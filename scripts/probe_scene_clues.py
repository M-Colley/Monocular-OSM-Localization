"""CPU-only, offline scene-text inventory over registered video windows.

Reads only manifest video/window fields, never GT files or geocoders. Cached
RapidOCR ONNX assets are mandatory; downloads and GPU providers are disabled.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _local_video(entry):
    source = entry["video"]
    if "path" in source:
        path = ROOT / source["path"]
    else:
        from monocular_osm.stages.acquire import pick_cached_input

        identifier = parse_qs(urlparse(source["url"]).query)["v"][0]
        prefix = re.sub(r"[^a-z0-9]+", "-", identifier.lower()).strip("-")
        candidates = [pick_cached_input(p) for p in (ROOT / "data").glob(prefix + "-*")]
        candidates = [p for p in candidates if p is not None]
        if len(candidates) != 1:
            raise RuntimeError(f"{entry['id']}: expected one local cached video, found {candidates}")
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _repeated_clues(detections):
    groups = defaultdict(list)
    for detection in detections:
        key = unicodedata.normalize("NFKC", detection.text).casefold()
        key = re.sub(r"[^\w]+", " ", key).strip()
        if key:
            groups[key].append(detection)
    out = []
    for key, values in groups.items():
        times = sorted({v.t_sec for v in values})
        if len(times) < 2:
            continue
        out.append({"literal_key": key, "spellings": sorted({v.text for v in values}),
                    "distinct_frames": len(times), "times_sec": times,
                    "mean_confidence": sum(v.confidence for v in values) / len(values),
                    "max_confidence": max(v.confidence for v in values),
                    "has_letters": any(c.isalpha() for c in key),
                    "bboxes": [v.bbox for v in values]})
    return sorted(out, key=lambda item: (-item["distinct_frames"], -item["mean_confidence"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", default=["kitti_0009", "kitti_0033", "overlay_y66ZkRpUh4k"])
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--ort-threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "scene_clue_probe")
    args = parser.parse_args()
    if not 20 <= args.frames <= 30 or not 1 <= args.ort_threads <= 4:
        parser.error("use 20-30 frames and 1-4 CPU threads")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = str(args.ort_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.ort_threads)
    os.environ["HF_HUB_OFFLINE"] = "1"

    import cv2
    import numpy as np
    from monocular_osm.scene_text import RapidOcrReader, extract_scene_text, is_position_stamp
    from rapidocr.utils.download_file import DownloadFile

    spec = importlib.util.find_spec("rapidocr")
    model_dir = Path(spec.origin).parent / "models"
    names = {"Det": "PP-OCRv6_det_small.onnx", "Rec": "PP-OCRv6_rec_small.onnx",
             "Cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx"}
    assets = {task: model_dir / name for task, name in names.items()}
    for asset in assets.values():
        if not asset.is_file() or asset.stat().st_size == 0:
            raise FileNotFoundError(f"Required cached model missing; downloads disabled: {asset}")

    def forbid_download(cls, *unused, **kwargs):
        raise RuntimeError("Model/font downloads are disabled for this offline probe")

    DownloadFile.run = classmethod(forbid_download)
    params = {f"{task}.model_path": str(path) for task, path in assets.items()}
    params.update({
        "Global.log_level": "error", "Global.text_score": 0.3,
        "EngineConfig.onnxruntime.intra_op_num_threads": args.ort_threads,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False,
        "EngineConfig.onnxruntime.use_dml": False,
        "EngineConfig.onnxruntime.use_cann": False,
        "EngineConfig.onnxruntime.use_coreml": False,
        "Rec.rec_batch_num": 4, "Cls.cls_batch_num": 4,
    })
    reader = RapidOcrReader(params=params)
    manifest = json.loads((ROOT / "benchmarks" / "manifest.json").read_text(encoding="utf-8"))
    # Strict allowlist: no GT fields are read by this probe.
    entries = {c["id"]: {k: c[k] for k in ("id", "name", "city", "video", "vo_segment")}
               for c in manifest["clips"] if c["id"] in args.clips}
    if set(args.clips) - set(entries):
        raise ValueError(f"Unknown clip IDs: {set(args.clips) - set(entries)}")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"purpose": "literal scene clues; no geocoding or accuracy evaluation",
               "cpu_only": True, "downloads_disabled": True,
               "rapidocr_version": importlib.metadata.version("rapidocr"),
               "onnxruntime_version": importlib.metadata.version("onnxruntime"),
               "ort_intra_op_threads": args.ort_threads, "ort_inter_op_threads": 1,
               "model_assets": {task: {"path": str(path), "bytes": path.stat().st_size,
                                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                for task, path in assets.items()}, "clips": []}
    for identifier in args.clips:
        entry = entries[identifier]
        video = _local_video(entry)
        start, end = map(float, entry["vo_segment"].split(":"))
        sampled = []

        def frame_reader(path, start_sec, end_sec, interval_sec):
            cap = cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open {path}")
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                first, stop = int(start_sec * fps), int(end_sec * fps)
                stop = min(stop, total) if total > 0 else stop
                indices = np.linspace(first, stop - 1, min(args.frames, stop - first)).astype(int)
                for index in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                    ok, image = cap.read()
                    if not ok:
                        raise RuntimeError(f"Decode failed at registered query frame {index}")
                    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    t_sec = pos_ms / 1000 if pos_ms > 0 else index / fps
                    sampled.append({"frame_index": int(index), "t_sec": float(t_sec),
                                    "shape": list(image.shape)})
                    yield float(t_sec), image
            finally:
                cap.release()

        class CountingReader:
            frames = 0
            rejected_stamps = 0

            def readtext(self, image):
                items = reader.readtext(image)
                self.frames += 1
                self.rejected_stamps += sum(is_position_stamp(str(item[1])) for item in items)
                if self.frames % 4 == 0:
                    print(f"{identifier}: OCR {self.frames}/{args.frames} frames", flush=True)
                return items

        counter = CountingReader()
        began = time.perf_counter()
        detections = extract_scene_text(
            video, start_sec=start, end_sec=end, min_confidence=0.3, min_len=2,
            ocr_reader=counter, use_gpu=False, frame_reader=frame_reader, cache_path=None)
        providers = {}
        for component in ("text_det", "text_rec", "text_cls"):
            session = getattr(reader._engine, component, None)
            if session is not None:
                providers[component] = session.session.session.get_providers()
        if any(value != ["CPUExecutionProvider"] for value in providers.values()):
            raise RuntimeError(f"Unexpected inference provider: {providers}")
        record = {**entry, "video_path": str(video), "video_bytes": video.stat().st_size,
                  "wall_s": round(time.perf_counter() - began, 3), "providers": providers,
                  "sampled_frames": sampled, "rejected_position_stamps": counter.rejected_stamps,
                  "detections": [asdict(d) for d in detections],
                  "repeated_literal_clues": _repeated_clues(detections)}
        (args.out / f"{identifier}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["clips"].append({"id": identifier, "frames": len(sampled),
                                  "detections": len(detections), "wall_s": record["wall_s"],
                                  "repeated_literal_clues": record["repeated_literal_clues"]})
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{identifier}: saved {len(detections)} detections in {record['wall_s']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
