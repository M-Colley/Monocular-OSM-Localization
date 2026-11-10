"""Build/QC one preregistered test4 clip; never localize or score it."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main(vid: str) -> int:
    import cv2
    import psutil
    import build_overlay_fleet as b
    import check_overlay_gt as q

    cv2.setNumThreads(2)
    # Explicit genuine-function dependency wrapper changes only ORT CPU pool
    # sizing. The unchanged builder still owns extraction/QC/window logic.
    from monocular_osm.scene_text import RapidOcrReader
    from functools import partial
    cpu_reader = RapidOcrReader(params={
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })
    original_extract = b.extract_gps_track
    b.extract_gps_track = partial(original_extract, ocr_reader=cpu_reader)
    proc = psutil.Process()
    proc.cpu_affinity(proc.cpu_affinity()[:2])
    here = Path(__file__).resolve().parent
    prereg = here / "selection_preregistration.json"
    receipt = json.loads((here / "preregistration_receipt.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(prereg.read_bytes()).hexdigest() == receipt["selection_sha256"]
    assert b.BY_ID[vid].split == "test4"
    out = here / "build_attempts"
    out.mkdir(exist_ok=True)
    attempt = 1
    while (out / f"{vid}_{attempt:02d}.json").exists():
        attempt += 1
    private = ROOT / "output" / "test4_data_acquisition_20261110"
    private.mkdir(parents=True, exist_ok=True)
    log_path = private / f"{vid}_{attempt:02d}.log"
    sources = ["scripts/build_overlay_fleet.py", "scripts/check_overlay_gt.py",
               "scripts/overlay_clips.py", "monocular_osm/gps_overlay.py",
               "monocular_osm/scene_text.py", "monocular_osm/download.py",
               "benchmarks/test4_screen/build_one_parallel.py"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    args = SimpleNamespace(data_dir="data/test4", max_height=1080, interval=5.0,
                           cpu=True, no_cache=False, min_fixes=12, max_kmh=160.0,
                           waypoints=20, out_dir="ground_truth", prefix="overlay_")
    result = {"id": vid, "city": b.BY_ID[vid].city_override, "split": "test4",
              "attempt": attempt, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "selection_sha256": receipt["selection_sha256"], "source_sha256": hashes,
              "localization_executed": False, "new_labels_restricted_to_data_qc": True,
              "OCR_operational_settings": {"ORT_intra": 2, "ORT_inter": 1,
                                            "weights_preprocessing_thresholds_unchanged": True}}
    code = 1
    with log_path.open("x", encoding="utf-8") as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                row = b._build(b.BY_ID[vid], args)
                if row is None:
                    result.update(status="GT_BUILD_DECLINED", available=False)
                else:
                    ok, problems, _stats = q.check(ROOT / row["gt_path"])
                    enough = row["span_km"] >= 0.5
                    for p, h in hashes.items():
                        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, p
                    result.update(video_path=row["video_path"], gt_path=row["gt_path"],
                                  plausibility_passed=ok, route_at_least_500m=enough)
                    cap = cv2.VideoCapture(str(ROOT / row["video_path"]))
                    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    result["download_duration_s"] = count / fps if fps > 0 else None
                    if ok and enough:
                        # Separate CPU2 builders may finish together. Serialize
                        # the unchanged index merge; each GT/video path is unique.
                        import msvcrt
                        lock_path = private / "overlay_index_publication.lock"
                        with lock_path.open("a+b") as lock:
                            if lock.tell() == 0:
                                lock.write(b"0")
                                lock.flush()
                            lock.seek(0)
                            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                            try:
                                b._write_index([row], ROOT / "ground_truth/overlay_fleet.json")
                            finally:
                                lock.seek(0)
                                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                        result.update(status="AVAILABLE_GT_QC_PASSED", available=True)
                        code = 0
                    else:
                        # Keep the new failed artifact and diagnostics; do not
                        # put it into the usable registry or repair its labels.
                        log.write(repr(problems) + "\n")
                        result.update(status="GT_QC_FAILED", available=False)
        except Exception as exc:
            text = str(exc).casefold()
            bot = any(k in text for k in ("not a bot", "sign in to confirm", "429", "too many requests"))
            result.update(status="UNVERIFIED_INFRASTRUCTURE", available=False,
                          exception_type=type(exc).__name__, bot_check=bot)
            if bot:
                result["retry_not_before_utc"] = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=30)).isoformat()
            # Transport/label details stay in the private QC log.
            log.write(type(exc).__name__ + ": " + str(exc) + "\n")
            code = 75
    result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    result["qc_log_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
    with (out / f"{vid}_{attempt:02d}.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, ensure_ascii=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
