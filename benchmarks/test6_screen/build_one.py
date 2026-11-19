"""Build and QC one test6 worldwide screen pass onto E:/claude-scoring/test6/data (never localizes)."""
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
HERE = Path(__file__).resolve().parent
DATA_DIR = "E:/claude-scoring/test6/data"


def screen_pass(vid: str) -> dict:
    rows = json.loads((HERE / "screen_results.json").read_text(encoding="utf-8"))
    passes = [r for r in rows if r.get("id") == vid and r.get("ok")]
    assert passes, f"{vid} has no screen pass"
    return passes[-1]


def main(vid: str) -> int:
    import cv2
    from functools import partial
    import build_overlay_fleet as b
    import check_overlay_gt as q
    from overlay_clips import OverlayClip
    from monocular_osm.scene_text import RapidOcrReader

    cv2.setNumThreads(2)
    cpu_reader = RapidOcrReader(params={
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })
    b.extract_gps_track = partial(b.extract_gps_track, ocr_reader=cpu_reader)

    receipt = json.loads((HERE / "preregistration_receipt.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((HERE / "selection_preregistration.json").read_bytes()).hexdigest() == receipt["selection_sha256"]

    row = screen_pass(vid)
    clip = OverlayClip(
        video_id=vid,
        region=row["region"],
        window=(0.0, 600.0),
        split="test6",
        city_override=row["registered_city"],
        overlay=f"GPS coordinate stamp ({row['region']})",
        note=f"test6 worldwide screen pass ({row.get('continent_region')}): {row['city']} query '{row['query']}' rank {row['rank']}. No localization used.",
    )

    out = HERE / "build_attempts"
    out.mkdir(exist_ok=True)
    attempt = 1
    while (out / f"{vid}_{attempt:02d}.json").exists():
        attempt += 1

    private = Path("E:/claude-scoring/test6/private_qc_logs")
    private.mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    log_path = private / f"{vid}_{attempt:02d}.log"

    args = SimpleNamespace(
        data_dir=DATA_DIR, max_height=1080, interval=5.0, cpu=True, no_cache=False,
        min_fixes=12, max_kmh=160.0, waypoints=20, out_dir="ground_truth", prefix="overlay_",
    )
    result = {
        "id": vid,
        "pool_city": row["city"],
        "city": clip.city_override,
        "continent_region": row.get("continent_region"),
        "split": "test6",
        "attempt": attempt,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_sha256": receipt["selection_sha256"],
        "localization_executed": False,
        "new_labels_restricted_to_data_qc": True,
    }
    code = 1
    with log_path.open("x", encoding="utf-8") as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                built = b._build(clip, args)
                if built is None:
                    result.update(status="GT_BUILD_DECLINED", available=False)
                else:
                    ok, problems, _stats = q.check(ROOT / built["gt_path"])
                    enough = built["span_km"] >= 0.5
                    vpath = Path(built["video_path"])
                    vpath_abs = vpath if vpath.is_absolute() else (ROOT / vpath)
                    result.update(
                        video_path=built["video_path"],
                        gt_path=built["gt_path"],
                        plausibility_passed=ok,
                        route_at_least_500m=enough,
                    )
                    cap = cv2.VideoCapture(str(vpath_abs))
                    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    result["download_duration_s"] = count / fps if fps > 0 else None
                    if ok and enough:
                        b._write_index([built], ROOT / "ground_truth/overlay_fleet.json")
                        result.update(status="AVAILABLE_GT_QC_PASSED", available=True)
                        code = 0
                    else:
                        log.write(repr(problems) + "\n")
                        result.update(status="GT_QC_FAILED", available=False)
        except Exception as exc:
            text = str(exc).casefold()
            bot = any(k in text for k in ("not a bot", "sign in to confirm", "429", "too many requests"))
            result.update(
                status="UNVERIFIED_INFRASTRUCTURE",
                available=False,
                exception_type=type(exc).__name__,
                bot_check=bot,
            )
            log.write(type(exc).__name__ + ": " + str(exc) + "\n")
            code = 75

    result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    result["qc_log_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
    with (out / f"{vid}_{attempt:02d}.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
