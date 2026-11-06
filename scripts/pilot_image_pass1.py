"""Pilot: ImAge (NeurIPS 2025, official GSV weights) against MegaLoc on the
city-wide FIRST retrieval pass.

For each clip, the first-pass reference set and MegaLoc similarity matrix
saved by an assisted baseline child (``retrieval_audit/0000.npz``) are reused;
ImAge encodes the same references (cached pixels, ``mapillary/ref_img_store.npz``)
and the same 80 query frames (seeked from the clip's video at the saved query
times), as the whole frame and as the best of three square crops. Both
models' matrices then go through the same diagnostics: the rank of the true
2 km area (``pass1_hypothesis_rank``) and the distance from the pipeline's
decoded robust centre to the route. Dev clips; GT for scoring only; CPU,
4 threads, below-normal priority.

    python scripts/pilot_image_pass1.py <assisted_run_dir> <run_data_dir> <image_inputs_dir> <out.json> [clip ...]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import psutil

psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import torch  # noqa: E402

from monocular_osm.geo import LocalFrame, geodesic_distance_m  # noqa: E402
from monocular_osm.kartaview_vpr import _robust_center, _square_query_crop, _viterbi_decode  # noqa: E402
from scripts.pass1_hypothesis_rank import TRUE_M, area_ranking  # noqa: E402
from scripts.pilot_image_paired import encode, load_model  # noqa: E402

torch.set_num_threads(4)


def frames_at(video: Path, times) -> list:
    cap = cv2.VideoCapture(str(video))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read {video} at {t:.2f} s")
        out.append(img)
    cap.release()
    return out


def diagnostics(sims, ref_ll, times, gt):
    frame = LocalFrame(float(gt[:, 0].mean()), float(gt[:, 1].mean()))
    gx, gy = frame.to_xy(gt[:, 0], gt[:, 1])
    route = np.c_[gx, gy]
    areas = area_ranking(sims, ref_ll, frame)
    dist = [float(np.min(np.hypot(*(route - c).T))) for c, _ in areas]
    rank = next((i + 1 for i, d in enumerate(dist) if d <= TRUE_M), None)
    dt = float((times[-1] - times[0]) / max(len(times) - 1, 1))
    path = _viterbi_decode(sims.astype(np.float32), ref_ll, dt)
    centre = _robust_center(ref_ll[path], sims[np.arange(len(path)), path])
    to_route = min(geodesic_distance_m(centre[0], centre[1], a, b) for a, b in gt)
    return {"true_area_rank": rank, "centre_to_route_m": round(float(to_route), 1),
            "top1_area_to_route_m": round(dist[0], 1)}


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    run, data, inputs, out = argv[:4]
    only = set(argv[4:])
    man = {c["id"]: c for c in json.loads((ROOT / "benchmarks/manifest.json").read_text(encoding="utf-8"))["clips"]}
    model, _audit = load_model(Path(inputs), torch)
    rows = []
    for cid in sorted(man):
        if only and cid not in only:
            continue
        f = sorted(glob.glob(f"{run}/{cid}/rep0/pipeline_out/*/children/*/baseline/retrieval_audit/0000.npz"))
        if not f:
            continue
        z = np.load(f[0], allow_pickle=False)
        ids = [str(i).split(":", 1)[-1] for i in z["ref_ids"]]
        slug = Path(f[0]).parts[-6]
        slug_dirs = [d for d in [Path(data) / slug] if (d / "mapillary/ref_img_store.npz").exists()]
        if not slug_dirs:
            print(cid, "no pixel store"); continue
        with np.load(slug_dirs[0] / "mapillary/ref_img_store.npz", allow_pickle=True) as st:
            pos = {str(k): i for i, k in enumerate(st["ids"])}
            keep = [j for j, i in enumerate(ids) if i in pos]
            raw = np.asarray(st["raw"])[[pos[ids[j]] for j in keep]]
        gt = np.array([[w["lat"], w["lon"]] for w in
                       json.loads((ROOT / man[cid]["gt"]).read_text(encoding="utf-8"))["waypoints"]])
        times = z["query_times"]
        video = (slug_dirs[0] / "input.mp4" if (slug_dirs[0] / "input.mp4").exists()
                 else ROOT / man[cid]["video"]["path"])          # local clips (KITTI)
        frames = frames_at(video, times)
        ref_ll = z["ref_latlon"][keep]
        megaloc = z["raw_similarities"][:, keep]
        ref_emb = encode([im[:, :, ::-1] for im in raw], model, torch)
        whole = encode([im[:, :, ::-1] for im in frames], model, torch)
        crops = encode([_square_query_crop(im, k)[:, :, ::-1] for im in frames for k in range(3)], model, torch)
        crop_sims = (crops @ ref_emb.T).reshape(len(frames), 3, -1).max(axis=1)
        row = {"clip": cid, "n_refs": len(keep),
               "megaloc": diagnostics(megaloc, ref_ll, times, gt),
               "image_whole": diagnostics(whole @ ref_emb.T, ref_ll, times, gt),
               "image_crops_max": diagnostics(crop_sims, ref_ll, times, gt)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        Path(out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
