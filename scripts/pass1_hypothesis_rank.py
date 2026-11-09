"""Where does the true area rank among the first retrieval pass's candidate areas?

For each clip, the saved pass-1 similarity matrix (assisted baseline child,
``retrieval_audit/0000.npz``) gives every query frame's top-k references. They
are pooled into 2 km grid areas, scored by summed similarity, and ranked; the
true area is any ranked area whose centre lies within 1 km of the ground-truth
route. A clip whose true area ranks 2nd or 3rd could be recovered by running
the second pass over several areas; one ranked far down could not. Dev clips,
in-sample; GT is used only to find the true area.

    python scripts/pass1_hypothesis_rank.py <assisted_run_dir> [out.json]
    python scripts/pass1_hypothesis_rank.py --plain <audit_out_dir> [out.json]

``--plain`` reads ``<dir>/<video id>/<slug>/retrieval_audit/0000.npz`` as
written by a plain ``--retrieval-only`` run with audit capture on. Each row
also reports the GT-free score ratio of the second to the first area and of
the true area to the first (what a multi-area rule would have to threshold).
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monocular_osm.geo import LocalFrame  # noqa: E402

CELL_M, TOP_K, TRUE_M = 2000.0, 5, 1000.0


def area_ranking(sims: np.ndarray, ref_latlon: np.ndarray, frame: LocalFrame):
    """[(centre_xy, score)] of 2 km areas, best first, from each query's top-k."""
    x, y = frame.to_xy(ref_latlon[:, 0], ref_latlon[:, 1])
    xy = np.c_[x, y]
    top = np.argsort(-sims, axis=1)[:, :TOP_K]
    cells: dict[tuple, list] = {}
    for q in range(sims.shape[0]):
        for r in top[q]:
            key = (int(np.floor(xy[r, 0] / CELL_M)), int(np.floor(xy[r, 1] / CELL_M)))
            cells.setdefault(key, []).append((float(sims[q, r]), xy[r]))
    ranked = sorted(cells.values(), key=lambda v: -sum(s for s, _ in v))
    return [(np.average([p for _, p in v], axis=0, weights=[s for s, _ in v]), sum(s for s, _ in v))
            for v in ranked]


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    plain = "--plain" in argv
    if plain:
        argv.remove("--plain")
    run = argv[0]
    man = {c["id"]: c for c in json.loads((ROOT / "benchmarks/manifest.json").read_text(encoding="utf-8"))["clips"]}
    rows = []
    for cid in sorted(man):
        pattern = (f"{run}/{cid.removeprefix('overlay_')}/*/retrieval_audit/0000.npz" if plain
                   else f"{run}/{cid}/rep0/pipeline_out/*/children/*/baseline/retrieval_audit/0000.npz")
        f = sorted(glob.glob(pattern))
        if not f:
            continue
        z = np.load(f[0], allow_pickle=False)
        wps = json.loads((ROOT / man[cid]["gt"]).read_text(encoding="utf-8"))["waypoints"]
        gt = np.array([[w["lat"], w["lon"]] for w in wps])
        frame = LocalFrame(float(gt[:, 0].mean()), float(gt[:, 1].mean()))
        gx, gy = frame.to_xy(gt[:, 0], gt[:, 1])
        route = np.c_[gx, gy]
        areas = area_ranking(z["raw_similarities"], z["ref_latlon"], frame)
        dist = [float(np.min(np.hypot(*(route - c).T))) for c, _ in areas]
        true_rank = next((i + 1 for i, d in enumerate(dist) if d <= TRUE_M), None)
        second = areas[1][1] / areas[0][1] if len(areas) > 1 else None
        true_ratio = areas[true_rank - 1][1] / areas[0][1] if true_rank else None
        rows.append({"clip": cid, "n_areas": len(areas), "true_area_rank": true_rank,
                     "top1_area_to_route_m": round(dist[0], 1),
                     "second_to_first_score": None if second is None else round(second, 3),
                     "true_to_first_score": None if true_ratio is None else round(true_ratio, 3)})
        print(f"{cid:22s} areas {len(areas):3d}  true area rank {true_rank!s:>4}  "
              f"top-1 area {dist[0]:7.0f} m from route  2nd/1st {second if second is None else round(second, 2)!s:>5}  "
              f"true/1st {true_ratio if true_ratio is None else round(true_ratio, 2)!s:>5}", flush=True)
    if len(argv) > 1:
        Path(argv[1]).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
