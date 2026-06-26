"""Run VGGT on a clip's keyframes, match the trajectory, evaluate vs GT.

VGGT gives a near-drift-free camera path (it closes loops VO leaves 27%
open). This measures whether that improves the localization: best-in-pool
(geometric ceiling), the #1 pick, and the distinct-place rank of the
GT-best candidate (selection). Keyframe density is the knob — too sparse
and the matcher interpolates straight segments through real curves.

    python scripts/test_vggt_match.py data/kitti/drive_0033.mp4 \
        <graphml> ground_truth/kitti_drive_0033.json --n 100 \
        --est 1400 1705 1900
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from scripts.test_vggt_traj import vggt_trajectory
from src.evaluator import evaluate_candidates_against_waypoints, load_gt_waypoints
from src.hypotheses import cluster_candidates
from src.osm_data import fetch_city_graph
from src.trajectory_matching import match_trajectory
from src.visual_odometry import trajectory_arc_length


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("graphml")
    ap.add_argument("gt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--est", type=float, nargs="+", default=[1705.0])
    ap.add_argument("--smooth", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(args.start * fps)
    f1 = int(args.end * fps) if args.end else total
    idxs = np.linspace(f0, min(f1, total) - 1, args.n).round().astype(int)
    frames = []
    for ix in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ix))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()

    xy = vggt_trajectory(frames, smooth=args.smooth)
    arc = float(trajectory_arc_length(xy)[-1])
    gap = 100 * np.linalg.norm(xy[-1] - xy[0]) / arc
    print(f"VGGT: {len(xy)} poses (smooth={args.smooth}), end-start gap {gap:.0f}% of arc")
    np.save("output/vggt_traj.npy", xy)

    road = fetch_city_graph("x", cache_path=Path(args.graphml))
    wp = load_gt_waypoints(Path(args.gt))
    for est in args.est:
        locked = est / arc
        pool = match_trajectory(xy, road, final_top_k=400, sample_every=1,
                                estimated_length_m=est, locked_scale=locked, progress=False)
        ev = evaluate_candidates_against_waypoints(pool, road, wp)
        g = np.array([e.mean_route_error_m for e in ev])
        g = np.where(np.isfinite(g), g, 1e9)
        bi = int(np.argmin(g))
        clusters = cluster_candidates(pool)
        brank = next((r for r, cl in enumerate(clusters, 1) if bi in cl), -1)
        best5 = min(g[cl[0]] for cl in clusters[:5])
        print(f"  est {est:5.0f}: best-in-pool {g.min():3.0f}m  pick {g[0]:3.0f}m  "
              f"GT-best place-rank #{brank}  best-of-top5 {best5:3.0f}m")


if __name__ == "__main__":
    main()
