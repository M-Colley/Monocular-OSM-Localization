"""Hybrid: VGGT selects the area, VO+loop-closure gives precise geometry.

VGGT's drift-free trajectory ranks the TRUE place #1 (selection — the
wall shape-matching couldn't break), but its local pose noise makes its
fine geometry worse than VO. VO+loop-closure has the precise geometry
(best-in-pool ~21 m) but ranks the true place #14 (drifted shape fits
parallel streets). So: match VGGT to get the area, then match the
loop-closed VO RESTRICTED to that area (anchor-gated) for the final
precise pick.

    python scripts/test_vggt_hybrid.py   # KITTI 0033, uses output/vggt_traj.npy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluator import evaluate_candidates_against_waypoints, load_gt_waypoints
from src.loop_closure import redistribute_drift
from src.osm_data import fetch_city_graph
from src.trajectory_matching import match_trajectory
from src.visual_odometry import trajectory_arc_length

DATA = Path("data")
GML = "local-36a50c34107a-drive-0033-karlsruhe-germany/Karlsruhe_Germany_around_48.9702_8.4788_968.graphml"
NPZ = "local-36a50c34107a-drive-0033-karlsruhe-germany/trajectory_v2_0-166.0_s3_fauto.npz"
GT = "ground_truth/kitti_drive_0033.json"


def main() -> None:
    road = fetch_city_graph("k", cache_path=DATA / GML)
    wp = load_gt_waypoints(Path(GT))

    # 1. VGGT trajectory -> match -> the selected area (top candidates' start nodes).
    vggt = np.load("output/vggt_traj.npy")
    v_arc = float(trajectory_arc_length(vggt)[-1])
    vpool = match_trajectory(vggt, road, final_top_k=15, sample_every=1,
                             estimated_length_m=1900.0, locked_scale=1900.0 / v_arc,
                             progress=False)
    seed = list(dict.fromkeys(c.start_node for c in vpool[:8]))
    print(f"VGGT selected {len(seed)} seed start-nodes (top-8 candidates)")

    # 2. VO + loop-closure, gated to the VGGT area, IPM-ish scale.
    xz = np.load(DATA / NPZ)["xz"]
    xz_lc = redistribute_drift(xz, 9, 531)   # known KITTI 0033 loop pair
    arc = float(trajectory_arc_length(xz_lc)[-1])
    for est in [1705.0, 1900.0, 2225.0]:
        # baseline: ungated VO+lc
        p_un = match_trajectory(xz_lc, road, final_top_k=20, sample_every=1,
                                estimated_length_m=est, locked_scale=est / arc, progress=False)
        e_un = evaluate_candidates_against_waypoints(p_un, road, wp)
        g_un = np.array([x.mean_route_error_m for x in e_un]); g_un = np.where(np.isfinite(g_un), g_un, 1e9)
        # hybrid: VO+lc gated to VGGT seeds
        p_h = match_trajectory(xz_lc, road, final_top_k=20, sample_every=1,
                               estimated_length_m=est, locked_scale=est / arc,
                               extra_start_nodes=seed, restrict_to_start_nodes=True,
                               progress=False)
        if p_h:
            e_h = evaluate_candidates_against_waypoints(p_h, road, wp)
            g_h = np.array([x.mean_route_error_m for x in e_h]); g_h = np.where(np.isfinite(g_h), g_h, 1e9)
            # Re-rank the gated candidates by bearing-corr to the VGGT
            # (drift-free) signature instead of the drifted VO's.
            from src.visual_odometry import bearing_signature, resample_uniform
            vsig = bearing_signature(vggt, n_samples=128)
            def vcorr(c):
                try:
                    s = bearing_signature(resample_uniform(c.walk_xy, 128), n_samples=128)
                except Exception:
                    return -1
                a, b = vsig - vsig.mean(), s - s.mean()
                d = np.sqrt((a @ a) * (b @ b)) + 1e-9
                return float((a @ b) / d)
            vr = np.array([vcorr(c) for c in p_h])
            g_vr = g_h[int(np.argmax(vr))]
            print(f"  est {est:5.0f}: ungated {g_un[0]:3.0f}m | gated {g_h[0]:3.0f}m | "
                  f"gated+VGGT-rerank {g_vr:3.0f}m  (best-in-gate {g_h.min():3.0f}m)")
        else:
            print(f"  est {est:5.0f}: hybrid found nothing in gate")


if __name__ == "__main__":
    main()
