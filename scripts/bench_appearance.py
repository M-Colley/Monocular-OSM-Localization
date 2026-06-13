"""Offline test of MULTI-FRAME cross-view appearance disambiguation.

The benchmark (scripts/bench_matching.py) proved trajectory shape can't
rank the true route to #1 — and a single mid-frame BevSplat score didn't
help either (it confidently picked a 620 m-wrong candidate on KITTI
0033). The hypothesis here: a wrong walk may match appearance at *one*
point by chance, but matching the video frame sequence at *many* points
along a specific walk is far less likely to be coincidental. So we score
appearance at K points along each candidate's walk and aggregate.

For candidate c and route-fraction f_k:
  * tile center  = c.aligned_traj_xy at arc-fraction f_k  (camera-location
                   hypothesis under candidate c), projected to lon/lat;
  * query frame  = the real video frame at time-fraction f_k;
  * score_k      = BevSplat(frame_k, tile_k);
the candidate's appearance score is mean_k score_k.

We then check: does aggregated appearance rank the GT-best candidate near
#1, where single-frame appearance and shape both failed?

    python scripts/bench_appearance.py --k 5 --top 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bev_splat_match import BevSplatConfig, _load_bev_splat_inference
from src.evaluator import evaluate_candidates_against_waypoints, load_gt_waypoints
from src.osm_data import fetch_city_graph
from src.satellite import fetch_satellite_tile
from src.trajectory_matching import match_trajectory
from src.visual_odometry import default_intrinsics, trajectory_arc_length

DATA = Path("data")

# KITTI drive_0033 — the residential loop where shape gets the right
# neighborhood (144 m) but not the street, and single-frame appearance hurt.
NPZ = "local-36a50c34107a-drive-0033-karlsruhe-germany/trajectory_v2_0-166.0_s3_fauto.npz"
GRAPHML = "local-36a50c34107a-drive-0033-karlsruhe-germany/Karlsruhe_Germany_around_48.9702_8.4788_968.graphml"
GT = "ground_truth/kitti_drive_0033.json"
VIDEO = "data/kitti/drive_0033.mp4"
EST_LEN = 876.0
WEIGHTS = "third_party/BevSplat-weights/KITTI_no_GPS.pth"
REPO = "third_party/BevSplat"


def _point_at_fraction(xy: np.ndarray, f: float) -> np.ndarray:
    arc = trajectory_arc_length(xy)
    target = f * arc[-1]
    j = int(np.searchsorted(arc, target))
    return xy[min(j, len(xy) - 1)]


def _frames_at_fractions(video: str, fractions: list[float]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in fractions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f * (total - 1)))
        ok, frame = cap.read()
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None)
    cap.release()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=5, help="frames/positions per candidate")
    p.add_argument("--top", type=int, default=40, help="candidates to score")
    args = p.parse_args()

    xz = np.load(DATA / NPZ)["xz"]
    road = fetch_city_graph("kitti", cache_path=DATA / GRAPHML)
    locked = EST_LEN / float(trajectory_arc_length(xz)[-1])
    pool = match_trajectory(xz, road, final_top_k=args.top, sample_every=1,
                            estimated_length_m=EST_LEN, locked_scale=locked, progress=False)
    wp = load_gt_waypoints(Path(GT))
    gt_err = np.array([e.mean_route_error_m
                       for e in evaluate_candidates_against_waypoints(pool, road, wp)])
    best_i = int(np.argmin(gt_err))
    print(f"pool {len(pool)}; GT-best is candidate #{best_i} ({gt_err[best_i]:.0f} m)")

    fractions = list(np.linspace(0.1, 0.9, args.k))
    frames = _frames_at_fractions(VIDEO, fractions)
    h, w = frames[0].shape[:2]
    K = default_intrinsics(w, h)
    to_ll = Transformer.from_crs(road.crs, "EPSG:4326", always_xy=True)

    inference, err = _load_bev_splat_inference(BevSplatConfig(
        weights_path=Path(WEIGHTS), repo_path=Path(REPO),
        model_module="models.models_kitti_nips", satellite_source="esri"))
    if inference is None:
        raise SystemExit(f"BevSplat inference unavailable: {err}")

    mf_score = np.full(len(pool), np.nan)   # multi-frame mean peak-correlation
    sf_score = np.full(len(pool), np.nan)   # single mid-frame (baseline)
    off_mag = np.full(len(pool), np.nan)    # mean predicted offset magnitude (lower=better)
    off_std = np.full(len(pool), np.nan)    # std of predicted offset across frames (lower=better)
    mid = len(fractions) // 2
    for i, cand in enumerate(pool):
        per_frame, offs = [], []
        for fk, frame in zip(fractions, frames):
            if frame is None:
                continue
            # Tile centre = the candidate's WALK (true OSM scale) at this
            # arc-fraction — scale-robust, unlike aligned_traj_xy which is
            # the VO squished onto the walk at the (possibly wrong) prior
            # scale. The query frame is the real view at the same fraction.
            pt = _point_at_fraction(cand.walk_xy, fk)
            lon, lat = to_ll.transform(float(pt[0]), float(pt[1]))
            try:
                tile = fetch_satellite_tile(lon, lat, half_extent_m=60.0, size=512, provider="esri")
                s, du, dv, _dh = inference(frame, tile, K)
            except Exception:
                continue
            per_frame.append(float(s))
            offs.append((float(du), float(dv)))
        if per_frame:
            mf_score[i] = float(np.mean(per_frame))
            sf_score[i] = per_frame[min(mid, len(per_frame) - 1)]
        if len(offs) >= 2:
            offs = np.array(offs)
            off_mag[i] = float(np.mean(np.hypot(offs[:, 0], offs[:, 1])))
            off_std[i] = float(np.mean(offs.std(axis=0)))
        print(f"  cand #{i:2d}  GTerr {gt_err[i]:5.0f} m  "
              f"mf {mf_score[i]:.3f}  |off| {off_mag[i]:.3f}  off_std {off_std[i]:.3f}",
              flush=True)

    def report(name, score, higher_better=True):
        ok = np.isfinite(score)
        if ok.sum() < 3:
            print(f"{name}: too few scored"); return
        s = score if higher_better else -score
        idx = np.where(ok)[0][np.argsort(-s[ok])]
        pick = idx[0]
        rank_best = int(np.where(idx == best_i)[0][0]) + 1 if best_i in idx else -1
        r = np.corrcoef(score[ok], gt_err[ok])[0, 1]
        print(f"{name}: pick #1 -> {gt_err[pick]:.0f} m   "
              f"GT-best ranked #{rank_best}/{ok.sum()}   corr(metric,GTerr)={r:+.2f}")

    print("\n=== RESULTS (KITTI 0033) ===")
    print(f"  shape pick (geometry #1) -> {gt_err[0]:.0f} m;  best-in-pool {gt_err[best_i]:.0f} m")
    report("  mf peak-score ", mf_score, higher_better=True)
    report("  offset |mag|  ", off_mag, higher_better=False)
    report("  offset std    ", off_std, higher_better=False)


if __name__ == "__main__":
    main()
