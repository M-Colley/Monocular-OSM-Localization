"""Evaluate the road-graph particle filter (monocular_osm.graph_filter) on KITTI.

GPS-free, odometry-only localization on OpenStreetMap, the setting of Brubaker,
Geiger & Urtasun, "Lost!" (CVPR 2013) / TPAMI 2016, on two KITTI raw drives:

    2011_09_26_drive_0009  (46 s, 447 frames)
    2011_09_30_drive_0033  (159 s, 1594 frames)

Maps (no ground-truth leak beyond the choice of region, stated here):

    A  "box"  a FIXED bounding box over the eastern Karlsruhe KITTI recording
              area, W 8.425 / S 48.958 / E 8.495 / N 49.020 (5.1 x 6.9 km,
              ~35 km^2), the SAME box for both drives. It is chosen to contain
              both drives with >= 0.9 km margin on every side, which is the
              Brubaker-style "the map covers the recording area" assumption;
              the drives are not centred in it. osmnx drive network.
    B  "city" the whole of Karlsruhe, ox.graph_from_place("Karlsruhe, Germany",
              network_type="drive") — the deployable "only the city is known"
              setting of the pipeline's README.

Odometry (all at 10 Hz, merged to 2 Hz filter steps):

    oracle       OXTS forward speed x dt (metres) + OXTS yaw. Bounds what the
                 filter can do with perfect relative motion. Not deployable.
    oracle_unit  OXTS yaw, but unit steps while moving (no speed): isolates
                 what monocular VO loses by having no scale.
    vo_oracle_speed  ablation: VO heading with OXTS metric speed — isolates
                 what monocular VO loses through its heading errors.
    vo           the repo's monocular VO (visual_odometry.estimate_trajectory
                 on image_02, stride 1, KITTI rectified intrinsics P_rect_02),
                 heading = optical-axis yaw, unit steps. Deployable.
    vo_gflow     vo, with step lengths replaced by the RELATIVE speed profile
                 of the repo's ground-flow estimator (ground_flow_scale.
                 step_lengths, camera height assumed 1.4 m); global scale left
                 unknown, as that module's own documentation prescribes.
                 Deployable; it abstains when the road-plane model fails.
    vo_plane     vo, with metric step lengths from the road-plane homography
                 and an ASSUMED camera height of 1.5 m (monocular_osm.
                 plane_speed; KITTI's is 1.65 m). Deployable. The filter's
                 global-scale prior is widened to log-sigma 0.25 for the
                 unknown height and the estimator's own bias (it read ~0.84x
                 at the true height on both drives, seen before these runs);
                 the per-step deviation sigma is the estimator's GT-free spread.
    vo_plane_rel the same profile normalised to mean 1, global scale left
                 unknown: exactly the vo_gflow treatment with a better profile.
    vo_lingbot_rel  LingBot-Map (arXiv 2604.14141) camera poses from
                 scripts/run_lingbot_map.py, heading from its rotations, its
                 step profile normalised to mean 1 (its scale is normalised
                 internally, ~0.05x metres on KITTI): the vo_gflow treatment.
    vo_lingbot_plane  the same poses with ONE global scale from the road-plane
                 estimate (median plane/LingBot step ratio over moving steps;
                 no ground truth): LingBot for shape and heading, the road for
                 metres. Metric filter settings as vo_plane, per-step noise
                 0.15 fixed before any run.

VO heading noise is self-calibrated without ground truth
(graph_filter.heading_increment_noise); the VO-vs-OXTS diagnostics in the
report use ground truth only to *describe* the VO, never to set a parameter.

Metrics per (drive x map x odometry x seed): error of the posterior mode at
every step, time-to-localize (first time error < 10 m and staying < 20 m),
final error, fraction of steps < 10 m / < 25 m, posterior mass within 25 m of
GT at the end, the smoothed (genealogy) route's mean error, runtime. Reported
as mean +- std over seeds. GT = OXTS lat/lon per frame
(kitti_raw.load_oxts_track).

Outputs under <root>/output/graph_filter/: results.json, results.md,
error_vs_time.png, plus cached graphs (graphml) and VO trajectories (npz).

    python scripts/eval_graph_filter.py --root <checkout with data/kitti>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from monocular_osm import graph_filter as gf  # noqa: E402

DRIVES = {
    "0009": {"date": "2011_09_26", "name": "2011_09_26_drive_0009_sync",
             "video": "drive_0009.mp4"},
    "0033": {"date": "2011_09_30", "name": "2011_09_30_drive_0033_sync",
             "video": "drive_0033.mp4"},
}
BOX_A = (8.425, 48.958, 8.495, 49.020)          # (west, south, east, north)
MAPS = {
    "A": {"file": "graph_A_box_karlsruhe_east.graphml", "bbox": BOX_A,
          "label": "A: fixed 35 km^2 box"},
    "B": {"file": "graph_B_karlsruhe_city.graphml", "place": "Karlsruhe, Germany",
          "label": "B: Karlsruhe city"},
}
# The repo's own GPS-free numbers for the same drives (README, 2026-07-23
# deployable table) and the city-centroid baseline (geocode of "Karlsruhe").
BASELINES = {
    "0009": {"pipeline_gps_free_mean_route_err_m": 5568, "city_centroid_err_m": 2645},
    "0033": {"pipeline_gps_free_mean_route_err_m": 7126, "city_centroid_err_m": 6843},
}
FILTER_HZ = 2.0


def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "kitti" / "raw").is_dir():
            return p
    raise SystemExit("could not find a checkout with data/kitti/raw; pass --root")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_oxts(drive_dir: Path) -> dict:
    """lat/lon/t from kitti_raw.load_oxts_track plus yaw and forward speed."""
    from monocular_osm.kitti_raw import load_oxts_track

    fixes = load_oxts_track(drive_dir)
    files = sorted((drive_dir / "oxts" / "data").glob("*.txt"))
    raw = np.array([np.array(f.read_text(encoding="utf-8").split()[:30], float) for f in files])
    t = np.array([f.t_sec for f in fixes])
    return {"t": t, "lat": raw[:, 0], "lon": raw[:, 1], "yaw": raw[:, 5],
            "vf": raw[:, 8], "latlon": np.column_stack([raw[:, 0], raw[:, 1]])}


def kitti_K(calib: Path, cam: str = "02") -> np.ndarray:
    for line in calib.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"P_rect_{cam}:"):
            P = np.array(line.split()[1:], float).reshape(3, 4)
            return P[:, :3].copy()
    raise ValueError(f"P_rect_{cam} not in {calib}")


def load_graph(kind: str, out: Path):
    spec = MAPS[kind]
    return gf.fetch_drive_graph(bbox=spec.get("bbox"), place=spec.get("place"),
                                cache_path=out / spec["file"])


def vo_trajectory(drive_dir: Path, K: np.ndarray, cache: Path):
    """The repo's monocular VO on image_02 (stride 1), cached as npz."""
    import cv2

    from monocular_osm.visual_odometry import (
        VO_ALGO_VERSION, Trajectory, estimate_trajectory)

    path = cache.with_name(f"{cache.stem}_v{VO_ALGO_VERSION}.npz")
    if path.exists():
        z = np.load(path)
        return Trajectory(centers=z["centers"], xz=z["xz"], valid=z["valid"],
                          n_inliers=list(z["n_inliers"]), rotations=z["rotations"],
                          translations=z["translations"]), path
    files = sorted((drive_dir / "image_02" / "data").glob("*.png"))
    frames = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    if any(f is None for f in frames):
        raise RuntimeError("unreadable PNG in " + str(drive_dir))
    t0 = time.perf_counter()
    traj = estimate_trajectory(frames, K)
    print(f"    VO on {len(frames)} frames: {time.perf_counter() - t0:.0f} s", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, centers=traj.centers, xz=traj.xz, valid=traj.valid,
                        n_inliers=np.asarray(traj.n_inliers), rotations=traj.rotations,
                        translations=traj.translations)
    return traj, path


def plane_lengths(drive_dir: Path, traj, K: np.ndarray, cache: Path,
                  camera_height_m: float | None = None) -> dict:
    """monocular_osm.plane_speed on the VO's own frames (cached npz)."""
    from monocular_osm.plane_speed import DEFAULT_CAMERA_HEIGHT_M, plane_step_lengths

    camera_height_m = camera_height_m or DEFAULT_CAMERA_HEIGHT_M
    if camera_height_m != DEFAULT_CAMERA_HEIGHT_M:
        cache = cache.with_name(f"{cache.stem}_h{camera_height_m:g}{cache.suffix}")

    if cache.exists():
        z = np.load(cache)
        return {"lengths": z["lengths"], "raw": z["raw"], "rel_spread": float(z["rel_spread"]),
                "coverage": float(z["coverage"]), "camera_height_m": float(z["camera_height_m"])}
    import cv2

    files = sorted((drive_dir / "image_02" / "data").glob("*.png"))
    frames = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    t0 = time.perf_counter()
    ps = plane_step_lengths(frames, traj, K, camera_height_m=camera_height_m)
    print(f"    road-plane speed: coverage {ps.coverage:.2f}, spread {ps.rel_spread:.2f} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)
    np.savez_compressed(cache, lengths=ps.lengths_m, raw=ps.raw_m, rel_spread=ps.rel_spread,
                        coverage=ps.coverage, camera_height_m=ps.camera_height_m)
    return {"lengths": ps.lengths_m, "raw": ps.raw_m, "rel_spread": ps.rel_spread,
            "coverage": ps.coverage, "camera_height_m": ps.camera_height_m}


def lingbot_trajectory(path: Path):
    """A visual_odometry.Trajectory from LingBot-Map's world-to-camera poses."""
    from monocular_osm.visual_odometry import Trajectory, _fit_plane_projection

    z = np.load(path)
    ext = np.asarray(z["extrinsic"], float)
    R, t = ext[:, :, :3], ext[:, :, 3]
    centers = -np.einsum("nji,nj->ni", R, t)
    up = -np.mean(R[:, 1, :], axis=0)                 # camera up (-y) in the world frame
    xz = _fit_plane_projection(centers, up=up / max(np.linalg.norm(up), 1e-12))
    n = len(centers)
    return Trajectory(centers=centers, xz=xz, valid=np.ones(n, bool), n_inliers=[0] * n,
                      rotations=R, translations=t), np.asarray(z["frame_index"], int)


def ground_flow_lengths(video: Path, traj, K: np.ndarray, size) -> tuple:
    from monocular_osm.ground_flow_scale import step_lengths, track_road_features

    tracks = track_road_features(video, stride=1, size=size)
    unit = np.linalg.norm(np.diff(traj.xz, axis=0), axis=1)
    lengths, conf = step_lengths(tracks, K, step_directions_valid=unit > 1e-9,
                                 fallback_lengths=unit)
    return lengths, float(conf)


# --------------------------------------------------------------------------
# odometry variants
# --------------------------------------------------------------------------

def odometry_variants(ox: dict, traj, gflow, frame_dt: float, plane=None,
                      lingbot=None) -> dict:
    t = ox["t"]
    dt = np.diff(t)
    vf = ox["vf"][1:]
    moving = vf > 0.2
    out = {}
    out["oracle"] = gf.odometry_from_arrays(np.where(moving, vf * dt, 0.0), ox["yaw"][1:], dt,
                                            metric=True)
    out["oracle_unit"] = gf.Odometry(step=moving.astype(float), heading=ox["yaw"][1:], dt=dt,
                                     metric=False, unit_seconds=frame_dt)
    if traj is not None:
        o = gf.odometry_from_trajectory(traj, frame_dt, heading_source="rotation")
        o.dt = dt.copy()
        out["vo"] = o
        # ablation: VO heading, OXTS speed — separates heading from scale
        out["vo_oracle_speed"] = gf.Odometry(
            step=np.where(moving, vf * dt, 0.0), heading=o.heading, dt=dt, metric=True,
            heading_var=o.heading_var)
        if gflow is not None and gflow[1] > 0:
            # Relative speed profile only: ground_flow_scale itself documents
            # that its absolute scale is worthless (assumed camera height,
            # self-calibrated horizon) and on KITTI it reads ~0.5x the true
            # distance. Normalise to mean 1 per moving frame, keep the global
            # scale unknown (non-metric, same speed prior as unit-step VO).
            lengths = np.asarray(gflow[0], float)
            mv = lengths > 0
            rel = lengths / max(float(lengths[mv].mean()), 1e-9) if mv.any() else lengths
            g = gf.odometry_from_trajectory(traj, frame_dt, heading_source="rotation",
                                            step_lengths_m=rel)
            g.metric, g.unit_seconds, g.dt = False, frame_dt, dt.copy()
            out["vo_gflow"] = g
        if plane is not None:
            L = np.asarray(plane["lengths"], float)
            pm = gf.odometry_from_trajectory(traj, frame_dt, heading_source="rotation",
                                             step_lengths_m=L)
            pm.dt = dt.copy()
            out["vo_plane"] = pm
            mv = L > 0
            rel = L / max(float(L[mv].mean()), 1e-9) if mv.any() else L
            pr = gf.odometry_from_trajectory(traj, frame_dt, heading_source="rotation",
                                             step_lengths_m=rel)
            pr.metric, pr.unit_seconds, pr.dt = False, frame_dt, dt.copy()
            out["vo_plane_rel"] = pr
    if lingbot is not None and len(lingbot.xz) == len(t):
        steps = np.linalg.norm(np.diff(lingbot.xz, axis=0), axis=1)
        mv = steps > 1e-9
        rel = steps / max(float(steps[mv].mean()), 1e-12) if mv.any() else steps
        lr = gf.odometry_from_trajectory(lingbot, frame_dt, heading_source="rotation",
                                         step_lengths_m=rel)
        lr.metric, lr.unit_seconds, lr.dt = False, frame_dt, dt.copy()
        out["vo_lingbot_rel"] = lr
        if plane is not None:
            L = np.asarray(plane["lengths"], float)
            both = mv & (L > 0)
            scale = float(np.median(L[both] / steps[both])) if both.sum() >= 10 else None
            if scale is not None:
                lp = gf.odometry_from_trajectory(lingbot, frame_dt, heading_source="rotation",
                                                 step_lengths_m=steps * scale)
                lp.dt = dt.copy()
                out["vo_lingbot_plane"] = lp
    factor = max(1, int(round(1.0 / (FILTER_HZ * frame_dt))))
    return {k: gf.aggregate_odometry(v, factor) for k, v in out.items()}


def filter_config(odom_kind: str, n: int, seed: int,
                  plane_spread: float | None = None) -> gf.FilterConfig:
    """Noise models fixed a priori per sensor (not tuned on these drives)."""
    if odom_kind == "vo_lingbot_plane":  # road-plane global scale, LingBot profile
        return gf.FilterConfig(n_particles=n, seed=seed, metric_logk_sigma=0.25,
                               metric_speed_dev_sigma=0.15)
    if odom_kind == "vo_plane":         # assumed camera height, GT-free step noise
        dev = 0.15 if plane_spread is None or not np.isfinite(plane_spread) else plane_spread
        return gf.FilterConfig(n_particles=n, seed=seed, metric_logk_sigma=0.25,
                               metric_speed_dev_sigma=float(np.clip(dev, 0.05, 0.3)))
    if odom_kind == "oracle":           # INS: speed ~1 %, yaw drift ~0
        return gf.FilterConfig(n_particles=n, seed=seed, metric_logk_sigma=0.05,
                               metric_speed_dev_sigma=0.03, delta_rw_deg=0.3)
    if odom_kind == "vo_oracle_speed":  # INS speed, VO heading noise model
        return gf.FilterConfig(n_particles=n, seed=seed, metric_logk_sigma=0.05,
                               metric_speed_dev_sigma=0.03)
    return gf.FilterConfig(n_particles=n, seed=seed)   # scale-free: defaults


def vo_diagnostics(ox: dict, traj, gflow, plane=None) -> dict:
    """How good is the monocular VO as odometry, measured against OXTS."""
    o = gf.odometry_from_trajectory(traj, 0.1, heading_source="rotation")
    ot = gf.odometry_from_trajectory(traj, 0.1, heading_source="translation")
    yaw = ox["yaw"][1:]
    moving = ox["vf"][1:] > 0.2
    out = {"frames": int(len(traj.xz)),
           "valid_pair_frac": float(np.mean(np.asarray(traj.valid)[1:])),
           "moving_frac_oxts": float(np.mean(moving)),
           "vo_zero_step_while_moving_frac": float(np.mean((o.step == 0) & moving))}
    for name, od in (("rotation", o), ("translation", ot)):
        ok = np.isfinite(od.heading) & moving
        if ok.sum() < 10:
            continue
        r = gf.wrap_angle(od.heading[ok] - yaw[ok])
        r = gf.wrap_angle(r - np.angle(np.mean(np.exp(1j * r))))     # remove yaw origin
        tt = ox["t"][1:][ok]
        slope = np.polyfit(tt, np.unwrap(r), 1)[0]
        dr = np.diff(r)
        out[f"heading_{name}"] = {
            "abs_err_deg_median": float(np.degrees(np.median(np.abs(r)))),
            "abs_err_deg_p90": float(np.degrees(np.percentile(np.abs(r), 90))),
            "drift_deg_per_min": float(np.degrees(slope) * 60),
            "step_noise_deg": float(np.degrees(np.std(gf.wrap_angle(dr)))),
            "mirror_check_corr": float(np.corrcoef(np.unwrap(od.heading[ok]),
                                                   np.unwrap(yaw[ok]))[0, 1]),
        }
    # turn-angle fidelity: heading change over the whole drive
    dist = np.cumsum(ox["vf"][1:] * np.diff(ox["t"]))
    out["distance_m"] = float(dist[-1])
    if gflow is not None:
        lengths, conf = gflow
        true = ox["vf"][1:] * np.diff(ox["t"])
        mv = moving & (lengths > 0)
        out["ground_flow"] = {"confidence": conf}
        if conf > 0 and mv.sum() > 10:
            ratio = lengths[mv] / true[mv]
            out["ground_flow"].update({
                "scale_median": float(np.median(ratio)),
                "rel_err_after_global_scale_median": float(np.median(np.abs(ratio / np.median(ratio) - 1))),
                "corr_with_true_step": float(np.corrcoef(lengths[mv], true[mv])[0, 1]),
            })
    if plane is not None:
        L = np.asarray(plane["lengths"], float)
        true = ox["vf"][1:] * np.diff(ox["t"])
        mv = moving & (L > 0)
        ratio = L[mv] / true[mv]
        out["road_plane"] = {
            "camera_height_assumed_m": plane["camera_height_m"],
            "coverage": plane["coverage"], "gt_free_rel_spread": plane["rel_spread"],
            "scale_median": float(np.median(ratio)),
            "rel_err_after_global_scale_median": float(
                np.median(np.abs(ratio / np.median(ratio) - 1))),
            "corr_with_true_step": float(np.corrcoef(L[mv], true[mv])[0, 1]),
            "distance_ratio": float(L.sum() / max(true.sum(), 1e-9)),
        }
    return out


# --------------------------------------------------------------------------
# runs (process pool)
# --------------------------------------------------------------------------

_NET_CACHE: dict = {}
_CRASH_DIR: Path | None = None
_CRASH_FH = None


def _init_worker(crash_dir: str) -> None:
    """Per-worker faulthandler: a native crash (access violation) otherwise
    surfaces only as an anonymous BrokenProcessPool in the parent."""
    import faulthandler

    global _CRASH_FH
    d = Path(crash_dir)
    d.mkdir(parents=True, exist_ok=True)
    _CRASH_FH = open(d / f"crash_{os.getpid()}.log", "w")
    faulthandler.enable(file=_CRASH_FH, all_threads=True)


def _net(graph_path: str) -> gf.RoadNetwork:
    if graph_path not in _NET_CACHE:
        import osmnx as ox
        _NET_CACHE[graph_path] = gf.RoadNetwork.from_graph(ox.load_graphml(graph_path))
    return _NET_CACHE[graph_path]


def safe_score_route(pred_latlon, wp) -> dict:
    """monocular_osm.metrics.score_route, guarded against its crash mode.

    score_route densifies the prediction to one vertex per 10 m and hands it to
    GEOS ``frechet_distance``, whose recursion overflows the C stack (a hard
    process kill, not an exception) beyond roughly 5-10 thousand vertices
    (~50-100 km of path). An online mode track that hops between distant
    hypotheses easily exceeds that, so the densification step is scaled to keep
    <= 3000 vertices. It only affects the Hausdorff/Frechet discretisation
    (bounded by the step); ``gt_to_pred_mean_m`` — the benchmark's headline —
    is exact point-to-curve distance and independent of it.
    """
    from monocular_osm.metrics import local_transformer, project, score_route

    pred = np.atleast_2d(np.asarray(pred_latlon, float))
    length = 0.0
    if len(pred) > 1:
        tf = local_transformer(float(wp[:, 0].mean()), float(wp[:, 1].mean()))
        xy = project(pred, tf)
        length = float(np.hypot(*np.diff(xy, axis=0).T).sum())
    step = max(10.0, length / 3000.0)
    out = score_route(pred, wp, densify_m=step).to_dict()
    out["densify_m_used"] = step
    return out


def run_one(job: dict) -> dict:
    net = _net(job["graph_path"])
    odom: gf.Odometry = job["odom"]
    gt_latlon = job["gt_latlon"]
    gt_xy = net.local_to_crs(net.latlon_to_local(gt_latlon))
    cfg = filter_config(job["odometry"], job["n"], job["seed"], job.get("plane_spread"))
    if job.get("heading_sigma_deg") is not None:
        cfg.heading_sigma_deg = float(job["heading_sigma_deg"])
    res = gf.run_filter(net, odom, cfg, diagnostic_xy=gt_xy)
    ev = gf.evaluate_against_gt(res, gt_latlon)
    ev.update({k: job[k] for k in ("drive", "map", "odometry", "seed", "n")})
    ev["runtime_s"] = res.runtime_s
    ev["confidence"] = res.confidence.tolist()
    ev["n_modes"] = res.n_modes.tolist()
    ev["gt_mass"] = res.gt_mass.tolist()
    ev["k_mode_final"] = float(res.k_mode[-1])
    ev["times"] = res.times.tolist()
    ev["mode_latlon_final"] = res.mode_latlon[-1].tolist()
    ev["n_steps"] = int(len(odom))
    # README-comparable route error: the smoothed (genealogy) route against the
    # clip's committed GT waypoints, with the benchmark's own scorer.
    wp = job["gt_waypoints"]
    if res.smoothed_latlon is not None:
        sm = res.smoothed_latlon[np.isfinite(res.smoothed_latlon).all(axis=1)]
        sc = safe_score_route(sm, wp) if len(sm) else None
    else:
        sc = None
    pt = safe_score_route(res.mode_latlon[-1], wp)         # final mode as a point
    mode_route = safe_score_route(res.mode_latlon, wp)     # online mode track
    ev["route_gt_to_pred_mean_m"] = (sc or pt)["gt_to_pred_mean_m"]
    ev["route_symmetric_mean_m"] = (sc or pt)["symmetric_mean_m"]
    ev["route_frechet_m"] = (sc or pt)["frechet_m"]
    ev["route_score"] = sc
    ev["mode_track_score"] = mode_route
    ev["final_point_score"] = pt
    return ev


def summarise(runs: list[dict]) -> dict:
    def ms(vals):
        v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
        return (float(v.mean()), float(v.std())) if len(v) else (None, None)
    ttl = [r["time_to_localize_s"] for r in runs]
    loc = [x for x in ttl if x is not None]
    out = {"n_seeds": len(runs), "localized_seeds": len(loc),
           "time_to_localize_s": ms(loc) if loc else (None, None)}
    for k in ("mean_err_m", "median_err_m", "final_err_m", "frac_steps_lt10m",
              "frac_steps_lt25m", "final_mass_within_25m_of_gt", "final_confidence",
              "smoothed_mean_err_m", "runtime_s", "k_mode_final",
              "route_gt_to_pred_mean_m", "route_symmetric_mean_m", "route_frechet_m"):
        out[k] = ms([r.get(k) for r in runs])
    return out


def fmt(mstd, nd=0, pct=False):
    m, s = mstd
    if m is None:
        return "—"
    if pct:
        return f"{100 * m:.0f} ± {100 * s:.0f} %"
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def write_markdown(path: Path, summary: dict, meta: dict) -> None:
    L = ["# Road-graph particle filter on KITTI (GPS-free, odometry + OSM only)", "",
         f"Generated by `scripts/eval_graph_filter.py` at commit `{meta.get('commit')}`; "
         f"{meta['seeds']} seeds per row, mean ± std over seeds. Filter at "
         f"{FILTER_HZ:.0f} Hz. Map A = fixed box {BOX_A} (W,S,E,N), same for both "
         "drives; map B = Karlsruhe city (osmnx drive network).", "",
         "| Drive | Map | Odometry | N | Localized seeds | Time to localize (s) | "
         "Final err (m) | Mean err (m) | Steps < 10 m | Mass ≤ 25 m of GT at end | "
         "Smoothed route err (m) | README-style route err (m) | Runtime (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, s in summary.items():
        d, m, o = key.split("|")
        L.append(f"| {d} | {m} | {o} | {meta['n'][m]:,} | {s['localized_seeds']}/{s['n_seeds']} | "
                 f"{fmt(s['time_to_localize_s'], 1)} | {fmt(s['final_err_m'])} | "
                 f"{fmt(s['mean_err_m'])} | {fmt(s['frac_steps_lt10m'], pct=True)} | "
                 f"{fmt(s['final_mass_within_25m_of_gt'], 2)} | {fmt(s['smoothed_mean_err_m'])} | "
                 f"{fmt(s['route_gt_to_pred_mean_m'])} | {fmt(s['runtime_s'])} |")
    L += ["", "README-style route err = `monocular_osm.metrics.score_route(...).gt_to_pred_mean_m` "
          "of the smoothed route vs the committed GT waypoints (the benchmark's headline "
          "metric; one-directional, see metrics.py).", "",
          "Existing pipeline on the same drives (README, GPS-free, mean route error): "
          "KITTI 0009 5,568 m, 0033 7,126 m; city-centroid baseline 2,645 m / 6,843 m.", ""]
    if meta.get("vo_diagnostics"):
        L += ["## Monocular VO vs OXTS", "", "```json",
              json.dumps(meta["vo_diagnostics"], indent=1), "```", ""]
    if meta.get("map_coverage"):
        L += ["## Ground truth vs map coverage (max distance GT -> nearest road)", "",
              "```json", json.dumps(meta["map_coverage"], indent=1), "```", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def plot(path: Path, runs: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drives = sorted({r["drive"] for r in runs})
    fig, axes = plt.subplots(1, len(drives), figsize=(7 * len(drives), 4.6), squeeze=False)
    styles = {"A": "-", "B": "--"}
    colors = {"oracle": "#1b7837", "oracle_unit": "#5aae61", "vo_oracle_speed": "#e08214",
              "vo": "#c51b7d", "vo_gflow": "#2166ac", "vo_plane": "#762a83",
              "vo_plane_rel": "#9970ab", "vo_lingbot_rel": "#35978f",
              "vo_lingbot_plane": "#01665e"}
    for ax, d in zip(axes[0], drives):
        combos = sorted({(r["map"], r["odometry"]) for r in runs if r["drive"] == d})
        for m, o in combos:
            rs = [r for r in runs if (r["drive"], r["map"], r["odometry"]) == (d, m, o)]
            t = np.asarray(rs[0]["times"])
            e = np.array([r["errors_m"] for r in rs])
            ax.plot(t, np.median(e, axis=0), styles.get(m, "-"), color=colors.get(o, "k"),
                    lw=1.6, label=f"map {m}, {o}")
            ax.fill_between(t, e.min(axis=0), e.max(axis=0), color=colors.get(o, "k"),
                            alpha=0.08, lw=0)
        ax.axhline(10, color="0.5", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(1, 3e4)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("posterior-mode error vs OXTS GT (m)")
        ax.set_title(f"KITTI {d} (median over seeds, band = min-max)")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--drives", nargs="+", default=list(DRIVES))
    ap.add_argument("--maps", nargs="+", default=list(MAPS))
    ap.add_argument("--odometry", nargs="+",
                    default=["oracle", "oracle_unit", "vo_oracle_speed", "vo", "vo_gflow"])
    ap.add_argument("--odometry-B", nargs="+", default=["oracle", "vo"],
                    help="odometry variants also run on the (expensive) city map B; "
                         "the ablations run on map A only")
    ap.add_argument("--heading-sigma-deg", type=float, default=None,
                    help="override FilterConfig.heading_sigma_deg (sensitivity runs)")
    ap.add_argument("--camera-height-m", type=float, default=None,
                    help="assumed camera height for vo_plane (default "
                         "plane_speed.DEFAULT_CAMERA_HEIGHT_M; sensitivity runs)")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument("--seeds", type=int, default=5, help="seeds per cell (>= 5 for the report)")
    ap.add_argument("--particles-A", type=int, default=300_000)
    ap.add_argument("--particles-B", type=int, default=600_000)
    ap.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    root = args.root or find_root(HERE)
    out = args.out or (root / "output" / "graph_filter")
    out.mkdir(parents=True, exist_ok=True)
    n_by_map = {"A": args.particles_A, "B": args.particles_B}

    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(out / "osmnx_http_cache")

    graph_paths = {}
    nets = {}
    for m in args.maps:
        G = load_graph(m, out)
        graph_paths[m] = str(out / MAPS[m]["file"])
        nets[m] = gf.RoadNetwork.from_graph(G)
        print(f"map {m}: {nets[m].stats()}", flush=True)

    jobs, meta_vo, coverage = [], {}, {}
    gt_wp = {}
    for d in args.drives:
        gj = json.loads((root / "ground_truth" / f"kitti_drive_{d}.json").read_text(
            encoding="utf-8"))
        gt_wp[d] = np.array([[w["lat"], w["lon"]] for w in gj["waypoints"]])
    for d in args.drives:
        spec = DRIVES[d]
        ddir = root / "data" / "kitti" / "raw" / spec["date"] / spec["name"]
        ox_ = load_oxts(ddir)
        frame_dt = float(np.median(np.diff(ox_["t"])))
        K = kitti_K(root / "data" / "kitti" / "raw" / spec["date"] / "calib_cam_to_cam.txt")
        traj = gflow = plane = None
        if any(o.startswith("vo") for o in args.odometry):
            traj, _ = vo_trajectory(ddir, K, out / f"vo_{d}.npz")
        if "vo_gflow" in args.odometry and traj is not None:
            gcache = out / f"gflow_{d}.npz"
            if gcache.exists():
                z = np.load(gcache)
                gflow = (z["lengths"], float(z["confidence"]))
            else:
                video = root / "data" / "kitti" / spec["video"]
                import cv2
                h, w = cv2.imread(str(next((ddir / "image_02" / "data").glob("*.png")))).shape[:2]
                t0 = time.perf_counter()
                gflow = ground_flow_lengths(video, traj, K, (w, h))
                print(f"    ground flow: confidence {gflow[1]:.2f} "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)
                np.savez_compressed(gcache, lengths=gflow[0], confidence=gflow[1])
        if traj is not None and any(o.startswith("vo_plane") or o == "vo_lingbot_plane"
                                    for o in args.odometry):
            plane = plane_lengths(ddir, traj, K, out / f"plane_{d}.npz",
                                  args.camera_height_m)
        lingbot = None
        if any(o.startswith("vo_lingbot") for o in args.odometry):
            lpath = out / f"lingbot_{d}.npz"
            if lpath.exists():
                lingbot, lfi = lingbot_trajectory(lpath)
                if len(lingbot.xz) != len(ox_["t"]) or (lfi != np.arange(len(lfi))).any():
                    print(f"  {d}: LingBot poses are not one per frame; skipped", flush=True)
                    lingbot = None
                else:
                    meta_vo.setdefault(f"{d}_lingbot", vo_diagnostics(ox_, lingbot, None))
                    st = np.linalg.norm(np.diff(lingbot.xz, axis=0), axis=1)
                    true = ox_["vf"][1:] * np.diff(ox_["t"])
                    mvl = (ox_["vf"][1:] > 0.2) & (st > 0)
                    meta_vo[f"{d}_lingbot"]["step_profile"] = {
                        "scale_median": float(np.median(st[mvl] / true[mvl])),
                        "corr_with_true_step": float(np.corrcoef(st[mvl], true[mvl])[0, 1]),
                        "rel_err_after_global_scale_median": float(np.median(np.abs(
                            (st[mvl] / true[mvl]) / np.median(st[mvl] / true[mvl]) - 1)))}
            else:
                print(f"  {d}: no {lpath.name} (run scripts/run_lingbot_map.py)", flush=True)
        if traj is not None:
            meta_vo[d] = vo_diagnostics(ox_, traj, gflow, plane)
        variants = odometry_variants(ox_, traj, gflow, frame_dt, plane, lingbot)
        for m in args.maps:
            gxy = nets[m].latlon_to_local(ox_["latlon"])
            dmin = _gt_to_road_m(nets[m], gxy)
            coverage[f"{d}|{m}"] = {"max_m": float(dmin.max()), "median_m": float(np.median(dmin))}
            for o in args.odometry:
                if m == "B" and o not in args.odometry_B:
                    continue
                if o not in variants:
                    print(f"  skip {d}/{m}/{o}: not available", flush=True)
                    continue
                odom = variants[o]
                idx = np.concatenate([[0], odom.frame_index])
                for s in range(args.seeds):
                    jobs.append({"drive": d, "map": m, "odometry": o, "seed": s,
                                 "n": n_by_map[m], "graph_path": graph_paths[m],
                                 "gt_waypoints": gt_wp[d],
                                 "heading_sigma_deg": args.heading_sigma_deg,
                                 "plane_spread": (plane or {}).get("rel_spread"),
                                 "odom": odom, "gt_latlon": ox_["latlon"][idx]})
    print(f"{len(jobs)} runs on {args.workers} workers", flush=True)
    t0 = time.perf_counter()
    runs = []
    # largest first so the pool drains evenly
    jobs.sort(key=lambda j: -j["n"] * len(j["odom"]))
    ckpt = out / f"results_partial{'_' + args.tag if args.tag else ''}.json"
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(str(out / "crash_logs"),)) as pool:
        futures = [pool.submit(run_one, j) for j in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            runs.append(r)
            print(f"  [{len(runs)}/{len(jobs)}] {r['drive']} {r['map']} {r['odometry']:15s} "
                  f"seed {r['seed']}: final {r['final_err_m']:8.1f} m  "
                  f"route {r['route_gt_to_pred_mean_m']:7.0f} m  "
                  f"ttl {r['time_to_localize_s']}  conf {r['final_confidence']:.2f}  "
                  f"{r['runtime_s']:.0f} s", flush=True)
            ckpt.write_text(json.dumps(runs), encoding="utf-8")   # survive interruption
    wall = time.perf_counter() - t0
    ckpt.unlink(missing_ok=True)
    shutil.rmtree(out / "crash_logs", ignore_errors=True)

    summary = {}
    for d in args.drives:
        for m in args.maps:
            for o in args.odometry:
                rs = [r for r in runs if (r["drive"], r["map"], r["odometry"]) == (d, m, o)]
                if rs:
                    summary[f"{d}|{m}|{o}"] = summarise(rs)
    meta = {"commit": _git_commit(), "seeds": args.seeds, "n": n_by_map,
            "box_A_wsen": BOX_A, "filter_hz": FILTER_HZ, "baselines": BASELINES,
            "maps": {m: nets[m].stats() for m in nets}, "vo_diagnostics": meta_vo,
            "map_coverage": coverage, "wall_time_s": wall,
            "filter_configs": {o: asdict(filter_config(o, 0, 0)) for o in args.odometry}}
    meta["heading_sigma_deg_override"] = args.heading_sigma_deg
    tag = f"_{args.tag}" if args.tag else ""
    (out / f"results{tag}.json").write_text(json.dumps(
        {"meta": meta, "summary": summary, "runs": runs}, indent=1), encoding="utf-8")
    write_markdown(out / f"results{tag}.md", summary, meta)
    plot(out / f"error_vs_time{tag}.png", runs)
    print(f"wrote {out / f'results{tag}.json'}, .md, .png ({wall:.0f} s wall)", flush=True)
    return 0


def _gt_to_road_m(net: gf.RoadNetwork, xy: np.ndarray) -> np.ndarray:
    """Distance from each GT point to the nearest road segment (diagnostic)."""
    from scipy.spatial import cKDTree

    # densify segments to <= 5 m points; nearest-point distance is then within 2.5 m
    seg_len = np.diff(np.append(net.seg_gstart, net.total_length_m))
    reps = np.maximum(1, np.ceil(seg_len / 5.0).astype(int))
    sid = np.repeat(np.arange(len(seg_len)), reps)
    frac = (np.arange(len(sid)) - np.repeat(np.cumsum(reps) - reps, reps)) / np.repeat(reps, reps)
    pts = net.seg_xy0[sid] + net.seg_dir[sid] * (frac * seg_len[sid])[:, None]
    d, _ = cKDTree(pts).query(xy)
    return d


def _git_commit() -> str | None:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
