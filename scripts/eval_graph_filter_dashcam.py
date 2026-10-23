"""Pre-registered dashcam test of the road-graph particle filter with
road-plane speed (``benchmarks/preregistered_rules.json``, rule
``graph_filter_dashcam``).

For every manifest clip with a cached VO trajectory (KITTI excluded: see
``scripts/eval_graph_filter.py``): rebuild the frames that VO used (same
stride and window, decoded exactly as ``frame_extraction.extract_frames`` does,
downscaled to ``--width`` px with the intrinsics scaled to match, so the
road-plane geometry is unchanged), estimate metric step lengths with
``monocular_osm.plane_speed`` (assumed camera height 1.5 m), and run
``monocular_osm.graph_filter`` with the ``--graph-filter`` defaults on the
drive network inside the pipeline's own city-extent search disc. Scored
against the clip's ground truth beside the pipeline's error from benchmark
runs v3 (core) / v4 (overlay); the registered hypotheses G1-G2 are computed
literally. Exploratory: the speed estimate against the ground truth's timing.

    python scripts/eval_graph_filter_dashcam.py --seeds 3 --workers 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from monocular_osm import graph_filter as gf  # noqa: E402

LOCALIZED_M = 50.0
_VO_RE = re.compile(r"trajectory_v2_(?P<s>[\d.]+)-(?P<e>[\d.]+)_s(?P<stride>\d+)_f(?P<mf>[^_]+)_vo3\.npz$")


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def find_vo(slug_dirs) -> tuple[Path, dict] | None:
    for d in slug_dirs:
        for f in sorted(d.glob("trajectory_v2_*_vo3.npz")):
            m = _VO_RE.search(f.name)
            if m:
                return f, {"start": float(m["s"]), "end": float(m["e"]),
                           "stride": int(m["stride"]),
                           "max_frames": None if m["mf"] == "auto" else int(m["mf"])}
    return None


def load_traj(path: Path):
    from monocular_osm.visual_odometry import Trajectory

    z = np.load(path)
    return Trajectory(centers=z["centers"], xz=z["xz"], valid=z["valid"],
                      n_inliers=list(z["n_inliers"]), rotations=z["rotations"],
                      translations=z["translations"])


def stream_gray(video: Path, start_sec: float, end_sec: float, stride: int,
                max_frames: int | None, width: int):
    """The frames ``extract_frames`` keeps (same selection and timestamps),
    as downscaled grayscale; returns (frames, timestamps, (w, h))."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame, end_frame = int(start_sec * fps), int(end_sec * fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames, ts, idx, size = [], [], start_frame, None
    while idx < end_frame and (max_frames is None or len(frames) < max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if (idx - start_frame) % stride == 0:
            h, w = frame.shape[:2]
            s = width / float(w)
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (width, int(round(h * s))), interpolation=cv2.INTER_AREA)
            size = (g.shape[1], g.shape[0])
            frames.append(g)
            ts.append(pos_ms / 1000.0 if pos_ms and pos_ms > 0 else idx / fps)
        idx += 1
    cap.release()
    return frames, np.asarray(ts, float), size


def haversine_m(a, b):
    a, b = np.radians(np.atleast_2d(a)), np.radians(np.atleast_2d(b))
    h = (np.sin((b[:, 0] - a[:, 0]) / 2) ** 2
         + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin((b[:, 1] - a[:, 1]) / 2) ** 2)
    return 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def speed_diagnostics(lengths, ts, wps) -> dict:
    """Exploratory: estimated vs ground-truth distance per GT interval."""
    t = np.array([w["t_sec"] for w in wps], float)
    ll = np.array([[w["lat"], w["lon"]] for w in wps], float)
    est, true = [], []
    step_t = ts[1:]
    for k in range(len(t) - 1):
        m = (step_t > t[k]) & (step_t <= t[k + 1])
        if m.sum() < 3:
            continue
        est.append(float(lengths[m].sum()))
        true.append(float(haversine_m(ll[k], ll[k + 1])[0]))
    est, true = np.array(est), np.array(true)
    ok = true > 5.0
    if ok.sum() < 5:
        return {"n_intervals": int(ok.sum())}
    r = est[ok] / true[ok]
    return {"n_intervals": int(ok.sum()), "scale_median": float(np.median(r)),
            "corr": float(np.corrcoef(est[ok], true[ok])[0, 1]),
            "rel_err_after_global_scale_median": float(np.median(np.abs(r / np.median(r) - 1))),
            "distance_ratio": float(est[ok].sum() / true[ok].sum())}


def search_disc(city: str) -> tuple[float, float, float] | None:
    """Centre (the centroid baseline's geocode) and the pipeline's city-extent
    radius (stages/prior.py: distance to bbox centre + half-diagonal, 3-8 km)."""
    from monocular_osm import baselines as Bl
    from monocular_osm.geo import geodesic_distance_m
    from monocular_osm.text_anchor import city_extent_radius

    cen = Bl.city_centroid(city, allow_network=True)
    ext = city_extent_radius(city, ROOT / "data" / "geocode_cache.json")
    if cen is None or ext is None:
        return None
    lat, lon = cen.route_latlon[0]
    r = geodesic_distance_m(lat, lon, ext[0], ext[1]) + float(ext[2])
    return float(lat), float(lon), float(np.clip(r, 3000.0, 8000.0))


def disc_graph(clip_id: str, city: str, disc, out: Path, attempts: int = 6) -> str:
    """The drive network of the search disc via the pipeline's own fetcher
    (osm_data.fetch_city_graph: transient-error retries, a pinned reachable
    Overpass address, atomic cache), with a slower outer backoff for the public
    server's 'request queue is full' rate limiting."""
    from monocular_osm.osm_data import fetch_city_graph

    path = out / f"graph_{clip_id}.graphml"
    for k in range(attempts):
        try:
            fetch_city_graph(city, path, around=(disc[0], disc[1], disc[2]))
            return str(path)
        except Exception as e:
            if k == attempts - 1:
                raise
            wait = min(900, 120 * 2 ** k)
            print(f"  graph for {clip_id} failed ({type(e).__name__}: {str(e)[:80]}); "
                  f"retrying in {wait} s", flush=True)
            time.sleep(wait)
    return str(path)


# --------------------------------------------------------------------------
# filter (one process per job, network cached per worker as in eval_graph_filter)
# --------------------------------------------------------------------------

def run_job(job: dict) -> dict:
    import eval_graph_filter as E

    net = E._net(job["graph_path"])
    cfg = gf.FilterConfig(n_particles=job["n"], seed=job["seed"], metric_logk_sigma=0.25,
                          metric_speed_dev_sigma=job["speed_dev"])
    gt_xy = net.local_to_crs(net.latlon_to_local(job["gt_latlon"]))
    res = gf.run_filter(net, job["odom"], cfg, diagnostic_xy=gt_xy)
    ev = gf.evaluate_against_gt(res, job["gt_latlon"])
    sm = res.smoothed_latlon
    sm = sm[np.isfinite(sm).all(axis=1)] if sm is not None else None
    sc = E.safe_score_route(sm if sm is not None and len(sm) else res.mode_latlon[-1],
                            job["gt_waypoints"])
    return {"clip": job["clip"], "seed": job["seed"], "final_err_m": ev["final_err_m"],
            "time_to_localize_s": ev["time_to_localize_s"],
            "frac_steps_lt10m": ev["frac_steps_lt10m"],
            "final_confidence": ev["final_confidence"],
            "route_symmetric_m": sc["symmetric_mean_m"], "runtime_s": res.runtime_s}


def pipeline_errors(manifest) -> dict:
    """Pipeline symmetric error per clip: v3 for core, v4 for overlay."""
    import benchmark as B

    runs = ROOT / ".claude" / "worktrees" / "bench" / "output" / "benchmark_runs"
    out = {}
    for run, fleet in (("v3", "core"), ("v4", "overlay")):
        if not (runs / run).is_dir():
            continue
        s = B.aggregate(runs / run, manifest, allow_network=False, n_resamples=99, root=ROOT)
        for cid in s["fleets"].get(fleet, {}).get("clips", []):
            m = s["clips"][cid]["methods"]
            out[cid] = {"run": run,
                        "pipeline": m["pipeline"]["symmetric_mean_m"]["median"],
                        "centre": m["city_centroid"]["symmetric_mean_m"]["median"]}
    return out


def main(argv=None) -> int:
    import benchmark as B
    import eval_graph_filter as E
    from monocular_osm.plane_speed import plane_step_lengths
    from monocular_osm.visual_odometry import default_intrinsics

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--particles", type=int, default=600_000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--camera-height-m", type=float, default=1.5)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "graph_filter_dashcam")
    ap.add_argument("--graph-dir", type=Path, default=None,
                    help="where the disc graphs are cached (default: --out)")
    ap.add_argument("--horizon", choices=("principal", "foe"), default="principal",
                    help="plane_speed road band (POST HOC: 'foe' was designed after the "
                         "registered run; the registered setting is 'principal')")
    ap.add_argument("--mask-static", action="store_true",
                    help="plane_speed static-overlay mask (POST HOC, see --horizon)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = B.load_manifest(ROOT / "benchmarks" / "manifest.json")
    clips = [c for c in manifest["clips"] if "url" in c["video"]
             and (c["id"] in args.clips if args.clips else True)]

    meta, jobs = {}, []
    for e in clips:
        slug_dirs = B.candidate_slug_dirs(e, e["city"], ROOT / "data")
        vo = find_vo(slug_dirs)
        video = next((d / "input.mp4" for d in slug_dirs if (d / "input.mp4").exists()), None)
        if vo is None or video is None:
            print(f"SKIP {e['id']}: no cached VO trajectory or video", flush=True)
            continue
        traj = load_traj(vo[0])
        cache = args.out / f"plane_{e['id']}.npz"
        if cache.exists():
            z = np.load(cache)
            lengths, ts, spread, cov = z["lengths"], z["ts"], float(z["spread"]), float(z["coverage"])
        else:
            t0 = time.perf_counter()
            frames, ts, size = stream_gray(video, vo[1]["start"], vo[1]["end"], vo[1]["stride"],
                                           vo[1]["max_frames"], args.width)
            if len(frames) != len(traj.xz):
                print(f"SKIP {e['id']}: {len(frames)} frames decoded, VO has {len(traj.xz)}",
                      flush=True)
                continue
            ps = plane_step_lengths(frames, traj, default_intrinsics(*size),
                                    camera_height_m=args.camera_height_m,
                                    horizon=args.horizon, mask_static=args.mask_static)
            lengths, spread, cov = ps.lengths_m, ps.rel_spread, ps.coverage
            np.savez_compressed(cache, lengths=lengths, ts=ts, spread=spread, coverage=cov)
            print(f"{e['id']}: plane speed on {len(frames)} frames in "
                  f"{time.perf_counter() - t0:.0f} s, coverage {cov:.2f}", flush=True)
        wps = json.loads((ROOT / e["gt"]).read_text(encoding="utf-8"))["waypoints"]
        disc = search_disc(e["city"])
        if disc is None:
            print(f"SKIP {e['id']}: no search disc for {e['city']!r}", flush=True)
            continue
        gt_ll = np.array([[w["lat"], w["lon"]] for w in wps])
        inside = float(np.mean(haversine_m(np.repeat([disc[:2]], len(gt_ll), 0), gt_ll) <= disc[2]))
        frame_dt = float(np.median(np.diff(ts)))
        odom = gf.odometry_from_trajectory(traj, frame_dt, heading_source="rotation",
                                           step_lengths_m=lengths)
        odom.dt = np.diff(ts)
        odom = gf.aggregate_odometry(odom, max(1, int(round(0.5 / frame_dt))))
        step_t = ts[np.concatenate([[0], odom.frame_index])]
        tw = np.array([w["t_sec"] for w in wps], float)
        gt_step = np.column_stack([np.interp(step_t, tw, gt_ll[:, 0]),
                                   np.interp(step_t, tw, gt_ll[:, 1])])
        graph_path = disc_graph(e["id"], e["city"], disc, args.graph_dir or args.out)
        meta[e["id"]] = {"city": e["city"], "fleet": e["fleet"], "disc": list(disc),
                         "route_inside_disc": inside,
                         "vo_valid_frac": float(np.mean(np.asarray(traj.valid))),
                         "plane_coverage": cov, "plane_spread": spread,
                         "speed_vs_gt": speed_diagnostics(lengths, ts, wps),
                         "n_steps": int(len(odom))}
        print(f"{e['id']}: disc r={disc[2]:.0f} m, route inside {inside:.0%}, "
              f"speed {meta[e['id']]['speed_vs_gt']}", flush=True)
        dev = float(np.clip(spread if np.isfinite(spread) else 0.15, 0.05, 0.3))
        for s in range(args.seeds):
            jobs.append({"clip": e["id"], "seed": s, "n": args.particles, "graph_path": graph_path,
                         "odom": odom, "gt_latlon": gt_step, "gt_waypoints": gt_ll,
                         "speed_dev": dev})

    runs = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=E._init_worker,
                             initargs=(str(args.out / "crash_logs"),)) as pool:
        for fut in as_completed([pool.submit(run_job, j) for j in jobs]):
            r = fut.result()
            runs.append(r)
            print(f"  {r['clip']:22s} seed {r['seed']}: final {r['final_err_m']:9.1f} m  "
                  f"route {r['route_symmetric_m']:8.0f} m  {r['runtime_s']:.0f} s", flush=True)
            (args.out / "runs_partial.json").write_text(json.dumps(runs), encoding="utf-8")

    pipe = pipeline_errors(manifest)
    table, wins, losses = {}, 0, 0
    for cid, m in meta.items():
        rs = [r for r in runs if r["clip"] == cid]
        if not rs:
            continue
        fin = float(np.median([r["final_err_m"] for r in rs]))
        route = float(np.median([r["route_symmetric_m"] for r in rs]))
        p = (pipe.get(cid) or {}).get("pipeline")
        if p is not None and np.isfinite(p) and route != p:
            wins += route < p
            losses += route > p
        table[cid] = {**m, "filter_final_m": fin, "filter_route_m": route,
                      "localized": fin < LOCALIZED_M, "pipeline_m": p,
                      "centre_m": (pipe.get(cid) or {}).get("centre"),
                      "pipeline_run": (pipe.get(cid) or {}).get("run")}
    from scipy.stats import binomtest
    res = {"clips": table, "runs": runs,
           "G1": any(t["localized"] for t in table.values()),
           "G2": {"filter_better": int(wins), "filter_worse": int(losses),
                  "sign_test_p": (float(binomtest(wins, wins + losses, 0.5).pvalue)
                                  if wins + losses else None),
                  "holds": wins > losses}}
    (args.out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "runs" and k != "clips"}, indent=1))
    for cid, t in table.items():
        print(f"{cid:22s} inside {t['route_inside_disc']:4.0%}  filter final {t['filter_final_m']:8.0f} "
              f"route {t['filter_route_m']:8.0f} | pipeline {t['pipeline_m']} | centre {t['centre_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
