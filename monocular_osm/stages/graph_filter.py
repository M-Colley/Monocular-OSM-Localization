"""Stage 9b: opt-in road-graph particle filter (``--graph-filter``).

Runs :mod:`monocular_osm.graph_filter` — sequential Monte Carlo localisation on
the directed OSM road graph (Brubaker et al., CVPR 2013 style) — on the visual
odometry and stores its route beside, never instead of, the matcher's answer:
``result["graph_filter"]`` carries the filter's summary and
``result["filter_position"]`` a position report the benchmark can score as a
separate method.

**Read this before relying on it.** The filter reaches Brubaker-level accuracy
given metric odometry (KITTI 0033 with OXTS speed + heading: localized in
~25 s, 7 m final error, on a box or the whole city). This repo's monocular VO
provides no speed — every step is normalised to unit length — and with those
unit steps the filter localizes on 2 of 5 seeds on a box-sized map and 0 of 5
on a whole city. So by default (``graph_filter_speed="plane"``) the steps are
given metric lengths from the road plane and an assumed camera height
(:mod:`monocular_osm.plane_speed`); with them KITTI 0033 localized on 5 of 5
seeds on the whole Karlsruhe map (final ~10 m), though only late in the drive,
and the short drive 0009 still fails. Evidence is one 165 s drive: the channel
stays off by default, never feeds the headline answer, and its confidence is
not calibrated (``confidence_calibrated: false`` in its output).
"""

from __future__ import annotations

import numpy as np

from .context import _record_channel, RunContext

_MAX_ROUTE_POINTS = 1000
#: The evaluation ran the filter at ~2 Hz; finer steps cost time, not accuracy.
_TARGET_STEP_S = 0.5


#: Below this fraction of moving steps with a road-plane estimate, the speed
#: profile is mostly interpolation: use unit steps and say so.
_MIN_PLANE_COVERAGE = 0.2


def _plane_speed(ctx: RunContext):
    """Road-plane step lengths for the VO steps, or ``None`` (unit steps)."""
    from ..plane_speed import plane_step_lengths
    from ..visual_odometry import default_intrinsics

    imgs = getattr(ctx.frames, "frames", None)
    if imgs is None or len(imgs) != len(ctx.traj.xz):
        print("      -> no frames for road-plane speed; unit steps")
        return None
    import cv2

    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if np.ndim(f) == 3 else f for f in imgs]
    h, w = gray[0].shape[:2]
    ps = plane_step_lengths(gray, ctx.traj, default_intrinsics(w, h),
                            camera_height_m=float(ctx.cfg.camera_height_m))
    print(f"      -> road-plane speed: {ps.coverage:.0%} of moving steps measured, "
          f"spread {ps.rel_spread:.2f}, camera height assumed {ps.camera_height_m} m")
    if ps.coverage < _MIN_PLANE_COVERAGE:
        print("      -> too few road-plane estimates; unit steps")
        return None
    return ps


def graph_filter_channel(ctx: RunContext) -> None:
    """Run the particle filter on the VO trajectory (opt-in); see module doc."""
    cfg = ctx.cfg
    if not cfg.graph_filter:
        return

    road, traj, frames = ctx.road, ctx.traj, ctx.frames
    if road is None or traj is None or frames is None or len(frames.timestamps) < 3:
        _record_channel(ctx.channel_status, "graph_filter", "inactive",
                        reason="no road graph / trajectory")
        return
    try:
        from ..graph_filter import (
            FilterConfig,
            PositionObservation,
            RoadNetwork,
            aggregate_odometry,
            odometry_from_trajectory,
            run_filter,
        )
        from ..position import latlon_to_xy, xy_to_latlon

        print(f"[9b] Road-graph particle filter "
              f"({cfg.graph_filter_particles:,} particles, seed {cfg.graph_filter_seed})")
        frame_dt = float(np.median(np.diff(np.asarray(frames.timestamps, float))))
        speed = _plane_speed(ctx) if cfg.graph_filter_speed == "plane" else None
        odom = odometry_from_trajectory(
            traj, frame_dt, heading_source="rotation",
            step_lengths_m=None if speed is None else speed.lengths_m)
        factor = max(1, int(round(_TARGET_STEP_S / max(frame_dt, 1e-3))))
        odom = aggregate_odometry(odom, factor)
        net = RoadNetwork.from_graph(road.graph)

        obs: list = []
        if cfg.graph_filter_vpr and ctx.vpr_track is not None and ctx.anchor_origin == "vpr":
            # Retrieval positions are absolute but noisy and sometimes wrong
            # (a look-alike street across the city): a wide sigma and a high
            # outlier probability keep one bad frame from pulling the cloud.
            idx, latlons, _sims = ctx.vpr_track
            xy = latlon_to_xy(np.asarray(latlons, float), road.crs)
            for fi, (x, y) in zip(np.asarray(idx, int), xy):
                obs.append(PositionObservation(step=int(round(fi / factor)),
                                               x=float(x), y=float(y),
                                               sigma_m=40.0, outlier=0.4))
        fcfg = FilterConfig(n_particles=int(cfg.graph_filter_particles),
                            seed=int(cfg.graph_filter_seed))
        if speed is not None:
            # As in the KITTI evaluation: a wide global-scale prior for the
            # assumed camera height, per-step noise from the estimator's own
            # (ground-truth-free) spread.
            fcfg.metric_logk_sigma = 0.25
            dev = speed.rel_spread if np.isfinite(speed.rel_spread) else 0.15
            fcfg.metric_speed_dev_sigma = float(np.clip(dev, 0.05, 0.3))
        res = run_filter(net, odom, fcfg, position_obs=obs)
    except Exception as e:  # an experimental channel must never sink the run
        print(f"      -> graph filter failed ({type(e).__name__}: {e})")
        _record_channel(ctx.channel_status, "graph_filter", "failed", error=e)
        return

    route = res.smoothed_latlon if res.smoothed_latlon is not None else res.mode_latlon
    route = np.asarray(route, dtype=float)
    keep = np.unique(np.linspace(0, len(route) - 1,
                                 min(len(route), _MAX_ROUTE_POINTS)).round().astype(int))
    route = route[keep]
    times = np.asarray(res.times, float)[keep] if len(res.times) == len(res.mode_latlon) else None
    final = np.asarray(res.mode_latlon, float)[-1]
    conf = float(res.confidence[-1])
    n_modes = int(res.n_modes[-1])
    print(f"      -> final mode {final[0]:.5f}, {final[1]:.5f}  "
          f"(posterior mass near it {conf:.2f}, {n_modes} mode(s), "
          f"{res.runtime_s:.0f} s) - confidence is NOT calibrated")

    ctx.result["graph_filter"] = {
        "n_particles": int(cfg.graph_filter_particles),
        "seed": int(cfg.graph_filter_seed),
        "metric_odometry": speed is not None,
        "speed_source": "road_plane" if speed is not None else "unit_steps",
        "road_plane": speed.to_dict() if speed is not None else None,
        "uses_vpr_observations": bool(obs),
        "n_position_observations": len(obs),
        "final_mode_latlon": [round(float(final[0]), 6), round(float(final[1]), 6)],
        "confidence": round(conf, 4),
        "confidence_calibrated": False,
        "n_modes": n_modes,
        "n_reinit": int(res.n_reinit),
        "runtime_s": round(float(res.runtime_s), 1),
        "map": res.map_stats,
    }
    report = {
        "source": "graph_filter",
        "latitude": round(float(route[0][0]), 6),
        "longitude": round(float(route[0][1]), 6),
        "route_latlon": [[round(float(a), 6), round(float(b), 6)] for a, b in route],
        "ranking": ("graph_filter(VO + road-plane speed)" if speed is not None
                    else "graph_filter(unit-step VO)"),
    }
    if times is not None:
        report["route_t_sec"] = [round(float(t), 2) for t in times]
    ctx.result["filter_position"] = report
    _record_channel(ctx.channel_status, "graph_filter", "ok",
                    confidence=round(conf, 4), n_modes=n_modes)
