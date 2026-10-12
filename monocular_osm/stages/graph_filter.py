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
provides no speed — every step is normalised to unit length — and its heading
drifts 8-14 deg/min; with it the filter localizes on 2 of 5 seeds on a
box-sized map and 0 of 5 on a whole city, and its reported confidence is badly
miscalibrated (only 3 of 14 runs with confidence > 0.8 were right). The channel
is therefore off by default, never feeds the headline answer, and says so in
its own output (``metric_odometry: false``, ``confidence_calibrated: false``).
A trajectory source with metric steps (VGGT-Long, DA3, MapAnything) is the
intended way to use it; see ``monocular_osm/graph_filter.py``.
"""

from __future__ import annotations

import numpy as np

from .context import _record_channel, RunContext

_MAX_ROUTE_POINTS = 1000
#: The evaluation ran the filter at ~2 Hz; finer steps cost time, not accuracy.
_TARGET_STEP_S = 0.5


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
        odom = odometry_from_trajectory(traj, frame_dt, heading_source="rotation")
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
        res = run_filter(net, odom,
                         FilterConfig(n_particles=int(cfg.graph_filter_particles),
                                      seed=int(cfg.graph_filter_seed)),
                         position_obs=obs)
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
          f"{res.runtime_s:.0f} s) - confidence is NOT calibrated for unit-step VO")

    ctx.result["graph_filter"] = {
        "n_particles": int(cfg.graph_filter_particles),
        "seed": int(cfg.graph_filter_seed),
        "metric_odometry": False,
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
        "ranking": "graph_filter(unit-step VO)",
    }
    if times is not None:
        report["route_t_sec"] = [round(float(t), 2) for t in times]
    ctx.result["filter_position"] = report
    _record_channel(ctx.channel_status, "graph_filter", "ok",
                    confidence=round(conf, 4), n_modes=n_modes)
