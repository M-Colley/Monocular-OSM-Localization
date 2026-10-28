"""Stage 9b: opt-in road-graph particle filter (``--graph-filter``).

Runs :mod:`monocular_osm.graph_filter` — sequential Monte Carlo localisation on
the directed OSM road graph (Brubaker et al., CVPR 2013 style) — on the visual
odometry. The companion mode stores its route beside the matcher's answer:
``result["graph_filter"]`` carries the filter's summary and
``result["filter_position"]`` a position report the benchmark can score as a
separate method. The explicit ``--answer filter`` mode skips retrieval/matching,
uses an independent whole-city graph and selects that route as the answer.

**Evidence has two input regimes.** The historical KITTI evaluation used raw
camera PNGs, calibrated rectified intrinsics and OXTS sensor timestamps. Its
road-plane-speed variant localized drive 0033 on the whole Karlsruhe map in
5 of 5 seeds, with final error around 10 m. Those results establish behavior
under that research setup; default pipeline settings require separate tests.

A fresh MP4-plus-city-only probe used generic 70-degree intrinsics, video
presentation times, assumed camera height 1.5 m, stride 1 and 600,000 particles.
Seeds 0-4 localized at 76.5-79.5 video seconds: mean mode error thereafter
5.35 +/- 0.17 m, final mode error 13.38 +/- 0.13 m, smoothed-route symmetric
error 5.02 +/- 0.24 m against the committed waypoints. Predictions and hashes
were frozen before separate scoring; seeds 1-4 reused the exact frozen
MP4-derived VO and plane odometry, changing only the PF seed. Whole-run online
mean error was still 4.36 km because the early modes were wrong. This tests
Monte Carlo robustness on one drive with explicit experimental settings; see
``docs/graph_filter_kitti.md`` for provenance and limitations.

By default (``graph_filter_speed="plane"``), metric lengths come from the road
plane and an assumed camera height (:mod:`monocular_osm.plane_speed`). Unit-step
VO alone failed all five historical city-map runs. The channel stays off by
default, never feeds the headline answer, and its confidence is uncalibrated
(``confidence_calibrated: false`` in its output).
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


def _whole_city_map(ctx):
    """Load the filter's independent city map, without any retrieval/seed crop."""
    from pathlib import Path
    import osmnx as ox
    from ..graph_filter import fetch_drive_graph
    from ..provenance import file_sha256

    explicit = getattr(ctx.cfg, "graph_filter_map_path", None)
    if explicit is not None:
        path = Path(explicit).resolve(strict=True)
        graph = ox.load_graphml(path)
        source = "explicit_cache"
    else:
        slug = ctx.cfg.city.replace(",", "").replace(" ", "_")
        path = (ctx.cfg.data_dir / f"{slug}_graph_filter_city.graphml").resolve()
        before = file_sha256(path) if path.is_file() else None
        graph = fetch_drive_graph(place=ctx.cfg.city, cache_path=path)
        source = "city_cache" if before is not None and before == file_sha256(path) else "named_city_fetch"
    metadata = {"scope": "whole_city", "city": ctx.cfg.city, "query": ctx.cfg.city,
                "cache_path": str(path), "configured_path": str(explicit) if explicit is not None else None,
                "sha256": file_sha256(path), "source": source, "around": None, "bbox": None}
    return graph, metadata


def _plane_speed(ctx: RunContext):
    """Road-plane step lengths for the VO steps, or ``None`` (unit steps)."""
    from ..plane_speed import plane_step_lengths
    from ..visual_odometry import default_intrinsics

    imgs = getattr(ctx.frames, "frames", None)
    if imgs is None or len(imgs) != len(ctx.traj.xz):
        print("      -> no frames for road-plane speed; unit steps")
        return None
    from ..frame_extraction import GrayscaleFrames

    gray = GrayscaleFrames(imgs)
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
    filter_only = cfg.answer == "filter"
    independent_map = filter_only or getattr(cfg, "graph_filter_map_path", None) is not None
    if (traj is None or frames is None or len(frames.timestamps) < 3
            or (road is None and not independent_map)):
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
        timestamps = np.asarray(frames.timestamps, float)
        intervals = np.diff(timestamps)
        if (len(timestamps) != len(traj.xz) or not np.isfinite(timestamps).all()
                or np.any(intervals <= 0)):
            raise ValueError("filter requires finite increasing video timestamps matching its VO poses")
        frame_dt = float(np.median(intervals))
        speed = getattr(ctx, "validated_metric_speed", None)
        if speed is None:
            speed = _plane_speed(ctx) if cfg.graph_filter_speed == "plane" else None
        if getattr(ctx, "require_metric_filter", False) and speed is None:
            _record_channel(ctx.channel_status, "graph_filter", "inactive",
                            reason="metric_assisted_profile_refuses_unit_step_fallback")
            return
        odom = odometry_from_trajectory(
            traj, frame_dt, heading_source="rotation",
            step_lengths_m=None if speed is None else speed.lengths_m)
        odom.dt = intervals
        factor = max(1, int(round(_TARGET_STEP_S / max(frame_dt, 1e-3))))
        odom = aggregate_odometry(odom, factor)
        map_input = None
        if independent_map:
            graph, map_input = _whole_city_map(ctx)
            if filter_only:
                from pathlib import Path
                from ..osm_data import _build_polyline_view
                from ..provenance import config_snapshot
                from .report import _write_result_json

                ctx.road = road = _build_polyline_view(graph)
                ctx.cache_path = Path(map_input["cache_path"])
                ctx.result["filter_inputs"]["map"] = map_input
                _write_result_json(cfg.output_dir / "filter_inputs.json",
                                   {**ctx.result["filter_inputs"], "config": config_snapshot(cfg)})
        else:
            graph = road.graph
        net = RoadNetwork.from_graph(graph)

        obs: list = []
        if cfg.graph_filter_vpr and ctx.vpr_track is not None and ctx.anchor_origin == "vpr":
            # Retrieval positions are absolute but noisy and sometimes wrong
            # (a look-alike street across the city): a wide sigma and a high
            # outlier probability keep one bad frame from pulling the cloud.
            idx, latlons, _sims = ctx.vpr_track
            xy = latlon_to_xy(np.asarray(latlons, float), net.crs)
            for fi, (x, y) in zip(np.asarray(idx, int), xy):
                obs.append(PositionObservation(step=int(round(fi / factor)),
                                               x=float(x), y=float(y),
                                               sigma_m=40.0, outlier=0.4))
        fcfg = FilterConfig(n_particles=int(cfg.graph_filter_particles),
                            seed=int(cfg.graph_filter_seed))
        budget_mib = getattr(cfg, "graph_filter_path_budget_mb", None)
        if budget_mib is not None:
            fcfg.path_budget_mb = float(budget_mib) * 1024 ** 2 / 1e6
        if speed is not None:
            # As in the KITTI evaluation: a wide global-scale prior for the
            # assumed camera height, per-step noise from the estimator's own
            # (ground-truth-free) spread.
            fcfg.metric_logk_sigma = 0.25
            dev = speed.rel_spread if np.isfinite(speed.rel_spread) else 0.15
            fcfg.metric_speed_dev_sigma = float(np.clip(dev, 0.05, 0.3))
        res = run_filter(net, odom, fcfg, position_obs=obs)
        if filter_only:
            prediction = res.to_json()
            prediction["video_times_s"] = timestamps[res.frame_index].tolist()
            prediction["input_policy"] = ctx.result["filter_inputs"]["input_policy"]
            _write_result_json(cfg.output_dir / "filter_prediction.json", prediction)
            np.savez_compressed(cfg.output_dir / "filter_prediction.npz", times=res.times,
                                frame_index=res.frame_index, video_times_s=timestamps[res.frame_index],
                                mode_latlon=res.mode_latlon, smoothed_latlon=res.smoothed_latlon,
                                final_xy=res.final_xy, final_w=res.final_w,
                                confidence=res.confidence, n_modes=res.n_modes)
    except Exception as e:  # an experimental channel must never sink the run
        print(f"      -> graph filter failed ({type(e).__name__}: {e})")
        _record_channel(ctx.channel_status, "graph_filter", "failed", error=e)
        return

    route = res.smoothed_latlon if res.smoothed_latlon is not None else res.mode_latlon
    route = np.asarray(route, dtype=float)
    keep = np.unique(np.linspace(0, len(route) - 1,
                                 min(len(route), _MAX_ROUTE_POINTS)).round().astype(int))
    route = route[keep]
    times = timestamps[np.asarray(res.frame_index, int)][keep]
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
        "map_input": map_input,
        "config": res.config,
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
