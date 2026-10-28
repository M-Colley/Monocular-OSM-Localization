"""Stage 11: diagnostics, channel summary, provenance and result.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from ..visual_odometry import VO_ALGO_VERSION
from .context import _record_channel, RunContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..pipeline import PipelineConfig


def _write_result_json(path: Path, result: dict) -> None:
    """Never truncate a usable retrieval checkpoint on a later write failure."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
            json.dump(result, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _summarize_channels(status: dict, cfg: "PipelineConfig", *, produced: dict,
                        fusion: dict | None) -> dict:
    """Fill in a status for every channel that did not record a failure.

    ``produced`` maps channel -> (enabled, produced_output). Fusion rank
    channels are "ok" when their effective weight was non-zero.
    """
    # A retrieval checkpoint may be finalized before geometry executes. Keep
    # derived statuses out of RunContext so a later report recomputes them;
    # explicitly recorded failures in the context still take precedence.
    status = dict(status)
    for name, (enabled, made) in produced.items():
        if name in status:
            continue
        if not enabled:
            _record_channel(status, name, "disabled")
        else:
            _record_channel(status, name, "ok" if made else "inactive")
    eff = (fusion or {}).get("weights_effective") or {}
    for name, w in eff.items():
        key = f"fusion_{name}"
        if key not in status:
            _record_channel(status, key, "ok" if w else "inactive", weight=w)
    return status


def finalize_result(ctx: RunContext) -> None:
    """Attach diagnostics, the channel summary and provenance; write result.json.

    Diagnostics are attached UNCONDITIONALLY: failed runs (no candidates, unusable
    CRS) are exactly the ones they are needed to debug.
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    result = ctx.result
    result.setdefault("city", cfg.city)
    frame_sampling = getattr(ctx.frames, "sampling", None)
    if frame_sampling is not None:
        result["frame_sampling"] = frame_sampling
    result_ipm_scale = ctx.result_ipm_scale
    vo_cache = ctx.vo_cache
    frame_seed_source = ctx.frame_seed_source
    vpr_center = ctx.vpr_center
    vpr_track = ctx.vpr_track
    c2f_info = ctx.c2f_info
    anchor_origin = ctx.anchor_origin
    vpr_source_used = ctx.vpr_source_used
    result_plate_anchor = ctx.result_plate_anchor
    plate_center = ctx.plate_center
    result_vlm_anchor = ctx.result_vlm_anchor
    road = ctx.road
    cache_path = ctx.cache_path
    result_sun_heading = ctx.result_sun_heading
    result_scale_recovery = ctx.result_scale_recovery
    result_loop_closure = ctx.result_loop_closure
    ocr_anchors = ctx.ocr_anchors
    street_anchors = ctx.street_anchors
    result_vggt = ctx.result_vggt

    if result_sun_heading is not None:
        # store a JSON-safe view (drop the per-frame numpy array)
        result["sun_heading"] = {k: v for k, v in result_sun_heading.items()
                                 if k != "headings"}

    # Diagnostics are attached UNCONDITIONALLY: failed runs (no candidates,
    # unusable CRS) are exactly the ones these are needed to debug, and
    # they used to vanish whenever the position report didn't build.
    if c2f_info is not None:
        result["vpr_coarse_to_fine"] = c2f_info
    if ctx.vpr_listing is not None:
        # A reference listing with lost tiles is blind there; record it.
        result["vpr_listing"] = ctx.vpr_listing
    if result_scale_recovery is not None:
        result["scale_recovery"] = result_scale_recovery
    if result_loop_closure is not None:
        result["loop_closure"] = result_loop_closure
    if result_vggt is not None:
        result["vggt_gating"] = result_vggt
    if result_ipm_scale is not None:
        result["ipm_scale"] = result_ipm_scale

    # Anchor-channel summaries: a run's only record of these expensive
    # computations must not be stdout.
    if result_plate_anchor is not None:
        result_plate_anchor.setdefault("applied", False)
        result["plate_anchor"] = result_plate_anchor
    elif cfg.use_plate_anchor:
        result["plate_anchor"] = {"applied": False,
                                  "status": "no_confident_district"}
    if result_vlm_anchor is not None:
        result["vlm_anchor"] = result_vlm_anchor
    elif cfg.use_vlm_anchor and vpr_center is None:
        result["vlm_anchor"] = {"applied_as_prior": False,
                                "status": "no_geocodable_prior"}
    if vpr_center is not None:
        from ..kartaview_vpr import backbone_info as _vpr_backbone
        result["vpr_prior"] = {
            "lat": float(vpr_center[0]), "lon": float(vpr_center[1]),
            "origin": anchor_origin,
            # The source actually queried: "mapillary" falls back to
            # "kartaview" without a token, and results differ by kilometres.
            "source": vpr_source_used,
            "backbone": _vpr_backbone() if anchor_origin == "vpr" else None,
            "track_frames": (int(len(vpr_track[0]))
                             if vpr_track is not None else None),
            # Retrieval confidence: median top-1 cosine similarity over the
            # final track (= pass2_sim when coarse-to-fine kept pass 2). On the
            # core fleet it separated right areas (0.21-0.26) from wrong ones
            # (0.11-0.13); recorded on every run so that can be tested.
            "track_sim_median": (round(float(np.median(vpr_track[2])), 4)
                                 if vpr_track is not None and len(vpr_track[2]) else None),
        }
        if vpr_track is not None and len(vpr_track[0]):
            # The final per-frame track itself, [t_sec, lat, lon, sim]: lets a
            # run's error be split into retrieval vs the trajectory fit.
            _ts = (ctx.frames.timestamps if ctx.frames is not None else None)
            result["vpr_prior"]["track"] = [
                [round(float(_ts[int(i)]), 2) if _ts is not None and int(i) < len(_ts) else None,
                 round(float(ll[0]), 6), round(float(ll[1]), 6), round(float(s), 4)]
                for i, ll, s in zip(vpr_track[0], vpr_track[1], vpr_track[2])]

    result["channels"] = _summarize_channels(
        channel_status, cfg,
        produced={
            "vpr": (cfg.use_vpr_prior, vpr_center is not None),
            "plate": (cfg.use_plate_anchor, plate_center is not None),
            "vlm": (cfg.use_vlm_anchor, result_vlm_anchor is not None),
            "sun": (cfg.use_sun_heading, bool((result_sun_heading or {}).get("available"))),
            "ocr_anchor": (cfg.enable_ocr_anchor, bool(ocr_anchors or street_anchors)),
            "loop_closure": (cfg.enable_loop_closure, result_loop_closure is not None),
            "coarse_from_frames": (cfg.coarse_from_frames and cfg.osm_around is None,
                                   frame_seed_source is not None),
            "graph_filter": (cfg.graph_filter,
                             "filter_position" in result),
            "poi": (cfg.answer == "poi", "poi_position" in result),
            "orienternet": (cfg.use_orienternet,
                            bool((result.get("orienternet") or {}).get("applied"))),
        },
        fusion=result.get("fusion"),
    )

    # The route the method answers with (cfg.answer, chosen by the
    # pre-registered answer_selection rule); result["position"] stays the
    # trajectory fit so runs remain comparable across versions.
    from .. import baselines as _bl
    _pick = {"fit": _bl.pipeline_headline, "track": _bl.vpr_track_only,
             "filter": _bl.graph_filter_only,
             "poi": _bl.poi_only,
             "checked": _bl.track_checked}[cfg.answer](result)
    if _pick is None and cfg.answer == "track":
        _pick = _bl.pipeline_headline(result)          # no track: the fit
    if _pick is not None:
        from ..position import build_answer_report, format_answer_summary, plot_answer

        result["answer"] = build_answer_report(
            _pick.route_latlon, mode=cfg.answer,
            method=_pick.meta.get("chose", _pick.name), source=_pick.source,
            disagreement_m=_pick.meta.get("disagreement_m"))
        if cfg.answer == "filter" and "route_t_sec" in result.get("filter_position", {}):
            result["answer"]["route_t_sec"] = result["filter_position"]["route_t_sec"]
        if cfg.answer == "poi":
            result["answer"].update(experimental=True, confidence_calibrated=False)
            result["answer"]["route_t_sec"] = result["poi_position"]["route_t_sec"]
        _answer_map = cfg.output_dir / "answer.png"
        try:
            plot_answer(result["answer"], _answer_map, road=road)
            result["answer"]["map_artifact"] = _answer_map.name
        except Exception as e:
            # An image-writing failure must not discard a successfully selected
            # route, or leave an older run's map looking like the current answer.
            _answer_map.unlink(missing_ok=True)
            result["answer"]["map_error"] = f"{type(e).__name__}: {e}"
            print(f"      -> selected-answer map unavailable ({e})")
        if (result.get("execution") or {}).get("stage") == "retrieval":
            print("      -> retrieval answer checkpointed; geometry diagnostics pending")
        else:
            print()
            if cfg.answer == "poi":
                print("EXPERIMENTAL POI component; confidence is not calibrated.")
            print(format_answer_summary(result["answer"]))
    else:
        result.pop("answer", None)
        (cfg.output_dir / "answer.png").unlink(missing_ok=True)
        print(f"      -> selected answer ({cfg.answer}) unavailable")

    # Provenance: commit, full config, library versions, whether credentials
    # were set (never their values). Without it a result cannot be traced to
    # the code and settings that produced it.
    from ..provenance import collect_provenance
    # The OSM snapshot matters too: OSM is edited continuously and the graph
    # is cached, so record when (and with what) this run's graph was fetched.
    _g = road.graph.graph if road is not None else {}
    result["provenance"] = collect_provenance(
        cfg, extra={"vo_algo_version": VO_ALGO_VERSION,
                    "graph_filter_inputs": result.get("filter_inputs"),
                    "poi_inputs": result.get("poi_inputs"),
                    "poi_localization_audit": result.get("poi_localization_audit"),
                    "vo_cache": str(vo_cache) if vo_cache is not None else None,
                    "osm_graph": {
                        "cache": str(cache_path) if cache_path is not None else None,
                        "created_date": _g.get("created_date"),
                        "created_with": _g.get("created_with"),
                        "nodes": road.graph.number_of_nodes(),
                        "edges": road.graph.number_of_edges(),
                    } if road is not None else None},
    )

    out_json = cfg.output_dir / "result.json"
    _write_result_json(out_json, result)
    print(f"      -> wrote {out_json}")
