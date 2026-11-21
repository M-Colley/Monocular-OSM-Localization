"""Native-grid worker adapter for the fixed all-literal-region image union."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from .poi_image_union import PROFILE_SHA256, plan_image_union, localize_image_union
from .provenance import file_sha256
from .stages.poi_only import _safe
from .stages.report import _write_result_json


def run_image_union_child(cfg, video, request):
    from .assisted_worker import _array_sha, _load_cues
    from .stages.context import RunContext
    from .stages.acquire import extract_video_frames
    from .stages.poi_only import _retrieval

    if cfg.vpr_query_mode != "crops" or cfg.poi_radius_m != 3000. or cfg.poi_ref_cap != 1500:
        raise ValueError("all-region image child requires its fixed native crop acquisition profile")
    cue = _load_cues(request)
    plan = plan_image_union(cue["clues"], cue["ocr_provenance"])
    _write_result_json(cfg.output_dir / "image_union_plan.json", _safe(plan))
    if plan["status"] != "ready":
        return {"status": "unavailable", "reason": plan["reason"], "image_union_plan": plan,
            "profile_sha256": PROFILE_SHA256, "retained_branch_count": plan["branch_count"],
            "successful_branch_count": 0, "uses_rigid_vo": False}
    ctx = RunContext.create(cfg)
    ctx.video_path = video
    query_cache = {}
    try:
        extract_video_frames(ctx)
        if list(ctx.frames.frames[0].shape[:2]) != cue["ocr_provenance"]["frame_shape_hw"]:
            raise ValueError("image query pixels and native OCR boxes use different grids")
        auto = cue["auto_quality"]
        times = np.asarray(ctx.frames.timestamps, np.float64)
        window = [cfg.vo_start_sec, cfg.vo_end_sec]
        if (cue.get("window") != window or auto.get("window") != window
                or auto.get("stride") != cfg.frame_stride or auto.get("fps") != float(ctx.frames.fps)
                or auto.get("n_frames") != len(ctx.frames.frames)
                or auto.get("timestamps_sha256") != _array_sha(times)):
            raise ValueError("image acquisition differs from its frozen native cue grid/window")
        inputs = {"inventory": cue["inventory"], "evidence_artifacts": [],
            "video": {"sha256": request["video_sha256"]}}
        retrieve = _retrieval(ctx, inputs, audit_query=True)

        def evidence(region, *, n_query, query_mode, query_cache):
            if n_query != 80 or query_mode != "crops":
                raise ValueError("image union requested a different query profile")
            return retrieve(ctx.frames.frames, times, tuple(region["center"]),
                radius_m=region["radius_m"], cap=region["cap"], n_query=n_query, query_cache=query_cache)

        candidate = localize_image_union(plan, timestamps=times,
            query_video_sha256=request["video_sha256"], retrieve_evidence=evidence, query_cache=query_cache)
        union = candidate.get("reference_union")
        if union is not None:
            array_names = ("query_indices", "query_times", "ref_latlon", "similarities")
            path = cfg.output_dir / "image_reference_union.npz"
            np.savez_compressed(path, ref_ids=np.asarray(union["ref_ids"]),
                **{name: union[name] for name in array_names})
            record = {"path": path.name, "sha256": file_sha256(path)}
            inputs["evidence_artifacts"].append(record)
            candidate["reference_union"] = {key: deepcopy(value) for key, value in union.items()
                if key not in array_names}
            candidate["reference_union"]["matrix_artifact"] = record
        candidate.update(image_union_plan=plan, evidence_artifacts=inputs["evidence_artifacts"],
            acquisition={"stride": cfg.frame_stride, "n_frames": len(ctx.frames.frames),
                "timestamps_sha256": _array_sha(times), "native_frame_shape_hw": list(ctx.frames.frames[0].shape[:2])})
        _write_result_json(cfg.output_dir / "image_union_prediction.json", _safe(candidate))
        return candidate
    finally:
        query_cache.clear()
        if ctx.frames is not None:
            ctx.frames.close()
