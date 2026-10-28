"""Experimental fixed-policy controller over isolated, frozen child runs."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import datetime
import hashlib
import json
from pathlib import Path
import re
import uuid
import numpy as np
from ..assisted_policy import CONFIG_SHA256, POLICY_VERSION, decide_assisted_policy, policy_config
from ..assisted_profiles import resolve_policy
from ..provenance import config_snapshot, file_sha256, snapshot_sources
from .context import _record_channel
from .poi_only import _safe
from .report import _write_result_json

REQUESTS = {"request_ocr_vo": "cues", "request_poi_joint": "poi_joint",
            "request_poi_image_track": "poi_image_track", "request_metric_pf": "metric_pf",
            "request_poi_image_union": "poi_image_union",
            "request_expanded_retrieval": "expanded"}
HASH = re.compile(r"[a-f0-9]{64}")
STATISTIC = "raw_cosine_at_decoded_reference"


def digest(value):
    return hashlib.sha256(json.dumps(_safe(value), sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def valid_route(candidate):
    if candidate.get("status") != "complete":
        return None
    try:
        route = np.asarray(candidate["route_latlon"], float)
        if (route.ndim != 2 or route.shape[1] != 2 or len(route) < 2 or not np.isfinite(route).all()
                or np.any(np.abs(route[:, 0]) > 90) or np.any(np.abs(route[:, 1]) > 180)):
            return None
        times = candidate.get("route_timestamps")
        if times is not None:
            times = np.asarray(times, float)
            if times.shape != (len(route),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                return None
        return route
    except (ValueError, TypeError, KeyError):
        return None


def score_identity(comparison):
    """Outer pool selection differs by design and remains an optimistic dev heuristic."""
    try:
        model, query, decoder = (comparison[name] for name in ("model", "query", "inner_decoder"))
        from ..kartaview_vpr import HUB_MEGALOC
        if (comparison.get("statistic") != STATISTIC or model["name"] != "megaloc"
                or model["hub"] != HUB_MEGALOC or query["mode"] != "warp"
                or not HASH.fullmatch(model["weights_sha256"])
                or decoder["actual"] != "viterbi" or decoder.get("fallback") is not False
                or not isinstance(decoder["settings"], dict) or not decoder["settings"]):
            return None
        indices, times = np.asarray(query["indices"]), np.asarray(query["times"], float)
        if (indices.ndim != 1 or len(indices) < 2 or times.shape != indices.shape
                or not np.issubdtype(indices.dtype, np.integer) or np.any(np.diff(indices) <= 0)
                or indices[0] < 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                or not isinstance(query["preprocessing"], dict) or not query["preprocessing"]
                or any(not HASH.fullmatch(query[name]) for name in ("pixels_sha256", "descriptors_sha256"))):
            return None
        return digest({"statistic": STATISTIC, "model": model, "query": query, "inner_decoder": decoder})
    except (KeyError, TypeError, ValueError):
        return None


def summary(kind, child):
    """Derive scalar inputs from actual decoded evidence, never rounded confidence."""
    status = child.get("status", "failed")
    if status not in {"complete", "partial", "failed", "unavailable"}:
        status = "failed"
    if kind == "cues":
        return {**deepcopy(child.get("cues") or {}), "status": status}
    route = valid_route({**child, "status": status})
    result = {"status": status, "route_valid": route is not None,
              "point_count": len(route) if route is not None else 0}
    if kind in {"baseline", "expanded"}:
        comparison = child.get("comparison") or {}
        identity = score_identity(comparison)
        try:
            scores = np.asarray(comparison["decoded_reference_cosines"], float)
            coordinates = np.asarray(comparison["decoded_reference_latlon"], float)
            valid = (identity is not None and scores.shape == (len(comparison["query"]["indices"]),)
                     and np.isfinite(scores).all() and np.all(np.abs(scores) <= 1)
                     and coordinates.shape == (len(scores), 2) and np.isfinite(coordinates).all())
        except (KeyError, TypeError, ValueError):
            valid = False
        result.update(score_kind="raw_cosine" if valid else None,
                      score_identity=identity if valid else None,
                      similarity_median=float(np.median(scores)) if valid else None)
        if valid and kind == "expanded":
            result["dominant_coordinate_fraction"] = max(Counter(map(tuple, coordinates)).values()) / len(coordinates)
    else:
        for name in ("retained_branch_count", "successful_branch_count", "uses_rigid_vo", "radius_m",
                     "cap", "query_mode", "score_normalization", "metric_only",
                     "uses_user_spatial_prior", "position_observation_count", "profile_sha256", "candidate_pruning"):
            if name in child:
                result[name] = child[name]
    return result


def child_record(kind, child, output, expected=None):
    artifacts = []
    for name in ("result_artifact", "frozen_artifact", "candidate_artifact"):
        record = child.get(name)
        if record is not None:
            path = Path(record["path"]).resolve(strict=True)
            if not path.is_relative_to(output.resolve()) or file_sha256(path) != record["sha256"]:
                raise ValueError(f"{kind}: child artifact escaped output or changed")
            artifacts.append({"role": name, "path": str(path), "sha256": record["sha256"]})
    roles = {item["role"]: item for item in artifacts}
    if child.get("status") in {"complete", "partial"} and len(roles) != 3:
        raise ValueError(f"{kind}: complete/partial child lacks frozen candidate/result artifacts")
    if len(roles) == 3:
        frozen_path = Path(roles["frozen_artifact"]["path"])
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen.get("scoring_has_not_run") is not True:
            raise ValueError(f"{kind}: child is not an inference-only freeze")
        for relative, expected_hash in frozen["artifacts"].items():
            path = (frozen_path.parent / relative).resolve(strict=True)
            if not path.is_relative_to(frozen_path.parent) or file_sha256(path) != expected_hash:
                raise ValueError(f"{kind}: frozen child artifact changed or escaped its run")
        for role in ("candidate_artifact", "result_artifact"):
            record = roles[role]
            relative = str(Path(record["path"]).relative_to(frozen_path.parent))
            if frozen["artifacts"].get(relative) != record["sha256"]:
                raise ValueError(f"{kind}: child freeze does not bind its {role}")
        payload = {key: value for key, value in child.items() if key not in
                   {*roles, "worker_returncode", "reported_status", "transport_error"}}
        if "reported_status" in child:
            if child.get("worker_returncode", 0) == 0:
                raise ValueError("child status promotion lacks a nonzero worker exit")
            payload["status"] = child["reported_status"]
        if digest(payload) != digest(json.loads(Path(roles["candidate_artifact"]["path"]).read_text(encoding="utf-8"))):
            raise ValueError(f"{kind}: response differs from the frozen candidate payload")
        if expected is not None and (payload.get("video_sha256") != expected["video"]["sha256"]
                or payload.get("window") != expected["registered_window"]
                or payload.get("source_sha256") != expected["source_sha256"]):
            raise ValueError(f"{kind}: child source/video/window differs from its parent")
    if child.get("status") == "complete" and child.get("worker_returncode") != 0:
        raise ValueError(f"{kind}: complete child did not exit successfully")
    route = valid_route(child)
    return {"kind": kind, "status": child.get("status", "failed"), "artifacts": artifacts,
            "source_sha256": deepcopy(child.get("source_sha256") or {}),
            "route_sha256": digest(child["route_latlon"]) if route is not None else None,
            "summary": summary(kind, child), "error": child.get("error") or child.get("transport_error"),
            "worker_returncode": child.get("worker_returncode"), "reported_status": child.get("reported_status"),
            "comparison": deepcopy(child.get("comparison")) if kind in {"baseline", "expanded"} else None}


def begin(ctx):
    root = Path(__file__).resolve().parents[2]
    sources = [root / "main.py", *sorted((root / "monocular_osm").rglob("*.py"))]
    execution_id = uuid.uuid4().hex
    policy = resolve_policy(getattr(ctx.cfg, "assisted_profile", "v1"))
    ctx.assisted_child_root = ctx.cfg.output_dir / "children" / execution_id
    inputs = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "execution_id": execution_id,
        "experimental": True, "validation_status": "posthoc_dev_unvalidated",
        "video": {"path": str(Path(ctx.video_path).resolve()), "sha256": file_sha256(ctx.video_path)},
        "registered_window": [ctx.cfg.vo_start_sec, ctx.cfg.vo_end_sec], "city": ctx.cfg.city,
        "input_policy": {name: False for name in ("ground_truth", "calibration", "sensor_timestamps",
            "raw_images", "gps_or_oxts", "manual_region_coordinates", "external_trajectory", "external_ocr_video")},
        "policy": {"version": policy.POLICY_VERSION, "config": policy.policy_config(), "config_sha256": policy.CONFIG_SHA256,
                   "source_sha256": file_sha256(Path(policy.__file__))},
        "source_sha256": snapshot_sources(sources, ctx.cfg.output_dir), "config": config_snapshot(ctx.cfg)}
    ctx.result["assisted_inputs"] = inputs
    _write_result_json(ctx.cfg.output_dir / "assisted_inputs.json", _safe(inputs))


def freeze(ctx):
    output = ctx.cfg.output_dir
    files = [output / name for name in ("result.json", "answer.png", "assisted_inputs.json", "assisted_audit.json")]
    files += [Path(item["path"]) for record in ctx.result["assisted_audit"]["reached_stages"] for item in record["artifacts"]]
    _write_result_json(output / "frozen_prediction.json", {
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "experimental": True,
        "scoring_has_not_run": True, "prediction_available": "answer" in ctx.result,
        "inputs": _safe(ctx.result["assisted_inputs"]),
        "artifacts": {str(path.resolve().relative_to(output.resolve())): file_sha256(path)
                      for path in files if path.is_file()}})


def run_assisted(ctx, *, finalize, run_child=None):
    """Execute only reached requests; incomplete assistance never replaces baseline."""
    if run_child is None:
        from ..assisted_worker import run_child
    begin(ctx)
    policy = resolve_policy(getattr(ctx.cfg, "assisted_profile", "v1"))
    ctx.result["execution"] = {"stage": "assisted", "experimental": True}
    for channel in ("vpr", "plate", "vlm", "sun", "ocr_anchor", "loop_closure", "coarse_from_frames",
                    "orienternet", "geometry", "graph_filter", "poi", "poi_retrieval"):
        _record_channel(ctx.channel_status, channel, "disabled", reason="isolated_assisted_children")
    evidence, children, trace, trace_inputs, reached = {}, {}, [], [], []
    def execute(kind):
        try:
            ctx.assisted_evidence = deepcopy(evidence)
            child = _safe(run_child(kind, ctx))
            record = child_record(kind, child, ctx.cfg.output_dir, ctx.result["assisted_inputs"])
        except Exception as error:
            child = {"status": "failed", "error": f"{type(error).__name__}: {_safe(str(error))}"}
            record = child_record(kind, child, ctx.cfg.output_dir)
        children[kind], evidence[kind] = child, record["summary"]
        reached.append(record)
    execute("baseline")
    terminal = None
    for _ in range(len(REQUESTS) + 1):
        trace_inputs.append(deepcopy(evidence))
        decision = policy.decide_assisted_policy(evidence)
        trace.append(decision)
        action = decision["next_action"]
        if action in {"keep_baseline", "select_candidate", "no_selection"}:
            terminal = decision
            break
        kind = REQUESTS.get(action)
        if kind is None or kind in children:
            raise ValueError("assisted policy requested an unknown or repeated stage")
        execute(kind)
    if terminal is None:
        raise ValueError("assisted policy did not terminate")
    selected = terminal["selected_candidate"]
    if selected is not None:
        child = children[selected]
        if valid_route(child) is None:
            raise ValueError("assisted policy selected an incomplete or invalid child route")
        position = {"route_latlon": deepcopy(child["route_latlon"]), "source": f"assisted_child:{selected}",
                    "experimental": True, "confidence_calibrated": False}
        if child.get("route_timestamps") is not None:
            position["route_t_sec"] = deepcopy(child["route_timestamps"])
        ctx.result["assisted_position"] = position
    audit = {"decision_trace": trace, "decision_inputs": trace_inputs, "evidence": evidence,
             "reached_stages": reached, "selected_candidate": selected,
             "selected_route_sha256": digest(children[selected]["route_latlon"]) if selected else None,
             "policy_version": policy.POLICY_VERSION, "config_sha256": policy.CONFIG_SHA256,
             "comparison_limit": "fixed expanded pool selection can inflate raw cosine; unvalidated dev heuristic"}
    audit["selected_child_artifact_sha256"] = next((item["sha256"] for record in reached
        if record["kind"] == selected for item in record["artifacts"]
        if item["role"] == "candidate_artifact"), None)
    ctx.result["assisted_audit"] = _safe(audit)
    _write_result_json(ctx.cfg.output_dir / "assisted_audit.json", ctx.result["assisted_audit"])
    _record_channel(ctx.channel_status, "assisted", "ok" if selected else "inactive",
                    selected_candidate=selected, validation_status="posthoc_dev_unvalidated")
    ctx.result["execution"].update(stage="complete", answer_available=selected is not None)
    finalize(ctx)
    freeze(ctx)
    return ctx.result
