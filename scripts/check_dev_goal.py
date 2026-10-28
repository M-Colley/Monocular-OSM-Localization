"""Audit a benchmark summary against the full registered development target.

This reads JSON only; it never runs a model or changes benchmark data. Missing
clips, failed repeats and unknown provenance cannot satisfy the goal.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_FLAGS = ("--osm-around", "--ground-truth", "--ground-truth-waypoints",
                    "--estimated-length-m")


def _finite_error(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _known_leak(args) -> bool:
    for arg in args or []:
        text = str(arg)
        flag = text.split("=", 1)[0]
        if (flag.startswith("--") and len(flag) > 2
                and any(f.startswith(flag) for f in _FORBIDDEN_FLAGS)):
            return True
        if ("ground_truth" in text.replace("\\", "/").lower()
                or re.search(r"-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+", text)):
            return True
    return False


def _absolute_path(value) -> bool:
    return (isinstance(value, str) and bool(value)
            and (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()))


def _sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _paths_match(left, right):
    if PureWindowsPath(left).drive or PureWindowsPath(right).drive:
        return PureWindowsPath(left) == PureWindowsPath(right)
    return PurePosixPath(left) == PurePosixPath(right)


def _filter_answer_issues(rep, score, city):
    """Audit video-only filtering; explicit city-map scope is a trusted declaration.

    A file digest binds the actual database, not its acquisition history. The
    caller must supply a whole-city map, rather than a clip/GT-centred crop.
    """
    problems = []
    provenance = rep.get("provenance") or {}
    cfg = provenance.get("config") or {}
    if ((score.get("meta") or {}).get("method") != "graph_filter_only"
            or cfg.get("answer") != "filter" or cfg.get("graph_filter") is not True
            or cfg.get("graph_filter_vpr") is not False):
        problems.append("filter answer lacks its enabled, retrieval-independent method configuration")
    if (cfg.get("city") != city or cfg.get("osm_around") is not None
            or cfg.get("estimated_length_m") is not None):
        problems.append("filter configuration violates the registered city-only inference contract")
    inputs = provenance.get("graph_filter_inputs") or {}
    policy = inputs.get("input_policy") or {}
    forbidden = ("calibration", "raw_images", "sensor_timestamps", "gps_or_oxts",
                 "ground_truth", "retrieval", "initial_position")
    if any(policy.get(name) is not False for name in forbidden):
        problems.append("strict filter input-policy audit is missing or reports external inputs")
    video = inputs.get("video") or {}
    if (not _absolute_path(video.get("path")) or not _sha256(video.get("sha256"))
            or _known_leak([video.get("path", "")])):
        problems.append("filter video provenance lacks a clean absolute path and content digest")
    if (inputs.get("timing") or {}).get("source") != "video_pts_or_encoded_fps":
        problems.append("filter timing does not come from the encoded video")
    graph = inputs.get("map") or {}
    if (graph.get("scope") != "whole_city" or graph.get("city") != city or graph.get("query") != city
            or "around" not in graph or graph["around"] is not None
            or "bbox" not in graph or graph["bbox"] is not None
            or graph.get("source") not in {"explicit_cache", "city_cache", "named_city_fetch"}
            or not _absolute_path(graph.get("cache_path")) or not _sha256(graph.get("sha256"))
            or _known_leak([graph.get("cache_path", "")])):
        problems.append("filter map lacks a fingerprinted whole-city database contract")
    configured = cfg.get("graph_filter_map_path")
    if configured is not None:
        if not isinstance(configured, str) or not configured.strip():
            problems.append("explicit filter map override is not a valid recorded path")
        if graph.get("source") != "explicit_cache":
            problems.append("explicit filter map override disagrees with recorded map provenance")
        if _absolute_path(configured):
            if not _paths_match(configured, str(graph.get("cache_path", ""))):
                problems.append("explicit filter map path differs from the map actually used")
        elif graph.get("configured_path") != configured:
            problems.append("relative filter map override lacks its exact configured-path provenance")
    elif graph.get("source") == "explicit_cache":
        problems.append("explicit filter map use is absent from the effective method configuration")
    return problems


def check_goal(summary: dict, manifest: dict, *, threshold_m: float = 200.0,
               expected_count: int = 15, method: str = "answer",
               min_repeats: int = 1) -> dict:
    """Require every available split=dev clip, with failures preserved.

    A clip's target is its median symmetric error over repeats. Every repeat
    must nevertheless have a real, finite answer; a median cannot hide a
    failed invocation. Recompute medians from the saved per-repeat scores.
    """
    reasons = []
    if summary.get("provisional") or (summary.get("review") or {}).get("status") == "in_progress_snapshot":
        reasons.append("summary is an in-progress snapshot, not a completed result")
    if not math.isfinite(threshold_m) or threshold_m < 0 or expected_count < 1 or min_repeats < 1:
        raise ValueError("threshold must be finite/nonnegative; counts must be positive")
    if any(word in method.lower() for word in ("oracle", "ground_truth")):
        reasons.append("the selected method uses ground truth or an oracle")
    expected = {c["id"]: c for c in manifest.get("clips", [])
                if c.get("split") == "dev" and c.get("available_locally") is True}
    if len(expected) != expected_count:
        reasons.append(f"manifest has {len(expected)} eligible dev clips, expected {expected_count}")
    clips = summary.get("clips") or {}
    missing, unexpected = sorted(set(expected) - set(clips)), sorted(set(clips) - set(expected))
    if missing:
        reasons.append("expected clips are missing: " + ", ".join(missing))
    if unexpected:
        reasons.append("summary contains clips outside the target: " + ", ".join(unexpected))

    prov = summary.get("provenance") or {}
    if prov.get("consistent") is not True or prov.get("problems"):
        reasons.append("benchmark provenance is inconsistent or unknown")
    if "leaks" not in prov or prov["leaks"]:
        reasons.append("ground-truth leak audit is missing or reports leakage")
    if prov.get("dirty") != ["False"]:
        reasons.append("prediction code is dirty or its clean state is unknown")
    commits = prov.get("commits") or []
    if not commits or any(c in (None, "None") for c in commits):
        reasons.append("prediction commits are missing or unknown")
    trees = prov.get("package_trees") or []
    if len(trees) != 1 or str(trees[0]).startswith("unknown@"):
        reasons.append("all target clips must use one known pipeline code tree")
    for field in ("method_config_fingerprints", "environment_fingerprints"):
        if len(prov.get(field) or []) != 1:
            reasons.append(f"one consistent {field} value is required")
    if len(prov.get("vpr_model_fingerprints") or []) > 1:
        reasons.append("target clips used different VPR models")

    execution = summary.get("execution") or {}
    if (execution.get("plan_recorded") is not True or execution.get("complete") is not True
            or execution.get("incomplete_reps")):
        reasons.append("a recorded, complete execution plan is required")
    metadata = summary.get("run_config") or {}
    plan = metadata.get("execution_plan") or {}
    if set(plan) != set(expected):
        reasons.append("execution plan does not cover exactly the expected dev clips")
    if _known_leak((metadata.get("config") or {}).get("pipeline_args")):
        reasons.append("run configuration contains known ground-truth-derived inputs")

    reps = summary.get("reps") or []
    if execution.get("expected_reps") != len(reps) or execution.get("n_completed") != len(reps):
        reasons.append("execution counts disagree with the recorded repeats")
    by_clip = {cid: [] for cid in expected}
    for rep in reps:
        if rep.get("clip") not in by_clip:
            reasons.append("a repeat belongs to a clip outside the development target")
            continue
        by_clip[rep["clip"]].append(rep)
        if rep.get("commit") not in commits:
            reasons.append(f"{rep['clip']}: repeat commit disagrees with provenance")
        rep_prov = rep.get("provenance") or {}
        cfg = rep_prov.get("config") or {}
        if (_known_leak(rep_prov.get("argv")) or cfg.get("ground_truth_waypoints")
                or cfg.get("ground_truth_streets") or rep_prov.get("leaks")
                or (cfg.get("osm_around") and cfg.get("osm_around_source") != "video_title")):
            reasons.append(f"{rep['clip']}: repeat provenance contains known ground-truth leakage")

    rows, answer_modes = [], set()
    for cid, entry in expected.items():
        errors = []
        issues = []
        clip = clips.get(cid) or {}
        records = by_clip[cid]
        if clip.get("split") != "dev":
            issues.append("development split is missing or incorrect")
        if clip.get("city") not in {entry["city"], *entry.get("city_alternatives", [])}:
            issues.append("city differs from the registered inputs")
        if clip.get("vo_segments") != [entry["vo_segment"]]:
            issues.append("analysis window differs from the registered inputs")
        if (not entry.get("gt_waypoints_sha256")
                or clip.get("gt_waypoints_sha256") != entry["gt_waypoints_sha256"]):
            issues.append("ground-truth fingerprint is stale or missing")
        method_summary = (clip.get("methods") or {}).get(method) or {}
        median = _finite_error((method_summary.get("symmetric_mean_m") or {}).get("median"))
        if median is None:
            issues.append("selected method has no finite symmetric error")
        if len(records) < min_repeats:
            issues.append(f"fewer than {min_repeats} recorded repeats")
        for count in (clip.get("n_reps"), clip.get("n_ok"), clip.get("n_completed"),
                      method_summary.get("n_reps"), method_summary.get("n_answered")):
            if count != len(records):
                issues.append("repeat/answer counts are incomplete or inconsistent")
                break
        indices = [r.get("rep") for r in records]
        planned = (plan.get(cid) or {}).get("reps") or []
        if len(set(indices)) != len(indices) or set(indices) != set(planned):
            issues.append("recorded repeats do not match the execution plan")
        for rep in records:
            score = (rep.get("scores") or {}).get(method) or {}
            error = _finite_error(score.get("symmetric_mean_m"))
            if (rep.get("ok") is not True or rep.get("completed") is not True
                    or rep.get("result_found") is not True or rep.get("rc") != 0
                    or score.get("answered") is not True or score.get("failed") is not False
                    or error is None):
                issues.append(f"rep{rep.get('rep')}: failed, unanswered or nonfinite")
            else:
                errors.append(error)
            source = str(score.get("source") or "").lower()
            chosen = str((score.get("meta") or {}).get("method") or "").lower()
            if (score.get("oracle") or "oracle" in source or "ground_truth" in source
                    or "oracle" in chosen or "ground_truth" in chosen):
                issues.append(f"rep{rep.get('rep')}: selected score is an oracle/GT result")
            if method == "answer":
                mode = (score.get("meta") or {}).get("mode")
                if source != "result.answer" or mode not in {"fit", "track", "checked", "filter"}:
                    issues.append(f"rep{rep.get('rep')}: selected answer is stale or missing")
                else:
                    answer_modes.add(mode)
                if mode == "filter":
                    issues.extend(f"rep{rep.get('rep')}: {problem}"
                                  for problem in _filter_answer_issues(rep, score, clip.get("city")))
        filter_inputs = [(rep.get("provenance") or {}).get("graph_filter_inputs") or {}
                         for rep in records if (((rep.get("scores") or {}).get(method) or {}).get(
                             "meta") or {}).get("mode") == "filter"]
        if len({((data.get("video") or {}).get("sha256"),
                 (data.get("map") or {}).get("sha256")) for data in filter_inputs}) > 1:
            issues.append("filter video/map content changes between repeats")
        measured = statistics.median(errors) if errors else None
        if measured is not None and median is not None and abs(round(measured, 1) - median) > 0.051:
            issues.append("reported median does not match the recorded repeat scores")
        if median is not None and median > threshold_m:
            issues.append(f"symmetric error exceeds {threshold_m:g} m")
        rows.append({"clip": cid, "n_repeats": len(records), "symmetric_mean_m": median,
                     "passes": not issues, "reasons": list(dict.fromkeys(issues))})
    if len(answer_modes) > 1:
        reasons.append("the selected answer mode differs between clips/repeats")
    passed = sum(row["passes"] for row in rows)
    return {
        "goal_met": not reasons and passed == expected_count,
        "method": method, "threshold_m": threshold_m, "expected_count": expected_count,
        "min_repeats": min_repeats, "n_passing": passed, "n_expected": len(expected),
        "missing_ids": missing, "unexpected_ids": unexpected,
        "reasons": list(dict.fromkeys(reasons)), "clips": rows,
        "input_contract_notes": (["Explicit filter maps are trusted whole-city database inputs; "
                                  "content hashes bind the file, not its acquisition history."]
                                 if "filter" in answer_modes else []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="benchmark aggregate summary.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/manifest.json")
    parser.add_argument("--threshold-m", type=float, default=200.0)
    parser.add_argument("--expected-count", type=int, default=15)
    parser.add_argument("--method", default="answer")
    parser.add_argument("--min-repeats", "--min-reps", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = check_goal(summary, manifest, threshold_m=args.threshold_m,
                            expected_count=args.expected_count, method=args.method,
                            min_repeats=args.min_repeats)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result = {"goal_met": False, "reasons": [f"cannot audit inputs: {error}"]}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["goal_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
