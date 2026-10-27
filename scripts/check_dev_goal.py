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
from pathlib import Path

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
                if source != "result.answer" or mode not in {"fit", "track", "checked"}:
                    issues.append(f"rep{rep.get('rep')}: selected answer is stale or missing")
                else:
                    answer_modes.add(mode)
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
