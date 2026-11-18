"""Score a held-out assisted-policy run against its pre-registered rule.

Written for ``test4_2026_11`` (benchmarks/preregistered_rules.json): for every
clip it scores, with the unchanged ``monocular_osm.metrics.score_route``,

* the run's assisted answer (``result.json`` ``assisted_position``, checked
  against ``answer``),
* the same run's baseline child answer (``children/*/baseline/result.json``),
* the city-centre guess from ``benchmarks/geocode_snapshot.json`` for the
  frozen manifest city (offline; no network),

against the frozen ground truth named by the frozen manifest, after checking
the manifest and every ground-truth file against their registered digests.
It then evaluates the rule's hypotheses V1-V3 and the k/n goal check.

    python scripts/score_heldout_assisted.py \
        --run-dir E:/claude-scoring/wt-test4-v4/output/benchmark_runs/assisted_v4_test4 \
        --out docs/heldout_test4_2026-11.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

import importlib.util  # noqa: E402

# Loaded by path: the dash 4.5.0 wheel installs a top-level ``benchmarks``
# package, which shadows this repository's namespace directory (as in CI; see
# tests/test_benchmark_manifest.py). Fix proposed upstream in plotly/dash#4148.
_spec = importlib.util.spec_from_file_location("_build_manifest", ROOT / "benchmarks" / "build_manifest.py")
_build_manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build_manifest)
gt_waypoints_digest = _build_manifest.gt_waypoints_digest
MANIFEST_PATH = _build_manifest.MANIFEST_PATH
from monocular_osm import baselines as B  # noqa: E402
from monocular_osm.metrics import score_route  # noqa: E402
from scripts import failure_taxonomy as FT  # noqa: E402

THRESHOLD_M = 200.0


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frozen_manifest(freeze: dict, path: Path | None) -> dict:
    revision = freeze.get("manifest_commit")
    if path is None and revision:
        source = f"{revision}:{MANIFEST_PATH.as_posix()}"
        try:
            content = subprocess.check_output(["git", "show", source], cwd=ROOT,
                                              stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"cannot read frozen manifest {source}; provide --manifest") from exc
    else:
        path = path if path is not None else ROOT / MANIFEST_PATH
        source = str(path)
        content = path.read_bytes()
    expected = freeze["manifest_sha256"]
    if hashlib.sha256(content).hexdigest() != expected:
        raise SystemExit(f"manifest {source} is not the frozen one ({expected})")
    return json.loads(content)


def _require_clean_commit(result: dict, commit: str, path: Path) -> None:
    git = ((result.get("provenance") or {}).get("git") or {})
    if git.get("commit") != commit or git.get("dirty") is not False:
        raise SystemExit(f"{path}: expected clean result from {commit}; "
                         f"found commit {git.get('commit')} (dirty={git.get('dirty')})")


def wilson(k: int, n: int) -> list[float]:
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return [float(lo), float(hi)]


def _one(paths: list[Path], what: str) -> Path:
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one {what}, found {len(paths)}: {paths}")
    return paths[0]


def _score(pred, gt) -> float | None:
    if pred is None:
        return None
    return float(score_route(pred.route_latlon, gt).symmetric_mean_m)


def score_clip(entry: dict, run_dir: Path, *, commit: str) -> dict:
    vid = entry["id"].removeprefix("overlay_")
    gt_doc = _json(ROOT / entry["gt"])
    if gt_waypoints_digest(gt_doc) != entry["gt_waypoints_sha256"]:
        raise SystemExit(f"{entry['id']}: ground truth differs from the frozen manifest digest")
    gt = [[float(w["lat"]), float(w["lon"])] for w in gt_doc["waypoints"]]

    rep = run_dir / entry["id"] / "rep0"
    status = _json(rep / "status.json")
    if status.get("rc") != 0 or status.get("timed_out") or not status.get("result_found"):
        raise SystemExit(f"{entry['id']}: run did not complete ({status})")
    out = _one(sorted((rep / "pipeline_out").glob("*/result.json")), f"{vid} result.json")
    result = _json(out)
    _require_clean_commit(result, commit, out)

    assisted = B.assisted_policy(result)
    answer = B.answer(result)
    if assisted is None or answer is None or assisted.route_latlon != answer.route_latlon:
        raise SystemExit(f"{entry['id']}: assisted_position and answer disagree")
    trace = result["assisted_audit"]["decision_trace"][-1]
    selected = trace["selected_candidate"]

    child = _one(sorted(out.parent.glob("children/*/baseline/result.json")), f"{vid} baseline child")
    child_result = _json(child)
    _require_clean_commit(child_result, commit, child)
    baseline = B.answer(child_result)
    centre = B.city_centroid(entry["city"], allow_network=False)
    if centre is None:
        raise SystemExit(f"{entry['id']}: no geocode snapshot for {entry['city']!r}")

    # Search-coverage taxonomy (scripts/failure_taxonomy.py's discs and modes),
    # read from the baseline child's own log.
    points = np.array(gt)
    log = (child.parent / "worker.log").read_text(encoding="utf-8", errors="replace")
    seed, fine = FT._RE_SEED.search(log), FT._RE_FINE.search(log)
    if seed:
        lat, lon, radius = float(seed.group(1)), float(seed.group(2)), float(seed.group(3))
    else:
        (lat, lon), radius = centre.route_latlon[0], FT.PASS1_CITY_DISC_M
    pass1_share = FT.share_inside(points, lat, lon, radius)
    fine_share = (FT.share_inside(points, float(fine.group(2)), float(fine.group(3)),
                                  float(fine.group(1))) if fine else None)

    # Post-hoc diagnostic only: the expanded candidate the policy declined.
    expanded = None
    for path in sorted(out.parent.glob("children/*/expanded/candidate.json")):
        route = _json(path).get("route_latlon") or []
        if len(route) >= 2:
            expanded = _score(B.Prediction("expanded_candidate", route, "route", source=str(path)), gt)

    runtime = _json(out.parent / "assisted_runtime.json")
    t0 = runtime["started_monotonic_s"]
    stages = {e["kind"]: {"status": e["status"], "end_s": round(e["monotonic_s"] - t0, 1),
                          **({"reason": e["reason"]} if e.get("reason") else {})}
              for e in runtime["events"]}
    rec = {
        "clip": entry["id"], "name": entry["name"], "frozen_city": entry["city"],
        "runner_city": status["city"], "vo_segment": status["vo_segment"],
        "assisted_m": _score(assisted, gt), "baseline_track_m": _score(baseline, gt),
        "city_centre_m": _score(centre, gt), "selected_stage": selected,
        "selection_reason": trace["selection_reason"], "stages": stages,
        "wall_s": status["wall_s"], "result_sha256": _sha256(out),
        "baseline_child_sha256": _sha256(child), "centre_source": centre.source,
        "pass1_radius_m": radius, "title_seeded": bool(seed),
        "pass1_share": pass1_share, "fine_share": fine_share,
        "declined_expanded_candidate_m_posthoc": expanded,
    }
    rec["mode"] = FT.failure_mode(rec["assisted_m"], pass1_share, fine_share,
                                  threshold_m=THRESHOLD_M)
    rec["assisted_within"] = rec["assisted_m"] is not None and rec["assisted_m"] <= THRESHOLD_M
    rec["baseline_within"] = rec["baseline_track_m"] is not None and rec["baseline_track_m"] <= THRESHOLD_M
    rec["beats_centre"] = rec["assisted_m"] is not None and rec["assisted_m"] < rec["city_centre_m"]
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--rule", default="test4_2026_11")
    ap.add_argument("--manifest", type=Path, help="frozen manifest file (defaults to the rule's Git snapshot)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    rule = _json(ROOT / "benchmarks" / "preregistered_rules.json")["rules"][args.rule]
    manifest = _frozen_manifest(rule["dataset_freeze"], args.manifest)
    frozen = rule["dataset_freeze"]["manifest_sha256"]
    run_config = _json(args.run_dir / "run_config.json")
    if run_config["manifest_sha256"] != rule["runner_manifest"]["sha256"]:
        raise SystemExit("run used a different runner manifest than the rule registers")
    runner_git = run_config["runner_git"]
    commit = runner_git.get("commit")
    dirty = runner_git.get("dirty")
    if not commit or not rule["method"].startswith(f"commit {commit[:7]}") or dirty is not False:
        raise SystemExit(f"run commit {commit} (dirty={dirty}) is not the rule's")

    entries = {c["id"]: c for c in manifest["clips"]}
    clips = [score_clip(entries[f"overlay_{vid}"], args.run_dir, commit=commit) for vid in rule["clips"]]

    n = len(clips)
    k = sum(c["assisted_within"] for c in clips)
    k_base = sum(c["baseline_within"] for c in clips)
    harmed = [c["clip"] for c in clips if c["baseline_within"] and not c["assisted_within"]]
    beats = sum(c["beats_centre"] for c in clips)
    summary = {
        "rule": args.rule, "run_dir": str(args.run_dir), "runner_commit": commit,
        "manifest_sha256": frozen, "threshold_m": THRESHOLD_M, "clips": clips,
        "V1": {"within": k, "n": n, "share": k / n, "wilson95": wilson(k, n),
               "in_sample": {"within": 14, "n": 15, "wilson95": wilson(14, 15)}},
        "V2": {"holds": not harmed, "baseline_within": k_base, "harmed": harmed},
        "V3": {"holds": beats >= 4, "beats_centre": beats, "n": n},
        "goal_check": {"within": k, "n": n, "holds": k == n},
        "median_m": {name: statistics.median(c[name] for c in clips)
                     for name in ("assisted_m", "baseline_track_m", "city_centre_m")},
    }

    print("| Clip | Assisted v4 (m) | Baseline track (m) | City centre (m) | Selected | "
          "Route in pass-1 / pass-2 disc | Mode | Declined expanded (m, post hoc) | Wall (s) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in clips:
        fine = "-" if c["fine_share"] is None else f"{c['fine_share']:.2f}"
        exp = "-" if c["declined_expanded_candidate_m_posthoc"] is None else \
            f"{c['declined_expanded_candidate_m_posthoc']:.1f}"
        print(f"| {c['name']} | {c['assisted_m']:.1f} | {c['baseline_track_m']:.1f} | "
              f"{c['city_centre_m']:.1f} | {c['selected_stage']} | "
              f"{c['pass1_share']:.2f} / {fine} | {c['mode']} | {exp} | {c['wall_s']:.0f} |")
    lo, hi = summary["V1"]["wilson95"]
    print(f"\nV1: {k}/{n} within {THRESHOLD_M:.0f} m (Wilson 95% {lo:.0%}-{hi:.0%}); "
          f"in-sample 14/15")
    print(f"V2 (no harm): {'holds' if not harmed else 'fails: ' + ', '.join(harmed)} "
          f"(baseline track {k_base}/{n})")
    print(f"V3 (beats centre on >=4): {'holds' if beats >= 4 else 'fails'} ({beats}/{n})")
    print(f"Goal 'all within 200 m': {k}/{n} -> {'holds' if k == n else 'not met'}")
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
