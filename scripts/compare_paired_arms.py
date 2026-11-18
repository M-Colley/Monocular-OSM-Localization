"""Paired comparison of two benchmark arms run on the same clips.

    python scripts/compare_paired_arms.py RUNS ORD_PREFIX NEW_PREFIX SPLIT [SPLIT ...] --out OUT.json

Reads ``RUNS/<prefix>_<split>/summary.json`` (from ``benchmark.py aggregate``)
for both arms, pairs clips that ran in both, and reports per clip: the answer
error of each arm, the 10 % better/worse/tie band, crossings of 200 m, the
pass-1 centre's distance to the ground-truth route, the coarse-to-fine choice
and the pass-1 reference count. Threshold decisions use repeat scores before
display rounding. Incomplete or ambiguous pairs are listed separately.
Re-aggregate the original run directory to resolve rounded legacy scores.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monocular_osm.geo import geodesic_distance_m  # noqa: E402

THRESHOLD_M = 200.0


def route(clip: str) -> np.ndarray:
    g = json.loads((ROOT / "ground_truth" / f"{clip}.json").read_text(encoding="utf-8"))
    w = g["waypoints"]
    w = ast.literal_eval(w) if isinstance(w, str) else w
    return np.array([[p["lat"], p["lon"]] for p in w], float)


def result(run_dir: Path, clip: str) -> dict:
    rep = run_dir / clip / "rep0"
    paths = [rep / "result.json", *sorted(rep.glob("pipeline_out/*/result.json"))]
    path = next((p for p in paths if p.is_file()), None)
    return json.loads(path.read_text(encoding="utf-8")) if path is not None else {}


def pass1_to_route(res: dict, gt: np.ndarray) -> float | None:
    c = (res.get("vpr_coarse_to_fine") or {}).get("pass1_center")
    return None if not c else float(np.min(geodesic_distance_m(c[0], c[1], gt[:, 0], gt[:, 1])))


def _answer_score(summary: dict, clip: str) -> tuple[float, bool] | None:
    """Return a median error and exact threshold classification when available."""
    clip_summary = summary["clips"][clip]
    n_reps = clip_summary.get("n_reps")
    if clip_summary.get("n_completed", n_reps) != n_reps:
        return None
    method = clip_summary["methods"].get("answer") or {}
    reps = [r for r in summary.get("reps", []) if r["clip"] == clip]
    if "reps" in summary and n_reps is not None and len(reps) != n_reps:
        return None
    if any(r.get("completed") is False for r in reps):
        return None
    scores = [r.get("scores", {}).get("answer") or {} for r in reps]
    values = ([s.get("symmetric_mean_m") for s in scores] if scores else
              [method.get("symmetric_mean_m", {}).get("median")])
    if any(v is None for v in values):
        return None
    errors = [float(v) for v in values]
    if any(math.isnan(v) or v < 0 for v in errors):
        return None
    error = median(errors)
    if scores and summary.get("repeat_score_precision") == "unrounded":
        return error, error <= THRESHOLD_M
    if error != THRESHOLD_M:
        return error, error < THRESHOLD_M
    # Old summaries can round repeat scores to two decimals, and may only
    # retain display medians. A strict majority of exact success flags still
    # settles where the median falls.
    flags = [s.get("success", {}).get("symmetric_mean_m", {}).get("200") for s in scores]
    if not flags:
        rate = method.get("success_rate", {}).get("symmetric_mean_m", {}).get("200")
        return (error, rate > 0.5) if rate is not None and rate != 0.5 else None
    if all(isinstance(flag, bool) for flag in flags):
        n_within = sum(flags)
        if 2 * n_within != len(flags):
            return error, 2 * n_within > len(flags)
        if all(v == round(v, 2) for v in errors):
            return None
    return error, True


def _report_metric(value: float) -> float | str:
    return "inf" if math.isinf(value) else value


def compare(runs: Path, ord_prefix: str, new_prefix: str, splits: list[str]) -> dict:
    from scipy.stats import binomtest

    rows = []
    unpaired = []
    within_ordinary = within_new = 0
    for split in splits:
        dirs = runs / f"{ord_prefix}_{split}", runs / f"{new_prefix}_{split}"
        summaries = [json.loads((d / "summary.json").read_text(encoding="utf-8")) for d in dirs]
        ordinary, new = (s["clips"] for s in summaries)
        for cid in ordinary | new:
            status_paths = [d / cid / "rep0" / "status.json" for d in dirs]
            match = {"split": split, "clip": cid, "name": (ordinary.get(cid) or new[cid])["name"]}
            if cid not in ordinary or cid not in new or not all(p.is_file() for p in status_paths):
                unpaired.append({**match, "reason": "missing summary or completed run in one arm"})
                continue
            status = [json.loads(p.read_text(encoding="utf-8")) for p in status_paths]
            scores = [_answer_score(s, cid) for s in summaries]
            if any(s.get("rc") is None for s in status) or any(score is None for score in scores):
                unpaired.append({**match, "reason": "incomplete or ambiguous score in one arm"})
                continue
            (a, a_within), (b, b_within) = scores
            within_ordinary += a_within
            within_new += b_within
            res = [result(d, cid) if (s["rc"] == 0 and not s.get("timed_out")
                                      and s.get("result_found") is not False) else {}
                   for d, s in zip(dirs, status)]
            gt = route(cid) if any((r.get("vpr_coarse_to_fine") or {}).get("pass1_center")
                                  for r in res) else None
            refs = [((r.get("vpr_decoding") or {}).get("pass1") or {}).get("n_references") for r in res]
            dist = [pass1_to_route(r, gt) if gt is not None else None for r in res]
            rows.append({
                **match,
                "ordinary_m": _report_metric(round(a, 1)), "new_m": _report_metric(round(b, 1)),
                "change": "tie" if a == b or abs(b - a) < 0.1 * a else "better" if b < a else "worse",
                "crossing": "gain" if b_within and not a_within else "loss" if a_within and not b_within else "",
                "pass1_to_route_m": [None if d is None else round(d) for d in dist],
                "c2f_kept": [(r.get("vpr_coarse_to_fine") or {}).get("kept") for r in res],
                "pass1_refs": refs,
                "wall_s": [round(s.get("wall_s", 0)) for s in status],
            })
    better = sum(r["change"] == "better" for r in rows)
    worse = sum(r["change"] == "worse" for r in rows)
    gains = [r["clip"] for r in rows if r["crossing"] == "gain"]
    losses = [r["clip"] for r in rows if r["crossing"] == "loss"]
    summary = {
        "n": len(rows), "better": better, "worse": worse,
        "tie": sum(r["change"] == "tie" for r in rows),
        "sign_test_one_sided_p": (round(binomtest(better, better + worse, 0.5,
                                                  alternative="greater").pvalue, 4)
                                  if better + worse else None),
        "gains": gains, "losses": losses,
        "within_200_ordinary": within_ordinary,
        "within_200_new": within_new,
        "median_ordinary_m": _report_metric(median(float(r["ordinary_m"]) for r in rows)) if rows else None,
        "median_new_m": _report_metric(median(float(r["new_m"]) for r in rows)) if rows else None,
    }
    report = {"summary": summary, "clips": rows}
    if unpaired:
        report["unpaired"] = unpaired
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path)
    ap.add_argument("ord_prefix")
    ap.add_argument("new_prefix")
    ap.add_argument("splits", nargs="+")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    out = compare(args.runs, args.ord_prefix, args.new_prefix, args.splits)
    for r in out["clips"]:
        print(f"{r['split']:5s} {r['name'][:32]:32s} "
              f"{float(r['ordinary_m']):>9,.0f} {float(r['new_m']):>9,.0f} "
              f"{r['change']:6s} {r['crossing']:4s} pass-1 to route {r['pass1_to_route_m']}")
    for r in out.get("unpaired", []):
        print(f"{r['split']:5s} {r['name'][:32]:32s} unpaired ({r['reason']})")
    print(json.dumps(out["summary"], indent=1))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
