"""Paired comparison of two benchmark arms run on the same clips.

    python scripts/compare_paired_arms.py RUNS ORD_PREFIX NEW_PREFIX SPLIT [SPLIT ...] --out OUT.json

Reads ``RUNS/<prefix>_<split>/summary.json`` (from ``benchmark.py aggregate``)
for both arms, pairs clips that ran in both, and reports per clip: the answer
error of each arm, the 10 % better/worse/tie band, crossings of 200 m, the
pass-1 centre's distance to the ground-truth route, the coarse-to-fine choice
and the pass-1 reference count. Used for ``dense_pass1_dev_2026_11``.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

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
    paths = sorted(rep.glob("pipeline_out/*/result.json")) or [rep / "result.json"]
    return json.loads(paths[0].read_text(encoding="utf-8"))


def pass1_to_route(res: dict, gt: np.ndarray) -> float | None:
    c = (res.get("vpr_coarse_to_fine") or {}).get("pass1_center")
    return None if not c else float(np.min(geodesic_distance_m(c[0], c[1], gt[:, 0], gt[:, 1])))


def compare(runs: Path, ord_prefix: str, new_prefix: str, splits: list[str]) -> dict:
    from scipy.stats import binomtest
    rows = []
    for split in splits:
        dirs = runs / f"{ord_prefix}_{split}", runs / f"{new_prefix}_{split}"
        summaries = [json.loads((d / "summary.json").read_text(encoding="utf-8"))["clips"] for d in dirs]
        for cid in summaries[0]:
            if not all((d / cid / "rep0" / "status.json").exists() for d in dirs):
                continue   # listed but never run in one arm
            err = [float(s[cid]["methods"]["answer"]["symmetric_mean_m"]["median"]) for s in summaries]
            res = [result(d, cid) for d in dirs]
            status = [json.loads((d / cid / "rep0" / "status.json").read_text(encoding="utf-8")) for d in dirs]
            gt = route(cid)
            refs = [((r.get("vpr_decoding") or {}).get("pass1") or {}).get("n_references") for r in res]
            a, b = err
            dist = [pass1_to_route(r, gt) for r in res]
            rows.append({
                "split": split, "clip": cid, "name": summaries[0][cid]["name"],
                "ordinary_m": round(a, 1), "new_m": round(b, 1),
                "change": "tie" if abs(b - a) < 0.1 * a else "better" if b < a else "worse",
                "crossing": ("gain" if a > THRESHOLD_M >= b else
                             "loss" if b > THRESHOLD_M >= a else ""),
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
        "within_200_ordinary": sum(r["ordinary_m"] <= THRESHOLD_M for r in rows),
        "within_200_new": sum(r["new_m"] <= THRESHOLD_M for r in rows),
        "median_ordinary_m": float(np.median([r["ordinary_m"] for r in rows])) if rows else None,
        "median_new_m": float(np.median([r["new_m"] for r in rows])) if rows else None,
    }
    return {"summary": summary, "clips": rows}


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
        print(f"{r['split']:5s} {r['name'][:32]:32s} {r['ordinary_m']:>9,.0f} {r['new_m']:>9,.0f} "
              f"{r['change']:6s} {r['crossing']:4s} pass-1 to route {r['pass1_to_route_m']}")
    print(json.dumps(out["summary"], indent=1))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
