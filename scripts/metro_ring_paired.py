"""Paired scoring of a --vpr-metro-ring run against its own ordinary track.

A metro-ring run records the ordinary retrieval track it had before the ring
step (``result["metro_ring_retrieval"]["baseline_track"]``). ``shadow`` copies
a benchmark run directory and, where the ring replaced that track, puts the
ordinary track back as the answer (running median of 5, exactly as
``--answer track`` builds it), so the ordinary benchmark aggregator scores both
runs the same way. ``compare`` reads the two aggregated summaries.

    python scripts/metro_ring_paired.py shadow RUNS/metro_test2 RUNS/metro_test2_ordinary
    python scripts/benchmark.py aggregate --run-name metro_test2 --runs-dir RUNS --offline
    python scripts/benchmark.py aggregate --run-name metro_test2_ordinary --runs-dir RUNS --offline
    python scripts/metro_ring_paired.py compare RUNS/metro_test2 RUNS/metro_test2_ordinary
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monocular_osm.baselines import vpr_track_only  # noqa: E402
from monocular_osm.position import build_answer_report  # noqa: E402

COPIED = ("argv.json", "log.txt", "status.json")


def ordinary_result(result: dict) -> dict:
    """``result`` with the ring's track and answer replaced by the ordinary ones."""
    ring = result.get("metro_ring_retrieval") or {}
    baseline = ring.get("baseline_track")
    if ring.get("kept") in (None, "baseline") or not baseline:
        return result
    out = json.loads(json.dumps(result))
    out["vpr_prior"]["track"] = baseline
    pick = vpr_track_only(out)
    out["answer"] = build_answer_report(pick.route_latlon, mode="track", method=pick.name,
                                        source=pick.source)
    out["metro_ring_retrieval"] = {**ring, "shadow": "ordinary track restored for paired scoring"}
    return out


def shadow(run_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists")
    out_dir.mkdir(parents=True)
    shutil.copy2(run_dir / "run_config.json", out_dir / "run_config.json")
    for rep in sorted(run_dir.glob("*/rep*")):
        target = out_dir / rep.relative_to(run_dir)
        target.mkdir(parents=True)
        for name in COPIED:
            if (rep / name).exists():
                shutil.copy2(rep / name, target / name)
        if (rep / "result.json").exists():
            result = json.loads((rep / "result.json").read_text(encoding="utf-8"))
            (target / "result.json").write_text(json.dumps(ordinary_result(result)), encoding="utf-8")
    print(f"wrote {out_dir}")


def compare(ring_dir: Path, ordinary_dir: Path, threshold_m: float = 200.0) -> None:
    ring = json.loads((ring_dir / "summary.json").read_text(encoding="utf-8"))["clips"]
    ordinary = json.loads((ordinary_dir / "summary.json").read_text(encoding="utf-8"))["clips"]
    print("| Clip | Confidence | Ring ran | Kept | Ordinary (m) | With ring (m) | Change | Wall (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for cid, clip in ring.items():
        rep = ring_dir / cid / "rep0"
        if not (rep / "status.json").exists():
            # listed by the manifest but never started (e.g. the runner's disk guard refused it)
            print(f"| {clip['name']} | - | not run | - | - | - | - | - |")
            continue
        result = json.loads((rep / "result.json").read_text(encoding="utf-8")) if (rep / "result.json").exists() else {}
        status = json.loads((rep / "status.json").read_text(encoding="utf-8"))
        metro = result.get("metro_ring_retrieval") or {}
        gate = metro.get("gate") or {}

        def err(c):
            method = (c or {}).get("methods", {}).get("answer") or {}
            value = (method.get("symmetric_mean_m") or {}).get("median")
            return None if value is None else float(value)

        a, b = err(ordinary.get(cid)), err(clip)
        change = ("-" if a is None or b is None else "tie" if abs(b - a) < 0.1 * a
                  else "better" if b < a else "worse")
        flag = ("" if a is None or b is None else
                " (gain)" if a > threshold_m >= b else " (loss)" if b > threshold_m >= a else "")
        conf = gate.get("confidence")
        print(f"| {clip['name']} | {'-' if conf is None else f'{conf:.3f}'} | "
              f"{'no' if metro.get('status') == 'not_triggered' else 'yes'} | {metro.get('kept', '-')} | "
              f"{'-' if a is None else f'{a:,.0f}'} | {'-' if b is None else f'{b:,.0f}'} | {change}{flag} | "
              f"{status.get('wall_s', 0):,.0f} |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shadow")
    s.add_argument("run_dir", type=Path)
    s.add_argument("out_dir", type=Path)
    c = sub.add_parser("compare")
    c.add_argument("ring_dir", type=Path)
    c.add_argument("ordinary_dir", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "shadow":
        shadow(args.run_dir, args.out_dir)
    else:
        compare(args.ring_dir, args.ordinary_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
