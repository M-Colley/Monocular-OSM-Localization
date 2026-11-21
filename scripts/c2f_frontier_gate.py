"""Would --vpr-c2f-extend fire? The pipeline's own gate on saved runs (GT-free).

For each clip of the given run directories this reads the saved ordinary
retrieval track and coarse-to-fine record and applies the gate of
``stages.prior._c2f_frontier_extend``: pass 2 kept, and a leading or trailing
run of at least 10 % of the decoded points in the outer quarter of the pass-2
disc. Ground truth is not read.

    python scripts/c2f_frontier_gate.py RUN_DIR [RUN_DIR ...] [--json out.json]

A run directory is a benchmark run (``<clip>/rep0/result.json``, e.g. an
ordinary shadow from ``metro_ring_paired.py``) or an assisted run, whose
baseline child (``<clip>/rep0/pipeline_out/*/children/*/baseline/result.json``)
holds the ordinary track.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monocular_osm.stages.prior import _EXT_MIN_RUN_FRAC, frontier_end_runs  # noqa: E402


def gate(result: dict) -> dict:
    info = result.get("vpr_coarse_to_fine") or {}
    track = (result.get("vpr_prior") or {}).get("track") or []
    if info.get("kept") != "pass-2" or not track:
        return {"applicable": False, "fires": False}
    centre = info.get("pass2_disc_center") or info["pass1_center"]
    lead, trail = frontier_end_runs([[p[1], p[2]] for p in track], centre, info["tight_radius_m"])
    min_run = max(3, math.ceil(_EXT_MIN_RUN_FRAC * len(track)))
    return {"applicable": True, "lead": lead, "trail": trail, "min_run": min_run,
            "fires": max(lead, trail) >= min_run}


def results(run_dir: Path):
    for clip in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        plain = clip / "rep0" / "result.json"
        child = sorted(clip.glob("rep0/pipeline_out/*/children/*/baseline/result.json"))
        path = plain if plain.exists() and not child else (child[0] if child else None)
        if path is not None:
            yield clip.name, path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    rows = []
    for run_dir in args.run_dirs:
        for clip, path in results(run_dir):
            row = {"run": str(run_dir), "clip": clip, "result": str(path),
                   **gate(json.loads(path.read_text(encoding="utf-8")))}
            rows.append(row)
            detail = (f"lead {row['lead']:2d} trail {row['trail']:2d} (min {row['min_run']})"
                      if row["applicable"] else "pass 2 not kept")
            print(f"{clip:26s} {'FIRES' if row['fires'] else '-':5s} {detail}")
    print(f"\n{sum(r['fires'] for r in rows)} of {len(rows)} clips fire")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
