"""Run the full pipeline end-to-end on every ground-truth clip and report.

Each clip reuses its cached VO trajectory + OSM graph, so this is a fast
regression/accuracy sweep across all GT clips with the current code. Prints
a final table: GT mean/start error for both the HEADLINE answer
(result["position"] — the anchored answer whenever an anchor fired) and the
raw matcher pick (result["matcher_position"]), plus rc so failed runs are
visible in the summary, not just in scrollback.

    python scripts/run_all_gt.py            # gated sweep (GT-centered discs)
    python scripts/run_all_gt.py --blind    # honest blind mode: drops the
                                            # --osm-around GT leak (mega-city
                                            # London keeps its point+radius
                                            # fetch — full-city is infeasible)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# (name, output-slug, [main.py args]) — config per clip is its best achievable
# with the current pipeline. Ulm uses the OCR-anchor path (street anchors);
# the rest are shape + scale-lock (no legible signage to anchor on).
ULM_4K = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input_4k.webm"
CLIPS = [
    ("Ulm (OCR-anchor 4K)", "ull8s4qydrk-ulm-germany-4k-drive-ulm-germany", [
        "--url", "https://www.youtube.com/watch?v=ULl8s4qydrk", "--skip-download",
        "--city", "Ulm, Germany", "--enable-ocr-anchor", "--ocr-video", ULM_4K,
        "--scale-lock", "--ground-truth-waypoints", "ground_truth/ulm_ULl8s4qydrk.json",
        "--no-splat"]),
    ("KITTI drive_0009", "local-05c0f063c75b-drive-0009-karlsruhe-germany", [
        "--video", "data/kitti/drive_0009.mp4", "--city", "Karlsruhe, Germany",
        "--osm-around", "49.009340,8.439418,737", "--vo-segment", "0:47",
        "--ground-truth-waypoints", "ground_truth/kitti_drive_0009.json",
        "--scale-lock", "--no-splat", "--no-aerial"]),
    ("KITTI drive_0033", "local-36a50c34107a-drive-0033-karlsruhe-germany", [
        "--video", "data/kitti/drive_0033.mp4", "--city", "Karlsruhe, Germany",
        "--osm-around", "48.970155,8.478770,968", "--vo-segment", "0:166",
        "--ground-truth-waypoints", "ground_truth/kitti_drive_0033.json",
        "--scale-lock", "--no-splat", "--no-aerial"]),
    ("comma2k19 (Daly City)", "local-88d9fe89bc4d-route-148-san-francisco-california-usa", [
        "--video", "data/comma/route_148.mp4", "--city", "San Francisco, California, USA",
        "--osm-around", "37.672466,-122.465576,1272", "--vo-segment", "0:240",
        "--ground-truth-waypoints", "ground_truth/comma_148.json",
        "--scale-lock", "--no-splat", "--no-aerial"]),
    ("London (Bloomsbury)", "local-73200bdd8068-input-london-uk", [
        "--video", "data/london_T4wTL3LpLqU/input.mp4", "--city", "London, UK",
        "--osm-around", "51.5223,-0.1267,1500", "--vo-segment", "0:295",
        "--ground-truth-waypoints", "ground_truth/london_T4wTL3LpLqU.json",
        "--scale-lock", "--no-splat", "--no-aerial"]),
]

# Mega-cities keep their point+radius graph fetch even in --blind mode: a
# full-city London graph is not enumerable, so the disc is an infra
# necessity there, not a GT leak we can drop.
MEGA_CITY_CLIPS = {"London (Bloomsbury)"}


def _strip_osm_around(args: list[str]) -> list[str]:
    """Return `args` without the `--osm-around <value>` GT leak."""
    out: list[str] = []
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "--osm-around":
            skip = True
            continue
        out.append(a)
    return out


def _stash_previous_result(res: Path) -> None:
    """Move an existing result.json aside so a failed run can't leave a
    stale file that the sweep would report as fresh numbers."""
    if res.exists():
        prev = res.with_name("result.prev.json")
        prev.unlink(missing_ok=True)
        res.replace(prev)


def _load_fresh_result(res: Path, rc: int, run_start: float) -> dict | None:
    """Read result.json only when the run succeeded AND the file was
    written by *this* run (mtime after the launch)."""
    if rc != 0 or not res.exists():
        return None
    if res.stat().st_mtime < run_start:
        return None  # stale leftover from an earlier run
    return json.load(open(res, encoding="utf-8"))


def _result_row(name: str, rc: int, result: dict | None) -> dict:
    """Extract the sweep-table row per the output contract:

    result["position"] is the HEADLINE answer (anchored when an anchor
    fired, check "source"); result["matcher_position"] is always the raw
    matcher pick. Older result.json files predate matcher_position — fall
    back to position so the sweep still degrades gracefully on them.
    """
    row: dict = {"name": name, "rc": rc}
    if result is None:
        return row
    p = result.get("position") or {}
    mp = result.get("matcher_position") or p
    sc = p.get("spatial_confidence", {})
    row.update(
        source=p.get("source", "matcher"),
        gt_mean=p.get("gt_mean_route_error_m"),
        gt_start=p.get("gt_start_error_m"),
        m_gt_mean=mp.get("gt_mean_route_error_m"),
        m_gt_start=mp.get("gt_start_error_m"),
        ranking=p.get("ranking", ""),
        streets=", ".join(p.get("street_names", [])[:3]),
        conf=sc.get("level"), spread=sc.get("spread_m"),
        n_hyp=len(p.get("hypotheses", [])),
    )
    return row


def _fmt(v, spec="%.0f", missing="-") -> str:
    return spec % v if v is not None else missing


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blind", action="store_true",
                    help="drop the GT-centered --osm-around discs (except "
                         "mega-city clips) for an honest blind number")
    opts = ap.parse_args(argv)

    rows = []
    for name, slug, args in CLIPS:
        if opts.blind and name not in MEGA_CITY_CLIPS:
            args = _strip_osm_around(args)
        print(f"\n{'='*70}\nRUNNING: {name}{' [blind]' if opts.blind else ''}\n{'='*70}",
              flush=True)
        res = ROOT / "output" / slug / "result.json"
        _stash_previous_result(res)
        log = ROOT / "output" / f"_gt_run_{slug[:20]}.log"
        run_start = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run([PY, "main.py", *args], cwd=ROOT,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        row = _result_row(name, rc, _load_fresh_result(res, rc, run_start))
        rows.append(row)
        print(f"  -> rc={rc}  source={row.get('source')}  "
              f"mean={row.get('gt_mean')}  start={row.get('gt_start')}  "
              f"matcher_mean={row.get('m_gt_mean')}", flush=True)

    mode = "BLIND (no GT discs)" if opts.blind else "gated"
    print(f"\n\n{'='*112}\nFINAL RESULTS — all GT clips, {mode} (current pipeline)\n{'='*112}")
    print(f"{'clip':24s} {'rc':>3s} {'source':>19s} {'mean(m)':>8s} {'start(m)':>9s} "
          f"{'mMean(m)':>9s} {'mStart(m)':>10s} {'conf':>7s} {'spread':>7s}  pick streets")
    print("-" * 112)
    for r in rows:
        print(f"{r['name']:24s} {r['rc']:>3d} {str(r.get('source', '-')):>19s} "
              f"{_fmt(r.get('gt_mean'), missing='FAIL'):>8s} "
              f"{_fmt(r.get('gt_start')):>9s} "
              f"{_fmt(r.get('m_gt_mean')):>9s} {_fmt(r.get('m_gt_start')):>10s} "
              f"{str(r.get('conf')):>7s} {_fmt(r.get('spread')):>7s}  "
              f"{r.get('streets', '')}")


if __name__ == "__main__":
    main()
