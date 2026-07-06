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
import os
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
    # Ulm #2 (LC9Sa--u5KE): added 2026-07-04 AFTER the Viterbi/fusion/top-K
    # placement stack was tuned on the other clips — a held-out
    # generalization check. Mapillary cache seeded from the Ulm-4K clip
    # (same geocode centre/radius/cap -> same fetch signature).
    ("Ulm #2 (held-out)", "lc9sa-u5ke-ulm-de-centre-ville-dashcam-4k-zhiroad-deutschland-ulmcity-ulm-german", [
        "--url", "https://www.youtube.com/watch?v=LC9Sa--u5KE", "--skip-download",
        "--city", "Ulm, Germany", "--vo-segment", "0:500", "--scale-lock",
        "--ground-truth-waypoints", "ground_truth/ulm_LC9Sa--u5KE.json",
        "--no-splat"]),
    # London: OCR super-res + the local OSM gazetteer (default on) recover the
    # sub-300 m anchors the rate-limited Nominatim path missed on this 720p
    # clip, dropping start error 1728 -> 295 m vs shape-only 1325 m.
    ("London (Bloomsbury)", "local-73200bdd8068-input-london-uk", [
        "--video", "data/london_T4wTL3LpLqU/input.mp4", "--city", "London, UK",
        "--osm-around", "51.5223,-0.1267,1500", "--vo-segment", "0:295",
        "--ground-truth-waypoints", "ground_truth/london_T4wTL3LpLqU.json",
        "--enable-ocr-anchor", "--ocr-super-res",
        "--ocr-video", "data/london_T4wTL3LpLqU/input_4k.webm",
        "--scale-lock", "--no-splat", "--no-aerial"]),
]

# Mega-cities keep their point+radius graph fetch even in --blind mode: a
# full-city London graph is not enumerable, so the disc is an infra
# necessity there, not a GT leak we can drop.
MEGA_CITY_CLIPS = {"London (Bloomsbury)"}

# Per-clip VGGT-Long trajectory override for --vggt-best. Only the clips where
# a staged VGGT-Long trajectory BEATS the default end-to-end (A/B 2026-07-05):
# London mean 70->43 / start 51->31, 0033 mean 137->118. Ulm-4K (172 vs 106)
# and comma (170 vs 102) REGRESS with VGGT-Long — its globally-consistent
# shape does not always yield a better matcher candidate pool — so they keep
# the default. Poses must be pre-staged (scratchpad/vggt_fleet.sh); a missing
# file makes the run fall back to the default trajectory (a warning, not an
# error), so the flag is safe on a fresh checkout.
VGGT_BEST = {
    "KITTI drive_0033":
        "data/local-36a50c34107a-drive-0033-karlsruhe-germany/vggt_long_poses.txt",
    "London (Bloomsbury)":
        "data/local-73200bdd8068-input-london-uk/vggt_long_poses.txt",
}


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
    ap.add_argument("--no-vpr", action="store_true",
                    help="skip the Mapillary VPR prior (on by default now — it "
                         "is the strongest blind lever; needs MLY_TOKEN env)")
    ap.add_argument("--orienternet", action="store_true",
                    help="add the gated OrienterNet metric head to every clip "
                         "(track+start-pin gated, so it can only help or "
                         "no-op; costs model load + Overpass tiles per run)")
    ap.add_argument("--vggt-best", action="store_true",
                    help="use a staged VGGT-Long trajectory on the clips where "
                         "it beats the default e2e (0033, London — see "
                         "VGGT_BEST); best-achievable benchmark, needs the "
                         "poses pre-staged (scratchpad/vggt_fleet.sh)")
    opts = ap.parse_args(argv)

    # Mapillary VPR is the best blind prior we have (3-31 m to route on every
    # clip); on by default. Search radius auto-caps to the osm_around disc.
    vpr = [] if opts.no_vpr else ["--use-vpr-prior", "--vpr-source", "mapillary"]
    if vpr and not os.environ.get("MLY_TOKEN"):
        print("NOTE: MLY_TOKEN not set — Mapillary refs are served from each "
              "clip's warm cache; clips without one fall back to kartaview.",
              flush=True)

    rows = []
    for name, slug, args in CLIPS:
        args = args + vpr
        if opts.orienternet:
            args = args + ["--use-orienternet"]
        if opts.vggt_best and name in VGGT_BEST:
            traj = ROOT / VGGT_BEST[name]
            if traj.exists():
                args = args + ["--vggt-long-trajectory", str(traj)]
            else:
                print(f"NOTE: {name} VGGT-Long poses not staged "
                      f"({traj}); using default trajectory.", flush=True)
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
