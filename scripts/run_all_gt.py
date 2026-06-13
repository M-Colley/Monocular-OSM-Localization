"""Run the full pipeline end-to-end on every ground-truth clip and report.

Each clip reuses its cached VO trajectory + OSM graph, so this is a fast
regression/accuracy sweep across all GT clips with the current code. Prints
a final table: GT mean/start error, the headline pick, and the calibrated
spatial confidence + top hypotheses.

    python scripts/run_all_gt.py
"""

from __future__ import annotations

import json
import subprocess
import sys
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


def main() -> None:
    rows = []
    for name, slug, args in CLIPS:
        print(f"\n{'='*70}\nRUNNING: {name}\n{'='*70}", flush=True)
        log = ROOT / "output" / f"_gt_run_{slug[:20]}.log"
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run([PY, "main.py", *args], cwd=ROOT,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        res = ROOT / "output" / slug / "result.json"
        row = {"name": name, "rc": rc}
        if res.exists():
            p = json.load(open(res, encoding="utf-8")).get("position", {})
            sc = p.get("spatial_confidence", {})
            row.update(
                gt_mean=p.get("gt_mean_route_error_m"),
                gt_start=p.get("gt_start_error_m"),
                ranking=p.get("ranking", ""),
                streets=", ".join(p.get("street_names", [])[:3]),
                conf=sc.get("level"), spread=sc.get("spread_m"),
                n_hyp=len(p.get("hypotheses", [])),
            )
        rows.append(row)
        print(f"  -> rc={rc}  mean={row.get('gt_mean')}  start={row.get('gt_start')}",
              flush=True)

    print(f"\n\n{'='*92}\nFINAL RESULTS — all GT clips (current pipeline)\n{'='*92}")
    print(f"{'clip':24s} {'mean(m)':>8s} {'start(m)':>9s} {'conf':>7s} "
          f"{'spread':>7s}  pick streets")
    print("-" * 92)
    for r in rows:
        m = r.get("gt_mean"); s = r.get("gt_start")
        print(f"{r['name']:24s} {('%.0f'%m if m is not None else 'FAIL'):>8s} "
              f"{('%.0f'%s if s is not None else '-'):>9s} {str(r.get('conf')):>7s} "
              f"{('%.0f'%r['spread'] if r.get('spread') is not None else '-'):>7s}  "
              f"{r.get('streets','')}")


if __name__ == "__main__":
    main()
