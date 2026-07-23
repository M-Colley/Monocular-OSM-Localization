"""Fleet A/B sweep: run every GT clip with the shape+VPR core, BASELINE vs
ENHANCED (+ the tile3d skyline channel, adaptive-weighted, auto source =
official LoD2 in Germany / worldwide OSM LoD1 elsewhere). Records the
headline + matcher GT errors and the tile3d channel status per clip, and
writes an incremental JSON the visualization reads.

    python scripts/tile3d_gt_sweep.py Berlin,Ulm4K,KITTI33   # subset
    python scripts/tile3d_gt_sweep.py all
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "scratchpad_sweep"
OUT.mkdir(exist_ok=True)
# Separate output files let a German (fast) and an OSM (slow) sweep run
# concurrently without racing the same results JSON.
RESULTS = OUT / os.environ.get("SWEEP_OUT", "gt_sweep_results.json")

# (key, city, country, slug, vo_segment, stride, gt_file, expect_source,
#  extra_args) — cached-VO segments with GT files trimmed to match.
CLIPS = {
    "Berlin": ("Berlin", "DE", "lblkr2ek0w4-berlin-germany-4k-driving-tour-alexanderplatz-potsdamer-platz-brande",
               "0:420", 3, "ground_truth/berlin_lBlKR2ek0w4_0-420.json", "LoD2",
               ["--url", "https://www.youtube.com/watch?v=lBlKR2ek0w4", "--skip-download",
                "--city", "Berlin, Germany", "--osm-around", "52.514106,13.397760,1945"]),
    "Ulm4K": ("Ulm", "DE", "ull8s4qydrk-ulm-germany-4k-drive-ulm-germany",
              "0:300", 3, "ground_truth/ulm_ULl8s4qydrk_0-300.json", "LoD2",
              ["--url", "https://www.youtube.com/watch?v=ULl8s4qydrk", "--skip-download",
               "--city", "Ulm, Germany", "--osm-around", "48.401751,9.986253,803"]),
    "Ulm2": ("Ulm #2", "DE", "lc9sa-u5ke-ulm-de-centre-ville-dashcam-4k-zhiroad-deutschland-ulmcity-ulm-german",
             "0:500", 4, "ground_truth/ulm_LC9Sa--u5KE.json", "LoD2",
             ["--url", "https://www.youtube.com/watch?v=LC9Sa--u5KE", "--skip-download",
              "--city", "Ulm, Germany"]),
    "KITTI9": ("Karlsruhe", "DE", "local-05c0f063c75b-drive-0009-karlsruhe-germany",
               "0:47", 3, "ground_truth/kitti_drive_0009.json", "LoD2",
               ["--video", "data/kitti/drive_0009.mp4", "--city", "Karlsruhe, Germany",
                "--osm-around", "49.009340,8.439418,737"]),
    "KITTI33": ("Karlsruhe", "DE", "local-36a50c34107a-drive-0033-karlsruhe-germany",
                "0:166", 3, "ground_truth/kitti_drive_0033.json", "LoD2",
                ["--video", "data/kitti/drive_0033.mp4", "--city", "Karlsruhe, Germany",
                 "--osm-around", "48.970155,8.478770,968"]),
    "comma": ("Daly City", "US", "local-88d9fe89bc4d-route-148-san-francisco-california-usa",
              "0:240", 3, "ground_truth/comma_148.json", "OSM",
              ["--video", "data/comma/route_148.mp4", "--city", "San Francisco, California, USA",
               "--osm-around", "37.672466,-122.465576,1272"]),
    "London": ("London", "UK", "local-73200bdd8068-input-london-uk",
               "0:295", 3, "ground_truth/london_T4wTL3LpLqU.json", "OSM",
               ["--video", "data/london_T4wTL3LpLqU/input.mp4", "--city", "London, UK",
                "--osm-around", "51.5223,-0.1267,1500"]),
    "Malaga": ("Malaga", "ES", "local-f71f13fb95d3-input-m-laga-spain",
               "0:105", 3, "ground_truth/malaga_extract07.json", "OSM",
               ["--video", "data/malaga-urban-extract07-spain/input.mp4", "--city", "Málaga, Spain",
                "--osm-around", "36.72424,-4.47621,1100"]),
    "Boreas": ("Vaughan", "CA", "local-79a9281e5d36-input-vaughan-ontario-canada",
               "0:179", 3, "ground_truth/boreas_glenshields.json", "OSM",
               ["--video", "data/boreas-glenshields-vaughan-canada/input.mp4",
                "--city", "Vaughan, Ontario, Canada", "--osm-around", "43.796,-79.476,1300"]),
}

CORE = ["--scale-lock", "--no-splat", "--no-aerial",
        "--use-vpr-prior", "--vpr-source", "mapillary"]

# Deployable (GT-free) mode: drop the GT-seeded --osm-around disc and seed the
# search from the city name + coarse-to-fine VPR instead. Some clips need the
# operator to name the right TOWN (not the metro) — the honest deployable
# assumption; comma's route is in Daly City, not San Francisco's centroid.
DEPLOY_CITY = {"comma": "Daly City, California, USA"}
RUN_TIMEOUT = 1500   # s per pipeline run; a hung mega-city graph fetch is marked, not left to hang


def _strip_osm_around(args: list) -> list:
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "--osm-around":
            skip = True
            continue
        out.append(a)
    return out


def _swap_city(args: list, city: str) -> list:
    out = list(args)
    for i, a in enumerate(out):
        if a == "--city" and i + 1 < len(out):
            out[i + 1] = city
    return out


def _gt(result: dict) -> dict:
    p = result.get("position") or {}
    mp = result.get("matcher_position") or p
    t3 = result.get("tile3d") or {}
    return {
        "source": p.get("source"),
        "start": p.get("gt_start_error_m"),
        "mean": p.get("gt_mean_route_error_m"),
        "m_start": mp.get("gt_start_error_m"),
        "m_mean": mp.get("gt_mean_route_error_m"),
        "streets": p.get("street_names", [])[:3],
        "tile3d_provider": t3.get("provider"),
        "tile3d_buildings": t3.get("n_buildings"),
        "tile3d_adaptive_weight": t3.get("adaptive_weight"),
    }


def _run(key, arm, extra, *, deployable=False):
    name, country, slug, seg, stride, gt, src, args = CLIPS[key]
    if deployable:
        args = _strip_osm_around(args)
        if key in DEPLOY_CITY:
            args = _swap_city(args, DEPLOY_CITY[key])
        extra = extra + ["--vpr-coarse-to-fine", "--coarse-from-video",
                         "--coarse-from-frames"]
    res_path = ROOT / "output" / slug / "result.json"
    if res_path.exists():
        res_path.replace(res_path.with_suffix(".sweepprev.json"))
    argv = ([PY, "main.py", *args, "--vo-segment", seg, "--frame-stride", str(stride),
             "--ground-truth-waypoints", gt] + CORE + extra)
    log = OUT / f"_{key}_{arm}.log"
    t0 = time.time()
    try:
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=RUN_TIMEOUT).returncode
    except subprocess.TimeoutExpired:
        return {"rc": -9, "secs": RUN_TIMEOUT, "note": "timeout"}
    dt = time.time() - t0
    row = {"rc": rc, "secs": round(dt)}
    if rc == 0 and res_path.exists() and res_path.stat().st_mtime >= t0:
        row.update(_gt(json.load(open(res_path, encoding="utf-8"))))
    return row


def main(argv):
    sel = argv[0] if argv else "all"
    keys = list(CLIPS) if sel == "all" else [k for k in sel.split(",") if k in CLIPS]
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    deploy_mode = bool(os.environ.get("DEPLOYABLE"))
    for key in keys:
        name, country = CLIPS[key][0], CLIPS[key][1]
        print(f"\n{'='*60}\n{key} ({name}, {country})"
              f"{' [DEPLOYABLE]' if deploy_mode else ''}\n{'='*60}", flush=True)
        entry = data.get(key, {"name": name, "country": country,
                               "expect_source": CLIPS[key][6]})
        if deploy_mode:
            # GT-free: no --osm-around, seed from city name + coarse-to-fine VPR
            dep = _run(key, "deployable", [], deployable=True)
            print(f"  deployable: {dep}", flush=True)
            entry["deployable"] = dep
        else:
            base = _run(key, "baseline", [])
            print(f"  baseline: {base}", flush=True)
            entry["baseline"] = base
            if not os.environ.get("BASELINE_ONLY"):
                enh = _run(key, "enhanced", ["--use-tile3d", "--tile3d-adaptive"])
                print(f"  enhanced: {enh}", flush=True)
                entry["enhanced"] = enh
        data[key] = entry
        RESULTS.write_text(json.dumps(data, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main(sys.argv[1:])
