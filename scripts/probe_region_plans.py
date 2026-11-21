"""Exploratory, cheap geography probe: freeze metadata-only plans, then score.

Run `plan` before `score`. Planning never opens a ground-truth file and never
lists/downloads imagery. Only Nominatim geocoding may use the network. Scoring
reads the frozen plan and registered GT offline; it cannot alter any regions.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLIPS = ("overlay_wJsEQTCAg1c", "overlay_g5lnpYCk1Ec", "overlay_kxEDNj5L_yQ")
DISCLOSURE = ("While inspecting file format before this probe, the operator saw the Phoenix "
              "GT header, including a coordinate in its description. No centers, radii, "
              "ordering or constants were selected from that information. The planning "
              "process reads only uploader metadata, city and the saved v6 input search "
              "signature. Plans are frozen before the separate offline scoring process.")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def plan(args):
    from monocular_osm.location_prior import _default_geocode, resolve_retrieval_regions

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "plans.json"
    if target.exists():
        raise SystemExit(f"{target} is frozen; use a fresh output directory")
    manifest_path = args.root / "benchmarks/manifest.json"
    manifest = read(manifest_path)
    entries = {c["id"]: c for c in manifest["clips"]}
    cache_path = out / "geocodes.json"
    cache = read(cache_path) if cache_path.exists() else {}

    def geocode(query):
        if query not in cache:
            if args.offline_geocodes:
                raise RuntimeError(f"no frozen geocode for {query!r}")
            cache[query] = _default_geocode(query)
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        return cache[query]

    records = []
    for cid in args.clips:
        entry = entries[cid]
        if entry["split"] != "dev":
            raise ValueError("this exploratory probe accepts development clips only")
        baseline_result_path = (args.benchmark_root / "output/benchmark_runs" /
                                args.baseline_run / cid / "rep0/result.json")
        cfg = read(baseline_result_path)["provenance"]["config"]
        metadata_path = (args.root / "data/metadata_cache" /
                         (hashlib.sha256(entry["video"]["url"].encode()).hexdigest()[:16] + ".json"))
        metadata = read(metadata_path)       # cached uploader text only; no video lookup
        directory = Path(cfg["data_dir"])
        if not directory.is_absolute():
            directory = args.root / directory
        archive = directory / "_benchmark_cold_vpr" / args.baseline_run / cid / "rep0/mapillary"
        candidates = sorted(archive.glob("mly_ref_meta_*.json"))
        if len(candidates) != 1:
            raise ValueError(f"{cid}: expected one archived first-pass signature in {archive}")
        signature_path = candidates[0]
        signature = read(signature_path)["signature"]
        original = {"name": "v6 original search", "center": signature["center"],
                    "radius_m": signature["radius_m"], "origin": cfg.get("osm_around_source") or "city"}
        # Exactly the production resolver's settings; the production caller
        # then preserves the original first region's actual 3/8 km radius.
        regions = resolve_retrieval_regions(
            entry["city"], metadata.get("title"), metadata.get("description"),
            center=tuple(original["center"]), radius_m=args.radius_m,
            max_regions=args.max_regions, geocode_fn=geocode)
        if regions:
            regions[0] = {**regions[0], "radius_m": original["radius_m"]}
        city_only = resolve_retrieval_regions(
            entry["city"], metadata.get("title"), metadata.get("description"),
            radius_m=args.radius_m, max_regions=args.max_regions, geocode_fn=geocode)
        records.append({
            "clip": cid, "city": entry["city"], "title": metadata.get("title"),
            "description": metadata.get("description"), "url": entry["video"]["url"],
            "metadata_path": str(metadata_path), "metadata_sha256": sha(metadata_path),
            "baseline_signature_path": str(signature_path), "baseline_signature_sha256": sha(signature_path),
            "baseline": original, "production_regions": regions,
            "city_metadata_only_regions": city_only,
        })
        print(f"{cid}: froze {len(regions)} production regions", flush=True)
    result = {
        "exploratory": True, "phase": "metadata-only planning", "created_utc": now(),
        "ground_truth_read": False, "image_listing_or_download": False,
        "operator_disclosure": DISCLOSURE,
        "settings": {"radius_m": args.radius_m, "max_regions": args.max_regions,
                     "first_region_radius": "preserve v6 original, as production caller does"},
        "manifest_sha256": sha(manifest_path), "resolver_sha256": sha(ROOT / "monocular_osm/location_prior.py"),
        "script_sha256": sha(__file__), "geocodes_path": str(cache_path), "records": records,
    }
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "plans.sha256").write_text(sha(target) + "\n", encoding="utf-8")
    print(f"Frozen plans: {target}", flush=True)
    return 0


def plan_corridors(args):
    """Freeze a new variant from the old plan alone; no geocoding or GT I/O."""
    source = args.source.resolve()
    source_plan = source / "plans.json"
    source_hash = sha(source_plan)
    if source_hash != (source / "plans.sha256").read_text(encoding="utf-8").strip():
        raise ValueError("source frozen plan fingerprint changed")
    out = args.out.resolve()
    target = out / "plans.json"
    if target.exists():
        raise SystemExit(f"{target} is frozen; use a fresh output directory")
    from monocular_osm.regional_retrieval import fill_region_corridors

    result = copy.deepcopy(read(source_plan))
    for record in result["records"]:
        original = record["production_regions"]
        record["pre_corridor_regions"] = copy.deepcopy(original)
        record["production_regions"] = fill_region_corridors(
            original, max_regions=args.max_regions, radius_m=args.radius_m)
        print(f"{record['clip']}: {len(original)} original -> "
              f"{len(record['production_regions'])} corridor-plan regions", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    if (source / "geocodes.json").exists():
        shutil.copyfile(source / "geocodes.json", out / "geocodes.json")
        result["geocodes_path"] = str(out / "geocodes.json")
        result["geocodes_sha256"] = sha(out / "geocodes.json")
    result.update({
        "phase": "metadata-only corridor planning from an already frozen input plan",
        "created_utc": now(), "ground_truth_read": False,
        "source_plans_path": str(source_plan), "source_plans_sha256": source_hash,
        "script_sha256": sha(__file__), "resolver_sha256": sha(ROOT / "monocular_osm/location_prior.py"),
        "corridor_helper_sha256": sha(ROOT / "monocular_osm/regional_retrieval.py"),
        "corridor_disclosure": ("This generic corridor policy was proposed after the original "
                                "development coverage scores were seen. This new planning process "
                                "uses only the original frozen regions; no GT, new geocodes, "
                                "image listings, imagery, or embeddings are read."),
        "settings": {**result["settings"], "fill_corridors": True,
                     "max_regions": args.max_regions, "radius_m": args.radius_m},
    })
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "plans.sha256").write_text(sha(target) + "\n", encoding="utf-8")
    print(f"Frozen corridor plans: {target}", flush=True)
    return 0


def _route_samples(waypoints, step_m):
    """Geodesic segment midpoints weighted by arc length; no road snapping."""
    from monocular_osm.geo import WGS84
    points, weights, times = [], [], []
    for a, b in zip(waypoints, waypoints[1:]):
        azimuth, _, length = WGS84.inv(a["lon"], a["lat"], b["lon"], b["lat"])
        count = max(1, math.ceil(length / step_m))
        for i in range(count):
            fraction = (i + 0.5) / count
            lon, lat, _ = WGS84.fwd(a["lon"], a["lat"], azimuth, length * fraction)
            points.append((lat, lon))
            weights.append(length / count)
            times.append(a["t_sec"] + fraction * (b["t_sec"] - a["t_sec"]))
    return points, weights, times


def _coverage(regions, points, weights, times, waypoints):
    import numpy as np
    from monocular_osm.geo import geodesic_distance_m
    samples = np.asarray(points)

    def margins(coords):
        if not regions:
            return np.full(len(coords), float("inf"))
        return np.min([geodesic_distance_m(coords[:, 0], coords[:, 1], *r["center"])
                       - r["radius_m"] for r in regions], axis=0)

    excess = margins(samples)
    inside = excess <= 0
    weights = np.asarray(weights)
    intervals, start = [], None
    for index in range(len(inside) + 1):
        if index < len(inside) and not inside[index]:
            start = index if start is None else start
        elif start is not None:
            intervals.append({"start_t_sec": times[start], "end_t_sec": times[index - 1],
                              "length_m": float(weights[start:index].sum())})
            start = None
    waypoint_coords = np.array([[w["lat"], w["lon"]] for w in waypoints])
    return {"arc_fraction_inside": float(weights[inside].sum() / weights.sum()),
            "waypoint_fraction_inside": float(np.mean(margins(waypoint_coords) <= 0)),
            "route_length_m": float(weights.sum()),
            "max_distance_outside_union_m": max(0.0, float(excess.max())),
            "longest_uncovered_m": max((i["length_m"] for i in intervals), default=0.0),
            "uncovered_intervals": intervals}


def score(args):
    out = args.out.resolve()
    plan_path = out / "plans.json"
    if sha(plan_path) != (out / "plans.sha256").read_text(encoding="utf-8").strip():
        raise ValueError("frozen plan fingerprint changed; refusing to score")
    plans = read(plan_path)
    manifest_path = args.root / "benchmarks/manifest.json"
    if sha(manifest_path) != plans["manifest_sha256"]:
        raise ValueError("manifest changed after planning")
    entries = {c["id"]: c for c in read(manifest_path)["clips"]}
    scored = []
    for record in plans["records"]:
        entry = entries[record["clip"]]
        gt_path = args.root / entry["gt"]   # first GT read, in the separate process
        gt = read(gt_path)
        wp = gt["waypoints"]
        rows = [[round(float(w["t_sec"]), 3), round(float(w["lat"]), 7),
                 round(float(w["lon"]), 7)] for w in wp]
        digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
        if digest != entry["gt_waypoints_sha256"]:
            raise ValueError(f"{record['clip']}: registered GT fingerprint changed")
        points, weights, times = _route_samples(wp, args.step_m)
        alternatives = {"v6_original": [record["baseline"]],
                        "production_union": record["production_regions"],
                        "city_metadata_only_union": record["city_metadata_only_regions"]}
        if "pre_corridor_regions" in record:
            alternatives["pre_corridor_union"] = record["pre_corridor_regions"]
        scored.append({"clip": record["clip"], "city": record["city"],
                       "gt_sha256": digest, "gt_file_sha256": sha(gt_path),
                       "coverage": {name: _coverage(regions, points, weights, times, wp)
                                    for name, regions in alternatives.items()},
                       "per_production_disc": [{**r, "coverage": _coverage([r], points, weights, times, wp)}
                                               for r in record["production_regions"]]})
    result = {"exploratory": True, "phase": "offline scoring of already frozen plans",
              "created_utc": now(), "plans_sha256": sha(plan_path),
              "sampling_step_m": args.step_m, "operator_disclosure": DISCLOSURE,
              "corridor_disclosure": plans.get("corridor_disclosure"),
              "caveat": "Coverage measures geometric opportunity only; it says nothing about available imagery or retrieval accuracy. GT waypoints are joined by geodesic chords.",
              "records": scored}
    (out / "scores.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    corridor = bool(plans.get("settings", {}).get("fill_corridors"))
    heading = "corridor-plan" if corridor else "regional-plan"
    comparison_heading = "Original named-region union" if corridor else "City/metadata-only union"
    lines = [f"# Exploratory {heading} coverage probe", "", result["caveat"], "",
             "Plans were frozen in a separate process before these GT scores.", "",
             f"| Clip | v6 disc | Production union | {comparison_heading} | Longest production gap |",
             "|---|---:|---:|---:|---:|"]
    for r in scored:
        c = r["coverage"]
        comparison = c["pre_corridor_union"] if corridor else c["city_metadata_only_union"]
        lines.append(f"| {r['clip']} | {c['v6_original']['arc_fraction_inside']:.1%} | "
                     f"{c['production_union']['arc_fraction_inside']:.1%} | "
                     f"{comparison['arc_fraction_inside']:.1%} | "
                     f"{c['production_union']['longest_uncovered_m']:.0f} m |")
    lines += ["", "Operator disclosure: " + DISCLOSURE]
    if result["corridor_disclosure"]:
        lines += ["", "Corridor disclosure: " + result["corridor_disclosure"]]
    (out / "scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("plan", help="freeze metadata-only regions before GT is read")
    p.add_argument("--benchmark-root", type=Path, default=ROOT / ".claude/worktrees/bench")
    p.add_argument("--baseline-run", default="v6")
    p.add_argument("--clips", nargs="+", default=list(CLIPS))
    p.add_argument("--radius-m", type=float, default=3000.0)
    p.add_argument("--max-regions", type=int, default=6)
    p.add_argument("--offline-geocodes", action="store_true")
    p.set_defaults(fn=plan)
    c = sub.add_parser("corridors", help="freeze a corridor variant from existing plans without GT/geocoding")
    c.add_argument("--source", type=Path, default=ROOT / "output/region_plan_probe")
    c.add_argument("--radius-m", type=float, default=3000.0)
    c.add_argument("--max-regions", type=int, default=6)
    c.set_defaults(fn=plan_corridors)
    s = sub.add_parser("score", help="score frozen plans offline without changing regions")
    s.add_argument("--step-m", type=float, default=20.0)
    s.set_defaults(fn=score)
    for command in (p, s, c):
        command.add_argument("--root", type=Path, default=ROOT)
        default_out = "region_plan_corridor_probe" if command is c else "region_plan_probe"
        command.add_argument("--out", type=Path, default=ROOT / "output" / default_out)
    args = parser.parse_args(argv)
    if getattr(args, "step_m", 1) <= 0:
        parser.error("--step-m must be positive")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
