"""Freeze all registered dev-city municipality plans, then score them offline.

This exploratory probe never runs retrieval or reads GT during planning. It
retains failed/over-budget cities. Existing public boundary responses are read
without modifying their caches; optional HTTP misses are bounded and saved only
under this probe's output directory. Score is a separate process and accepts
only the unchanged frozen plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import probe_region_plans as previous  # noqa: E402


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _frozen_source(source, out):
    path = source / "plans.json"
    if previous.sha(path) != (source / "plans.sha256").read_text(encoding="utf-8").strip():
        raise ValueError(f"source frozen plan changed: {path}")
    target = out / "inputs" / f"{source.name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    return previous.read(target), {"path": str(path.resolve()), "sha256": previous.sha(target),
                                   "saved_copy": str(target.resolve())}


class BoundaryInputs:
    """Read production-equivalent first-result geocodes with a strict HTTP bound."""

    def __init__(self, root, out, *, allow_network=False, max_network_requests=12):
        self.root, self.out = root, out
        self.allow_network, self.max_network_requests = allow_network, max_network_requests
        self.requests = []
        self.observed = {}

    def __call__(self, city):
        import requests
        from shapely.geometry import shape

        tag = hashlib.sha256(city.encode()).hexdigest()[:16]
        url = requests.Request("GET", "https://nominatim.openstreetmap.org/search", params={
            "format": "json", "polygon_geojson": 1, "dedupe": 0, "limit": 1, "q": city,
        }).prepare().url
        cache_name = hashlib.sha1(url.encode()).hexdigest() + ".json"
        response, origin = None, None
        # Cached production boundary, followed by exact OSMnx request caches.
        dedicated = self.root / "data/_city_boundaries" / f"city_{tag}.geojson"
        if dedicated.exists():
            blob = previous.read(dedicated)
            if blob.get("city") == city:
                geometry, source = blob["geometry"], blob.get("source", {})
                origin = {"kind": "existing_dedicated_boundary", "path": str(dedicated.resolve()),
                          "sha256": previous.sha(dedicated)}
                return self._save(city, tag, geometry, source, origin)
        for folder in (self.root / "cache", self.root / ".claude/worktrees/bench/cache"):
            path = folder / cache_name
            if path.exists():
                response = previous.read(path)
                origin = {"kind": "existing_osmnx_http_cache", "path": str(path.resolve()),
                          "sha256": previous.sha(path), "request_url": url}
                break
        if response is None:
            if not self.allow_network:
                raise RuntimeError("no cached first-result municipality boundary; network disabled")
            if len(self.requests) >= self.max_network_requests:
                raise RuntimeError("bounded municipality geocoding request budget exhausted")
            time.sleep(1.05)  # Nominatim permits at most one request per second.
            event = {"city": city, "url": url, "started_utc": previous.now()}
            self.requests.append(event)
            try:
                reply = requests.get(url, timeout=(8, 25), headers={
                    "User-Agent": "Monocular-OSM-Localization municipality coverage research",
                })
                event["status_code"] = reply.status_code
                reply.raise_for_status()  # No retries, including 429/504.
                response = reply.json()
                raw = self.out / "http_cache" / cache_name
                _write(raw, response)
                origin = {"kind": "bounded_new_http_request", "request_url": url,
                          "path": str(raw.resolve()), "sha256": previous.sha(raw)}
            except Exception as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise
        if not isinstance(response, list) or not response:
            raise ValueError("first-result municipality geocode returned no results")
        # Same first-result choice as geocode_to_gdf(city, which_result=1).
        result = sorted(response, key=lambda x: x["importance"], reverse=True)[0]
        source = {"query": city, "provider": "Nominatim (production-equivalent first result)",
                  "osm_id": int(result["osm_id"]), "osm_type": result["osm_type"],
                  "display_name": result.get("display_name")}
        source.update({key: result[key] for key in
                       ("class", "category", "type", "addresstype", "place_rank") if key in result})
        return self._save(city, tag, result["geojson"], source, origin)

    def _save(self, city, tag, geometry, source, origin):
        from shapely.geometry import shape

        target = self.out / "boundary_inputs" / f"city_{tag}.geojson"
        _write(target, {"city": city, "geometry": geometry, "source": source, "origin": origin})
        self.observed[city] = {"boundary_input": str(target.resolve()),
                               "boundary_input_sha256": previous.sha(target),
                               "geometry_sha256": hashlib.sha256(json.dumps(
                                   geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                               "source": source, "origin": origin}
        return shape(geometry), source


def plan(args):
    from monocular_osm.city_retrieval import CityCoverageLimitError, plan_city_regions

    out = args.out.resolve()
    target = out / "plans.json"
    if target.exists():
        raise SystemExit(f"{target} is already frozen; choose a fresh directory")
    manifest_path = args.root / "benchmarks/manifest.json"
    manifest = previous.read(manifest_path)
    entries = [c for c in manifest["clips"] if c["split"] == "dev"]
    cities = sorted({c["city"] for c in entries})
    originals, original_source = _frozen_source(args.metadata_source, out)
    corridors, corridor_source = _frozen_source(args.corridor_source, out)
    original_by_id = {r["clip"]: r for r in originals["records"]}
    corridor_by_id = {r["clip"]: r for r in corridors["records"]}
    boundary = BoundaryInputs(args.root, out, allow_network=args.allow_network,
                              max_network_requests=args.max_network_requests)
    records = []
    for city in cities:
        record = {"city": city, "clips": [c["id"] for c in entries if c["city"] == city],
                  "status": "unavailable", "regions": []}
        try:
            regions, audit = plan_city_regions(city, out / "planner_boundary_cache",
                                               radius_m=args.radius_m, max_cells=args.max_cells,
                                               boundary_fn=boundary)
            record.update(status="ok", regions=regions, audit=audit)
        except CityCoverageLimitError as exc:
            record.update(status="refused_over_budget", error=str(exc),
                          required_cells=exc.required_cells, lower_bound=exc.lower_bound,
                          max_cells=exc.max_cells)
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record.update(boundary.observed.get(city, {}))
        records.append(record)
        print(f"{city}: {record['status']} ({len(record['regions'])} published cells)", flush=True)
    clips = []
    for c in entries:
        item = {"clip": c["id"], "city": c["city"], "available_locally": c["available_locally"],
                "gt_path": c["gt"], "registered_gt_sha256": c["gt_waypoints_sha256"]}
        if c["id"] in original_by_id:
            r, cr = original_by_id[c["id"]], corridor_by_id[c["id"]]
            if r["city"] != c["city"] or cr["city"] != c["city"]:
                raise ValueError("frozen metadata-plan city differs from registered city")
            item.update(metadata_regions=r["production_regions"],
                        metadata_corridor_regions=cr["production_regions"],
                        original_disc=[r["baseline"]])
        clips.append(item)
    result = {"exploratory": True, "phase": "municipality planning before GT scoring",
              "created_utc": previous.now(), "ground_truth_read": False,
              "image_listing_or_download": False, "model_forward": False,
              "settings": {"radius_m": args.radius_m, "max_cells": args.max_cells,
                           "allow_network": args.allow_network,
                           "max_network_requests": args.max_network_requests,
                           "network_retry_count": 0, "geocode_result": 1,
                           "membership": "every distinct registered dev city, including unavailable clips"},
              "manifest_path": str(manifest_path.resolve()), "manifest_sha256": previous.sha(manifest_path),
              "script_sha256": previous.sha(__file__),
              "planner_sha256": previous.sha(ROOT / "monocular_osm/city_retrieval.py"),
              "coverage_scorer_sha256": previous.sha(ROOT / "scripts/probe_region_plans.py"),
              "metadata_sources": [original_source, corridor_source],
              "inherited_operator_disclosure": originals.get("operator_disclosure"),
              "disclosure": "Exploratory generic grid chosen after earlier dev geography diagnosis. "
                            "All registered dev city strings are included; no GT selects cells or parameters. "
                            "Over-budget/non-polygon municipalities remain failures with no partial plan.",
              "network_requests": boundary.requests, "cities": records, "clips": clips}
    _write(target, result)
    (out / "plans.sha256").write_text(previous.sha(target) + "\n", encoding="utf-8")
    print(f"FROZEN {target}: {len(cities)} cities, {len(clips)} clips; SHA256 {previous.sha(target)}", flush=True)
    return 0


def score(args):
    target = args.out / "plans.json"
    if previous.sha(target) != (args.out / "plans.sha256").read_text(encoding="utf-8").strip():
        raise ValueError("frozen plan changed")
    plans = previous.read(target)
    if previous.sha(args.root / "benchmarks/manifest.json") != plans["manifest_sha256"]:
        raise ValueError("manifest changed after planning")
    if previous.sha(ROOT / "scripts/probe_region_plans.py") != plans["coverage_scorer_sha256"]:
        raise ValueError("coverage scorer changed after planning")
    cities = {r["city"]: r for r in plans["cities"]}
    scored = []
    for entry in plans["clips"]:
        city = cities[entry["city"]]
        record = {"clip": entry["clip"], "city": entry["city"],
                  "available_locally": entry["available_locally"], "city_plan_status": city["status"],
                  "n_cells": len(city["regions"]), "coverage": {}}
        try:
            path = args.root / entry["gt_path"]  # First GT read; separate score process.
            wp = previous.read(path)["waypoints"]
            rows = [[round(float(w["t_sec"]), 3), round(float(w["lat"]), 7),
                     round(float(w["lon"]), 7)] for w in wp]
            digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
            if digest != entry["registered_gt_sha256"]:
                raise ValueError("registered GT fingerprint changed")
            points, weights, times = previous._route_samples(wp, args.step_m)
            alternatives = {}
            if city["status"] == "ok":
                alternatives["municipality_union"] = city["regions"]
            if "metadata_regions" in entry:
                alternatives.update(metadata_union=entry["metadata_regions"],
                                    metadata_corridor_union=entry["metadata_corridor_regions"],
                                    original_disc=entry["original_disc"])
                if city["status"] == "ok":
                    alternatives.update(municipality_plus_metadata=city["regions"] + entry["metadata_regions"],
                                        municipality_plus_corridors=city["regions"] + entry["metadata_corridor_regions"])
            record.update(gt_status="ok", gt_sha256=digest, gt_file_sha256=previous.sha(path),
                          coverage={name: previous._coverage(regions, points, weights, times, wp)
                                    for name, regions in alternatives.items()})
        except Exception as exc:
            record.update(gt_status="failed", error=f"{type(exc).__name__}: {exc}")
        scored.append(record)
    result = {"exploratory": True, "created_utc": previous.now(), "plans_sha256": previous.sha(target),
              "phase": "separate offline scoring of frozen municipality plans", "sampling_step_m": args.step_m,
              "caveat": "Geometric opportunity only; no imagery availability or retrieval accuracy is measured. "
                        "Registered GT waypoints are joined by geodesic chords. Refused cities have no municipal score.",
              "records": scored}
    _write(args.out / "scores.json", result)
    lines = ["# Exploratory registered-dev municipality coverage", "", result["caveat"], "",
             "All plans, failures, source fingerprints and settings were frozen before this separate scoring process.", "",
             "| Clip | Local video | City plan | Cells | Municipal coverage | + metadata | + corridors |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in scored:
        cov = r["coverage"]
        def pct(key):
            return f"{cov[key]['arc_fraction_inside']:.1%}" if key in cov else "—"
        lines.append(f"| {r['clip']} | {r['available_locally']} | {r['city_plan_status']} | {r['n_cells']} | "
                     f"{pct('municipality_union')} | {pct('municipality_plus_metadata')} | "
                     f"{pct('municipality_plus_corridors')} |")
    lines += ["", "Failed city plans:"] + [f"- {r['city']}: {r['error']}" for r in plans["cities"] if r["status"] != "ok"]
    lines += ["", plans["disclosure"]]
    (args.out / "scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--radius-m", type=float, default=3000.0)
    p.add_argument("--max-cells", type=int, default=64)
    p.add_argument("--allow-network", action="store_true")
    p.add_argument("--max-network-requests", type=int, default=12)
    p.add_argument("--metadata-source", type=Path, default=ROOT / "output/region_plan_probe")
    p.add_argument("--corridor-source", type=Path, default=ROOT / "output/region_plan_corridor_probe")
    p.set_defaults(fn=plan)
    s = sub.add_parser("score")
    s.add_argument("--step-m", type=float, default=20.0)
    s.set_defaults(fn=score)
    for command in (p, s):
        command.add_argument("--root", type=Path, default=ROOT)
        command.add_argument("--out", type=Path, default=ROOT / "output/municipality_fleet_probe")
    args = parser.parse_args(argv)
    if getattr(args, "step_m", 1) <= 0 or getattr(args, "max_network_requests", 0) < 0:
        parser.error("step must be positive; request budget must be nonnegative")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
