"""Exploratory municipality coverage: freeze a city-polygon grid, then score GT.

Planning uses a previously frozen metadata plan and cached Nominatim boundaries.
It reads no GT and makes no network requests. Every selected hexagonal cell
gets its own existing 3 km retrieval disc; oversized plans are refused instead
of silently dropping part of a municipality. No imagery or embeddings are run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import probe_region_plans as previous  # noqa: E402


def municipality_grid(geometry, city, *, radius_m=3000.0, max_cells=64):
    """Cover an arbitrary municipality with radius-r discs on a hex lattice.

    Hexagons are the lattice's Voronoi cells. Select every cell intersecting
    the municipality, including boundary cells with centres just outside it.
    A fixed one-metre geometric margin absorbs projection/buffer roundoff;
    each retrieval disc retains exactly the requested radius.
    """
    from pyproj import CRS, Transformer
    from shapely.geometry import Point, Polygon
    from shapely.ops import transform, unary_union

    if geometry.geom_type not in {"Polygon", "MultiPolygon"} or geometry.is_empty:
        raise ValueError("a nonempty municipality polygon is required")
    if radius_m <= 1 or max_cells < 1:
        raise ValueError("radius must exceed one metre and max_cells must be positive")
    centre = geometry.centroid
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={centre.y} +lon_0={centre.x} +datum=WGS84 +units=m")
    forward = Transformer.from_crs(4326, crs, always_xy=True)
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    polygon = transform(forward.transform, geometry)
    cell_radius = radius_m - 1.0
    dx, dy = 1.5 * cell_radius, math.sqrt(3) * cell_radius
    xmin, ymin, xmax, ymax = polygon.bounds
    cells = []
    for q in range(math.floor((xmin - cell_radius) / dx), math.ceil((xmax + cell_radius) / dx) + 1):
        rmin = math.floor((ymin - cell_radius) / dy - q / 2)
        rmax = math.ceil((ymax + cell_radius) / dy - q / 2)
        for r in range(rmin, rmax + 1):
            x, y = dx * q, dy * (r + q / 2)
            hexagon = Polygon([(x + cell_radius * math.cos(i * math.pi / 3),
                                y + cell_radius * math.sin(i * math.pi / 3)) for i in range(6)])
            if hexagon.intersects(polygon):
                cells.append((q, r, x, y))
    if len(cells) > max_cells:
        raise ValueError(f"municipality needs {len(cells)} cells, exceeds cap {max_cells}; no truncated plan")
    regions, discs = [], []
    for q, r, x, y in cells:
        lon, lat = inverse.transform(x, y)
        regions.append({"name": f"{city} cell {q},{r}", "center": [lat, lon],
                        "radius_m": float(radius_m), "origin": "municipality_grid",
                        "cell_axial": [q, r]})
        discs.append(Point(x, y).buffer(radius_m, quad_segs=96))
    uncovered = polygon.difference(unary_union(discs)).area
    if uncovered > 1.0:
        raise ValueError(f"numerical coverage audit failed: {uncovered:g} square metres uncovered")
    return regions, {"n_cells": len(cells), "municipality_area_km2": polygon.area / 1e6,
                     "uncovered_polygon_area_m2": uncovered, "disc_radius_m": radius_m,
                     "hex_circumradius_m": cell_radius, "projection_origin_latlon": [centre.y, centre.x],
                     "boundary_cells_may_be_outside_municipality": True,
                     "dense_reference_budget_at_1500_per_cell": len(cells) * 1500}


def plan(args):
    import osmnx as ox

    source = args.source.resolve()
    source_plan = source / "plans.json"
    if previous.sha(source_plan) != (source / "plans.sha256").read_text().strip():
        raise ValueError("source frozen plan changed")
    target = args.out / "plans.json"
    if target.exists():
        raise SystemExit(f"{target} is already frozen")
    original = previous.read(source_plan)
    result = copy.deepcopy(original)
    result["records"] = [r for r in result["records"] if r["clip"] in args.clips]
    if {r["clip"] for r in result["records"]} != set(args.clips):
        raise ValueError("requested clip missing from frozen source plan")
    boundaries = {"type": "FeatureCollection", "features": []}
    for record in result["records"]:
        # The same city lookup was cached by the earlier metadata resolver.
        # A transport guard makes accidental new HTTP requests fail visibly.
        with patch("requests.sessions.Session.send", side_effect=RuntimeError("planning is offline")):
            row = ox.geocode_to_gdf(record["city"], which_result=1).iloc[0]
        geometry = row.geometry
        grid, audit = municipality_grid(geometry, record["city"], radius_m=args.radius_m,
                                        max_cells=args.max_cells)
        record["pre_corridor_regions"] = copy.deepcopy(record["production_regions"])
        named = [r for r in record["production_regions"] if r.get("origin") == "video_named"]
        record["production_regions"] = grid + named
        record["municipality_grid_audit"] = audit
        boundaries["features"].append({"type": "Feature", "geometry": geometry.__geo_interface__,
                                        "properties": {"city": record["city"],
                                                       "osm_id": int(row["osm_id"]),
                                                       "osm_type": row["osm_type"]}})
        print(f"{record['clip']}: {len(grid)} municipal cells + {len(named)} named-place discs; "
              f"{audit['municipality_area_km2']:.1f} km²; polygon uncovered {audit['uncovered_polygon_area_m2']:.3f} m²")
    args.out.mkdir(parents=True, exist_ok=True)
    boundary_path = args.out / "city_boundaries.geojson"
    boundary_path.write_text(json.dumps(boundaries, ensure_ascii=False), encoding="utf-8")
    result.update({"phase": "offline city-polygon planning without GT", "created_utc": previous.now(),
                   "ground_truth_read": False, "network_requests": False,
                   "source_plans_path": str(source_plan), "source_plans_sha256": previous.sha(source_plan),
                   "script_sha256": previous.sha(__file__), "city_boundaries_path": str(boundary_path),
                   "city_boundaries_sha256": previous.sha(boundary_path),
                   "municipality_disclosure": ("Exploratory development iteration after geographic failure diagnosis. "
                                              "The generic planner covers the entire named municipality, accepts no GT, "
                                              "uses the existing 3 km radius, and refuses plans beyond the management cell cap."),
                   "settings": {"strategy": "municipality_hex_cover", "radius_m": args.radius_m,
                                "max_cells": args.max_cells, "original_city_disc_replaced_by_local_cells": True}})
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "plans.sha256").write_text(previous.sha(target) + "\n", encoding="utf-8")
    print(f"Frozen plan: {target}")
    return 0


def score(args):
    previous.score(args)              # first GT read, in this separate process
    plans = previous.read(args.out / "plans.json")
    scores = previous.read(args.out / "scores.json")
    lines = ["# Exploratory municipality-grid coverage probe", "", scores["caveat"], "",
             "Plans were frozen from the cached named-city polygon before separate GT scoring.", "",
             "| Clip | Original named regions | Municipality grid | Grid cells | Longest uncovered route |",
             "|---|---:|---:|---:|---:|"]
    for record, scored in zip(plans["records"], scores["records"]):
        coverage = scored["coverage"]
        before, after = coverage["pre_corridor_union"], coverage["production_union"]
        lines.append(f"| {record['clip']} | {before['arc_fraction_inside']:.1%} | "
                     f"{after['arc_fraction_inside']:.1%} | {record['municipality_grid_audit']['n_cells']} | "
                     f"{after['longest_uncovered_m']:.0f} m |")
    lines += ["", plans["municipality_disclosure"], "",
              "This is a geometry result. Independent dense budgets for every cell are expensive; "
              "a future hierarchical retrieval policy must be evaluated separately."]
    (args.out / "scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--source", type=Path, default=ROOT / "output/region_plan_probe")
    p.add_argument("--clips", nargs="+", default=["overlay_g5lnpYCk1Ec"])
    p.add_argument("--radius-m", type=float, default=3000.0)
    p.add_argument("--max-cells", type=int, default=64)
    p.set_defaults(fn=plan)
    s = sub.add_parser("score")
    s.add_argument("--step-m", type=float, default=20.0)
    s.set_defaults(fn=score)
    for command in (p, s):
        command.add_argument("--root", type=Path, default=ROOT)
        command.add_argument("--out", type=Path, default=ROOT / "output/region_plan_city_grid_probe")
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
