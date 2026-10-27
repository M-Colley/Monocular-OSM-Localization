"""Opt-in, GT-free hierarchical retrieval over a whole municipality.

Every city-polygon cell gets a coarse local budget; visual evidence and temporal
continuity choose cells to densify. Ordinary retrieval remains a candidate.
These experimental heuristics do not provide calibrated confidence.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import operator
import os
from pathlib import Path

import numpy as np

from .regional_retrieval import merge_region_tracks

MAX_CITY_CELLS = 64
CITY_GRID_VERSION = 1


class CityCoverageLimitError(ValueError):
    """Full-city coverage cannot fit the management bound; never truncate it."""

    def __init__(self, required_cells, max_cells, *, lower_bound=False):
        self.required_cells = int(required_cells)
        self.max_cells = int(max_cells)
        self.lower_bound = bool(lower_bound)
        super().__init__(f"municipality needs {'at least ' if lower_bound else ''}{required_cells} "
                         f"cells, exceeds limit {max_cells}; full-city plan refused")


def _positive_integer(value, name):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, (bool, np.bool_)) or result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def municipality_regions(geometry, city, *, radius_m=3000.0, max_cells=MAX_CITY_CELLS):
    """Cover all intersecting municipal hex cells with independent local discs.

    The existing radius is preserved; hex cells use a fixed one-metre numerical
    margin. Boundary-cell centres may be just outside the city. No GT is accepted.
    """
    from pyproj import CRS, Transformer
    from shapely import transform, union_all
    from shapely.geometry import Point, Polygon

    max_cells = _positive_integer(max_cells, "max_cells")
    radius_m = float(radius_m)
    if not np.isfinite(radius_m) or radius_m <= 1:
        raise ValueError("radius_m must be finite and exceed one metre")
    if (geometry.geom_type not in {"Polygon", "MultiPolygon"} or geometry.is_empty
            or not geometry.is_valid):
        raise ValueError("a valid nonempty municipality polygon is required")
    if not np.isfinite(geometry.bounds).all():
        raise ValueError("municipality coordinates must be finite")
    centre = geometry.centroid
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={centre.y} +lon_0={centre.x} +datum=WGS84 +units=m")
    forward = Transformer.from_crs(4326, crs, always_xy=True)
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    polygon = transform(geometry, forward.transform, interleaved=False)
    if not np.isfinite(polygon.bounds).all() or not np.isfinite(polygon.area):
        raise ValueError("municipality has no finite local projection")
    cell_radius = radius_m - 1.0
    hex_area = 1.5 * math.sqrt(3) * cell_radius ** 2
    lower_bound = math.ceil(polygon.area / hex_area)
    if lower_bound > max_cells:
        raise CityCoverageLimitError(lower_bound, max_cells, lower_bound=True)
    dx, dy = 1.5 * cell_radius, math.sqrt(3) * cell_radius
    xmin, ymin, xmax, ymax = polygon.bounds
    regions, discs = [], []
    for q in range(math.floor((xmin - cell_radius) / dx), math.ceil((xmax + cell_radius) / dx) + 1):
        rmin = math.floor((ymin - cell_radius) / dy - q / 2)
        rmax = math.ceil((ymax + cell_radius) / dy - q / 2)
        for r in range(rmin, rmax + 1):
            x, y = dx * q, dy * (r + q / 2)
            cell = Polygon([(x + cell_radius * math.cos(i * math.pi / 3),
                             y + cell_radius * math.sin(i * math.pi / 3)) for i in range(6)])
            if not cell.intersects(polygon):
                continue
            if len(regions) >= max_cells:
                raise CityCoverageLimitError(len(regions) + 1, max_cells, lower_bound=True)
            lon, lat = inverse.transform(x, y)
            regions.append({"name": f"{city} cell {q},{r}", "center": [float(lat), float(lon)],
                            "radius_m": radius_m, "origin": "municipality_grid", "cell_axial": [q, r]})
            discs.append(Point(x, y).buffer(radius_m, quad_segs=96))
    uncovered = float(polygon.difference(union_all(discs)).area)
    if uncovered > 1.0:
        raise ValueError(f"municipality coverage audit failed: {uncovered:g} square metres uncovered")
    return regions, {
        "grid_version": CITY_GRID_VERSION, "n_cells": len(regions), "max_cells": max_cells,
        "radius_m": radius_m, "municipality_area_km2": float(polygon.area / 1e6),
        "uncovered_polygon_area_m2": uncovered, "hex_circumradius_m": cell_radius,
        "projection_origin_latlon": [float(centre.y), float(centre.x)],
    }


def _city_boundary(city):
    import osmnx as ox

    row = ox.geocode_to_gdf(city, which_result=1).iloc[0]
    return row.geometry, {"query": city, "provider": "Nominatim via osmnx",
                          "osm_id": int(row["osm_id"]), "osm_type": str(row["osm_type"])}


def plan_city_regions(city, cache_dir, *, radius_m=3000.0, max_cells=MAX_CITY_CELLS,
                      boundary_fn=None):
    """Reuse a shared city polygon, then make a complete bounded cell plan.

    First use may read osmnx's HTTP cache or geocode the named city. No roads,
    imagery or GT are fetched. A dedicated boundary snapshot is shared by city.
    """
    from shapely.geometry import mapping, shape

    cache_dir = Path(cache_dir)
    tag = hashlib.sha256(city.encode()).hexdigest()[:16]
    path = cache_dir / f"city_{tag}.geojson"
    cached = False
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        if blob.get("city") != city:
            raise ValueError("cached municipality belongs to a different city")
        geometry, source = shape(blob["geometry"]), blob.get("source") or {}
        cached = True
    else:
        geometry, source = (boundary_fn or _city_boundary)(city)
        # Refuse an unusable/oversized polygon before publishing its cache.
        regions, audit = municipality_regions(geometry, city, radius_m=radius_m, max_cells=max_cells)
        blob = {"schema": 1, "city": city, "geometry": mapping(geometry), "source": source}
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    if cached:
        regions, audit = municipality_regions(geometry, city, radius_m=radius_m, max_cells=max_cells)
    geometry_hash = hashlib.sha256(json.dumps(mapping(geometry), sort_keys=True,
                                              separators=(",", ":")).encode()).hexdigest()
    return regions, {**audit, "city": city, "boundary_cache": str(path),
                     "boundary_cache_hit": cached, "boundary_sha256": geometry_hash,
                     "boundary_source": source}


def _validated_track(track, timestamps, expected_indices):
    indices, coords, similarities = track
    indices = np.asarray(indices)
    coords, similarities = np.asarray(coords, float), np.asarray(similarities, float)
    if (indices.ndim != 1 or not len(indices) or not np.isfinite(indices).all()
            or not np.array_equal(indices, indices.astype(int)) or np.any(np.diff(indices) <= 0)
            or indices[0] < 0 or indices[-1] >= len(timestamps)
            or coords.shape != (len(indices), 2) or similarities.shape != (len(indices),)):
        raise ValueError("city retrieval track has invalid query indices or shapes")
    indices = indices.astype(int)
    if expected_indices is not None and not np.array_equal(indices, expected_indices):
        raise ValueError("city retrieval track describes different query frames")
    valid = (np.isfinite(coords).all(axis=1) & np.isfinite(similarities)
             & (np.abs(coords[:, 0]) <= 90) & (np.abs(coords[:, 1]) <= 180))
    if not valid.any():
        raise ValueError("city retrieval track has no finite WGS84 candidates")
    return indices, coords, similarities


def select_city_tracks(regions, baseline_track, timestamps, retrieve, *,
                       coarse_cap=250, fine_regions=3, fine_cap=1500):
    """Search every cell, densify supported cells, retain baseline evidence.

    ``retrieve(region, cap=..., phase='coarse'|'fine', index=...)`` returns
    ``(track_or_None, provenance_dict)``. The existing regional cosine/continuity
    rule ranks fine cells by supported-query count, median cosine, then index.
    Coarse candidates remain available if fine retrieval fails or the drive
    spans more cells than the fine budget. No model or GT is accessed here.
    """
    coarse_cap = _positive_integer(coarse_cap, "coarse_cap")
    fine_regions = _positive_integer(fine_regions, "fine_regions")
    fine_cap = _positive_integer(fine_cap, "fine_cap")
    if len(regions) > MAX_CITY_CELLS:
        raise CityCoverageLimitError(len(regions), MAX_CITY_CELLS)
    times = np.asarray(timestamps, float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("city retrieval timestamps must be finite and ordered")
    expected = None
    if baseline_track is not None:
        expected = _validated_track(baseline_track, times, None)[0]
    coarse, fine, coarse_records, fine_records = [], [], [], []

    def fetch(index, phase, cap):
        nonlocal expected
        region = regions[index]
        record = {**deepcopy(region), "cell_index": index, "phase": phase, "ref_cap": cap}
        try:
            track, provenance = retrieve(deepcopy(region), cap=cap, phase=phase, index=index)
            record.update(provenance or {})
            if track is None:
                record["status"] = "failed" if record.get("error") else "unavailable"
                return None, record
            track = _validated_track(track, times, expected)
            expected = track[0]
            record.update(status="ok", median_similarity=float(np.nanmedian(track[2])))
            return track, record
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            return None, record

    for index in range(len(regions)):
        track, record = fetch(index, "coarse", coarse_cap)
        coarse_records.append(record)
        if track is not None:
            coarse.append((index, track))
    baseline = [baseline_track] if baseline_track is not None else []
    # Coarse candidates share a reference budget. Let them compete for the
    # fine budget without the ordinary region's larger pool suppressing a
    # useful thin cell before it can be densified. Baseline remains in the
    # final comparison, so this does not force the answer away from it.
    coarse_candidates = [track for _, track in coarse]
    _coarse_track, coarse_selection = merge_region_tracks(coarse_candidates, times)
    counts = coarse_selection.get("region_frame_counts") or []
    support = []
    for candidate_index, (index, track) in enumerate(coarse):
        count = counts[candidate_index] if candidate_index < len(counts) else 0
        if count:
            support.append({"cell_index": index, "supported_frames": count,
                            "median_similarity": float(np.nanmedian(track[2]))})
    support.sort(key=lambda s: (-s["supported_frames"], -s["median_similarity"], s["cell_index"]))
    selected_cells = [s["cell_index"] for s in support[:fine_regions]]
    for index in selected_cells:
        track, record = fetch(index, "fine", fine_cap)
        fine_records.append(record)
        if track is not None:
            fine.append((index, track))
    labels = ([{"phase": "baseline", "cell_index": None}] if baseline else [])
    labels += [{"phase": "coarse", "cell_index": index} for index, _ in coarse]
    labels += [{"phase": "fine", "cell_index": index} for index, _ in fine]
    candidates = baseline + coarse_candidates + [track for _, track in fine]
    merged, final_selection = merge_region_tracks(candidates, times)
    selected_sources = [labels[i] for i in final_selection.get("selected_regions", [])]
    selected_city_frames = sum(s["phase"] != "baseline" for s in selected_sources)
    if baseline_track is not None and not selected_city_frames:
        merged = baseline_track           # Preserve the exact ordinary track.
    diagnostics = {
        "experimental": True, "confidence_calibrated": False,
        "status": "ok" if coarse and merged is not None else "unavailable", "n_cells": len(regions),
        "coarse_cap": coarse_cap, "fine_cap": fine_cap, "fine_region_limit": fine_regions,
        "reference_budget": {"coarse_max": len(regions) * coarse_cap,
                             "fine_max": min(len(regions), fine_regions) * fine_cap,
                             "fine_requested": len(selected_cells) * fine_cap},
        "n_usable_coarse": len(coarse), "n_usable_fine": len(fine),
        "coarse_regions": coarse_records, "fine_regions": fine_records,
        "coarse_selection": coarse_selection, "visually_supported_cells": support,
        "coarse_ranking_pool": "equal-budget city cells only; ordinary baseline retained for final comparison",
        "comparison_calibration": {"calibrated": False,
                                   "different_reference_pool_sizes": True,
                                   "raw_cosine_across_pools_is_experimental": True},
        "densified_cell_indices": selected_cells,
        "final_selection": {**final_selection, "selected_sources": selected_sources},
        "selected_city_frames": selected_city_frames,
    }
    return merged, diagnostics
