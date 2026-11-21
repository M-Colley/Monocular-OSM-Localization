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


class CityBoundaryTypeError(ValueError):
    """A geocoded polygon cannot establish an administrative city boundary."""

    def __init__(self, message, validation, source):
        self.validation = validation
        self.source = source
        super().__init__(message)


def _boundary_validation(source, legacy_unknown_policy):
    if legacy_unknown_policy not in {"refuse", "allow_unverified"}:
        raise ValueError("legacy_unknown_policy must be refuse or allow_unverified")
    classification = {key: source.get(key) for key in
                      ("class", "category", "type", "addresstype", "place_rank")}
    kinds = {str(source[key]).strip().lower() for key in ("class", "category")
             if source.get(key) is not None}
    feature_type = str(source.get("type") or "").strip().lower()
    validation = {"classification": classification, "legacy_unknown_policy": legacy_unknown_policy,
                  "administrative_boundary_verified": False}
    if (kinds and kinds != {"boundary"}) or (feature_type and feature_type != "administrative"):
        validation["status"] = "non_administrative"
        raise CityBoundaryTypeError(
            f"geocoded polygon is not an administrative boundary ({classification}); municipality plan refused",
            validation, source)
    if feature_type == "administrative":
        return {**validation, "status": "administrative", "administrative_boundary_verified": True}
    validation["status"] = "classification_unknown"
    if legacy_unknown_policy == "refuse":
        raise CityBoundaryTypeError(
            "city boundary classification is missing; legacy snapshot cannot establish an administrative "
            "municipality (policy=refuse)", validation, source)
    return validation


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


def metro_ring_regions(city, city_centre, *, outer_radius_m, searched_centre, searched_radius_m,
                       cell_radius_m, max_cells=MAX_CITY_CELLS):
    """Cover a disc around the city's geocode, minus the disc already searched.

    The area is ``disc(city_centre, outer_radius_m)`` without
    ``disc(searched_centre, searched_radius_m)``: for an ordinary run the
    pass-1 city disc (so an annulus), for a title- or plate-seeded run the seed
    disc. It reuses the municipality hex cover and its coverage audit, so cells
    near the inner edge may reach inside the searched disc. No GT is accepted.
    """
    from pyproj import CRS, Transformer
    from shapely import transform
    from shapely.geometry import Point

    outer_radius_m, searched_radius_m = float(outer_radius_m), float(searched_radius_m)
    if not (np.isfinite(outer_radius_m) and np.isfinite(searched_radius_m)) or outer_radius_m <= 0:
        raise ValueError("metro ring radii must be finite and positive")
    lat0, lon0 = map(float, city_centre)
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m")
    forward = Transformer.from_crs(4326, crs, always_xy=True)
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    sx, sy = forward.transform(float(searched_centre[1]), float(searched_centre[0]))
    area = Point(0.0, 0.0).buffer(outer_radius_m, quad_segs=96).difference(
        Point(sx, sy).buffer(max(searched_radius_m, 0.0), quad_segs=96))
    if area.is_empty:
        raise ValueError("the searched disc already covers the metro disc")
    geometry = transform(area, inverse.transform, interleaved=False)
    regions, audit = municipality_regions(geometry, city, radius_m=cell_radius_m, max_cells=max_cells)
    for region in regions:
        region.update(origin="metro_ring", name=region["name"].replace(" cell ", " metro cell "))
    return regions, {**audit, "plan": "metro_ring", "city_centre": [lat0, lon0],
                     "outer_radius_m": outer_radius_m,
                     "searched_centre": [float(searched_centre[0]), float(searched_centre[1])],
                     "searched_radius_m": searched_radius_m,
                     "ring_area_km2": float(area.area / 1e6)}


def _city_boundary(city):
    import osmnx as ox

    row = ox.geocode_to_gdf(city, which_result=1).iloc[0]
    source = {"query": city, "provider": "Nominatim via osmnx",
              "osm_id": int(row["osm_id"]), "osm_type": str(row["osm_type"])}
    for key in ("class", "category", "type", "addresstype", "place_rank", "display_name"):
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            source[key] = int(value) if key == "place_rank" else str(value)
    return row.geometry, source


def plan_city_regions(city, cache_dir, *, radius_m=3000.0, max_cells=MAX_CITY_CELLS,
                      boundary_fn=None, legacy_unknown_policy="refuse"):
    """Reuse a shared city polygon, then make a complete bounded cell plan.

    First use may read osmnx's HTTP cache or geocode the named city. No roads,
    imagery or GT are fetched. A dedicated boundary snapshot is shared by city.
    Known non-administrative polygons are refused. Legacy snapshots without
    taxonomy are refused by default; an explicit research-only allow_unverified
    policy keeps their missing administrative identity visible in diagnostics.
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
        validation = _boundary_validation(source, legacy_unknown_policy)
        # Refuse an unusable/oversized polygon before publishing its cache.
        regions, audit = municipality_regions(geometry, city, radius_m=radius_m, max_cells=max_cells)
        blob = {"schema": 2, "city": city, "geometry": mapping(geometry), "source": source}
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    if cached:
        validation = _boundary_validation(source, legacy_unknown_policy)
        regions, audit = municipality_regions(geometry, city, radius_m=radius_m, max_cells=max_cells)
    geometry_hash = hashlib.sha256(json.dumps(mapping(geometry), sort_keys=True,
                                              separators=(",", ":")).encode()).hexdigest()
    return regions, {**audit, "city": city, "boundary_cache": str(path),
                     "boundary_cache_hit": cached, "boundary_sha256": geometry_hash,
                     "boundary_source": source, "boundary_validation": validation}


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


def _raw_cell_vote_input(audit, track, timestamps):
    """Validate the actual consumer audit, then retain only its row maxima."""
    from .kartaview_vpr import _audit_array_sha

    if not isinstance(audit, dict) or audit.get("status") != "complete":
        raise ValueError("raw-votes requires a complete actual retrieval audit")
    indices, coordinates, cosines = track
    raw = np.asarray(audit["raw_similarities"])
    refs, ids = np.asarray(audit["ref_latlon"]), np.asarray(audit["ref_ids"])
    ix, times = np.asarray(audit["query_indices"]), np.asarray(audit["query_times"])
    chosen = np.asarray(audit["decoded_ref_indices"])
    if (raw.dtype != np.float32 or raw.ndim != 2 or not 1 <= raw.shape[0] <= 80
            or not 1 <= raw.shape[1] <= 20000 or raw.size > 1600000
            or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1.0001)
            or refs.dtype != np.float64 or refs.shape != (raw.shape[1], 2)
            or not np.isfinite(refs).all() or np.any(np.abs(refs[:, 0]) > 90)
            or np.any(np.abs(refs[:, 1]) > 180)
            or ids.shape != (raw.shape[1],) or ids.dtype.kind not in "US"
            or len(set(ids.astype(str).tolist())) != len(ids) or not np.all(ids.astype(str) != "")
            or ix.shape != (raw.shape[0],) or ix.dtype.kind not in "iu"
            or not np.array_equal(ix, indices) or np.any(ix < 0)
            or np.any(ix >= len(timestamps)) or np.any(ix[1:] <= ix[:-1])
            or times.dtype != np.float64 or times.shape != ix.shape
            or not np.array_equal(times, timestamps[ix.astype(np.intp)])
            or chosen.shape != ix.shape or chosen.dtype.kind not in "iu"
            or np.any(chosen < 0) or np.any(chosen >= len(refs))):
        raise ValueError("raw-votes matrix/reference/query grid is invalid or nonfinite")
    actual = raw[np.arange(len(ix)), chosen.astype(np.intp)].astype(np.float64)
    if (not np.array_equal(actual, cosines)
            or not np.array_equal(actual, audit["decoded_cosines"])
            or not np.array_equal(refs[chosen.astype(np.intp)], coordinates)):
        raise ValueError("raw-votes audit does not describe the actual returned track")
    identity = {key: audit[key] for key in ("query_pixels_sha256", "query_descriptors_sha256",
                "actual_model_state_sha256", "backbone", "preprocessing", "query_mode", "score_kind", "decoder")}
    identity.update(query_indices_sha256=_audit_array_sha(ix), query_times_sha256=_audit_array_sha(times))
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                                    allow_nan=False).encode()).hexdigest()
    hashes = [audit[key] for key in ("query_pixels_sha256", "query_descriptors_sha256",
                                   "actual_model_state_sha256", "ref_embeddings_sha256")]
    if (any(type(h) is not str or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes)
            or audit["score_kind"] != "raw_cosine_at_decoded_reference"
            or identity != audit["comparison_identity"] or digest(identity) != audit["comparison_identity_sha256"]):
        raise ValueError("raw-votes actual model/query identity is incomplete")
    # Decoder choices can differ by pool. Raw appearance must be identical.
    appearance = {key: value for key, value in identity.items() if key != "decoder"}
    return raw.max(axis=1), appearance, {"status": "complete", "n_references": raw.shape[1],
        "raw_similarities_sha256": _audit_array_sha(raw), "appearance_sha256": digest(appearance)}


def select_city_tracks(regions, baseline_track, timestamps, retrieve, *,
                       coarse_cap=250, fine_regions=3, fine_cap=1500, cell_selector="decoded"):
    """Search every cell, densify supported cells, retain baseline evidence.

    ``retrieve(region, cap=..., phase='coarse'|'fine', index=...)`` returns
    ``(track_or_None, provenance_dict)``. The existing regional cosine/continuity
    rule ranks fine cells by supported-query count, median cosine, then index.
    Opt-in ``raw-votes`` takes a third return value: the actual populated audit.
    Each query votes for its top three coarse-cell raw maxima; total votes,
    then existing cell order choose the same bounded number of fine cells.
    Coarse candidates remain available if fine retrieval fails or the drive
    spans more cells than the fine budget. No model or GT is accessed here.
    """
    coarse_cap = _positive_integer(coarse_cap, "coarse_cap")
    fine_regions = _positive_integer(fine_regions, "fine_regions")
    fine_cap = _positive_integer(fine_cap, "fine_cap")
    if cell_selector not in {"decoded", "raw-votes"}:
        raise ValueError("city cell selector must be decoded or raw-votes")
    if len(regions) > MAX_CITY_CELLS:
        raise CityCoverageLimitError(len(regions), MAX_CITY_CELLS)
    times = np.asarray(timestamps, float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("city retrieval timestamps must be finite and ordered")
    expected = None
    if baseline_track is not None:
        expected = _validated_track(baseline_track, times, None)[0]
    coarse, fine, coarse_records, fine_records = [], [], [], []
    raw_rows, appearance, raw_errors = {}, None, []

    def fetch(index, phase, cap):
        nonlocal expected, appearance
        region = regions[index]
        record = {**deepcopy(region), "cell_index": index, "phase": phase, "ref_cap": cap}
        track = None
        try:
            returned = retrieve(deepcopy(region), cap=cap, phase=phase, index=index)
            track, provenance = returned[:2]
            record.update(provenance or {})
            if track is None:
                record["status"] = "failed" if record.get("error") else "unavailable"
                return None, record
            track = _validated_track(track, times, expected)
            expected = track[0]
            record.update(status="ok", median_similarity=float(np.nanmedian(track[2])))
            if cell_selector == "raw-votes":
                try:
                    maxima, identity, proof = _raw_cell_vote_input(returned[2] if len(returned) == 3 else None, track, times)
                    if appearance is not None and identity != appearance:
                        raise ValueError("raw-votes coarse cells have different actual model/query identities")
                    appearance = identity
                    record["raw_evidence"] = proof
                    if phase == "coarse":
                        raw_rows[index] = maxima
                except (KeyError, TypeError, ValueError, IndexError) as exc:
                    record["raw_evidence"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
                    if phase == "coarse":
                        raw_errors.append(index)
                    else:
                        record.update(status="failed", error=record["raw_evidence"]["reason"])
                        return None, record
            return track, record
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if cell_selector == "raw-votes" and phase == "coarse" and track is not None:
                raw_errors.append(index)
                record["raw_evidence"] = {"status": "unavailable", "reason": record["error"]}
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
    if cell_selector == "decoded":
        _coarse_track, coarse_selection = merge_region_tracks(coarse_candidates, times)
    else:
        coarse_selection = {}
    counts = coarse_selection.get("region_frame_counts") or []
    support = []
    raw_available = cell_selector != "raw-votes" or bool(coarse) and not raw_errors
    if cell_selector == "raw-votes" and raw_available:
        maxima = np.column_stack([raw_rows[index] for index, _ in coarse])
        top = np.argsort(-maxima, axis=1, kind="stable")[:, :min(3, len(coarse))]
        counts = np.bincount(top.ravel(), minlength=len(coarse)).tolist()
        coarse_selection = {"selector": "raw-votes", "region_frame_counts": counts,
            "top_cell_indices_per_query": [[coarse[position][0] for position in row] for row in top]}
        for position, (index, _track) in enumerate(coarse):
            if counts[position]:
                support.append({"cell_index": index, "supported_frames": counts[position]})
        support.sort(key=lambda s: (-s["supported_frames"], s["cell_index"]))
    elif cell_selector == "decoded":
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
    candidates = baseline + coarse_candidates + [track for _, track in fine] if raw_available else baseline
    if not raw_available:
        labels = labels[:len(baseline)]
    merged, final_selection = merge_region_tracks(candidates, times)
    selected_sources = [labels[i] for i in final_selection.get("selected_regions", [])]
    selected_city_frames = sum(s["phase"] != "baseline" for s in selected_sources)
    if baseline_track is not None and not selected_city_frames:
        merged = baseline_track           # Preserve the exact ordinary track.
    diagnostics = {
        "experimental": True, "confidence_calibrated": False,
        "status": "ok" if raw_available and coarse and merged is not None else "unavailable", "n_cells": len(regions),
        "cell_selector": cell_selector,
        "coarse_cap": coarse_cap, "fine_cap": fine_cap, "fine_region_limit": fine_regions,
        "reference_budget": {"coarse_max": len(regions) * coarse_cap,
                             "fine_max": min(len(regions), fine_regions) * fine_cap,
                             "fine_requested": len(selected_cells) * fine_cap},
        "n_usable_coarse": len(coarse), "n_usable_fine": len(fine),
        "coarse_regions": coarse_records, "fine_regions": fine_records,
        "coarse_selection": coarse_selection, "visually_supported_cells": support,
        "coarse_ranking_pool": ("actual raw query-wise top3 coarse-cell maxima votes, cell-order ties"
                                if cell_selector == "raw-votes" else
                                "equal-budget city cells only; ordinary baseline retained for final comparison"),
        "comparison_calibration": {"calibrated": False,
                                   "different_reference_pool_sizes": True,
                                   "raw_cosine_across_pools_is_experimental": True},
        "densified_cell_indices": selected_cells,
        "final_selection": {**final_selection, "selected_sources": selected_sources},
        "selected_city_frames": selected_city_frames,
    }
    if cell_selector == "raw-votes":
        diagnostics["raw_votes"] = {"status": "complete" if raw_available else "unavailable",
            "top_cells_per_query": 3, "unavailable_coarse_cell_indices": raw_errors,
            "reason": None if raw_available else "missing_or_incomparable_actual_coarse_raw_evidence",
            "appearance_identity": appearance}
    return merged, diagnostics
