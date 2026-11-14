"""Experimental POI localization over already acquired inputs.

Only the injected retrieval callback acquires evidence. No maps/OCR/files/GT
are opened here, and diagnostic alignment never replaces a pipeline answer.
Every literal-clue branch receives the same explicit neighborhood budget.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from . import poi_alignment as core
from .poi_clues import build_poi_hypotheses

ALGORITHM_VERSION = "poi-neighborhood-localization-v1"


@dataclass(frozen=True)
class POILocalizationConfig:
    """Experimental retrieval budgets; alignment weights remain fixed."""

    radius_m: float
    cap_per_branch: int
    n_query: int = 80
    max_evidence_values: int = 4_000_000
    max_road_grid_cells: int = 4_000_000

    def __post_init__(self):
        if isinstance(self.radius_m, bool) or not np.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("neighborhood radius must be finite and positive")
        object.__setattr__(self, "radius_m", float(self.radius_m))
        for name in ("cap_per_branch", "n_query", "max_evidence_values", "max_road_grid_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if not 2 <= self.n_query <= core.POIAlignmentConfig().max_query_samples:
            raise ValueError("query budget must be between two and the fixed core limit")


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _array_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _finite_xy(value, name):
    xy = np.asarray(value, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError(f"{name} must be nonempty finite metric xy")
    return xy


def _region_plans(clues, project, inverse, config):
    plans = []
    for index, branch in enumerate(clues["candidate_clusters"]):
        positions = np.array([project.transform(p["lon"], p["lat"]) for p in branch["pois"]])
        lon, lat = inverse.transform(*positions.mean(axis=0))
        plans.append({"hypothesis_id": str(branch.get("id", f"hypothesis_{index}")),
                      "origin": "literal_poi_branch_centroid", "center": [float(lat), float(lon)],
                      "poi_ids": list(branch["poi_ids"]), "radius_m": float(config.radius_m),
                      "cap": int(config.cap_per_branch)})
    return plans


def _evidence(raw, timestamps, cap, n_query):
    if not isinstance(raw, dict) or any(key not in raw for key in (
            "query_indices", "query_times", "similarities", "ref_latlon", "ref_ids")):
        raise ValueError(raw.get("reason", "retrieval evidence is unavailable")
                         if isinstance(raw, dict) else "retrieval evidence is unavailable")
    indices = np.asarray(raw["query_indices"])
    times = np.asarray(raw["query_times"], float)
    refs = np.asarray(raw["ref_latlon"], float)
    ids = [str(value) for value in raw["ref_ids"]]
    scores = np.asarray(raw["similarities"])
    if (indices.ndim != 1 or not 2 <= len(indices) <= n_query
            or not np.issubdtype(indices.dtype, np.integer) or np.any(np.diff(indices) <= 0)
            or indices[0] < 0 or indices[-1] >= len(timestamps)
            or times.shape != indices.shape or not np.array_equal(times, timestamps[indices])):
        raise ValueError("retrieval query indices/times do not identify the supplied video grid")
    if (not 1 <= len(ids) <= cap or len(set(ids)) != len(ids) or any(not v for v in ids)
            or refs.shape != (len(ids), 2) or not np.isfinite(refs).all()
            or np.any(np.abs(refs[:, 0]) > 90) or np.any(np.abs(refs[:, 1]) > 180)
            or scores.shape != (len(times), len(ids)) or scores.dtype.kind not in "fiu"
            or not np.isfinite(scores).all()):
        raise ValueError("retrieval reference identities, finite matrix and cap must agree")
    provenance = deepcopy(raw.get("provenance") or {})
    backbone, preprocessing = provenance.get("backbone"), provenance.get("preprocessing")
    if (not isinstance(backbone, dict) or not backbone.get("name") or not backbone.get("hub")
            or not isinstance(preprocessing, dict) or not preprocessing
            or not isinstance(provenance.get("query_mode"), str) or not provenance["query_mode"]):
        raise ValueError("retrieval backbone and preprocessing identities are required")
    identity = _json_sha({"backbone": backbone, "preprocessing": preprocessing,
                          "query_mode": provenance["query_mode"]})
    return indices, times, refs, ids, np.ascontiguousarray(scores, dtype=np.float32), provenance, identity


def _append_pool(pool, evidence):
    indices, times, refs, ids, scores, provenance, identity = evidence
    if pool["identity"] is not None and (identity != pool["identity"]
            or not np.array_equal(indices, pool["indices"]) or not np.array_equal(times, pool["times"])):
        raise ValueError("neighborhood query grid/backbone/preprocessing differs")
    # Check every duplicate before mutating the union. A conflicted region
    # cannot contribute a silently selected subset of its references.
    for column, identifier in enumerate(ids):
        if identifier in pool["columns"]:
            old = pool["columns"][identifier]
            if (not np.array_equal(refs[column], old[0])
                    or not np.allclose(scores[:, column], old[1], rtol=0, atol=1e-6)):
                raise ValueError("duplicate reference ID has conflicting coordinates or similarities")
    added = 0
    for column, identifier in enumerate(ids):
        if identifier not in pool["columns"]:
            pool["columns"][identifier] = refs[column].copy(), scores[:, column].copy()
            added += 1
    pool.update(identity=identity, indices=indices.copy(), times=times.copy())
    return added


def _map_fields(roads, refs, duration, query_count, config, limits, supplied_starts):
    """Fixed pilot all-city raster/road-start policy, with preflight bounds."""
    from scipy.ndimage import distance_transform_edt

    margin = duration * config.average_speed_range_mps[1] + 3 * config.appearance_sigma_m
    low = np.floor((np.minimum(roads.min(axis=0), refs.min(axis=0)) - margin)
                   / config.field_cell_m) * config.field_cell_m
    high = np.ceil((np.maximum(roads.max(axis=0), refs.max(axis=0)) + margin)
                   / config.field_cell_m) * config.field_cell_m
    span = high - low
    if not np.isfinite(span).all():
        raise ValueError("map extent is not finite")
    shape = tuple(math.ceil(float(v) / config.field_cell_m) + 1 for v in span[::-1])
    road_shape = tuple(math.ceil(float(v) / config.road_cell_m) + 1 for v in span[::-1])
    if math.prod(shape) * query_count > core.MAX_FIELD_VALUES:
        raise ValueError("appearance raster budget exceeded before allocation")
    if math.prod(road_shape) > limits.max_road_grid_cells:
        raise ValueError("road raster budget exceeded before allocation")
    mask = np.zeros(road_shape, dtype=bool)
    pixels = np.rint((roads - low) / config.road_cell_m).astype(int)
    mask[pixels[:, 1], pixels[:, 0]] = True
    distance = (distance_transform_edt(~mask) * config.road_cell_m).astype(np.float32)
    if supplied_starts is not None:
        starts = _finite_xy(supplied_starts, "full-city starts")
        uncapped = len(starts)
        if uncapped > config.max_starts:
            raise ValueError("provided full-city start budget exceeded")
        start_policy = "caller-supplied full-city starts"
    else:
        spacing = config.start_spacing_m
        bins = np.floor((roads - low) / spacing).astype(int)
        width = math.ceil(float(span[0]) / spacing) + 1
        flat = bins[:, 1] * width + bins[:, 0]
        centers = low + (bins + .5) * spacing
        offsets = np.sum((roads - centers) ** 2, axis=1)
        order = np.lexsort((np.arange(len(flat)), offsets, flat))
        first = np.r_[True, flat[order[1:]] != flat[order[:-1]]]
        starts = roads[order[first]]
        uncapped = len(starts)
        if uncapped > config.max_starts:
            starts = starts[np.linspace(0, uncapped - 1, config.max_starts).astype(int)]
        start_policy = "nearest road sample per fixed 100m bin; uniform cap over all-city bins"
    return low, shape, distance, starts, {
        "origin_xy": low.tolist(), "appearance_shape": list(shape), "road_shape": list(road_shape),
        "start_policy": start_policy, "uncapped_start_count": uncapped, "start_count": len(starts),
        "start_samples_sha256": _array_sha(starts)}


def localize_with_poi(frames, vo_xy, timestamps, intrinsics, city, road_xy, map_crs,
                      detections, inventory, retrieve, *, config, query_cache=None,
                      start_samples=None, map_provenance=None):
    """Return diagnostic alignment; leave every pipeline answer untouched.

    ``retrieve(frames, timestamps, center, radius_m=..., cap=..., n_query=...,
    query_cache=...)`` returns the public ``kartaview_vpr_evidence`` schema.
    A caller closure supplies source/cache/query mode. IDs globally identify
    images (source-prefixed for multiple providers). Conflicting duplicates
    invalidate their region. A failed region keeps the result inactive, even
    when partial evidence permits diagnostic alignment. Metric arrays share
    ``map_crs``. Frames and K/bboxes must use the same original pixel grid.
    """
    result = {"experimental": True, "confidence_calibrated": False, "status": "inactive",
              "selected_answer": None, "diagnostic_position": None, "regions": [],
              "region_plans": [], "failed_regions": [],
              "provenance": {"algorithm": ALGORITHM_VERSION,
                "identity_policy": "compare declared backbone/preprocessing; preserve cache identity uncertainty"}}
    try:
        from pyproj import CRS, Transformer

        if not isinstance(config, POILocalizationConfig):
            raise ValueError("explicit POILocalizationConfig is required")
        fixed = core.POIAlignmentConfig()
        result["provenance"].update(config=asdict(config), fixed_alignment_config=fixed.to_dict())
        road = _finite_xy(road_xy, "full-city road samples")
        vo = _finite_xy(vo_xy, "VO positions")
        times = np.asarray(timestamps, float)
        if (times.shape != (len(vo),) or len(times) < 2 or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or len(frames) != len(times)):
            raise ValueError("frames, increasing VO timestamps and positions must agree")
        K = np.asarray(intrinsics, float)
        if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.allclose(K[2], [0, 0, 1]) or np.linalg.norm(np.diff(vo, axis=0), axis=1).sum() <= 1e-8):
            raise ValueError("finite pinhole intrinsics and a nonstationary VO shape are required")
        detections = list(detections)
        if any(not times[0] <= float(d["t_sec"]) <= times[-1] for d in detections):
            raise ValueError("OCR observations must lie within the supplied video timeline")
        crs = CRS.from_user_input(map_crs)
        if not crs.is_projected or not np.isclose(crs.axis_info[0].unit_conversion_factor, 1):
            raise ValueError("map CRS must be projected in metres")
        if not isinstance(city, str) or not city.strip() or inventory.get("city", city) != city:
            raise ValueError("inventory must belong to the supplied named city")
        source = deepcopy(map_provenance or {})
        if source.get("city", city) != city or source.get("scope", "whole_city") != "whole_city":
            raise ValueError("road samples must come from the declared whole-city map")
        _json_sha(source)
        clues = build_poi_hypotheses(detections, inventory)
        result["clues"] = clues
        branches = clues["candidate_clusters"]
        if not branches:
            result["reason"] = "no_literal_poi_conjunctions"
            return result
        if len(branches) > fixed.max_hypotheses:
            raise ValueError("clue hypothesis budget exceeded; no branches were dropped")
        fine_bound = (len(branches) * fixed.basins_per_hypothesis * len(fixed.refine_xy_offsets_m) ** 2
                      * len(fixed.refine_yaw_offsets_deg) * len(fixed.refine_length_multipliers))
        if fine_bound > fixed.max_fine_poses:
            raise ValueError("fixed core refinement budget exceeded before requesting evidence")
        if len(branches) * config.cap_per_branch * config.n_query > config.max_evidence_values:
            raise ValueError("retrieval matrix budget exceeded before requesting evidence")
        project = Transformer.from_crs(4326, crs, always_xy=True)
        inverse = Transformer.from_crs(crs, 4326, always_xy=True)
        plans = _region_plans(clues, project, inverse, config)
        result["region_plans"] = deepcopy(plans)  # All plans precede the first callback.
        result["provenance"].update(config=asdict(config), fixed_alignment_config=fixed.to_dict(),
            city=city, map_crs=crs.to_string(), road_samples_sha256=_array_sha(road), map_sources=source,
            region_plans_sha256=_json_sha(plans), query_cache_shared=True,
            query_cache_provided=query_cache is not None)
        cache = {} if query_cache is None else query_cache
        if not isinstance(cache, dict):
            raise ValueError("query_cache must be one explicit mutable dictionary")
        pool = {"identity": None, "indices": None, "times": None, "columns": {}}
        for plan in plans:
            record = {**deepcopy(plan), "status": "failed"}
            try:
                raw = retrieve(frames, times, tuple(plan["center"]), radius_m=plan["radius_m"],
                               cap=plan["cap"], n_query=config.n_query, query_cache=cache)
                record["provenance"] = deepcopy(raw.get("provenance") or {}) if isinstance(raw, dict) else {}
                evidence = _evidence(raw, times, config.cap_per_branch, config.n_query)
                record.update(reference_count=len(evidence[3]), query_count=len(evidence[1]),
                              ref_ids_sha256=_json_sha(evidence[3]),
                              similarities_sha256=_array_sha(evidence[4]), identity_sha256=evidence[6])
                added = _append_pool(pool, evidence)
                record.update(status="ok", new_unique_reference_count=added,
                              duplicate_reference_count=len(evidence[3]) - added)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            result["regions"].append(record)
        result["failed_regions"] = [r["hypothesis_id"] for r in result["regions"] if r["status"] != "ok"]
        if not pool["columns"]:
            result["reason"] = "no_compatible_reference_evidence"
            return result
        return _align_result(result, pool, road, vo, times, intrinsics, clues, crs, project,
                             fixed, config, source, start_samples)
    except Exception as exc:
        result.update(status="inactive", reason="poi_localization_failed", error=f"{type(exc).__name__}: {exc}")
    return result


def _align_result(result, pool, road, vo, times, intrinsics, clues, crs, project,
                  fixed, config, source, start_samples):
    ids = list(pool["columns"])
    refs = np.array([pool["columns"][identifier][0] for identifier in ids])
    similarities = np.ascontiguousarray(np.column_stack([
        pool["columns"][identifier][1] for identifier in ids]), dtype=np.float32)
    ref_xy = np.column_stack(project.transform(refs[:, 1], refs[:, 0]))
    result["reference_union"] = {
        "ids": ids, "count": len(ids), "ids_sha256": _json_sha(ids),
        "ref_latlon_sha256": _array_sha(refs), "similarities_sha256": _array_sha(similarities),
        "query_indices": pool["indices"].tolist(), "query_times": pool["times"].tolist(),
        "identity_sha256": pool["identity"]}
    low, shape, distance, starts, grid = _map_fields(
        road, ref_xy, float(times[-1] - times[0]), len(pool["times"]), fixed, config, start_samples)
    result["grid"] = grid
    appearance = core.build_appearance_field(ref_xy, similarities, pool["times"],
        origin_xy=low, shape=shape, reference_mean=False, config=fixed,
        provenance={"region_plans_sha256": result["provenance"]["region_plans_sha256"],
                    "reference_union": result["reference_union"]})
    road_field = core.RasterField(distance, tuple(low), fixed.road_cell_m, 200.,
                                  provenance={"road_samples_sha256": _array_sha(road), "sources": source})
    aligned = core.align_candidate_clusters(vo, times, intrinsics, clues, map_crs=crs,
        appearance_field=appearance, road_distance_field=road_field, start_samples=starts,
        query_times=pool["times"], config=fixed, provenance=result["provenance"])
    result["alignment"] = aligned
    result["diagnostic_position"] = {
        "source": "experimental_poi_alignment", "hypothesis_id": aligned["selected"]["hypothesis_id"],
        "route_latlon": aligned["selected"]["route_latlon"],
        "route_timestamps": aligned["selected"]["route_timestamps"]}
    result.update(status="inactive" if result["failed_regions"] else "ok",
                  reason="incomplete_retrieval" if result["failed_regions"] else "diagnostic_alignment_only")
    return result
