"""Bounded input-only alignment of motion, timed POIs and map evidence.

Callers match literal OCR to an OSM inventory and supply every branch.
No fetching, model inference, file access or evaluation occurs here. Map
coordinates/rasters share one projected metric frame. Scores are evidence,
not probabilities of correctness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from typing import Mapping
import numpy as np

ALGORITHM_VERSION = "timed-poi-similarity-v1"
MAX_FIELD_VALUES = 16000000

@dataclass(frozen=True)
class POIAlignmentConfig:
    field_cell_m: float = 100.0
    appearance_sigma_m: float = 150.0
    row_std_floor: float = 0.005
    z_clip: float = 6.0
    empty_appearance: float = -6.0
    road_cell_m: float = 25.0
    road_sigma_m: float = 50.0
    road_weight: float = 0.25
    road_distance_cap_sigmas: float = 4.0
    average_speed_range_mps: tuple = (2.0, 25.0)
    length_count: int = 9
    duration_prior_speed_mps: float = 5.5
    length_log_sigma: float = 0.7
    length_prior_weight: float = 0.1
    start_spacing_m: float = 100.0
    max_starts: int = 15000
    yaw_step_deg: float = 15.0
    basin_position_bin_m: float = 500.0
    basin_yaw_bin_deg: float = 30.0
    basin_log_length_bin: float = 0.35
    refine_xy_offsets_m: tuple = (-150, -100, -50, 0, 50, 100, 150)
    refine_yaw_offsets_deg: tuple = (-10, -5, 0, 5, 10)
    refine_length_multipliers: tuple = (0.8, 0.9, 1.0, 1.1, 1.2)
    transform_batch: int = 256
    visibility_m: float = 100.0
    distance_sigma_m: float = 100.0
    distance_penalty_cap: float = 8.0
    bearing_free_deg: float = 20.0
    bearing_sigma_deg: float = 45.0
    bearing_penalty_cap: float = 4.0
    heading_window_s: float = 1.5
    poi_weight: float = 1.0
    basins_per_hypothesis: int = 16
    seed_top_scores: int = 16384
    missing_observation_penalty: float = 8.0
    max_query_samples: int = 80
    max_coarse_poses: int = 4000000
    max_total_coarse_scores: int = 20000000
    max_fine_poses: int = 250000
    max_hypotheses: int = 64
    max_observations: int = 64
    max_choices_per_observation: int = 32
    retained_candidates: int = 256
    anchor_search_radius_m: float = 200.0
    max_start_proposals: int = 250000

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        for definition in fields(self):
            key, value = definition.name, getattr(self, definition.name)
            values = np.asarray(value)
            if not np.isfinite(values).all():
                raise ValueError(f"{key} must be finite")
            if isinstance(definition.default, int):
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
                object.__setattr__(self, key, int(value))
            elif isinstance(definition.default, tuple):
                if values.ndim != 1:
                    raise ValueError(f"{key} must be a one-dimensional grid")
                object.__setattr__(self, key, tuple(float(v) for v in values))
            else:
                if value < 0 and key != "empty_appearance":
                    raise ValueError(f"{key} must be nonnegative")
                object.__setattr__(self, key, float(value))
        positive=("distance_sigma_m","bearing_sigma_deg","heading_window_s","road_sigma_m","length_log_sigma","yaw_step_deg","basin_position_bin_m","basin_yaw_bin_deg","basin_log_length_bin","start_spacing_m","duration_prior_speed_mps","field_cell_m","road_cell_m","appearance_sigma_m","row_std_floor","z_clip")
        if any(getattr(self,k)<=0 for k in positive) or self.yaw_step_deg>360:
            raise ValueError("metric scales and angular steps must be positive")
        if len(self.average_speed_range_mps)!=2 or self.average_speed_range_mps[0]<=0 or self.average_speed_range_mps[1]<self.average_speed_range_mps[0]:
            raise ValueError("speed range must be positive and ordered")
        for name in ("refine_xy_offsets_m","refine_yaw_offsets_deg","refine_length_multipliers"):
            if not len(getattr(self,name)):
                raise ValueError("refinement grids must be nonempty")
        if min(self.refine_length_multipliers)<=0 or 1 not in self.refine_length_multipliers:
            raise ValueError("refinement factors must be positive and retain the coarse scale")
        if 0 not in self.refine_xy_offsets_m or 0 not in self.refine_yaw_offsets_deg:
            raise ValueError("refinement offsets must retain the coarse pose")
        if self.max_query_samples<2 or self.seed_top_scores<self.basins_per_hypothesis:
            raise ValueError("query/seed budgets must support the search")


def _digest(array):
    array=np.ascontiguousarray(array)
    h=hashlib.sha256(str((array.shape,array.dtype.str)).encode("ascii"))
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def _json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()


def _finite(value,shape=None,name="array",dtype=float):
    result=np.asarray(value,dtype=dtype)
    if result.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    if (shape is not None and result.shape!=shape) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have the required shape and finite values")
    return result

def transform_path(parameters, unit_shape):
    angle = np.radians(parameters[:, 2])[:, None]
    cos, sin = np.cos(angle), np.sin(angle)
    x = cos * unit_shape[None, :, 0] - sin * unit_shape[None, :, 1]
    y = sin * unit_shape[None, :, 0] + cos * unit_shape[None, :, 1]
    return np.stack([x, y], axis=2) * parameters[:, None, 3:4] + parameters[:, None, :2]

def _sample_grid(grid, xy, low, cell, outside):
    """Bilinear samples for [candidate,query,xy]; temporal rows never mix."""
    col, row = (xy[..., 0] - low[0]) / cell, (xy[..., 1] - low[1]) / cell
    height, width = grid.shape[-2:]
    inside = (col >= 0) & (row >= 0) & (col <= width - 1) & (row <= height - 1)
    x = np.floor(np.clip(col, 0, width - 2)).astype(int)
    y = np.floor(np.clip(row, 0, height - 2)).astype(int)
    fx, fy = np.clip(col - x, 0, 1), np.clip(row - y, 0, 1)
    if grid.ndim == 3:
        q = np.arange(xy.shape[1])[None, :]
        a, b, c, d = grid[q, y, x], grid[q, y, x + 1], grid[q, y + 1, x], grid[q, y + 1, x + 1]
    else:
        a, b, c, d = grid[y, x], grid[y, x + 1], grid[y + 1, x], grid[y + 1, x + 1]
    value = (1 - fy) * ((1 - fx) * a + fx * b) + fy * ((1 - fx) * c + fx * d)
    return np.where(inside, value, outside)

def _appearance_values(scores, reference_xy, low, shape, config):
    settings = config.to_dict()
    from scipy.ndimage import gaussian_filter

    cell = settings["field_cell_m"]
    indices = np.rint((reference_xy - low) / cell).astype(int)
    counts = np.zeros(shape, dtype=np.float32)
    np.add.at(counts, (indices[:, 1], indices[:, 0]), 1)
    sigma = settings["appearance_sigma_m"] / cell
    density = gaussian_filter(counts, sigma=sigma, mode="constant", truncate=3)
    supported = density > 1e-6
    z = (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(
        scores.std(axis=1, keepdims=True), settings["row_std_floor"])
    z = np.clip(z, -settings["z_clip"], settings["z_clip"])
    field = np.full((len(scores), *shape), settings["empty_appearance"], dtype=np.float32)
    for q in range(len(scores)):
        weight = np.zeros(shape, dtype=np.float32)
        np.add.at(weight, (indices[:, 1], indices[:, 0]), np.exp(z[q]))
        weight = gaussian_filter(weight, sigma=sigma, mode="constant", truncate=3)
        field[q, supported] = np.log(np.maximum(weight[supported] / density[supported], 1e-8))
    return field

def _poi_loss_modes(parameters, geometry, allowed, config):
    """Return [visibility mode,hypothesis,candidate] soft losses, without labels."""
    settings = config.to_dict()
    camera = transform_path(parameters, geometry["unit_positions"])
    yaw = np.radians(parameters[:, 2])
    losses = np.zeros((2, len(allowed), len(parameters)), dtype=float)
    for hypothesis, choices in enumerate(allowed):
        for obs, pois in enumerate(choices):
            if not len(pois):
                losses[:, hypothesis] += geometry["weights"][obs] * settings["missing_observation_penalty"]
                continue
            delta = pois[None, :, :] - camera[:, obs, None, :]
            distance = np.linalg.norm(delta, axis=2)
            excess = np.maximum(distance - settings["visibility_m"], 0)
            radial = np.minimum(0.5 * (excess / settings["distance_sigma_m"]) ** 2,
                                settings["distance_penalty_cap"])
            angular = np.zeros_like(radial)
            if geometry["bearing_available"][obs]:
                # In the VO x/z plane, image-right is clockwise from camera
                # forward, so a positive image bearing subtracts from XY yaw.
                expected = yaw + geometry["headings"][obs] - geometry["rays"][obs]
                bearing = np.arctan2(delta[:, :, 1], delta[:, :, 0])
                error = np.abs((bearing - expected[:, None] + np.pi) % (2 * np.pi) - np.pi)
                excess_angle = np.maximum(error - np.radians(settings["bearing_free_deg"]), 0)
                angular = np.minimum(0.5 * (excess_angle / np.radians(settings["bearing_sigma_deg"])) ** 2,
                                     settings["bearing_penalty_cap"])
            losses[0, hypothesis] += geometry["weights"][obs] * radial.min(axis=1)
            losses[1, hypothesis] += geometry["weights"][obs] * (radial + angular).min(axis=1)
    return losses

def _seeds_for_hypothesis(scores, starts, yaws, lengths, config):
    settings = config.to_dict()
    flat = scores.ravel()
    keep = min(settings["seed_top_scores"], len(flat))
    selected = np.argpartition(-flat, keep - 1)[:keep]
    selected = selected[np.argsort(-flat[selected], kind="stable")]
    seeds, seen = [], set()
    for index in selected:
        i, j, k = np.unravel_index(index, scores.shape)
        p = np.array([*starts[k], yaws[j], lengths[i]])
        key = (round(p[0] / settings["basin_position_bin_m"]),
               round(p[1] / settings["basin_position_bin_m"]),
               round(p[2] / settings["basin_yaw_bin_deg"]),
               round(np.log(p[3]) / settings["basin_log_length_bin"]))
        if key not in seen:
            seen.add(key)
            seeds.append(p)
            if len(seeds) == settings["basins_per_hypothesis"]:
                break
    return seeds

def _refinement_parameters(seeds_by_hypothesis, lengths, config):
    settings = config.to_dict()
    parameters, owners = [], []
    for hypothesis, seeds in enumerate(seeds_by_hypothesis):
        for seed in seeds:
            for factor in settings["refine_length_multipliers"]:
                length = seed[3] * factor
                if not lengths[0] <= length <= lengths[-1]:
                    continue
                for da in settings["refine_yaw_offsets_deg"]:
                    for dx in settings["refine_xy_offsets_m"]:
                        for dy in settings["refine_xy_offsets_m"]:
                            parameters.append([seed[0] + dx, seed[1] + dy, (seed[2] + da) % 360, length])
                            owners.append(hypothesis)
    return np.asarray(parameters), np.asarray(owners)


@dataclass(frozen=True)
class RasterField:
    """Metric raster with optional exact query-time identity and content provenance."""
    values: np.ndarray
    origin_xy: tuple
    cell_m: float
    outside_value: float
    query_times: np.ndarray | None = None
    provenance: Mapping | None = None

    def __post_init__(self):
        values = np.asarray(self.values)
        if values.ndim not in (2, 3) or min(values.shape[-2:]) < 2 or values.size > MAX_FIELD_VALUES:
            raise ValueError("field must be a bounded two/three-dimensional raster")
        if not np.isfinite(values).all() or not np.isfinite(self.outside_value):
            raise ValueError("field and outside values must be finite")
        _finite(self.origin_xy, (2,), "field origin")
        if not np.isfinite(self.cell_m) or self.cell_m <= 0:
            raise ValueError("field cell size must be finite and positive")
        object.__setattr__(self, "values", values)
        if values.ndim == 3:
            times = _finite(self.query_times, (len(values),), "field query times")
            if not len(times) or np.any(np.diff(times) <= 0):
                raise ValueError("field query times must increase strictly")
            object.__setattr__(self, "query_times", times)
        elif self.query_times is not None:
            raise ValueError("static fields must not have query times")

    def sample(self, positions, times):
        values = self.values
        if values.ndim == 3:
            indices = np.searchsorted(self.query_times, times)
            if (np.any(indices >= len(self.query_times))
                    or not np.allclose(self.query_times[np.minimum(indices, len(self.query_times) - 1)],
                                       times, rtol=0, atol=1e-8)):
                raise ValueError("appearance rows must match the requested query times")
            if len(indices) != len(values) or not np.array_equal(indices, np.arange(len(values))):
                values = values[indices]
        return _sample_grid(values, positions, np.asarray(self.origin_xy), self.cell_m, self.outside_value)

    def source_record(self):
        return {"shape": list(self.values.shape), "values_sha256": _digest(self.values),
                "origin_xy": list(self.origin_xy), "cell_m": float(self.cell_m),
                "outside_value": float(self.outside_value),
                "query_times_sha256": _digest(self.query_times) if self.query_times is not None else None,
                "sources": dict(self.provenance or {})}


def build_appearance_field(reference_xy, similarities, query_times, *, origin_xy, shape,
                           reference_mean=False, config=None, provenance=None):
    """Use every supplied reference; never substitute a denser reference pool."""
    config = config or POIAlignmentConfig()
    refs = _finite(reference_xy, name="references")
    scores = _finite(similarities, name="similarities", dtype=None)
    if scores.dtype.kind != "f" or scores.dtype.itemsize > 8:
        scores = scores.astype(float)
    times = _finite(query_times, name="query times")
    if (refs.ndim != 2 or refs.shape[1] != 2 or not len(refs) or times.ndim != 1 or not len(times)
            or scores.shape != (len(times), len(refs))):
        raise ValueError("query/reference identity and matrix shapes must agree")
    if (len(shape) != 2 or any(int(v) != v or v < 2 for v in shape)
            or len(times) * np.prod(shape) > MAX_FIELD_VALUES):
        raise ValueError("appearance field exceeds its bounded grid or has invalid dimensions")
    origin = _finite(origin_xy, (2,), "field origin")
    shape = tuple(int(v) for v in shape)
    indices = np.rint((refs - origin) / config.field_cell_m).astype(int)
    if np.any(indices < 0) or np.any(indices[:, 0] >= shape[1]) or np.any(indices[:, 1] >= shape[0]):
        raise ValueError("appearance bounds must include every supplied reference")
    evidence = scores - scores.mean(axis=0, keepdims=True) if reference_mean else scores
    values = _appearance_values(evidence, refs, origin, shape, config)
    source = {"kind": "all-reference Gaussian log-mean-exp", "reference_count": len(refs),
              "reference_xy_sha256": _digest(refs), "similarities_sha256": _digest(scores),
              "reference_mean": bool(reference_mean), "config": config.to_dict(),
              "caller": dict(provenance or {})}
    return RasterField(values, tuple(origin), config.field_cell_m, config.empty_appearance, times, source)


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _prepare(vo_xy, timestamps, intrinsics, observations, pois, hypotheses, config):
    xy, times = _finite(vo_xy, name="VO path"), _finite(timestamps, name="VO timestamps")
    if (times.ndim != 1 or len(times) < 2 or xy.shape != (len(times), 2)
            or np.any(np.diff(times) <= 0)):
        raise ValueError("VO path and strictly increasing timestamps must agree")
    K = _finite(intrinsics, (3, 3), "intrinsics")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("intrinsics must describe a finite pinhole camera")
    arc = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if arc <= 1e-8:
        raise ValueError("VO has no route shape to align")
    if not 1 <= len(observations) <= config.max_observations or not 1 <= len(hypotheses) <= config.max_hypotheses:
        raise ValueError("observation/hypothesis counts exceed configured bounds or are empty")
    obs, seen = [], set()
    for i, observation in enumerate(observations):
        item = {"id": str(_get(observation, "id", f"observation_{i}")),
                "t_sec": float(_get(observation, "t_sec")), "confidence": float(_get(observation, "confidence")),
                "text": str(_get(observation, "text", "")), "bbox": _get(observation, "bbox")}
        if (item["id"] in seen or not np.isfinite([item["t_sec"], item["confidence"]]).all()
                or not times[0] <= item["t_sec"] <= times[-1] or not 0 < item["confidence"] <= 1):
            raise ValueError("OCR IDs/times/confidence must be unique, finite and within the video")
        if item["bbox"] is not None:
            box = _finite(item["bbox"], (4,), "OCR box")
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("OCR boxes must have positive width and height")
            item["bbox"] = box.tolist()
        obs.append(item)
        seen.add(item["id"])
    poi_map = {}
    for poi in pois:
        key = str(_get(poi, "id"))
        if key in poi_map:
            raise ValueError("POI IDs must be unique")
        poi_map[key] = _finite(_get(poi, "xy"), (2,), "POI map position")
    normalized, allowed, seen = [], [], set()
    for i, hypothesis in enumerate(hypotheses):
        key = str(_get(hypothesis, "id", f"hypothesis_{i}"))
        assignments = _get(hypothesis, "assignments")
        if key in seen or not isinstance(assignments, Mapping) or set(assignments).difference(o["id"] for o in obs):
            raise ValueError("hypothesis IDs/observation assignments must be unique and known")
        clean = {}
        for o in obs:
            targets = assignments.get(o["id"], [])
            targets = [targets] if isinstance(targets, str) else list(targets)
            targets = list(dict.fromkeys(str(v) for v in targets))
            if len(targets) > config.max_choices_per_observation or set(targets).difference(poi_map):
                raise ValueError("hypothesis references unknown POIs or exceeds its choice budget")
            clean[o["id"]] = targets
        if not any(clean.values()):
            raise ValueError("each hypothesis must assign a timed observation")
        normalized.append({"id": key, "assignments": clean})
        allowed.append([np.asarray([poi_map[p] for p in clean[o["id"]]]).reshape(-1, 2) for o in obs])
        seen.add(key)
    def at(t):
        return np.c_[np.interp(t, times, xy[:, 0]), np.interp(t, times, xy[:, 1])]
    when = np.array([o["t_sec"] for o in obs])
    tangent = (at(np.minimum(when + config.heading_window_s / 2, times[-1]))
               - at(np.maximum(when - config.heading_window_s / 2, times[0])))
    rays, valid = [], []
    for i, o in enumerate(obs):
        box = o["bbox"]
        use = box is not None and np.linalg.norm(tangent[i]) > 1e-6
        rays.append(np.arctan(((box[0] + box[2]) / 2 - K[0, 2]) / K[0, 0]) if use else 0.)
        valid.append(use)
    weights = np.array([o["confidence"] for o in obs])
    geometry = {"unit_positions": (at(when) - xy[0]) / arc,
                "headings": np.arctan2(tangent[:, 1], tangent[:, 0]), "rays": np.asarray(rays),
                "bearing_available": np.asarray(valid), "weights": weights / weights.sum()}
    return {"xy": xy, "times": times, "K": K, "arc": arc, "at": at, "observations": obs,
            "pois": poi_map, "hypotheses": normalized, "allowed": allowed, "geometry": geometry}


def _base_scores(parameters, unit_shape, times, duration, appearance, road, config):
    poses = transform_path(parameters, unit_shape)
    app = appearance.sample(poses, times).mean(axis=1) if appearance is not None else np.zeros(len(parameters))
    if road is None:
        road_penalty = np.zeros(len(parameters))
    else:
        distance = road.sample(poses, times)
        road_penalty = config.road_weight * np.minimum(distance / config.road_sigma_m,
                                                      config.road_distance_cap_sigmas) ** 2
        road_penalty = road_penalty.mean(axis=1)
    length_penalty = config.length_prior_weight * (
        np.log(parameters[:, 3] / (duration * config.duration_prior_speed_mps)) / config.length_log_sigma) ** 2
    return np.c_[parameters, app - road_penalty - length_penalty, app, road_penalty, length_penalty]


def _derived_starts(data, lengths, yaws, config):
    """Timed-anchor proposals; a bounded union over every branch/yaw/scale."""
    span = (2 * config.anchor_search_radius_m + 1e-6) / config.start_spacing_m
    if not np.isfinite(span) or span > config.max_start_proposals:
        raise ValueError("timed-anchor proposal budget exceeded before grid allocation")
    offset_count = int(np.ceil(span))
    anchor_count = 0
    for choices in data["allowed"]:
        anchor = min((i for i, p in enumerate(choices) if len(p)),
                     key=lambda i: data["observations"][i]["t_sec"])
        anchor_count += len(choices[anchor])
    proposals = offset_count ** 2 * len(lengths) * len(yaws) * anchor_count
    if proposals > config.max_start_proposals:
        raise ValueError("timed-anchor proposal budget exceeded before grid allocation")
    offsets = np.arange(-config.anchor_search_radius_m, config.anchor_search_radius_m + 1e-6,
                        config.start_spacing_m)
    grid = np.array([(x, y) for x in offsets for y in offsets])
    starts, generated = {}, 0
    for length in lengths:
        for yaw in yaws:
            shifted = transform_path(np.array([[0., 0., yaw, length]]), data["geometry"]["unit_positions"])[0]
            for choices in data["allowed"]:
                anchor = min((i for i, p in enumerate(choices) if len(p)),
                             key=lambda i: data["observations"][i]["t_sec"])
                for point in choices[anchor]:
                    for offset in grid:
                        generated += 1
                        if generated > config.max_start_proposals:
                            raise ValueError("timed-anchor proposal budget exceeded; supply bounded map starts")
                        position = point - shifted[anchor] + offset
                        key = tuple(np.floor(position / config.start_spacing_m).astype(int))
                        center = (np.asarray(key) + 0.5) * config.start_spacing_m
                        distance = float(np.sum((position - center) ** 2))
                        if key not in starts or distance < starts[key][0]:
                            starts[key] = distance, position
                        if len(starts) > config.max_starts:
                            raise ValueError("timed-anchor start budget exceeded; supply bounded map starts")
    return np.asarray([starts[k][1] for k in sorted(starts)])


def _rank(rows):
    return np.lexsort((rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0], rows[:, 9], -rows[:, 10]))


def _row_key(row):
    return (-row[10], int(row[9]), *row[:4])


def _candidate(row, data, config):
    h = int(row[9])
    hypothesis = data["hypotheses"][h]
    camera = transform_path(row[None], data["geometry"]["unit_positions"])[0]
    evidence = []
    for i, observation in enumerate(data["observations"]):
        ids = hypothesis["assignments"][observation["id"]]
        item = {"observation_id": observation["id"], "t_sec": observation["t_sec"],
                "bearing_available": bool(data["geometry"]["bearing_available"][i])}
        if ids:
            delta = data["allowed"][h][i] - camera[i]
            distance = np.linalg.norm(delta, axis=1)
            radial = np.minimum(0.5 * (np.maximum(distance - config.visibility_m, 0) / config.distance_sigma_m) ** 2,
                                config.distance_penalty_cap)
            angular = np.zeros_like(radial)
            error = np.full(len(ids), np.nan)
            if item["bearing_available"]:
                expected = np.radians(row[2]) + data["geometry"]["headings"][i] - data["geometry"]["rays"][i]
                error = np.abs((np.arctan2(delta[:, 1], delta[:, 0]) - expected + np.pi) % (2 * np.pi) - np.pi)
                angular = np.minimum(0.5 * (np.maximum(error - np.radians(config.bearing_free_deg), 0)
                                           / np.radians(config.bearing_sigma_deg)) ** 2, config.bearing_penalty_cap)
            pick = int((radial + angular).argmin())
            item.update(poi_id=ids[pick], distance_m=float(distance[pick]),
                        bearing_error_deg=float(np.degrees(error[pick])) if item["bearing_available"] else None,
                        weighted_loss=float(data["geometry"]["weights"][i] * (radial[pick] + angular[pick])))
        else:
            item.update(poi_id=None, weighted_loss=float(data["geometry"]["weights"][i]
                                                       * config.missing_observation_penalty))
        evidence.append(item)
    return {"hypothesis_id": hypothesis["id"], "start_xy": row[:2].tolist(), "yaw_deg": float(row[2]),
            "length_m": float(row[3]), "objective": float(row[10]), "appearance": float(row[5]),
            "road_penalty": float(row[6]), "length_penalty": float(row[7]), "poi_loss": float(row[8]),
            "observations": evidence}


def align_poi_hypotheses(vo_xy, timestamps, intrinsics, observations, pois, hypotheses, *,
                         appearance_field=None, road_distance_field=None, start_samples=None,
                         query_times=None, config=None, provenance=None):
    """Compare all branches using timed visibility, bearing and optional map evidence.

    Observations provide id/t_sec/confidence/bbox; POIs provide id and metric xy.
    Each hypothesis provides id and assignments {observation_id: [poi_ids]}.
    Supply every alternative. Optional rasters and starts use the POI map frame.
    Without starts, bounded proposals are derived from every timed POI branch.
    Appearance is optional, but its absence is recorded: POI/road constraints
    alone may leave the correct branch unresolved.
    """
    config = config or POIAlignmentConfig()
    if not isinstance(config, POIAlignmentConfig):
        raise ValueError("config must be a POIAlignmentConfig")
    sources = dict(provenance or {})
    _json_digest(sources)
    data = _prepare(vo_xy, timestamps, intrinsics, observations, pois, hypotheses, config)
    if appearance_field is not None and not isinstance(appearance_field, RasterField):
        raise ValueError("appearance must be a bounded RasterField")
    if road_distance_field is not None and (
            not isinstance(road_distance_field, RasterField) or road_distance_field.values.ndim != 2
            or road_distance_field.values.min() < 0 or road_distance_field.outside_value < 0):
        raise ValueError("road evidence must be a nonnegative static distance field")
    if query_times is None:
        query_times = (appearance_field.query_times if appearance_field is not None and appearance_field.query_times is not None
                       else data["times"])
    query_times = _finite(query_times, name="alignment query times")
    if query_times.ndim != 1 or len(query_times) < 2 or np.any(np.diff(query_times) <= 0):
        raise ValueError("alignment query times must increase strictly")
    if query_times[0] < data["times"][0] or query_times[-1] > data["times"][-1]:
        raise ValueError("alignment query times fall outside the VO timeline")
    if len(query_times) > config.max_query_samples:
        query_times = query_times[np.linspace(0, len(query_times) - 1, config.max_query_samples).astype(int)]
    unit_query = (data["at"](query_times) - data["xy"][0]) / data["arc"]
    if appearance_field is not None:
        appearance_field.sample(np.zeros((1, len(query_times), 2)), query_times)
    duration = float(data["times"][-1] - data["times"][0])
    yaw_count = 360. / config.yaw_step_deg
    if (not np.isfinite(yaw_count) or yaw_count > config.max_coarse_poses
            or config.length_count * int(np.ceil(yaw_count)) > config.max_coarse_poses
            or config.length_count * int(np.ceil(yaw_count)) * len(data["hypotheses"])
            > config.max_total_coarse_scores):
        raise ValueError("coarse pose/hypothesis budget exceeded before grid allocation")
    lengths = np.geomspace(duration * config.average_speed_range_mps[0], duration * config.average_speed_range_mps[1],
                           config.length_count)
    yaws = np.arange(0, 360, config.yaw_step_deg)
    starts = (_derived_starts(data, lengths, yaws, config) if start_samples is None
              else _finite(start_samples, name="map starts"))
    if starts.ndim != 2 or starts.shape[1] != 2 or not 1 <= len(starts) <= config.max_starts:
        raise ValueError("map starts must be nonempty metric xy within their configured budget")
    count = len(lengths) * len(yaws) * len(starts)
    hypotheses_count = len(data["hypotheses"])
    if count > config.max_coarse_poses or count * hypotheses_count > config.max_total_coarse_scores:
        raise ValueError("coarse pose/hypothesis budget exceeded")
    fine_bound = (hypotheses_count * config.basins_per_hypothesis * len(config.refine_xy_offsets_m) ** 2
                  * len(config.refine_yaw_offsets_deg) * len(config.refine_length_multipliers))
    if fine_bound > config.max_fine_poses:
        raise ValueError("fine pose budget exceeded; reduce the equal per-hypothesis refinement budget")
    scores = np.empty((hypotheses_count, len(lengths), len(yaws), len(starts)), dtype=np.float32)
    batch = config.transform_batch
    for i, length in enumerate(lengths):
        for j, yaw in enumerate(yaws):
            for k in range(0, len(starts), batch):
                positions = starts[k:k + batch]
                parameters = np.c_[positions, np.full(len(positions), yaw), np.full(len(positions), length)]
                base = _base_scores(parameters, unit_query, query_times, duration, appearance_field, road_distance_field, config)
                loss = _poi_loss_modes(parameters, data["geometry"], data["allowed"], config)[1]
                scores[:, i, j, k:k + len(positions)] = base[None, :, 4] - config.poi_weight * loss
    seeds = [_seeds_for_hypothesis(s, starts, yaws, lengths, config) for s in scores]
    parameters, owners = _refinement_parameters(seeds, lengths, config)
    fine_parts, best_by_hypothesis = [], [None] * hypotheses_count
    for k in range(0, len(parameters), batch):
        p = parameters[k:k + batch]
        base = _base_scores(p, unit_query, query_times, duration, appearance_field, road_distance_field, config)
        loss = _poi_loss_modes(p, data["geometry"], data["allowed"], config)[1]
        total = base[None, :, 4] - config.poi_weight * loss
        chosen = total.argmax(axis=0)
        q = np.arange(len(p))
        fine_parts.append(np.c_[base, loss[chosen, q], chosen, total[chosen, q], owners[k:k + len(p)]])
        for h in range(hypotheses_count):
            pick = int(np.lexsort((p[:, 3], p[:, 2], p[:, 1], p[:, 0], -total[h]))[0])
            row = np.r_[base[pick], loss[h, pick], h, total[h, pick], owners[k + pick]]
            if best_by_hypothesis[h] is None or _row_key(row) < _row_key(best_by_hypothesis[h]):
                best_by_hypothesis[h] = row
    fine = np.concatenate(fine_parts)
    order = _rank(fine)
    selected_row = fine[order[0]]
    selected = _candidate(selected_row, data, config)
    unit_full = (data["xy"] - data["xy"][0]) / data["arc"]
    selected["route_xy"] = transform_path(selected_row[None], unit_full)[0].tolist()
    selected["route_timestamps"] = data["times"].tolist()
    branches = [{**hypothesis, "best": _candidate(best_by_hypothesis[h], data, config),
                 "refinement_basins": len(seeds[h])} for h, hypothesis in enumerate(data["hypotheses"])]
    ranked_hypotheses = [branch for _, branch in sorted(
        enumerate(branches), key=lambda item: (-item[1]["best"]["objective"], item[0]))]
    retained, seen = [], set()
    for index in order:
        row = fine[index]
        key = (*np.round(row[:4], 6), int(row[9]))
        if key not in seen:
            seen.add(key)
            retained.append(_candidate(row, data, config))
            if len(retained) == config.retained_candidates:
                break
    return {"experimental": True, "confidence_calibrated": False, "selected": selected,
            "hypotheses": branches, "ranked_hypotheses": ranked_hypotheses, "ranked_candidates": retained,
            "hypothesis_objective_margin": (ranked_hypotheses[0]["best"]["objective"] - ranked_hypotheses[1]["best"]["objective"]
                                            if len(branches) > 1 else None),
            "search": {"coarse_pose_count": count, "coarse_hypothesis_scores": count * hypotheses_count,
                       "fine_pose_count": len(parameters), "hypothesis_count": hypotheses_count,
                       "query_count": len(query_times), "retained_candidates": len(retained),
                       "start_mode": "provided" if start_samples is not None else "timed_poi_grid"},
            "provenance": {"algorithm": ALGORITHM_VERSION, "implementation_module": __name__,
                           "config": config.to_dict(), "numpy": np.__version__, "caller_sources": sources,
                           "vo_sha256": _digest(data["xy"]), "times_sha256": _digest(data["times"]),
                           "intrinsics_sha256": _digest(data["K"]), "observations_sha256": _json_digest(data["observations"]),
                           "pois_sha256": _json_digest([(k, v.tolist()) for k, v in data["pois"].items()]),
                           "hypotheses_sha256": _json_digest(data["hypotheses"]), "start_samples_sha256": _digest(starts),
                           "query_times_sha256": _digest(query_times),
                           "appearance": appearance_field.source_record() if appearance_field is not None else None,
                           "road_distance": road_distance_field.source_record() if road_distance_field is not None else None}}


def align_candidate_clusters(vo_xy, timestamps, intrinsics, candidate_clusters, *, map_crs, **kwargs):
    """Convert supplied literal-conjunction clusters; never match/fetch/select them.

    Accept a clue builder's complete result or its candidate_clusters list.
    POIs are converted from WGS84 into the caller's projected metric frame.
    """
    from pyproj import CRS, Transformer

    clue_provenance = None
    if isinstance(candidate_clusters, Mapping):
        clue_provenance = {k: v for k, v in candidate_clusters.items() if k != "candidate_clusters"}
        candidate_clusters = candidate_clusters["candidate_clusters"]
    crs = CRS.from_user_input(map_crs)
    if not crs.is_projected or not np.isclose(crs.axis_info[0].unit_conversion_factor, 1.0):
        raise ValueError("map_crs must be projected in metres")
    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    observations, keys, pois, hypotheses = [], {}, {}, []
    for index, cluster in enumerate(candidate_clusters):
        for poi in cluster["pois"]:
            lat, lon = float(poi["lat"]), float(poi["lon"])
            if not np.isfinite([lat, lon]).all() or abs(lat) > 90 or abs(lon) > 180:
                raise ValueError("OSM POIs must contain finite WGS84 coordinates")
            point = np.asarray(project.transform(lon, lat))
            if poi["id"] in pois and not np.allclose(pois[poi["id"]]["xy"], point, atol=1e-6, rtol=0):
                raise ValueError("the same OSM POI ID has conflicting coordinates")
            pois[poi["id"]] = {"id": poi["id"], "xy": point}
        assignments = {}
        for edge in cluster["conjunctions"]:
            for witness in edge["witnesses"]:
                for observation in witness["observations"]:
                    box = observation.get("bbox")
                    key = (float(observation["t_sec"]), str(observation["text"]).casefold(),
                           tuple(box) if box is not None else ())
                    if key not in keys:
                        identifier = f"observation_{len(observations)}"
                        keys[key] = identifier
                        observations.append({**observation, "id": identifier})
                    assignments.setdefault(keys[key], []).append(observation["poi_id"])
        hypotheses.append({"id": str(cluster.get("id", f"hypothesis_{index}")), "assignments": assignments})
    result = align_poi_hypotheses(vo_xy, timestamps, intrinsics, observations, list(pois.values()), hypotheses, **kwargs)
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    route = np.asarray(result["selected"]["route_xy"])
    lon, lat = inverse.transform(route[:, 0], route[:, 1])
    result["selected"]["route_latlon"] = np.c_[lat, lon].tolist()
    result["provenance"]["map_crs"] = crs.to_string()
    # Retain the complete inventory-backed alternatives, including every
    # witness, rather than replacing them with only the selected branch.
    result["candidate_clusters"] = json.loads(json.dumps(candidate_clusters, allow_nan=False))
    result["provenance"]["candidate_clusters_sha256"] = _json_digest(candidate_clusters)
    if clue_provenance is not None:
        result["provenance"]["literal_clues_sha256"] = _json_digest(clue_provenance)
        result["provenance"]["literal_clue_sources"] = clue_provenance
    return result
