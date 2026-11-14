"""Bounded experimental alignment to signposted junction TARGETS.

Exit refs are neither camera fixes nor evidence that an exit is taken. Every
supplied alternative and an unsupported/null interpretation are preserved.
No IO, models or labels. The fixed near/advance rule is post-hoc DEV research,
not a calibrated probability or an adopted localization selection policy.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import product
import hashlib
import json
import math
import numpy as np
from .poi_alignment import RasterField, transform_path


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_sha(value):
    array = np.ascontiguousarray(value)
    result = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    result.update(memoryview(array).cast("B"))
    return result.hexdigest()


@dataclass(frozen=True)
class RoadTargetConfig:
    version: str = "road-target-near-advance-v1"
    validation_status: str = "posthoc_dev_unvalidated"
    near_radius_m: float = 300.
    advance_radius_m: float = 2000.
    advance_penalty: float = .5
    distance_sigma_m: float = 750.
    distance_loss_cap: float = 6.
    bearing_free_deg: float = 90.
    bearing_sigma_deg: float = 60.
    bearing_loss_cap: float = 2.
    undefined_heading_distance_m: float = 25.
    heading_window_s: float = 2.
    null_loss: float = 2.
    sign_weight: float = 1.
    mean_speed_range_mps: tuple = (2., 35.)
    length_count: int = 9
    length_prior_weight: float = .1
    length_log_sigma: float = .9
    yaw_step_deg: float = 15.
    anchor_spacing_m: float = 200.
    refine_xy_m: tuple = (-100., 0., 100.)
    refine_yaw_deg: tuple = (-10., -5., 0., 5., 10.)
    refine_length_factors: tuple = (.9, 1., 1.1)
    basins_per_branch: int = 4
    position_basin_m: float = 500.
    yaw_basin_deg: float = 30.
    log_length_basin: float = .35
    max_observations: int = 8
    max_targets_per_observation: int = 16
    max_branches: int = 128
    max_queries: int = 80
    max_motion_poses: int = 100000
    max_coarse_poses: int = 2000000
    max_fine_poses: int = 100000
    batch_size: int = 256
    retained_per_branch: int = 16

    def __post_init__(self):
        if self.version != "road-target-near-advance-v1" or self.validation_status != "posthoc_dev_unvalidated":
            raise ValueError("this is the declared experimental road-target profile")
        for name, value in asdict(self).items():
            if isinstance(value, str):
                continue
            values = np.asarray(value)
            signed = {"near_radius_m", "advance_penalty", "bearing_free_deg", "undefined_heading_distance_m",
                      "length_prior_weight", "refine_xy_m", "refine_yaw_deg"}
            if values.dtype.kind not in "fiu" or not np.isfinite(values).all() or (np.any(values <= 0) and name not in signed):
                raise ValueError(f"invalid positive finite setting: {name}")
            if isinstance(value, tuple):
                if values.ndim != 1 or not len(values):
                    raise ValueError("search grids must be nonempty one-dimensional tuples")
                object.__setattr__(self, name, tuple(float(v) for v in values))
            elif name in {"length_count", "basins_per_branch", "max_observations", "max_targets_per_observation",
                         "max_branches", "max_queries", "max_motion_poses", "max_coarse_poses", "max_fine_poses",
                         "batch_size", "retained_per_branch"} and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError("search budgets must be integers")
        if (len(self.mean_speed_range_mps) != 2 or self.mean_speed_range_mps[1] < self.mean_speed_range_mps[0]
                or self.near_radius_m < 0 or self.advance_radius_m < self.near_radius_m
                or not 0 <= self.bearing_free_deg <= 180 or self.yaw_step_deg > 360
                or min(self.advance_penalty, self.undefined_heading_distance_m, self.length_prior_weight) < 0
                or 0 not in self.refine_xy_m or 0 not in self.refine_yaw_deg or 1 not in self.refine_length_factors):
            raise ValueError("ordered metric bounds and identity refinement are required")


def _prepare(vo_xy, timestamps, valid, observations, query_times, config):
    xy, times, quality = np.asarray(vo_xy), np.asarray(timestamps), np.asarray(valid)
    if (xy.dtype.kind not in "fiu" or xy.ndim != 2 or xy.shape[1] != 2
            or not 2 <= len(xy) <= config.max_motion_poses or not np.isfinite(xy).all() or np.any(np.abs(xy)>1e8)
            or times.shape != (len(xy),) or times.dtype.kind not in "fiu" or not np.isfinite(times).all()
            or times[0] < 0 or np.any(np.diff(times.astype(float)) <= 0)
            or quality.shape != times.shape or quality.dtype.kind != "b"):
        raise ValueError("finite aligned motion, actual ordered times and boolean validity required")
    xy, times = xy.astype(float), times.astype(float)
    arc = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if not np.isfinite(arc) or arc <= 1e-6 or times[-1]-times[0] > np.finfo(float).max/config.mean_speed_range_mps[1]:
        raise ValueError("nondegenerate finite motion duration/shape required")
    if not isinstance(observations, (list, tuple)) or not 1 <= len(observations) <= config.max_observations:
        raise ValueError("timed observation budget exceeded or missing")
    clean, seen, target_positions = [], set(), {}
    count = 1
    for row in observations:
        identifier, ref = row.get("id"), row.get("ref")
        when, confidence = row.get("t_sec"), row.get("confidence")
        if (not isinstance(identifier, str) or not identifier or identifier in seen or not isinstance(ref, str) or not ref
                or isinstance(when, bool) or not isinstance(when, (float, int)) or not times[0] <= when <= times[-1]
                or isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not 0 < confidence <= 1):
            raise ValueError("observation IDs/refs/times/confidences must be finite, unique and within motion")
        candidates = row.get("candidates")
        if not isinstance(candidates, (list, tuple)) or len(candidates) > config.max_targets_per_observation:
            raise ValueError("all supplied alternatives must fit the target budget")
        choices, used = [], set()
        for target in candidates:
            key, point = target.get("id"), np.asarray(target.get("xy"))
            if (not isinstance(key, str) or not key or key in used or point.shape != (2,)
                    or point.dtype.kind not in "fiu" or not np.isfinite(point).all() or np.any(np.abs(point)>1e8)
                    or key in target_positions and not np.array_equal(target_positions[key], point)):
                raise ValueError("target IDs and finite metric positions must be unique and consistent")
            target_positions[key] = point.astype(float); used.add(key)
            choices.append({"id": key, "xy": point.astype(float).tolist()})
        choices.sort(key=lambda target: target["id"])
        clean.append({"id": identifier, "ref": ref, "t_sec": float(when), "confidence": float(confidence), "candidates": choices})
        seen.add(identifier); count *= len(choices) + 1
        if count > config.max_branches:
            raise ValueError("all assignment/null combinations exceed branch budget before allocation")
    clean.sort(key=lambda row: (row["t_sec"], row["id"]))
    if not target_positions:
        raise ValueError("no target alternatives to anchor a route; retain the external baseline")
    query = np.asarray(query_times if query_times is not None else times[np.linspace(0, len(times)-1, min(len(times), config.max_queries)).astype(int)])
    if (query.ndim != 1 or not 2 <= len(query) <= config.max_queries or query.dtype.kind not in "fiu"
            or not np.isfinite(query).all() or np.any(np.diff(query.astype(float)) <= 0)
            or query[0] < times[0] or query[-1] > times[-1]):
        raise ValueError("actual query times must be ordered, bounded and within the motion timeline")
    def at(when):
        return np.column_stack([np.interp(when, times, xy[:, axis]) for axis in (0, 1)])
    obs_times = np.array([row["t_sec"] for row in clean])
    tangent = at(np.minimum(obs_times + config.heading_window_s/2, times[-1])) - at(np.maximum(obs_times - config.heading_window_s/2, times[0]))
    return {"xy": xy, "times": times, "arc": arc, "unit_query": (at(query)-xy[0])/arc,
            "unit_observed": (at(obs_times)-xy[0])/arc, "query_times": query.astype(float), "observations": clean,
            "headings": np.arctan2(tangent[:, 1], tangent[:, 0]), "heading_valid": np.linalg.norm(tangent, axis=1) > 1e-6,
            "weights": np.array([row["confidence"] for row in clean])/sum(row["confidence"] for row in clean),
            "valid_fraction": float(quality.mean())}


def _loss(parameters, assignment, data, config, *, details=False):
    cameras = transform_path(parameters, data["unit_observed"])
    losses = np.zeros(len(parameters)); evidence = []
    for index, target in enumerate(assignment):
        if target is None:
            loss = np.full(len(parameters), config.null_loss)
            if details:
                evidence.append({"observation_id": data["observations"][index]["id"], "target_id": None,
                                 "interpretation": "unsupported/outside-inventory/misreading", "loss": float(loss[0])})
        else:
            delta = np.asarray(target["xy"])[None] - cameras[:, index]
            distance = np.linalg.norm(delta, axis=1)
            near = .5*(np.maximum(distance-config.near_radius_m, 0)/config.distance_sigma_m)**2
            advance = config.advance_penalty + .5*(np.maximum(distance-config.advance_radius_m, 0)/config.distance_sigma_m)**2
            radial = np.minimum(np.minimum(near, advance), config.distance_loss_cap)
            angle = np.abs((np.arctan2(delta[:, 1], delta[:, 0])-np.radians(parameters[:, 2])-data["headings"][index]+np.pi)%(2*np.pi)-np.pi)
            usable = data["heading_valid"][index] & (distance > config.undefined_heading_distance_m)
            bearing = np.where(usable, np.minimum(.5*(np.maximum(np.degrees(angle)-config.bearing_free_deg, 0)/config.bearing_sigma_deg)**2,
                                                       config.bearing_loss_cap), 0.)
            loss = radial + bearing
            if details:
                evidence.append({"observation_id": data["observations"][index]["id"], "target_id": target["id"],
                    "distance_m": float(distance[0]), "range_mode": "near" if near[0] <= advance[0] else "advance",
                    "bearing_used": bool(usable[0]), "bearing_error_deg": float(np.degrees(angle[0])) if usable[0] else None,
                    "radial_loss": float(radial[0]), "bearing_loss": float(bearing[0]), "loss": float(loss[0]),
                    "constraint": "signposted target only; no camera fix or exit-taken claim"})
        losses += data["weights"][index]*loss
    return (losses, evidence) if details else losses


def _evaluate(parameters, assignment, data, appearance, config):
    poses = transform_path(parameters, data["unit_query"])
    app = appearance.sample(poses, data["query_times"]).mean(axis=1) if appearance is not None else np.zeros(len(parameters))
    duration = data["times"][-1]-data["times"][0]
    middle_speed = math.sqrt(np.prod(config.mean_speed_range_mps))
    prior = config.length_prior_weight*(np.log(parameters[:, 3]/(duration*middle_speed))/config.length_log_sigma)**2
    loss = _loss(parameters, assignment, data, config)
    return np.column_stack([parameters, app-prior-config.sign_weight*loss, app, prior, loss])


def _order(rows):
    return np.lexsort((rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0], -rows[:, 4]))


def _candidate(row, assignment, data, config):
    _, evidence = _loss(row[None, :4], assignment, data, config, details=True)
    route = transform_path(row[None, :4], data["unit_query"])[0]
    return {"start_xy": row[:2].tolist(), "yaw_deg": float(row[2]), "length_m": float(row[3]),
            "objective": float(row[4]), "appearance": float(row[5]), "length_prior_loss": float(row[6]),
            "target_loss": float(row[7]), "evidence": evidence, "route_xy": route.tolist(),
            "route_timestamps": data["query_times"].tolist()}


def align_road_targets(vo_xy, timestamps, valid, observations, *, query_times=None,
                       appearance_field=None, config=None, provenance=None):
    """All anchored assignment routes plus the unsupported interpretation.

    Observations have id/ref/t_sec/confidence/candidates[{id,xy}], in one metric
    frame. Supply ALL exact inventory matches. Every observation also gets a
    null assignment. All-null has no geographic route and remains unresolved.
    The winner is only an ANCHORED diagnostic, not baseline admission. Scale
    bounds apply to total duration, never force per-frame speeds or ban stops.
    """
    config = config or RoadTargetConfig()
    if not isinstance(config, RoadTargetConfig) or appearance_field is not None and not isinstance(appearance_field, RasterField):
        raise ValueError("declared config and optional bounded appearance RasterField required")
    sources = json.loads(json.dumps(provenance or {}, allow_nan=False))
    data = _prepare(vo_xy, timestamps, valid, observations, query_times, config)
    choices = [[None, *row["candidates"]] for row in data["observations"]]
    assignments = [entry for entry in product(*choices) if any(target is not None for target in entry)]
    span, yaw_span = config.advance_radius_m/config.anchor_spacing_m, 360/config.yaw_step_deg
    if not np.isfinite([span, yaw_span]).all() or max(span, yaw_span) > config.max_coarse_poses:
        raise ValueError("anchor/yaw grid exceeds budget before allocation")
    yaw_count = math.ceil(yaw_span)
    half_axis = math.floor(span)
    axis_count = 2*half_axis+1
    offset_bound = axis_count**2
    coarse_bound = len(assignments)*config.length_count*yaw_count*offset_bound
    fine_bound = len(assignments)*config.basins_per_branch*len(config.refine_xy_m)**2*len(config.refine_yaw_deg)*len(config.refine_length_factors)
    if coarse_bound > config.max_coarse_poses or fine_bound > config.max_fine_poses:
        raise ValueError("complete equal-branch search exceeds budget before allocation")
    # Count indices BEFORE allocation. Absolute epsilons would create enormous
    # grids at tiny metric scales and defeat the budget above.
    axis = np.arange(-half_axis, half_axis+1, dtype=float)*config.anchor_spacing_m
    radius_limit = config.advance_radius_m*(1+8*np.finfo(float).eps)
    offsets = np.array([(x, y) for x in axis for y in axis if np.hypot(x, y) <= radius_limit])
    duration = data["times"][-1]-data["times"][0]
    lengths = np.geomspace(duration*config.mean_speed_range_mps[0], duration*config.mean_speed_range_mps[1], config.length_count)
    yaws = np.arange(0., 360., config.yaw_step_deg)
    branches = []; coarse_count = fine_count = 0
    for branch_number, assignment in enumerate(assignments):
        anchor = next(index for index, target in enumerate(assignment) if target is not None)
        center = np.asarray(assignment[anchor]["xy"])
        parts = []
        for length in lengths:
            for yaw in yaws:
                shift = transform_path(np.array([[0., 0., yaw, length]]), data["unit_observed"])[0, anchor]
                parameters = np.column_stack([center+offsets-shift, np.full(len(offsets), yaw), np.full(len(offsets), length)])
                for start in range(0, len(parameters), config.batch_size):
                    parts.append(_evaluate(parameters[start:start+config.batch_size], assignment, data, appearance_field, config))
        coarse = np.concatenate(parts); coarse_count += len(coarse)
        seeds, seen = [], set()
        for index in _order(coarse):
            row = coarse[index]
            key = (math.floor(row[0]/config.position_basin_m), math.floor(row[1]/config.position_basin_m),
                   math.floor(row[2]/config.yaw_basin_deg), math.floor(math.log(row[3])/config.log_length_basin))
            if key not in seen:
                seen.add(key); seeds.append(row[:4])
            if len(seeds) == config.basins_per_branch:
                break
        fine = []
        minimum, maximum = lengths[0], lengths[-1]
        for seed, (dx, dy, dyaw, factor) in product(seeds, product(config.refine_xy_m, config.refine_xy_m, config.refine_yaw_deg, config.refine_length_factors)):
            length = seed[3]*factor
            if minimum <= length <= maximum:
                fine.append([seed[0]+dx, seed[1]+dy, (seed[2]+dyaw)%360, length])
        fine = np.unique(np.asarray(fine), axis=0)
        scored = np.concatenate([_evaluate(fine[start:start+config.batch_size], assignment, data, appearance_field, config)
                                 for start in range(0, len(fine), config.batch_size)])
        fine_count += len(fine)
        ordered = _order(scored)
        best = _candidate(scored[ordered[0]], assignment, data, config)
        alternatives = [_candidate(scored[index], assignment, data, config) for index in ordered[:config.retained_per_branch]]
        branches.append({"id": f"assignment_{branch_number:03d}",
            "assignments": {row["id"]: (target["id"] if target is not None else None) for row, target in zip(data["observations"], assignment)},
            "best": best, "alternatives": alternatives, "coarse_pose_count": len(coarse), "fine_pose_count": len(fine)})
    ranked = sorted(branches, key=lambda branch: (-branch["best"]["objective"], branch["id"]))
    return {"status": "complete", "experimental": True, "confidence_calibrated": False, "policy_adopted": False,
        "selected": {"branch_id": ranked[0]["id"], **ranked[0]["best"]}, "selection_scope": "anchored diagnostic hypotheses only",
        "unlocalized_alternative": {"assignments": {row["id"]: None for row in data["observations"]},
            "route_xy": None, "reason": "all signs may be unsupported/outside inventory; retain external baseline", "target_loss": config.null_loss},
        "branches": branches, "ranked_branch_ids": [branch["id"] for branch in ranked], "observations": data["observations"],
        "motion_quality": {"valid_fraction": data["valid_fraction"], "limitation": "valid pose fraction does not calibrate accumulated heading/shape error"},
        "search": {"coarse_poses": coarse_count, "fine_poses": fine_count, "anchored_branches": len(branches),
                   "query_count": len(data["query_times"]), "anchor_rule": "earliest assigned target; equal radius/grid for every branch",
                   "scale_scope": "global mean speed over observed motion duration; stops are allowed"},
        "provenance": {"profile": asdict(config), "profile_sha256": _json_sha(asdict(config)), "caller_sources": sources,
            "motion_xy_sha256": _array_sha(vo_xy), "motion_times_sha256": _array_sha(timestamps), "motion_valid_sha256": _array_sha(valid),
            "query_times_sha256": _array_sha(data["query_times"]), "observations_sha256": _json_sha(data["observations"]),
            "appearance": appearance_field.source_record() if appearance_field is not None else None}}
