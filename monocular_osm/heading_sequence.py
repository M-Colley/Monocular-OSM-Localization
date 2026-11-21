"""Experimental all-reference position/heading MAP sequence model.

Pure numerical code: no files, models, network, labels or pipeline policy.
All accuracy constants below are fixed posthoc DEV hypotheses, not calibrated
probabilities or a validated default. Camera headings are optical directions,
not reference-vehicle travel bearings. Unknown/panoramic cameras are neutral.
"""
from __future__ import annotations

import hashlib
import json
import math
from types import MappingProxyType

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalFrame


_H = 16
_MODES = ("erased", "reference_heading", "reference_heading_vo")
_VARIANTS = ("raw", "reference_mean")
_FIXED_SCRATCH = 64 * 1024
_EDGE_SCRATCH = 8192  # Two [edge,16,16] float64 tiles plus bounded reductions.
_EMISSION_SCRATCH = 2048  # Per-reference single-row angular/view workspace.
PROFILE = {
    "version": "all-reference-heading-sequence-v1", "experimental": True,
    "validation_status": "posthoc_DEV_unvalidated_fixed_profile",
    "modes": _MODES, "variants": _VARIANTS, "heading_bins": _H,
    "heading_convention": "degrees clockwise from north; bins 0,22.5,...337.5",
    "appearance_temperature": .02,
    "reference_heading_sigma_deg": 45., "reference_heading_outlier": .2,
    "reference_heading_cameras": ("perspective",),
    "reference_heading_kernel": "Gaussian shortest-arc weights normalized over all16 bins, mixed uniform",
    "view_pooling": "max_v(cosine/temperature + log_heading_probability)",
    "heading_mass": "subtract logmeanexp_h(augmented - pooled_max/temperature)",
    "reference_mean": "mean over time of max-view cosine; same reference constant subtracted from every view",
    "spatial_free_base_m": 30., "spatial_free_speed_mps": 40.,
    "reference_position_sigma_m": 150., "radius_sigma_allowance": 2.,
    "spatial_potential": "Gaussian excess-distance penalty beyond free_base+free_speed*actual_dt",
    "spatial_boundary": "nextafter(radius,+inf) for KDTree query and measured Euclidean distance",
    "bearing_sigma_deg": 60., "bearing_outlier": .2,
    "bearing_weight": "min(1,distance/150); neutral at distance<=20m or undefined circular mean",
    "bearing_neutral_distance_m": 20.,
    "turn_sigma_deg": 45., "turn_time_base_s": 7.5, "turn_outlier": .1,
    "vo_sigma_deg": 30., "vo_outlier": .25, "vo_weight": .5,
    "vo_input": "signed CLOCKWISE camera yaw increment in degrees; NaN/zero quality is missing; no unit-step distance",
    "angular_transition_kernel": "shortest-arc Gaussian normalized on[-pi,pi] plus uniform; log density ratio to zero residual",
    "objective": "unnormalized additive MAP potentials; no transition row normalization",
    "initial_state": "uniform over all reference/heading states",
    "ties": "lexicographic reference ID then ascending heading bin at predecessor and terminal",
    "primary_route": "unsmoothed joint MAP reference/heading path",
    "confidence": None, "candidate_pruning": False,
    "search_claim": "exact max-product over every supplied reference/heading and every physical-radius edge",
    "max_queries": 80, "max_references": 10000, "max_views": 16,
    "max_cosine_values": 12800000, "max_ID_itemsize": 512,
    "max_spatial_edges_per_step": 2000000,
    "default_temp_bytes": 64 * 1024**2, "hard_max_temp_bytes": 256 * 1024**2,
    "fixed_scratch_bytes": _FIXED_SCRATCH, "scratch_bytes_per_edge": _EDGE_SCRATCH,
    "scratch_bytes_per_emission_reference": _EMISSION_SCRATCH,
    "max_joint_backpointer_bytes": 512 * 1024**2,
    "default_policy_adopted": False, "selected_variant": None,
}


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


PROFILE_SHA256 = _json_sha(PROFILE)
PROFILE = MappingProxyType(PROFILE)
HEADINGS_DEG = tuple(float(v) for v in np.arange(_H) * (360. / _H))
_HEADINGS = np.radians(HEADINGS_DEG)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _array_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    if a.size:
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _wrap(value):
    return (np.asarray(value) + np.pi) % (2. * np.pi) - np.pi


def _angular_penalty(delta, sigma, outlier, *, out=None):
    """Proper circular density mixture, expressed relative to its peak.

    This constant subtraction makes missing and perfectly aligned evidence
    neutral. It avoids preferring stationary/undefined bearings merely because
    a valid continuous density has an arbitrary dimensional normalizer.
    """
    result = np.array(delta, dtype=np.float64, copy=True) if out is None else out
    if out is not None:
        np.copyto(result, delta)
    result += np.pi
    np.remainder(result, 2. * np.pi, out=result)
    result -= np.pi
    result /= sigma
    # Beyond40 sigma the Gaussian is already below float64's smallest
    # subnormal. Clipping before squaring avoids overflow for tiny actual dt.
    np.clip(result, -40., 40., out=result)
    np.square(result, out=result)
    result *= -.5
    np.exp(result, out=result)
    normalizer = sigma * math.sqrt(2. * np.pi) * math.erf(np.pi / (math.sqrt(2.) * sigma))
    signal, noise = (1. - outlier) / normalizer, outlier / (2. * np.pi)
    result *= signal
    result += noise
    np.log(result, out=result)
    result -= math.log(signal + noise)
    return result


def _inputs(view_cosines, ref_ids, ref_latlon, query_times, camera_types,
            reference_headings_deg, crop_offsets_deg, vo_yaw_delta_deg,
            vo_quality, max_temp_bytes, max_backpointer_bytes):
    score, ids, refs, times = map(np.asarray, (view_cosines, ref_ids, ref_latlon, query_times))
    _require(score.ndim == 3 and score.dtype.kind == "f" and score.dtype.itemsize in (4, 8),
             "float32/float64 [query,view,reference] cosines required")
    q, views, n = score.shape
    _require(2 <= q <= PROFILE["max_queries"] and 1 <= views <= PROFILE["max_views"]
             and 1 <= n <= PROFILE["max_references"] and score.size <= PROFILE["max_cosine_values"],
             "cosine dimensions exceed fixed profile bounds")
    _require(ids.shape == (n,) and ids.dtype.kind in "US" and ids.dtype.itemsize <= PROFILE["max_ID_itemsize"],
             "bounded string reference IDs required")
    strings = ids.astype(str)
    _require(len(set(strings.tolist())) == n and all(strings), "nonempty unique reference IDs required")
    _require(refs.shape == (n, 2) and refs.dtype.kind in "fiu" and np.isfinite(refs).all()
             and np.all(np.abs(refs[:, 0]) <= 90) and np.all(np.abs(refs[:, 1]) <= 180),
             "finite WGS84 reference coordinates required")
    _require(times.shape == (q,) and times.dtype.kind in "fiu" and np.isfinite(times).all()
             and times[0] >= 0 and np.all(np.diff(times.astype(np.float64)) > 0),
             "finite increasing actual query times required")
    intervals = np.diff(times.astype(np.float64))
    _require(np.isfinite(intervals).all() and intervals.max() < np.finfo(float).max / 80.,
             "actual intervals overflow physical-radius arithmetic")
    _require(np.isfinite(score).all() and np.max(np.abs(score)) <= 1.00001, "finite raw cosines required")
    _require(camera_types is not None and not isinstance(camera_types, (str, bytes)),
             "one camera-type token per reference required")
    try:
        cameras = list(camera_types)
    except TypeError as error:
        raise ValueError("one camera-type token per reference required") from error
    _require(len(cameras) == n and all(v is None or isinstance(v, str) and len(v) <= 64 for v in cameras),
             "one bounded camera-type token per reference required")
    camera_tokens = [v or "" for v in cameras]
    cameras = np.asarray(camera_tokens)
    angles, offsets = np.asarray(reference_headings_deg), np.asarray(crop_offsets_deg)
    _require(angles.shape == (n,) and angles.dtype.kind in "fiu" and not np.isinf(angles).any()
             and np.all(np.isnan(angles) | ((angles >= 0) & (angles <= 360))),
             "reference headings must be degrees in[0,360] or NaN for missing")
    _require(offsets.shape == (views,) and offsets.dtype.kind in "fiu" and np.isfinite(offsets).all()
             and np.all(np.abs(offsets) <= 180), "finite signed crop offsets in degrees required")
    yaw = np.full(q-1, np.nan) if vo_yaw_delta_deg is None else np.asarray(vo_yaw_delta_deg)
    quality = np.ones(q-1) if vo_quality is None else np.asarray(vo_quality)
    _require(yaw.shape == (q-1,) and yaw.dtype.kind in "fiu" and not np.isinf(yaw).any()
             and np.all(np.isnan(yaw) | (np.abs(yaw) <= 1e6)), "bounded signed clockwise VO increments or NaN required")
    _require(quality.shape == (q-1,) and quality.dtype.kind in "fiu" and np.isfinite(quality).all()
             and np.all((quality >= 0) & (quality <= 1)), "VO quality must lie in[0,1]")
    _require(type(max_temp_bytes) is int and _FIXED_SCRATCH + _EDGE_SCRATCH <= max_temp_bytes <= PROFILE["hard_max_temp_bytes"],
             "temporary memory budget is outside fixed bounds")
    _require(type(max_backpointer_bytes) is int and 0 < max_backpointer_bytes <= PROFILE["max_joint_backpointer_bytes"],
             "backpointer budget is outside fixed bounds")
    back_bytes = len(_MODES) * len(_VARIANTS) * q * n * _H * np.dtype(np.int32).itemsize
    _require(back_bytes <= max_backpointer_bytes, "joint backpointer allocation exceeds budget")
    _require(_FIXED_SCRATCH + n*8 + _EDGE_SCRATCH <= max_temp_bytes,
             "spatial-count workspace exceeds temporary budget")
    order = np.argsort(strings, kind="stable")
    refs64 = refs.astype(np.float64)[order]
    lat = float(refs64[:, 0].mean())
    longitude = np.radians(refs64[:, 1])
    sin, cos = float(np.sin(longitude).mean()), float(np.cos(longitude).mean())
    _require(np.hypot(sin, cos) > 1e-8, "references have no stable local metric origin")
    frame = LocalFrame(lat, float(np.degrees(np.arctan2(sin, cos))))
    xy = np.column_stack(frame.to_xy(refs64[:, 0], refs64[:, 1])).astype(np.float64)
    _require(np.isfinite(xy).all(), "references cannot form a finite local metric frame")
    active = (cameras == "perspective") & np.isfinite(angles)
    return {"scores": np.ascontiguousarray(score[:, :, order]), "ids": strings[order], "refs": refs64,
        "times": times.astype(np.float64), "xy": xy, "order": order, "frame": frame,
        "active": active[order], "cameras": cameras[order], "angles": angles.astype(np.float64)[order],
        "offsets": np.radians(offsets.astype(np.float64)), "yaw": np.radians(yaw.astype(np.float64)),
        "quality": np.where(np.isfinite(yaw), quality, 0.).astype(np.float64),
        "back_bytes": back_bytes, "temp_bytes": max_temp_bytes,
        "hashes": {"view_cosines": _array_sha(score), "ref_ids": _array_sha(ids), "ref_latlon": _array_sha(refs),
            "query_times": _array_sha(times), "camera_types": _json_sha(camera_tokens),
            "reference_headings_deg": _array_sha(angles), "crop_offsets_deg": _array_sha(offsets),
            "vo_yaw_delta_deg": _array_sha(yaw), "vo_quality": _array_sha(quality)}}


def _reference_means(scores, budget):
    q, _, n = scores.shape
    width = max(1, (budget-_FIXED_SCRATCH) // (q * scores.dtype.itemsize + 64))
    means = np.empty(n, np.float64)
    for start in range(0, n, width):
        stop = min(n, start+width)
        pooled = np.max(scores[:, :, start:stop], axis=1)
        means[start:stop] = pooled.mean(axis=0, dtype=np.float64)
    return means


def _emission_row(scores, active, angles, offsets, budget):
    """Return pooled and mass-neutral heading emissions for one full row."""
    n = scores.shape[1]
    pooled = np.max(scores, axis=0).astype(np.float64) / PROFILE["appearance_temperature"]
    headed = np.empty((n, _H), np.float64)
    width = max(1, (budget-_FIXED_SCRATCH) // _EMISSION_SCRATCH)
    sigma = np.radians(PROFILE["reference_heading_sigma_deg"])
    for start in range(0, n, width):
        stop = min(n, start+width)
        a = active[start:stop]
        target = headed[start:stop]; target.fill(-np.inf)
        effective_angles = np.radians(np.where(a, angles[start:stop], 0.))
        work = np.empty_like(target)
        for view, offset in enumerate(offsets):
            np.subtract(_HEADINGS[None, :] + offset, effective_angles[:, None], out=work)
            work += np.pi; np.remainder(work, 2.*np.pi, out=work); work -= np.pi
            work /= sigma; np.square(work, out=work); work *= -.5; np.exp(work, out=work)
            work /= work.sum(axis=1, keepdims=True)
            work *= 1.-PROFILE["reference_heading_outlier"]
            work += PROFILE["reference_heading_outlier"]/_H
            np.log(work, out=work)
            work[~a] = -math.log(_H)
            work += scores[view, start:stop].astype(np.float64)[:, None] / PROFILE["appearance_temperature"]
            np.maximum(target, work, out=target)
        np.subtract(target, pooled[start:stop, None], out=work)
        np.exp(work, out=work)
        target -= np.log(work.mean(axis=1))[:, None]
        target[~a] = pooled[start:stop][~a, None]
    return pooled, headed


def _neighbor_blocks(tree, xy, radius, budget):
    """Bound cardinalities before allocating Python neighbor lists or tiles."""
    upper = float(np.nextafter(radius, np.inf))
    counts = tree.query_ball_point(xy, upper, return_length=True)
    edge_count, degree = int(counts.sum()), int(counts.max())
    _require(edge_count <= PROFILE["max_spatial_edges_per_step"], "physical spatial-edge budget exceeded before allocation")
    reserve = _FIXED_SCRATCH + counts.nbytes + degree * 64
    _require(reserve + _EDGE_SCRATCH <= budget, "neighbor-list workspace exceeds temporary budget")
    capacity = (budget-reserve) // _EDGE_SCRATCH
    source = np.empty(min(capacity, edge_count), np.int32)
    destination = np.empty_like(source)
    used = 0
    for current in range(len(xy)):
        neighbors = tree.query_ball_point(xy[current], upper, return_sorted=True)
        _require(len(neighbors) == int(counts[current]), "KDTree cardinality changed")
        for offset in range(0, len(neighbors), len(source)):
            values = neighbors[offset:offset+len(source)]
            consumed = 0
            while consumed < len(values):
                take = min(len(source)-used, len(values)-consumed)
                source[used:used+take] = values[consumed:consumed+take]
                destination[used:used+take] = current
                used += take; consumed += take
                if used == len(source):
                    yield source, destination, edge_count, degree, reserve + len(source)*_EDGE_SCRATCH
                    used = 0
    if used:
        yield source[:used], destination[:used], edge_count, degree, reserve + len(source)*_EDGE_SCRATCH


def _heading_geometry():
    delta = _wrap(_HEADINGS[None, :] - _HEADINGS[:, None])
    east = np.sin(_HEADINGS)[:, None] + np.sin(_HEADINGS)[None, :]
    north = np.cos(_HEADINGS)[:, None] + np.cos(_HEADINGS)[None, :]
    undefined = np.hypot(east, north) < 1e-12
    return delta, np.arctan2(east, north), undefined


def _advance(previous, tree, xy, dt, yaw, quality, budget):
    n = len(xy)
    following = np.full_like(previous, -np.inf)
    predecessor = np.full(previous.shape, np.iinfo(np.int32).max, np.int32)
    free = PROFILE["spatial_free_base_m"] + PROFILE["spatial_free_speed_mps"] * dt
    radius = free + PROFILE["radius_sigma_allowance"] * PROFILE["reference_position_sigma_m"]
    _require(np.isfinite(radius), "physical radius is not finite")
    delta, mean, undefined = _heading_geometry()
    turn = _angular_penalty(delta, np.radians(PROFILE["turn_sigma_deg"])*math.sqrt(dt)/math.sqrt(PROFILE["turn_time_base_s"]), PROFILE["turn_outlier"])
    vo = np.zeros_like(turn) if not np.isfinite(yaw) or quality == 0 else (
        PROFILE["vo_weight"] * quality * _angular_penalty(delta-yaw, np.radians(PROFILE["vo_sigma_deg"]), PROFILE["vo_outlier"]))
    largest = edges = degree = 0
    for src, dst, edges, degree, bound in _neighbor_blocks(tree, xy, radius, budget):
        largest = max(largest, bound)
        dx, dy = xy[dst, 0]-xy[src, 0], xy[dst, 1]-xy[src, 1]
        distance = np.hypot(dx, dy)
        keep = distance <= np.nextafter(radius, np.inf)
        if not keep.all():
            src, dst, dx, dy, distance = [v[keep] for v in (src, dst, dx, dy, distance)]
        if not len(src): continue
        bearing = np.arctan2(dx, dy)
        potential = np.empty((len(src), _H, _H), np.float64)
        np.subtract(mean[None, :, :], bearing[:, None, None], out=potential)
        _angular_penalty(potential, np.radians(PROFILE["bearing_sigma_deg"]), PROFILE["bearing_outlier"], out=potential)
        potential[:, undefined] = 0.
        weight = np.minimum(1., distance/PROFILE["reference_position_sigma_m"])
        weight[distance <= PROFILE["bearing_neutral_distance_m"]] = 0.
        potential *= weight[:, None, None]
        potential += turn[None, :, :]
        spatial = -.5 * (np.maximum(0., distance-free)/PROFILE["reference_position_sigma_m"])**2
        potential += spatial[:, None, None]
        work = np.empty_like(potential)
        boundaries = np.r_[0, np.flatnonzero(np.diff(dst))+1]
        groups = np.repeat(np.arange(len(boundaries)), np.diff(np.r_[boundaries, len(src)]))
        current_ids = dst[boundaries]
        for method in range(len(previous)):
            np.add(potential, previous[method, src, :, None], out=work)
            if method % len(_MODES) == 2: work += vo[None, :, :]
            best_heading = work.argmax(axis=1)
            edge_value = np.take_along_axis(work, best_heading[:, None, :], axis=1)[:, 0, :]
            group_value = np.maximum.reduceat(edge_value, boundaries, axis=0)
            keys = src[:, None].astype(np.int64)*_H + best_heading
            keys[edge_value != group_value[groups]] = np.iinfo(np.int32).max
            group_key = np.minimum.reduceat(keys, boundaries, axis=0).astype(np.int32)
            old, old_key = following[method, current_ids], predecessor[method, current_ids]
            replace = (group_value > old) | ((group_value == old) & (group_key < old_key))
            following[method, current_ids] = np.where(replace, group_value, old)
            predecessor[method, current_ids] = np.where(replace, group_key, old_key)
    _require(np.isfinite(following).all(), "one or more physical-radius states became unreachable")
    return following, predecessor, {"edge_count": edges, "max_degree": degree, "radius_m": radius,
        "free_distance_m": free, "temporary_bound_bytes": largest}


def _selected_views(data, row, reference, heading, mode):
    score = data["scores"][row, :, reference].astype(np.float64) / PROFILE["appearance_temperature"]
    if mode != "erased" and data["active"][reference]:
        delta = _wrap(_HEADINGS[None, :] + data["offsets"][:, None] - np.radians(data["angles"][reference]))
        weights = np.exp(-.5*(delta/np.radians(PROFILE["reference_heading_sigma_deg"]))**2)
        weights /= weights.sum(axis=1, keepdims=True)
        weights = (1.-PROFILE["reference_heading_outlier"])*weights + PROFILE["reference_heading_outlier"]/_H
        score += np.log(weights[:, heading])
    return int(score.argmax())


def decode_heading_tracks(view_cosines, ref_ids, ref_latlon, query_times, camera_types,
                          reference_headings_deg, crop_offsets_deg, *, vo_yaw_delta_deg=None,
                          vo_quality=None, max_temp_bytes=PROFILE["default_temp_bytes"],
                          max_backpointer_bytes=PROFILE["max_joint_backpointer_bytes"]):
    """Freeze-ready six-way exact MAP diagnostic, without selecting a method.

    ``view_cosines`` is [query, view, reference]. Crop offsets and supplied VO
    increments are CLOCKWISE compass angles. Missing reference angles/VO are
    NaN. Only the exact camera token ``perspective`` activates heading input.
    VO quality scales optional yaw evidence; it never supplies metric distance.

    The temporary ceiling bounds block/list workspaces. Linear input/state
    buffers and joint int32 backpointers have independent fixed cardinality and
    byte caps. Excess physical-graph cardinality refuses before neighbor lists.
    """
    data = _inputs(view_cosines, ref_ids, ref_latlon, query_times, camera_types,
        reference_headings_deg, crop_offsets_deg, vo_yaw_delta_deg, vo_quality,
        max_temp_bytes, max_backpointer_bytes)
    scores, times, xy = data["scores"], data["times"], data["xy"]
    q, _, n = scores.shape
    means = _reference_means(scores, max_temp_bytes) / PROFILE["appearance_temperature"]
    pooled, headed = _emission_row(scores[0], data["active"], data["angles"], data["offsets"], max_temp_bytes)
    previous = np.stack([np.broadcast_to(pooled[:, None], (n, _H)), headed, headed]*2)
    previous[len(_MODES):] -= means[None, :, None]
    back = np.empty((len(_MODES)*len(_VARIANTS), q, n, _H), np.int32)
    back[:, 0] = -1
    tree = cKDTree(xy)
    steps = []
    for row, interval in enumerate(np.diff(times), 1):
        following, predecessor, allocation = _advance(previous, tree, xy, float(interval),
            data["yaw"][row-1], data["quality"][row-1], max_temp_bytes)
        back[:, row] = predecessor
        pooled, headed = _emission_row(scores[row], data["active"], data["angles"], data["offsets"], max_temp_bytes)
        for method in range(len(following)):
            emission = pooled[:, None] if method % len(_MODES) == 0 else headed
            following[method] += emission
            if method >= len(_MODES): following[method] -= means[:, None]
        previous = following
        steps.append(allocation)
    outputs = {}
    for method, (variant, mode) in enumerate((v, m) for v in _VARIANTS for m in _MODES):
        state = int(previous[method].argmax())
        objective = float(previous[method].reshape(-1)[state])
        references, headings = np.empty(q, np.int64), np.empty(q, np.int64)
        for row in range(q-1, -1, -1):
            reference, heading = divmod(state, _H)
            references[row], headings[row] = reference, heading
            if row: state = int(back[method, row, reference, heading])
        views = [_selected_views(data, row, int(ref), int(head), mode) for row, (ref, head) in enumerate(zip(references, headings))]
        outputs[variant+"/"+mode] = {"status": "complete", "variant": variant, "mode": mode,
            "objective": objective, "route_latlon": data["refs"][references].tolist(), "route_timestamps": times.tolist(),
            "chosen_ref_ids": data["ids"][references].tolist(), "chosen_ref_indices": data["order"][references].tolist(),
            "heading_bin_indices": headings.tolist(), "camera_heading_deg": np.asarray(HEADINGS_DEG)[headings].tolist(),
            "selected_view_indices": views, "reference_heading_available": data["active"][references].tolist(),
            "reference_heading_active": (data["active"][references] & (mode != "erased")).tolist(),
            "selected_raw_view_cosines": [float(scores[row, view, ref]) for row, (view, ref) in enumerate(zip(views, references))],
            "candidate_pruning": False, "confidence": None, "smoothed": False, "selected_answer": None}
    return {"profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256, "inputs": data["hashes"],
        "outputs": outputs, "selected_variant": None, "confidence": None,
        "allocation": {"reference_count": n, "heading_states": n*_H, "joint_backpointer_bytes": data["back_bytes"],
            "temporary_ceiling_bytes": max_temp_bytes, "steps": steps},
        "limitations": ["unnormalized MAP scores are uncalibrated", "reference positions approximate query-camera positions",
            "only perspective camera optical headings are used; panoramas remain heading agnostic",
            "missing VO is not a measured stop; no metric step lengths", "no default or policy adoption"]}
