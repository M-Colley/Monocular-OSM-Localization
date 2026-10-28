"""Experimental timed POI factors on the unchanged erased heading objective.

Pure numerical/input validation code. No files, models, network, VO or labels.
The background is conditional on the same physical reference-state graph;
neither named-city area nor a larger geographic prior enters the likelihood.
"""
from __future__ import annotations
from dataclasses import asdict
import math
from types import MappingProxyType

import numpy as np
from scipy.spatial import cKDTree

from . import heading_sequence as base
from .poi_clues import PoiClueConfig, build_poi_hypotheses

PROFILE = {"version": "timed-poi-erased-heading-v1", "posthoc_DEV": True,
    "validation_status": "unvalidated_composite_likelihood_diagnostic", "confidence": None,
    "base_profile_sha256": base.PROFILE_SHA256, "reference_headings": "all_erased",
    "VO": False, "appearance_views": 1, "appearance_crop_offset_deg": 0.,
    "variants": ["raw", "reference_mean"], "min_branches": 1, "max_branches": 6,
    "max_events": 128, "max_detections": 2000, "max_inventory": 100000,
    "max_POIs_per_branch": 128, "max_associations_per_event": 32,
    "max_capture_times_per_query_interval": 1,
    "visibility_plateau_m": 100., "reference_uncertainty_m": 150.,
    "distance_sigma": "sqrt(150^2+[0.5*(30+40*actual_dt)]^2);150_once",
    "bearing_plateau_deg": 20., "bearing_tail_sigma_deg": 45., "outlier_floor": .25,
    "undefined_bearing_m": 1e-9, "missing_bearing": "neutral_angular_factor_one",
    "sign_semantics": "OCR rays target signs_including_directories;OSM_business_centres_are_uncertain_proxies;not_triangulation",
    "event_identity": "same_time_and_native_bbox;missing_bbox_also_requires_same_text",
    "capture_weights": "uniform_over_deduplicated_events_at_identical_actual_time",
    "associations": "uniform_normalized_sum_over_unique_matched_primary_POI_IDs_no_max",
    "capture_factor": "exp(sum_event_weight*log(.25+.75*sum_association_weight*visibility*bearing))",
    "background": "uniform_source_reference_and_heading;uniform_actual_physical_destination;uniform_destination_heading",
    "normalization": "exact_capture_factor_expectation_under_shared_background;LR=factor/expectation",
    "normalizer_arithmetic": "float64_heading_mean_then_edge_order_add_at_source_then_fsum_source_means",
    "null_prior": .5, "branch_prior": ".5/B", "branch_scope": "one_global_conjunction_branch_no_switching",
    "interpolation": "metric_linear_position;shortest_arc_heading;antipodal_arcs_marginalized_once_for_shared_capture",
    "exact_query_time": "preceding_interval_except_first_query", "cue_weight": 1.,
    "joint_selection": "joint_MAP_component_with_normalized_prior;ties_null_then_lexicographic_branch_ID",
    "primary_route": "unsmoothed_MAP", "candidate_pruning": False,
    "serial_components": True, "neighbor_budget_fraction": 4, "linear_reserve_bytes_per_reference": 64,
    "temporary_bytes": 64*1024**2, "hard_temporary_bytes": 256*1024**2,
    "default_policy_adopted": False, "selected_variant": None}
PROFILE_SHA256 = base._json_sha(PROFILE)
PROFILE["variants"] = tuple(PROFILE["variants"])
PROFILE = MappingProxyType(PROFILE)
require = base._require
H = len(base.HEADINGS_DEG)
HEADINGS = np.radians(base.HEADINGS_DEG)


def _box(value, shape):
    if value is None:
        return None
    require(isinstance(value, (list, tuple, np.ndarray)) and not any(isinstance(v, (bool, np.bool_)) for v in value), "numeric native xyxy box required")
    box = np.asarray(value)
    require(box.shape == (4,) and box.dtype.kind in "fiu" and np.isfinite(box).all(), "finite native xyxy box required")
    h, w = shape
    require(0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h, "OCR box is outside its native grid")
    return box.astype(float).tolist()


def build_plan(clues, detections, inventory, provenance, K, query_times):
    """Rebuild every literal branch before deriving shared timed observations."""
    times, K = np.asarray(query_times), np.asarray(K)
    require(times.ndim == 1 and 2 <= len(times) <= 80 and times.dtype.kind in "fiu"
            and np.isfinite(times).all() and times[0] >= 0 and np.all(np.diff(times) > 0), "finite increasing query times required")
    shape, window = provenance.get("frame_shape_hw"), np.asarray(provenance.get("window"))
    require(isinstance(shape, (list, tuple)) and len(shape) == 2
            and all(type(v) is int and 0 < v <= 16384 for v in shape), "bounded native frame shape required")
    require(window.shape == (2,) and window.dtype.kind in "fiu" and np.isfinite(window).all()
            and 0 <= window[0] <= times[0] < times[-1] <= window[1], "query times outside native OCR window")
    require(K.shape == (3, 3) and K.dtype.kind == "f" and np.isfinite(K).all()
            and K[0, 0] > 0 and K[1, 1] > 0 and np.array_equal(K[2], [0, 0, 1])
            and K[0, 1] == K[1, 0] == 0 and 0 <= K[0, 2] <= shape[1] and 0 <= K[1, 2] <= shape[0],
            "finite zero-skew native pinhole K required")
    require(provenance.get("source") == "same_video_native_frames" and provenance.get("native_bbox_grid") is True
            and provenance.get("position_stamps_rejected") is True and provenance.get("K_sha256") == base._array_sha(K),
            "same-video/native-grid/K identity is missing")
    token = provenance.get("video_sha256")
    require(isinstance(token, str) and len(token) == 64 and all(c in "0123456789abcdef" for c in token), "native video identity is missing")
    require(isinstance(detections, list) and len(detections) <= PROFILE["max_detections"]
            and isinstance(inventory.get("pois"), list) and len(inventory["pois"]) <= PROFILE["max_inventory"], "bounded OCR/inventory required")
    require(clues.get("config") == asdict(PoiClueConfig()), "literal matching profile differs")
    rebuilt = build_poi_hypotheses(detections, inventory)
    require(base._json_sha(rebuilt) == base._json_sha(clues)
            and provenance.get("detections_sha256") == rebuilt["detections_sha256"], "literal observations/branches or detection identity differ")
    for row in detections:
        require(window[0] <= row["t_sec"] <= window[1], "OCR time is outside native window")
        _box(row.get("bbox"), shape)
    clusters = clues["candidate_clusters"]
    require(PROFILE["min_branches"] <= len(clusters) <= PROFILE["max_branches"], "retain all branches within fixed budget")
    grouped = {}
    for row in rebuilt["matched_observations"]:
        box = _box(row.get("bbox"), shape)
        key = base._json_sha([float(row["t_sec"]), box, row["text"] if box is None else None])
        event = grouped.setdefault(key, {"id": key, "t_sec": float(row["t_sec"]), "bbox": box,
            "texts": set(), "matched_poi_ids": set(), "confidence": 0.})
        event["texts"].add(row["text"]);event["matched_poi_ids"].add(row["poi_id"])
        event["confidence"] = max(event["confidence"], float(row["confidence"]))
    require(0 < len(grouped) <= PROFILE["max_events"], "bounded nonempty physical event set required")
    events = sorted(grouped.values(), key=lambda e: (e["t_sec"], e["id"]))
    for event in events:
        t = event["t_sec"]
        require(times[0] <= t <= times[-1], "matched OCR event outside shared query support")
        hi = min(max(int(np.searchsorted(times, t, side="left")), 1), len(times)-1)
        event.update(row=hi, alpha=float((t-times[hi-1])/(times[hi]-times[hi-1])),
            bearing_rad=None if event["bbox"] is None else float(np.arctan2(
                (event["bbox"][0]+event["bbox"][2])/2-K[0, 2], K[0, 0])),
            weight=1./sum(e["t_sec"] == t for e in events))
        event["texts"], event["matched_poi_ids"] = sorted(event["texts"]), sorted(event["matched_poi_ids"])
    branches = []
    for cluster in clusters:
        pois = sorted(cluster["pois"], key=lambda p: p["id"])
        ids = [p["id"] for p in pois]
        require(ids == sorted(set(cluster["poi_ids"])) and len(ids) == len(set(ids)), "branch POI inventory differs")
        assignments = {event["id"]: [p["id"] for p in pois if p["id"] in event["matched_poi_ids"]] for event in events}
        branches.append({"id": "poi_"+base._json_sha(ids)[:24], "pois": pois, "assignments": assignments})
    branches.sort(key=lambda b: b["id"])
    require(len({b["id"] for b in branches}) == len(branches), "duplicate literal branch")
    plan = {"profile_sha256": PROFILE_SHA256, "events": events, "branches": branches,
        "input_hashes": {"clues": base._json_sha(clues), "detections": base._json_sha(detections),
            "inventory": base._json_sha(inventory), "K": base._array_sha(K), "query_times": base._array_sha(times)},
        "native_identity": {k: provenance.get(k) for k in ("source", "video_sha256", "window", "frame_shape_hw", "K_sha256", "detections_sha256")},
        "selected_variant": None, "confidence": None}
    plan["plan_sha256"] = base._json_sha(plan)
    _validate_plan(plan, times)
    return plan


def _validate_plan(plan, times):
    require(isinstance(plan, dict) and plan.get("profile_sha256") == PROFILE_SHA256, "timed plan profile differs")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    require(base._json_sha(body) == plan.get("plan_sha256")
            and plan["input_hashes"]["query_times"] == base._array_sha(times), "timed plan/query identity differs")
    events, branches = plan["events"], plan["branches"]
    require(0 <= len(events) <= PROFILE["max_events"] and PROFILE["min_branches"] <= len(branches) <= PROFILE["max_branches"], "timed plan cardinality exceeds bounds")
    require(len({e["id"] for e in events}) == len(events), "duplicate physical OCR event")
    for event in events:
        t = event["t_sec"]
        require(isinstance(t, (int, float, np.number)) and not isinstance(t, (bool, np.bool_))
                and np.isfinite(t) and times[0] <= t <= times[-1], "timed event outside query window")
        row = min(max(int(np.searchsorted(times, t, side="left")), 1), len(times)-1)
        alpha = float((t-times[row-1])/(times[row]-times[row-1]))
        weight = 1./sum(e["t_sec"] == t for e in events)
        require(type(event["row"]) is int and event["row"] == row
                and not isinstance(event["alpha"], bool) and event["alpha"] == alpha
                and not isinstance(event["weight"], bool) and event["weight"] == weight
                and (event["bearing_rad"] is None or np.isfinite(event["bearing_rad"]) and abs(event["bearing_rad"]) <= np.pi),
                "event bracket/bearing/capture weight differs")
    for row in {event["row"] for event in events}:
        require(len({event["t_sec"] for event in events if event["row"] == row}) <= PROFILE["max_capture_times_per_query_interval"],
                "v1 requires at most one distinct capture time per query interval; no observations were dropped")
    require([b["id"] for b in branches] == sorted({b["id"] for b in branches}), "branch identities must be unique and ordered")
    for branch in branches:
        ids = [p["id"] for p in branch["pois"]]
        require(0 < len(ids) <= PROFILE["max_POIs_per_branch"] and all(isinstance(i, str) and i for i in ids)
                and ids == sorted(set(ids)), "bounded complete branch POIs required")
        for poi in branch["pois"]:
            require(not isinstance(poi["lat"], bool) and not isinstance(poi["lon"], bool)
                    and np.isfinite([poi["lat"], poi["lon"]]).all()
                    and abs(poi["lat"]) <= 90 and abs(poi["lon"]) <= 180, "finite WGS84 POI required")
        require(set(branch["assignments"]) == {e["id"] for e in events}, "every branch needs the same global event set")
        for values in branch["assignments"].values():
            require(isinstance(values, list) and len(values) <= PROFILE["max_associations_per_event"] and values == sorted(set(values))
                    and set(values) <= set(ids), "invalid or unnormalized association set")


def distance_sigma(dt):
    require(np.isfinite(dt) and dt > 0, "positive finite actual interval required")
    return math.hypot(PROFILE["reference_uncertainty_m"], .5*(30.+40.*dt))


def _blocks(tree, xy, dt, budget):
    free = base.PROFILE["spatial_free_base_m"] + base.PROFILE["spatial_free_speed_mps"]*dt
    radius = free + base.PROFILE["radius_sigma_allowance"]*base.PROFILE["reference_position_sigma_m"]
    internal = (budget-len(xy)*PROFILE["linear_reserve_bytes_per_reference"])//PROFILE["neighbor_budget_fraction"]
    require(internal >= base._FIXED_SCRATCH+base._EDGE_SCRATCH, "timed edge workspace exceeds budget")
    for src, dst, count, degree, bound in base._neighbor_blocks(tree, xy, radius, internal):
        delta = xy[dst]-xy[src]
        distance = np.hypot(delta[:, 0], delta[:, 1])
        keep = distance <= np.nextafter(radius, np.inf)
        if keep.any():
            require(4*bound+len(xy)*64 <= budget, "timed scratch bound exceeds ceiling")
            yield src[keep], dst[keep], delta[keep], distance[keep], {
                "radius_m": radius, "free_distance_m": free, "edge_count_upper": count,
                "max_degree_upper": degree, "temporary_bound_bytes": 4*bound+len(xy)*64}


def _angular_shape(delta):
    result = np.maximum(np.abs(base._wrap(delta))-np.radians(PROFILE["bearing_plateau_deg"]), 0.)
    result /= np.radians(PROFILE["bearing_tail_sigma_deg"])
    np.minimum(result, 40., out=result)
    np.square(result, out=result);result *= -.5;np.exp(result, out=result)
    return result


def _event_factor(xy, src, dst, event, targets, dt, *, arc="primary"):
    """One observation conditional on one common camera-heading arc."""
    require(arc in ("primary", "alternate"), "known shared heading arc required")
    n, alpha = len(src), event["alpha"]
    position = xy[src] + alpha*(xy[dst]-xy[src])
    delta = base._wrap(HEADINGS[None, :]-HEADINGS[:, None])
    heading = HEADINGS[:, None] + (1 if arc == "primary" else -1)*alpha*delta
    signal = np.zeros((n, H, H), np.float64)
    for target in targets:
        difference = target-position
        distance = np.hypot(difference[:, 0], difference[:, 1])
        visibility = np.maximum(distance-PROFILE["visibility_plateau_m"], 0.)/distance_sigma(dt)
        np.minimum(visibility, 40., out=visibility)
        np.square(visibility, out=visibility);visibility *= -.5;np.exp(visibility, out=visibility)
        if event["bearing_rad"] is None:
            signal += visibility[:, None, None]/len(targets)
        else:
            bearing = np.arctan2(difference[:, 0], difference[:, 1])
            error = bearing[:, None, None]-heading[None, :, :]-event["bearing_rad"]
            angular = _angular_shape(error)
            angular[distance <= PROFILE["undefined_bearing_m"]] = 1.
            angular *= visibility[:, None, None]/len(targets)
            signal += angular
    signal *= 1.-PROFILE["outlier_floor"]
    signal += PROFILE["outlier_floor"]
    return signal


def _capture_conditional(xy, src, dst, capture, dt, arc):
    combined = np.zeros((len(src), H, H), np.float64)
    for event, targets in capture:
        factor = _event_factor(xy, src, dst, event, targets, dt, arc=arc)
        np.log(factor, out=factor);factor *= event["weight"]
        combined += factor
    np.exp(combined, out=combined)
    require(np.isfinite(combined).all() and combined.min() >= PROFILE["outlier_floor"]-1e-14
            and combined.max() <= 1.+1e-14, "bounded capture factor became invalid")
    return combined


def _capture_factor(xy, src, dst, capture, dt):
    require(bool(capture) and all(e["alpha"] == capture[0][0]["alpha"] for e, _ in capture)
            and abs(math.fsum(e["weight"] for e, _ in capture)-1.) <= 1e-12,
            "capture must share one timestamp/heading interpolation and normalized weights")
    combined = _capture_conditional(xy, src, dst, capture, dt, "primary")
    alpha = capture[0][0]["alpha"]
    if 0 < alpha < 1:
        delta = base._wrap(HEADINGS[None, :]-HEADINGS[:, None])
        opposite = np.abs(np.abs(delta)-np.pi) <= 1e-12
        other = _capture_conditional(xy, src, dst, capture, dt, "alternate")
        combined[:, opposite] = .5*(combined[:, opposite]+other[:, opposite])
    return combined


def _captures(branch, events, frame):
    poi = {p["id"]: np.asarray(frame.to_xy(p["lat"], p["lon"]), np.float64) for p in branch["pois"]}
    result = {}
    for event in events:
        values = np.array([poi[key] for key in branch["assignments"][event["id"]]], np.float64).reshape(-1, 2)
        require(np.isfinite(values).all(), "POIs cannot form finite metric positions")
        associated = {**event, "association_poi_ids": branch["assignments"][event["id"]]}
        result.setdefault(event["row"], {}).setdefault(event["t_sec"], []).append((associated, values))
    return result


def capture_normalizers(tree, xy, dt, captures, budget):
    """Exact E_mu[r] with uniform source and degree-normalized destinations."""
    degrees = np.zeros(len(xy), np.int64)
    for src, _, _, _, _ in _blocks(tree, xy, dt, budget):
        np.add.at(degrees, src, 1)
    require((degrees > 0).all(), "physical proposal has a source with no destinations")
    result = {}
    for when, capture in sorted(captures.items()):
        sums = np.zeros(len(xy), np.float64)
        for src, dst, _, _, _ in _blocks(tree, xy, dt, budget):
            factors = _capture_factor(xy, src, dst, capture, dt)
            # Fixed per-edge reduction and ordered add.at make block boundaries
            # immaterial to the source accumulation order.
            np.add.at(sums, src, factors.mean(axis=(1, 2), dtype=np.float64))
        value = math.fsum((sums/degrees).tolist())/len(xy)
        require(np.isfinite(value) and PROFILE["outlier_floor"]-1e-13 <= value <= 1.+1e-13,
                "conditional background normalizer is invalid")
        result[when] = value
    return result, {"source_count": len(xy), "directed_edges": int(degrees.sum()),
        "minimum_degree": int(degrees.min()), "maximum_degree": int(degrees.max()),
        "background_probability_mass": 1.}


def _base_potential(delta_xy, distance, dt):
    delta, mean, undefined = base._heading_geometry()
    turn = base._angular_penalty(delta, np.radians(base.PROFILE["turn_sigma_deg"])*math.sqrt(dt)
        /math.sqrt(base.PROFILE["turn_time_base_s"]), base.PROFILE["turn_outlier"])
    bearing = np.arctan2(delta_xy[:, 0], delta_xy[:, 1])
    potential = np.empty((len(distance), H, H), np.float64)
    np.subtract(mean[None, :, :], bearing[:, None, None], out=potential)
    base._angular_penalty(potential, np.radians(base.PROFILE["bearing_sigma_deg"]),
                          base.PROFILE["bearing_outlier"], out=potential)
    potential[:, undefined] = 0.
    weight = np.minimum(1., distance/base.PROFILE["reference_position_sigma_m"])
    weight[distance <= base.PROFILE["bearing_neutral_distance_m"]] = 0.
    potential *= weight[:, None, None]
    potential += turn[None, :, :]
    free = base.PROFILE["spatial_free_base_m"]+base.PROFILE["spatial_free_speed_mps"]*dt
    spatial = -.5*(np.maximum(0., distance-free)/base.PROFILE["reference_position_sigma_m"])**2
    potential += spatial[:, None, None]
    return potential


def _advance(previous, tree, xy, dt, captures, normalizers, budget):
    if not captures:
        return base._advance(previous, tree, xy, dt, np.nan, 0., budget)
    following = np.full_like(previous, -np.inf)
    predecessor = np.full(previous.shape, np.iinfo(np.int32).max, np.int32)
    last, largest = {}, 0
    for src, dst, delta, distance, allocation in _blocks(tree, xy, dt, budget):
        last, largest = allocation, max(largest, allocation["temporary_bound_bytes"])
        potential = _base_potential(delta, distance, dt)
        for when, capture in sorted(captures.items()):
            cue = _capture_factor(xy, src, dst, capture, dt)
            np.log(cue, out=cue);cue -= math.log(normalizers[when])
            potential += cue
        work = np.empty_like(potential)
        boundaries = np.r_[0, np.flatnonzero(np.diff(dst))+1]
        groups = np.repeat(np.arange(len(boundaries)), np.diff(np.r_[boundaries, len(src)]))
        current = dst[boundaries]
        for method in range(len(previous)):
            np.add(potential, previous[method, src, :, None], out=work)
            best_heading = work.argmax(axis=1)
            edge_value = np.take_along_axis(work, best_heading[:, None, :], axis=1)[:, 0, :]
            group_value = np.maximum.reduceat(edge_value, boundaries, axis=0)
            keys = src[:, None].astype(np.int64)*H+best_heading
            keys[edge_value != group_value[groups]] = np.iinfo(np.int32).max
            group_key = np.minimum.reduceat(keys, boundaries, axis=0).astype(np.int32)
            old, old_key = following[method, current], predecessor[method, current]
            replace = (group_value > old) | ((group_value == old) & (group_key < old_key))
            following[method, current] = np.where(replace, group_value, old)
            predecessor[method, current] = np.where(replace, group_key, old_key)
    require(np.isfinite(following).all(), "physical graph state became unreachable")
    return following, predecessor, {**last, "temporary_bound_bytes": largest}


def _diagnostics(xy, refs, headings, captures, normalizers, times):
    diagnostics, total = [], 0.
    for row, at_times in sorted(captures.items()):
        src, dst = np.array([refs[row-1]]), np.array([refs[row]])
        first, last = int(headings[row-1]), int(headings[row])
        dt = float(times[row]-times[row-1])
        for when, capture in sorted(at_times.items()):
            raw = float(_capture_factor(xy, src, dst, capture, dt)[0, first, last])
            log_lr = math.log(raw)-math.log(normalizers[row][when])
            total += log_lr
            alpha = capture[0][0]["alpha"]
            delta = float(base._wrap(HEADINGS[last]-HEADINGS[first]))
            arcs = ["primary", "alternate"] if abs(abs(delta)-np.pi) <= 1e-12 and 0 < alpha < 1 else ["primary"]
            arc_evidence = []
            for arc in arcs:
                conditional = float(_capture_conditional(xy, src, dst, capture, dt, arc)[0, first, last])
                arc_evidence.append({"arc": arc, "conditional_capture_factor": conditional,
                                     "responsibility": conditional/(len(arcs)*raw)})
            details = []
            for event, targets in capture:
                outlier, choices, per_arc = 0., [], []
                for index, target in enumerate(targets):
                    choices.append({"poi_id": event.get("association_poi_ids", [None]*len(targets))[index],
                        "target_xy": target.tolist(), "association_weight": 1./len(targets), "responsibility": 0.})
                for arc in arc_evidence:
                    label, gamma = arc["arc"], arc["responsibility"]
                    event_raw = float(_event_factor(xy, src, dst, event, targets, dt, arc=label)[0, first, last])
                    outlier += gamma*PROFILE["outlier_floor"]/event_raw
                    per_arc.append({"arc": label, "raw_event_factor": event_raw})
                    for index, target in enumerate(targets):
                        single = float(_event_factor(xy, src, dst, event, [target], dt, arc=label)[0, first, last])
                        contribution = (single-PROFILE["outlier_floor"])/len(targets)
                        choices[index]["responsibility"] += gamma*contribution/event_raw
                details.append({"event_id": event["id"], "capture_weight": event["weight"],
                    "arc_conditioned_event_factors": per_arc, "outlier_responsibility": outlier,
                    "choices": choices})
            diagnostics.append({"row": row, "t_sec": when, "distance_sigma_m": distance_sigma(dt),
                "raw_capture_factor": raw, "background_normalizer": normalizers[row][when],
                "log_likelihood_ratio": log_lr, "shared_arc_responsibilities": arc_evidence, "events": details})
    return diagnostics, total


def _decode_component(data, captures, probability, component, budget):
    scores, times, xy = data["scores"], data["times"], data["xy"]
    q, _, n = scores.shape
    means = base._reference_means(scores, budget)/base.PROFILE["appearance_temperature"]
    first = scores[0, 0].astype(np.float64)/base.PROFILE["appearance_temperature"]
    previous = np.stack([np.broadcast_to(first[:, None], (n, H))]*2)
    previous[1] -= means[:, None]
    back = np.empty((2, q, n, H), np.int32);back[:, 0] = -1
    tree, normalizers, steps = cKDTree(xy), {}, []
    for row, interval in enumerate(np.diff(times), 1):
        dt, active = float(interval), captures.get(row, {})
        norms, background = (capture_normalizers(tree, xy, dt, active, budget) if active else ({}, None))
        normalizers[row] = norms
        following, predecessor, allocation = _advance(previous, tree, xy, dt, active, norms, budget)
        back[:, row] = predecessor
        emission = scores[row, 0].astype(np.float64)/base.PROFILE["appearance_temperature"]
        following[0] += emission[:, None]
        following[1] += emission[:, None];following[1] -= means[:, None]
        previous = following
        steps.append({"row": row, "allocation": allocation, "background": background,
            "normalizers": [{"t_sec": when, "value": value} for when, value in sorted(norms.items())]})
    outputs = {}
    for method, variant in enumerate(PROFILE["variants"]):
        state = int(previous[method].argmax())
        unprior = float(previous[method].reshape(-1)[state])
        references, headings = np.empty(q, np.int64), np.empty(q, np.int64)
        for row in range(q-1, -1, -1):
            reference, head = divmod(state, H)
            references[row], headings[row] = reference, head
            if row:
                state = int(back[method, row, reference, head])
        diagnostics, cue_objective = _diagnostics(xy, references, headings, captures, normalizers, times)
        outputs[variant] = {"status": "complete", "variant": variant, "component": component,
            "objective": unprior+math.log(probability), "objective_without_prior": unprior,
            "log_prior": math.log(probability), "component_prior": probability,
            "cue_objective": cue_objective, "base_objective": unprior-cue_objective,
            "route_latlon": data["refs"][references].tolist(), "route_timestamps": times.tolist(),
            "chosen_ref_ids": data["ids"][references].tolist(), "chosen_ref_indices": data["order"][references].tolist(),
            "heading_bin_indices": headings.tolist(), "camera_heading_deg": np.asarray(base.HEADINGS_DEG)[headings].tolist(),
            "capture_evidence": diagnostics, "candidate_pruning": False, "smoothed": False,
            "confidence": None, "selected_answer": None}
    return outputs, {"backpointer_bytes": back.nbytes, "steps": steps}


def decode_timed_tracks(raw_cosines, ref_ids, ref_latlon, query_times, plan, *,
                        max_temp_bytes=PROFILE["temporary_bytes"]):
    """All branches/null for both normalizations, plus fixed joint MAP outputs.

    Every branch shares the same global candidate graph and the same physical
    OCR event set. Components run serially. No reference heading, rigid path,
    VO distance, learned model or geographic-area prior is accepted.
    """
    raw, times = np.asarray(raw_cosines), np.asarray(query_times)
    require(raw.ndim == 2 and 2 <= raw.shape[0] <= 80 and 1 <= raw.shape[1] <= 10000
            and raw.dtype.kind == "f" and raw.dtype.itemsize in (4, 8), "bounded pooled[query,reference] appearance required")
    require(times.shape == (len(raw),) and times.dtype.kind in "fiu" and np.isfinite(times).all()
            and times[0] >= 0 and np.all(np.diff(times.astype(float)) > 0), "finite increasing shared query grid required")
    _validate_plan(plan, times)
    require(type(max_temp_bytes) is int and max_temp_bytes <= PROFILE["hard_temporary_bytes"], "temporary ceiling exceeds fixed cap")
    data = base._inputs(raw[:, None, :], ref_ids, ref_latlon, times, [None]*raw.shape[1],
        np.full(raw.shape[1], np.nan), np.array([0.]), None, np.zeros(len(times)-1), max_temp_bytes,
        base.PROFILE["max_joint_backpointer_bytes"])
    # Refuse the extra cue workspace before any neighbor/kernel allocation.
    require((max_temp_bytes-len(data["xy"])*64)//4 >= base._FIXED_SCRATCH+base._EDGE_SCRATCH,
            "temporary ceiling cannot hold bounded timed factors")
    components = [("null", None, PROFILE["null_prior"])]
    components += [(branch["id"], branch, (1.-PROFILE["null_prior"])/len(plan["branches"])) for branch in plan["branches"]]
    outputs, allocations = {}, {}
    for name, branch, probability in components:
        captures = {} if branch is None else _captures(branch, plan["events"], data["frame"])
        result, allocations[name] = _decode_component(data, captures, probability, name, max_temp_bytes)
        for variant, value in result.items():
            outputs[variant+"/"+name] = value
    for variant in PROFILE["variants"]:
        # Iteration order gives null, then lexicographic branch ID, on ties.
        chosen = max((name for name, _, _ in components), key=lambda name: outputs[variant+"/"+name]["objective"])
        outputs[variant+"/joint"] = {**outputs[variant+"/"+chosen], "joint_selected_component": chosen,
                                    "selection_rule": PROFILE["joint_selection"]}
    return {"profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256, "plan_sha256": plan["plan_sha256"],
        "inputs": data["hashes"], "outputs": outputs, "component_order": [name for name, _, _ in components],
        "allocation": allocations, "selected_variant": None, "confidence": None,
        "limitations": ["uncalibrated appearance/motion/composite-likelihood objectives",
            "conditional on OCR-selected shared gallery coverage; not full-city posterior",
            "generic native K is not physical calibration; POI centroids are not sign locations",
            "directory text may share one sign ray while mapped business centres differ; broad kernels are only an approximation",
            "one global conjunction branch per component; no production-policy adoption"]}
