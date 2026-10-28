"""Synthetic tests only: no data files/models/labels/network."""
from copy import deepcopy
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from monocular_osm import timed_poi_heading as core
from monocular_osm.poi_clues import build_poi_hypotheses


def case():
    pois = []
    for i, latitude in enumerate((40., 40.01)):
        for j, name in enumerate(("AlphaMarket", "BetaCafe")):
            pois.append({"id": f"node/{i*2+j}", "lat": latitude+j*.00005, "lon": -80.,
                         "tags": {"name": name, "shop": "supermarket"}})
    inventory = {"city": "Synthetic", "scope": "named_city_bbox", "pois": pois}
    detections = [{"text": "AlphaMarket", "confidence": .99, "t_sec": .5, "bbox": [60., 20., 80., 30.]},
                  {"text": "BetaCafe", "confidence": .98, "t_sec": .5, "bbox": [20., 20., 40., 30.]}]
    clues = build_poi_hypotheses(detections, inventory)
    K = np.array([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]])
    provenance = {"source": "same_video_native_frames", "native_bbox_grid": True,
        "position_stamps_rejected": True, "video_sha256": "a"*64, "frame_shape_hw": [80, 100],
        "window": [0., 1.], "K_sha256": core.base._array_sha(K), "detections_sha256": clues["detections_sha256"]}
    times = np.array([0., 1.])
    plan = core.build_plan(clues, detections, inventory, provenance, K, times)
    refs = np.array([[40., -80.], [40.01, -80.]])
    ids, scores = np.array(["mapillary:b", "mapillary:a"]), np.array([[.1, .2], [.3, .1]], np.float32)
    return clues, detections, inventory, provenance, K, times, plan, refs, ids, scores


def test_plan_shared_events_native_bearings_and_normalized_priors():
    *_, plan, refs, ids, scores = case()
    assert len(plan["branches"]) == 2 and len(plan["events"]) == 2
    assert all(e["row"] == 1 and e["alpha"] == .5 and e["weight"] == .5 for e in plan["events"])
    np.testing.assert_allclose(sorted(e["bearing_rad"] for e in plan["events"]), [-math.atan(.2), math.atan(.2)])
    assert all(set(b["assignments"]) == {e["id"] for e in plan["events"]} for b in plan["branches"])


def test_alias_rows_and_duplicate_native_detection_do_not_add_events():
    clues, detections, inventory, provenance, K, times, plan, *_ = case()
    duplicates = detections + [dict(detections[0])]
    rebuilt = build_poi_hypotheses(duplicates, inventory)
    provenance["detections_sha256"] = rebuilt["detections_sha256"]
    changed = core.build_plan(rebuilt, duplicates, inventory, provenance, K, times)
    assert changed["events"] == plan["events"]
    assert changed["branches"] == plan["branches"]


def test_multiple_captures_in_one_interval_decline_without_dropping_any():
    _, detections, inventory, provenance, K, times, *_ = case()
    detections[0]["t_sec"], detections[1]["t_sec"] = .25, .75
    clues = build_poi_hypotheses(detections, inventory)
    provenance["detections_sha256"] = clues["detections_sha256"]
    with pytest.raises(ValueError, match="one distinct capture time"):
        core.build_plan(clues, detections, inventory, provenance, K, times)


@pytest.mark.parametrize("change", ["bbox", "K", "time", "identity", "video", "grid"])
def test_input_identity_and_native_geometry_refuse_drift(change):
    clues, detections, inventory, provenance, K, times, *_ = case()
    if change == "bbox": detections[0]["bbox"][2] = 101.
    if change == "K": K[0, 0] *= 2
    if change == "time": detections[0]["t_sec"] = 2.
    if change == "identity": provenance["detections_sha256"] = "b"*64
    if change == "video": provenance["video_sha256"] = None
    if change == "grid": provenance["native_bbox_grid"] = False
    with pytest.raises(ValueError):
        core.build_plan(clues, detections, inventory, provenance, K, times)


def test_uncertainty_formula_uses_reference_term_once_and_has_no_hard_gate():
    assert core.distance_sigma(3.8) == math.hypot(150., .5*(30.+40.*3.8))
    with pytest.raises(ValueError): core.distance_sigma(0)
    xy = np.array([[0., 0.]])
    event = {"alpha": .5, "bearing_rad": None}
    factor = core._event_factor(xy, np.array([0]), np.array([0]), event, [np.array([126., 0.])], 3.8)
    assert core.PROFILE["outlier_floor"] < factor.min() < 1


def scalar_angular(error):
    error = (error+math.pi)%(2*math.pi)-math.pi
    return math.exp(-.5*(max(abs(error)-math.radians(20.), 0.)/math.radians(45.))**2)


def scalar_event(x, h0, h1, event, targets, dt, arc="primary"):
    delta = (h1-h0+math.pi)%(2*math.pi)-math.pi
    angle = h0+(1 if arc == "primary" else -1)*event["alpha"]*delta
    values = []
    sigma = math.hypot(150., .5*(30+40*dt))
    for target in targets:
        dx, dy = target-x
        distance = math.hypot(dx, dy)
        radial = math.exp(-.5*(max(distance-100., 0.)/sigma)**2)
        angular = 1.
        if event["bearing_rad"] is not None and distance > 1e-9:
            bearing = math.atan2(dx, dy)-event["bearing_rad"]
            angular = scalar_angular(bearing-angle)
        values.append(radial*angular)
    return .25+.75*(sum(values)/len(values) if values else 0.)


def scalar_capture(x, h0, h1, capture, dt):
    primary = math.exp(sum(e["weight"]*math.log(scalar_event(x, h0, h1, e, targets, dt)) for e, targets in capture))
    delta = (h1-h0+math.pi)%(2*math.pi)-math.pi
    if abs(abs(delta)-math.pi) <= 1e-12 and 0 < capture[0][0]["alpha"] < 1:
        alternate = math.exp(sum(e["weight"]*math.log(scalar_event(x, h0, h1, e, targets, dt, "alternate")) for e, targets in capture))
        return .5*(primary+alternate)
    return primary


def test_conditional_normalizer_matches_independent_degree_weighted_enumeration():
    xy = np.array([[0., 0.], [200., 0.], [400., 0.]])
    events = [({"alpha": .4, "bearing_rad": .3, "weight": .5}, [np.array([180., 50.])]),
              ({"alpha": .4, "bearing_rad": None, "weight": .5}, [np.array([210., -20.])])]
    tree, dt = cKDTree(xy), 1.
    normalizers, allocation = core.capture_normalizers(tree, xy, dt, {.4: events}, 2*1024**2)
    weighted = []
    for i in range(3):
        destinations = [j for j in range(3) if np.linalg.norm(xy[j]-xy[i]) <= 370.]
        for j in destinations:
            position = xy[i]+.4*(xy[j]-xy[i])
            for h0 in core.HEADINGS:
                for h1 in core.HEADINGS:
                    raw = scalar_capture(position, h0, h1, events, dt)
                    weighted.append((raw, 1./(3*len(destinations)*16**2)))
    expected = math.fsum(raw*w for raw, w in weighted)
    assert normalizers[.4] == pytest.approx(expected, abs=2e-14)
    assert math.fsum(raw/normalizers[.4]*w for raw, w in weighted) == pytest.approx(1., abs=2e-14)
    assert allocation["directed_edges"] == 7
    assert allocation["minimum_degree"] == 2 and allocation["maximum_degree"] == 3


def test_antipodal_arcs_wrap_missing_and_undefined_bearings():
    xy = np.array([[0., 0.]])
    event = {"alpha": .4, "bearing_rad": .3}
    target = np.array([20., 60.])
    values = core._event_factor(xy, np.array([0]), np.array([0]), event, [target], 1.)[0]
    for a, b in ((0, 8), (15, 0), (3, 9)):
        assert values[a, b] == pytest.approx(scalar_event(xy[0], core.HEADINGS[a], core.HEADINGS[b], event, [target], 1.), abs=1e-14)
    for target in (np.array([0., 0.]), np.array([20., 60.])):
        neutral = core._event_factor(xy, np.array([0]), np.array([0]), {"alpha": .4, "bearing_rad": None}, [target], 1.)[0]
        assert np.ptp(neutral) == 0
    undefined = core._event_factor(xy, np.array([0]), np.array([0]), event, [xy[0]], 1.)[0]
    assert np.ptp(undefined) == 0


@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("alpha", [0., .5, 1.])
def test_simultaneous_events_share_one_antipodal_camera_heading(aligned, alpha):
    xy = np.array([[0., 0.]])
    event = {"alpha": alpha, "bearing_rad": 0., "weight": .5}
    first, second = np.array([100., 0.]), np.array([100. if aligned else -100., 0.])
    capture = [(event, [first]), (dict(event), [second])]
    value = core._capture_factor(xy, np.array([0]), np.array([0]), capture, 1.)[0, 0, 8]
    expected = scalar_capture(xy[0], 0., math.pi, capture, 1.)
    assert value == pytest.approx(expected, abs=1e-14)
    if alpha == .5 and not aligned:
        assert value == pytest.approx(.5013468111, abs=1e-10)
        wrong_eventwise = math.prod((.5*(scalar_event(xy[0], 0., math.pi, e, targets, 1.)+
            scalar_event(xy[0], 0., math.pi, e, targets, 1., "alternate")))**e["weight"] for e, targets in capture)
        assert wrong_eventwise-value > .1


def test_null_path_and_objective_are_exact_existing_erased_control_plus_prior():
    *_, times, plan, refs, ids, scores = case()
    result = core.decode_timed_tracks(scores, ids, refs, times, plan)
    expected = core.base.decode_heading_tracks(scores[:, None], ids, refs, times, [None]*2, np.full(2, np.nan), np.array([0.]))
    assert len(result["outputs"]) == 8
    for variant in ("raw", "reference_mean"):
        actual, erased = result["outputs"][variant+"/null"], expected["outputs"][variant+"/erased"]
        assert actual["chosen_ref_ids"] == erased["chosen_ref_ids"]
        assert actual["heading_bin_indices"] == erased["heading_bin_indices"]
        assert actual["objective_without_prior"] == erased["objective"]
        assert actual["objective"] == erased["objective"]+math.log(.5)
        assert actual["cue_objective"] == 0
    assert sum(result["outputs"]["raw/"+b]["component_prior"] for b in result["component_order"]) == 1


def test_input_reference_permutation_and_scratch_chunking_preserve_outputs():
    *_, times, plan, refs, ids, scores = case()
    first = core.decode_timed_tracks(scores, ids, refs, times, plan, max_temp_bytes=512*1024)
    second = core.decode_timed_tracks(scores[:, ::-1].copy(), ids[::-1], refs[::-1], times, plan, max_temp_bytes=2*1024**2)
    for key in first["outputs"]:
        a, b = first["outputs"][key], second["outputs"][key]
        assert a["chosen_ref_ids"] == b["chosen_ref_ids"]
        assert a["heading_bin_indices"] == b["heading_bin_indices"]
        assert a["objective"] == b["objective"]
        for capture in a["capture_evidence"]:
            for event in capture["events"]:
                assert event["outlier_responsibility"]+sum(p["responsibility"] for p in event["choices"]) == pytest.approx(1.)


@pytest.mark.parametrize("budget", [True, 256*1024**2+1, 1000])
def test_scratch_budget_refuses_before_neighbor_allocation(monkeypatch, budget):
    *_, times, plan, refs, ids, scores = case()
    monkeypatch.setattr(core.base, "_neighbor_blocks", lambda *a: pytest.fail("allocated neighbors"))
    with pytest.raises(ValueError):
        core.decode_timed_tracks(scores, ids, refs, times, plan, max_temp_bytes=budget)


def test_modified_bracket_or_duplicate_associations_are_refused():
    *_, times, plan, refs, ids, scores = case()
    for field in ("alpha", "association"):
        changed = deepcopy(plan)
        if field == "alpha": changed["events"][0]["alpha"] = .75
        else:
            row = changed["branches"][0]["assignments"][changed["events"][0]["id"]]
            row.append(row[0])
        changed["plan_sha256"] = core.base._json_sha({k: v for k, v in changed.items() if k != "plan_sha256"})
        with pytest.raises(ValueError): core.decode_timed_tracks(scores, ids, refs, times, changed)


def scalar_circle_penalty(error, sigma, outlier):
    error = (error+math.pi)%(2*math.pi)-math.pi
    normalizer = sigma*math.sqrt(2*math.pi)*math.erf(math.pi/(math.sqrt(2)*sigma))
    signal, noise = (1-outlier)/normalizer, outlier/(2*math.pi)
    return math.log((signal*math.exp(-.5*(error/sigma)**2)+noise)/(signal+noise))


def scalar_motion(start, end, h0, h1, dt):
    delta = end-start;distance = math.hypot(*delta)
    free = 30+40*dt
    if distance > np.nextafter(free+300, np.inf): return -math.inf
    turn = scalar_circle_penalty(h1-h0, math.radians(45)*math.sqrt(dt/7.5), .1)
    east, north = math.sin(h0)+math.sin(h1), math.cos(h0)+math.cos(h1)
    bearing = 0.
    if distance > 20 and math.hypot(east, north) >= 1e-12:
        bearing = min(1., distance/150)*scalar_circle_penalty(math.atan2(east, north)-math.atan2(delta[0], delta[1]), math.radians(60), .2)
    return turn+bearing-.5*(max(distance-free, 0.)/150)**2


def test_joint_MAP_matches_tiny_exhaustive_normalized_pseudo_posterior():
    *_, times, plan, refs, ids, scores = case()
    result = core.decode_timed_tracks(scores, ids, refs, times, plan)
    data = core.base._inputs(scores[:, None], ids, refs, times, [None]*2, np.full(2, np.nan),
        np.array([0.]), None, np.zeros(1), core.PROFILE["temporary_bytes"], core.base.PROFILE["max_joint_backpointer_bytes"])
    xy, ordered_scores = data["xy"], data["scores"][:, 0]
    components = [("null", None, .5)]+[(b["id"], b, .25) for b in plan["branches"]]
    candidates = {"raw": [], "reference_mean": []}
    for name, branch, prior in components:
        choices, normalizer = None, 1.
        if branch:
            points = {p["id"]: np.array(data["frame"].to_xy(p["lat"], p["lon"])) for p in branch["pois"]}
            choices = [(e, [points[p] for p in branch["assignments"][e["id"]]]) for e in plan["events"]]
        values = []
        for j in range(2):
            for h1 in range(16):
                for i in range(2):
                    degree = sum(np.linalg.norm(xy[k]-xy[i]) <= 370 for k in range(2))
                    for h0 in range(16):
                        motion = scalar_motion(xy[i], xy[j], core.HEADINGS[h0], core.HEADINGS[h1], 1.)
                        if not math.isfinite(motion): continue
                        raw_factor = 1.
                        if choices:
                            position = .5*(xy[i]+xy[j])
                            raw_factor = scalar_capture(position, core.HEADINGS[h0], core.HEADINGS[h1], choices, 1.)
                        values.append((i, h0, j, h1, motion, raw_factor, 1./(2*degree*16**2)))
        if choices: normalizer = math.fsum(row[5]*row[6] for row in values)
        for i, h0, j, h1, motion, raw_factor, _ in values:
            for variant in candidates:
                emission = float(ordered_scores[0, i])/.02+float(ordered_scores[1, j])/.02
                if variant == "reference_mean":
                    means = ordered_scores.mean(axis=0, dtype=np.float64)
                    emission -= (means[i]+means[j])/.02
                value = emission+motion+math.log(raw_factor/normalizer)+math.log(prior)
                candidates[variant].append((value, name, i, h0, j, h1))
    for variant, rows in candidates.items():
        peak = max(row[0] for row in rows)
        mass = math.fsum(math.exp(row[0]-peak) for row in rows)
        probabilities = [math.exp(row[0]-peak)/mass for row in rows]
        assert math.fsum(probabilities) == pytest.approx(1.)
        expected = rows[int(np.argmax(probabilities))]
        for component in result["component_order"]:
            component_peak = max(row[0] for row in rows if row[1] == component)
            assert result["outputs"][variant+"/"+component]["objective"] == pytest.approx(component_peak, abs=1e-12)
        actual = result["outputs"][variant+"/joint"]
        assert actual["joint_selected_component"] == expected[1]
        assert actual["chosen_ref_ids"] == data["ids"][[expected[2], expected[4]]].tolist()
        assert actual["heading_bin_indices"] == [expected[3], expected[5]]
        assert actual["objective"] == pytest.approx(expected[0], abs=1e-12)
