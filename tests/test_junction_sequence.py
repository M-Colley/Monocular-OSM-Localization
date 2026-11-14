"""Private synthetic roads/motion only; no models, cases, network or labels."""
import math

import networkx as nx
import numpy as np
import pytest

from monocular_osm.graph_filter import FilterConfig, Odometry, RoadNetwork
from monocular_osm import junction_sequence as j
from monocular_osm.road_sign_clues import extract_road_sign_clues


def roads(two=False):
    g = nx.MultiDiGraph(crs="EPSG:32632")
    for lane in range(2 if two else 1):
        for i in range(3):
            g.add_node(lane * 10 + i + 1, x=455000. + i * 500., y=5428000. + lane * 5000.)
        for i in range(2):
            g.add_edge(lane * 10 + i + 1, lane * 10 + i + 2)
    return RoadNetwork.from_graph(g)


def motion(n=80, dt=.25, *, status="measured", start=4.):
    step = np.ones(n) if status == "measured" else np.zeros(n)
    ts = start + np.arange(n + 1, dtype=np.float64) * dt
    od = Odometry(step, np.zeros(n), np.diff(ts), metric=False, unit_seconds=dt)
    return od, np.full(n, status, dtype="U10"), ts


def candidate(net, edge=1, offset=None, identity=None):
    offset = net.edge_len[edge] if offset is None else offset
    identity = f"node/{net.edge_keys[edge][1]}" if identity is None else identity
    x, y, _, _ = net.locate(np.array([edge]), np.array([offset]))
    lat, lon = net.local_to_latlon(np.column_stack([x, y]))[0]
    return {"id": identity, "lat": float(lat), "lon": float(lon)}


def matches(rows):
    return {"status": "complete", "camera_fix": None, "selected_answer": None,
            "matches": [{"observation": extract_road_sign_clues(
                            [{"text": f"EXIT {i + 1}", "confidence": .99,
                              "t_sec": float(t), "bbox": [100., 100., 180., 125.]}],
                            [480, 640])["observations"][0],
                         "candidates": cs, "camera_fix": None}
                        for i, (t, cs) in enumerate(rows)]}


def test_initial_importance_ratio_exactly_recovers_road_length_prior():
    length = np.array([1., 10., 100.])
    q, ratio = j.initial_proposal(length, np.array([.001, 1., .2]))
    np.testing.assert_allclose(q * np.exp(ratio), length / length.sum(), rtol=1e-15)
    assert np.all(q >= .25 * length / length.sum())


def test_auxiliary_resampling_includes_inverse_future_guide_correction():
    class Fake:
        def resample(self, q):
            self.q = q.copy()
            return np.array([1, 0, 1])
    pf = Fake()
    w, guide = np.array([.1, .3, .6]), np.array([.1, .8, .2])
    idx = j.auxiliary_resample(pf, w, guide)
    np.testing.assert_allclose(pf.q, w * guide / (w @ guide))
    corrected = 1 / guide[idx]
    np.testing.assert_allclose(np.exp(pf.logw), corrected / corrected.sum())
    # Expected weighted ancestor measure recovers w, not w*g.
    expected = pf.q / guide
    np.testing.assert_allclose(expected / expected.sum(), w)


def test_turn_proposal_has_actual_prior_over_proposal_ratio(monkeypatch):
    net = roads(two=True)
    # Synthetic fork; all actual transitions have unnormalised prior weight 1.
    net.succ_table = np.array([[1, 2], [-1, -1], [-1, -1], [-1, -1]])
    net.succ_head = np.zeros((4, 2))
    net.succ_logprior = np.array([[0., 0.], [-np.inf, -np.inf], [-np.inf, -np.inf], [-np.inf, -np.inf]])
    od, st, ts = motion(2)
    m = j.coarsen_motion(od, st, ts)
    pf = j._OfflineFilter(net, FilterConfig(n_particles=16, path_storage=False), m, [], lambda: None)
    monkeypatch.setattr(pf, "guide", lambda e, s, k: np.where(e == 1, .1, .9))
    edges, correction, dead = pf._choose_successors(np.zeros(2000, dtype=int), [])
    expected_q = np.where(edges == 1, .2, .8)
    np.testing.assert_allclose(correction, -np.log(expected_q), rtol=1e-15)
    assert not dead.any()


def test_directed_target_distance_does_not_turn_behind_into_ahead():
    net = roads()
    target = candidate(net, 0)
    cue = j.prepare_cues(net, matches([(4., [target])]), np.array([4., 5.]))[0]
    rem = cue.targets[0].remaining(net, np.array([0, 1, 1]), np.array([0., 0., 300.]))
    assert rem[0] == pytest.approx(500., abs=1e-6)
    assert rem[1] == 0.
    assert np.isinf(rem[2])


def test_advance_sign_and_outlier_semantics_not_camera_fix():
    net = roads()
    cue = j.prepare_cues(net, matches([(4., [candidate(net, 0)])]), np.array([4., 5.]))[0]
    factor = j.sign_factor(net, cue.targets, np.array([0, 0, 1]), np.array([100., 500., 100.]), 10.)
    assert .15 < factor[0] < factor[1]
    assert factor[1] == pytest.approx(1.)
    assert factor[2] == .15


def test_all_junction_alternatives_equal_mixture_not_product_or_first_choice():
    net = roads(two=True)
    a, b = candidate(net, 1), candidate(net, 3)
    times = np.array([4., 5.])
    forward = j.prepare_cues(net, matches([(4., [a, b])]), times)[0]
    reverse = j.prepare_cues(net, matches([(4., [b, a])]), times)[0]
    edge, off = np.array([1, 3]), np.array([0., 0.])
    x = j.sign_factor(net, forward.targets, edge, off, 8.)
    np.testing.assert_allclose(x, j.sign_factor(net, reverse.targets, edge, off, 8.))
    assert x[0] == pytest.approx(x[1])
    assert len(forward.targets) == 2


def test_unknown_and_stop_composition_keep_full_counts_and_time():
    od, st, ts = motion(8)
    st[[1, 2]] = "unknown"
    st[[4, 5]] = "stationary"
    od.step[[1, 2, 4, 5]] = 0.
    m = j.coarsen_motion(od, st, ts)
    np.testing.assert_array_equal(m.counts.sum(axis=0), [4, 2, 2])
    assert m.unknown_seconds.sum() == .5
    assert m.stationary_seconds.sum() == .5
    assert m.step.sum() == 4.
    assert m.times[0] == ts[0] and m.times[-1] == ts[-1]
    assert np.isnan(m.se2[m.counts[:, 1] > 0]).all()


def test_full_5950_native_steps_keep_late_cue_and_exact_endpoints():
    od, st, ts = motion(5950, .1, start=3.986065518858444)
    st[100:110] = "unknown"
    od.step[100:110] = 0.
    m = j.coarsen_motion(od, st, ts)
    assert len(m.step) <= 1200
    assert m.counts.sum() == 5950
    assert m.native_end_indices[-1] == 5950
    assert m.times[0].hex() == ts[0].hex()
    assert m.times[-1].hex() == ts[-1].hex()
    assert m.unknown_seconds.sum() == pytest.approx(1.)
    late = matches([(float(ts[-2]), [])])
    cue = j.prepare_cues(roads(), late, m.times)[0]
    assert cue.step == len(m.step)
    assert not cue.targets


def test_composition_preserves_se2_order_and_rotation_not_angle_average():
    a = np.array([[0., -1., 0.], [1., 0., 1.], [0., 0., 1.]])
    b = a.copy()
    expected = a @ b
    np.testing.assert_array_equal(j.compose_se2(np.stack([a, b])), expected)
    od, st, ts = motion(2, .25)
    od.heading[:] = [math.pi / 2, math.pi]
    m = j.coarsen_motion(od, st, ts)
    np.testing.assert_allclose(m.se2[0], expected, atol=1e-15)
    assert m.step[0] == 2.  # path length, not chord length
    assert m.heading[0] == math.pi


def test_overlong_complete_window_is_unavailable_not_truncated():
    od, st, ts = motion(1201, .5)
    with pytest.raises(j.Unavailable, match="complete native window"):
        j.coarsen_motion(od, st, ts)


@pytest.mark.parametrize("kind", ["metric", "clock", "status", "false_stop"])
def test_motion_admission_refuses_semantic_aliases(kind):
    od, st, ts = motion(8)
    if kind == "metric": od.metric = True
    if kind == "clock": ts[3] += .01
    if kind == "status": st[2] = "bad"
    if kind == "false_stop": st[2] = "stationary"
    with pytest.raises(ValueError):
        j.coarsen_motion(od, st, ts)


def test_both_late_cue_alternatives_retained_and_offline_full_path_recovered():
    net = roads(two=True)
    od, st, ts = motion()
    a, b = candidate(net, 1), candidate(net, 3)
    out = j.align(net, od, st, ts, matches([(10., [a, b]), (23.5, [b])]),
                  config=j.Config(particles=800, seed=4))
    assert out["status"] == "complete_diagnostic_only"
    assert out["cues"][0]["junction_ids"] == ["node/13", "node/3"]
    assert out["cues"][1]["junction_ids"] == ["node/13"]
    assert np.all(out["route_edge"] >= 2)  # whole lineage, including before first cue
    assert out["times"][0] == ts[0] and out["times"][-1] == ts[-1]
    assert out["online"] is False and out["accuracy_admitted"] is False
    assert out["genealogy_payload_bytes"] == 16 * 800 * len(out["times"])


@pytest.mark.parametrize("status", ["stationary", "unknown"])
def test_unknown_moves_under_speed_prior_but_explicit_stop_does_not(status):
    net = roads()
    od, st, ts = motion(12, status=status)
    out = j.align(net, od, st, ts, matches([(ts[-1], [candidate(net)])]),
                  config=j.Config(particles=300, seed=2))
    assert out["status"] == "complete_diagnostic_only"
    x, _, _, _ = net.locate(out["route_edge"], out["route_offset_m"])
    if status == "stationary":
        np.testing.assert_array_equal(x, np.full_like(x, x[0]))
    else:
        assert x[-1] > x[0]


def test_whole_window_genealogy_budget_refuses_before_filter(monkeypatch):
    od, st, ts = motion()
    monkeypatch.setattr(j, "_OfflineFilter", lambda *a, **k: pytest.fail("allocated particles"))
    out = j.align(roads(), od, st, ts, matches([(10., [])]), config=j.Config(particles=100, genealogy_bytes=1))
    assert out["status"] == "unavailable"
    assert out["route_latlon"] is None
    assert "genealogy" in out["reason"]


def test_no_match_is_explicit_outlier_only_not_dropped():
    od, st, ts = motion(4)
    out = j.align(roads(), od, st, ts, matches([(ts[-1], [])]), config=j.Config(particles=30))
    assert out["status"] == "complete_diagnostic_only"
    assert out["cues"][0]["no_matches_outlier_only"] is True
    assert out["cues"][0]["factor_min"] == .15


def test_inner_deadline_returns_no_partial_success():
    class Clock:
        value = 0.
        def __call__(self):
            self.value += 1.
            return self.value
    od, st, ts = motion()
    out = j.align(roads(), od, st, ts, matches([(10., [])]), config=j.Config(inner_s=.5), clock=Clock())
    assert out["status"] == "unavailable" and out["route_latlon"] is None


def test_map_cannot_silently_drop_one_alternative():
    net = roads()
    a = candidate(net)
    b = {**a, "id": "node/2", "lat": a["lat"] + 1.}
    with pytest.raises(j.Unavailable, match="coordinate"):
        j.prepare_cues(net, matches([(4., [a, b])]), np.array([4., 5.]))


def test_nearby_geometry_cannot_forge_missing_topology_junction():
    net = roads()
    # The coordinate is exactly on a road, but that OSM node was not retained.
    a = candidate(net, 0, 200., identity="node/999")
    with pytest.raises(j.Unavailable, match="retained OSM topology"):
        j.prepare_cues(net, matches([(4., [a])]), np.array([4., 5.]))


def test_bare_numeric_reference_cannot_be_promoted_to_verified_exit():
    net = roads()
    observed = matches([(4., [candidate(net)])])
    observed["matches"][0]["observation"] = {"t_sec": 4., "ref": "186B"}
    with pytest.raises(ValueError, match="bare refs"):
        j.prepare_cues(net, observed, np.array([4., 5.]))
    clues = extract_road_sign_clues([{"text": "186B", "confidence": .99, "t_sec": 4.,
                                     "bbox": [100., 100., 180., 125.]}], [480, 640])
    assert clues["observations"] == []


def test_worst_genealogy_bound_is_honest():
    assert 16 * 50_000 * 1201 == 960_800_000
    assert 960_800_000 <= j.PROFILE["genealogy_bytes"] == 1 << 30


def fork_filter(degree, *, conditional, monkeypatch, tilted=False):
    net = roads(two=True)
    net.succ_table = np.full((4, 3), -1, dtype=np.int64)
    net.succ_table[0, :degree] = np.arange(1, degree + 1)
    net.succ_head = np.zeros((4, 3))
    net.succ_logprior = np.where(net.succ_table >= 0, 0., -np.inf)
    if conditional:
        net = net.with_turn_prior(0., normalise=True)
    od, st, ts = motion(2)
    pf = j._OfflineFilter(net, FilterConfig(n_particles=16, path_storage=False),
                          j.coarsen_motion(od, st, ts), [], lambda: None)
    monkeypatch.setattr(pf, "guide", lambda e, s, k: (e + .25) if tilted else np.ones(np.shape(e)))
    return pf


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_conditional_uniform_q_equals_p_has_zero_correction_and_mass_one(degree, monkeypatch):
    pf = fork_filter(degree, conditional=True, monkeypatch=monkeypatch)
    edge, correction, dead = pf._choose_successors(np.zeros(100, dtype=np.int64), [])
    np.testing.assert_allclose(correction, 0., atol=1e-15)
    assert not dead.any() and np.all((edge >= 1) & (edge <= degree))
    lp = pf.net.succ_logprior[0, :degree]
    np.testing.assert_allclose(np.exp(lp), np.full(degree, 1. / degree))
    assert np.exp(lp).sum() == pytest.approx(1.)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_conditional_importance_correction_recovers_target_under_tilted_guide(degree, monkeypatch):
    pf = fork_filter(degree, conditional=True, monkeypatch=monkeypatch, tilted=True)
    edge, correction, _ = pf._choose_successors(np.zeros(200, dtype=np.int64), [])
    p = np.full(degree, 1. / degree)
    guide = np.arange(1, degree + 1) + .25
    q = .25 * p + .75 * (p * guide) / (p @ guide)
    np.testing.assert_allclose(correction, np.log(p[edge - 1]) - np.log(q[edge - 1]), atol=1e-15)
    np.testing.assert_allclose(q * np.exp(np.log(p) - np.log(q)), p, atol=1e-15)


def test_multiple_crossings_preserve_markov_mass_and_degree_one_subdivision():
    from itertools import product
    from monocular_osm.graph_filter import _turn_logprior
    def total_mass(degrees, conditional):
        kernels = []
        for degree in degrees:
            succ = np.arange(degree, dtype=np.int64)[None]
            lp = _turn_logprior(succ, np.zeros_like(succ, dtype=float), np.zeros(1), 0., conditional)[0]
            q = np.arange(1., degree + 1)
            q /= q.sum()
            kernels.append((lp, q))
        mass = 0.
        for choices in product(*(range(d) for d in degrees)):
            proposal, correction = 1., 0.
            for choice, (lp, q) in zip(choices, kernels):
                proposal *= q[choice]
                correction += lp[choice] - np.log(q[choice])
            mass += proposal * np.exp(correction)
        return mass
    assert total_mass([2, 3], True) == pytest.approx(1.)
    assert total_mass([2, 3], False) == pytest.approx(6.)
    assert total_mass([1, 2, 1, 3, 1], True) == pytest.approx(total_mass([2, 3], True))
    assert total_mass([1, 2, 1, 3, 1], False) == pytest.approx(total_mass([2, 3], False))


def test_conditional_dead_ends_and_padding_never_nan_or_select_padding(monkeypatch):
    pf = fork_filter(2, conditional=True, monkeypatch=monkeypatch)
    old = np.array([0, 1, 0, 2, 3], dtype=np.int64)
    edge, correction, dead = pf._choose_successors(old, [])
    np.testing.assert_array_equal(dead, [False, True, False, True, True])
    assert np.all(np.isneginf(correction[dead])) and not np.isnan(correction).any()
    np.testing.assert_array_equal(edge[dead], old[dead])
    assert np.all(np.isin(edge[~dead], [1, 2]))


def test_local_conditional_view_shares_all_geometry_and_keeps_base_unit_weights():
    net = roads(two=True)
    before = net.succ_logprior.copy()
    new = net.with_turn_prior(0., normalise=True)
    assert new is not net and new.normalise_turn_prior and not net.normalise_turn_prior
    for key in net.__dataclass_fields__:
        if isinstance(getattr(net, key), np.ndarray) and key != "succ_logprior":
            assert getattr(new, key) is getattr(net, key)
    assert new.edge_keys is net.edge_keys
    np.testing.assert_array_equal(net.succ_logprior, before)


def test_default_legacy_is_unchanged_and_profiles_record_full_filter_configuration():
    from dataclasses import asdict
    net = roads()
    od, st, ts = motion(8)
    cues = matches([(ts[-1], [candidate(net)])])
    default = j.align(net, od, st, ts, cues, config=j.Config(particles=60, seed=3))
    legacy = j.align(net, od, st, ts, cues, config=j.Config(particles=60, seed=3, junction_prior="legacy"))
    conditional = j.align(net, od, st, ts, cues,
                          config=j.Config(particles=60, seed=3, junction_prior="conditional-uniform-v3"))
    for key in default:
        if isinstance(default[key], np.ndarray):
            np.testing.assert_array_equal(default[key], legacy[key])
    assert default["cues"] == legacy["cues"]
    assert default["profile"] == j.PROFILE and conditional["profile"] == j.PROFILE_V3
    assert default["method"] != conditional["method"]
    assert default["base_turn_profile_verified"] is conditional["base_turn_profile_verified"] is True
    old_cfg, new_cfg = default["effective_filter_config"], conditional["effective_filter_config"]
    assert set(old_cfg) == set(asdict(FilterConfig()))
    assert old_cfg["normalise_turn_prior"] is False and new_cfg["normalise_turn_prior"] is True
    assert {k: v for k, v in old_cfg.items() if k != "normalise_turn_prior"} == {
        k: v for k, v in new_cfg.items() if k != "normalise_turn_prior"}
    assert old_cfg["speed_range_mps"] == (.5, 55.)
    assert old_cfg["logk_bounds_margin"] == .5 and old_cfg["speed_dev_sigma"] == .3
    assert old_cfg["step_noise"] == .05
    assert conditional["profile"]["speed_range_is_hard_cap"] is False


@pytest.mark.parametrize("fault", ["normalized_input", "changed_base_weight"])
def test_v3_requires_original_unit_weight_base_before_local_transformation(fault):
    net = roads()
    od, st, ts = motion(2)
    if fault == "normalized_input": net = net.with_turn_prior(0., normalise=True)
    else: net.succ_logprior[net.succ_table >= 0] = .5
    with pytest.raises(ValueError, match="unnormalised|unit legal"):
        j.align(net, od, st, ts, matches([(ts[-1], [candidate(net)])]),
                config=j.Config(particles=2, junction_prior="conditional-uniform-v3"))


@pytest.mark.parametrize("value", [True, None, "conditional", 1])
def test_turn_prior_choice_is_explicit_and_no_aliases(value):
    with pytest.raises(ValueError):
        j.Config(junction_prior=value).validate()
