"""Pure synthetic bank/roads/clocks only; no IO, actual cases, models or GT."""
import copy
import math
import networkx as nx
import numpy as np
import pytest
from monocular_osm import raw_vpr_appearance as a, raw_vpr_bank as b, raw_vpr_native as n
from monocular_osm import junction_sequence as j, poi_alignment as poi
from monocular_osm.graph_filter import RoadNetwork, Odometry, FilterConfig
from tests.test_raw_vpr_bank import fixture as bank_fixture, reseal, run as admit
from tests.test_junction_sequence import roads, matches


def case(equal=False):
    net = roads(two=True)
    times = 4. + np.arange(101, dtype=np.float64) * .01
    selected = np.linspace(0, 100, 80).astype(int)
    query_times = times[selected]
    base = bank_fixture(keys=("baseline/a",), names={"baseline/a": ["mapillary:0", "mapillary:1", "mapillary:2"]}, descriptors=False)
    comparison, records, facts, regions = base
    comparison["query"]["times"] = query_times.tolist()
    region = regions[records[0]["key"]]
    z, meta = region["arrays"], region["audit"]
    z["query_times"] = query_times.copy()
    z["ref_latlon"] = net.local_to_latlon(np.array([[500., 0.], [500., 5000.], [500., 2500.]]))
    z["raw_similarities"][:] = [.2, .2, .2] if equal else [.35, .35, .1]
    z["decoded_cosines"] = z["raw_similarities"][:, 0].copy()
    z["returned_cosines"] = z["decoded_cosines"].astype(np.float64)
    z["returned_latlon"] = z["ref_latlon"][z["decoded_ref_indices"]]
    meta["comparison_identity"]["query_times_sha256"] = n.typed(query_times)
    meta["comparison_identity_sha256"] = b.legacy.json_sha(meta["comparison_identity"])
    reseal(records[0], region)
    flat = facts["appearance_identity"]
    flat["query_times_header"] = copy.deepcopy(meta["array_artifact"]["arrays"]["query_times"])
    physical = facts["all_physical_calls_and_aliases"][0]
    physical["query_times_sha256"] = n.typed(query_times)
    physical["query_to_vo"] = n.query_vo_join(query_times, times, 1.)
    for i, row in enumerate(physical["query_to_vo"]):
        row.update(query_row=i, local_query_index=i, dense_index=int(selected[i]), native_ordinal=int(selected[i]))
    facts["signed_vo_clock_sha256"] = n.typed(times)
    facts["video_sha256"] = "d" * 64
    bank = admit(base)
    assert bank["status"] == "complete_research_bank", bank
    odom = Odometry(np.ones(100), np.zeros(100), np.diff(times), metric=False, unit_seconds=.01)
    status = np.full(100, "measured")
    motion = j.coarsen_motion(odom, status, times)
    return net, times, odom, status, motion, bank, facts


def test_genuine_field_parameters_raw_mode_and_multimodal_support():
    net, times, _, _, motion, bank, facts = case()
    appearance = a.prepare(bank, net, facts, times, motion.times)
    cfg = appearance.field.provenance["config"]
    assert [cfg[k] for k in ("field_cell_m", "appearance_sigma_m", "row_std_floor", "z_clip", "empty_appearance")] == [100.,150.,.005,6.,-6.]
    assert appearance.field.provenance["reference_mean"] is False
    local = np.array([[500., 0.], [500., 5000.], [500., 2500.]])
    logf = appearance.log_factor(0, net, local, time=bank["query_times"][0])
    assert logf[0] == pytest.approx(logf[1], abs=1e-12) and logf[0] > logf[2]
    assert math.exp(logf[2]) >= .1
    assert appearance.receipt["all_references"] == 3


def test_scalar_floor_mixture_oracle_and_nonzero_origin_crs():
    net, times, _, _, motion, bank, facts = case()
    app = a.prepare(bank, net, facts, times, motion.times)
    point = np.array([[500., 0.]])
    value = app.field.sample(net.local_to_crs(point)[:, None, :], np.array([bank["query_times"][0]]))[0,0]
    expected = math.log(.1 + .9 * math.exp(float(value) - app.maxima[0]))
    assert app.log_factor(0, net, point, time=bank["query_times"][0])[0] == pytest.approx(expected, abs=1e-15)
    assert np.linalg.norm(net.origin) > 1_000_000
    distant = np.array([[1e8, 1e8]])
    oracle = math.log(.1 + .9 * math.exp(-6. - app.maxima[0]))
    assert app.log_factor(0, net, distant, time=bank["query_times"][0])[0] == pytest.approx(oracle, abs=1e-15)
    with pytest.raises(ValueError, match="exact original"):
        app.log_factor(0, net, point, time=bank["query_times"][0] + 1e-10)


def test_zero_row_spread_uses_std_floor_without_nan():
    net, times, _, _, motion, bank, facts = case(equal=True)
    app = a.prepare(bank, net, facts, times, motion.times)
    assert np.isfinite(app.field.values).all()
    assert np.max(app.field.values) == pytest.approx(0., abs=1e-6)


def test_first_observation_maxspan_groups_and_duplicate_evidence_power():
    ids, powers = a.temporal_groups(np.arange(80, dtype=np.float64) * 20.)
    np.testing.assert_array_equal(ids, np.repeat(np.arange(40), 2))
    np.testing.assert_array_equal(powers, np.full(80, .5))
    ids, powers = a.temporal_groups(np.linspace(0., 29., 80))
    assert np.all(ids == 0) and np.sum(powers * math.log(.3)) == pytest.approx(math.log(.3))


@pytest.mark.parametrize("fault", ["clock", "step", "time", "lag", "model", "missing_alias", "composed_clock"])
def test_native_schedule_or_identity_mismatch_refuses(fault):
    net, times, _, _, motion, bank, facts = case()
    row = facts["all_physical_calls_and_aliases"][0]["query_to_vo"][2]
    if fault == "clock": facts["signed_vo_clock_sha256"] = "0" * 64
    if fault == "step": row["composed_step"] += 1
    if fault == "time": row["t_sec"] += .001
    if fault == "lag": row["composed_lag_s"] += .001
    if fault == "model": bank["appearance_identity"]["model"]["weights_sha256"] = "f" * 64
    if fault == "missing_alias": facts["all_physical_calls_and_aliases"] = []
    if fault == "composed_clock": motion.times[1] += .01
    with pytest.raises(ValueError): a.prepare(bank, net, facts, times, motion.times)


def test_whole_extent_memory_refuses_before_builder_instead_of_cropping(monkeypatch):
    net, times, _, _, motion, bank, facts = case()
    net.seg_xy0[-1, 0] = 1e8  # Private synthetic geometry, never a real map.
    monkeypatch.setattr(poi, "build_appearance_field", lambda *a, **k: pytest.fail("allocation before whole-extent cap"))
    with pytest.raises(ValueError, match="16M.*no crop"):
        a.prepare(bank, net, facts, times, motion.times)


def test_extent_contains_terminal_road_vertices_and_every_reference():
    graph = nx.MultiDiGraph(crs="EPSG:32632")
    graph.add_node(1, x=1000., y=1000.)
    graph.add_node(2, x=-1000., y=-2000.)
    graph.add_edge(1, 2)
    net = RoadNetwork.from_graph(graph)
    origin, shape, receipt = a.full_extent(net, np.array([[2000., 2500.]]))
    assert receipt["full_geometry_min_xy"] == pytest.approx([-1000., -2000.], abs=1e-9)
    assert receipt["full_geometry_max_xy"] == [2000., 2500.]
    assert np.all(origin <= [-1450., -2450.])
    assert shape[0] * shape[1] * 80 == receipt["field_values"]
    with pytest.raises(ValueError, match="16M"):
        a.full_extent(net, np.array([[1e8, 1e8]]))
    net.crs = "EPSG:4326"
    with pytest.raises(ValueError, match="projected metre"):
        a.full_extent(net, np.array([[1., 2.]]))


def test_initial_proposal_is_exit_times_first_power_and_retains_exact_p_over_q(monkeypatch):
    net, times, _, _, motion, bank, facts = case()
    app = a.prepare(bank, net, facts, times, motion.times)
    pf = j._OfflineFilter(net, FilterConfig(n_particles=32, seed=0, path_storage=False), motion, [], lambda: None)
    monkeypatch.setattr(pf, "guide", lambda e, s, k: np.full(np.shape(e), .4))
    nc = np.ceil(net.edge_len / 100.).astype(np.int64)
    edges = np.repeat(np.arange(net.n_edges), nc)
    starts = np.concatenate([np.arange(k) * (length / k) for k, length in zip(nc, net.edge_len)])
    lengths = np.repeat(net.edge_len / nc, nc)
    expected = .4 * app.initial_weight(net, edges, starts + lengths / 2.)
    original, seen = j.initial_proposal, []
    def proposal(actual_lengths, guide, defensive=.25):
        np.testing.assert_array_equal(guide, expected)
        q, ratio = original(actual_lengths, guide, defensive)
        np.testing.assert_allclose(q * np.exp(ratio), lengths / lengths.sum(), rtol=1e-14)
        seen.append(True)
        return q, ratio
    monkeypatch.setattr(j, "initial_proposal", proposal)
    pf.initialise_offline(app)
    assert seen == [True]
    np.testing.assert_array_equal(pf.guide(edges, starts, np.zeros(len(edges))), np.full(len(edges), .4))


def test_tiny_fused_full_50k_run_consumes_all80_once_and_preserves_ocr():
    net, times, odom, status, _, bank, facts = case()
    cue = matches([(times[-1], [])])  # Literal EXIT with no map match stays outlier-only.
    result = j.align_with_appearance(net, odom, status, times, cue, bank, facts)
    assert result["status"] == "complete_diagnostic_only", result
    events = result["appearance_events"]
    assert [row["query_row"] for row in events] == list(range(80))
    assert [row["t_sec"] for row in events] == bank["query_times"].tolist()
    assert all(0 <= row["composition_lag_s"] <= 1. for row in events)
    assert sum(row["power"] for row in events) == pytest.approx(1.)
    assert result["cues"][0]["no_matches_outlier_only"] is True
    assert result["cues"][0]["factor_min"] == result["cues"][0]["factor_max"] == .15
    assert result["config"]["particles"] == 50_000 and result["config"]["seed"] == 0
    assert result["effective_filter_config"]["normalise_turn_prior"] is True
    assert result["accuracy_admitted"] is False and result["selection"] is None


def test_default_align_never_builds_an_appearance_field(monkeypatch):
    net, times, odom, status, _, _, _ = case()
    monkeypatch.setattr(a, "prepare", lambda *x, **k: pytest.fail("default secretly fused appearance"))
    result = j.align(net, odom, status, times, matches([(times[-1], [])]),
                     config=j.Config(particles=60, seed=3, junction_prior="conditional-uniform-v3"))
    assert result["status"] == "complete_diagnostic_only"
    assert "appearance" not in result and "appearance_events" not in result
    assert result["profile"] == j.PROFILE_V3


@pytest.mark.parametrize("config", [j.Config(), j.Config(particles=10, junction_prior="conditional-uniform-v3"),
                                   j.Config(seed=1, junction_prior="conditional-uniform-v3")])
def test_fused_profile_cannot_silently_change_prior_particle_count_or_seed(config):
    with pytest.raises(ValueError, match="conditional-uniform-v3 / 50k / seed0"):
        j.align_with_appearance(None, None, None, None, None, None, None, config=config)
