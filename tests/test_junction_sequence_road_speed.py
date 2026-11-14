"""Private original run9 controls plus exact road-hook witnesses; no data/GT."""
from copy import deepcopy
import math
import networkx as nx
import numpy as np
import pytest

from monocular_osm import junction_sequence as j, road_speed_context as context, road_speed_kernel as kernel
from monocular_osm import road_segment_semantics as sem
from monocular_osm.graph_filter import GraphParticleFilter, RoadNetwork, FilterConfig, Odometry
from tests.test_reference_snapshot_future_guide_probe import case
from tests.test_junction_sequence import matches
from tests.test_junction_sequence_raw_query_veto import capture, assert_state

PAIRS = ((j.align_with_snapshot_future_guide, j.align_with_snapshot_future_guide_road_speed),
         (j.align_with_snapshot_normalized_heading_future_guide,
          j.align_with_snapshot_normalized_heading_future_guide_road_speed))


def context_for(net, mode="all"):
    # Reconstruct ONLY this invented fixture's known straight source graph.
    # This is not a production geometry-to-source join or admission shortcut.
    graph = nx.MultiDiGraph(crs=net.crs)
    for index, (u, v, key) in enumerate(net.edge_keys):
        lo, hi = net.seg_lo[index], net.seg_hi[index]
        assert hi - lo == 1
        start = net.seg_xy0[lo] + net.origin
        end = net.seg_xy0[lo] + net.seg_dir[lo] * net.edge_len[index] + net.origin
        graph.add_node(u, x=float(start[0]), y=float(start[1]))
        graph.add_node(v, x=float(end[0]), y=float(end[1]))
        tags = {"osmid": index + 1, "length": float(net.edge_len[index]), "highway": "primary"}
        if mode == "all" or (mode == "mixed" and index % 2): tags["maxspeed"] = "50"
        graph.add_edge(u, v, key=key, **tags)
    records = []; catalog = sem.attach(graph, emit_record=records.append)
    trace = []; rebuilt = RoadNetwork.from_graph(graph, construction_trace=trace)
    assert sem.net_binding(rebuilt) == sem.net_binding(net)
    for row in trace: row["source_keyed_path"] = graph.edges[row["edge_key"]][sem.TOKEN]
    packet = sem.bind_mapping(catalog, trace, rebuilt)
    return context.prepare(rebuilt, packet, iter(records), catalog_sha256=catalog.encoded_sha256,
                           catalog_bytes=catalog.encoded_bytes)


@pytest.mark.parametrize("old,new", PAIRS)
def test_all_neutral_full_parent_arrays_metadata_PF_and_RNG_exact(monkeypatch, old, new):
    net, times, odom, status, _, bank, facts = case("baseline")
    base, ctx = context_for(net, "none")
    filters = capture(monkeypatch); cue = matches([(times[-1], [])])
    reference = old(net, odom, status, times, cue, bank, facts)
    out = new(base, odom, status, times, cue, bank, facts, road_context=ctx)
    assert out["status"] == reference["status"] == "complete_diagnostic_only"
    assert_state(filters[0], filters[1])
    for name, value in reference.items():
        if isinstance(value, np.ndarray): np.testing.assert_array_equal(out[name], value)
    for name in ("appearance", "future_guide", "cues", "appearance_events", "effective_filter_config"):
        assert out[name] == reference[name]
    assert out["road_model_eligible"][1:].all()
    assert not out["road_context_count"].any() and not out["road_component_count"].any()
    assert np.all(out["road_witness_uniform"] == -1.)


def replay(out, pf):
    motion = pf.motion; p = pf.cfg.n_particles
    for row, step in enumerate(out["road_witness_steps"]):
        step = int(step); dt = float(motion.times[step] - motion.times[step - 1])
        rng = np.random.default_rng(); rng.bit_generator.state = deepcopy(out["road_speed_witness_rng_before"][row])
        before = out["road_witness_logk_before"][row]
        lo, hi = pf._k_bounds; margin = pf.cfg.logk_bounds_margin
        after = np.clip(before + rng.normal(0., pf.cfg.logk_rw * math.sqrt(dt), p), lo - margin, hi + margin)
        np.testing.assert_array_equal(after, out["road_witness_logk_after"][row])
        a = math.exp(-dt / 10.)
        innovation = rng.normal(0., .3 * math.sqrt(1. - a * a), p)
        np.testing.assert_array_equal(innovation, out["road_witness_innovation"][row])
        previous = pf._road_context.lookup_previous(pf.net,
            out["road_witness_previous_edge"][row], out["road_witness_previous_offset_m"][row])
        eligible = previous["eligible"] & bool(out["road_model_eligible"][step])
        uniform = np.full(p, -1., np.float64)
        ids = np.flatnonzero(eligible) if a < 1. else np.empty(0, np.int64)
        if len(ids): uniform[ids] = rng.random(len(ids))
        np.testing.assert_array_equal(uniform, out["road_witness_uniform"][row])
        mean = a * out["road_witness_eps_before"][row]
        expected = mean + innovation
        road = ids[uniform[ids] >= .25]
        if len(road):
            center = np.log(previous["mps"][road]) - after[road] - (math.log(float(motion.step[step - 1])) - math.log(dt))
            expected[road] = mean[road] + (1. - a) * center + innovation[road] * math.sqrt((.3**2 + .75**2) / .3**2)
        np.testing.assert_array_equal(expected, out["road_witness_eps_after"][row])
        assert out["road_context_count"][step] == np.count_nonzero(eligible)
        assert out["road_component_count"][step] == len(road)


@pytest.mark.parametrize("bank_key", ["baseline", "expanded"])
@pytest.mark.parametrize("old,new", PAIRS)
def test_four_fixed_controls_full50k_witness_exact_replay_and_original80_events(monkeypatch, bank_key, old, new):
    net, times, odom, status, _, bank, facts = case(bank_key)
    base, ctx = context_for(net)
    filters = capture(monkeypatch)
    out = new(base, odom, status, times, matches([(times[-1], [])]), bank, facts, road_context=ctx)
    assert out["status"] == "complete_diagnostic_only"
    pf = filters[-1]; replay(out, pf)
    assert out["road_witness_previous_edge"].dtype == np.int32
    assert out["road_witness_previous_edge"].shape[1] == 50_000
    for name, value in out.items():
        if name.startswith("road_witness_") and name not in {"road_witness_steps", "road_witness_previous_edge"}:
            assert value.dtype == np.float64 and value.shape == out["road_witness_previous_edge"].shape
    assert len(out["appearance_events"]) == 80
    assert [r["query_row"] for r in out["appearance_events"]] == list(range(80))
    np.testing.assert_array_equal(out["motion_counts"], pf.motion.counts)
    assert out["profile"]["motion_target_changed"] is True
    assert out["profile"]["proposal_only"] is False
    assert out["profile"]["appearance_target_unchanged"] is True
    assert out["profile"]["future_guide_is_proposal_only"] is True
    assert out["road_speed_completed_prediction_steps"] == len(pf.motion.step)
    assert out["road_speed_epsilon_steps_computed"] == len(pf.motion.step)
    np.testing.assert_array_equal(out["road_witness_steps"], kernel.witness_steps(len(pf.motion.step)))


class UnitFuture:
    def sample(self, index, net, edge, offset):
        return np.ones(np.broadcast_shapes(np.shape(edge), np.shape(offset)))


def low_level(total=6, particles=50_000, statuses=None, tiny=False, mode="all"):
    net = case("baseline")[0]
    base, ctx = context_for(net, mode)
    runtime = ctx.bind_runtime(base.with_turn_prior(0., normalise=True))
    dt = 1e-300 if tiny else .5
    statuses = ["measured"] * total if statuses is None else statuses
    steps = np.array([1. if s == "measured" else 0. for s in statuses])
    odom = Odometry(steps, np.zeros(total), np.full(total, dt), metric=False, unit_seconds=dt)
    times = np.arange(total + 1, dtype=np.float64) * dt
    motion = j.coarsen_motion(odom, statuses, times)
    cfg = FilterConfig(n_particles=particles, seed=0, path_storage=False,
                       turn_prior_kappa=0., normalise_turn_prior=True)
    pf = j._RoadSpeedFutureGuideFilter(runtime, cfg, motion, [], lambda: None)
    pf.install_road_context(ctx); pf.future_guide = UnitFuture(); pf.initialise_offline()
    return pf, ctx, cfg


def test_all_five_predetermined_steps_capture_every50000_before_position_advance():
    pf, _, _ = low_level()
    for t in range(1, len(pf.motion.step) + 1):
        pf.index = t
        pf.predict(float(pf._road_schedule["parent_step"][t - 1]), 0., float(pf._road_schedule["dt"][t - 1]))
    out = pf.road_speed_exports()
    assert out["road_witness_steps"].shape == (5,)
    assert out["road_witness_previous_edge"].shape == (5, 50_000)
    assert out["road_speed_witness_payload_bytes"] <= 64 << 20
    replay(out, pf)


@pytest.mark.parametrize("status,tiny", [("unknown", False), ("stationary", False), ("measured", True)])
def test_unknown_stationary_and_a1_degeneracy_preserve_parent_transition_RNG(status, tiny):
    pf, _, cfg = low_level(total=1, particles=64, statuses=[status], tiny=tiny)
    parent = j._SnapshotFutureGuideFilter(pf.net, cfg, pf.motion, [], lambda: None)
    parent.future_guide = UnitFuture(); parent.initialise_offline()
    for target in (parent, pf):
        target.index = 1
        target.predict(float(pf._road_schedule["parent_step"][0]), 0., float(pf._road_schedule["dt"][0]))
    assert_state(parent, pf)
    out = pf.road_speed_exports()
    assert not out["road_component_count"].any()
    assert np.all(out["road_witness_uniform"] == -1.)
    assert out["road_model_eligible"][1] == (status == "measured")
    if tiny: assert out["road_context_count"][1] == 64
    replay(out, pf)


def test_mixed_neutral_current_epsilon_exact_without_later_path_claim(monkeypatch):
    net, times, odom, status, _, bank, facts = case("baseline")
    base, ctx = context_for(net, "mixed")
    filters = capture(monkeypatch)
    out = j.align_with_snapshot_future_guide_road_speed(base, odom, status, times, matches([(times[-1], [])]), bank, facts, road_context=ctx)
    pf = filters[-1]; replay(out, pf)
    assert out["status"] == "complete_diagnostic_only"
    for row, t in enumerate(out["road_witness_steps"]):
        a = math.exp(-float(pf._road_schedule["dt"][int(t) - 1]) / 10.)
        neutral = out["road_witness_uniform"][row] == -1.
        assert neutral.any() and not neutral.all()
        np.testing.assert_array_equal(out["road_witness_eps_after"][row, neutral],
            a * out["road_witness_eps_before"][row, neutral] + out["road_witness_innovation"][row, neutral])


@pytest.mark.parametrize("entry", [p[1] for p in PAIRS])
@pytest.mark.parametrize("change", [{"particles": 10}, {"seed": 1}, {"junction_prior": "legacy"}])
def test_fixed_entries_reject_nonfixed_profile_before_dispatch(monkeypatch, entry, change):
    monkeypatch.setattr(j, "_align", lambda *a, **k: pytest.fail("nonfixed dispatch"))
    cfg = {"particles": 50_000, "seed": 0, "junction_prior": "conditional-uniform-v3", **change}
    with pytest.raises(ValueError): entry(*([None] * 7), road_context=None, config=j.Config(**cfg))


@pytest.mark.parametrize("change", [{"common_null": True}, {"adaptive_resampling": True},
    {"high_particle_budget": True}, {"no_adaptive_heading_q": True}, {"raw_query_veto": True},
    {"joint_initial_proposal": True}, {"compatible_appearance": True}, {"future_proposal": False}])
def test_road_mode_refuses_other_studies_before_motion_or_input_values(monkeypatch, change):
    base, ctx = context_for(case("baseline")[0])
    monkeypatch.setattr(j, "coarsen_motion", lambda *a: pytest.fail("motion before mode guard"))
    flags = {"snapshot_appearance": True, "future_proposal": True, "road_speed": True,
             "road_context": ctx, "appearance_input": (None, None), **change}
    with pytest.raises(ValueError):
        j._align(base, None, None, None, None, config=j.Config(junction_prior="conditional-uniform-v3"), **flags)


def test_early_unavailable_never_invents_step_or_witness_exports(monkeypatch):
    net, times, odom, status, _, bank, facts = case("baseline"); base, ctx = context_for(net)
    monkeypatch.setattr(j, "coarsen_motion", lambda *a: (_ for _ in ()).throw(j.Unavailable("private early")))
    out = j.align_with_snapshot_future_guide_road_speed(base, odom, status, times, matches([(times[-1], [])]), bank, facts, road_context=ctx)
    assert out["status"] == "unavailable" and out["route_latlon"] is None
    assert "road_witness_steps" not in out and "road_context_count" not in out


def test_later_turn_failure_preserves_completed_epsilon_witness_and_honest_prefix(monkeypatch):
    net, times, odom, status, _, bank, facts = case("baseline"); base, ctx = context_for(net)
    original = j._OfflineFilter._choose_successors
    def stop(self, *args, **kwargs):
        if isinstance(self, j._RoadSpeedMixin) and self.index == 1:
            raise j.Unavailable("private after epsilon, during original turn")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(j._OfflineFilter, "_choose_successors", stop)
    filters = capture(monkeypatch)
    out = j.align_with_snapshot_future_guide_road_speed(base, odom, status, times, matches([(times[-1], [])]), bank, facts, road_context=ctx)
    assert out["status"] == "unavailable" and out["route_latlon"] is None
    assert out["road_speed_completed_prediction_steps"] == 0
    assert out["road_speed_epsilon_steps_computed"] == 1
    np.testing.assert_array_equal(out["road_witness_steps"], [1])
    assert not out["road_context_count"][2:].any()
    replay(out, filters[-1])


def test_witness_capacity_refuses_before_witness_allocations(monkeypatch):
    pf, ctx, cfg = low_level(total=1, particles=50_000)
    other = j._RoadSpeedFutureGuideFilter(pf.net, cfg, pf.motion, [], lambda: None)
    monkeypatch.setattr(kernel, "witness_steps", lambda t: np.arange(1000, dtype=np.int32))
    monkeypatch.setattr(j.np, "empty", lambda *a, **k: pytest.fail("witness allocation before cap"))
    with pytest.raises(j.Unavailable, match="64MiB"): other.install_road_context(ctx)
