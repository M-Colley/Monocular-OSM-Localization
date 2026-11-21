"""Private joint-measure/witness tests; no real maps, video, models or GT."""
from copy import deepcopy
import math
import weakref
import numpy as np
import pytest
from monocular_osm import junction_sequence as j, joint_initial_proposal as core
from monocular_osm import raw_vpr_appearance as appearance
from tests.test_reference_snapshot_junction_probe_v2 import case
from tests.test_junction_sequence import matches, candidate

PAIRS = ((j.align_with_snapshot_future_guide, j.align_with_snapshot_future_guide_joint_initial_proposal),
         (j.align_with_snapshot_normalized_heading_future_guide,
          j.align_with_snapshot_normalized_heading_future_guide_joint_initial_proposal))
WITNESS = {"initial_speed_representatives_mps": (np.float64, (16,)),
    "initial_speed_log_boundaries": (np.float64, (15,)), "initial_speed_bin_counts": (np.int32, (16,)),
    **{key: (np.int32, (50_000,)) for key in ("initial_particle_cell", "initial_particle_speed_bin", "initial_particle_edge")},
    **{key: (np.float64, (50_000,)) for key in ("initial_particle_offset_m", "initial_particle_logk", "initial_particle_eps", "initial_particle_logw")}}


def partition(net):
    counts = np.ceil(net.edge_len / 100.).astype(np.int64)
    edge = np.repeat(np.arange(net.n_edges), counts)
    start = np.concatenate([np.arange(n) * length/n for n, length in zip(counts, net.edge_len)])
    length = np.repeat(net.edge_len/counts, counts)
    return edge, start, length


@pytest.mark.parametrize("old,entry", PAIRS)
def test_genuine_prior_first_exact_conditional_weights_and_pre_target_witness(monkeypatch, old, entry):
    net, times, odom, status, _, bank, facts = case(versions=True)
    initial, prior_draws, proposals, g0_calls = [], [], [], []
    original_init = j._JointInitialProposalMixin.initialise_offline
    original_q, original_g0 = j.initial_proposal, appearance.Appearance.initial_weight
    previous_q, previous_guide = [], []
    def proposal(length, guide, *args, **kwargs):
        if previous_q:
            assert previous_q[-1]() is None and previous_guide[-1]() is None, "previous conditional layer retained"
        q, ratio = original_q(length, guide, *args, **kwargs)
        proposals.append((length.copy(), guide.copy(), q.copy(), ratio.copy()))
        previous_q.append(weakref.ref(q)); previous_guide.append(weakref.ref(guide))
        return q, ratio
    def g0(self, *args, **kwargs):
        value = original_g0(self, *args, **kwargs)
        g0_calls.append(value.copy())
        return value
    def initialise(self, app=None):
        oracle = np.random.default_rng()
        oracle.bit_generator.state = deepcopy(self.rng.bit_generator.state)
        lo, hi = self._k_bounds
        expected_k = np.clip(oracle.normal(math.log(self.cfg.speed_median_mps*self.unit_seconds),
            self.cfg.speed_log_sigma, self.cfg.n_particles), lo, hi)
        expected_eps = oracle.normal(0., self._dev_sigma, self.cfg.n_particles)
        prior_draws.append((expected_k, expected_eps))
        original_init(self, app)
        initial.append({key: value.copy() for key, value in self.joint_initial_witness.items()})
    monkeypatch.setattr(j, "initial_proposal", proposal)
    monkeypatch.setattr(appearance.Appearance, "initial_weight", g0)
    monkeypatch.setattr(j._JointInitialProposalMixin, "initialise_offline", initialise)
    result = entry(net, odom, status, times, matches([(times[-1], [candidate(net, edge=0)])]), bank, facts)
    assert result["status"] == "complete_diagnostic_only"
    assert len(initial) == len(g0_calls) == 1
    assert previous_q[-1]() is None and previous_guide[-1]() is None
    for name, (dtype, shape) in WITNESS.items():
        value = result[name]
        assert value.dtype == dtype and value.shape == shape and value.flags.c_contiguous
        np.testing.assert_array_equal(value, initial[0][name])
    np.testing.assert_array_equal(result["initial_particle_logk"], prior_draws[0][0])
    np.testing.assert_array_equal(result["initial_particle_eps"], prior_draws[0][1])
    bins = core.bin_indices(result["initial_particle_logk"], result["initial_particle_eps"], odom.unit_seconds)
    np.testing.assert_array_equal(result["initial_particle_speed_bin"], bins)
    np.testing.assert_array_equal(result["initial_speed_bin_counts"], np.bincount(bins, minlength=16).astype(np.int32))
    edges, starts, lengths = partition(net)
    selected = result["initial_particle_cell"]
    np.testing.assert_array_equal(result["initial_particle_edge"], edges[selected].astype(np.int32))
    assert np.all(result["initial_particle_offset_m"] >= starts[selected])
    assert np.all(result["initial_particle_offset_m"] <= starts[selected]+lengths[selected])
    occupied = np.flatnonzero(result["initial_speed_bin_counts"])
    assert len(proposals) == len(occupied)
    assert any(not np.array_equal(proposals[0][1], row[1]) for row in proposals[1:]), "speed never conditioned the guide"
    for b, (length, guide, q, ratio) in zip(occupied, proposals):
        p = length/length.sum()
        np.testing.assert_allclose(q, .25*p+.75*p*guide/np.sum(p*guide), rtol=1e-15, atol=0)
        ids = np.flatnonzero(bins == b)
        np.testing.assert_array_equal(result["initial_particle_logw"][ids], ratio[selected[ids]])
    assert result["initial_particle_logw"].max() <= math.log(4.)+1e-12
    assert np.any(np.exp(result["initial_particle_logk"]+result["initial_particle_eps"])/odom.unit_seconds > 55.)
    receipt = result["joint_initial_proposal_receipt"]
    assert receipt["particle_count"] == 50_000 and receipt["cell_count"] == len(lengths)
    assert receipt["occupied_bin_count"] == len(occupied) and sum(receipt["bin_counts"]) == 50_000
    assert receipt["mode"] == core.MODE and receipt["profile_sha256"] == core.PROFILE_SHA256
    assert receipt["G0_computed_once"] is True and receipt["witness_before_any_target_factor"] is True
    assert result["profile"]["proposal_only"] is True and result["profile"]["target_changed"] is False
    assert result["profile"]["appearance_target_unchanged"] is True
    assert [row["query_row"] for row in result["appearance_events"]] == list(range(80))
    assert not {"raw_query_enabled", "resampled_after_step", "effective_native_heading_var"} & result.keys()


@pytest.mark.parametrize("entry", [p[1] for p in PAIRS])
@pytest.mark.parametrize("change", [{"particles": 100}, {"seed": 1}, {"junction_prior": "legacy"}])
def test_fixed_entries_refuse_nonfixed_configuration_before_dispatch(monkeypatch, entry, change):
    monkeypatch.setattr(j, "_align", lambda *a, **k: pytest.fail("nonfixed dispatched"))
    cfg = j.Config(**{"junction_prior": "conditional-uniform-v3", **change})
    with pytest.raises(ValueError):
        entry(None, None, None, None, None, None, None, config=cfg)


@pytest.mark.parametrize("change", [{"appearance_input": None}, {"snapshot_appearance": False},
    {"future_proposal": False}, {"compatible_appearance": True}, {"common_null": True},
    {"raw_query_veto": True}, {"no_adaptive_heading_q": True}, {"adaptive_resampling": True},
    {"high_particle_budget": True}, {"joint_initial_proposal": 1}])
def test_only_original_run9_target_is_compatible_before_values(monkeypatch, change):
    monkeypatch.setattr(j, "coarsen_motion", lambda *a: pytest.fail("incompatible input values"))
    args = dict(config=j.Config(junction_prior="conditional-uniform-v3"),
        appearance_input=(None, None), snapshot_appearance=True, future_proposal=True, joint_initial_proposal=True)
    args.update(change)
    with pytest.raises(ValueError):
        j._align(None, None, None, None, None, **args)


def test_completed_witness_survives_later_failure_unchanged(monkeypatch):
    net, times, odom, status, _, bank, facts = case(versions=True)
    def fail(*a, **k):
        raise j.Unavailable("private subsequent prediction bound")
    monkeypatch.setattr(j.GraphParticleFilter, "predict", fail)
    result = PAIRS[0][1](net, odom, status, times, matches([(times[-1], [])]), bank, facts)
    assert result["status"] == "unavailable" and result["route_latlon"] is None
    assert all(name in result for name in WITNESS)
    assert result["joint_initial_proposal_receipt"]["witness_before_any_target_factor"] is True


def test_early_failure_never_fabricates_witness(monkeypatch):
    def fail(*a, **k):
        raise j.Unavailable("private complete motion bound")
    monkeypatch.setattr(j, "coarsen_motion", fail)
    result = PAIRS[0][1](None, None, None, None, None, None, None)
    assert result["status"] == "unavailable"
    assert not set(WITNESS) & result.keys() and "joint_initial_proposal_receipt" not in result


@pytest.mark.parametrize("kind", ["cells", "scratch"])
def test_full_partition_reserve_refuses_before_prior_or_guide(monkeypatch, kind):
    from types import SimpleNamespace
    pf = j._JointInitialFutureGuideFilter.__new__(j._JointInitialFutureGuideFilter)
    pf.index = 0
    pf.cfg = SimpleNamespace(n_particles=50_000)
    pf.net = SimpleNamespace(n_edges=1, edge_len=np.array([100.*(core.MAX_CELLS+1) if kind == "cells" else 100.]))
    pf._sample_logk = lambda *a: pytest.fail("prior draw before reserve")
    if kind == "scratch":
        monkeypatch.setattr(core, "SCRATCH_BYTES", 1)
    with pytest.raises(j.Unavailable, match="reserve"):
        pf.initialise_offline(object())
    assert not hasattr(pf, "joint_initial_witness")


def test_all_initial_bins_reuse_one_genuine_future_active_layer(monkeypatch):
    net, times, odom, status, _, bank, facts = case(versions=True)
    original = j._JointInitialProposalMixin.initialise_offline
    observed = []
    def init(self, app=None):
        before = self.future_guide.gaussian_calls
        original(self, app)
        observed.append((self.future_guide.gaussian_calls-before, self.future_guide._index,
                         self.joint_initial_receipt["occupied_bin_count"]))
    monkeypatch.setattr(j._JointInitialProposalMixin, "initialise_offline", init)
    result = PAIRS[0][1](net, odom, status, times, matches([(times[-1], [])]), bank, facts)
    assert result["status"] == "complete_diagnostic_only"
    assert len(observed) == 1 and observed[0][0:2] == (1, 0) and observed[0][2] > 1


def test_exact_joint_expectation_uses_conditional_not_marginal_position_q():
    lengths = np.array([1., 2., 3.])
    speed_mass = np.array([.3, .7])
    guides = (np.array([8., 1., .5]), np.array([.1, 1., 20.]))
    q, ratio = zip(*(j.initial_proposal(lengths, g) for g in guides))
    q, ratio = np.asarray(q), np.asarray(ratio)
    p = lengths/lengths.sum()
    np.testing.assert_allclose(speed_mass[:, None]*q*np.exp(ratio), speed_mass[:, None]*p, rtol=1e-15)
    test_function = np.array([[2., -1., .5], [-3., 4., 1.]])
    correct = np.sum(speed_mass[:, None]*q*np.exp(ratio)*test_function)
    expected = np.sum(speed_mass[:, None]*p*test_function)
    assert correct == pytest.approx(expected, abs=1e-15)
    marginal_q = np.sum(speed_mass[:, None]*q, axis=0)
    wrong = np.sum(speed_mass[:, None]*q*(p/marginal_q)*test_function)
    assert abs(wrong-expected) > .1
