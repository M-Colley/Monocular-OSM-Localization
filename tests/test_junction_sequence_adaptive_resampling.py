"""Private target-ESS/lineage/RNG controls; no cases, labels or model inputs."""
import numpy as np
import pytest
from monocular_osm import junction_sequence as j, adaptive_auxiliary_resampling as r
from monocular_osm.graph_filter import GraphParticleFilter
from tests.test_reference_snapshot_junction_probe_v2 import case
from tests.test_junction_sequence import matches


PAIRS = ((j.align_with_snapshot_future_guide, j.align_with_snapshot_future_guide_adaptive_resampling),
         (j.align_with_snapshot_normalized_heading_future_guide,
          j.align_with_snapshot_normalized_heading_future_guide_adaptive_resampling))
STATE = ("edge", "offset", "logk", "eps", "delta", "delta_var", "x", "y", "road_heading", "prev_head", "logw")


def capture_filters(monkeypatch):
    instances, original = [], GraphParticleFilter.__init__
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        instances.append(self)
    monkeypatch.setattr(GraphParticleFilter, "__init__", init)
    return instances


def assert_identical_state_and_route(ref, new, first, second):
    for name in STATE: np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.rng.bit_generator.state == second.rng.bit_generator.state
    for name, value in ref.items():
        if isinstance(value, np.ndarray): np.testing.assert_array_equal(new[name], value)
    assert new["appearance_events"] == ref["appearance_events"]
    assert new["cues"] == ref["cues"]


@pytest.mark.parametrize("old_entry,new_entry", PAIRS)
def test_no_trigger_preserves_every_state_weight_rng_and_uses_no_aux_guide(monkeypatch, old_entry, new_entry):
    net, times, odom, status, _, bank, facts = case(versions=True)
    instances = capture_filters(monkeypatch)
    aux_calls = []
    def identity(pf, weights, guide):
        aux_calls.append(True)
        return np.arange(len(weights), dtype=np.int32)
    monkeypatch.setattr(j, "auxiliary_resample", identity)
    cue = matches([(times[-1], [])])
    ref = old_entry(net, odom, status, times, cue, bank, facts)
    old_aux_count = len(aux_calls)
    original_guide, decisions = j._SnapshotFutureGuideFilter.guide, []
    def no_aux_guide(self, edge, offset, logk):
        assert np.shape(edge) != (50_000,), "guide computed on no-trigger auxiliary branch"
        return original_guide(self, edge, offset, logk)
    def never(weights):
        assert weights.dtype == np.float64 and weights.shape == (50_000,)
        decisions.append(weights.copy())
        return {"resample": False}
    monkeypatch.setattr(j._SnapshotFutureGuideFilter, "guide", no_aux_guide)
    monkeypatch.setattr(r, "decision", never)
    new = new_entry(net, odom, status, times, cue, bank, facts)
    assert new["status"] == ref["status"] == "complete_diagnostic_only"
    assert len(aux_calls) == old_aux_count == len(decisions)
    assert_identical_state_and_route(ref, new, instances[0], instances[1])
    assert new["resampled_after_step"].dtype == np.bool_
    assert not new["resampled_after_step"].any()
    np.testing.assert_array_equal(new["initial_ancestor_count"], np.full(len(new["times"]), 50_000, np.int32))
    assert "resampled_after_step" not in ref and "initial_ancestor_count" not in ref
    assert {k for k, v in new.items() if isinstance(v, np.ndarray)} - {
        k for k, v in ref.items() if isinstance(v, np.ndarray)} == {"resampled_after_step", "initial_ancestor_count"}


@pytest.mark.parametrize("old_entry,new_entry", PAIRS)
def test_always_trigger_matches_genuine_old_aux_corrections_and_rng_and_label_oracle(monkeypatch, old_entry, new_entry):
    net, times, odom, status, _, bank, facts = case(versions=True)
    instances = capture_filters(monkeypatch)
    cue = matches([(times[-1], [])])
    ref = old_entry(net, odom, status, times, cue, bank, facts)
    original_aux, parents, order = j.auxiliary_resample, [], []
    def always(weights):
        assert np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)
        order.append("decision")
        return {"resample": True}
    def aux(pf, weights, guide):
        assert order[-1] == "decision"
        result = original_aux(pf, weights, guide)
        parents.append(result.copy()); order.append("resampled")
        return result
    monkeypatch.setattr(r, "decision", always)
    monkeypatch.setattr(j, "auxiliary_resample", aux)
    new = new_entry(net, odom, status, times, cue, bank, facts)
    assert_identical_state_and_route(ref, new, instances[0], instances[1])
    expected, labels = [50_000], np.arange(50_000, dtype=np.int32)
    for chosen in parents:
        labels = labels[chosen]
        expected.append(np.count_nonzero(np.bincount(labels, minlength=50_000)))
    np.testing.assert_array_equal(new["initial_ancestor_count"], np.asarray(expected, np.int32))
    np.testing.assert_array_equal(new["resampled_after_step"], [True] * len(parents) + [False])
    assert new["profile"]["native_heading_variance_unchanged"] is True
    assert not {"effective_native_heading_var", "composed_heading_var"} & new.keys()


def test_real_decision_uses_target_weights_before_even_a_peaked_aux_guide(monkeypatch):
    net, times, odom, status, _, bank, facts = case(versions=True)
    instances = capture_filters(monkeypatch)
    actual_decision, actual_guide = r.decision, j._SnapshotFutureGuideFilter.guide
    decisions, guide_steps = [], []
    def decision(weights):
        pf = instances[-1]
        np.testing.assert_array_equal(weights, np.exp(pf.logw))
        result = actual_decision(weights)
        assert result["resample"] == (result["ess"] < .5 * result["particle_count"])
        decisions.append((pf.index, result))
        return result
    def guide(self, edge, offset, logk):
        if np.shape(edge) == (50_000,):
            assert decisions[-1][0] == self.index and decisions[-1][1]["resample"]
            guide_steps.append(self.index)
            peaked = np.full(50_000, 1e-6); peaked[0] = 1.
            return peaked
        return actual_guide(self, edge, offset, logk)
    monkeypatch.setattr(r, "decision", decision)
    monkeypatch.setattr(j._SnapshotFutureGuideFilter, "guide", guide)
    result = PAIRS[0][1](net, odom, status, times, matches([(times[-1], [])]), bank, facts)
    assert result["status"] == "complete_diagnostic_only"
    assert decisions[0][1]["resample"] is False and 0 not in guide_steps
    np.testing.assert_array_equal(result["resampled_after_step"], [d["resample"] for _, d in decisions] + [False])
    assert result["initial_ancestor_count"].dtype == np.int32
    assert np.all(np.diff(result["initial_ancestor_count"]) <= 0)


@pytest.mark.parametrize("entry", [pair[1] for pair in PAIRS])
@pytest.mark.parametrize("change", [{"particles": 100}, {"seed": 1}, {"junction_prior": "legacy"}])
def test_explicit_entries_refuse_nonfixed_sampler_before_dispatch(monkeypatch, entry, change):
    monkeypatch.setattr(j, "_align", lambda *a, **k: pytest.fail("nonfixed sampler entered"))
    cfg = j.Config(**{"junction_prior": "conditional-uniform-v3", **change})
    with pytest.raises(ValueError): entry(None, None, None, None, None, None, None, config=cfg)


@pytest.mark.parametrize("entry", [pair[0] for pair in PAIRS] + [
    j.align_with_snapshot_common_null_appearance, j.align_with_snapshot_common_null_normalized_heading,
    j.align_with_snapshot_future_guide_no_adaptive_q,
    j.align_with_snapshot_normalized_heading_future_guide_no_adaptive_q])
def test_old_entries_do_not_enable_new_resampling_policy(monkeypatch, entry):
    seen = []
    monkeypatch.setattr(j, "_align", lambda *a, **k: seen.append(k))
    entry(None, None, None, None, None, None, None)
    assert len(seen) == 1 and seen[0].get("adaptive_resampling", False) is False


@pytest.mark.parametrize("invalid", [{"future_proposal": False}, {"common_null": True},
    {"compatible_appearance": True}, {"no_adaptive_heading_q": True}])
def test_resampling_mode_does_not_change_other_target_contracts(monkeypatch, invalid):
    monkeypatch.setattr(j, "coarsen_motion", lambda *a: pytest.fail("invalid policy reached motion"))
    options = dict(appearance_input=(None, None), snapshot_appearance=True, future_proposal=True, adaptive_resampling=True)
    options.update(invalid)
    with pytest.raises(ValueError): j._align(None, None, None, None, None, **options)
