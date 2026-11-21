"""Private variance-policy integration; no actual motion/assets/models/labels."""
import math
import numpy as np
import pytest
from monocular_osm import junction_sequence as j, no_adaptive_heading_variance as q
from monocular_osm import common_null_appearance as null_core, reference_snapshot_appearance as appearance
from monocular_osm.graph_filter import GraphParticleFilter
from tests.test_reference_snapshot_junction_probe_v2 import case
from tests.test_junction_sequence import matches


ENTRIES = (j.align_with_snapshot_future_guide_no_adaptive_q,
           j.align_with_snapshot_normalized_heading_future_guide_no_adaptive_q)


@pytest.mark.parametrize("entry", ENTRIES)
def test_clone_zero_variance_precedes_real_coarsen_without_mean_clock_or_target_change(monkeypatch, entry):
    net, times, source, status, _, bank, facts = case(versions=True)
    source.heading_var = np.linspace(.002, .01, len(source), dtype=np.float64)
    original = {k: getattr(source, k).copy() for k in ("step", "heading", "dt", "frame_index", "heading_var")}
    expected = j.coarsen_motion(source, status, times)
    real_coarsen, real_predict = j.coarsen_motion, GraphParticleFilter.predict
    seen, predicted = [], []
    def coarsen(effective, actual_status, clock):
        assert effective is not source and actual_status is status and clock is times
        for k in ("step", "heading", "dt", "frame_index"):
            np.testing.assert_array_equal(getattr(effective, k), original[k])
            assert not np.shares_memory(getattr(effective, k), getattr(source, k))
        np.testing.assert_array_equal(effective.heading_var, np.zeros(len(source)))
        motion = real_coarsen(effective, actual_status, clock)
        for k in ("times", "step", "heading", "unknown_seconds", "stationary_seconds", "native_end_indices", "counts", "se2"):
            np.testing.assert_array_equal(getattr(motion, k), getattr(expected, k))
        seen.append(motion)
        return motion
    def predict(self, step, heading, dt, **kwargs):
        predicted.append(kwargs["heading_var"])
        return real_predict(self, step, heading, dt, **kwargs)
    monkeypatch.setattr(j, "coarsen_motion", coarsen)
    monkeypatch.setattr(GraphParticleFilter, "predict", predict)
    monkeypatch.setattr(null_core, "build_common_null_field", lambda *a, **k: pytest.fail("run10 target used"))
    result = entry(net, source, status, times, matches([(times[-1], [])]), bank, facts)
    assert result["status"] == "complete_diagnostic_only" and len(seen) == 1
    assert all(v == 0. for v in predicted)
    for k, value in original.items(): np.testing.assert_array_equal(getattr(source, k), value)
    np.testing.assert_array_equal(result["effective_native_heading_var"], np.zeros(len(source)))
    np.testing.assert_array_equal(result["composed_heading_var"], seen[0].heading_var)
    assert np.any(expected.heading_var > 0.)
    assert result["profile"]["appearance_profile_sha256"] == appearance.PROFILE_SHA256
    assert result["profile"]["variance_mode"] == "no-adaptive-extra-Q-v1"
    assert result["profile"]["heading_variance_profile_sha256"] == q.PROFILE_SHA256
    assert result["profile"]["proposal_only"] is False
    assert result["profile"]["future_guide_is_proposal_only"] is True
    assert [r["query_row"] for r in result["appearance_events"]] == list(range(80))
    assert result["future_guide"]["target_changed"] is False
    cfg = result["effective_filter_config"]
    assert (cfg["delta_rw_deg"], cfg["heading_sigma_deg"], cfg["heading_outlier"]) == (2., 15., .2)


def test_mixed_unknown_stationary_means_and_SE2_unchanged_unknown_Q_retained(monkeypatch):
    net, times, source, status, _, bank, facts = case()
    status = status.astype("U10")
    source.heading_var = np.full(len(source), .03)
    status[7:13] = "unknown"; source.heading[7:13] = np.nan
    status[25:32] = "stationary"; source.step[25:32] = 0.
    before = j.coarsen_motion(source, status, times)
    original_status, original_time = status.copy(), times.copy()
    real = j.coarsen_motion
    class Checked(Exception): pass
    def check_clone(effective, actual_status, actual_times):
        motion = real(effective, actual_status, actual_times)
        for name in ("times", "step", "heading", "unknown_seconds", "stationary_seconds", "native_end_indices", "counts", "se2"):
            np.testing.assert_array_equal(getattr(motion, name), getattr(before, name))
        np.testing.assert_array_equal(actual_status, original_status)
        np.testing.assert_array_equal(actual_times, original_time)
        np.testing.assert_allclose(motion.heading_var, math.radians(5.)**2 * motion.unknown_seconds, rtol=1e-15)
        assert np.any(motion.heading_var > 0.) and not np.shares_memory(effective.heading_var, source.heading_var)
        raise Checked
    monkeypatch.setattr(j, "coarsen_motion", check_clone)
    with pytest.raises(Checked): ENTRIES[0](net, source, status, times, matches([]), bank, facts)
    np.testing.assert_array_equal(source.heading_var, np.full(len(source), .03))


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("change", [{"particles": 100}, {"seed": 1}, {"junction_prior": "legacy"}])
def test_explicit_entries_reject_nonfixed_sampler_before_dispatch(monkeypatch, entry, change):
    monkeypatch.setattr(j, "_align", lambda *a, **k: pytest.fail("nonfixed sampler entered"))
    cfg = j.Config(**{"junction_prior": "conditional-uniform-v3", **change})
    with pytest.raises(ValueError): entry(None, None, None, None, None, None, None, config=cfg)


@pytest.mark.parametrize("entry", [j.align_with_snapshot_future_guide,
    j.align_with_snapshot_normalized_heading_future_guide, j.align_with_snapshot_common_null_appearance,
    j.align_with_snapshot_common_null_normalized_heading])
def test_old_future_and_common_null_entries_never_enable_new_policy(monkeypatch, entry):
    seen = []
    monkeypatch.setattr(j, "_align", lambda *a, **k: seen.append(k))
    entry(None, None, None, None, None, None, None)
    assert len(seen) == 1 and seen[0].get("no_adaptive_heading_q", False) is False


@pytest.mark.parametrize("invalid", [{"future_proposal": False}, {"common_null": True}, {"compatible_appearance": True}])
def test_variance_mode_cannot_leak_to_other_visual_or_proposal_contract(monkeypatch, invalid):
    monkeypatch.setattr(j, "coarsen_motion", lambda *a: pytest.fail("invalid policy reached motion"))
    options = dict(appearance_input=(None, None), snapshot_appearance=True, future_proposal=True, no_adaptive_heading_q=True)
    options.update(invalid)
    with pytest.raises(ValueError): j._align(None, None, None, None, None, **options)
