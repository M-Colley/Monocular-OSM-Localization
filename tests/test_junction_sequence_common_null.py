"""Private target/sampler fixtures only; no case files, models or labels."""
from types import SimpleNamespace
import numpy as np
import pytest
from monocular_osm import junction_sequence as j, raw_vpr_appearance as old, poi_alignment as poi
from monocular_osm import snapshot_future_guide as future
from monocular_osm.graph_filter import FilterConfig
from tests.test_reference_snapshot_junction_probe_v2 import case
from tests.test_junction_sequence import matches


def test_direct_log_ratio_and_initial_weight_do_not_apply_old_wrapper(monkeypatch):
    times = np.arange(80, dtype=np.float64)
    values = np.broadcast_to(np.array([[-2., 0.], [1., 2.]], np.float32), (80, 2, 2)).copy()
    field = poi.RasterField(values, (0., 0.), 1., 0., times, {"kind": "private log-ratio"})
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [20., 20.]])
    net = SimpleNamespace(local_to_crs=lambda x: x,
        locate=lambda e, s: (points[:, 0], points[:, 1], np.zeros(4), np.zeros(4)))
    app = j.CommonNullAppearance(field, np.zeros(80, int), np.full(80, .5), np.arange(80), {})
    monkeypatch.setattr(old.Appearance, "log_factor", lambda *a, **k: pytest.fail("old wrapper called"))
    actual = app.log_factor(0, net, points, time=times[0])
    np.testing.assert_array_equal(actual, [-2., 0., 2., 0.])
    np.testing.assert_array_equal(app.initial_weight(net, np.arange(4), np.zeros(4)), np.exp(.5 * actual))
    assert app.initial_weight(net, np.arange(4), np.zeros(4))[2] > 1.
    with pytest.raises(ValueError): app.log_factor(True, net, points, time=0.)
    with pytest.raises(ValueError): app.log_factor(0, net, points, time=1e-10)


def test_native_adapter_preserves_all_weights_schedule_and_direct_field(monkeypatch):
    net, times, _, _, motion, evidence, facts = case(versions=True)
    monkeypatch.setattr(old.Appearance, "log_factor", lambda *a, **k: pytest.fail("old wrapper called"))
    app = j.prepare_common_null_appearance(evidence, net, facts, times, motion.times)
    groups, powers = old.temporal_groups(evidence["query_times"])
    np.testing.assert_array_equal(app.groups, groups)
    np.testing.assert_array_equal(app.powers, powers)
    assert app.receipt["snapshot_count"] == 4 and app.receipt["all_physical_calls"] == 2
    assert app.receipt["direct_log_ratio_sampling"] is True
    assert app.receipt["old_row_max_or_outlier_wrapper"] is False
    assert app.field.outside_value == 0. and app.field.values.dtype == np.float32
    local = np.array([[500., 0.], [500., 2500.], [1e8, 1e8]])
    expected = app.field.sample(net.local_to_crs(local)[:, None, :], evidence["query_times"][:1])[:, 0]
    np.testing.assert_array_equal(app.log_factor(0, net, local, time=evidence["query_times"][0]), expected)
    assert expected[-1] == 0.


@pytest.mark.parametrize("fault", ["clock", "alias", "weight", "extent"])
def test_native_and_whole_extent_refusal_precedes_new_builder(monkeypatch, fault):
    from monocular_osm import common_null_appearance as core
    net, times, _, _, motion, evidence, facts = case(versions=True)
    if fault == "clock": facts["signed_vo_clock_sha256"] = "0" * 64
    if fault == "alias": facts["all_physical_calls_and_aliases"].pop()
    if fault == "weight": evidence["snapshot_weights"][0] += .001
    if fault == "extent": net.seg_xy0[-1, 0] = 1e8
    monkeypatch.setattr(core, "build_common_null_field", lambda *a, **k: pytest.fail("builder before authority/cap"))
    with pytest.raises(ValueError): j.prepare_common_null_appearance(evidence, net, facts, times, motion.times)


def test_initial_common_null_guide_has_exact_original_p_over_q(monkeypatch):
    net, times, _, _, motion, evidence, facts = case(versions=True)
    app = j.prepare_common_null_appearance(evidence, net, facts, times, motion.times)
    pf = j._OfflineFilter(net, FilterConfig(n_particles=32, seed=0, path_storage=False), motion, [], lambda: None)
    monkeypatch.setattr(pf, "guide", lambda e, s, k: np.full(np.shape(e), .4))
    nc = np.ceil(net.edge_len / 100.).astype(np.int64)
    edges = np.repeat(np.arange(net.n_edges), nc)
    starts = np.concatenate([np.arange(k) * (length / k) for k, length in zip(nc, net.edge_len)])
    lengths = np.repeat(net.edge_len / nc, nc)
    expected = .4 * app.initial_weight(net, edges, starts + lengths / 2.)
    original, seen = j.initial_proposal, []
    def proposal(actual_lengths, guide, defensive=.25):
        np.testing.assert_array_equal(actual_lengths, lengths)
        np.testing.assert_array_equal(guide, expected)
        q, ratio = original(actual_lengths, guide, defensive)
        np.testing.assert_allclose(q * np.exp(ratio), lengths / lengths.sum(), rtol=1e-14)
        seen.append(True)
        return q, ratio
    monkeypatch.setattr(j, "initial_proposal", proposal)
    pf.initialise_offline(app)
    assert seen == [True]


@pytest.mark.parametrize("entry, normalized", [
    (j.align_with_snapshot_common_null_appearance, False),
    (j.align_with_snapshot_common_null_normalized_heading, True)])
def test_new_targets_keep_all80_powers_and_exit_only_proposals(monkeypatch, entry, normalized):
    net, times, odom, status, _, evidence, facts = case(versions=True)
    monkeypatch.setattr(future, "build", lambda *a, **k: pytest.fail("future build in target-only control"))
    monkeypatch.setattr(future, "defensive_auxiliary_guide", lambda *a, **k: pytest.fail("new auxiliary proposal"))
    monkeypatch.setattr(old.Appearance, "log_factor", lambda *a, **k: pytest.fail("old wrapper called"))
    result = entry(net, odom, status, times, matches([(times[-1], [])]), evidence, facts)
    assert result["status"] == "complete_diagnostic_only"
    profile = result["profile"]
    assert profile["visual_target"] == "common-null-v1" and profile["future_proposal"] is False
    assert profile["base_heading_target"] == ("normalized-junction-v1" if normalized else "legacy")
    assert ("heading_mode" in profile) == normalized
    assert [r["query_row"] for r in result["appearance_events"]] == list(range(80))
    assert [r["power"] for r in result["appearance_events"]] == result["appearance"]["group_powers"]
    assert "future_guide" not in result and result["accuracy_admitted"] is False


@pytest.mark.parametrize("entry", [j.align_with_snapshot_common_null_appearance,
                                   j.align_with_snapshot_common_null_normalized_heading])
@pytest.mark.parametrize("change", [{"particles": 100}, {"seed": 1}, {"junction_prior": "legacy"}])
def test_new_entries_reject_nonfixed_sampler_before_dispatch(monkeypatch, entry, change):
    monkeypatch.setattr(j, "_align", lambda *a, **k: pytest.fail("nonfixed sampler entered"))
    cfg = j.Config(**{"junction_prior": "conditional-uniform-v3", **change})
    with pytest.raises(ValueError): entry(None, None, None, None, None, None, None, config=cfg)


@pytest.mark.parametrize("entry", [j.align_with_snapshot_appearance, j.align_with_snapshot_normalized_heading])
def test_old_entries_do_not_build_common_null(monkeypatch, entry):
    from monocular_osm import common_null_appearance as core
    net, times, odom, status, _, evidence, facts = case()
    monkeypatch.setattr(core, "build_common_null_field", lambda *a, **k: pytest.fail("old target changed"))
    result = entry(net, odom, status, times, matches([(times[-1], [])]), evidence, facts)
    assert result["status"] == "complete_diagnostic_only"
    assert "visual_target" not in result["profile"]
