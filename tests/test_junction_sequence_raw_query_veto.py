"""Private TARGET-only controls: no case arrays, video, maps, models or labels."""
from copy import deepcopy
import numpy as np
import pytest
from monocular_osm import junction_sequence as j, raw_query_veto as gate
from monocular_osm import reference_snapshot_appearance as snapshot, raw_vpr_appearance as appearance
from monocular_osm import snapshot_future_guide as future
from monocular_osm.graph_filter import GraphParticleFilter
from tests.test_reference_snapshot_junction_probe_v2 import case
from tests import test_raw_vpr_appearance as field_fixture
from tests.test_junction_sequence import matches

PAIRS = ((j.align_with_snapshot_future_guide, j.align_with_snapshot_future_guide_raw_query_veto),
         (j.align_with_snapshot_normalized_heading_future_guide,
          j.align_with_snapshot_normalized_heading_future_guide_raw_query_veto))
STATE = ("edge", "offset", "logk", "eps", "delta", "delta_var", "x", "y", "road_heading", "prev_head", "logw")


def capture(monkeypatch):
    filters, original = [], GraphParticleFilter.__init__
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        filters.append(self)
    monkeypatch.setattr(GraphParticleFilter, "__init__", init)
    return filters


def assert_state(first, second):
    for name in STATE:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.rng.bit_generator.state == second.rng.bit_generator.state


def mixed_case(monkeypatch, weak_rows):
    original = field_fixture.case
    def private_source():
        values = original()
        values[5]["raw_similarities"][weak_rows] = [.08, .10, .06]
        return values
    # Rebuild/reseal NEW private physical source pools, never accepted headers.
    monkeypatch.setattr(field_fixture, "case", private_source)
    return case(versions=True)


@pytest.mark.parametrize("old,new", PAIRS)
def test_all_enabled_is_exact_parent_state_route_rng_and_original_events(monkeypatch, old, new):
    net, times, odom, status, _, bank, facts = case(versions=True)
    filters = capture(monkeypatch)
    cue = matches([(times[-1], [])])
    reference = old(net, odom, status, times, cue, bank, facts)
    result = new(net, odom, status, times, cue, bank, facts)
    assert result["status"] == reference["status"] == "complete_diagnostic_only"
    assert_state(filters[0], filters[1])
    for name, value in reference.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(result[name], value)
    for name in ("appearance", "future_guide", "cues"):
        assert result[name] == reference[name]
    assert result["raw_query_enabled"].dtype == np.bool_ and result["raw_query_enabled"].shape == (80,)
    assert result["raw_query_enabled"].all()
    assert result["raw_query_maxima"].dtype == np.float64
    np.testing.assert_array_equal(result["raw_query_maxima"], bank["raw_similarities"].max(axis=1).astype(np.float64))
    for previous, current in zip(reference["appearance_events"], result["appearance_events"]):
        assert {key: current[key] for key in previous} == previous
        assert current["enabled"] is True and current["effective_power"] == previous["power"]
    assert "raw_query_enabled" not in reference and "raw_query_veto_receipt" not in reference
    assert result["raw_query_veto_receipt"]["profile_sha256"] == gate.PROFILE_SHA256
    profile = result["profile"]
    assert profile["raw_query_veto_mode"] == "absolute-max-dev-veto-v1"
    assert profile["proposal_only"] is False and profile["future_guide_is_proposal_only"] is True
    assert profile["target_changed"] is True and profile["appearance_object_and_all_proposals_ungated"] is True
    assert profile["appearance_target_unchanged"] is False


@pytest.mark.parametrize("old,new", PAIRS)
def test_real_disabled_rows_keep_genuine_appearance_full_backward_guides_and_initial_proposal(monkeypatch, old, new):
    weak = [0, 1, 3, 27, 78, 79]
    net, times, odom, status, _, bank, facts = mixed_case(monkeypatch, weak)
    arrays_before = {key: value.copy() for key, value in bank.items() if isinstance(value, np.ndarray)}
    original_prepare, original_build = snapshot.prepare_snapshot, future.build
    apps, guide_records, initial = [], [], []
    def prepare(*args, **kwargs):
        app = original_prepare(*args, **kwargs)
        assert type(app) is appearance.Appearance
        apps.append((app, app.field.values.copy(), app.maxima.copy(), deepcopy(app.receipt)))
        return app
    def build(app, *args, **kwargs):
        assert app is apps[-1][0]
        value = original_build(app, *args, **kwargs)
        guide_records.append((value.backward_H.copy(), deepcopy(value.receipt)))
        return value
    original_initial = j._OfflineFilter.initialise_offline
    def initialise(self, app=None):
        assert app is apps[-1][0]
        original_initial(self, app)
        initial.append(({key: getattr(self, key).copy() for key in STATE}, deepcopy(self.rng.bit_generator.state)))
    monkeypatch.setattr(snapshot, "prepare_snapshot", prepare)
    monkeypatch.setattr(future, "build", build)
    monkeypatch.setattr(j._OfflineFilter, "initialise_offline", initialise)
    cue = matches([(times[-1], [])])
    previous = old(net, odom, status, times, cue, bank, facts)
    result = new(net, odom, status, times, cue, bank, facts)
    assert result["status"] == previous["status"] == "complete_diagnostic_only"
    for key in STATE:
        np.testing.assert_array_equal(initial[0][0][key], initial[1][0][key])
    assert initial[0][1] == initial[1][1]
    np.testing.assert_array_equal(guide_records[0][0], guide_records[1][0])
    assert guide_records[0][1] == guide_records[1][1]
    for app, values, maxima, receipt in apps:
        np.testing.assert_array_equal(app.field.values, values)
        np.testing.assert_array_equal(app.maxima, maxima)
        assert app.receipt == receipt
    for key, value in arrays_before.items():
        np.testing.assert_array_equal(bank[key], value)
    enabled = result["raw_query_enabled"]
    assert np.flatnonzero(~enabled).tolist() == weak
    events = result["appearance_events"]
    assert [row["query_row"] for row in events] == list(range(80))
    for row, original in zip(events, previous["appearance_events"]):
        assert row["power"] == original["power"]
        assert row["enabled"] == bool(enabled[row["query_row"]])
        assert row["effective_power"] == row["power"] * float(row["enabled"])
        if not row["enabled"]:
            assert row["log_factor_min"] == row["log_factor_max"] == 0.
    assert sum(row["effective_power"] for row in events) < sum(row["power"] for row in events)


@pytest.mark.parametrize("old,new", PAIRS)
def test_disabled_finite_factor_values_have_exactly_zero_target_effect(monkeypatch, old, new):
    weak = [1, 27, 79]  # G0 is deliberately unchanged in this independent oracle.
    net, times, odom, status, _, bank, facts = mixed_case(monkeypatch, weak)
    filters = capture(monkeypatch)
    original = appearance.Appearance.log_factor
    value = [-2.]
    def factor(self, query, net, local_xy, **kwargs):
        if query in weak:
            return np.full(len(local_xy), value[0], np.float64)
        return original(self, query, net, local_xy, **kwargs)
    monkeypatch.setattr(appearance.Appearance, "log_factor", factor)
    cue = matches([(times[-1], [])])
    first = new(net, odom, status, times, cue, bank, facts)
    value[0] = -900.
    second = new(net, odom, status, times, cue, bank, facts)
    assert first["status"] == second["status"] == "complete_diagnostic_only"
    assert_state(filters[0], filters[1])
    for name, array in first.items():
        if isinstance(array, np.ndarray):
            np.testing.assert_array_equal(array, second[name])
    assert first["appearance_events"] == second["appearance_events"]


def test_disabled_NaN_is_refused_before_zero_multiplication(monkeypatch):
    net, times, odom, status, _, bank, facts = mixed_case(monkeypatch, [1])
    original = appearance.Appearance.log_factor
    def factor(self, query, net, local_xy, **kwargs):
        if query == 1:
            return np.full(len(local_xy), np.nan)
        return original(self, query, net, local_xy, **kwargs)
    monkeypatch.setattr(appearance.Appearance, "log_factor", factor)
    with pytest.raises(ValueError, match="finite per-particle"):
        PAIRS[0][1](net, odom, status, times, matches([(times[-1], [])]), bank, facts)


@pytest.mark.parametrize("entry", [pair[1] for pair in PAIRS])
@pytest.mark.parametrize("change", [{"particles": 100}, {"seed": 1}, {"junction_prior": "legacy"}])
def test_new_entries_reject_nonfixed_sampler_before_dispatch(monkeypatch, entry, change):
    monkeypatch.setattr(j, "_align", lambda *a, **k: pytest.fail("invalid configuration dispatched"))
    config = j.Config(**{"junction_prior": "conditional-uniform-v3", **change})
    with pytest.raises(ValueError):
        entry(None, None, None, None, None, None, None, config=config)


@pytest.mark.parametrize("change", [{"snapshot_appearance": False}, {"future_proposal": False},
    {"common_null": True}, {"compatible_appearance": True}, {"no_adaptive_heading_q": True},
    {"adaptive_resampling": True}, {"high_particle_budget": True}, {"appearance_input": None},
    {"raw_query_veto": 1}])
def test_veto_requires_exact_run9_parent_before_motion_or_input_access(monkeypatch, change):
    monkeypatch.setattr(j, "coarsen_motion", lambda *a: pytest.fail("invalid method reached input values"))
    options = dict(config=j.Config(junction_prior="conditional-uniform-v3"),
        appearance_input=(None, None), snapshot_appearance=True, future_proposal=True, raw_query_veto=True)
    options.update(change)
    with pytest.raises(ValueError):
        j._align(None, None, None, None, None, **options)


@pytest.mark.parametrize("entry", [pair[0] for pair in PAIRS])
def test_original_future_entry_never_builds_a_veto(monkeypatch, entry):
    net, times, odom, status, _, bank, facts = case(versions=True)
    monkeypatch.setattr(gate, "build_gate", lambda *a: pytest.fail("old entry enabled new target"))
    result = entry(net, odom, status, times, matches([(times[-1], [])]), bank, facts)
    assert result["status"] == "complete_diagnostic_only"
    assert not {"raw_query_enabled", "raw_query_maxima", "raw_query_veto_receipt"} & result.keys()


def test_all_disabled_retains_every_event_with_zero_effective_power(monkeypatch):
    net, times, odom, status, _, bank, facts = mixed_case(monkeypatch, list(range(80)))
    result = PAIRS[0][1](net, odom, status, times, matches([(times[-1], [])]), bank, facts)
    assert result["status"] == "complete_diagnostic_only"
    assert result["raw_query_veto_receipt"]["enabled_count"] == 0
    assert not result["raw_query_enabled"].any()
    assert [row["query_row"] for row in result["appearance_events"]] == list(range(80))
    assert sum(row["effective_power"] for row in result["appearance_events"]) == 0.
    assert all(row["power"] > 0 and row["enabled"] is False for row in result["appearance_events"])


def test_computed_gate_diagnostics_survive_later_unavailable_without_fake_events(monkeypatch):
    net, times, odom, status, _, bank, facts = case(versions=True)
    def refuse(*args, **kwargs):
        raise ValueError("private fixed future bound")
    monkeypatch.setattr(future, "build", refuse)
    result = PAIRS[0][1](net, odom, status, times, matches([(times[-1], [])]), bank, facts)
    assert result["status"] == "unavailable" and result["route_latlon"] is None
    assert result["raw_query_enabled"].shape == result["raw_query_maxima"].shape == (80,)
    assert result["raw_query_veto_receipt"]["enabled_count"] == 80
    assert "appearance_events" not in result


def test_early_unavailable_does_not_fabricate_uncomputed_gate(monkeypatch):
    def refuse(*args, **kwargs):
        raise j.Unavailable("private motion bound")
    monkeypatch.setattr(j, "coarsen_motion", refuse)
    result = PAIRS[0][1](None, None, None, None, None, None, None)
    assert result["status"] == "unavailable"
    assert not {"raw_query_enabled", "raw_query_maxima", "raw_query_veto_receipt"} & result.keys()
