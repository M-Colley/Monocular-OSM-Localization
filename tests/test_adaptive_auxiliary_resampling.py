"""Private scalar/ancestry evidence only; no actual cases or filter IO."""
import hashlib
import json

import numpy as np
import pytest

from monocular_osm import adaptive_auxiliary_resampling as a


@pytest.mark.parametrize("weights,expected", [
    ([.25, .25, .25, .25], False),
    ([.5, .5, 0., 0.], False),  # equality is a skip
    ([.6, .4, 0., 0.], True),
    ([.5, .49, .01, 0.], False),
    ([1.], False),
    ([1., 0., 0., 0.], True),
])
def test_fixed_strict_half_target_ess(weights, expected):
    w = np.array(weights, dtype=np.float64)
    result = a.decision(w)
    assert result == {"ess": float(1. / np.sum(w * w)),
                      "resample": expected, "particle_count": len(w)}


def test_readonly_strided_weights_and_rng_are_bitwise_unchanged():
    w = np.full(24, 1. / 12., dtype=np.float64)[::2]
    w.flags.writeable = False
    before = w.tobytes(), w.strides, w.flags.writeable
    rng = np.random.get_state()
    assert not a.decision(w)["resample"]
    assert (w.tobytes(), w.strides, w.flags.writeable) == before
    after = np.random.get_state()
    assert rng[0] == after[0] and rng[2:] == after[2:]
    np.testing.assert_array_equal(rng[1], after[1])


def test_skip_must_preserve_carried_inverse_guide_weights():
    previous_guide = np.array([.5, .75, 1.])
    logw = -np.log(previous_guide)
    logw -= np.log(np.exp(logw).sum())
    weights = np.exp(logw)
    assert not a.decision(weights)["resample"]
    # A uniform reset would silently delete an earlier proposal correction.
    expected_mean = float(weights @ np.arange(3.))
    assert expected_mean == pytest.approx(10. / 13.)
    assert expected_mean != pytest.approx(float(np.arange(3.).mean()))
    before = logw.tobytes()
    a.decision(weights)
    assert logw.tobytes() == before


def test_target_ess_is_not_tilted_guide_ess():
    weights = np.full(4, .25)
    raw = np.array([1., 1e-6, 1e-6, 1e-6])
    effective = .25 + .75 * raw / float(weights @ raw)
    proposal = weights * effective
    proposal /= proposal.sum()
    assert not a.decision(weights)["resample"]
    assert a.decision(proposal)["resample"]


def test_trigger_keeps_existing_defensive_auxiliary_measure_correction():
    weights = np.array([.97, .01, .01, .01])
    raw = np.array([.01, .2, .7, 1.])
    assert a.decision(weights)["resample"]
    effective = .25 + .75 * raw / float(weights @ raw)
    proposal = weights * effective
    proposal /= proposal.sum()
    recovered = proposal / effective
    np.testing.assert_allclose(recovered / recovered.sum(), weights, rtol=1e-15)
    assert np.all(proposal >= .25 * weights - 1e-16)
    assert np.max(1. / effective) <= 4.
    # Multiplying the old weight again after resampling targets w**2 instead.
    wrong = recovered * weights
    wrong /= wrong.sum()
    assert not np.allclose(wrong, weights)


def test_identity_parent_required_after_skipping_prior_resampling():
    first = np.array([10, 20, 30])
    sampled = np.array([2, 0, 2], dtype=np.int32)
    second = first[sampled]
    third = second.copy()  # no transition or resampling
    final_index = 1
    identity = np.arange(3, dtype=np.int32)
    parent_second = int(identity[final_index])
    correct = [first[sampled[parent_second]], second[parent_second], third[final_index]]
    stale_second = int(sampled[final_index])
    stale = [first[sampled[stale_second]], second[stale_second], third[final_index]]
    assert correct == [10, 10, 10]
    assert stale == [30, 30, 10]


@pytest.mark.parametrize("bad", [
    [1.], np.array([1.], dtype=np.float32), np.array([True]),
    np.empty(0, dtype=np.float64), np.ones((1, 1)),
    np.array([np.nan]), np.array([np.inf]), np.array([-.1, 1.1]),
    np.array([0., 0.]), np.array([.4, .4]), np.array([.5, .50000000001]),
])
def test_invalid_target_weights_refused(bad):
    with pytest.raises(ValueError):
        a.decision(bad)


def test_cap_is_checked_before_numerical_scratch(monkeypatch):
    bad = np.zeros(50_001, dtype=np.float64)
    monkeypatch.setattr(a.np, "isfinite", lambda _: pytest.fail("oversize numerical validation"))
    with pytest.raises(ValueError, match="bounded"):
        a.decision(bad)


def test_full_fixed_capacity_and_profile_digest():
    w = np.full(50_000, 1. / 50_000)
    result = a.decision(w)
    assert not result["resample"] and result["particle_count"] == 50_000
    expected = hashlib.sha256(json.dumps(dict(a.PROFILE), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert a.PROFILE_SHA256 == expected and a.PROFILE["proposal_only"]
    with pytest.raises(TypeError):
        a.PROFILE["maximum_particles"] = 500_000
