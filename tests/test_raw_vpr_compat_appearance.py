"""Synthetic compatible-bank creation; never relabel an old strict bank."""
from copy import deepcopy
import numpy as np
import pytest
from monocular_osm import raw_vpr_compat_appearance as a, raw_vpr_appearance as strict
from monocular_osm import raw_vpr_compat_bank as compatible, junction_sequence as j
from tests import test_raw_vpr_appearance as old
from tests.test_raw_vpr_bank import fixture, reseal


def case():
    net, times, odom, status, motion, strict_bank, facts = old.case()
    original = fixture(keys=("baseline/a",), names={"baseline/a": strict_bank["ref_ids"].tolist()})
    comparison, records, _, regions = original
    comparison["query"]["times"] = times[np.linspace(0, 100, 80).astype(int)].tolist()
    region = regions["baseline/a"]
    z, meta = region["arrays"], region["audit"]
    for name in ("query_times", "ref_latlon", "raw_similarities"):
        z[name] = strict_bank[name].copy()
    z["decoded_cosines"] = z["raw_similarities"][:, 0].copy()
    z["returned_cosines"] = z["decoded_cosines"].astype(np.float64)
    z["returned_latlon"] = z["ref_latlon"][z["decoded_ref_indices"]]
    meta["comparison_identity"]["query_times_sha256"] = a.strict.native.typed(z["query_times"])
    meta["comparison_identity_sha256"] = compatible.strict.legacy.json_sha(meta["comparison_identity"])
    reseal(records[0], region)
    bank = compatible.admit_bank("baseline", records, comparison, facts, lambda r: regions[r["key"]])
    assert bank["status"] == "complete_compatible_research_bank", bank
    return net, times, odom, status, motion, bank, facts, strict_bank


def test_explicit_compatible_field_has_bit_exact_shared_math_without_strict_cast():
    net, times, _, _, motion, bank, facts, strict_bank = case()
    result = a.prepare_compatible(bank, net, facts, times, motion.times)
    control = strict.prepare(strict_bank, net, facts, times, motion.times)
    np.testing.assert_array_equal(result.field.values, control.field.values)
    np.testing.assert_array_equal(result.maxima, control.maxima)
    np.testing.assert_array_equal(result.groups, control.groups)
    np.testing.assert_array_equal(result.powers, control.powers)
    assert result.receipt["compatibility_not_bitwise_identity"] is True
    assert a.PROFILE_SHA256 != strict.PROFILE_SHA256
    with pytest.raises(ValueError): strict.prepare(bank, net, facts, times, motion.times)
    with pytest.raises(ValueError): a.prepare_compatible(strict_bank, net, facts, times, motion.times)


def test_compatible_and_strict_sampler_math_and_rng_identical_for_identical_inputs():
    net, times, odom, status, _, bank, facts, strict_bank = case()
    matched = old.matches([(times[-1], [])])
    result = j.align_with_compatible_appearance(net, odom, status, times, matched, bank, facts)
    control = j.align_with_appearance(net, odom, status, times, matched, strict_bank, facts)
    assert result["status"] == control["status"] == "complete_diagnostic_only"
    for name, value in control.items():
        if isinstance(value, np.ndarray): np.testing.assert_array_equal(result[name], value)
    assert result["appearance_events"] == control["appearance_events"]
    assert result["profile"]["compatibility_not_bitwise_identity"] is True
    assert result["accuracy_admitted"] is False and result["selection"] is None


@pytest.mark.parametrize("fault", ["status", "profile", "identity_claim", "clock"])
def test_compatible_contract_remains_explicit_and_native(fault):
    net, times, _, _, motion, bank, facts, _ = case()
    if fault == "status": bank["status"] = "complete_research_bank"
    if fault == "profile": bank["profile_sha256"] = strict.PROFILE_SHA256
    if fault == "identity_claim": bank["compatibility_not_bitwise_identity"] = False
    if fault == "clock": facts["signed_vo_clock_sha256"] = "f" * 64
    with pytest.raises(ValueError): a.prepare_compatible(bank, net, facts, times, motion.times)
