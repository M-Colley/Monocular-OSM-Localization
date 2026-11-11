"""Private in-memory version/alias fixtures; no cases, IO, models or GT."""
from copy import deepcopy
import gc
import math
import weakref
import numpy as np
import pytest
from monocular_osm import reference_snapshot_evidence as s, raw_vpr_compat_bank as numeric
from tests.test_raw_vpr_bank import fixture, reseal


ARRAYS = ("provider_ids", "ref_latlon", "raw_similarities", "snapshot_weights", "query_indices", "query_times")


def case(keys=("baseline/a", "baseline/b"), names=None):
    return fixture(keys=keys, names=names or {key: ["mapillary:0"] for key in keys})


def run(c, loader=None, records=None):
    comparison, original, facts, regions = c
    return s.admit_bank("baseline", original if records is None else records, comparison, facts,
                        loader or (lambda record: regions[record["key"]]))


def descriptor(c, key, col=0):
    region = c[3][key]
    region["descriptor_rows"][col] = 0.
    region["descriptor_rows"][col, 1] = 1.
    region["audit"]["ref_embeddings_sha256"] = s.strict.tensor_sha(region["descriptor_rows"])


def score(c, key, value, col=0):
    region = c[3][key]; z = region["arrays"]
    z["raw_similarities"][:, col] = value
    chosen = z["decoded_ref_indices"]
    z["decoded_cosines"] = z["raw_similarities"][np.arange(80), chosen]
    z["returned_cosines"] = z["decoded_cosines"].astype(np.float64)
    reseal(next(r for r in c[1] if r["key"] == key), region)


def test_profile_and_array_abi_are_new_immutable_and_bounded():
    with pytest.raises(TypeError): s.PROFILE["maximum_snapshots"] = 1
    assert s.PROFILE_SHA256 not in (s.strict.PROFILE_SHA256, numeric.PROFILE_SHA256)
    assert s.PROFILE["maximum_reference_occurrences"] == s.PROFILE["maximum_snapshots"] == 64 * 1500
    assert s.PROFILE["maximum_provider_ids"] == 30000
    assert s.PROFILE["maximum_raw_matrix_bytes"] == 30_720_000
    # Worst IDs allowed by strict validation are128 Unicode chars (512B).
    worst_arrays = 96000 * (512 + 16 + 8 + 80 * 4) + 80 * 16
    assert worst_arrays < s.PROFILE["maximum_output_npz_bytes"] == 128 * 1024**2
    result = run(case())
    assert result["status"] == "complete_snapshot_research_evidence"
    assert result["provider_ids"].shape == (1,) and result["raw_similarities"].shape == (80, 1)
    assert result["ref_latlon"].dtype == result["snapshot_weights"].dtype == np.float64
    assert result["raw_similarities"].dtype == np.float32
    assert result["reference_encoding_identity"] == "UNVERIFIED"
    for flag in ("historical_encoder_verified", "descriptor_equivalence_assumed", "old_strict_or_numeric_status_upgraded",
                 "field_built", "decoder_run", "policy_admission"):
        assert result[flag] is False
    assert result["query_observations_per_time"] == 1


@pytest.mark.parametrize("kind", ["descriptor", "raw", "both"])
def test_distinct_versions_are_retained_without_descriptor_or_raw_equivalence(kind):
    c = case()
    if kind in {"descriptor", "both"}: descriptor(c, "baseline/b")
    if kind in {"raw", "both"}: score(c, "baseline/b", .8)
    result = run(c)
    assert result["status"] == "complete_snapshot_research_evidence"
    assert result["snapshot_count"] == 2 and result["unique_provider_count"] == 1
    assert result["provider_ids"].tolist() == ["mapillary:0"] * 2
    np.testing.assert_array_equal(result["snapshot_weights"], [.5, .5])
    assert s.strict.admit_bank("baseline", c[1], c[0], c[2], lambda r: c[3][r["key"]])["status"] == "UNAVAILABLE"
    assert numeric.admit_bank("baseline", c[1], c[0], c[2], lambda r: c[3][r["key"]])["status"] == "UNAVAILABLE"
    for col, row in enumerate(result["mapping"]):
        src = c[3][row["canonical"]["key"]]
        j = row["canonical"]["column"]
        assert row["descriptor_row_sha256"] == s.strict.legacy.full._array_sha(src["descriptor_rows"][j:j+1])
        assert row["raw_column_sha256"] == s.strict.legacy.full._array_sha(src["arrays"]["raw_similarities"][:, j])
        assert result["raw_similarities"][:, col].tobytes() == src["arrays"]["raw_similarities"][:, j].tobytes()


def test_exact_aliases_cannot_change_snapshot_counts_weights_or_values():
    original = case(); descriptor(original, "baseline/b")
    extra = case(keys=("baseline/a", "baseline/b", "baseline/c-alias"))
    descriptor(extra, "baseline/b")
    before, after = run(original), run(extra)
    for name in ARRAYS: np.testing.assert_array_equal(before[name], after[name])
    assert after["physical_calls_validated"] == after["reference_occurrences"] == 3
    assert before["snapshot_count"] == after["snapshot_count"] == 2
    assert sorted(len(row["occurrences"]) for row in after["mapping"]) == [1, 2]
    assert after["inputs"]["baseline/c-alias"]["actual_descriptor_tensor_sha256"] == extra[3]["baseline/c-alias"]["audit"]["ref_embeddings_sha256"]


def test_per_provider_unit_weight_not_global_snapshot_weight_and_stable_order():
    keys = ("baseline/c", "baseline/a", "baseline/b")
    c = case(keys, {key: ["mapillary:10", "mapillary:2"] for key in keys})
    descriptor(c, "baseline/b", 0); score(c, "baseline/c", .6, 0)
    descriptor(c, "baseline/b", 1)
    result = run(c); shuffled = run(c, records=list(reversed(c[1])))
    assert result["status"] == "complete_snapshot_research_evidence"
    assert result["unique_provider_count"] == 2 and result["snapshot_count"] == 5
    tuples = [(m["provider_id"], m["descriptor_row_sha256"], m["raw_column_sha256"]) for m in result["mapping"]]
    assert tuples == sorted(tuples)
    for provider, k in (("mapillary:10", 3), ("mapillary:2", 2)):
        weights = result["snapshot_weights"][result["provider_ids"] == provider]
        np.testing.assert_array_equal(weights, np.full(k, 1. / k))
        assert math.fsum(map(float, weights)) == pytest.approx(1., abs=2e-16)
    for name in ARRAYS: np.testing.assert_array_equal(result[name], shuffled[name])
    assert result["mapping"] == shuffled["mapping"]
    assert all(m["canonical"] == min(m["occurrences"], key=lambda p: (p["key"], p["column"])) for m in result["mapping"])


@pytest.mark.parametrize("fault", ["coordinates", "model", "query", "tensor", "alias_control", "header", "missing_body", "strided", "nonfinite", "loader"])
def test_any_per_pool_or_coordinate_conflict_refuses_entire_representation(fault):
    c = case(); descriptor(c, "baseline/b"); score(c, "baseline/b", .8)
    region = c[3]["baseline/b"]; z, audit = region["arrays"], region["audit"]
    if fault == "coordinates":
        z["ref_latlon"][0, 0] += .001; z["returned_latlon"][:] = z["ref_latlon"][0]
        reseal(c[1][1], region)
    elif fault == "model": audit["actual_model_state_sha256"] = "f" * 64
    elif fault == "query": audit["query_pixels_sha256"] = "f" * 64
    elif fault == "tensor": audit["ref_embeddings_sha256"] = "f" * 64
    elif fault == "alias_control":
        z["returned_cosines"][17] += .01; reseal(c[1][1], region)
    elif fault == "header": audit["array_artifact"]["arrays"]["raw_similarities"]["sha256"] = "f" * 64
    elif fault == "missing_body": region.pop("descriptor_rows")
    elif fault == "strided": region["descriptor_rows"] = np.zeros((1, 16896), np.float32)[:, ::2]
    elif fault == "nonfinite":
        region["descriptor_rows"][0, 7] = np.nan
        audit["ref_embeddings_sha256"] = s.strict.tensor_sha(region["descriptor_rows"])
    def loader(record):
        if fault == "loader" and record["key"] == "baseline/b": raise EOFError("private incomplete physical pool")
        return c[3][record["key"]]
    result = run(c, loader)
    assert result["status"] == "UNAVAILABLE"
    assert not set(ARRAYS + ("mapping", "inputs")).intersection(result)


@pytest.mark.parametrize("cap,value", [("maximum_reference_occurrences", 1), ("maximum_provider_ids", 1),
                                     ("maximum_snapshots", 1), ("maximum_score_values", 80)])
def test_declared_bounds_refuse_without_subset(monkeypatch, cap, value):
    c = fixture() if cap == "maximum_provider_ids" else case()
    if cap in {"maximum_snapshots", "maximum_score_values"}: descriptor(c, "baseline/b")
    monkeypatch.setattr(s, "PROFILE", {**s.PROFILE, cap: value})
    result = run(c)
    assert result["status"] == "UNAVAILABLE" and "bound" in result["reason"]
    assert not set(ARRAYS).intersection(result)


def test_original_descriptor_and_array_views_are_released_between_loads():
    keys = ("baseline/a", "baseline/b", "baseline/c")
    c = fixture(keys=keys, names={key: ["mapillary:0"] for key in keys}, descriptors=False)
    previous, calls = [], []
    def loader(record):
        gc.collect()
        assert all(ref() is None for ref in previous), "prior descriptor/pool view retained"
        region = deepcopy(c[3][record["key"]])
        body = np.zeros((1, 8448), np.float32); body[0, len(calls)] = 1.
        region["descriptor_rows"] = body
        region["audit"]["ref_embeddings_sha256"] = s.strict.tensor_sha(body)
        previous.extend(weakref.ref(v) for v in (body, region["arrays"]["ref_latlon"], region["arrays"]["raw_similarities"]))
        calls.append(record["key"])
        return region
    result = run(c, loader)
    assert result["status"] == "complete_snapshot_research_evidence" and result["snapshot_count"] == 3
    gc.collect()
    assert all(ref() is None for ref in previous)
    assert not any("descriptor_rows" in value for value in result["inputs"].values())


def test_original_metadata_arrays_and_old_statuses_are_not_mutated():
    c = case(); before = deepcopy(c)
    assert run(c)["status"] == "complete_snapshot_research_evidence"
    for key, region in c[3].items():
        assert region["audit"] == before[3][key]["audit"]
        for name, array in region["arrays"].items(): np.testing.assert_array_equal(array, before[3][key]["arrays"][name])
    assert s.strict.admit_bank("baseline", c[1], c[0], c[2], lambda r: c[3][r["key"]])["status"] == "complete_research_bank"
    assert numeric.admit_bank("baseline", c[1], c[0], c[2], lambda r: c[3][r["key"]])["status"] == "complete_compatible_research_bank"


def test_last_identical_alias_gets_full_validation_before_folding():
    c = case(keys=("baseline/a", "baseline/b", "baseline/c"))
    c[3]["baseline/c"]["arrays"]["returned_cosines"][17] += .001
    reseal(c[1][2], c[3]["baseline/c"])
    seen = []
    result = run(c, lambda r: seen.append(r["key"]) or c[3][r["key"]])
    assert seen == ["baseline/a", "baseline/b", "baseline/c"]
    assert result["status"] == "UNAVAILABLE" and "original regional raw controls" in result["reason"]


@pytest.mark.parametrize("fault", ["subset", "extra", "native", "model", "query", "over_calls"])
def test_native_common_identity_and_call_bound_precede_loader(fault):
    c = case(); comparison, records, facts, _ = c
    if fault == "subset": records.pop()
    elif fault == "extra": records.append({**deepcopy(records[0]), "key": "baseline/extra"})
    elif fault == "native": facts["status"] = "UNAVAILABLE"
    elif fault == "model": comparison["model"]["weights_sha256"] = "f" * 64
    elif fault == "query": comparison["query"]["times"][3] += .01
    elif fault == "over_calls": records[:] = [deepcopy(records[0]) for _ in range(65)]
    result = run(c, lambda _: pytest.fail("numeric callback before native/header membership"))
    assert result["status"] == "UNAVAILABLE"
