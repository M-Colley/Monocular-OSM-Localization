"""Private compatibility study fixtures; no IO, cases, models, or GT."""
from copy import deepcopy
import gc
import weakref

import numpy as np
import pytest

from monocular_osm import raw_vpr_compat_bank as b
from tests.test_raw_vpr_bank import fixture, reseal


def run(case, loader=None, records=None):
    comparison, original, facts, regions = case
    return b.admit_bank("baseline", original if records is None else records, comparison, facts,
                        loader or (lambda record: regions[record["key"]]))


def shared(keys=("baseline/a", "baseline/b")):
    return fixture(keys=keys, names={k: ["mapillary:0"] for k in keys})


def change_descriptor(case, key, values):
    region = case[3][key]
    for column, value in values.items():
        region["descriptor_rows"][0, column] = value
    region["audit"]["ref_embeddings_sha256"] = b.strict.tensor_sha(region["descriptor_rows"])


def change_score(case, key, value):
    region = case[3][key]
    z = region["arrays"]
    z["raw_similarities"][:] = value
    z["decoded_cosines"][:] = value
    z["returned_cosines"][:] = value
    reseal(next(r for r in case[1] if r["key"] == key), region)


def test_profile_is_separate_immutable_and_uses_explicit_linear_bounds():
    with pytest.raises(TypeError): b.PROFILE["maximum_references"] = 1
    assert b.PROFILE_SHA256 != b.strict.PROFILE_SHA256
    assert b.PROFILE["maximum_references"] == 30000
    assert b.PROFILE["maximum_score_values"] == 80 * 30000
    assert b.PROFILE["maximum_canonical_descriptor_bytes"] == 30000 * 8448 * 4
    assert b.PROFILE["maximum_reference_occurrences"] == 64 * 1500
    assert b.PROFILE["distance_workspace_bytes"] == 8448 * 8
    assert b.strict.PROFILE["maximum_references"] == 10000
    c, records, facts, regions = fixture()
    assert b.strict.admit_bank("baseline", records, c, facts, lambda r: regions[r["key"]])["status"] == "complete_research_bank"


@pytest.mark.parametrize("distance,accepted", [
    (np.nextafter(np.float32(1e-6), np.float32(0)), True),
    (np.float32(1e-6), True),
    (np.nextafter(np.float32(1e-6), np.float32(np.inf)), False),
])
def test_actual_f32_near_boundary_distance_is_compared_in_f64(distance, accepted):
    case = shared()
    change_descriptor(case, "baseline/b", {1: distance})
    result = run(case)
    assert (result["status"] == "complete_compatible_research_bank") is accepted
    if accepted:
        measured = float(abs(np.float64(distance)))
        occurrences = result["mapping"][0]["occurrences"]
        assert result["maximum_descriptor_l2_from_canonical"] == measured
        assert occurrences[0]["descriptor_l2_from_canonical"] == 0.
        assert occurrences[1]["descriptor_l2_from_canonical"] == measured
        assert occurrences[0]["descriptor_sha256"] != occurrences[1]["descriptor_sha256"]
        assert result["compatibility_not_bitwise_identity"] is True
        assert result["historical_encoder_verified"] is False
        assert result["reference_encoding_identity"] == "UNVERIFIED"
        assert not result["policy_admission"] and not result["field_built"]
        assert b.strict.admit_bank("baseline", case[1], case[0], case[2], lambda r: case[3][r["key"]])["status"] == "UNAVAILABLE"
    else:
        assert "fixed canonical" in result["reason"]
        assert not {"mapping", "raw_similarities"}.intersection(result)


def test_full_vector_l2_matches_scalar_f64_oracle_not_component_tolerance():
    case = shared()
    row = case[3]["baseline/b"]["descriptor_rows"][0]
    row[1:] = np.float32(6e-9)
    case[3]["baseline/b"]["audit"]["ref_embeddings_sha256"] = b.strict.tensor_sha(row[None])
    canonical = case[3]["baseline/a"]["descriptor_rows"][0]
    delta = row.astype(np.float64) - canonical.astype(np.float64)
    oracle = float(np.sqrt(np.sum(delta * delta, dtype=np.float64)))
    result = run(case)
    assert result["maximum_descriptor_l2_from_canonical"] == oracle
    assert oracle < 1e-6
    row[1:] = np.float32(2e-8)  # Every component is close, full norm is not.
    case[3]["baseline/b"]["audit"]["ref_embeddings_sha256"] = b.strict.tensor_sha(row[None])
    assert run(case)["status"] == "UNAVAILABLE"


def test_lexical_canonical_is_fixed_not_chained_or_input_order_dependent():
    keys = ("baseline/c", "baseline/a", "baseline/b")
    case = shared(keys)
    change_descriptor(case, "baseline/b", {1: 7.5e-7})
    change_descriptor(case, "baseline/c", {1: 1.5e-6})
    calls = []
    result = run(case, lambda r: calls.append(r["key"]) or case[3][r["key"]])
    assert calls == sorted(keys) and result["status"] == "UNAVAILABLE"
    assert "fixed canonical" in result["reason"]
    change_descriptor(case, "baseline/c", {1: -7.5e-7})
    result = run(case)
    assert result["status"] == "complete_compatible_research_bank"
    assert result["mapping"][0]["canonical"]["key"] == "baseline/a"
    assert np.array_equal(result["raw_similarities"][:, 0], case[3]["baseline/a"]["arrays"]["raw_similarities"][:, 0])


@pytest.mark.parametrize("delta,accepted", [(3.75e-7, True), (7.5e-7, False)])
def test_full_three_occurrence_raw_envelope_not_only_distance_to_canonical(delta, accepted):
    keys = ("baseline/a-middle", "baseline/b-high", "baseline/c-low")
    case = shared(keys)
    values = np.asarray([.2, .2 + delta, .2 - delta], np.float32)
    for key, value in zip(keys, values): change_score(case, key, value)
    result = run(case)
    assert (result["status"] == "complete_compatible_research_bank") is accepted
    if accepted:
        assert result["mapping"][0]["maximum_raw_score_spread"] == float(values.max()) - float(values.min())
    else:
        assert "raw-score envelope" in result["reason"] and "mapping" not in result


@pytest.mark.parametrize("fault", ["missing", "fortran", "shape", "nonfinite", "tensor", "coordinate",
                                    "model", "query", "alias_control", "array_header", "loader"])
def test_any_body_identity_tensor_coordinate_or_alias_conflict_refuses_whole_bank(fault):
    case = shared()
    region = case[3]["baseline/b"]
    if fault == "missing": region.pop("descriptor_rows")
    if fault == "fortran":
        # A shape-correct strided body must not trigger a whole-pool C copy.
        region["descriptor_rows"] = np.zeros((1, 16896), np.float32)[:, ::2]
    if fault == "shape": region["descriptor_rows"] = np.zeros((1, 8447), np.float32)
    if fault == "nonfinite": change_descriptor(case, "baseline/b", {3: np.nan})
    if fault == "tensor": region["audit"]["ref_embeddings_sha256"] = "e" * 64
    if fault == "coordinate":
        region["arrays"]["ref_latlon"][0, 0] += .00001
        region["arrays"]["returned_latlon"][:] = region["arrays"]["ref_latlon"][0]
        reseal(case[1][1], region)
    if fault == "model": region["audit"]["actual_model_state_sha256"] = "e" * 64
    if fault == "query": region["audit"]["query_pixels_sha256"] = "e" * 64
    if fault == "alias_control":
        region["arrays"]["returned_cosines"][17] += .001
        reseal(case[1][1], region)
    if fault == "array_header": region["audit"]["array_artifact"]["arrays"]["raw_similarities"]["sha256"] = "e" * 64
    def loader(record):
        if fault == "loader" and record["key"] == "baseline/b": raise EOFError("private incomplete pool")
        return case[3][record["key"]]
    result = run(case, loader)
    assert result["status"] == "UNAVAILABLE"
    assert not {"raw_similarities", "mapping", "inputs"}.intersection(result)


def test_missing_body_is_refused_even_without_any_duplicate_reference():
    assert run(fixture(descriptors=False))["status"] == "UNAVAILABLE"


@pytest.mark.parametrize("fault", ["subset", "extra", "native", "model", "query"])
def test_all_native_membership_and_common_identity_gates_precede_loader(fault):
    case = fixture()
    c, records, facts, _ = case
    if fault == "subset": records.pop()
    if fault == "extra": records.append({**deepcopy(records[0]), "key": "baseline/extra"})
    if fault == "native": facts["status"] = "UNAVAILABLE"
    if fault == "model": c["model"]["weights_sha256"] = "e" * 64
    if fault == "query": c["query"]["times"][3] += .01
    assert run(case, lambda _: pytest.fail("must refuse before loader"))["status"] == "UNAVAILABLE"


def test_original_loaded_arrays_are_released_and_canonical_rows_are_independent_copies():
    keys = ("baseline/a", "baseline/b", "baseline/c")
    case = fixture(keys=keys, names={k: ["mapillary:0"] for k in keys}, descriptors=False)
    previous, seen = [], []
    def loader(record):
        gc.collect()
        assert all(r() is None for r in previous), "prior original pool retained"
        region = deepcopy(case[3][record["key"]])
        body = np.zeros((1, 8448), np.float32)
        body[0, 0] = 1.
        body[0, 1] = np.float32(len(seen) * 3e-7)
        region["descriptor_rows"] = body
        region["audit"]["ref_embeddings_sha256"] = b.strict.tensor_sha(body)
        previous.extend(weakref.ref(a) for a in (body, region["arrays"]["ref_latlon"], region["arrays"]["raw_similarities"]))
        seen.append(record["key"])
        return region
    result = run(case, loader)
    assert result["status"] == "complete_compatible_research_bank" and seen == list(keys)
    gc.collect()
    assert all(r() is None for r in previous)
    assert result["maximum_descriptor_l2_from_canonical"] == float(np.float32(6e-7))
    assert not any("descriptor_rows" in record for record in result["inputs"].values())


@pytest.mark.parametrize("cap,value", [("maximum_references", 1), ("maximum_score_values", 80),
                                     ("maximum_canonical_descriptor_bytes", 8448 * 4),
                                     ("maximum_reference_occurrences", 1)])
def test_linear_budget_conflicts_refuse_complete_bank_without_partial_output(monkeypatch, cap, value):
    # Same accounting branches with small private caps, no giant allocation.
    monkeypatch.setattr(b, "PROFILE", {**b.PROFILE, cap: value})
    result = run(fixture())
    assert result["status"] == "UNAVAILABLE" and "budget" in result["reason"]
    assert "raw_similarities" not in result


def test_more_than_ten_thousand_unique_rows_use_only_new_linear_gallery_budget():
    keys = tuple(f"baseline/{i}" for i in range(7))
    names = {key: [f"mapillary:{i * 1500 + j}" for j in range(1500 if i < 6 else 1001)]
             for i, key in enumerate(keys)}
    case = fixture(keys=keys, names=names, descriptors=False)
    def loader(record):
        region = deepcopy(case[3][record["key"]])
        n = len(region["arrays"]["ref_ids"])
        body = np.zeros((n, 8448), np.float32)
        body[:, 0] = 1.
        region["descriptor_rows"] = body
        region["audit"]["ref_embeddings_sha256"] = b.strict.tensor_sha(body)
        return region
    result = run(case, loader)
    assert result["status"] == "complete_compatible_research_bank"
    assert result["raw_similarities"].shape == (80, 10001)
    assert result["reference_occurrences"] == 10001
    assert result["physical_calls_validated"] == 7
    assert result["maximum_descriptor_l2_from_canonical"] == 0.
