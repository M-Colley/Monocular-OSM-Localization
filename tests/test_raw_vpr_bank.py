"""Private in-memory bank fixtures: no cases, IO, models, or GT."""
from copy import deepcopy
import numpy as np
import pytest

from monocular_osm import raw_vpr_bank as b


def fixture(keys=("baseline/a", "baseline/b"), names=None, descriptors=True):
    q = 80
    comparison = {"statistic": b.legacy.STATISTIC,
                  "model": {"name": "megaloc", "hub": b.legacy.PROFILE["backbone_hub"],
                            "device": "cpu", "weights_sha256": "c" * 64},
                  "query": {"indices": list(range(q)), "times": (np.arange(q) * .5).tolist(),
                            "mode": "warp", "preprocessing": {"dimensions": 8448, "resize": [512, 512]},
                            "pixels_sha256": "a" * 64, "descriptors_sha256": "b" * 64}}
    regions, records = {}, []
    for i, key in enumerate(keys):
        identifiers = (names or {}).get(key, [f"mapillary:{i}"])
        n = len(identifiers)
        raw = np.full((q, n), .2, np.float32)
        coords = np.array([[40., -80. + int(v.rsplit(":", 1)[1]) * 1e-6] for v in identifiers])
        chosen = np.zeros(q, np.int64)
        arrays = {"query_indices": np.arange(q, dtype=np.int64), "query_times": np.arange(q) * .5,
                  "ref_ids": np.array(identifiers), "ref_latlon": coords, "raw_similarities": raw,
                  "decoded_ref_indices": chosen, "decoded_cosines": raw[:, 0].copy(),
                  "returned_indices": np.arange(q, dtype=np.int64), "returned_latlon": coords[chosen],
                  "returned_cosines": raw[:, 0].astype(np.float64)}
        record = {"key": key, **{role: {"path": "/private/" + key + suffix, "sha256": str(i % 10) * 64}
                                 for role, suffix in (("request_ref", "/request.json"), ("metadata_ref", "/0000.json"),
                                                      ("array_ref", "/0000.npz"))}}
        audit = {"status": "complete", "issues": [], "score_kind": comparison["statistic"],
                 "backbone": {k: v for k, v in comparison["model"].items() if k != "weights_sha256"},
                 "actual_model_state_sha256": "c" * 64, "query_mode": "warp",
                 "preprocessing": deepcopy(comparison["query"]["preprocessing"]),
                 "query_pixels_sha256": "a" * 64, "query_descriptors_sha256": "b" * 64,
                 "decoder": {"requested": True, "actual": "viterbi"},
                 "reference_encoding_identity": "UNVERIFIED",
                 "reference_pool": {"source": "mapillary", "count": n,
                                    "fetch": {"listing": "tiles", "listing_stats_known": True, "lost": 0,
                                              "snapshot": {"n_refs": n}}}}
        original = {k: deepcopy(audit[k]) for k in (*b.native.APPEARANCE, "score_kind", "decoder")}
        original.update(query_indices_sha256=b.legacy.full._array_sha(arrays["query_indices"]),
                        query_times_sha256=b.legacy.full._array_sha(arrays["query_times"]))
        audit.update(comparison_identity=original, comparison_identity_sha256=b.legacy.json_sha(original))
        region = {"key": key, "audit": audit, "arrays": arrays}
        if descriptors:
            rows = np.zeros((n, 8448), np.float32)
            rows[:, 0] = 1.
            region["descriptor_rows"] = rows
            audit["ref_embeddings_sha256"] = b.tensor_sha(rows)
        regions[key] = region
        records.append(record)
        reseal(record, region)
    first = regions[keys[0]]
    flat = {k: deepcopy(first["audit"][k]) for k in b.native.APPEARANCE}
    for name in ("query_indices", "query_times"):
        flat[name + "_header"] = deepcopy(first["audit"]["array_artifact"]["arrays"][name])
    facts = {"schema": "raw-vpr-own-native80-replay-v1",
             "status": "VERIFIED_ALL_BOUND_OWN_NATIVE80_PIXELS_CLOCK_NO_NN", "profile": deepcopy(b.native.PROFILE),
             "reference_encoding_upgraded": False, "query_descriptors_and_model_values_recomputed": False,
             "reference_encoding": "UNVERIFIED",
             "original_child_completion_certified": False, "appearance_identity": flat,
             "bank_membership": {"baseline": list(keys)}, "all_physical_calls_and_aliases": []}
    for record in records:
        facts["all_physical_calls_and_aliases"].append({**deepcopy(record), "profile_key": "d" * 64,
            "query_pixels_sha256": "a" * 64, "query_indices_sha256": flat["query_indices_header"]["sha256"],
            "query_times_sha256": flat["query_times_header"]["sha256"], "query_to_vo": [{} for _ in range(80)]})
    return comparison, records, facts, regions


def reseal(record, region):
    region["audit"]["array_artifact"] = {**record["array_ref"], "arrays": {
        name: {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": b.legacy.full._array_sha(a)}
        for name, a in region["arrays"].items()}}


def run(case, loader=None, records=None):
    comparison, original, facts, regions = case
    return b.admit_bank("baseline", original if records is None else records, comparison, facts,
                        loader or (lambda record: regions[record["key"]]))


def test_complete_bank_preserves_all_columns_sorted_and_stable_query_identity():
    case = fixture(keys=("baseline/z", "baseline/a"), names={"baseline/z": ["mapillary:2", "mapillary:0"]})
    result = run(case)
    assert result["status"] == "complete_research_bank"
    assert result["ref_ids"].tolist() == ["mapillary:0", "mapillary:1", "mapillary:2"]
    assert result["raw_similarities"].shape == (80, 3) and result["reference_occurrences"] == 3
    assert result["reference_encoding_identity"] == "UNVERIFIED" and not result["policy_admission"]
    assert np.array_equal(result["query_indices"], np.arange(80))
    assert all("descriptor_rows" not in row for row in result["mapping"])


def test_alias_is_validated_and_deduplicated_only_after_complete_physical_call_checks():
    keys = ("baseline/z", "baseline/a")
    case = fixture(keys=keys, names={k: ["mapillary:0"] for k in keys})
    calls = []
    result = run(case, lambda record: calls.append(record["key"]) or case[3][record["key"]])
    assert calls == sorted(keys) and result["physical_calls_validated"] == 2
    assert result["raw_similarities"].shape == (80, 1) and result["query_observations_per_time"] == 1
    assert result["mapping"][0]["canonical"]["key"] == "baseline/a"
    assert len(result["mapping"][0]["occurrences"]) == 2
    case[3]["baseline/z"]["audit"]["status"] = "partial"
    assert run(case)["status"] == "UNAVAILABLE"


@pytest.mark.parametrize("fault", ["native", "subset", "extra", "duplicate", "role", "bank", "query", "model", "unsigned"])
def test_all_authority_and_native_common_identity_checks_precede_callbacks(fault):
    case = fixture(); c, records, facts, _ = case
    if fault == "native": facts["status"] = "unavailable"
    elif fault == "subset": records.pop()
    elif fault == "extra": records.append({**deepcopy(records[0]), "key": "baseline/extra"})
    elif fault == "duplicate": records[1] = deepcopy(records[0])
    elif fault == "role": records[0]["array_ref"]["sha256"] = "e" * 64
    elif fault == "bank": facts["bank_membership"] = {"another": facts["bank_membership"]["baseline"]}
    elif fault == "query": c["query"]["times"][10] += .01
    elif fault == "model": c["model"]["weights_sha256"] = "e" * 64
    else: c["query"]["indices"] = np.arange(80, dtype=np.uint64)
    result = run(case, lambda _: pytest.fail("gate must precede callback"))
    assert result["status"] == "UNAVAILABLE"


def test_native_proof_other_bank_is_not_an_extra_observation():
    case = fixture(); facts = case[2]
    other = deepcopy(facts["all_physical_calls_and_aliases"][0]); other["key"] = "expanded/a"
    facts["all_physical_calls_and_aliases"].append(other)
    facts["bank_membership"]["expanded"] = [other["key"]]
    assert run(case)["physical_calls_validated"] == 2


@pytest.mark.parametrize("authenticated", [True, False])
def test_nested_alias_retains_only_authenticated_same_sha_original_pointer(authenticated):
    keys = ("baseline/original", "baseline/nested-alias")
    case = fixture(keys=keys, names={key: ["mapillary:0"] for key in keys})
    _, records, facts, regions = case
    original_ref = deepcopy(records[0]["array_ref"])
    records[1]["array_ref"]["sha256"] = original_ref["sha256"]
    facts["all_physical_calls_and_aliases"][1]["array_ref"] = deepcopy(records[1]["array_ref"])
    artifact = regions[keys[1]]["audit"]["array_artifact"]
    artifact.update(original_ref)
    if not authenticated: artifact["path"] += ".not-in-native-inventory"
    result = run(case)
    assert (result["status"] == "complete_research_bank") is authenticated
    if authenticated:
        proof = result["inputs"][keys[1]]
        assert proof["source"]["array_ref"] == records[1]["array_ref"]
        assert proof["metadata_array_pointer"] == original_ref
        assert result["physical_calls_validated"] == 2 and len(result["mapping"][0]["occurrences"]) == 2
    else:
        assert "not an authenticated same-SHA" in result["reason"]
        assert "raw_similarities" not in result


@pytest.mark.parametrize("fault", ["coordinate", "descriptor", "missing_descriptor", "envelope", "claimed_tensor", "array_header", "array_role", "raw_identity", "loader", "loader_eof"])
def test_any_conflict_declines_whole_bank_without_partial_arrays(fault):
    keys = ("baseline/a", "baseline/b")
    case = fixture(keys=keys, names={k: ["mapillary:0"] for k in keys})
    region = case[3][keys[1]]; a, z = region["audit"], region["arrays"]
    if fault == "coordinate": z["ref_latlon"][0, 0] += .01; z["returned_latlon"][:] = z["ref_latlon"][0]
    elif fault == "descriptor": region["descriptor_rows"][0, 1] = 1.; a["ref_embeddings_sha256"] = b.tensor_sha(region["descriptor_rows"])
    elif fault == "missing_descriptor": region.pop("descriptor_rows")
    elif fault == "envelope": z["raw_similarities"][:] += np.float32(2e-6); z["decoded_cosines"][:] = z["raw_similarities"][:, 0]; z["returned_cosines"][:] = z["decoded_cosines"]
    elif fault == "claimed_tensor": a["ref_embeddings_sha256"] = "e" * 64
    elif fault == "array_header": a["array_artifact"]["arrays"]["raw_similarities"]["sha256"] = "e" * 64
    elif fault == "array_role": a["array_artifact"]["path"] += ".wrong"
    elif fault == "raw_identity": a["query_pixels_sha256"] = "e" * 64
    if fault in {"coordinate", "envelope"}: reseal(case[1][1], region)
    def loader(record):
        if fault == "loader" and record["key"] == keys[1]: raise RuntimeError("private load failure")
        if fault == "loader_eof" and record["key"] == keys[1]: raise EOFError("private incomplete load")
        return case[3][record["key"]]
    result = run(case, loader)
    assert result["status"] == "UNAVAILABLE" and "raw_similarities" not in result and "mapping" not in result


def test_whole_tensor_hash_uses_original_compact_torch_contract():
    import hashlib
    rows = np.zeros((1, 8448), np.float32)
    assert b.tensor_sha(rows) == hashlib.sha256(b'[[1,8448],"torch.float32"]' + rows.tobytes()).hexdigest()


def test_unique_only_pool_keeps_old_optional_descriptor_semantics():
    assert run(fixture(descriptors=False))["status"] == "complete_research_bank"


def test_more_than_six_calls_are_supported_but_physical_call_cap_is_pre_callback():
    keys = tuple(f"baseline/{i:02}" for i in range(40))
    case = fixture(keys=keys, descriptors=False)
    assert run(case)["physical_calls_validated"] == 40
    c, records, facts, _ = fixture(keys=tuple(f"baseline/{i:02}" for i in range(65)), descriptors=False)
    result = b.admit_bank("baseline", records, c, facts, lambda _: pytest.fail("cap before callback"))
    assert result["status"] == "UNAVAILABLE"


def test_unique_cap_never_silently_truncates_columns():
    keys = tuple(f"baseline/{i}" for i in range(7))
    names = {key: [f"mapillary:{i * 1500 + j}" for j in range(1500)] for i, key in enumerate(keys)}
    case = fixture(keys=keys, names=names, descriptors=False)
    assert run(case)["status"] == "UNAVAILABLE"


def test_input_arrays_and_metadata_are_unchanged():
    case = fixture(); before = deepcopy(case)
    assert run(case)["status"] == "complete_research_bank"
    for key, region in case[3].items():
        assert region["audit"] == before[3][key]["audit"]
        assert all(np.array_equal(v, before[3][key]["arrays"][name]) for name, v in region["arrays"].items())
