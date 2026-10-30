"""Tiny cross-region search/comparison/admission proofs, no real inputs."""
from copy import deepcopy
from collections import Counter
import numpy as np
import pytest
from monocular_osm import metadata_union as m


def comparison(values, coords, ids, q=None):
    values = np.asarray(values, np.float32).astype(float)
    q = len(values) if q is None else q
    return {"statistic": m.STATISTIC, "model": {"name": "megaloc", "hub": m.PROFILE["backbone_hub"], "device": "cpu", "weights_sha256": "c"*64},
        "query": {"indices": list(range(q)), "times": (np.arange(q)*5.).tolist(), "mode": "warp",
            "preprocessing": {"dimensions": 8448, "resize": [512, 512]}, "pixels_sha256": "a"*64, "descriptors_sha256": "b"*64},
        "inner_decoder": {"actual": "viterbi", "fallback": False, "settings": {"requested": True, "query_dt_s": 5., "maximum_viterbi_references": 6000}},
        "decoded_reference_cosines": values.tolist(), "decoded_reference_latlon": np.asarray(coords).tolist(), "decoded_reference_ids": list(ids),
        "selected_audit_rows": [[0,i] for i in range(q)]}


def summary(c, expanded=False):
    result = {"status": "complete", "route_valid": True, "point_count": len(c["query"]["indices"]), "score_kind": "raw_cosine",
        "score_identity": m.json_sha({k: c[k] for k in ("statistic", "model", "query", "inner_decoder")}),
        "similarity_median": float(np.median(c["decoded_reference_cosines"]))}
    if expanded: result["dominant_coordinate_fraction"] = max(Counter(map(tuple, c["decoded_reference_latlon"])).values())/result["point_count"]
    return result


def coords(xs):
    return np.column_stack([np.full(len(xs), 40.), -80.+np.asarray(xs)/85000])


def make_region(key, raw, positions, names, *, chosen=None):
    raw = np.ascontiguousarray(raw, np.float32); q, n = raw.shape
    ids = np.array(["mapillary:"+v for v in names]); ll = np.asarray(positions, np.float64)
    indices, times = np.arange(q), np.arange(q)*5.
    if chosen is None: chosen = np.array(m.full.decode_full_tracks(raw, ids, ll, times)["outputs"]["raw"]["chosen_ref_indices"])
    chosen = np.asarray(chosen, np.int64); selected = raw[np.arange(q), chosen]
    arrays = {"query_indices": indices, "query_times": times, "ref_ids": ids, "ref_latlon": ll, "raw_similarities": raw,
        "decoded_ref_indices": chosen, "decoded_cosines": selected.copy(), "returned_indices": indices.copy(),
        "returned_latlon": ll[chosen], "returned_cosines": selected.astype(float)}
    c = comparison(selected, ll[chosen], ids[chosen])
    decoder = {"requested": True, "actual": "viterbi", "query_dt_s": 5., "maximum_viterbi_references": 6000, "fallback_reason": None}
    audit = {"status": "complete", "issues": [], "score_kind": m.STATISTIC, "backbone": {k: v for k, v in c["model"].items() if k != "weights_sha256"},
        "actual_model_state_sha256": "c"*64, "query_mode": "warp", "preprocessing": c["query"]["preprocessing"],
        "query_pixels_sha256": "a"*64, "query_descriptors_sha256": "b"*64, "decoder": decoder,
        "reference_encoding_identity": "unverified legacy reference fingerprint", "reference_pool": {"source": "mapillary", "count": n,
            "fetch": {"listing": "tiles", "listing_stats_known": True, "lost": 0, "snapshot": {"n_refs": n}}}}
    identity = {k: audit[k] for k in ("query_pixels_sha256", "query_descriptors_sha256", "actual_model_state_sha256", "backbone", "preprocessing", "query_mode", "score_kind", "decoder")}
    identity.update(query_indices_sha256=m.full._array_sha(indices), query_times_sha256=m.full._array_sha(times))
    audit.update(comparison_identity=identity, comparison_identity_sha256=m.json_sha(identity))
    return {"key": key, "audit": audit, "arrays": arrays}


def case():
    q = 4
    seed = make_region("seed", np.full((q, 1), .1), coords([-4000]), ["base"])
    raw = np.column_stack([np.eye(q)*.8+.1, np.full(q, .95)])
    gallery = make_region("other", raw, coords([0, 100, 200, 300, 5000]), ["a0", "a1", "a2", "a3", "hub"])
    baseline = comparison(np.full(q, .1), np.repeat(seed["arrays"]["ref_latlon"], q, axis=0), ["mapillary:base"]*q)
    metadata = comparison(np.full(q, .95), np.repeat(gallery["arrays"]["ref_latlon"][-1:], q, axis=0), ["mapillary:hub"]*q)
    metadata["selected_audit_rows"] = [[1, i] for i in range(q)]
    return [seed, gallery], baseline, summary(baseline), metadata, summary(metadata, True), [seed]


def test_fixed_reference_mean_recovers_distributed_evidence_and_raw_is_not_primary():
    result = m.replay_metadata_union(*case())
    assert result["status"] == "complete" and result["selection"] == "metadata_union_reference_mean"
    assert result["outputs"]["raw"]["chosen_ref_ids"] == ["mapillary:hub"]*4
    assert result["outputs"]["reference_mean"]["chosen_ref_ids"] == ["mapillary:a"+str(i) for i in range(4)]
    primary = result["admission"]["sole_primary_summary"]
    assert primary["dominant_coordinate_fraction"] == .25 and primary["similarity_median"] > .89
    assert result["comparisons"]["raw"]["similarity_median"] > primary["similarity_median"]
    assert result["comparisons"]["reference_mean"]["actual_decoder_profile_sha256"] == m.full.PROFILE_SHA256
    assert primary["score_identity"] != result["admission"]["baseline_original"]["score_identity"]


def test_joint_path_contains_reference_pruned_by_every_regional_winner():
    seed, baseline, *_ = case()
    a = make_region("A", [[.95, .4], [0., .4], [0., .4], [0., .4]], coords([0, 2000]), ["a0", "a1"])
    b = make_region("B", [[0., .4], [.8, .4], [.8, .4], [.8, .4]], coords([100, 2100]), ["b0", "b1"])
    union = m.build_union([seed[0], a, b], baseline)
    result = m.full.decode_full_tracks(union["raw_similarities"], union["ref_ids"], union["ref_latlon"], union["query_times"])
    first = result["outputs"]["raw"]["chosen_ref_ids"][0]
    regional_first = {r["arrays"]["ref_ids"][r["arrays"]["decoded_ref_indices"][0]] for r in (seed[0], a, b)}
    assert first == "mapillary:a0" and first not in regional_first


def test_selected_raw_proof_maps_exact_original_pool_column_and_query():
    regions, *rest = case(); result = m.replay_metadata_union(regions, *rest)
    by_key = {r["key"]: r for r in regions}
    for proof in result["comparisons"].values():
        for row in proof["selected_source_rows"]:
            source = by_key[row["region"]]["arrays"]
            assert row["raw_cosine"] == float(source["raw_similarities"][row["query_row"], row["region_column"]])
            assert row["ref_id"] == source["ref_ids"][row["region_column"]]
    assert result["admission"]["decoder_identity_not_forged"] is True


def test_already_admitted_metadata_wins_without_union_or_decode(monkeypatch):
    _, baseline, bsum, metadata, *_ = case()
    metadata["decoded_reference_latlon"] = coords([0, 100, 200, 300]).tolist()
    msum = summary(metadata, True)
    monkeypatch.setattr(m.full, "decode_full_tracks", lambda *_: pytest.fail("admitted legacy metadata must bypass union decode"))
    result = m.replay_metadata_union([], baseline, bsum, metadata, msum)
    assert result["selection"] == "legacy_metadata" and result["outputs"] == {} and result["union_decoded"] is False


def test_raw_diagnostic_cannot_rescue_a_declined_fixed_primary(monkeypatch):
    args = case(); result = m.replay_metadata_union(*args)
    full_result = {**result["full_decoder_evidence"], "outputs": deepcopy(result["outputs"])}
    full_result["outputs"]["raw"], full_result["outputs"]["reference_mean"] = full_result["outputs"]["reference_mean"], full_result["outputs"]["raw"]
    for name, value in full_result["outputs"].items(): value["variant"] = name
    monkeypatch.setattr(m.full, "decode_full_tracks", lambda *_: full_result)
    declined = m.replay_metadata_union(*args)
    assert declined["comparisons"]["raw"]["dominant_coordinate_fraction"] == .25
    assert declined["comparisons"]["reference_mean"]["dominant_coordinate_fraction"] == 1.
    assert declined["selection"] == "baseline_or_existing_municipality_fallback"


@pytest.mark.parametrize("bad", ["partial", "query", "model", "rawNaN", "unserved", "winner"])
def test_incomplete_incomparable_or_conflicting_regions_refuse_without_filtering(bad):
    regions, baseline, *_ = case(); region = regions[1]
    if bad == "partial": region["audit"]["status"] = "partial"
    if bad == "query": region["arrays"]["query_times"][1] += .1
    if bad == "model": region["audit"]["actual_model_state_sha256"] = "d"*64
    if bad == "rawNaN": region["arrays"]["raw_similarities"][0, 0] = np.nan
    if bad == "unserved": region["audit"]["reference_pool"]["fetch"]["lost"] = 1
    if bad == "winner": region["arrays"]["returned_cosines"][0] = 0
    with pytest.raises(ValueError): m.build_union(regions, baseline)


def test_overlaps_need_actual_identical_descriptors_not_hash_claims():
    regions, baseline, *_ = case(); a = regions[0]; b = deepcopy(a); b["key"] = "overlap"
    a["audit"]["reference_row_embeddings_sha256"] = ["a"*64]; b["audit"]["reference_row_embeddings_sha256"] = ["a"*64]
    with pytest.raises(ValueError, match="descriptor"): m.build_union([a, b], baseline)
    for r in (a, b): r["descriptor_rows"] = np.ones((1, 8448), np.float32)
    result = m.build_union([b, a], baseline)
    assert len(result["ref_ids"]) == 1 and result["reference_occurrences"] == 2
    b["descriptor_rows"][0, 0] = 2
    with pytest.raises(ValueError, match="descriptor"): m.build_union([a, b], baseline)


def test_coordinate_conflict_and_all_occurrence_score_envelope_refuse():
    regions, baseline, *_ = case(); a = regions[0]; b = deepcopy(a); b["key"] = "overlap"
    for r in (a, b): r["descriptor_rows"] = np.ones((1, 8448), np.float32)
    b["arrays"]["ref_latlon"][0, 0] += .001
    b["arrays"]["returned_latlon"] = b["arrays"]["ref_latlon"][b["arrays"]["decoded_ref_indices"]].copy()
    with pytest.raises(ValueError, match="coordinates"): m.build_union([a, b], baseline)
    b["arrays"]["ref_latlon"] = a["arrays"]["ref_latlon"].copy()
    b["arrays"]["returned_latlon"] = b["arrays"]["ref_latlon"][b["arrays"]["decoded_ref_indices"]].copy()
    b["arrays"]["raw_similarities"] += np.float32(.000002)
    b["arrays"]["decoded_cosines"] = b["arrays"]["raw_similarities"][:, 0].copy()
    b["arrays"]["returned_cosines"] = b["arrays"]["decoded_cosines"].astype(float)
    with pytest.raises(ValueError, match="envelope"): m.build_union([a, b], baseline)


@pytest.mark.parametrize("field", ["similarity_median", "dominant_coordinate_fraction", "score_identity", "status"])
def test_old_scalar_claims_must_match_actual_raw_evidence(field):
    args = list(case()); args[4][field] = np.nan if field != "status" else "partial"
    with pytest.raises(ValueError): m.replay_metadata_union(*args)


def test_no_smoothing_loophole_and_gate_constants_are_unchanged():
    result = m.replay_metadata_union(*case())
    for name, proof in result["comparisons"].items():
        assert proof["dominant_coordinate_fraction"] == max(Counter(map(tuple, result["outputs"][name]["raw_route_latlon"])).values())/4
    assert m.PROFILE["admission_policy_config_sha256"] == m.policy.CONFIG_SHA256
    assert m.PROFILE["sole_new_primary"] == "reference_mean" and m.PROFILE["candidate_pruning"] is False


def test_independent_baseline_gallery_is_proved_without_union_membership():
    args = list(case()); args[0] = args[0][1:]
    args[3]["selected_audit_rows"] = [[0, i] for i in range(4)]
    result = m.replay_metadata_union(*args)
    assert "mapillary:base" not in result["union"]["ref_ids"].tolist()
    assert all(r["ref_id"] == "mapillary:base" and r["region"] == "seed"
        for r in result["admission"]["original_selected_sources"]["baseline"])
    assert result["selection"] == "metadata_union_reference_mean"


@pytest.mark.parametrize("bad", ["missing", "raw", "identity"])
def test_original_baseline_requires_own_actual_pool_before_new_decode(monkeypatch, bad):
    args = list(case())
    if bad == "missing": args[-1] = None
    elif bad == "raw": args[-1][0]["arrays"]["decoded_cosines"][0] = 0
    else: args[-1][0]["audit"]["query_pixels_sha256"] = "d"*64
    monkeypatch.setattr(m.full, "decode_full_tracks", lambda *_: pytest.fail("invalid baseline proof must fail before DP"))
    with pytest.raises(ValueError): m.replay_metadata_union(*args)


@pytest.mark.parametrize("score,dominant,expected", [(.162,.25,True),(.161999,.25,False),(.162,.250001,False),
    (float("nan"),.25,False),(.9,float("nan"),False)])
def test_existing_gate_thresholds_are_inclusive_and_nonfinite_declines(score, dominant, expected):
    baseline = {"similarity_median": .14, "score_identity": "a"*64}
    candidate = {"status": "complete", "route_valid": True, "point_count": 80, "score_kind": "raw_cosine",
        "score_identity": "a"*64, "similarity_median": score, "dominant_coordinate_fraction": dominant}
    assert (m.legacy_gate(baseline,candidate)["selected_candidate"] == "expanded") is expected


@pytest.mark.parametrize("mixed",[False,True])
def test_baseline_selected_rows_can_use_second_or_mixed_independent_pools(mixed):
    args=list(case()); first=args[-1][0]
    second=make_region("fine",np.full((4,1),.11),coords([-8000]),["fine"],chosen=np.zeros(4,int))
    rows=[[i%2 if mixed else 1,i] for i in range(4)]; pools=[first,second]
    ids=[];ll=[];scores=[]
    for audit,row in rows:
        a=pools[audit]["arrays"];col=int(a["decoded_ref_indices"][row])
        ids.append(str(a["ref_ids"][col]));ll.append(a["ref_latlon"][col]);scores.append(a["raw_similarities"][row,col])
    args[1]=comparison(scores,ll,ids);args[1]["selected_audit_rows"]=rows;args[2]=summary(args[1]);args[-1]=pools
    result=m.replay_metadata_union(*args)
    assert [v["audit_index"] for v in result["admission"]["original_selected_sources"]["baseline"]]==[v[0] for v in rows]
    assert "mapillary:fine" not in result["union"]["ref_ids"]


def test_bad_unused_baseline_pool_is_not_silently_filtered():
    args=list(case());bad=deepcopy(args[-1][0]);bad["key"]="unused";bad["audit"]["status"]="partial"
    args[-1].append(bad)
    with pytest.raises(ValueError,match="complete"):m.replay_metadata_union(*args)


def duplicate_occurrence_case(q=4, changed_rows=None):
    seed = make_region("seed", np.full((q, 1), .1, np.float32), coords([-4000]), ["base"], chosen=np.zeros(q, int))
    baseline = comparison(np.full(q, .1), seed["arrays"]["ref_latlon"][[0]*q], ["mapillary:base"]*q)
    canonical = make_region("a_canonical", np.full((q, 1), .2, np.float32), coords([0]), ["duplicate"], chosen=np.zeros(q, int))
    selected = deepcopy(canonical); selected["key"] = "b_selected"
    matrix = selected["arrays"]["raw_similarities"]
    rows = np.arange(q) if changed_rows is None else np.asarray(changed_rows)
    matrix[rows] = np.nextafter(matrix[rows], np.float32(np.inf))
    selected["arrays"]["decoded_cosines"] = matrix[:, 0].copy()
    selected["arrays"]["returned_cosines"] = matrix[:, 0].astype(np.float64)
    for region in (canonical, selected):region["descriptor_rows"] = np.ones((1, 8448), np.float32)
    original = comparison(matrix[:, 0], selected["arrays"]["ref_latlon"][[0]*q], ["mapillary:duplicate"]*q)
    original["selected_audit_rows"] = [[1, i] for i in range(q)]
    return [[canonical, selected], baseline, summary(baseline), original, summary(original, True), [seed]]


def test_duplicate_float32_rounding_keeps_exact_original_and_canonical_proofs_separate():
    args = duplicate_occurrence_case(); before = deepcopy(args)
    result = m.replay_metadata_union(*args)
    canonical, selected = args[0]
    assert result["status"] == "complete"
    np.testing.assert_array_equal(result["union"]["raw_similarities"], canonical["arrays"]["raw_similarities"])
    assert not np.array_equal(canonical["arrays"]["raw_similarities"], selected["arrays"]["raw_similarities"])
    for q, row in enumerate(result["admission"]["original_selected_sources"]["metadata"]):
        assert row["audit_index"] == 1 and row["audit_query_row"] == q and row["region"] == "b_selected"
        assert row["raw_cosine"] == args[3]["decoded_reference_cosines"][q] == float(selected["arrays"]["raw_similarities"][q, 0])
        assert row["canonical_union_source"] == {"region": "a_canonical", "region_column": 0,
            "raw_cosine": float(canonical["arrays"]["raw_similarities"][q, 0])}
        assert row["canonical_union_source"]["raw_cosine"] != row["raw_cosine"]
    assert args[3] == before[3] and args[4] == before[4]
    for actual, old in zip(args[0], before[0]):
        for name in actual["arrays"]:np.testing.assert_array_equal(actual["arrays"][name], old["arrays"][name])


def test_duplicate_union_canonical_order_does_not_follow_selected_pool_order():
    args = duplicate_occurrence_case(); first = m.replay_metadata_union(*args)
    args[0].reverse(); args[3]["selected_audit_rows"] = [[0, i] for i in range(4)]
    second = m.replay_metadata_union(*args)
    np.testing.assert_array_equal(first["union"]["raw_similarities"], second["union"]["raw_similarities"])
    assert first["outputs"] == second["outputs"]
    assert all(r["audit_index"] == 0 and r["region"] == "b_selected" for r in second["admission"]["original_selected_sources"]["metadata"])


def test_full80_control_with22_ULP_different_duplicate_rows_matches_runtime_failure_pattern():
    changed = np.linspace(0, 79, 22).astype(int)
    args = duplicate_occurrence_case(q=80, changed_rows=changed)
    canonical, selected = args[0]
    assert np.count_nonzero(canonical["arrays"]["raw_similarities"] != selected["arrays"]["raw_similarities"]) == 22
    assert np.max(abs(canonical["arrays"]["raw_similarities"]-selected["arrays"]["raw_similarities"])) < 1e-6
    result = m.replay_metadata_union(*args)
    proof = result["admission"]["original_selected_sources"]["metadata"]
    assert result["status"] == "complete" and len(proof) == 80
    assert [q for q, row in enumerate(proof) if row["raw_cosine"] != row["canonical_union_source"]["raw_cosine"]] == changed.tolist()
    assert all(row["raw_cosine"] == args[3]["decoded_reference_cosines"][q] for q, row in enumerate(proof))
    np.testing.assert_array_equal(result["union"]["raw_similarities"], canonical["arrays"]["raw_similarities"])


def test_full80_duplicate_ULP_pattern_does_not_tolerate_a_false_original_scalar(monkeypatch):
    args = duplicate_occurrence_case(q=80, changed_rows=np.linspace(0, 79, 22).astype(int))
    args[3]["decoded_reference_cosines"][3] += 1e-12
    args[4] = summary(args[3], True)
    monkeypatch.setattr(m.full, "decode_full_tracks", lambda *_: pytest.fail("false original scalar must refuse before DP"))
    with pytest.raises(ValueError, match="selected raw evidence differs"):
        m.replay_metadata_union(*args)


@pytest.mark.parametrize("bad", ["original_cosine", "coordinate", "wrong_pool", "wrong_query", "missing_mapping",
                                "missing_descriptors", "descriptor_conflict", "score_envelope"])
def test_duplicate_occurrence_fix_keeps_exact_source_and_compatibility_guards(monkeypatch, bad):
    args = duplicate_occurrence_case(); canonical, selected = args[0]
    if bad == "original_cosine":args[3]["decoded_reference_cosines"][0] += 1e-12; args[4] = summary(args[3], True)
    elif bad == "coordinate":args[3]["decoded_reference_latlon"][0][0] += 1e-12
    elif bad == "wrong_pool":args[3]["selected_audit_rows"][0] = [0, 0]
    elif bad == "wrong_query":args[3]["selected_audit_rows"][1] = [1, 0]
    elif bad == "missing_mapping":args[3].pop("selected_audit_rows")
    elif bad == "missing_descriptors":selected.pop("descriptor_rows")
    elif bad == "descriptor_conflict":selected["descriptor_rows"][0, 0] += 1
    else:
        selected["arrays"]["raw_similarities"] += np.float32(2e-6)
        selected["arrays"]["decoded_cosines"] = selected["arrays"]["raw_similarities"][:, 0].copy()
        selected["arrays"]["returned_cosines"] = selected["arrays"]["decoded_cosines"].astype(np.float64)
        args[3]["decoded_reference_cosines"] = selected["arrays"]["decoded_cosines"].astype(np.float64).tolist()
        args[4] = summary(args[3], True)
    monkeypatch.setattr(m.full, "decode_full_tracks", lambda *_: pytest.fail("invalid exact original/duplicate evidence must refuse before DP"))
    with pytest.raises(ValueError):m.replay_metadata_union(*args)
