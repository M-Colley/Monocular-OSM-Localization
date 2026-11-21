"""Replay frozen all-branch image evidence without frames, models, or labels."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from monocular_osm import poi_image_union as union
from monocular_osm.provenance import file_sha256
from monocular_osm.stages.poi_only import _safe
from scripts import assisted_goal_audit as audit
from tests.test_poi_image_union import VIDEO, evidence, inputs


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(value), allow_nan=False), encoding="utf8")


def union_case(tmp_path, count=2):
    base = tmp_path / "union"
    base.mkdir()
    clues, ocr = inputs(count)
    plan = union.plan_image_union(clues, ocr)
    write(base / "image_union_plan.json", plan)
    times = np.arange(100, dtype=np.float64) * .3
    native = tmp_path / "cues_vo.npz"
    np.savez(native, timestamps=times)
    runtime = {"device": "cpu", "intra_op_threads": 4, "inter_op_threads": 1, "torch_version": "synthetic"}
    records, raw_by_id = [], {}
    for index, region in enumerate(plan["region_plans"]):
        ids = ["mapillary:shared", f"mapillary:region_{index}"]
        scores = np.column_stack([np.full(80, .2),
            np.linspace(.1, .7, 80) if index == 0 else np.linspace(.7, .1, 80)]).astype(np.float32)
        raw = evidence(region, ids, scores, timestamps=times)
        path = base / "poi_evidence" / f"{index:04d}_pool.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **{key: value for key, value in raw.items() if key not in {"status", "provenance"}})
        record = {"path": str(path.relative_to(base)), "sha256": file_sha256(path)}
        raw["provenance"].update(matrix_artifact=record, cpu_runtime=runtime,
            reference_identities={key: {"encoding_identity": "UNVERIFIED"} for key in ids})
        raw_by_id[region["hypothesis_id"]] = raw
        records.append(record)
    candidate = union.localize_image_union(plan, timestamps=times, query_video_sha256=VIDEO,
        retrieve_evidence=lambda region, **kwargs: raw_by_id[region["hypothesis_id"]])
    assert candidate["status"] == "complete", candidate
    arrays = candidate["reference_union"]
    path = base / "image_reference_union.npz"
    np.savez_compressed(path, **{key: np.asarray(arrays[key]) for key in
        ("query_indices", "query_times", "ref_ids", "ref_latlon", "similarities")})
    record = {"path": path.name, "sha256": file_sha256(path)}
    candidate["reference_union"] = {key: value for key, value in arrays.items() if key not in
        {"query_indices", "query_times", "ref_latlon", "similarities"}}
    candidate["reference_union"]["matrix_artifact"] = record
    auto = {"stride": 3, "fps": 10., "n_frames": len(times), "window": [0., 160.],
        "timestamps_sha256": union._array_sha(times), "vo_artifact": {"path": str(native), "sha256": file_sha256(native)}}
    candidate.update(image_union_plan=plan, evidence_artifacts=[*records, record],
        acquisition={"stride": 3, "n_frames": len(times), "timestamps_sha256": union._array_sha(times),
            "native_frame_shape_hw": ocr["frame_shape_hw"]},
        candidate_artifact={"path": str(base / "candidate.json")}, cpu_runtime=runtime,
        actual_backbone=next(iter(raw_by_id.values()))["provenance"]["backbone"], actual_model_state={"sha256": "f"*64})
    candidate = _safe(candidate)
    cue = {"clues": clues, "ocr_provenance": ocr, "auto_quality": auto, "window": [0., 160.]}
    ledger = {path: file_sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    return SimpleNamespace(child=candidate, cue=cue, ledger=ledger, files=audit.Files(tmp_path),
        inputs={"video": {"sha256": VIDEO}, "registered_window": [0., 160.]}, base=base)


def check(case):
    audit.image_union_child(case.files, case.child, case.ledger, case.inputs, case.cue)


def rehash(case, record):
    path = case.base / record["path"]
    record["sha256"] = file_sha256(path)
    case.ledger[path] = record["sha256"]


@pytest.mark.parametrize("count", [1, 2, 6])
def test_frozen_complete_union_replays_all_regions_and_every_reference(tmp_path, count):
    case = union_case(tmp_path, count)
    assert case.child["reference_union"]["reference_count"] == count + 1
    check(case)
    assert case.child["candidate_pruning"] is False and case.child["uses_rigid_vo"] is False


@pytest.mark.parametrize("field", ["route_latlon", "chosen_ref_ids", "objective", "candidate_pruning",
    "uses_rigid_vo", "profile_sha256", "successful_branch_count", "score_normalization"])
def test_rehashed_claims_cannot_change_full_dp_result_or_fixed_profile(tmp_path, field):
    case = union_case(tmp_path)
    if field == "route_latlon": case.child[field][0][0] += .001
    elif field == "chosen_ref_ids": case.child[field][0] = "mapillary:foreign"
    elif field == "objective": case.child[field] += .01
    elif field == "profile_sha256": case.child[field] = "0"*64
    elif field == "score_normalization": case.child[field] = "raw"
    elif field == "successful_branch_count": case.child[field] = 1
    else: case.child[field] = True
    with pytest.raises(ValueError): check(case)


@pytest.mark.parametrize("field", ["n_frames", "stride", "timestamps_sha256", "native_frame_shape_hw", "window"])
def test_native_grid_and_bbox_must_match_frozen_cues(tmp_path, field):
    case = union_case(tmp_path)
    if field == "window": case.cue["window"][1] += 1
    else: case.child["acquisition"][field] = None
    with pytest.raises(ValueError, match="native cue grid"): check(case)


@pytest.mark.parametrize("field", ["memberships", "canonical_occurrences", "reference_identities", "hashes"])
def test_union_membership_canonical_order_and_legacy_unknown_claims_are_bound(tmp_path, field):
    case = union_case(tmp_path)
    case.child["reference_union"][field] = {}
    with pytest.raises(ValueError, match="membership"): check(case)


@pytest.mark.parametrize("field", ["query_identity", "cpu_runtime", "backbone"])
def test_each_region_must_bind_actual_child_query_model_and_cpu_runtime(tmp_path, field):
    case = union_case(tmp_path)
    provenance = case.child["regions"][1]["provenance"]
    if field == "query_identity": provenance[field]["actual_model_state_sha256"] = "0"*64
    elif field == "cpu_runtime": provenance[field]["intra_op_threads"] = 8
    else: provenance[field]["device"] = "cuda"
    with pytest.raises(ValueError, match="model/runtime"): check(case)


@pytest.mark.parametrize("fault", ["omitted_branch", "failed_branch", "reordered", "omitted_matrix", "extra_matrix", "aliased_matrix", "unfrozen_plan"])
def test_every_branch_plan_and_exact_artifact_set_is_required(tmp_path, fault):
    case = union_case(tmp_path)
    if fault == "omitted_branch": case.child["regions"].pop()
    elif fault == "failed_branch": case.child["regions"][1]["status"] = "failed"
    elif fault == "reordered": case.child["regions"].reverse()
    elif fault == "omitted_matrix": case.child["evidence_artifacts"].pop(0)
    elif fault == "extra_matrix": case.child["evidence_artifacts"].append(deepcopy(case.child["evidence_artifacts"][0]))
    elif fault == "aliased_matrix":
        case.child["regions"][1]["provenance"]["matrix_artifact"] = deepcopy(case.child["regions"][0]["provenance"]["matrix_artifact"])
    else: del case.ledger[case.base / "image_union_plan.json"]
    with pytest.raises(ValueError): check(case)


@pytest.mark.parametrize("role,field", [("union", "similarities"), ("union", "ref_ids"), ("union", "query_indices"),
    ("union", "ref_latlon"), ("region", "query_times"), ("region", "similarities"), ("union", "dtype")])
def test_rehashed_matrix_content_still_must_match_complete_replay(tmp_path, role, field):
    case = union_case(tmp_path)
    record = case.child["reference_union"]["matrix_artifact"] if role == "union" else case.child["regions"][0]["provenance"]["matrix_artifact"]
    path = case.base / record["path"]
    arrays = audit.load_npz(path)
    if field == "ref_ids": arrays[field][0] = "mapillary:foreign"
    elif field == "dtype": arrays["similarities"] = arrays["similarities"].astype(np.float64)
    elif field in {"query_indices", "query_times"}: arrays[field][0] += 1
    else: arrays[field].flat[0] += .01
    np.savez_compressed(path, **arrays)
    rehash(case, record)
    for declared in case.child["evidence_artifacts"]:
        if declared["path"] == record["path"]: declared["sha256"] = record["sha256"]
    with pytest.raises(ValueError): check(case)


def test_v2_controller_strong_baseline_keeps_original_route_and_goal_passes(tmp_path, monkeypatch):
    from tests import test_check_assisted_goal as legacy
    config = legacy.PipelineConfig
    monkeypatch.setattr(legacy, "PipelineConfig", lambda *a, **k: config(*a, **k, assisted_profile="v2"))
    case = legacy.fixture(tmp_path, monkeypatch)
    assert case.ctx.result["assisted_audit"]["policy_version"].endswith("v2")
    assert [record["kind"] for record in case.ctx.result["assisted_audit"]["reached_stages"]] == ["baseline"]
    assert case.ctx.result["answer"]["route_latlon"] == case.children["baseline"]["route_latlon"]
    report = legacy.check(case)
    assert report["goal_met"], legacy.reasons(report)


def test_unknown_assisted_profile_declines_before_artifact_access():
    result = audit.audit_assisted({"provenance": {"config": {"assisted_profile": "foreign"},
        "assisted_inputs": {}, "assisted_audit": {}}}, {}, "city", "0:1",
        poi_issues=None, filter_issues=None, reader_issues=None, ocr_identity=None)
    assert "assisted_profile must be v1, v2, v3 or v4" in str(result["issues"])


def test_historical_v1_freeze_without_profile_field_remains_auditable(tmp_path, monkeypatch):
    from tests import test_check_assisted_goal as legacy
    from monocular_osm import assisted_worker as worker, provenance
    from monocular_osm.stages import assisted as stage
    original = provenance.config_snapshot
    def historical_snapshot(cfg):
        result = original(cfg)
        result.pop("assisted_profile", None)
        return result
    for module in (legacy, worker, stage):
        monkeypatch.setattr(module, "config_snapshot", historical_snapshot)
    case = legacy.fixture(tmp_path, monkeypatch)
    assert "assisted_profile" not in case.ctx.result["assisted_inputs"]["config"]
    report = legacy.check(case)
    assert report["goal_met"], legacy.reasons(report)


@pytest.mark.parametrize("profile,kind", [("v1", "poi_image_union"), ("v2", "poi_image_track")])
def test_profile_cannot_substitute_the_other_image_component_before_file_access(tmp_path, profile, kind):
    with pytest.raises(ValueError, match="incompatible reached stage"):
        audit.verify_child(audit.Files(tmp_path), {"kind": kind}, {"config": {"assisted_profile": profile}}, {})


@pytest.mark.parametrize("fault", ["policy_source", "omitted_source", "v1_policy"])
def test_v2_goal_binds_new_policy_and_all_union_source_modules(tmp_path, monkeypatch, fault):
    from tests import test_check_assisted_goal as legacy
    from monocular_osm import assisted_policy as v1
    config = legacy.PipelineConfig
    monkeypatch.setattr(legacy, "PipelineConfig", lambda *a, **k: config(*a, **k, assisted_profile="v2"))
    case = legacy.fixture(tmp_path, monkeypatch)
    inputs = case.ctx.result["assisted_inputs"]
    source_key = lambda name: next(key for key in inputs["source_sha256"] if key.replace("\\", "/") == name)
    if fault == "policy_source":
        inputs["policy"]["source_sha256"] = inputs["source_sha256"][source_key("monocular_osm/assisted_policy.py")]
    elif fault == "omitted_source":
        del inputs["source_sha256"][source_key("monocular_osm/poi_image_union.py")]
    else:
        inputs["policy"].update(version=v1.POLICY_VERSION, config=v1.policy_config(), config_sha256=v1.CONFIG_SHA256)
    legacy.refresh_parent(case)
    report = legacy.check(case)
    assert not report["goal_met"]
    assert "assisted frozen-input audit" in legacy.reasons(report)


@pytest.mark.parametrize("count", [0, 1, 29])
@pytest.mark.parametrize("listing", ["graph", None, "tiles"])
def test_completed_sparse_search_requires_known_untruncated_tile_listing(count, listing):
    from tests.test_check_assisted_goal import sparse_record
    record = sparse_record(count)
    proof = record["track_decoding"]["terminal_search"]
    proof["listing"]["listing"] = listing
    if listing == "tiles":
        assert audit.terminal_search(record)
        proof["listing"]["truncated"] = True
    with pytest.raises(ValueError, match="sparse listing is partial"):
        audit.terminal_search(record)
