"""Frozen controller/worker evidence is replayed without models or labels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from monocular_osm import assisted_worker as worker, kartaview_vpr as kv
from monocular_osm.pipeline import PipelineConfig
from monocular_osm.provenance import config_snapshot, file_sha256
from monocular_osm.stages import assisted as stage
from monocular_osm.stages.context import RunContext
from scripts import assisted_goal_audit as audit
from tests.test_benchmark import M as benchmark
from tests.test_check_dev_goal import M, _valid


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def actual_audit(monkeypatch, score=.17, *, model="3", pixels="1"):
    n = 5
    indices = np.arange(n, dtype=np.int64)
    times = np.arange(n, dtype=np.float64)
    refs = np.c_[49. + np.arange(40) * .0001, np.full(40, 8.4)]
    scores = np.full((n, 40), .05, dtype=np.float32)
    chosen = np.arange(n, dtype=np.int64)
    scores[indices, chosen] = score
    cosines = scores[indices, chosen].copy()
    value = {"issues": [], "query_indices": indices, "query_pixels_sha256": pixels * 64,
        "query_descriptors_sha256": "2" * 64, "actual_model_state_sha256": model * 64,
        "backbone": {"name": "megaloc", "hub": kv.HUB_MEGALOC, "device": "cpu"},
        "preprocessing": {"version": 1, "warp_size": [512, 512], "descriptor_normalization": "L2"},
        "query_mode": "warp", "ref_ids": np.array([f"mapillary:{i}" for i in range(40)]),
        "ref_latlon": refs, "ref_embeddings_sha256": "4" * 64, "raw_similarities": scores,
        "reference_pool": {"source": "mapillary", "center": [49., 8.4], "radius_m": 3000., "cap": 1500}}
    monkeypatch.setattr(kv, "LAST_TRACK_STATS", {"decoder": "viterbi"})
    kv._finish_track_audit(value, indices, scores, chosen, cosines, times, True, 1.)
    value["returned_track"] = {"indices": indices, "latlon": refs[chosen], "cosines": cosines}
    return value


def fixture(tmp_path, monkeypatch, *, score=.17, model="3", pixels="1", failed_assistance=False, runtime_patch=None,
            relative_paths=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic registered encoded pixels")
    if relative_paths:
        monkeypatch.chdir(tmp_path)
    cfg = PipelineConfig(url="", city="Registered city", data_dir=Path("data/video") if relative_paths else tmp_path / "data/video",
        output_dir=Path("runs/output") if relative_paths else tmp_path / "out",
        video_path=Path("video.mp4") if relative_paths else video, answer="assisted", frame_stride=6,
        vo_start_sec=0., vo_end_sec=300.)
    cfg.output_dir.mkdir(parents=True)
    ctx = RunContext(cfg=cfg, fusion_w=None, video_path=cfg.video_path)
    returned = {}

    def child(kind, parent):
        if kind != "baseline" and failed_assistance:
            raise RuntimeError("bounded synthetic unavailable stage")
        assert kind == "baseline", "unexpected future stage"
        directory = parent.assisted_child_root / kind
        directory.mkdir(parents=True)
        child_cfg = worker.child_config(cfg, kind, parent.video_path)
        child_cfg.output_dir = directory
        request = {"kind": kind, "config": config_snapshot(child_cfg),
            "video_sha256": parent.result["assisted_inputs"]["video"]["sha256"],
            "source_sha256": parent.result["assisted_inputs"]["source_sha256"],
            "parent_output": str(cfg.output_dir.resolve()), "policy_evidence": deepcopy(parent.assisted_evidence),
            "launch_commit_headroom_bytes": 9 * 1024**3}
        write(directory / "request.json", request)
        raw = actual_audit(monkeypatch, score, model=model, pixels=pixels)
        grid = {"mode": "linspace", "dense_count": 5, "dense_indices": list(range(5)),
                "requested_sample_count": 80, "sample_count": 5}
        rawctx = SimpleNamespace(retrieval_audits=[raw], vpr_track=tuple(raw["returned_track"][k]
            for k in ("indices", "latlon", "cosines")), frames=SimpleNamespace(sampling=grid, timestamps=np.arange(5.)))
        comp = worker._comparison(rawctx, directory)
        track = [[round(float(t), 2), round(float(ll[0]), 6), round(float(ll[1]), 6), round(float(s), 4)]
                 for t, ll, s in zip(comp["query"]["times"], comp["decoded_reference_latlon"], comp["decoded_reference_cosines"])]
        physical = {"vpr_prior": {"track": track}}
        from monocular_osm.baselines import vpr_track_only
        route = vpr_track_only(physical).route_latlon
        physical["answer"] = {"route_latlon": route, "mode": "track", "method": "vpr_track_only"}
        write(directory / "result.json", physical)
        candidate = {"status": "complete", "route_latlon": route,
            "route_timestamps": list(map(float, range(5))), "comparison": comp,
            "cpu_runtime": {"device": "cpu", "intra_op_threads": 4, "inter_op_threads": 1, "torch_version": "synthetic"},
            "actual_backbone": raw["backbone"], "actual_model_state": {"sha256": model * 64},
            "model_downloads_disabled": True}
        candidate["cpu_runtime"].update(runtime_patch or {})
        response = {**worker._finish(child_cfg, request, candidate), "worker_returncode": 0}
        returned[kind] = response
        return response

    def finalize(parent):
        parent.result["answer"] = {**parent.result["assisted_position"], "mode": "assisted",
            "method": "assisted_policy", "source": "result.assisted_position"}
        parent.result["provenance"] = {"config": config_snapshot(cfg), "git": {"commit": "commit-A", "dirty": False},
            "assisted_inputs": parent.result["assisted_inputs"], "assisted_audit": parent.result["assisted_audit"]}
        write(cfg.output_dir / "result.json", parent.result)

    stage.run_assisted(ctx, finalize=finalize, run_child=child)
    result = SimpleNamespace(ctx=ctx, children=returned, finalize=finalize, output=cfg.output_dir.resolve())
    refresh_parent(result)
    return result


def refresh_parent(case):
    ctx = case.ctx
    write(ctx.cfg.output_dir / "assisted_inputs.json", ctx.result["assisted_inputs"])
    write(ctx.cfg.output_dir / "assisted_audit.json", ctx.result["assisted_audit"])
    case.finalize(ctx)
    stage.freeze(ctx)
    summary, manifest = _valid(count=1)
    summary["reps"][0]["provenance"] = benchmark._provenance_of(ctx.result)
    summary["reps"][0]["scores"]["answer"]["meta"] = {"mode": "assisted", "method": "assisted_policy"}
    case.summary, case.manifest = summary, manifest


def check(case):
    return M.check_goal(case.summary, case.manifest, expected_count=1)


def reasons(report):
    return " ".join(report["reasons"] + [r for c in report["clips"] for r in c["reasons"]])


def test_real_controller_freezes_and_full_matrix_adapter_pass_without_future_stages(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch)
    report = check(case)
    assert report["goal_met"], reasons(report)
    assert [x["kind"] for x in case.ctx.result["assisted_audit"]["reached_stages"]] == ["baseline"]
    assert "unvalidated heuristic" in " ".join(report["input_contract_notes"])


def test_failed_reached_assistance_is_retained_and_safe_baseline_fallback_can_pass(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch, score=.12, failed_assistance=True)
    report = check(case)
    assert report["goal_met"], reasons(report)
    assert [x["status"] for x in case.ctx.result["assisted_audit"]["reached_stages"]] == ["complete", "failed", "failed"]


@pytest.mark.parametrize("relative_paths", [True, False])
def test_frozen_goal_audit_is_independent_of_verifier_working_directory(tmp_path, monkeypatch, relative_paths):
    case = fixture(tmp_path / "original", monkeypatch, relative_paths=relative_paths)
    elsewhere = tmp_path / "unrelated"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = check(case)
    assert result["goal_met"], reasons(result)
    inputs, transcript = case.ctx.result["assisted_inputs"], case.ctx.result["assisted_audit"]
    files = audit.parent_files(inputs["config"], inputs, transcript)
    assert files.root == case.output
    if relative_paths:
        assert files.cwd == tmp_path / "original"
        nested = files.child(case.output / "children" / inputs["execution_id"] / "baseline")
        assert nested.cwd == files.cwd
        assert nested.path("candidate.json") == nested.root / "candidate.json"
        assert nested.config_path("data/video", strict=False) == tmp_path / "original/data/video"


@pytest.mark.parametrize("fault", ["suffix", "traversal", "execution", "tamper", "absolute", "split_roles", "relative_role"])
def test_canonical_output_anchor_never_overrides_mismatch_or_tampered_layout(tmp_path, monkeypatch, fault):
    case = fixture(tmp_path / "original", monkeypatch, relative_paths=True)
    inputs, transcript = deepcopy(case.ctx.result["assisted_inputs"]), deepcopy(case.ctx.result["assisted_audit"])
    cfg = inputs["config"]
    if fault == "suffix": cfg["output_dir"] = "different/output"
    elif fault == "traversal": cfg["output_dir"] = "../runs/output"
    elif fault == "execution": inputs["execution_id"] = "f" * 32
    elif fault == "tamper": transcript["reached_stages"][0]["artifacts"][0]["sha256"] = "f" * 64
    elif fault == "absolute":
        other = tmp_path / "different_absolute_output"; other.mkdir(); cfg["output_dir"] = str(other)
    elif fault == "relative_role": transcript["reached_stages"][0]["artifacts"][0]["path"] = "candidate.json"
    else:
        import shutil
        item = transcript["reached_stages"][0]["artifacts"][0]
        other = tmp_path / "other/runs/output/children" / inputs["execution_id"] / "baseline" / Path(item["path"]).name
        other.parent.mkdir(parents=True); shutil.copyfile(item["path"], other); item["path"] = str(other)
    elsewhere = tmp_path / "unrelated"; elsewhere.mkdir(); monkeypatch.chdir(elsewhere)
    with pytest.raises(ValueError): audit.parent_files(cfg, inputs, transcript)


@pytest.mark.parametrize("field", ["decision_trace", "decision_inputs", "evidence", "selected_route_sha256",
                                   "selected_child_artifact_sha256", "policy_version", "config_sha256"])
def test_stale_policy_transcript_or_selected_identity_is_rejected(tmp_path, monkeypatch, field):
    case = fixture(tmp_path, monkeypatch)
    transcript = case.ctx.result["assisted_audit"]
    if field in {"decision_trace", "decision_inputs"}:
        transcript[field][0] = {}
    elif field == "evidence": transcript[field] = {}
    else: transcript[field] = "f" * 64
    refresh_parent(case)
    assert not check(case)["goal_met"]


@pytest.mark.parametrize("change", ["gt", "manual", "source_omitted", "runtime", "child_rc", "answer_route",
                                    "window", "mode", "method", "unrequested_stage", "selected_partial"])
def test_malformed_or_external_assisted_contract_cannot_pass(tmp_path, monkeypatch, change):
    case = fixture(tmp_path, monkeypatch)
    inputs, transcript = case.ctx.result["assisted_inputs"], case.ctx.result["assisted_audit"]
    if change == "gt": inputs["input_policy"]["ground_truth"] = True
    elif change == "manual": inputs["input_policy"]["manual_region_coordinates"] = True
    elif change == "source_omitted": inputs["source_sha256"].pop("main.py")
    elif change == "window": inputs["registered_window"] = [0., 5.]
    elif change == "runtime": transcript["reached_stages"][0]["comparison"]["inner_decoder"]["actual"] = "argmax"
    elif change == "child_rc": transcript["reached_stages"][0]["worker_returncode"] = 1
    elif change == "selected_partial": transcript["reached_stages"][0]["status"] = "partial"
    elif change == "unrequested_stage": transcript["reached_stages"].append(deepcopy(transcript["reached_stages"][0]))
    refresh_parent(case)
    if change == "answer_route":
        case.ctx.result["answer"]["route_latlon"][0][0] += .1
        write(case.ctx.cfg.output_dir / "result.json", case.ctx.result)
        stage.freeze(case.ctx)
        case.summary["reps"][0]["provenance"] = benchmark._provenance_of(case.ctx.result)
    elif change in {"mode", "method"}:
        case.summary["reps"][0]["scores"]["answer"]["meta"][change] = "track"
    assert not check(case)["goal_met"]


def test_changed_matrix_or_source_bytes_are_rejected_before_replay(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch)
    matrix = Path(case.children["baseline"]["comparison"]["audit_artifacts"][0]["path"])
    matrix.write_bytes(b"changed frozen matrix")
    assert not check(case)["goal_met"]


@pytest.mark.parametrize("runtime", [{"device": "cuda"}, {"intra_op_threads": 5},
                                     {"inter_op_threads": 0}, {"torch_version": None}])
def test_frozen_success_still_requires_actual_cpu_runtime_proof(tmp_path, monkeypatch, runtime):
    case = fixture(tmp_path, monkeypatch, runtime_patch=runtime)
    assert not check(case)["goal_met"]


def test_goal_uses_saved_repeat_error_not_one_decimal_display_median():
    summary, manifest = _valid()
    summary["clips"]["clip00"]["methods"]["answer"]["symmetric_mean_m"]["median"] = 200.
    summary["reps"][0]["scores"]["answer"]["symmetric_mean_m"] = 200.04
    result = M.check_goal(summary, manifest)
    assert not result["goal_met"] and result["clips"][0]["symmetric_mean_m"] == 200.04
    assert result["n_passing"] == 14


@pytest.mark.parametrize("fault", ["raw_cosines", "query_pixels", "rows", "times", "sampling", "matrix_inventory"])
def test_comparison_rejects_rehashed_stale_scalar_and_query_declarations(tmp_path, monkeypatch, fault):
    raw = actual_audit(monkeypatch)
    ctx = SimpleNamespace(retrieval_audits=[raw], vpr_track=tuple(raw["returned_track"][k]
        for k in ("indices", "latlon", "cosines")), frames=SimpleNamespace(timestamps=np.arange(5.),
        sampling={"mode": "linspace", "dense_count": 5, "dense_indices": list(range(5)),
                  "requested_sample_count": 80, "sample_count": 5}))
    comp = json.loads(json.dumps(worker._comparison(ctx, tmp_path)))
    ledger = {Path(r["path"]): r["sha256"] for r in comp["audit_artifacts"]}
    assert audit.comparison(audit.Files(tmp_path), comp, ledger, [0., 300.])
    if fault == "raw_cosines": comp["decoded_reference_cosines"][0] += .01
    elif fault == "query_pixels": comp["query"]["pixels_sha256"] = "f" * 64
    elif fault == "rows": comp["selected_audit_rows"][0][1] = 1
    elif fault == "times": comp["query"]["times"][0] = -.1
    elif fault == "sampling": comp["source_sampling"]["dense_count"] = 6
    else: comp["audit_artifacts"].pop()
    with pytest.raises((ValueError, KeyError)):
        audit.comparison(audit.Files(tmp_path), comp, ledger, [0., 300.])


def test_incomplete_unused_pool_is_declined_for_complete_expansion(tmp_path, monkeypatch):
    raw, bad = actual_audit(monkeypatch), actual_audit(monkeypatch)
    bad["raw_similarities"][0, 0] += .1
    ctx = SimpleNamespace(retrieval_audits=[raw, bad], vpr_track=tuple(raw["returned_track"][k]
        for k in ("indices", "latlon", "cosines")), frames=SimpleNamespace(timestamps=np.arange(5.),
        sampling={"mode": "linspace", "dense_count": 5, "dense_indices": list(range(5)),
                  "requested_sample_count": 80, "sample_count": 5}))
    comp = json.loads(json.dumps(worker._comparison(ctx, tmp_path)))
    ledger = {Path(r["path"]): r["sha256"] for r in comp["audit_artifacts"]}
    assert audit.comparison(audit.Files(tmp_path), comp, ledger, [0., 300.])
    with pytest.raises(ValueError, match="completed listing proof"):
        audit.comparison(audit.Files(tmp_path), comp, ledger, [0., 300.], require_all=True)


def sparse_record(count):
    refs = [{"id": str(i), "lat": 49., "lon": 8.4} for i in range(count)]
    snapshot = kv._reference_snapshot(refs)
    return {"status": "unavailable", "center": [49., 8.4], "radius_m": 3000., "ref_cap": 1500,
        "track_decoding": {"terminal_search": {"status": "complete_empty" if count == 0 else "complete_sparse",
            "count": count, "source": "mapillary", "snapshot": snapshot, "reference_inventory": refs,
            "fetch_signature": kv._fetch_signature([49., 8.4], 3000., 1500),
            "listing": {"listing": "tiles", "listing_stats_known": True, "lost": 0, "snapshot": snapshot}}}}


@pytest.mark.parametrize("count", [0, 1, 29])
def test_verified_empty_sparse_region_is_a_completed_search(count):
    assert audit.terminal_search(sparse_record(count))


@pytest.mark.parametrize("fault", ["unknown", "lost", "inventory", "signature", "error", "count", "source"])
def test_partial_or_stale_empty_region_proof_cannot_hide_a_failure(fault):
    record = sparse_record(1)
    proof = record["track_decoding"]["terminal_search"]
    if fault == "unknown": proof["listing"]["listing_stats_known"] = False
    elif fault == "lost": proof["listing"]["lost"] = 1
    elif fault == "inventory": proof["reference_inventory"][0]["lat"] += .01
    elif fault == "signature": proof["fetch_signature"] = {}
    elif fault == "error": record["error"] = "request interrupted"
    elif fault == "count": proof["count"] = 30
    else: proof["source"] = "unknown"
    with pytest.raises(ValueError): audit.terminal_search(record)


def metadata_pool(tmp_path, monkeypatch, missing=2, *, recorded=True):
    """One frozen regional control whose capped snapshot had URL-less members."""
    listed = [{"id": str(100 + i), "lat": 49. + i * 1e-4, "lon": 8.4} for i in range(40)]
    served = listed[missing:]
    fetch = {"listing": "tiles", "listing_stats_known": True, "lost": 0, "url_lookups": 40,
             "url_missing": missing, "snapshot": kv._reference_snapshot(listed)}
    if recorded:
        fetch["url_missing_refs"] = deepcopy(listed[:missing])
    np.savez(tmp_path / "0000.npz", query_indices=np.arange(80),
             ref_ids=np.array(["mapillary:" + r["id"] for r in served]),
             ref_latlon=np.array([[r["lat"], r["lon"]] for r in served]))
    record = {"status": "complete", "issues": [], "reference_pool": {"source": "mapillary", "count": len(served), "fetch": fetch}}
    monkeypatch.setattr(audit, "comparison", lambda *args, **kwargs: [record])
    files = SimpleNamespace(record=lambda row, ledger: Path(row["path"]))
    value = {"audit_artifacts": [{"path": str(tmp_path / "0000.json")}, {"path": str(tmp_path / "0000.npz")}]}
    return files, value, record


@pytest.mark.parametrize("missing,recorded", [(0, True), (2, True), (2, False), (5, True)])
def test_metadata_pool_accepts_recorded_bounded_url_less_shortfall(tmp_path, monkeypatch, missing, recorded):
    files, value, record = metadata_pool(tmp_path, monkeypatch, missing, recorded=recorded)
    regions = audit.metadata_regions(files, value, {}, [0., 300.])
    assert len(regions) == 1 and len(regions[0]["arrays"]["ref_ids"]) == 40 - missing


@pytest.mark.parametrize("fault", ["over_bound", "uncounted", "count_mismatch", "substituted_id", "served_id", "truncated"])
def test_metadata_pool_refuses_unaccounted_url_less_shortfall(tmp_path, monkeypatch, fault):
    files, value, record = metadata_pool(tmp_path, monkeypatch, 6 if fault == "over_bound" else 2)
    fetch = record["reference_pool"]["fetch"]
    if fault == "uncounted": fetch.pop("url_missing"); fetch.pop("url_missing_refs")
    elif fault == "count_mismatch": fetch["url_missing"] = 3
    elif fault == "substituted_id": fetch["url_missing_refs"][0]["id"] = "999"
    elif fault == "served_id": fetch["url_missing_refs"][0]["id"] = "139"
    elif fault == "truncated": fetch["truncated"] = True
    with pytest.raises(ValueError, match="incomplete capped served imagery"):
        audit.metadata_regions(files, value, {}, [0., 300.])


def test_sparse_listing_proof_counts_listed_members_so_url_less_ones_change_nothing():
    """Terminal proofs compare LISTED rows; URL-less members are listed, not a shortfall."""
    record = sparse_record(3)
    listing = record["track_decoding"]["terminal_search"]["listing"]
    refs = record["track_decoding"]["terminal_search"]["reference_inventory"]
    listing.update(url_lookups=3, url_missing=2, url_missing_refs=deepcopy(refs[:2]))
    assert audit.terminal_search(record)


def image_case(tmp_path):
    from tests.test_poi_image_track import _plan, _evidence, VIDEO
    from monocular_osm.poi_image_track import decode_single_branch_image_track
    plan = _plan()
    evidence = _evidence(plan)
    evidence["query_indices"] = np.arange(3)
    evidence["provenance"]["cpu_runtime"] = {"device": "cpu", "intra_op_threads": 4, "inter_op_threads": 1}
    child = decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)
    matrix = tmp_path / "poi_evidence/matrix.npz"
    matrix.parent.mkdir()
    np.savez(matrix, **{k: v for k, v in evidence.items() if k != "provenance"})
    native = tmp_path / "cues_vo.npz"
    np.savez(native, timestamps=np.arange(3.))
    child.update(image_plan=plan, candidate_artifact={"path": str(tmp_path / "candidate.json")},
        evidence_artifacts=[{"path": "poi_evidence/matrix.npz", "sha256": file_sha256(matrix)}],
        acquisition={"n_frames": 3, "stride": 6, "timestamps_sha256": worker._array_sha(np.arange(3.))})
    cue = {"clues": plan["clues"], "ocr_provenance": plan["ocr_provenance"],
           "auto_quality": {"stride": 6, "vo_artifact": {"path": str(native), "sha256": file_sha256(native)}}}
    ledger = {matrix: file_sha256(matrix), native: file_sha256(native)}
    return child, cue, ledger, {"video": {"sha256": VIDEO}}


def test_image_gate_resolves_relative_matrix_and_binds_native_grid_without_vo_poses(tmp_path):
    child, cue, ledger, inputs = image_case(tmp_path)
    audit.image_child(audit.Files(tmp_path), child, ledger, inputs, cue)


@pytest.mark.parametrize("fault", ["raw_score", "route", "grid", "profile", "vo", "provenance"])
def test_image_gate_rejects_stale_scores_route_and_profile(tmp_path, fault):
    child, cue, ledger, inputs = image_case(tmp_path)
    if fault == "raw_score": child["selected_raw_cosines"][0] += .01
    elif fault == "route": child["route_latlon"][0][0] += .01
    elif fault == "grid": child["acquisition"]["stride"] = 3
    elif fault == "profile": child["cap"] = 500
    elif fault == "vo": child["uses_rigid_vo"] = True
    else: child["provenance"]["similarities_sha256"] = "f" * 64
    with pytest.raises(ValueError): audit.image_child(audit.Files(tmp_path), child, ledger, inputs, cue)


def metric_case(tmp_path, monkeypatch):
    from tests.test_assisted_motion import _case, _prepare
    native = _case(tmp_path, monkeypatch, coverage=1., spread=0.)
    prepared = _prepare(native)
    ledger = {path: file_sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    inputs = {"video": {"sha256": file_sha256(native.video)}, "registered_window": [0., 130.],
        "source_sha256": {"monocular_osm/" + name: sha for name, sha in prepared["source_sha256"].items()}}
    return prepared, ledger, inputs, native.auto


def test_metric_gate_rederives_native_vo_and_plane_scalars_from_full_arrays(tmp_path, monkeypatch):
    prepared, ledger, inputs, auto = metric_case(tmp_path, monkeypatch)
    result = audit.metric_package(audit.Files(tmp_path), prepared, ledger, inputs, auto)
    assert result["eligible"] and result["plane"]["coverage"] == 1. and result["plane"]["rel_spread"] == 0.


@pytest.mark.parametrize("fault", ["quality", "window", "source", "profile", "summary", "array_hash"])
def test_metric_gate_rejects_stale_or_partial_metric_package(tmp_path, monkeypatch, fault):
    prepared, ledger, inputs, auto = metric_case(tmp_path, monkeypatch)
    if fault == "quality": auto["valid_fraction"] = .94
    elif fault == "window": inputs["registered_window"] = [0., 100.]
    elif fault == "source": inputs["source_sha256"]["monocular_osm/plane_speed.py"] = "f" * 64
    elif fault == "profile": prepared["profile_sha256"] = "f" * 64
    elif fault == "summary": prepared["summary"]["plane"]["coverage"] = .5
    else: Path(prepared["artifacts"]["native_metric_inputs.npz"]["path"]).write_bytes(b"changed native arrays")
    with pytest.raises(ValueError): audit.metric_package(audit.Files(tmp_path), prepared, ledger, inputs, auto)


def test_actual_weight_identity_cannot_change_between_clips(tmp_path, monkeypatch):
    a = fixture(tmp_path / "a", monkeypatch)
    b = fixture(tmp_path / "b", monkeypatch, model="9")
    summary, manifest = _valid(count=2)
    for i, case in enumerate((a, b)):
        summary["reps"][i]["provenance"] = case.summary["reps"][0]["provenance"]
        summary["reps"][i]["scores"]["answer"]["meta"] = {"mode": "assisted", "method": "assisted_policy"}
    result = M.check_goal(summary, manifest, expected_count=2)
    assert not result["goal_met"] and "actual retrieval weights" in reasons(result)


def cue_case(tmp_path, *, fault=None):
    from tests.test_check_poi_goal import _reader_proof
    from monocular_osm.visual_odometry import default_intrinsics
    from monocular_osm.poi_clues import build_poi_hypotheses
    times, valid = np.arange(5.), np.array([True, True, True, False, False])
    K = default_intrinsics(640, 480)
    if fault == "camera": K[0, 0] += 1
    path = tmp_path / "cues_vo.npz"
    np.savez(path, xz=np.c_[times, times], timestamps=times, valid=valid, K=K)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic video")
    signature = {"schema": 2, "vo_algo_version": 1, "video": {"path": str(video)}, "frames_sha256": "d" * 64,
        "odometry_runtime": {"workers": 2, "numpy": "synthetic", "opencv": "synthetic"},
        "extraction": {"start_sec": 0., "end_sec": 300., "stride": 6, "fps": 6., "n_frames": 5,
            "max_frames": None, "frame_max_width": None, "frame_memory_mb": 128.,
            "timestamps_sha256": hashlib.sha256(times.astype("<f8").tobytes()).hexdigest()}}
    if fault == "signature": signature["extraction"]["frame_max_width"] = 320
    elif fault == "workers": signature["odometry_runtime"]["workers"] = 4
    signed = tmp_path / "cues_vo_cache.npz"
    np.savez(signed, xz=np.c_[times, times], valid=valid, cache_signature=np.array(json.dumps(signature)), vo_algo_version=np.int64(1))
    quality = {"status": "complete", "n_frames": 5, "n_valid": 3, "valid_fraction": .6,
        "stride": 6, "fps": 6., "signed_vo_cache": {"path": str(signed), "sha256": file_sha256(signed)},
        "duration_s": 4., "timestamps_sha256": worker._array_sha(times), "video_sha256": "a" * 64,
        "window": [0., 300.], "vo_artifact": {"path": str(path), "sha256": file_sha256(path)}}
    if fault == "vo_counts": quality["valid_fraction"] = .9
    detections = []
    if fault == "position_stamp": detections = [{"text": "49.12345, 8.12345", "t_sec": 1., "confidence": .99,
                                                  "bbox": [0, 0, 100, 30]}]
    meta = {"source": "same_video_native_frames", "native_bbox_grid": True, "sample_count": 24,
        "min_len": 2, "min_confidence": .3, "super_res": False, "use_gpu": False,
        "position_stamps_rejected": True, "position_stamp_filter": "scene_text.is_position_stamp",
        "video_sha256": "a" * 64, "frame_shape_hw": [480, 640], "window": [0., 4.],
        "K_sha256": worker._array_sha(K), "detections_sha256": audit.digest(detections),
        "reader_sidecar_path": str(tmp_path / "reader.json"), "reader_sidecar_sha256": "c" * 64,
        "engine": "rapidocr", "model_weight_identity": "explicit_cached_sha256", "reader": _reader_proof()}
    if fault == "reader": meta["reader"]["actual_sessions"]["text_det"]["providers"] = ["CUDAExecutionProvider"]
    elif fault == "ocr_window": meta["window"] = [0., 3.]
    elif fault == "stamp_filter": meta["position_stamp_filter"] = "none"
    write(tmp_path / "poi_ocr.json", {"provenance": meta, "detections": detections})
    inventory = {"city": "Registered city", "scope": "named_city_bbox", "pois": [],
        "snapshot_sha256": audit.digest([]), "response_sha256": "b" * 64, "bbox_wsen": [8., 48., 9., 49.],
        "query": '[out:json];nwr["shop"](48.0000000,8.0000000,49.0000000,9.0000000);out center tags;'}
    if fault == "inventory_city": inventory["city"] = "Another city"
    elif fault == "inventory_query": inventory["query"] = '[out:json];nwr["shop"](0,0,1,1);out center tags;'
    write(tmp_path / "poi_inventory.json", inventory)
    inventory_meta = {k: inventory[k] for k in ("city", "scope", "snapshot_sha256", "response_sha256", "bbox_wsen", "query")}
    inventory_meta.update(path=str(tmp_path / "poi_inventory.json"), file_sha256=file_sha256(tmp_path / "poi_inventory.json"))
    clues = build_poi_hypotheses(detections, inventory)
    child = {"candidate_artifact": {"path": str(tmp_path / "candidate.json")}, "auto_quality": quality,
        "ocr_provenance": meta, "inventory": inventory_meta, "clues": clues,
        "cues": {"vo": quality, "duration_s": 4., "ocr": {"status": "complete", "clue_branch_count": 0,
            "detections_sha256": meta["detections_sha256"], "inventory_sha256": inventory["snapshot_sha256"]},
            "plane": {"status": "not_run", "measured": False}}}
    ledger = {p: file_sha256(p) for p in tmp_path.iterdir() if p.is_file()}
    return child, ledger, {"video": {"sha256": "a" * 64, "path": str(video)}, "registered_window": [0., 300.], "city": "Registered city"}


def test_native_cue_gate_preserves_true_zero_branches_and_measured_vo(tmp_path):
    child, ledger, inputs = cue_case(tmp_path)
    audit.cues(audit.Files(tmp_path), child, ledger, inputs, M._poi_reader_issues)


@pytest.mark.parametrize("fault", ["camera", "vo_counts", "position_stamp", "reader", "ocr_window", "stamp_filter",
                                   "inventory_city", "inventory_query", "signature", "workers"])
def test_cue_gate_rejects_rehashed_external_pixels_runtime_and_inventory(tmp_path, fault):
    child, ledger, inputs = cue_case(tmp_path, fault=fault)
    with pytest.raises(ValueError): audit.cues(audit.Files(tmp_path), child, ledger, inputs, M._poi_reader_issues)


@pytest.mark.parametrize("fault", ["object", "dimensions", "huge_header", "duplicate", "non_array"])
def test_artifact_loader_refuses_unsafe_npz_before_large_allocation(tmp_path, fault):
    import io
    import zipfile
    path = tmp_path / "evidence.npz"
    if fault in {"object", "dimensions"}:
        np.savez(path, value=np.array([object()]) if fault == "object" else np.zeros((1, 1, 1, 1)))
    else:
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False,
                                                      "shape": (2**30,) if fault == "huge_header" else (0,)})
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("value.npy" if fault != "non_array" else "unexpected.txt", header.getvalue())
            if fault == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate"):
                    archive.writestr("value.npy", header.getvalue())
    with pytest.raises(ValueError): audit.load_npz(path)
