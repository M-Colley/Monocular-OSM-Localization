"""Isolated worker transport and actual retrieval-audit adaptation; no models."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from monocular_osm import assisted_worker as worker
from monocular_osm import kartaview_vpr as kv
from monocular_osm.stages import assisted as controller


def parent(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic encoded input")
    return SimpleNamespace(cfg=SimpleNamespace(output_dir=tmp_path / "output"),
        video_path=video, assisted_child_root=tmp_path / "output/children/unique_execution",
        assisted_evidence={"baseline": {"status": "not_run"}},
        result={"assisted_inputs": {"video": {"sha256": worker.file_sha256(video)},
                                    "source_sha256": {"main.py": "a" * 64}}})


def fake_process(monkeypatch, tmp_path, *, returncode=0, response=True, interrupted=False):
    calls = []
    monkeypatch.setattr(worker, "_commit_headroom", lambda: 9 * 1024**3)
    monkeypatch.setattr(worker, "child_config", lambda *args: SimpleNamespace(output_dir=tmp_path / "unused"))
    payload = {"status": "complete", "route_latlon": [[49., 8.], [49.001, 8.001]]}

    class Process:
        pid = 1234567
        def wait(self):
            if interrupted:
                raise KeyboardInterrupt("cancelled by caller")
            return returncode

    process = Process()
    def popen(command, **kwargs):
        request = Path(command[-1])
        if response:
            request.with_name("response.json").write_text(json.dumps(payload), encoding="utf8")
        kwargs["stdout"].write("synthetic worker log\n")
        calls.append((command, kwargs))
        return process

    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    return process, calls, payload


@pytest.mark.parametrize("returncode", [1, -9])
def test_nonzero_exit_never_promotes_a_complete_frozen_response(monkeypatch, tmp_path, returncode):
    _, calls, payload = fake_process(monkeypatch, tmp_path, returncode=returncode)
    ctx = parent(tmp_path)
    result = worker.run_child("baseline", ctx)
    assert result["status"] == "failed" and result["reported_status"] == "complete"
    assert result["worker_returncode"] == returncode and "frozen diagnostics retained" in result["transport_error"]
    assert json.loads((ctx.assisted_child_root / "baseline/response.json").read_text()) == payload
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert calls[0][1]["env"]["OMP_NUM_THREADS"] == "4"


def test_success_records_returncode_and_frozen_request_identity(monkeypatch, tmp_path):
    _, _, payload = fake_process(monkeypatch, tmp_path)
    ctx = parent(tmp_path)
    result = worker.run_child("baseline", ctx)
    assert result == {**payload, "worker_returncode": 0}
    request = json.loads((ctx.assisted_child_root / "baseline/request.json").read_text())
    assert request["video_sha256"] == ctx.result["assisted_inputs"]["video"]["sha256"]
    assert request["source_sha256"] == ctx.result["assisted_inputs"]["source_sha256"]
    assert request["policy_evidence"] == ctx.assisted_evidence
    assert request["launch_commit_headroom_bytes"] == 9 * 1024**3


def test_headroom_refuses_before_any_spawn_or_child_output(monkeypatch, tmp_path):
    ctx = parent(tmp_path)
    monkeypatch.setattr(worker, "_commit_headroom", lambda: 8 * 1024**3 - 1)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn"))
    with pytest.raises(RuntimeError, match="less than 8 GiB"):
        worker.run_child("baseline", ctx)
    assert not ctx.assisted_child_root.exists()


def test_cancellation_stops_only_the_owned_child_and_propagates(monkeypatch, tmp_path):
    process, _, _ = fake_process(monkeypatch, tmp_path, interrupted=True)
    stopped = []
    monkeypatch.setattr(worker, "_stop_owned_child", stopped.append)
    with pytest.raises(KeyboardInterrupt, match="cancelled by caller"):
        worker.run_child("baseline", parent(tmp_path))
    assert stopped == [process]


def test_missing_response_exposes_worker_failure(monkeypatch, tmp_path):
    fake_process(monkeypatch, tmp_path, returncode=7, response=False)
    with pytest.raises(RuntimeError, match="exited 7 without a frozen response"):
        worker.run_child("baseline", parent(tmp_path))


def test_stale_execution_directory_is_never_reused(monkeypatch, tmp_path):
    _, calls, _ = fake_process(monkeypatch, tmp_path)
    ctx = parent(tmp_path)
    (ctx.assisted_child_root / "baseline").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        worker.run_child("baseline", ctx)
    assert not calls


@pytest.mark.parametrize("system", ["nt", "posix"])
def test_cleanup_targets_only_the_owned_process_tree(monkeypatch, system):
    calls = []
    child = SimpleNamespace(pid=1234567, poll=lambda: None, wait=lambda: calls.append("wait"))
    if system == "nt":
        monkeypatch.setattr(worker, "os", SimpleNamespace(name="nt"))
        monkeypatch.setattr(worker.subprocess, "run", lambda command, **kwargs: calls.append(command))
        expected = ["taskkill", "/PID", "1234567", "/T", "/F"]
    else:
        import signal
        monkeypatch.setattr(worker, "os", SimpleNamespace(name="posix", killpg=lambda *args: calls.append(args)))
        expected = (1234567, signal.SIGTERM)
    worker._stop_owned_child(child)
    assert calls == [expected, "wait"]


def audit(monkeypatch, *, offset=0.):
    indices = np.arange(0, 10, 2, dtype=np.int64)
    timeline = np.arange(9, dtype=np.float64) * .55
    references = np.column_stack([49. + offset + np.arange(40) * .00001, np.full(40, 8.4)])
    references[-1, 0] += .1
    scores = np.zeros((5, 40), dtype=np.float32)
    scores[:, 0] = [.5, .16196, .16196, .16196, .5]
    scores[2, -1] = .95
    chosen = np.zeros(5, dtype=np.int64)
    cosines = scores[np.arange(5), chosen].copy()
    value = {"issues": [], "query_indices": indices.copy(), "query_pixels_sha256": "1" * 64,
        "query_descriptors_sha256": "2" * 64, "actual_model_state_sha256": "3" * 64,
        "backbone": {"name": "megaloc", "hub": kv.HUB_MEGALOC, "device": "cpu"},
        "preprocessing": {"version": 1, "warp_size": [512, 512], "descriptor_normalization": "L2"},
        "query_mode": "warp", "ref_ids": np.array([f"mapillary:{offset}:{i}" for i in range(40)]),
        "ref_latlon": references.copy(), "ref_embeddings_sha256": "4" * 64,
        "raw_similarities": scores.copy()}
    monkeypatch.setattr(kv, "LAST_TRACK_STATS", {"decoder": "viterbi"})
    kv._finish_track_audit(value, indices, scores, chosen, cosines, timeline, True, 1.1)
    assert value["status"] == "complete", value.get("issues")
    value["returned_track"] = {"indices": indices.copy(), "latlon": references[chosen].copy(),
                               "cosines": cosines.copy()}
    return value


def context(audits, *, mixed=False):
    first = audits[0]["returned_track"]
    coordinates = first["latlon"].copy()
    if mixed:
        coordinates[1] = audits[1]["returned_track"]["latlon"][1]
    return SimpleNamespace(retrieval_audits=audits,
        vpr_track=(first["indices"].copy(), coordinates, first["cosines"].copy()),
        frames=SimpleNamespace(sampling={"mode": "synthetic_native", "stride": 2}))


def test_actual_adapter_uses_unrounded_decoded_cosines_and_saves_full_matrix(monkeypatch, tmp_path):
    original = audit(monkeypatch)
    value = worker._comparison(context([original]), tmp_path)
    assert value["decoded_reference_cosines"] == original["decoded_cosines"].tolist()
    assert value["decoded_reference_cosines"][2] < float(original["raw_similarities"][2].max())
    scalar = controller.summary("baseline", {"status": "complete", "route_latlon": [[49., 8.], [49.1, 8.]],
                                               "comparison": value})
    assert scalar["similarity_median"] == float(np.float32(.16196)) < .162
    assert scalar["score_identity"]
    assert value["query"]["indices"] == [0, 2, 4, 6, 8]
    assert value["inner_decoder"]["actual"] == "viterbi" and value["inner_decoder"]["fallback"] is False
    for artifact in value["audit_artifacts"]:
        assert worker.file_sha256(artifact["path"]) == artifact["sha256"]
        if Path(artifact["path"]).suffix == ".npz":
            with np.load(artifact["path"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["raw_similarities"], original["raw_similarities"])


def test_mixed_regional_rows_keep_actual_reference_ids_and_pool_provenance(monkeypatch, tmp_path):
    a, b = audit(monkeypatch), audit(monkeypatch, offset=.001)
    value = worker._comparison(context([a, b], mixed=True), tmp_path)
    assert value["selected_audit_rows"] == [(0, 0), (1, 1), (0, 2), (0, 3), (0, 4)]
    assert value["decoded_reference_ids"][1] == str(b["ref_ids"][0])
    assert len(value["audit_artifacts"]) == 4


def test_unmatched_selected_track_cannot_claim_comparison(monkeypatch, tmp_path):
    ctx = context([audit(monkeypatch)])
    ctx.vpr_track[1][0, 0] += .01
    assert "unavailable_reason" in worker._comparison(ctx, tmp_path)


@pytest.mark.parametrize("fault", ["identity", "matrix", "coordinates", "decoded_index"])
def test_stale_or_inconsistent_actual_audit_is_declined(monkeypatch, tmp_path, fault):
    a, b = audit(monkeypatch), audit(monkeypatch, offset=.001)
    if fault == "identity":
        b["comparison_identity"]["query_pixels_sha256"] = "f" * 64
    elif fault == "matrix":
        b["raw_similarities"][1, 0] += .1
    elif fault == "coordinates":
        b["ref_latlon"][0, 0] += .01
    else:
        b["decoded_ref_indices"][1] = 1
    result = worker._comparison(context([a, b], mixed=True), tmp_path)
    assert "unavailable_reason" in result
