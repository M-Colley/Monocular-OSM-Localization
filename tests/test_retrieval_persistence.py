"""Real assisted/prior wrappers with tiny fake retrievals; no models or inputs."""
from copy import deepcopy
from functools import wraps
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from monocular_osm import assisted_worker as worker, kartaview_vpr as kv, pipeline
from monocular_osm import public_reference_cache as cache, retrieval_catalog as catalog
from monocular_osm.runtime_budget import DeadlineExpired, OwnedProcessTree, RuntimeBudget
from monocular_osm.stages import assisted, prior
from monocular_osm.stages.context import RunContext


def payload():
    return {"status": "complete", "ref_ids": np.array(["kartaview:1", "kartaview:2"]),
        "ref_latlon": np.array([[1., 2.], [1., 2.001]]),
        "raw_similarities": np.array([[.1, .2], [.3, .4]], np.float32),
        "query_indices": np.array([0, 1]), "query_times": np.array([0., 1.]),
        "ref_embeddings_sha256": "a" * 64, "query_pixels_sha256": "b" * 64}


def config(root):
    root.mkdir(parents=True, exist_ok=True)
    return pipeline.PipelineConfig(url="", city="Synthetic", data_dir=root / "data",
        output_dir=root, video_path=root / "video.mp4", vpr_source="kartaview",
        use_vpr_prior=True, vpr_multi_region=True, vpr_coarse_to_fine=False,
        vpr_city_cover=False, vo_end_sec=1., assisted_profile="v1")


def install_pipeline(monkeypatch, cfg, *, block_second=False, clear_audit=False):
    contexts, seen = [], []
    monkeypatch.setitem(sys.modules, "osmnx", SimpleNamespace())
    monkeypatch.setattr(prior, "select_retrieval_device", lambda: "cpu")
    monkeypatch.setattr(kv, "_robust_center", lambda ll, cosines: tuple(ll.mean(axis=0)))
    original = kv.kartaview_vpr_track

    @wraps(original)
    def retrieve(*args, **kwargs):
        if seen and block_second:
            # The first call must already be a real pair before this call blocks.
            saved = json.loads((cfg.output_dir / "retrieval_audit/0000.json").read_text())
            assert saved["status"] == "complete" and saved["persisted_writer"]["ordinal"] == 0
            assert saved["public_cache_request"]["source"] == "kartaview"
            (cfg.output_dir / "second_call_blocked").write_text("ready")
            time.sleep(60)
        audit = kwargs["audit"]
        if clear_audit:
            audit.clear()  # Same acquisition-index loss as the KV cache-hit path.
        audit.update(payload())
        seen.append(audit)
        return (audit["query_indices"].copy(), audit["ref_latlon"].copy(),
                np.array([.1, .4]))

    monkeypatch.setattr(kv, "kartaview_vpr_track", retrieve)

    def regions(cfg, frames, center, radius, cap, source, token, device, dt, track, **kwargs):
        args = dict(source=source, radius_m=radius, cap=cap, device=device,
            cache_dir=str(cfg.data_dir / source), query_cache=kwargs["query_cache"])
        first = track(frames.frames, center, **args)
        second = track(frames.frames, center, **args)
        return second or first, {"n_usable_regions": 2}, {}

    monkeypatch.setattr(prior, "_multi_region_vpr", regions)

    def stages(ctx):
        contexts.append(ctx)
        ctx.frames = SimpleNamespace(frames=[None, None], timestamps=np.array([0., 1.]), close=lambda: None)
        ctx.osm_around = (1., 2., 100.)
        ctx.estimated_length_m = 100.
        prior.vpr_coarse_prior(ctx)
        return ctx.result

    monkeypatch.setattr(pipeline, "_run_pipeline_stages", stages)
    return contexts, seen


def test_real_wrapper_checkpoint_survives_owned_child_timeout_and_parent_freeze(tmp_path, monkeypatch):
    cfg = config(tmp_path / "out")
    cfg.video_path.write_bytes(b"fake video")
    parent = RunContext(cfg=cfg, fusion_w=None, video_path=cfg.video_path)
    parent.assisted_child_root = cfg.output_dir / "children/execution"
    parent.runtime_budget = RuntimeBudget(time.monotonic() + 60., 2., 1., "test", time.monotonic(), time.monotonic)
    original_answer = {"route_latlon": [[1., 2.], [1., 2.001]]}
    parent.result = {"answer": deepcopy(original_answer), "assisted_inputs": {
        "execution_id": "execution", "video": {"sha256": worker.file_sha256(cfg.video_path)},
        "source_sha256": {"fixture.py": "c" * 64}, "registered_window": [0., 1.]}}
    monkeypatch.setattr(worker, "_commit_headroom", lambda: 9 * 1024**3)
    popen = subprocess.Popen

    def launch(argv, **kwargs):
        return popen([sys.executable, "-B", str(Path(__file__).resolve()),
            "--blocked-child", argv[-1]], **kwargs)

    monkeypatch.setattr(worker.subprocess, "Popen", launch)

    class Owner(OwnedProcessTree):
        def wait(self, deadline):
            marker = parent.assisted_child_root / "baseline/second_call_blocked"
            until = time.monotonic() + 20.
            while not marker.exists():
                self.capture()
                if self.child.poll() is not None or time.monotonic() >= until:
                    raise AssertionError("fake child did not reach its blocked second retrieval")
                time.sleep(.02)
            raise subprocess.TimeoutExpired("fake second retrieval", 0)

    # Exercise the actual run_child timeout/cleanup/catalog path, not a manually
    # persisted pair in a fake Popen callback.
    import monocular_osm.runtime_budget as budgets
    monkeypatch.setattr(budgets, "OwnedProcessTree", Owner)
    with pytest.raises(DeadlineExpired):
        worker.run_child("baseline", parent)
    record = parent.assisted_retrieval_catalogs["baseline"]
    value = catalog.read(record["artifact"]["path"])
    assert value["terminal_authoritative"] and len(value["pairs"]) == 1
    assert value["pairs"][0]["status"] == "complete_pair"
    root = parent.assisted_child_root / "baseline"
    assert not (root / "response.json").exists() and not (root / "child_frozen.json").exists()
    parent.result["assisted_audit"] = {"reached_stages": [],
        "persisted_retrieval_catalogs": parent.assisted_retrieval_catalogs}
    worker._write_result_json(cfg.output_dir / "result.json", parent.result)
    assisted.freeze(parent)
    frozen = json.loads((cfg.output_dir / "frozen_prediction.json").read_text())
    assert any(name.replace("\\", "/").endswith("baseline/retrieval_audit/0000.npz")
               for name in frozen["artifacts"])
    assert parent.result["answer"] == original_answer


def test_cache_cleared_audit_keeps_request_reuse_ordinals_and_idempotent_final_export(tmp_path, monkeypatch):
    cfg = config(tmp_path / "child")
    contexts, seen = install_pipeline(monkeypatch, cfg, clear_audit=True)
    checkpoint_bytes = []

    def seal(audits, cfg):
        for index, audit in enumerate(audits):
            path = cfg.output_dir / f"retrieval_audit/{index:04d}.npz"
            checkpoint_bytes.append(path.read_bytes())
            audit["public_reference_cache"] = {"status": "declined", "reason": "tiny fake encoder"}

    monkeypatch.setattr(cache, "seal_pool_audits", seal)
    worker._pipeline_child(cfg, "baseline")
    assert len(seen) == 2 and len(checkpoint_bytes) == 2
    for index, audit in enumerate(seen):
        path = cfg.output_dir / f"retrieval_audit/{index:04d}.npz"
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert "call_index" not in metadata
        assert metadata["persisted_writer"]["ordinal"] == index
        assert metadata["public_cache_request"] == audit["public_cache_request"]
        assert metadata["public_cache_reuse"] == audit["public_cache_reuse"]
        assert metadata["public_reference_cache"]["status"] == "declined"
        assert path.read_bytes() == checkpoint_bytes[index]
        with np.load(path, allow_pickle=False) as arrays:
            assert np.array_equal(arrays["returned_latlon"], audit["returned_track"]["latlon"])
    before = {p.name: p.read_bytes() for p in (cfg.output_dir / "retrieval_audit").iterdir()}
    worker._save_audits(contexts[0].retrieval_audits, cfg.output_dir)
    assert before == {p.name: p.read_bytes() for p in (cfg.output_dir / "retrieval_audit").iterdir()}


@pytest.mark.parametrize("phase", ["arrays", "metadata"])
def test_interrupted_first_checkpoint_never_advertises_a_complete_pair(tmp_path, monkeypatch, phase):
    root = tmp_path / "child"
    root.mkdir()
    request = {"kind": "baseline", "execution_id": "execution", "video_sha256": "a" * 64,
        "source_sha256": {}, "config": {"vo_start_sec": 0., "vo_end_sec": 1.}}
    (root / "request.json").write_text(json.dumps(request))

    def fail(*args, **kwargs):
        raise OSError("synthetic interrupted write")

    monkeypatch.setattr(worker.np if phase == "arrays" else worker,
        "savez_compressed" if phase == "arrays" else "_write_result_json", fail)
    with pytest.raises(OSError):
        worker._save_audit(payload(), root, 0)
    rows = catalog.collect(root, request, {"all_tracked_owned_births_gone": True})["pairs"]
    assert not rows if phase == "arrays" else rows[0]["status"] == "orphan"
    assert not list((root / "retrieval_audit").glob("*.tmp"))


def test_interrupted_enrichment_preserves_previous_complete_pair_and_rejects_array_drift(tmp_path, monkeypatch):
    audit = payload()
    worker._save_audit(audit, tmp_path, 0)
    before = {p.name: p.read_bytes() for p in (tmp_path / "retrieval_audit").iterdir()}
    audit["public_reference_cache"] = {"status": "declined"}

    def fail(*args, **kwargs):
        raise OSError("synthetic final metadata interruption")

    monkeypatch.setattr(worker, "_write_result_json", fail)
    # Recompression would risk replacing the earlier pair before JSON commits.
    monkeypatch.setattr(worker.np, "savez_compressed", lambda *a, **k: pytest.fail("recompressed checkpoint"))
    with pytest.raises(OSError):
        worker._save_audits([audit], tmp_path)
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "retrieval_audit").iterdir()}
    audit["raw_similarities"][0, 0] += .1
    with pytest.raises(ValueError, match="arrays changed"):
        worker._save_audits([audit], tmp_path)
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "retrieval_audit").iterdir()}


def test_optional_sink_failure_does_not_change_completed_retrieval(tmp_path, monkeypatch):
    cfg = config(tmp_path / "child")
    contexts, seen = install_pipeline(monkeypatch, cfg)

    def fail(*args):
        raise OSError("synthetic disk full")

    monkeypatch.setattr(worker, "_save_audit", fail)
    worker._pipeline_child(cfg, "baseline")
    ctx = contexts[0]
    assert len(seen) == 2 and all(a["status"] == "complete" for a in seen)
    assert all("provider_error" not in a for a in seen)
    assert np.array_equal(ctx.vpr_track[1], seen[-1]["returned_track"]["latlon"])
    assert [e["ordinal"] for e in ctx.result["retrieval_audit_persistence_errors"]] == [0, 1]


def test_context_without_sink_keeps_in_memory_audit_and_interrupts_are_not_swallowed():
    ctx = SimpleNamespace(retrieval_audits=[], result={})
    audit = payload()
    prior._append_retrieval_audit(ctx, audit)
    assert ctx.retrieval_audits == [audit] and ctx.result == {}

    def interrupt(*args):
        raise KeyboardInterrupt

    ctx.retrieval_audit_sink = interrupt
    with pytest.raises(KeyboardInterrupt):
        prior._append_retrieval_audit(ctx, payload())
    assert len(ctx.retrieval_audits) == 2 and ctx.result == {}


if __name__ == "__main__":
    assert sys.argv[1] == "--blocked-child"
    request = json.loads(Path(sys.argv[2]).read_text())
    cfg = worker._from_snapshot(request["config"])
    with pytest.MonkeyPatch.context() as patch:
        install_pipeline(patch, cfg, block_second=True)
        worker._pipeline_child(cfg, request["kind"], request)
