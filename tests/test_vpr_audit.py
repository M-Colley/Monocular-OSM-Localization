"""Opt-in evidence identities preserve ordinary retrieval numerics."""
import hashlib

import numpy as np
import pytest

from monocular_osm import kartaview_vpr as kv


def setup_track(monkeypatch):
    torch = pytest.importorskip("torch")
    frames = [np.full((3, 6, 3), i, np.uint8) for i in range(3)]
    refs = [{"id": str(i), "lat": 50 + i * .00001, "lon": 8.} for i in range(40)]
    refs[1]["lat"] = 50.01
    monkeypatch.setattr(kv, "_MODEL", torch.nn.Linear(2, 2))
    monkeypatch.setattr(kv, "_MODEL_NAME", "megaloc")
    monkeypatch.setattr(kv, "_MODEL_DEVICE", "cpu")
    monkeypatch.setattr(kv, "_resolve_backbone", lambda *a: "megaloc")
    monkeypatch.setattr(kv, "_fetch_refs_for", lambda *a: refs)
    monkeypatch.setattr(kv, "_embed_refs", lambda *a, **k: (
        torch.eye(40), np.array([[r["lat"], r["lon"]] for r in refs])))
    descriptors = torch.zeros((3, 40))
    descriptors[0, 0] = descriptors[2, 0] = descriptors[1, 1] = 1.
    monkeypatch.setattr(kv, "_embed", lambda *a, **k: descriptors.clone())
    return frames, descriptors


def test_complete_audit_binds_actual_encoder_pixels_times_and_decoded_cosines(monkeypatch):
    frames, descriptors = setup_track(monkeypatch)
    ordinary = kv.kartaview_vpr_track(frames, (50, 8), source="mapillary", device="cpu", query_dt_s=1.5)
    audit = {}
    result = kv.kartaview_vpr_track(frames, (50, 8), source="mapillary", device="cpu", query_dt_s=1.5,
                                    audit=audit, query_timestamps=[0., .7, 3.])
    assert audit["status"] == "complete", audit.get("issues")
    for expected, actual in zip(ordinary, result):
        np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(audit["raw_similarities"], descriptors.numpy())
    np.testing.assert_array_equal(audit["decoded_cosines"], [1., 0., 1.])
    assert audit["decoded_cosines"][1] < audit["raw_similarities"][1].max()
    np.testing.assert_array_equal(audit["query_times"], [0., .7, 3.])
    assert audit["ref_ids"].tolist() == [f"mapillary:{i}" for i in range(40)]
    assert audit["actual_model_state_sha256"] == kv._audit_model_state()["sha256"]
    assert audit["decoder"]["actual"] == "viterbi"
    assert audit["decoder"]["query_dt_s"] == 1.5
    assert audit["score_kind"] == "raw_cosine_at_decoded_reference"
    assert len(audit["comparison_identity_sha256"]) == 64


def test_default_track_never_runs_expensive_audit(monkeypatch):
    frames, _ = setup_track(monkeypatch)
    monkeypatch.setattr(kv, "_audit_model_state", lambda: pytest.fail("audit must stay opt-in"))
    monkeypatch.setattr(kv, "_audit_pixels", lambda *a: pytest.fail("audit must stay opt-in"))
    assert kv.kartaview_vpr_track(frames, (50, 8), device="cpu") is not None


@pytest.mark.parametrize("timeline", [None, [0, 1], [0, 0, 1], [2, 1, 0], [0, float("nan"), 2]])
def test_bad_audit_grid_is_incomparable_without_losing_original_route(monkeypatch, timeline):
    frames, _ = setup_track(monkeypatch)
    ordinary = kv.kartaview_vpr_track(frames, (50, 8), device="cpu")
    audit = {}
    result = kv.kartaview_vpr_track(frames, (50, 8), device="cpu", audit=audit, query_timestamps=timeline)
    assert audit["status"] == "incomplete" and audit["issues"]
    for expected, actual in zip(ordinary, result):
        np.testing.assert_array_equal(expected, actual)


def test_changed_pixels_model_and_times_change_comparison_identity(monkeypatch):
    torch = pytest.importorskip("torch")
    frames, _ = setup_track(monkeypatch)
    def run(times=(0., 1., 2.)):
        audit = {}
        kv.kartaview_vpr_track(frames, (50, 8), device="cpu", audit=audit, query_timestamps=times)
        assert audit["status"] == "complete", audit["issues"]
        return audit
    initial = run()
    frames[0][0, 0, 0] += 1
    pixel_change = run()
    assert initial["query_pixels_sha256"] != pixel_change["query_pixels_sha256"]
    with torch.no_grad():
        kv._MODEL.weight[0, 0] += 1
    model_change = run()
    assert pixel_change["actual_model_state_sha256"] != model_change["actual_model_state_sha256"]
    time_change = run((0., 1.1, 2.))
    assert len({r["comparison_identity_sha256"] for r in (initial, pixel_change, model_change, time_change)}) == 4


def test_reference_pool_does_not_change_query_comparison_identity(monkeypatch):
    torch = pytest.importorskip("torch")
    frames, _ = setup_track(monkeypatch)
    def run():
        audit = {}
        kv.kartaview_vpr_track(frames, (50, 8), device="cpu", audit=audit, query_timestamps=[0., 1., 2.])
        assert audit["status"] == "complete", audit["issues"]
        return audit
    first = run()
    original = kv._embed_refs
    def changed_refs(*args, **kwargs):
        emb, coords = original(*args, **kwargs)
        emb = torch.roll(emb, shifts=1, dims=0)
        return emb, coords
    monkeypatch.setattr(kv, "_embed_refs", changed_refs)
    second = run()
    assert first["ref_embeddings_sha256"] != second["ref_embeddings_sha256"]
    assert first["comparison_identity_sha256"] == second["comparison_identity_sha256"]


def test_audit_failure_is_explicit_and_preserves_numeric_answer(monkeypatch):
    frames, _ = setup_track(monkeypatch)
    ordinary = kv.kartaview_vpr_track(frames, (50, 8), device="cpu")
    def unsupported():
        raise ValueError("unsupported state layout")
    monkeypatch.setattr(kv, "_audit_model_state", unsupported)
    audit = {}
    result = kv.kartaview_vpr_track(frames, (50, 8), device="cpu", audit=audit, query_timestamps=[0, 1, 2])
    assert audit["status"] == "incomplete"
    assert any("unsupported state layout" in issue for issue in audit["issues"])
    for expected, actual in zip(ordinary, result):
        np.testing.assert_array_equal(expected, actual)


def test_argmax_memory_fallback_is_recorded_truthfully_before_comparison(monkeypatch):
    pytest.importorskip("torch")  # kartaview_vpr_track returns None without torch
    scores = np.zeros((3, 6001), np.float32)
    scores[:, 2] = 1
    monkeypatch.setattr(kv, "_prepare_refs_and_query", lambda *a, **k: (
        np.arange(3), scores, np.tile([50., 8.], (6001, 1))))
    audit = {}
    route = kv.kartaview_vpr_track([object()] * 3, (50, 8), device="cpu",
                                   audit=audit, query_timestamps=[0., 1., 2.])
    assert route is not None
    assert audit["decoder"]["actual"] == "argmax"
    assert "6000" in audit["decoder"]["fallback_reason"]
    np.testing.assert_array_equal(audit["decoded_ref_indices"], [2, 2, 2])


def test_streamed_state_hash_includes_scalar_buffers_and_uses_bounded_transfers(monkeypatch):
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.register_buffer("counter", torch.tensor(3))
    model.register_buffer("weights", torch.arange(3_000_000, dtype=torch.float32))
    monkeypatch.setattr(kv, "_MODEL", model)
    transfers = []
    original = torch.Tensor.cpu
    def observe(tensor, *args, **kwargs):
        transfers.append(tensor.numel() * tensor.element_size())
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", observe)
    first = kv._audit_model_state()
    assert max(transfers) <= 8 * 1024 ** 2 and len(transfers) >= 3
    model.counter += 1
    assert kv._audit_model_state()["sha256"] != first["sha256"]
    assert first["element_count"] == 3_000_001


def test_query_audit_on_cache_hit_hashes_the_actual_reused_descriptors(monkeypatch):
    torch = pytest.importorskip("torch")
    frames = [np.zeros((2, 4, 3), np.uint8) for _ in range(2)]
    monkeypatch.setattr(kv, "_MODEL", torch.nn.Linear(2, 2))
    calls = []
    def embed(*args, **kwargs):
        calls.append(1)
        return torch.eye(2)
    monkeypatch.setattr(kv, "_embed", embed)
    cache, first, second = {}, {}, {}
    a = kv._query_similarities(frames, [0, 1], torch.eye(2), "cpu", batch_size=2,
                               query_cache=cache, audit=first)
    b = kv._query_similarities(frames, [0, 1], torch.eye(2), "cpu", batch_size=2,
                               query_cache=cache, audit=second)
    assert calls == [1]
    np.testing.assert_array_equal(a, b)
    assert first == second
