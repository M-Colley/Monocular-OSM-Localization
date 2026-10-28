"""Audited evidence captures existing numerical inputs without extra forwards."""
import numpy as np
import pytest

from monocular_osm import kartaview_vpr as kv


@pytest.fixture
def inputs(monkeypatch):
    torch = pytest.importorskip("torch")
    frames = [np.full((3, 6, 3), i, np.uint8) for i in range(3)]
    refs = [{"id": str(i+1), "lat": 50+i*.0001, "lon": 8.} for i in range(40)]
    monkeypatch.setattr(kv, "_MODEL", torch.nn.Linear(2, 2))
    monkeypatch.setattr(kv, "_MODEL_NAME", "megaloc")
    monkeypatch.setattr(kv, "_MODEL_DEVICE", "cpu")
    monkeypatch.setattr(kv, "_resolve_backbone", lambda *args: "megaloc")
    def fetch(*args):
        kv.LAST_FETCH_STATS = {"listing": "tiles", "tiles": 1, "lost": 0,
            "listing_stats_known": True, "snapshot": kv._reference_snapshot(refs)}
        return refs
    monkeypatch.setattr(kv, "_fetch_refs_for", fetch)
    reference = torch.eye(40)
    monkeypatch.setattr(kv, "_embed_refs", lambda *args, **kw: (
        reference, np.array([[r["lat"], r["lon"]] for r in refs])))
    calls = []
    def embed(device, images, **kwargs):
        calls.append(len(images))
        return torch.eye(40)[:len(images)].clone()
    monkeypatch.setattr(kv, "_embed", embed)
    return frames, reference, calls


def call(frames, **kwargs):
    return kv.kartaview_vpr_evidence(frames, [4., 5.1, 6.4], (50., 8.),
        n_query=3, cap=1500, device="cpu", query_mode="crops", **kwargs)


def test_default_evidence_numerics_and_forward_count_unchanged_by_optin_audit(inputs):
    frames, refs, calls = inputs
    ordinary = call(frames)
    audit = {}
    captured = call(frames, audit=audit, query_video_sha256="a"*64)
    assert ordinary is not None and captured is not None, kv.LAST_FAILURE
    assert calls == [9, 9]  # three native crops, with no audit/model re-encoding
    for key in ("query_indices", "query_times", "similarities", "ref_ids", "ref_latlon"):
        np.testing.assert_array_equal(ordinary[key], captured[key])
    assert "query_identity" not in ordinary["provenance"]
    assert captured["status"] == audit["status"] == "complete"
    proof = captured["provenance"]["query_identity"]
    assert proof["query_video_sha256"] == "a"*64
    assert proof["actual_model_state_sha256"] == kv._audit_model_state()["sha256"]
    assert proof["query_indices_sha256"] == kv._audit_array_sha(captured["query_indices"])
    assert proof["query_times_sha256"] == kv._audit_array_sha(captured["query_times"])
    assert proof["views"] == 3 and proof["query_mode"] == "crops"
    rows = captured["provenance"]["reference_identities"]
    assert set(rows) == set(captured["ref_ids"])
    assert rows["mapillary:1"]["embedding_sha256"] == kv._audit_tensor_sha(refs[:1])
    assert rows["mapillary:1"]["pixel_sha256"] is None
    assert "unverified" in rows["mapillary:1"]["encoding_identity"]
    acquisition = captured["provenance"]["acquisition"]
    assert acquisition["status"] == "complete" and acquisition["listing_complete"] is True
    assert acquisition["listed_ref_ids"] == captured["ref_ids"].tolist()
    assert acquisition["listed_references"] == [
        {"id": str(i+1), "lat": 50+i*.0001, "lon": 8.} for i in range(40)]
    assert acquisition["unserved_references"] == []


def test_same_descriptor_cache_binds_actual_reused_query_without_another_forward(inputs):
    frames, _, calls = inputs
    cache = {}
    a, b = {}, {}
    first = call(frames, query_cache=cache, audit=a, query_video_sha256="a"*64)
    second = call(frames, query_cache=cache, audit=b, query_video_sha256="a"*64)
    assert first is not None and second is not None, kv.LAST_FAILURE
    assert calls == [9]
    assert first["provenance"]["query_identity"] == second["provenance"]["query_identity"]
    assert a["query_descriptors_sha256"] == b["query_descriptors_sha256"]


@pytest.mark.parametrize("sha", [None, "x", "z"*64, "A"*64])
def test_invalid_video_identity_fails_before_fetch(inputs, monkeypatch, sha):
    frames, _, calls = inputs
    monkeypatch.setattr(kv, "_prepare_refs_and_query", lambda *a, **k: pytest.fail("must not fetch"))
    audit = {}
    assert call(frames, audit=audit, query_video_sha256=sha) is None
    assert audit["status"] == "incomplete" and not calls


def test_unavailable_model_identity_cannot_be_reported_as_complete(inputs, monkeypatch):
    frames, _, _ = inputs
    monkeypatch.setattr(kv, "_audit_model_state", lambda: {"sha256": "unverified"})
    audit = {}
    assert call(frames, audit=audit, query_video_sha256="a"*64) is None
    assert audit["status"] == "incomplete"


def test_bad_reference_row_audit_declines_without_promoting_partial_identity(inputs, monkeypatch):
    frames, _, _ = inputs
    original = kv._prepare_refs_and_query
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        kwargs["audit"]["reference_row_embeddings_sha256"].pop()
        return result
    monkeypatch.setattr(kv, "_prepare_refs_and_query", changed)
    audit = {}
    assert call(frames, audit=audit, query_video_sha256="a"*64) is None
    assert audit["status"] == "incomplete"


def test_default_evidence_does_not_hash_model_or_pixels(inputs, monkeypatch):
    frames, _, _ = inputs
    monkeypatch.setattr(kv, "_audit_model_state", lambda: pytest.fail("opt-in only"))
    monkeypatch.setattr(kv, "_audit_pixels", lambda *a: pytest.fail("opt-in only"))
    assert call(frames) is not None


def test_listing_unknown_does_not_become_complete_from_valid_query_identity(inputs, monkeypatch):
    frames, _, _ = inputs
    original = kv._fetch_refs_for
    def partial(*args):
        refs = original(*args)
        kv.LAST_FETCH_STATS["lost"] = 1
        return refs
    monkeypatch.setattr(kv, "_fetch_refs_for", partial)
    audit = {}
    result = call(frames, audit=audit, query_video_sha256="a"*64)
    assert result["status"] == audit["status"] == "complete"
    assert result["provenance"]["acquisition"]["status"] == "unknown"
    assert result["provenance"]["acquisition"]["listing_complete"] is False


def test_unserved_reference_partition_is_explicit_without_inventing_download_reason():
    identity = {"listed_ref_ids": ["mapillary:1", "mapillary:2"],
        "listed_ref_latlon": [[50., 8.], [51., 9.]],
        "listing_snapshot": {"n_refs": 2, "refs_sha256": "a"*64}}
    fetch = {"listing": "tiles", "lost": 0, "listing_stats_known": True,
             "snapshot": {"n_refs": 2, "refs_sha256": "a"*64}}
    result = kv._evidence_acquisition(identity, ["mapillary:1"], "mapillary", 1500, fetch)
    assert result["status"] == "complete"
    assert result["listed_references"] == [{"id": "1", "lat": 50., "lon": 8.},
        {"id": "2", "lat": 51., "lon": 9.}]
    assert result["unserved_references"] == [{"ref_id": "mapillary:2", "reason": "not_in_embedded_reference_snapshot"}]
    with pytest.raises(ValueError, match="partition"):
        kv._evidence_acquisition(identity, ["mapillary:3"], "mapillary", 1500, fetch)


def test_legacy_graph_listing_zero_loss_is_not_a_pagination_completeness_proof(inputs, monkeypatch):
    frames, _, _ = inputs
    original = kv._fetch_refs_for
    def graph(*args):
        refs = original(*args)
        kv.LAST_FETCH_STATS["listing"] = "graph"
        return refs
    monkeypatch.setattr(kv, "_fetch_refs_for", graph)
    result = call(frames, audit={}, query_video_sha256="a"*64)
    assert result["status"] == "complete"
    assert result["provenance"]["acquisition"]["listing_complete"] is False
