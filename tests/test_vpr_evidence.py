"""Joint evidence preserves every score and the exact servable photo identity."""
import numpy as np
import pytest

from monocular_osm import kartaview_vpr as kv


def test_full_evidence_uses_actual_video_times_and_same_query_cache(monkeypatch):
    torch = pytest.importorskip("torch")
    refs = [{"id": i, "lat": 50 + i * .001, "lon": 8} for i in range(40)]
    frames = [np.zeros((2, 4, 3), np.uint8) for _ in range(3)]
    cache = {}
    monkeypatch.setattr(kv, "_resolve_backbone", lambda *args: "megaloc")
    monkeypatch.setattr(kv, "_MODEL_NAME", "megaloc")
    monkeypatch.setattr(kv, "_MODEL_DEVICE", "cpu")
    def fetch(*args):
        kv.LAST_FETCH_STATS = {"tiles_failed": 2, "snapshot": {"hash": "example"}}
        return refs
    monkeypatch.setattr(kv, "_fetch_refs_for", fetch)
    monkeypatch.setattr(kv, "_embed_refs", lambda *args, **kwargs: (
        torch.ones((40, 3)), np.array([[r["lat"], r["lon"]] for r in refs])))
    def similarities(source, indices, embeddings, device, **kwargs):
        assert source is frames and kwargs["query_cache"] is cache
        np.testing.assert_array_equal(indices, [0, 2])
        return np.arange(80, dtype=np.float32).reshape(2, 40)
    monkeypatch.setattr(kv, "_query_similarities", similarities)
    result = kv.kartaview_vpr_evidence(frames, [4, 5.1, 6.4], (50, 8), n_query=2,
                                        device="cpu", query_cache=cache)
    assert result is not None, kv.LAST_FAILURE
    np.testing.assert_array_equal(result["query_times"], [4, 6.4])
    np.testing.assert_array_equal(result["similarities"], np.arange(80).reshape(2, 40))
    assert result["ref_ids"].tolist() == [f"mapillary:{i}" for i in range(40)]
    assert result["provenance"]["preprocessing"]["dimensions"] == 3
    assert "unverified" in result["provenance"]["reference_encoding_identity"]
    kv.LAST_FETCH_STATS["snapshot"]["hash"] = "later pool"
    assert result["provenance"]["fetch"]["snapshot"]["hash"] == "example"


def test_partial_pool_ids_follow_embedded_rows_instead_of_store_order(tmp_path):
    refs = [{"id": "a"}, {"id": "missing"}, {"id": "c"}]
    np.savez(tmp_path / "ref_img_store.npz", ids=np.array(["c", "a"]), xy=[[5, 6], [1, 2]])
    result = kv._embedded_reference_ids(refs, np.array([[1, 2], [5, 6]]), tmp_path, "mapillary")
    assert result.tolist() == ["mapillary:a", "mapillary:c"]
    with pytest.raises(ValueError, match="coordinate order"):
        kv._embedded_reference_ids(refs, np.array([[5, 6], [1, 2]]), tmp_path, "mapillary")


@pytest.mark.parametrize("times", [[1], [1, 1], [2, 1], [1, float("nan")]])
def test_invalid_video_timeline_never_fetches_references(monkeypatch, times):
    monkeypatch.setattr(kv, "_prepare_refs_and_query", lambda *args, **kwargs: pytest.fail("must not fetch"))
    assert kv.kartaview_vpr_evidence([object(), object()], times, (50, 8)) is None
    assert "timestamps" in kv.LAST_FAILURE


def test_unidentified_union_is_explicitly_unavailable(monkeypatch):
    monkeypatch.setattr(kv, "_prepare_refs_and_query", lambda *args, **kwargs: pytest.fail("must not fetch"))
    assert kv.kartaview_vpr_evidence([object(), object()], [0, 1], (50, 8), source="union") is None
    assert "single reference source" in kv.LAST_FAILURE


def test_partial_without_per_id_cache_cannot_invent_identity():
    with pytest.raises(ValueError, match="per-ID cache"):
        kv._embedded_reference_ids([{"id": 1}, {"id": 2}], np.zeros((1, 2)), None, "mapillary")


@pytest.mark.parametrize("kwargs", [{"n_query": 1.5}, {"n_query": True}, {"cap": 0},
                                    {"query_batch_size": 0}, {"radius_m": float("inf")},
                                    {"source": "typo"}])
def test_invalid_budget_or_disc_never_downloads_references(monkeypatch, kwargs):
    monkeypatch.setattr(kv, "_prepare_refs_and_query", lambda *args, **kw: pytest.fail("must not fetch"))
    assert kv.kartaview_vpr_evidence([object(), object()], [0, 1], (50, 8), **kwargs) is None
    assert kv.LAST_FAILURE
