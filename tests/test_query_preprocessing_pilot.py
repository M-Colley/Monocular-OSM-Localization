"""A cached pilot must never pair embeddings with unrelated coordinates."""

import json

import numpy as np
import pytest

from monocular_osm.kartaview_vpr import _refs_fingerprint
from scripts.pilot_query_preprocessing import load_cached_references


def test_dense_cache_keeps_reference_order_and_records_content_hash(tmp_path):
    source = tmp_path / "dense.npz"
    ll = np.array([[48.4, 9.99], [48.41, 10.0]])
    np.savez(source, ids=np.array(["z", "a"]), latlon=ll,
             emb=np.array([[1, 2], [3, 4]], dtype=np.float16))
    emb, coords, info = load_cached_references(source)
    np.testing.assert_array_equal(coords, ll)
    np.testing.assert_array_equal(emb, [[1, 2], [3, 4]])
    assert emb.dtype == np.float32 and len(info["sha256"]) == 64


def test_archived_metadata_must_match_embedding_fingerprint(tmp_path):
    refs = [{"id": "z", "lat": 48.4, "lon": 9.99},
            {"id": "a", "lat": 48.41, "lon": 10.0}]
    (tmp_path / "mly_ref_meta_test.json").write_text(json.dumps({"refs": refs}), encoding="utf-8")
    source = tmp_path / "ref_emb_megaloc_test.npz"
    np.savez(source, emb=np.eye(2), fingerprint="wrong reference pool")
    with pytest.raises(ValueError, match="fingerprint-matched"):
        load_cached_references(tmp_path)
    np.savez(source, emb=np.eye(2), fingerprint=_refs_fingerprint(refs))
    _emb, coords, _info = load_cached_references(tmp_path)
    np.testing.assert_array_equal(coords, [[48.4, 9.99], [48.41, 10.0]])
