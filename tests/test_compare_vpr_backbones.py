"""scripts/compare_vpr_backbones.py: scoring and the archive read path.

No models: the scoring is checked on synthetic similarities whose answers are
known, and the archive reader on a tiny fake cold-run archive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_vpr_backbones as C  # noqa: E402

from monocular_osm.kartaview_vpr import _refs_fingerprint  # noqa: E402


def _grid_refs(n=40, lat0=48.40, lon0=9.99, step=0.001):
    return np.array([[lat0 + step * (i // 8), lon0 + step * (i % 8)] for i in range(n)])


def test_score_rewards_retrieving_the_right_place() -> None:
    refs = _grid_refs()
    q_gt = refs[[3, 10, 17, 24, 31]] + 1e-5                  # each query ~1 m from a ref
    route = q_gt.copy()
    perfect = np.full((5, len(refs)), 0.1)
    perfect[np.arange(5), [3, 10, 17, 24, 31]] = 0.9
    s = C.score(perfect, refs, q_gt, route)
    assert s["r1@25m"] == 1.0 and s["top1_err_median_m"] < 5 and s["centre_err_m"] < 150
    wrong = np.full((5, len(refs)), 0.1)
    wrong[:, 39] = 0.2                                        # always the far corner
    w = C.score(wrong, refs, q_gt, route)
    assert w["r1@100m"] == 0.0 and w["top1_err_median_m"] > 300


def test_auroc_of_the_confidence() -> None:
    assert C.auroc(np.array([0.3, 0.4]), np.array([0.1, 0.2])) == 1.0
    assert C.auroc(np.array([0.1]), np.array([0.3])) == 0.0
    assert C.auroc(np.array([]), np.array([0.3])) is None


def _fake_archive(tmp_path: Path, n=35):
    d = tmp_path / "rep0" / "mapillary"
    d.mkdir(parents=True)
    ll = _grid_refs(n)
    refs = [{"id": str(1000 + i), "lat": float(a), "lon": float(b)} for i, (a, b) in enumerate(ll)]
    (d / "mly_ref_meta_abc.json").write_text(json.dumps(
        {"signature": {"center": [48.4, 9.99], "radius_m": 3000.0, "cap": 1500},
         "refs": refs + [{"id": "9999", "lat": 0.0, "lon": 0.0}]}), encoding="utf-8")
    raw = np.zeros((n, 8, 8, 3), np.uint8)
    np.savez(d / "ref_img_store.npz", ids=np.array([r["id"] for r in refs]), raw=raw)
    return tmp_path / "rep0", refs


def test_archived_refs_keep_only_stored_images_in_order(tmp_path: Path) -> None:
    rep, refs = _fake_archive(tmp_path)
    ids, img, ll, sig, dicts = C.archived_refs(rep, "mapillary")
    assert ids == [r["id"] for r in refs] and len(img) == len(refs)   # unstored "9999" dropped
    assert sig["radius_m"] == 3000.0 and ll.shape == (len(refs), 2)


def test_archived_megaloc_embeddings_are_used_only_with_a_matching_fingerprint(tmp_path) -> None:
    rep, refs = _fake_archive(tmp_path)
    emb = np.random.default_rng(0).normal(size=(len(refs), 4)).astype(np.float32)
    np.savez(rep / "mapillary" / "ref_emb_megaloc_x.npz", emb=emb,
             fingerprint=np.array(_refs_fingerprint(refs)))
    got = C.archived_megaloc(rep, "mapillary", refs)
    assert got is not None and np.allclose(got, emb)
    assert C.archived_megaloc(rep, "mapillary", refs[::-1]) is None    # other order: refuse


def test_gt_is_interpolated_at_query_times() -> None:
    wps = [{"t_sec": 0.0, "lat": 48.0, "lon": 9.0}, {"t_sec": 10.0, "lat": 48.1, "lon": 9.2}]
    assert C.gt_at(wps, np.array([5.0]))[0] == pytest.approx([48.05, 9.1])
