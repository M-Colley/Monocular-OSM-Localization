"""Tests for the KartaView VPR caches + robust centre (network faked, no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import src.kartaview_vpr as kv
from src.kartaview_vpr import (
    _fetch_refs,
    _geometric_median,
    _load_ref_images,
    _refs_fingerprint,
    _robust_center,
)

ULM = (48.3984, 9.9916)


def _jpg_bytes() -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.full((8, 8, 3), 128, np.uint8))
    assert ok
    return buf.tobytes()


class _FakeRequests:
    """Stands in for the ``requests`` module (imported inside the functions).

    ``api_items`` -> nearby-photos POST payload; ``image_bytes`` -> CDN GET.
    Raises when ``fail`` so tests can prove a call was (not) made.
    """

    def __init__(self, api_items=None, image_bytes=b"", fail=False):
        self.api_items = api_items or []
        self.image_bytes = image_bytes
        self.fail = fail
        self.post_calls: list[dict] = []
        self.get_calls: list[str] = []

    def Session(self):  # noqa: N802 - mimics requests API
        outer = self

        class _Resp:
            def __init__(self, payload=None, content=b""):
                self._payload = payload
                self.content = content
                self.status_code = 200

            def json(self):
                return self._payload

        class _Sess:
            def post(self, url, data=None, timeout=None):
                if outer.fail:
                    raise AssertionError("network must not be hit")
                outer.post_calls.append(dict(data))
                return _Resp(payload={"currentPageItems": outer.api_items})

            def get(self, url, timeout=None):
                if outer.fail:
                    raise AssertionError("network must not be hit")
                outer.get_calls.append(url)
                return _Resp(content=outer.image_bytes)

        return _Sess()


def _api_item(pid, lat, lon):
    return {"id": pid, "lat": lat, "lng": lon, "lth_name": "storage6/x.jpg"}


def _refs(coords):
    return [{"id": i, "lat": la, "lon": lo, "url": f"http://x/{i}"}
            for i, (la, lo) in enumerate(coords)]


# ---------------------------------------------------------------------------
# _fetch_refs — cache must be keyed by (center, radius, cap)
# ---------------------------------------------------------------------------


def test_fetch_refs_ignores_legacy_unsigned_cache(tmp_path: Path, monkeypatch) -> None:
    # Legacy format: a bare list with no fetch signature — params unknowable.
    stale = [{"id": 1, "lat": 1.0, "lon": 1.0, "url": "http://old"}]
    (tmp_path / "ref_meta.json").write_text(json.dumps(stale), encoding="utf-8")
    fake = _FakeRequests(api_items=[_api_item(7, 48.40, 9.99)])
    monkeypatch.setitem(sys.modules, "requests", fake)
    refs = _fetch_refs(ULM, 500.0, str(tmp_path))
    assert [r["id"] for r in refs] == [7]          # refetched, not the stale list
    assert len(fake.post_calls) > 0


def test_fetch_refs_cache_hit_and_param_invalidation(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeRequests(api_items=[_api_item(7, 48.40, 9.99)])
    monkeypatch.setitem(sys.modules, "requests", fake)
    first = _fetch_refs(ULM, 500.0, str(tmp_path))
    assert [r["id"] for r in first] == [7]

    # Same params -> served from cache, network untouched.
    offline = _FakeRequests(fail=True)
    monkeypatch.setitem(sys.modules, "requests", offline)
    again = _fetch_refs(ULM, 500.0, str(tmp_path))
    assert again == first

    # Changed radius -> stale cache must NOT be served (the old code returned
    # it, making --vpr-search-radius a silent no-op).
    fake2 = _FakeRequests(api_items=[_api_item(9, 48.41, 9.98)])
    monkeypatch.setitem(sys.modules, "requests", fake2)
    wider = _fetch_refs(ULM, 8000.0, str(tmp_path))
    assert [r["id"] for r in wider] == [9]
    assert len(fake2.post_calls) > 0


def test_fetch_refs_grid_lon_step_is_latitude_aware(tmp_path: Path, monkeypatch) -> None:
    def lon_step_at(lat):
        fake = _FakeRequests(api_items=[])
        monkeypatch.setitem(sys.modules, "requests", fake)
        _fetch_refs((lat, 0.0), 800.0, None)
        lons = sorted({round(float(c["lng"]), 6) for c in fake.post_calls})
        return lons[1] - lons[0]

    # At 60N (cos = 0.5) the lon step must be ~2x the equator step so query
    # discs tile uniformly in METRES, not degrees.
    assert lon_step_at(60.0) / lon_step_at(0.0) == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# _load_ref_images — npz must be fingerprinted against the refs list
# ---------------------------------------------------------------------------


def test_load_ref_images_serves_matching_cache_offline(tmp_path: Path, monkeypatch) -> None:
    refs = _refs([(48.40, 9.99), (48.41, 9.98)])
    fake = _FakeRequests(image_bytes=_jpg_bytes())
    monkeypatch.setitem(sys.modules, "requests", fake)
    raw, ref_xy, fp = _load_ref_images(refs, str(tmp_path))
    assert raw.shape[0] == 2 and fp == _refs_fingerprint(refs)
    np.testing.assert_allclose(ref_xy, [[48.40, 9.99], [48.41, 9.98]])

    # Warm rerun: identical refs -> cache hit, no network.
    monkeypatch.setitem(sys.modules, "requests", _FakeRequests(fail=True))
    raw2, ref_xy2, _ = _load_ref_images(refs, str(tmp_path))
    np.testing.assert_array_equal(raw2, raw)
    np.testing.assert_allclose(ref_xy2, ref_xy)


def test_load_ref_images_invalidates_on_refs_change(tmp_path: Path, monkeypatch) -> None:
    """A regenerated refs list must NOT be paired with the stale npz (the old
    code dereferenced stale keep-indices into the new list — silently wrong
    GPS labels for every retrieval hit)."""
    refs_a = _refs([(48.40, 9.99), (48.41, 9.98)])
    monkeypatch.setitem(sys.modules, "requests",
                        _FakeRequests(image_bytes=_jpg_bytes()))
    _load_ref_images(refs_a, str(tmp_path))

    # "User deleted ref_meta.json and refetched": different, longer refs list.
    refs_b = _refs([(51.50, -0.12), (51.51, -0.10), (51.52, -0.09)])
    fake_b = _FakeRequests(image_bytes=_jpg_bytes())
    monkeypatch.setitem(sys.modules, "requests", fake_b)
    raw, ref_xy, fp = _load_ref_images(refs_b, str(tmp_path))
    assert len(fake_b.get_calls) == 3              # re-downloaded, not served stale
    assert raw.shape[0] == 3
    np.testing.assert_allclose(
        ref_xy, [[51.50, -0.12], [51.51, -0.10], [51.52, -0.09]])
    assert fp == _refs_fingerprint(refs_b)


def test_load_ref_images_ignores_legacy_npz(tmp_path: Path, monkeypatch) -> None:
    # Pre-fingerprint cache layout: raw + keep only. Must be refetched.
    np.savez(tmp_path / "ref_imgs.npz",
             raw=np.zeros((1, 512, 512, 3), np.uint8), keep=np.array([0]))
    refs = _refs([(48.40, 9.99), (48.41, 9.98)])
    fake = _FakeRequests(image_bytes=_jpg_bytes())
    monkeypatch.setitem(sys.modules, "requests", fake)
    raw, ref_xy, _ = _load_ref_images(refs, str(tmp_path))
    assert len(fake.get_calls) == 2
    assert raw.shape[0] == 2 and len(ref_xy) == 2


# ---------------------------------------------------------------------------
# _embed_refs — embedding cache keyed by (image fingerprint, model)
# ---------------------------------------------------------------------------


def test_embed_refs_embedding_cache_skips_reembed(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    refs = _refs([(48.40, 9.99), (48.41, 9.98)])
    monkeypatch.setitem(sys.modules, "requests",
                        _FakeRequests(image_bytes=_jpg_bytes()))
    kv._MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    kv._STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    calls = []

    def fake_embed(device, imgs):
        calls.append(len(imgs))
        return torch.ones((len(imgs), 4))

    monkeypatch.setattr(kv, "_embed", fake_embed)
    emb1, xy1 = kv._embed_refs(refs, "cpu", str(tmp_path), model_name="megaloc")
    assert calls == [2]

    # Warm rerun, same refs/model: served from the emb cache — no GPU pass.
    emb2, xy2 = kv._embed_refs(refs, "cpu", str(tmp_path), model_name="megaloc")
    assert calls == [2]                            # _embed not called again
    np.testing.assert_allclose(emb2.numpy(), emb1.numpy())
    np.testing.assert_allclose(xy2, xy1)

    # Different model name -> its own cache entry -> re-embed.
    kv._embed_refs(refs, "cpu", str(tmp_path), model_name="eigenplaces")
    assert calls == [2, 2]


# ---------------------------------------------------------------------------
# _robust_center — Weiszfeld must run in a local METRIC frame
# ---------------------------------------------------------------------------


def test_robust_center_matches_metric_geometric_median() -> None:
    import math

    # Asymmetric cloud at 60N (cos(lat)=0.5) where degree-space and
    # metric-space medians clearly differ.
    latlons = np.array([
        [60.000, 0.000], [60.000, 0.050], [60.001, 0.025],
        [59.999, 0.025], [60.010, 0.000],
    ])
    sims = np.ones(len(latlons))

    # Reference: geometric median computed in the local metric frame.
    dm = 111320.0
    coslat = math.cos(math.radians(latlons[:, 0].mean()))
    xy = np.column_stack([latlons[:, 1] * dm * coslat, latlons[:, 0] * dm])
    ref = _geometric_median(xy, weights=sims)
    ref_lat, ref_lon = ref[1] / dm, ref[0] / (dm * coslat)

    lat, lon = _robust_center(latlons, sims)
    # Must agree with the metric median to ~1 m.
    assert abs(lat - ref_lat) * dm < 1.0
    assert abs(lon - ref_lon) * dm * coslat < 1.0

    # ... and the raw-degree-space median (the old behaviour) is measurably
    # elsewhere, so this test discriminates.
    deg = _geometric_median(latlons, weights=sims)
    d_deg = math.hypot((deg[0] - ref_lat) * dm,
                       (deg[1] - ref_lon) * dm * coslat)
    assert d_deg > 10.0
