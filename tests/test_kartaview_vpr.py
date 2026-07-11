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
# _fetch_refs_mapillary — warm cache must serve WITHOUT a token (offline sweeps)
# ---------------------------------------------------------------------------


def _write_mly_cache(tmp_path: Path, center, radius_m, cap=1500,
                     store: str | None = "match") -> list[dict]:
    """Write a warm Mapillary cache: a signature-tagged meta + the per-id
    image store. ``store``: "match" stores every ref's pixels (genuine warm
    cache); "partial" stores only some (interrupted rebuild — warm serve must
    decline); None writes no image store."""
    refs = [{"id": i, "lat": center[0], "lon": center[1]} for i in range(3)]
    sig = kv._fetch_signature(center, radius_m, cap)
    (tmp_path / f"mly_ref_meta_{kv._sig_tag(sig)}.json").write_text(
        json.dumps({"signature": sig, "refs": refs}), encoding="utf-8")
    if store is not None:
        ids = [str(r["id"]) for r in refs]
        if store == "partial":
            ids = ids[:1]
        # merge into any existing store so two signatures can share one
        p = tmp_path / "ref_img_store.npz"
        if p.exists():
            with np.load(p, allow_pickle=True) as d:
                have = [str(x) for x in d["ids"]]
                raw0 = np.asarray(d["raw"])
            add = [i for i in ids if i not in have]
            allids = have + add
            allraw = np.concatenate(
                [raw0, np.zeros((len(add), 4, 4, 3), np.uint8)]) if add else raw0
        else:
            allids = ids
            allraw = np.zeros((len(ids), 4, 4, 3), np.uint8)
        np.savez(p, ids=np.array(allids), raw=allraw)
    return refs


def test_fetch_refs_mapillary_warm_cache_needs_no_token(
    tmp_path: Path, monkeypatch
) -> None:
    """A signature-matching ref cache is served before the token guard, with
    zero network — offline GT sweeps must not silently lose the VPR prior."""
    refs = _write_mly_cache(tmp_path, ULM, 500.0)
    monkeypatch.delenv("MLY_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "requests", _FakeRequests(fail=True))
    got = kv._fetch_refs_mapillary(ULM, 500.0, str(tmp_path), token=None)
    assert [r["id"] for r in got] == [r["id"] for r in refs]


def test_fetch_refs_mapillary_no_token_no_cache_returns_empty(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("MLY_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "requests", _FakeRequests(fail=True))
    assert kv._fetch_refs_mapillary(ULM, 500.0, str(tmp_path), token=None) == []


def test_fetch_refs_mapillary_two_signatures_coexist(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression (2026-07-11): with a single untagged meta the deployable
    (wide-disc) and GT-seeded (tight-disc) configs of one clip CLOBBERED each
    other's cache — the auto-sizing runs overwrote Ulm's GT-seeded cache and
    every later tokenless run lost the VPR channel. Signature-tagged metas +
    a shared per-id image store must let BOTH serve from one dir."""
    monkeypatch.delenv("MLY_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "requests", _FakeRequests(fail=True))
    _write_mly_cache(tmp_path, ULM, 500.0)               # variant A (tight)
    _write_mly_cache(tmp_path, ULM, 8000.0, cap=3000)    # variant B (wide)
    got_a = kv._fetch_refs_mapillary(ULM, 500.0, str(tmp_path), token=None)
    got_b = kv._fetch_refs_mapillary(ULM, 8000.0, str(tmp_path), token=None,
                                     cap=3000)
    assert [r["id"] for r in got_a] == [0, 1, 2]
    assert [r["id"] for r in got_b] == [0, 1, 2]
    assert kv.has_mapillary_cache(str(tmp_path))


def test_fetch_refs_mapillary_partial_store_declines_tokenless(
    tmp_path: Path, monkeypatch
) -> None:
    """The per-id image store must hold EVERY served id's pixels before the
    tokenless warm serve fires — a partial store (interrupted rebuild) would
    otherwise serve a meta whose missing ids get silently dropped downstream.
    With only some pixels present and no token, degrade to [] gracefully."""
    _write_mly_cache(tmp_path, ULM, 500.0, store="partial")
    monkeypatch.delenv("MLY_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "requests", _FakeRequests(fail=True))
    assert kv._fetch_refs_mapillary(ULM, 500.0, str(tmp_path), token=None) == []
    # complete the store -> warm serve fires
    _write_mly_cache(tmp_path, ULM, 500.0, store="match")
    got = kv._fetch_refs_mapillary(ULM, 500.0, str(tmp_path), token=None)
    assert [r["id"] for r in got] == [0, 1, 2]


def test_fetch_refs_mapillary_stale_signature_still_needs_token(
    tmp_path: Path, monkeypatch
) -> None:
    """A cache built for OTHER params cannot cover the query: without a token
    the fetch degrades to no refs instead of serving the wrong disc."""
    _write_mly_cache(tmp_path, ULM, 500.0)
    monkeypatch.delenv("MLY_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "requests", _FakeRequests(fail=True))
    assert kv._fetch_refs_mapillary(ULM, 900.0, str(tmp_path), token=None) == []


def test_has_mapillary_cache(tmp_path: Path) -> None:
    assert not kv.has_mapillary_cache(None)
    assert not kv.has_mapillary_cache(str(tmp_path))
    _write_mly_cache(tmp_path, ULM, 500.0)
    assert kv.has_mapillary_cache(str(tmp_path))


# ---------------------------------------------------------------------------
# _resolve_backbone — the MegaLoc->eigenplaces fallback must survive a warm cache
# ---------------------------------------------------------------------------


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self


class _FakeTorch:
    """Minimal torch stand-in: hub.load fails for MegaLoc, succeeds otherwise."""

    def __init__(self):
        self.loads: list[str] = []

        class _Hub:
            def load(_self, repo, fn, **kw):
                self.loads.append(repo)
                if "MegaLoc" in repo:
                    raise RuntimeError("hub unreachable")
                return _FakeModel()
        self.hub = _Hub()

        class _Cuda:
            @staticmethod
            def is_available():
                return False
        self.cuda = _Cuda()


def test_resolve_backbone_fallback_survives_warm_cache(monkeypatch) -> None:
    """When MegaLoc's hub fetch fails, _resolve_backbone falls back to
    eigenplaces AND remembers it — a later call with a warm _MODEL must not
    relabel the resident eigenplaces weights as 'megaloc' (which would key the
    ref-embedding cache to the wrong backbone and dot two embedding spaces)."""
    monkeypatch.setattr(kv, "_MODEL", None)
    monkeypatch.setattr(kv, "_MODEL_NAME", None)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    # 1st call: MegaLoc fails -> resolves + loads eigenplaces, remembers it
    assert kv._resolve_backbone("megaloc", "cpu") == "eigenplaces"
    assert kv._MODEL is not None and kv._MODEL_NAME == "eigenplaces"
    # 2nd call with a warm _MODEL: still eigenplaces, NOT megaloc (the bug)
    assert kv._resolve_backbone("megaloc", "cpu") == "eigenplaces"


def test_resolve_backbone_megaloc_success_labels_megaloc(monkeypatch) -> None:
    """When MegaLoc loads, the resolved name is 'megaloc' and is remembered."""
    class _OkTorch(_FakeTorch):
        def __init__(self):
            super().__init__()

            class _Hub:
                def load(_self, repo, fn, **kw):
                    return _FakeModel()
            self.hub = _Hub()

    monkeypatch.setattr(kv, "_MODEL", None)
    monkeypatch.setattr(kv, "_MODEL_NAME", None)
    monkeypatch.setitem(sys.modules, "torch", _OkTorch())
    assert kv._resolve_backbone("megaloc", "cpu") == "megaloc"
    assert kv._MODEL_NAME == "megaloc"


# ---------------------------------------------------------------------------
# _viterbi_decode — continuity kills confident-but-wrong retrievals
# ---------------------------------------------------------------------------


def test_viterbi_overrides_confident_outlier() -> None:
    """Refs A0..A4 lie along a street ~100 m apart; ref B sits 3 km away.
    Frame 2's argmax confidently picks B; its neighbours pick the A-chain.
    The decode must keep frame 2 on the street."""
    lat0 = 48.4
    step = 100.0 / 111320.0
    ref_ll = np.array([[lat0 + i * step, 9.99] for i in range(5)]
                      + [[lat0 + 3000.0 / 111320.0, 9.99]])
    sims = np.full((5, 6), 0.1)
    for q in range(5):
        sims[q, q] = 0.8                  # true chain
    sims[2, 5] = 0.95                     # confident outlier at frame 2
    sims[2, 2] = 0.60
    path = kv._viterbi_decode(sims, ref_ll, dt_s=2.0)
    assert path[2] == 2                   # continuity beats the outlier
    assert list(path) == [0, 1, 2, 3, 4]
    # per-frame argmax WOULD have taken the bait
    assert int(sims[2].argmax()) == 5


def test_viterbi_free_radius_allows_normal_motion() -> None:
    """Consecutive refs within the plausible-drive radius carry no penalty:
    a clean argmax chain is returned unchanged."""
    lat0 = 48.4
    step = 60.0 / 111320.0
    ref_ll = np.array([[lat0 + i * step, 9.99] for i in range(4)])
    sims = np.eye(4) * 0.9 + 0.05
    path = kv._viterbi_decode(sims, ref_ll, dt_s=2.0)
    assert list(path) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# _fetch_refs_panoramax — STAC search, signed cache, stable ordering
# ---------------------------------------------------------------------------


class _FakePanoramax:
    """requests stand-in for the Panoramax STAC ``/search`` endpoint."""

    def __init__(self, features=None, fail=False):
        self.features = features or []
        self.fail = fail
        self.calls: list[dict] = []

    def Session(self):  # noqa: N802 - mimics requests API
        outer = self

        class _Resp:
            def __init__(self, payload):
                self._payload = payload

            def json(self):
                return self._payload

        class _Sess:
            def get(self, url, params=None, timeout=None):
                if outer.fail:
                    raise AssertionError("network must not be hit")
                outer.calls.append(dict(params or {}))
                return _Resp({"features": outer.features})

        return _Sess()


def _pnx_feature(fid, lat, lon, assets=None):
    if assets is None:
        assets = {"sd": {"href": f"http://pnx/{fid}/sd.jpg"},
                  "hd": {"href": f"http://pnx/{fid}/hd.jpg"}}
    return {"id": fid, "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "assets": assets}


def test_fetch_refs_panoramax_parses_and_sorts(tmp_path: Path, monkeypatch) -> None:
    fake = _FakePanoramax(features=[
        _pnx_feature("b", 48.40, 9.99),
        _pnx_feature("a", 48.41, 9.98),
        {"id": "broken", "geometry": {"coordinates": [9.97, 48.39]}, "assets": {}},
    ])
    monkeypatch.setitem(sys.modules, "requests", fake)
    refs = kv._fetch_refs_panoramax(ULM, 500.0, str(tmp_path))
    # sorted by id, the asset-less feature dropped, sd asset preferred
    assert [r["id"] for r in refs] == ["a", "b"]
    assert refs[0]["url"].endswith("/a/sd.jpg")
    assert refs[0]["lat"] == pytest.approx(48.41)
    assert len(fake.calls) > 0 and "bbox" in fake.calls[0]


def test_fetch_refs_panoramax_cache_hit_and_invalidation(
    tmp_path: Path, monkeypatch
) -> None:
    fake = _FakePanoramax(features=[_pnx_feature("x", 48.40, 9.99)])
    monkeypatch.setitem(sys.modules, "requests", fake)
    refs = kv._fetch_refs_panoramax(ULM, 500.0, str(tmp_path))
    assert [r["id"] for r in refs] == ["x"]
    # warm hit: same params, network forbidden
    monkeypatch.setitem(sys.modules, "requests", _FakePanoramax(fail=True))
    assert kv._fetch_refs_panoramax(ULM, 500.0, str(tmp_path)) == refs
    # different radius -> signature mismatch -> refetch
    fake2 = _FakePanoramax(features=[_pnx_feature("y", 48.40, 9.99)])
    monkeypatch.setitem(sys.modules, "requests", fake2)
    assert [r["id"] for r in
            kv._fetch_refs_panoramax(ULM, 900.0, str(tmp_path))] == ["y"]


def test_fetch_refs_panoramax_caps_deterministically(
    tmp_path: Path, monkeypatch
) -> None:
    feats = [_pnx_feature(f"id{i:04d}", 48.40 + i * 1e-5, 9.99) for i in range(40)]
    fake = _FakePanoramax(features=feats)
    monkeypatch.setitem(sys.modules, "requests", fake)
    refs = kv._fetch_refs_panoramax(ULM, 500.0, str(tmp_path), cap=10)
    assert len(refs) == 10
    assert refs == sorted(refs, key=lambda r: r["id"])   # subsample keeps order


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


def test_load_ref_images_per_id_reuse_and_own_coords(tmp_path: Path, monkeypatch) -> None:
    """Per-id store (R1): an overlapping ref set reuses the shared ids' pixels
    (no re-download — the old monolithic cache refetched EVERYTHING on any
    drift), and each image keeps ITS OWN stored lat/lon, so a cached photo can
    never be mislabelled with another ref's coordinates."""
    refs_a = _refs([(48.40, 9.99), (48.41, 9.98)])       # ids 0, 1
    monkeypatch.setitem(sys.modules, "requests",
                        _FakeRequests(image_bytes=_jpg_bytes()))
    _load_ref_images(refs_a, str(tmp_path))

    # Overlap id 0 (a wrong coord claim it must IGNORE) + two genuinely new ids.
    refs_b = [{"id": 0, "lat": 99.0, "lon": 99.0, "url": "http://x/0"},
              {"id": 2, "lat": 48.50, "lon": 9.80, "url": "http://x/2"},
              {"id": 3, "lat": 48.51, "lon": 9.81, "url": "http://x/3"}]
    fake_b = _FakeRequests(image_bytes=_jpg_bytes())
    monkeypatch.setitem(sys.modules, "requests", fake_b)
    raw, ref_xy, fp = _load_ref_images(refs_b, str(tmp_path))
    assert sorted(fake_b.get_calls) == ["http://x/2", "http://x/3"]  # id 0 reused
    assert raw.shape[0] == 3
    # id 0 keeps its ORIGINAL (48.40, 9.99), NOT refs_b's bogus (99, 99)
    np.testing.assert_allclose(
        ref_xy, [[48.40, 9.99], [48.50, 9.80], [48.51, 9.81]])
    assert fp == _refs_fingerprint(refs_b)


def test_load_ref_images_serves_url_less_refs_from_store(tmp_path: Path, monkeypatch) -> None:
    """Union-store ids reach _load_ref_images with url=None; they must still be
    served from the per-id store (the tokenless warm path depends on it), and
    ids with neither a url nor a stored image are dropped, not fetched-as-None."""
    refs = _refs([(48.40, 9.99), (48.41, 9.98)])
    monkeypatch.setitem(sys.modules, "requests",
                        _FakeRequests(image_bytes=_jpg_bytes()))
    _load_ref_images(refs, str(tmp_path))
    url_less = [{"id": 0, "lat": 48.40, "lon": 9.99},          # stored -> served
                {"id": 9, "lat": 48.60, "lon": 9.70}]          # unknown -> dropped
    monkeypatch.setitem(sys.modules, "requests", _FakeRequests(fail=True))
    raw, ref_xy, _ = _load_ref_images(url_less, str(tmp_path))
    assert raw.shape[0] == 1
    np.testing.assert_allclose(ref_xy, [[48.40, 9.99]])


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
