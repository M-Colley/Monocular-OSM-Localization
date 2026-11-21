"""Tests for worldwide Unicode/exonym support, metro-dense VPR, and Stage 0 city inference."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import benchmark as bm  # noqa: E402
from monocular_osm import city_inference, global_geoloc, kartaview_vpr as kv, metro_vpr  # noqa: E402
from monocular_osm import location_prior, osm_gazetteer, text_anchor  # noqa: E402
from monocular_osm.cli import _validate_input_args, build_arg_parser  # noqa: E402
from monocular_osm.pipeline import PipelineConfig, RunContext  # noqa: E402
from monocular_osm.stages.prior import vpr_coarse_prior  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Worldwide Unicode scripts, camera stopwords, extent sizing & fairness
# ---------------------------------------------------------------------------


def test_gazetteer_preserves_and_matches_world_scripts():
    assert osm_gazetteer._norm_name("Łódź Fabryczna") == "lodz fabryczna"
    assert osm_gazetteer._norm_name("København H") == "kobenhavn h"
    assert osm_gazetteer._norm_name("Булевард Витоша") == "булевард витоша"
    assert osm_gazetteer._norm_name("新宿駅東口") == "新宿駅東口"
    assert osm_gazetteer._norm_name("강남역") == "강남역"
    assert osm_gazetteer._norm_name("صيدلية النهدي") == "صيدلية النهدي"

    entries = [
        {"name": "新宿駅", "norm": osm_gazetteer._norm_name("新宿駅"),
         "lat": 35.6909, "lon": 139.7003, "kind": "station"},
        {"name": "강남역", "norm": osm_gazetteer._norm_name("강남역"),
         "lat": 37.4979, "lon": 127.0276, "kind": "station"},
        {"name": "Булевард Витоша", "norm": osm_gazetteer._norm_name("Булевард Витоша"),
         "lat": 42.6915, "lon": 23.3199, "kind": "street"},
        {"name": "Łódź Kaliska", "norm": osm_gazetteer._norm_name("Łódź Kaliska"),
         "lat": 51.7578, "lon": 19.4267, "kind": "station"},
    ]
    hits = osm_gazetteer.match_texts(
        [
            ("新宿駅", 0.95, 1.0),
            ("강남역", 0.95, 2.0),
            ("Булевард Витоша", 0.95, 3.0),
            ("Lodz Kaliska", 0.95, 4.0),
        ],
        {"entries": entries},
    )
    matched_names = {h.name for h in hits}
    assert matched_names == {"新宿駅", "강남역", "Булевард Витоша", "Łódź Kaliska"}


def test_is_geocodable_text_accepts_non_latin_scripts_and_rejects_noise():
    assert text_anchor.is_geocodable_text("新宿駅")
    assert text_anchor.is_geocodable_text("강남역")
    assert text_anchor.is_geocodable_text("Београд")
    assert text_anchor.is_geocodable_text("صيدلية النهدي")
    assert text_anchor.is_geocodable_text("Łódź")
    assert not text_anchor.is_geocodable_text("12345")
    assert not text_anchor.is_geocodable_text("10:45")
    assert not text_anchor.is_geocodable_text("A")


def test_city_extent_radius_probes_wider_nominatim_ranks_for_tiny_polygons(monkeypatch, tmp_path):
    import osmnx as ox

    class _FakeGDF:
        def __init__(self, bounds):
            self.total_bounds = bounds

    def fake_geocode_to_gdf(query, which_result=1):
        if which_result == 1:
            # Tiny 1.2 km inner ward (the Glasgow rank-1 polygon trap)
            return _FakeGDF((-4.260, 55.855, -4.240, 55.865))
        if which_result == 2:
            # True Glasgow council boundary (~11 km half-diagonal)
            return _FakeGDF((-4.390, 55.780, -4.080, 55.930))
        raise ValueError("no more results")

    monkeypatch.setattr(ox, "geocode_to_gdf", fake_geocode_to_gdf)
    ext = text_anchor.city_extent_radius("Glasgow, UK", tmp_path / "geo.json")
    assert ext is not None
    assert ext[2] > 9000.0


def test_location_prior_filters_camera_gear_and_keeps_multilingual_places():
    cands = location_prior.extract_place_candidates(
        "Driving Munich 4K | VIOFO Dash Camera 1440p | Marienplatz to Schwabing",
        None,
    )
    lowered = {c.lower() for c in cands}
    assert "camera" not in lowered
    assert "viofo" not in lowered
    assert "marienplatz" in lowered
    assert "schwabing" in lowered

    cyr = location_prior.extract_place_candidates(
        "Вожња Београд: Нови Београд - Земун",
        None,
    )
    assert any("Земун" in c or "Нови Београд" in c for c in cyr)


def test_city_inference_covers_global_countries():
    assert city_inference.guess_city_from_title("4K Night Drive - Tokyo, Japan") == "Tokyo, Japan"
    assert city_inference.guess_city_from_title("Driving in São Paulo, Brazil") == "São Paulo, Brazil"
    assert city_inference.guess_city_from_title("Morning traffic in Nairobi, Kenya") == "Nairobi, Kenya"
    assert city_inference.guess_city_from_title("Dashcam Seoul, South Korea") == "Seoul, South Korea"


def test_benchmark_fairness_accepts_exonyms_and_scripts_but_rejects_hidden_suburbs():
    entry_ro = {"id": "ro1", "city": "Bucharest, Romania", "video": {"url": "https://youtu.be/x"}}
    assert bm.city_fairness_problem(entry_ro, "Drumetie auto Buzau - Bucuresti DN2", "") is None

    entry_jp = {"id": "jp1", "city": "Tokyo, Japan", "video": {"url": "https://youtu.be/y"}}
    assert bm.city_fairness_problem(entry_jp, "【車載動画】夜の東京ドライブ 4K", "") is None

    entry_rs = {"id": "rs1", "city": "Belgrade, Serbia", "video": {"url": "https://youtu.be/z"}}
    assert bm.city_fairness_problem(entry_rs, "Вожња кроз Београд 2026", "") is None

    entry_de = {"id": "de1", "city": "Cologne, Germany", "video": {"url": "https://youtu.be/k"}}
    assert bm.city_fairness_problem(entry_de, "Dashcam Fahrt durch Köln Innenstadt", "") is None

    unfair = {"id": "bad1", "city": "Daly City, California, USA", "video": {"url": "https://youtu.be/u"}}
    problem = bm.city_fairness_problem(unfair, "Driving around the Bay Area in 4K", "")
    assert problem is not None and "not named" in problem


# ---------------------------------------------------------------------------
# 2. Tiled Viterbi, spatial micro-cell dedup & streamed warm-cache hits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_tiled_viterbi_matches_dense_and_numpy_exact(seed):
    rng = np.random.default_rng(seed)
    n_q, n_r = 20, 310
    latlon = np.column_stack([
        48.4 + rng.uniform(-0.08, 0.08, n_r),
        9.98 + rng.uniform(-0.10, 0.10, n_r),
    ])
    sims = rng.uniform(0.05, 0.35, (n_q, n_r)).astype(np.float32)
    for q in range(n_q):
        sims[q, (q * 11 + 3) % n_r] += 0.35

    ref_np = kv._viterbi_decode(sims, latlon, 4.0)
    dense_torch = kv._viterbi_decode_torch(sims, latlon, 4.0, device="cpu", block_size=2048)
    # Force tiled branch (n_r > block_size * 4 when block_size=64 -> 310 > 256)
    tiled_torch = kv._viterbi_decode_torch(sims, latlon, 4.0, device="cpu", block_size=64)

    assert np.array_equal(dense_torch, ref_np)
    assert np.array_equal(tiled_torch, ref_np)


def test_spatial_dedup_eliminates_traverse_stacking_deterministically():
    listed = {}
    # 40 stacked downtown photos within a 5 m spot
    for k in range(40):
        listed[f"downtown_{k}"] = [48.13510 + k * 1e-7, 11.58200 + k * 1e-7]
    # 10 suburban single-traverse photos spaced ~200 m apart
    for s in range(10):
        listed[f"suburb_{s}"] = [48.2000 + s * 0.002, 11.6500]

    dedup1 = kv._spatial_dedup_dict(listed, 48.15, cell_m=25.0)
    dedup2 = kv._spatial_dedup_dict(dict(reversed(list(listed.items()))), 48.15, cell_m=25.0)

    assert dedup1 == dedup2
    suburb_kept = [k for k in dedup1 if k.startswith("suburb_")]
    downtown_kept = [k for k in dedup1 if k.startswith("downtown_")]
    assert len(suburb_kept) == 10
    assert len(downtown_kept) == 1


def test_mly_tile_listing_handles_empty_water_tiles_with_spatial_dedup(monkeypatch):
    from monocular_osm import mvt

    class _FakeResp:
        def __init__(self, content):
            self.status_code = 200
            self.content = content

    class _FakeSess:
        def get(self, url, params=None, timeout=60):
            # Alternate empty tiles (water/forest) and populated tiles
            y = int(url.rsplit("/", 1)[1])
            return _FakeResp(b"empty" if y % 2 == 0 else b"full")

    def fake_decode(content, layer, z, x, y):
        if content == b"empty":
            return [], np.zeros((0, 2))
        return ["id_a", "id_b"], np.array([[52.5200, 13.4050], [52.52001, 13.40501]])

    monkeypatch.setattr(mvt, "decode_points", fake_decode)
    stats = {"tiles": 0, "lost": 0}
    got = kv._mly_tile_listing(
        (52.5200, 13.4050), 2000.0, "tok", _FakeSess(),
        lambda k, n=1: None, stats, spatial_dedup_m=25.0,
    )
    assert stats["tiles"] >= 1
    assert len(got) == 1  # two stacked points in same 25m cell -> 1 kept, empty tiles ignored cleanly


def test_mapillary_fetch_serves_warm_streamed_descriptor_cache_offline(tmp_path, monkeypatch):
    center = (48.4011, 9.9876)
    sig = kv._fetch_signature(center, 20000.0, 24000, spatial_dedup_m=25.0)
    refs = [{"id": str(5000 + i), "lat": 48.40 + i * 1e-4, "lon": 9.98} for i in range(35)]
    fp = kv._refs_fingerprint(refs)
    snapshot = kv._reference_snapshot(refs, created_utc="2026-11-18T00:00:00Z")

    meta_path = tmp_path / f"mly_ref_meta_{kv._sig_tag(sig)}.json"
    meta_path.write_text(json.dumps({
        "signature": sig,
        "refs": refs,
        "listing_stats": {"listing": "tiles", "tiles": 16, "split": 0, "lost": 0},
        "snapshot": snapshot,
    }), encoding="utf-8")

    emb = np.ones((len(refs), 4), dtype=np.float32)
    xy = np.array([[r["lat"], r["lon"]] for r in refs], dtype=np.float64)
    np.savez(
        tmp_path / f"ref_emb_stream_megaloc_{fp[:10]}.npz",
        emb=emb,
        xy=xy,
        ids=np.array([r["id"] for r in refs]),
        request_fingerprint=np.array(fp),
    )

    monkeypatch.delenv("MLY_TOKEN", raising=False)
    assert kv.has_mapillary_cache(str(tmp_path))
    got = kv._fetch_refs_mapillary(center, 20000.0, str(tmp_path), cap=24000,
                                   token=None, spatial_dedup_m=25.0)
    assert len(got) == 35
    assert kv.LAST_FETCH_STATS.get("cache_hit") is True


# ---------------------------------------------------------------------------
# 3. Metro-Dense VPR, Corridor Pass 2, Additive Seeds & Stage 0 Inference
# ---------------------------------------------------------------------------


def test_plan_corridor_discs_adapts_to_short_vs_highway_routes():
    # Compact 600 m urban route -> 1 disc
    short_ll = np.column_stack([
        np.linspace(48.400, 48.405, 40),
        np.linspace(9.980, 9.985, 40),
    ])
    short_sims = np.full(40, 0.35)
    c_short = metro_vpr.plan_corridor_discs(short_ll, short_sims, radius_m=2500.0, max_discs=4)
    assert len(c_short) == 1

    # 9 km highway drive (~0.08 deg lat) + 1 wild single-frame outlier 18 km away
    hwy_ll = np.column_stack([
        np.linspace(48.40, 48.48, 80),
        np.full(80, 9.98),
    ])
    hwy_ll[37] = [48.62, 10.20]  # isolated single-frame outlier must be ignored
    hwy_sims = np.full(80, 0.35)
    c_hwy = metro_vpr.plan_corridor_discs(hwy_ll, hwy_sims, radius_m=2500.0, max_discs=4)
    assert 2 <= len(c_hwy) <= 6
    # No disc should be placed on the isolated outlier at 48.62
    assert all(c[0] < 48.52 for c in c_hwy)

    # 18 km inter-city highway drive (e.g. Howwood -> Glasgow): span-adaptive expansion > 4 discs
    inter_ll = np.column_stack([
        np.linspace(55.81, 55.97, 80),
        np.full(80, -4.40),
    ])
    c_inter = metro_vpr.plan_corridor_discs(inter_ll, np.full(80, 0.30), radius_m=2500.0, max_discs=4)
    assert 5 <= len(c_inter) <= 8


def test_metro_dense_recovers_suburban_route_despite_bad_3km_seed_and_falls_back(tmp_path, monkeypatch):
    """Simulate a suburban drive 13 km north of the city centre while a noisy
    coarse seed points 4 km south (the Melbourne / Belgrade seed-clamping trap),
    AND verify open-provider fallback when Mapillary returns 0 refs."""
    city_center = (48.0000, 11.0000)
    bad_seed = (47.9650, 11.0000, 2500.0)   # 3.9 km south (wrong district)
    true_route_lats = np.linspace(48.1150, 48.1240, 12)  # ~13 km north!
    true_route_lons = np.full(12, 11.0000)

    monkeypatch.setattr(kv, "_resolve_backbone", lambda m, d: "fake")
    monkeypatch.setattr(kv, "_MODEL_NAME", "megaloc")
    monkeypatch.setattr(kv, "_MODEL_DEVICE", "cpu")

    def fake_fetch_for(src, center, radius_m, cache_dir, cap=1500, token=None, *, spatial_dedup_m=None):
        if src == "mapillary":
            return []  # simulate thin/empty Mapillary region -> triggers open fallback!
        if src == "panoramax":
            # Generate grid of refs inside (center, radius_m), including the true suburban route
            refs = []
            rng = np.random.default_rng(42)
            for i in range(45):
                la = center[0] + rng.uniform(-0.02, 0.02)
                lo = center[1] + rng.uniform(-0.02, 0.02)
                refs.append({"id": f"bg_{i}_{round(radius_m)}", "lat": la, "lon": lo, "url": f"https://p/{i}"})
            # If the disc covers the true suburban route (~13 km from city_center), include route refs!
            dist_to_route = abs(center[0] - 48.119) * 111320.0
            if dist_to_route <= radius_m:
                for k, (la, lo) in enumerate(zip(true_route_lats, true_route_lons)):
                    refs.append({"id": f"route_{k}", "lat": float(la), "lon": float(lo), "url": f"https://r/{k}"})
            return refs
        return []

    def fake_embed_streaming(refs, device, cache_dir, model_name="model", *, batch_size=24, chunk=1024):
        vecs = []
        xy = []
        for r in refs:
            rid = str(r["id"])
            if rid.startswith("route_"):
                k = int(rid.split("_")[1])
                ang = (k / 12.0) * (np.pi / 2.0)
                v = np.array([np.cos(ang), np.sin(ang), 1.0, 0.0], dtype=np.float32)
            else:
                v = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v)
            xy.append([float(r["lat"]), float(r["lon"])])
        return torch.from_numpy(np.stack(vecs)), np.asarray(xy, dtype=np.float64)

    def fake_query_sims(frames_bgr, q_idx, ref_emb, device, **kw):
        q_vecs = []
        for qi in range(len(q_idx)):
            ang = (qi / max(len(q_idx), 1)) * (np.pi / 2.0)
            v = np.array([np.cos(ang), np.sin(ang), 1.0, 0.0], dtype=np.float32)
            v /= np.linalg.norm(v)
            q_vecs.append(v)
        q_mat = torch.from_numpy(np.stack(q_vecs))
        return (q_mat @ ref_emb.T).numpy()

    monkeypatch.setattr(kv, "_fetch_refs_for", fake_fetch_for)
    monkeypatch.setattr(kv, "_embed_refs_streaming", fake_embed_streaming)
    monkeypatch.setattr(kv, "_query_similarities", fake_query_sims)

    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(12)]
    ts = np.arange(12, dtype=np.float64) * 4.0

    cfg = PipelineConfig(
        url="https://youtu.be/suburb",
        city="TestMetro, Country",
        data_dir=tmp_path / "data" / "clip1",
        output_dir=tmp_path / "out" / "clip1",
        vpr_metro_dense=True,
        vpr_region_cache_dir=tmp_path / "ext_cache",
        retrieval_only=True,
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "geocode_cache.json").write_text(
        json.dumps({"TestMetro, Country": [city_center[0], city_center[1]]}),
        encoding="utf-8",
    )

    class _FramesWrap:
        def __init__(self, f, t):
            self.frames = f
            self.timestamps = t

    ctx = RunContext.create(cfg)
    ctx.frames = _FramesWrap(frames, ts)
    ctx.estimated_length_m = 1200.0
    ctx.osm_around = bad_seed  # noisy 2.5 km seed 4 km south of city centre!
    monkeypatch.setenv("MLY_TOKEN", "dummy_test_token")

    vpr_coarse_prior(ctx)

    assert ctx.vpr_track is not None
    assert ctx.vpr_center is not None
    assert ctx.vpr_source_used == "union"  # Mapillary augmented by Panoramax
    # Located right on the suburban route at 48.119, NOT trapped at bad_seed (47.965)!
    assert abs(ctx.vpr_center[0] - 48.119) < 0.01
    diag = ctx.result.get("metro_dense_retrieval")
    assert diag is not None and diag["status"] == "ok"


def test_stage0_global_geoloc_and_cli_validation(monkeypatch, tmp_path):
    # Without --infer-city-from-video, --video still requires --city
    with pytest.raises(ValueError, match="--city is required"):
        _validate_input_args([Path("a.mp4")], None, None)

    # With infer_city=True, --video without --city is accepted
    _validate_input_args([Path("a.mp4")], None, None, infer_city=True)

    parser = build_arg_parser()
    args = parser.parse_args([
        "--video", "a.mp4",
        "--infer-city-from-video",
        "--vpr-metro-dense",
        "--vpr-metro-cap", "24000",
    ])
    assert args.infer_city_from_video is True
    assert args.vpr_metro_dense is True
    assert args.vpr_metro_cap == 24000

    # Title fast-path in infer_city_worldwide
    g_title = global_geoloc.infer_city_worldwide(title="Evening drive through Osaka, Japan")
    assert g_title is not None and g_title.city == "Osaka, Japan"

    # Visual fallback path in infer_city_worldwide
    fake_guess = global_geoloc.GlobalCityGuess(
        city="Singapore", lat=1.3521, lon=103.8198, confidence=0.82,
        method="zero_shot_clip:mock", top_candidates=[{"city": "Singapore", "probability": 0.82}],
    )
    monkeypatch.setattr(global_geoloc, "_clip_rank_cities", lambda imgs, **kw: fake_guess)
    dummy_frames = [np.zeros((16, 16, 3), dtype=np.uint8)]
    g_vis = global_geoloc.infer_city_worldwide(frames_bgr=dummy_frames, title=None)
    assert g_vis is not None and g_vis.city == "Singapore"


def test_low_cosine_night_drive_recovers_corroborated_seed(tmp_path, monkeypatch):
    """When pitch-black night frames produce weak cosines (<0.155) everywhere in
    the 20 km metro and drift 40 km away from a corroborated OCR seed, Step 3
    recovers the dense seed corridor."""
    city_center = (-37.814, 144.963)
    good_ocr_seed = (-38.069, 145.313, 3500.0)  # 42 km SE (e.g. Berwick, Melbourne)

    monkeypatch.setattr(kv, "_resolve_backbone", lambda m, d: "fake")
    monkeypatch.setattr(kv, "_MODEL_NAME", "megaloc")
    monkeypatch.setattr(kv, "_MODEL_DEVICE", "cpu")

    def fake_fetch_for(src, center, radius_m, cache_dir, cap=1500, token=None, **kw):
        if src != "mapillary":
            return []
        near_seed = abs(center[0] - good_ocr_seed[0]) < 0.05
        base_lat, base_lon = (good_ocr_seed[0], good_ocr_seed[1]) if near_seed else city_center
        tag = "seed" if near_seed else "city"
        return [
            {"id": f"{tag}_{i}", "lat": base_lat + i * 1e-4, "lon": base_lon, "url": f"https://u/{tag}_{i}"}
            for i in range(40)
        ]

    def fake_embed(refs, device, cache_dir, model_name="model", **kw):
        vecs, xy = [], []
        for r in refs:
            # City-centre daytime distractor scores 0.132 against dark query; seed scores 0.125
            # unless dense_seed (cap=6000) is queried where a streetlamp ref scores 0.130.
            is_seed = str(r["id"]).startswith("seed_")
            c_val = 0.128 if is_seed else 0.132
            s_val = float(np.sqrt(1.0 - c_val ** 2))
            vecs.append([c_val, s_val, 0.0, 0.0])
            xy.append([float(r["lat"]), float(r["lon"])])
        return torch.tensor(vecs, dtype=torch.float32), np.asarray(xy, dtype=np.float64)

    def fake_qsims(frames_bgr, q_idx, ref_emb, device, **kw):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(q_idx), dtype=torch.float32)
        return (q @ ref_emb.T).numpy()

    monkeypatch.setattr(kv, "_fetch_refs_for", fake_fetch_for)
    monkeypatch.setattr(kv, "_embed_refs_streaming", fake_embed)
    monkeypatch.setattr(kv, "_query_similarities", fake_qsims)

    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(10)]
    ts = np.arange(10, dtype=np.float64) * 4.0
    tr, diag = metro_vpr.run_metro_dense_vpr(
        frames, ts,
        city="Melbourne, Australia",
        city_center=city_center,
        seed_around=good_ocr_seed,
        data_dir=tmp_path / "d",
        shared_cache_root=tmp_path / "shared",
        token="tok",
        device="cpu",
    )
    assert tr is not None
    assert diag.get("seed_recovery", {}).get("accepted") is True
    assert abs(float(np.median(tr[1][:, 0])) - good_ocr_seed[0]) < 0.02

