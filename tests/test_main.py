from __future__ import annotations

import sys
from pathlib import Path

import pytest

import main as main_mod
from main import (
    DEFAULT_CITY,
    DEFAULT_URL,
    _parse_segment,
    _resolve_city,
    _validate_input_args,
    build_arg_parser,
)


def test_resolve_city_prefers_explicit_city() -> None:
    assert _resolve_city("Paris, France", "https://example.com", "Driving in Ulm, Germany") == "Paris, France"


def test_resolve_city_uses_title_guess() -> None:
    assert _resolve_city(None, "https://example.com", "Driving in Ulm, Germany") == "Ulm, Germany"


def test_resolve_city_falls_back_for_default_demo_video() -> None:
    assert _resolve_city(None, DEFAULT_URL, None) == DEFAULT_CITY


def test_resolve_city_requires_explicit_city_when_guess_fails() -> None:
    with pytest.raises(ValueError, match="Could not infer a city"):
        _resolve_city(None, "https://example.com", "Dashcam compilation")


def test_arg_parser_accepts_comparison_options() -> None:
    args = build_arg_parser().parse_args(
        [
            "--enable-sliding-window",
            "--sliding-window-size", "80",
            "--sliding-window-step", "20",
            "--embedding-sources", "osm", "geotessera",
            "--embedding-model", "resnet18",
            "--geotessera-year", "2025",
        ]
    )
    assert args.enable_sliding_window is True
    assert args.sliding_window_size == 80
    assert args.sliding_window_step == 20
    assert args.embedding_sources == ["osm", "geotessera"]
    assert args.embedding_model == "resnet18"
    assert args.geotessera_year == 2025


def test_arg_parser_accepts_bev_splat_options() -> None:
    args = build_arg_parser().parse_args(
        [
            "--enable-bev-splat",
            "--bev-splat-weights", "checkpoints/bevsplat_kitti.pth",
            "--bev-splat-repo-path", "third_party/BevSplat",
            "--bev-splat-model-module", "models.models_kitti_vfa",
            "--bev-splat-source", "osm",
            "--bev-splat-tile-size", "384",
            "--bev-splat-half-extent-m", "80.0",
        ]
    )
    assert args.enable_bev_splat is True
    assert args.bev_splat_weights == Path("checkpoints/bevsplat_kitti.pth")
    assert args.bev_splat_repo_path == Path("third_party/BevSplat")
    assert args.bev_splat_model_module == "models.models_kitti_vfa"
    assert args.bev_splat_source == "osm"
    assert args.bev_splat_tile_size == 384
    assert args.bev_splat_half_extent_m == 80.0


def test_arg_parser_bev_splat_defaults_are_safe() -> None:
    """Defaults must keep BevSplat disabled so a vanilla `python main.py` doesn't try
    to load weights or import the upstream repo."""
    args = build_arg_parser().parse_args([])
    assert args.enable_bev_splat is False
    assert args.bev_splat_weights is None
    assert args.bev_splat_repo_path is None
    assert args.bev_splat_model_module == "models.models_kitti_nips"
    assert args.bev_splat_source == "esri"
    assert args.bev_splat_tile_size == 512
    assert args.bev_splat_half_extent_m == 60.0


# ---------------------------------------------------------------------------
# _parse_segment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0:420", (0.0, 420.0)),
        ("30:300", (30.0, 300.0)),
        ("30:", (30.0, None)),
        (":120", (0.0, 120.0)),
        ("45", (45.0, None)),
        (":", (0.0, None)),
        ("1.5:2.5", (1.5, 2.5)),
    ],
)
def test_parse_segment(text: str, expected: tuple[float, float | None]) -> None:
    assert _parse_segment(text) == expected


# ---------------------------------------------------------------------------
# --video argument + input validation
# ---------------------------------------------------------------------------


def test_arg_parser_accepts_local_videos() -> None:
    args = build_arg_parser().parse_args(
        ["--video", "a.mp4", "b.mp4", "--city", "Ulm, Germany"]
    )
    assert args.video == [Path("a.mp4"), Path("b.mp4")]
    assert args.url is None
    assert args.city == "Ulm, Germany"


def test_arg_parser_defaults_have_no_inputs() -> None:
    """The DEFAULT_URL fallback lives in main(), not the parser, so the
    parser itself must report 'nothing given'."""
    args = build_arg_parser().parse_args([])
    assert args.video is None
    assert args.url is None


def test_arg_parser_ground_truth_waypoints() -> None:
    args = build_arg_parser().parse_args(
        ["--ground-truth-waypoints", "ground_truth/ulm.json"]
    )
    assert args.ground_truth_waypoints == Path("ground_truth/ulm.json")
    assert build_arg_parser().parse_args([]).ground_truth_waypoints is None


def test_arg_parser_ocr_anchor_flags() -> None:
    args = build_arg_parser().parse_args(
        ["--enable-ocr-anchor", "--ocr-sample-interval-sec", "10",
         "--ocr-min-confidence", "0.6"]
    )
    assert args.enable_ocr_anchor is True
    assert args.ocr_sample_interval_sec == 10.0
    assert args.ocr_min_confidence == 0.6
    # Safe defaults: OCR anchor off so a vanilla run needs no easyocr/network.
    defaults = build_arg_parser().parse_args([])
    assert defaults.enable_ocr_anchor is False
    assert defaults.ocr_sample_interval_sec == 6.0


def test_arg_parser_estimated_length_defaults_to_auto() -> None:
    args = build_arg_parser().parse_args([])
    assert args.estimated_length_m is None
    args = build_arg_parser().parse_args(["--estimated-length-m", "2600"])
    assert args.estimated_length_m == 2600.0


def test_validate_rejects_video_and_url_together() -> None:
    with pytest.raises(ValueError, match="not both"):
        _validate_input_args([Path("a.mp4")], ["https://example.com"], "Ulm, Germany")


def test_validate_requires_city_for_local_video() -> None:
    with pytest.raises(ValueError, match="--city is required"):
        _validate_input_args([Path("a.mp4")], None, None)


@pytest.mark.parametrize(
    ("videos", "urls", "city"),
    [
        ([Path("a.mp4")], None, "Ulm, Germany"),   # video + city
        (None, ["https://example.com"], None),      # url, city inferred later
        (None, None, None),                         # nothing → default demo clip
        (None, None, "Ulm, Germany"),               # city only → default clip
    ],
)
def test_validate_accepts_valid_combinations(
    videos: list[Path] | None, urls: list[str] | None, city: str | None
) -> None:
    _validate_input_args(videos, urls, city)  # must not raise


# ---------------------------------------------------------------------------
# main() wiring for local videos
# ---------------------------------------------------------------------------


def _forbid_network_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(url: str) -> None:
        raise AssertionError(f"fetch_video_metadata must not be called (got {url!r})")

    monkeypatch.setattr(main_mod, "fetch_video_metadata", _boom)


def test_main_wires_local_video_into_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "ulm drive.mp4"
    video.write_bytes(b"\x00")
    captured: dict = {}

    def fake_run_pipeline(cfg):  # noqa: ANN001 - matches PipelineConfig
        captured["cfg"] = cfg
        return {"city": cfg.city, "matches": []}

    monkeypatch.setattr(main_mod, "run_pipeline", fake_run_pipeline)
    _forbid_network_metadata(monkeypatch)
    monkeypatch.setattr(
        sys, "argv",
        [
            "main.py",
            "--video", str(video),
            "--city", "Ulm, Germany",
            "--data-dir", str(tmp_path / "data"),
            "--output-dir", str(tmp_path / "out"),
        ],
    )

    main_mod.main()

    cfg = captured["cfg"]
    assert cfg.video_path == video
    assert cfg.city == "Ulm, Germany"
    assert cfg.url.startswith("file://")
    # Slug folders must be derived from the local metadata (stem + city).
    assert "ulm" in str(cfg.data_dir).lower()


def test_main_rejects_video_plus_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "--video", "a.mp4", "--url", "https://example.com",
         "--city", "Ulm, Germany"],
    )
    with pytest.raises(SystemExit):
        main_mod.main()


def test_main_rejects_video_without_city(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", "--video", "a.mp4"])
    with pytest.raises(SystemExit):
        main_mod.main()


def test_main_missing_local_video_reports_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonexistent --video path must fail with a SystemExit naming the
    file, not crash inside the pipeline."""
    monkeypatch.setattr(
        main_mod, "run_pipeline",
        lambda cfg: pytest.fail("pipeline must not run for a missing file"),
    )
    _forbid_network_metadata(monkeypatch)
    missing = tmp_path / "missing.mp4"
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "--video", str(missing), "--city", "Ulm, Germany"],
    )
    with pytest.raises(SystemExit, match="missing.mp4"):
        main_mod.main()


# ---------------------------------------------------------------------------
# --analyze-minutes + auto frame stride
# ---------------------------------------------------------------------------


def test_arg_parser_analyze_minutes() -> None:
    args = build_arg_parser().parse_args(["--analyze-minutes", "10"])
    assert args.analyze_minutes == 10.0
    assert build_arg_parser().parse_args([]).analyze_minutes is None


@pytest.mark.parametrize(
    ("duration_sec", "expected_stride"),
    [
        (420.0, 3),    # 7 min — the historical default
        (600.0, 4),    # 10 min
        (900.0, 6),    # 15 min
        (60.0, 3),     # short clips never go below 3
        (None, 3),     # open-ended segment
    ],
)
def test_auto_frame_stride(duration_sec: float | None, expected_stride: int) -> None:
    from main import _auto_frame_stride

    assert _auto_frame_stride(duration_sec) == expected_stride


def test_main_analyze_minutes_sets_segment_and_stride(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    captured: dict = {}

    def fake_run_pipeline(cfg):  # noqa: ANN001
        captured["cfg"] = cfg
        return {"city": cfg.city, "matches": []}

    monkeypatch.setattr(main_mod, "run_pipeline", fake_run_pipeline)
    _forbid_network_metadata(monkeypatch)
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "--video", str(video), "--city", "Ulm, Germany",
         "--analyze-minutes", "10",
         "--data-dir", str(tmp_path / "d"), "--output-dir", str(tmp_path / "o")],
    )
    main_mod.main()
    cfg = captured["cfg"]
    assert cfg.vo_start_sec == 0.0
    assert cfg.vo_end_sec == 600.0
    assert cfg.frame_stride == 4      # auto for 10 min
    assert cfg.max_frames is None     # uncapped


def test_main_explicit_stride_overrides_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    captured: dict = {}
    monkeypatch.setattr(
        main_mod, "run_pipeline",
        lambda cfg: captured.update(cfg=cfg) or {"city": cfg.city, "matches": []},
    )
    _forbid_network_metadata(monkeypatch)
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "--video", str(video), "--city", "Ulm, Germany",
         "--analyze-minutes", "15", "--frame-stride", "2",
         "--data-dir", str(tmp_path / "d"), "--output-dir", str(tmp_path / "o")],
    )
    main_mod.main()
    assert captured["cfg"].frame_stride == 2


def test_auto_frame_stride_uses_real_fps() -> None:
    """A 60 fps dashcam upload must not get double the intended frame
    budget: 7 min @60fps needs stride 6, not the nominal-30 answer 3."""
    from main import _auto_frame_stride

    assert _auto_frame_stride(420.0, fps=60.0) == 6
    assert _auto_frame_stride(420.0, fps=30.0) == 3
    # Unknown/broken fps falls back to the nominal 30.
    assert _auto_frame_stride(420.0, fps=None) == 3
    assert _auto_frame_stride(420.0, fps=0.0) == 3


# ---------------------------------------------------------------------------
# Refuted gating flags are gone (--vpr-gate-radius / --plate-gate-radius)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--vpr-gate-radius", "500"],
        ["--plate-gate-radius", "3000"],
    ],
)
def test_dead_gate_flags_removed(argv: list[str]) -> None:
    """The gate radii were parsed but never read (gating was refuted by
    experiment); they must now be rejected instead of silently no-oping."""
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(argv)


def test_use_vpr_sequence_flag_wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert build_arg_parser().parse_args([]).use_vpr_sequence is False
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    captured: dict = {}
    monkeypatch.setattr(
        main_mod, "run_pipeline",
        lambda cfg: captured.update(cfg=cfg) or {"city": cfg.city, "matches": []},
    )
    _forbid_network_metadata(monkeypatch)
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "--video", str(video), "--city", "Ulm, Germany",
         "--use-vpr-prior", "--use-vpr-sequence",
         "--data-dir", str(tmp_path / "d"), "--output-dir", str(tmp_path / "o")],
    )
    main_mod.main()
    assert captured["cfg"].use_vpr_prior is True
    assert captured["cfg"].use_vpr_sequence is True


# ---------------------------------------------------------------------------
# --skip-download offline re-runs (cached metadata; no network fetch)
# ---------------------------------------------------------------------------


def test_skip_download_uses_cached_metadata_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline re-run of a fully cached clip: --skip-download must not
    require a yt-dlp network metadata fetch (it used to die in
    fetch_video_metadata before ever consulting the cache)."""
    from src.download import VideoMetadata

    url = "https://www.youtube.com/watch?v=abc123def45"
    data_dir = tmp_path / "data"
    metadata = VideoMetadata(url=url, title="Driving in Ulm, Germany",
                             video_id="abc123def45")
    main_mod._write_cached_metadata(data_dir, url, metadata)

    captured: dict = {}
    monkeypatch.setattr(
        main_mod, "run_pipeline",
        lambda cfg: captured.update(cfg=cfg) or {"city": cfg.city, "matches": []},
    )
    _forbid_network_metadata(monkeypatch)   # any fetch -> test failure
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "--url", url, "--skip-download",
         "--data-dir", str(data_dir), "--output-dir", str(tmp_path / "o")],
    )
    main_mod.main()
    cfg = captured["cfg"]
    assert cfg.skip_download is True
    assert cfg.url == url
    assert cfg.city == "Ulm, Germany"       # inferred from the cached title


def test_metadata_cache_roundtrip(tmp_path: Path) -> None:
    from src.download import VideoMetadata

    url = "https://example.com/watch?v=xyz"
    meta = VideoMetadata(url=url, title="t", video_id="xyz")
    assert main_mod._load_cached_metadata(tmp_path, url) is None
    main_mod._write_cached_metadata(tmp_path, url, meta)
    loaded = main_mod._load_cached_metadata(tmp_path, url)
    assert loaded == meta
    # A different URL must not hit this cache entry.
    assert main_mod._load_cached_metadata(tmp_path, url + "2") is None


def test_metadata_fetch_failure_does_not_suppress_batch_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bad/offline URL in a batch must not suppress the summary of
    the runs that succeeded (the SystemExit used to fire before the
    batch_results.json write)."""
    import json as _json

    from src.download import DownloadError, VideoMetadata

    good = ["https://example.com/a", "https://example.com/b"]
    bad = "https://example.com/broken"

    def fake_fetch(url: str) -> VideoMetadata:
        if url == bad:
            raise DownloadError("network unreachable")
        return VideoMetadata(url=url, title=f"Driving in Ulm, Germany {url[-1]}",
                             video_id=url[-1])

    monkeypatch.setattr(main_mod, "fetch_video_metadata", fake_fetch)
    monkeypatch.setattr(
        main_mod, "run_pipeline",
        lambda cfg: {"city": cfg.city, "matches": []},
    )
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "--url", good[0], good[1], bad, "--city", "Ulm, Germany",
         "--data-dir", str(tmp_path / "d"), "--output-dir", str(out_dir)],
    )
    with pytest.raises(SystemExit, match="broken"):
        main_mod.main()
    batch = out_dir / "batch_results.json"
    assert batch.exists()
    payload = _json.loads(batch.read_text(encoding="utf-8"))
    assert len(payload["results"]) == 2
