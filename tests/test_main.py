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
