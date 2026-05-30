from __future__ import annotations

from pathlib import Path

import pytest

from main import DEFAULT_CITY, DEFAULT_URL, _resolve_city, build_arg_parser


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
