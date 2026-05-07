from __future__ import annotations

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
