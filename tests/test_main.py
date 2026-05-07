from __future__ import annotations

import pytest

from main import DEFAULT_CITY, DEFAULT_URL, _resolve_city


def test_resolve_city_prefers_explicit_city() -> None:
    assert _resolve_city("Paris, France", "https://example.com", "Driving in Ulm, Germany") == "Paris, France"


def test_resolve_city_uses_title_guess() -> None:
    assert _resolve_city(None, "https://example.com", "Driving in Ulm, Germany") == "Ulm, Germany"


def test_resolve_city_falls_back_for_default_demo_video() -> None:
    assert _resolve_city(None, DEFAULT_URL, None) == DEFAULT_CITY


def test_resolve_city_requires_explicit_city_when_guess_fails() -> None:
    with pytest.raises(ValueError, match="Could not infer a city"):
        _resolve_city(None, "https://example.com", "Dashcam compilation")
