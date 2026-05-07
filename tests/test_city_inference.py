from __future__ import annotations

from src.city_inference import guess_city_from_title, slugify_submission


def test_guess_city_from_title_with_country() -> None:
    assert guess_city_from_title("Driving in Ulm, Germany - 4K Dashcam") == "Ulm, Germany"


def test_guess_city_from_title_without_country() -> None:
    assert guess_city_from_title("Tokyo night drive POV") == "Tokyo"


def test_guess_city_from_title_returns_none_when_not_confident() -> None:
    assert guess_city_from_title("Best dashcam compilation 2026") is None


def test_slugify_submission_prefers_human_readable_parts() -> None:
    assert slugify_submission("ULl8s4qydrk", "Driving in Ulm, Germany", "Ulm, Germany", fallback_seed="x") == (
        "ull8s4qydrk-driving-in-ulm-germany-ulm-germany"
    )


def test_slugify_submission_falls_back_to_hash() -> None:
    assert slugify_submission(None, None, None, fallback_seed="https://example.com") == "video-327c3fda87ce"
