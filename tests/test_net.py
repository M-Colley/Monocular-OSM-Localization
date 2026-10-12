"""Retries cover transient network failures and nothing else."""

from __future__ import annotations

import pytest
import requests

from monocular_osm import net
from monocular_osm.net import is_transient, transient_retry


@pytest.mark.parametrize("exc,expected", [
    (requests.exceptions.ConnectTimeout("connect timeout=180"), True),
    (requests.exceptions.ConnectionError("reset by peer"), True),
    (TimeoutError("timed out"), True),
    (RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden"), True),
    (RuntimeError("HTTP Error 429: Too Many Requests"), True),
    (ValueError("Nominatim could not geocode Qwertyville"), False),
    (RuntimeError("Private video. Sign in if you've been granted access"), False),
    (KeyError("lat"), False),
])
def test_is_transient(exc, expected) -> None:
    assert is_transient(exc) is expected


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)


def test_retries_transient_then_succeeds(capsys) -> None:
    calls = []

    @transient_retry(attempts=4, first_wait_s=1.0)
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ConnectTimeout("connect timeout=180")
        return "graph"

    assert flaky() == "graph" and len(calls) == 3
    out = capsys.readouterr().out
    assert out.count("retrying in") == 2 and "flaky: attempt 1 failed" in out


def test_does_not_retry_permanent_errors() -> None:
    calls = []

    @transient_retry(attempts=4, first_wait_s=1.0)
    def bad_city():
        calls.append(1)
        raise ValueError("Nominatim could not geocode Qwertyville")

    with pytest.raises(ValueError):
        bad_city()
    assert len(calls) == 1


def test_reraises_the_last_error_when_exhausted() -> None:
    calls = []

    @transient_retry(attempts=3, first_wait_s=1.0)
    def always_down():
        calls.append(1)
        raise requests.exceptions.ConnectTimeout("connect timeout=180")

    with pytest.raises(requests.exceptions.ConnectTimeout):
        always_down()
    assert len(calls) == 3


def test_overpass_download_is_retried_on_connect_timeout(monkeypatch, capsys) -> None:
    import monocular_osm.osm_data as osm_data

    attempts = []

    def flaky_graph_from_point(center, dist, network_type):
        attempts.append(1)
        if len(attempts) < 3:
            raise requests.exceptions.ConnectTimeout(
                "Connection to overpass-api.de timed out. (connect timeout=180)")
        return "G"

    monkeypatch.setattr(osm_data.ox, "graph_from_point", flaky_graph_from_point)
    assert osm_data._download_graph("Ulm, Germany", "drive", (48.4, 9.99, 800.0)) == "G"
    assert len(attempts) == 3
    assert "_download_graph: attempt 1 failed" in capsys.readouterr().out


def test_geocode_failure_is_not_retried(monkeypatch) -> None:
    import monocular_osm.osm_data as osm_data

    attempts = []

    def no_such_place(place, network_type):
        attempts.append(1)
        raise ValueError("Nominatim could not geocode query 'Qwertyville'")

    monkeypatch.setattr(osm_data.ox, "graph_from_place", no_such_place)
    with pytest.raises(ValueError):
        osm_data._download_graph("Qwertyville", "drive", None)
    assert len(attempts) == 1


def test_network_error_message_does_not_blame_the_spelling(monkeypatch, tmp_path) -> None:
    import monocular_osm.pipeline as pipeline

    def down(*a, **k):
        raise requests.exceptions.ConnectTimeout("connect timeout=180")

    monkeypatch.setattr(pipeline, "fetch_city_graph", down)
    with pytest.raises(ValueError, match="Could not reach Overpass") as ei:
        pipeline._fetch_road_graph("Ulm, Germany", tmp_path / "g.graphml")
    assert "spelling" not in str(ei.value)
