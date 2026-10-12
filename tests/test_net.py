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
def _no_sleep_no_network(monkeypatch):
    import contextlib

    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    # Unit tests must not probe real Overpass addresses.
    monkeypatch.setattr("monocular_osm.osm_data.prefer_reachable_host",
                        lambda url, **k: contextlib.nullcontext(None))


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


# --- reachable-address pinning ------------------------------------------------

def _fake_network(monkeypatch, addrs, alive):
    """getaddrinfo returns ``addrs``; only addresses in ``alive`` connect."""
    import socket

    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port, family=0, type=0, proto=0, flags=0:
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addrs])

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_create_connection(address, timeout=None):
        if address[0] in alive:
            return _Conn()
        raise TimeoutError("timed out")

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)


def test_reachable_address_skips_a_black_holed_machine(monkeypatch) -> None:
    _fake_network(monkeypatch, ["162.55.144.139", "65.109.112.52"], {"65.109.112.52"})
    assert net.reachable_address("overpass-api.de") == "65.109.112.52"


def test_reachable_address_none_when_all_down(monkeypatch) -> None:
    _fake_network(monkeypatch, ["10.0.0.1", "10.0.0.2"], set())
    assert net.reachable_address("overpass-api.de") is None


def test_prefer_reachable_host_patches_only_inside_the_block(monkeypatch) -> None:
    import socket

    _fake_network(monkeypatch, ["162.55.144.139", "65.109.112.52"], {"65.109.112.52"})
    real = lambda name: "162.55.144.139"                  # the dead one, as Windows caches it
    monkeypatch.setattr(socket, "gethostbyname", real)
    with net.prefer_reachable_host("https://overpass-api.de/api") as addr:
        assert addr == "65.109.112.52"
        assert socket.gethostbyname("overpass-api.de") == "65.109.112.52"
        assert socket.gethostbyname("example.org") == "162.55.144.139"   # others untouched
    assert socket.gethostbyname("overpass-api.de") == "162.55.144.139"   # restored


def test_prefer_reachable_host_restores_even_on_error(monkeypatch) -> None:
    import socket

    _fake_network(monkeypatch, ["1.1.1.1"], {"1.1.1.1"})
    real = lambda name: "9.9.9.9"
    monkeypatch.setattr(socket, "gethostbyname", real)
    with pytest.raises(RuntimeError):
        with net.prefer_reachable_host("https://overpass-api.de/api"):
            raise RuntimeError("boom")
    assert socket.gethostbyname("overpass-api.de") == "9.9.9.9"


def test_prefer_reachable_host_is_a_noop_when_nothing_answers(monkeypatch) -> None:
    import socket

    _fake_network(monkeypatch, ["1.1.1.1"], set())
    real = lambda name: "9.9.9.9"
    monkeypatch.setattr(socket, "gethostbyname", real)
    with net.prefer_reachable_host("https://overpass-api.de/api") as addr:
        assert addr is None
        assert socket.gethostbyname("overpass-api.de") == "9.9.9.9"
