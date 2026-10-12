"""Retry policy for the network calls a run depends on.

A single run touches Overpass (road graph), Nominatim, Mapillary and
YouTube, and each of them intermittently times out or answers 403/429/5xx.
Without retries one transient failure killed a run that had already spent
minutes on visual odometry and image retrieval: in a 15-clip benchmark,
12 runs died at the OSM graph fetch with ``ConnectTimeout`` to
overpass-api.de (while the server answered fine minutes later) and two
YouTube downloads returned a one-off 403.

Retries are bounded and exponential, only for errors that are plausibly
transient — a misspelt city must still fail immediately — and each retry
is announced, so a run's log says what it waited for.
"""

from __future__ import annotations

import contextlib
import socket
import sys
from urllib.parse import urlparse

import requests
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# Substrings of yt-dlp / osmnx error text that mark a transient failure.
_TRANSIENT_TEXT = (
    "403", "429", "500", "502", "503", "504", "timed out", "timeout",
    "connection", "temporarily", "try again", "reset by peer",
)


def is_transient(exc: BaseException) -> bool:
    """True for failures worth retrying: network-level errors, and HTTP
    errors whose text carries a retryable status or timeout wording."""
    if isinstance(exc, (requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                        ConnectionError, TimeoutError)):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(t in text for t in _TRANSIENT_TEXT)


def _announce(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    wait = state.next_action.sleep if state.next_action else 0
    name = getattr(state.fn, "__name__", "call")
    print(f"      -> {name}: attempt {state.attempt_number} failed "
          f"({type(exc).__name__}: {str(exc)[:120]}); retrying in {wait:.0f} s",
          file=sys.stdout, flush=True)


def transient_retry(*, attempts: int = 4, first_wait_s: float = 20.0,
                    max_wait_s: float = 240.0):
    """Decorator: retry on transient errors with exponential backoff
    (20 s, 40 s, 80 s ... capped), re-raising the last error when exhausted."""
    return retry(
        retry=retry_if_exception(is_transient),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=first_wait_s, min=first_wait_s, max=max_wait_s),
        before_sleep=_announce,
        reraise=True,
    )


def reachable_address(hostname: str, port: int = 443, probe_timeout: float = 4.0) -> str | None:
    """The first IPv4 address of ``hostname`` that accepts a TCP connection.

    overpass-api.de is round-robin DNS over several machines, and from some
    networks one of them is black-holed (connections hang until the timeout).
    """
    try:
        infos = socket.getaddrinfo(hostname, port, family=socket.AF_INET,
                                   type=socket.SOCK_STREAM)
    except OSError:
        return None
    for addr in dict.fromkeys(i[4][0] for i in infos):
        try:
            with socket.create_connection((addr, port), timeout=probe_timeout):
                return addr
        except OSError:
            continue
    return None


@contextlib.contextmanager
def prefer_reachable_host(url: str, probe_timeout: float = 4.0):
    """While active, ``socket.gethostbyname(<host of url>)`` returns a
    *reachable* address.

    osmnx resolves the Overpass hostname once with ``gethostbyname`` and pins
    every request of the call to that single address (so its rate-limit check
    and its query reach the same server). On Windows the OS returns the cached
    first record, so when that record is the black-holed machine EVERY attempt
    - and every later run - times out for the full 180 s connect timeout,
    which retrying cannot fix: in a 15-clip benchmark 12 runs died this way
    while the other address answered in 60 ms. This probes the candidates
    briefly and makes osmnx's single resolution land on one that works.

    If no address is reachable (or the host is an IP / unresolvable) nothing is
    patched and behaviour is unchanged. Not thread-safe (it swaps a stdlib
    function for the duration of the call).
    """
    host = urlparse(url).hostname
    port = urlparse(url).port or (443 if urlparse(url).scheme == "https" else 80)
    addr = reachable_address(host, port, probe_timeout) if host else None
    real = socket.gethostbyname
    if addr is None:
        yield None
        return

    def _gethostbyname(name):
        return addr if name == host else real(name)

    socket.gethostbyname = _gethostbyname
    try:
        yield addr
    finally:
        socket.gethostbyname = real
