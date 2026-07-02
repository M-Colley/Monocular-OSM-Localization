"""Tests for license-plate district parsing + voting (ALPR/geocoder injected)."""

from __future__ import annotations

from types import SimpleNamespace

import src.plate_anchor as pa
from src.plate_anchor import _candidate_codes, plate_district_anchor


# ---------------------------------------------------------------------------
# _candidate_codes — all-splits enumeration (audit plate_anchor:32 examples)
# ---------------------------------------------------------------------------


def test_bab123_is_ambiguous_not_bamberg() -> None:
    # Berlin plate B-AB 123 stripped: the greedy prefix read it as BA=Bamberg.
    codes = _candidate_codes("BAB123")
    assert set(codes) == {"B", "BA"}       # both splits valid -> no unique code


def test_mab123_is_ambiguous_not_mannheim() -> None:
    # Munich M-AB: greedy prefix read MA=Mannheim.
    assert set(_candidate_codes("MAB123")) == {"M", "MA"}


def test_bot1234_is_ambiguous_not_bochum() -> None:
    # B-OT: greedy prefix read BO=Bochum. BOT (Bottrop) is not a valid split
    # here because a real plate needs 1-2 serial letters before the digits.
    assert set(_candidate_codes("BOT1234")) == {"B", "BO"}


def test_ulab123_resolves_to_ulm() -> None:
    # UL-AB: the greedy prefix took 'ULA' (not a code) and dropped the vote.
    assert _candidate_codes("ULAB123") == ["UL"]


def test_preserved_separator_disambiguates() -> None:
    # OCR that kept the separator makes even 2-serial big-city plates unique.
    assert _candidate_codes("B-AB 123") == ["B"]
    assert _candidate_codes("UL-AB 123") == ["UL"]
    assert _candidate_codes("M AB 123") == ["M"]


def test_candidate_codes_rejects_junk() -> None:
    assert _candidate_codes("XQ-ZZ 999") == []      # XQ not a district
    assert _candidate_codes("ULABCD1234") == []     # > 8 chars / 3 serial letters


# ---------------------------------------------------------------------------
# plate_district_anchor — voting + margin gate (ALPR faked)
# ---------------------------------------------------------------------------


class _FakeAlpr:
    """One list of plate texts per frame."""

    def __init__(self, per_frame: list[list[str]]) -> None:
        self._pf = per_frame
        self._i = 0

    def predict(self, img):
        texts = self._pf[self._i] if self._i < len(self._pf) else []
        self._i += 1
        return [SimpleNamespace(ocr=SimpleNamespace(text=t, confidence=0.9))
                for t in texts]


def _anchor(monkeypatch, per_frame, geocode=lambda q: (48.4, 9.99)):
    monkeypatch.setattr(pa, "_alpr", lambda: _FakeAlpr(per_frame))
    frames = [object()] * len(per_frame)
    return plate_district_anchor("unused.mp4", frames=frames, geocode_fn=geocode)


def test_two_unique_ulm_plates_yield_anchor(monkeypatch) -> None:
    a = _anchor(monkeypatch, [["UL-AB 123"], ["UL-CD 456"]])
    assert a is not None
    assert a.code == "UL" and a.district == "Ulm" and a.votes == 2


def test_berlin_two_serial_plates_vote_berlin_not_siblings(monkeypatch) -> None:
    # The audit's Berlin failure: B-AB / B-OT used to vote Bamberg / Bochum.
    a = _anchor(monkeypatch, [["B-AB 123"], ["B-OT 456"]])
    assert a is not None
    assert a.code == "B" and a.district == "Berlin"


def test_stripped_ambiguous_plates_do_not_vote(monkeypatch) -> None:
    # Separator lost in OCR -> every split ambiguous -> zero votes -> no anchor.
    a = _anchor(monkeypatch, [["BAB123"], ["MAB123"], ["BOT1234"]])
    assert a is None


def test_single_vote_is_not_enough(monkeypatch) -> None:
    assert _anchor(monkeypatch, [["UL-AB 123"]]) is None


def test_margin_gate_rejects_close_ballot(monkeypatch) -> None:
    # 3 vs 2 fails n > 1.5 * runner-up; 4 vs 2 passes.
    close = [["UL-AB 123"], ["UL-CD 45"], ["UL-EF 6"], ["NU-GH 78"], ["NU-JK 9"]]
    assert _anchor(monkeypatch, close) is None
    clear = close + [["UL-XY 77"]]
    a = _anchor(monkeypatch, clear)
    assert a is not None and a.code == "UL" and a.margin == 2.0


def test_same_plate_seen_twice_votes_once(monkeypatch) -> None:
    # One tracked car must not stuff the ballot (still below the 2-vote bar).
    assert _anchor(monkeypatch, [["UL-AB 123"], ["UL-AB 123"]]) is None
