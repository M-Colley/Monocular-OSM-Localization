"""Tests for selecting the OCR backend.

Two engines read the SAME pixels differently, so the choice has to reach
every consumer and must never let one engine's cached output be served to
the other. These tests use stubs — no model download, no GPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from monocular_osm.scene_text import (
    RapidOcrReader,
    _cache_signature,
    _default_reader,
    extract_scene_text,
)

REPO = Path(__file__).resolve().parents[1]


class _FakeRapid:
    """Mimics rapidocr's parallel-tuple result object."""

    class _Res:
        txts = ("N:41.8933 W:87.6216", "HDR")
        scores = (0.98, 0.99)
        boxes = [[(1, 2), (3, 2), (3, 4), (1, 4)], [(5, 6), (7, 6), (7, 8), (5, 8)]]

    def __call__(self, image):
        return self._Res()


def test_rapid_adapter_converts_to_the_easyocr_shape(monkeypatch) -> None:
    """RapidOCR returns parallel tuples; the rest of the pipeline expects
    per-detection (bbox, text, confidence) triples."""
    reader = RapidOcrReader.__new__(RapidOcrReader)
    reader._engine = _FakeRapid()
    out = reader.readtext(object())
    assert len(out) == 2
    bbox, text, conf = out[0]
    assert text == "N:41.8933 W:87.6216"
    assert conf == pytest.approx(0.98)
    assert bbox == [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]]


def test_rapid_adapter_handles_an_empty_read() -> None:
    class _Empty:
        class _Res:
            txts = ()
            scores = ()
            boxes = None

        def __call__(self, image):
            return self._Res()

    reader = RapidOcrReader.__new__(RapidOcrReader)
    reader._engine = _Empty()
    assert reader.readtext(object()) == []


def test_default_reader_rejects_an_unknown_engine() -> None:
    with pytest.raises(ValueError, match="unknown OCR engine"):
        _default_reader(("en",), False, "tesseract")


def test_the_default_engine_is_rapidocr() -> None:
    import inspect

    from monocular_osm.gps_overlay import extract_gps_track

    assert inspect.signature(_default_reader).parameters["engine"].default == "rapidocr"
    assert inspect.signature(extract_scene_text).parameters["engine"].default == "rapidocr"
    assert inspect.signature(extract_gps_track).parameters["engine"].default == "rapidocr"
    # PipelineConfig lives behind an open3d import, so read its default from
    # source rather than importing it — this must hold in a CPU-only checkout.
    src = (REPO / "monocular_osm" / "pipeline.py").read_text(encoding="utf-8")
    assert 'ocr_engine: str = "rapidocr"' in src


def _break_import(monkeypatch, missing: str) -> list[str]:
    """Make one backend unimportable; record which readers were attempted."""
    tried: list[str] = []
    import monocular_osm.scene_text as st

    class _Reader:
        def readtext(self, image):
            return []

    def fake_rapid():
        tried.append("rapidocr")
        if missing == "rapidocr":
            raise ImportError("No module named 'rapidocr'")
        return _Reader()

    def fake_easy(languages, use_gpu):
        tried.append("easyocr")
        if missing == "easyocr":
            raise ImportError("No module named 'easyocr'")
        return _Reader()

    monkeypatch.setattr(st, "RapidOcrReader", fake_rapid)
    monkeypatch.setattr(st, "_easyocr_reader", fake_easy)
    return tried


def test_missing_default_backend_falls_back_with_a_warning(monkeypatch) -> None:
    """Making rapidocr the default must not break installs that only have
    easyocr — the OCR channel is one of several, not the whole run."""
    tried = _break_import(monkeypatch, missing="rapidocr")
    with pytest.warns(RuntimeWarning, match="not installed"):
        reader = _default_reader(("en",), False, "rapidocr")
    assert reader is not None
    assert tried == ["rapidocr", "easyocr"]


def test_explicitly_requesting_the_missing_one_also_falls_back(monkeypatch) -> None:
    tried = _break_import(monkeypatch, missing="easyocr")
    with pytest.warns(RuntimeWarning, match="not installed"):
        _default_reader(("en",), False, "easyocr")
    assert tried == ["easyocr", "rapidocr"]


def test_no_backend_at_all_names_both_extras(monkeypatch) -> None:
    import monocular_osm.scene_text as st

    def boom(*a, **k):
        raise ImportError("nope")

    monkeypatch.setattr(st, "RapidOcrReader", boom)
    monkeypatch.setattr(st, "_easyocr_reader", boom)
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ImportError, match=r"\[rapidocr\]"):
            _default_reader(("en",), False, "rapidocr")


def test_cache_signature_separates_the_engines() -> None:
    """A cache built with one engine must not be served to the other — they
    read different text off identical pixels, which is the whole point."""
    common = dict(sample_interval_sec=6.0, start_sec=0.0, end_sec=None,
                  languages=("en",), min_confidence=0.3, min_len=3)
    a = _cache_signature(**common, engine="easyocr")
    b = _cache_signature(**common, engine="rapidocr")
    assert a != b
    assert a["engine"] == "easyocr" and b["engine"] == "rapidocr"


def test_extract_scene_text_passes_the_engine_through(monkeypatch, tmp_path) -> None:
    seen = {}

    def _fake_default(languages, use_gpu, engine="easyocr"):
        seen["engine"] = engine

        class _R:
            def readtext(self, image):
                return []
        return _R()

    monkeypatch.setattr("monocular_osm.scene_text._default_reader", _fake_default)
    extract_scene_text(tmp_path / "v.mp4", engine="rapidocr",
                       frame_reader=lambda *a, **k: [])
    assert seen["engine"] == "rapidocr"


def test_gps_overlay_track_cache_separates_the_engines(tmp_path) -> None:
    from monocular_osm.gps_overlay import _VARIANTS, _track_cache_signature

    args = (tmp_path / "v.mp4", 5.0, 0.0, None, "bottom", 0.2, 400.0, _VARIANTS)
    assert _track_cache_signature(*args, "easyocr") != \
        _track_cache_signature(*args, "rapidocr")


# ---------------------------------------------------------------------------
# scope check — cfg is not available everywhere in pipeline.py
# ---------------------------------------------------------------------------


def test_pipeline_never_reads_cfg_outside_a_function_that_has_it() -> None:
    """`cfg` is a local of run_pipeline, not a global.

    Threading the engine through touched a module-level helper
    (_frame_coarse_seed) where writing `cfg.ocr_engine` parses fine and
    raises NameError only when that branch runs — which is behind a
    default-on flag and an OCR pass, so no cheap test would reach it.
    This walks the AST instead.
    """
    tree = ast.parse((REPO / "monocular_osm" / "pipeline.py").read_text(encoding="utf-8"))
    # MODULE-LEVEL functions only. A nested helper (there are several inside
    # run_pipeline) closes over cfg perfectly legitimately; a module-level one
    # cannot, so a bare `cfg.` there is always a latent NameError.
    offenders = []
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        names = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        if "cfg" in names:
            continue                       # cfg is a parameter here: fine
        nested = {n for f in ast.walk(fn) if isinstance(f, ast.FunctionDef) and f is not fn
                  for n in ast.walk(f)}
        for node in ast.walk(fn):
            if node in nested:
                continue                   # belongs to an inner closure
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "cfg"):
                offenders.append(f"{fn.name}:{node.lineno} reads cfg.{node.attr}")
    assert not offenders, "cfg referenced in a module-level function:\n  " + \
        "\n  ".join(offenders)
