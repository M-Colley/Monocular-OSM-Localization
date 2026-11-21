"""Synthetic native decoding and frozen-file/crop boundaries only."""
import importlib
from unittest.mock import patch
import numpy as np
import pytest

with patch.dict("sys.modules", {"orb_rerank": importlib.import_module("scripts.orb_rerank")}):
    from scripts import probe_orb_rerank as runner


def query():
    return {"fps": 10., "sampling": {"dense_count": 6, "dense_indices": [0, 2, 5]},
            "timestamps": [0., .3, .9], "native_frame_shape": [8, 12, 3]}


def capture(monkeypatch, fault=None):
    import cv2
    closed = []
    class Capture:
        current = -1
        def isOpened(self): return fault != "open"
        def grab(self):
            self.current += 1
            return not (fault == "truncated" and self.current == 4)
        def retrieve(self):
            shape = (7, 12, 3) if fault == "shape" else (8, 12, 3)
            return fault != "retrieve", np.full(shape, self.current, np.uint8)
        def get(self, key):
            if key == cv2.CAP_PROP_FPS: return 20. if fault == "fps" else 10.
            if key == cv2.CAP_PROP_POS_FRAMES: return self.current+1+(1 if fault == "ordinal" else 0)
            return {0: 0., 2: 300., 5: 900.}.get(self.current, 0)+(1 if fault == "pts" else 0)
        def release(self): closed.append(True)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: Capture())
    return closed


def test_stream_queries_preserves_vfr_pts_and_exact_native_ordinals(monkeypatch):
    closed = capture(monkeypatch)
    rows = list(runner.stream_queries("synthetic.mp4", query()))
    assert [row for row, _ in rows] == [0, 1, 2]
    assert [int(image[0, 0, 0]) for _, image in rows] == [0, 2, 5]
    assert closed


@pytest.mark.parametrize("fault", ["fps", "pts", "ordinal", "shape", "retrieve", "truncated", "open"])
def test_stream_queries_refuses_inconsistent_or_partial_native_inputs_and_releases(monkeypatch, fault):
    closed = capture(monkeypatch, fault)
    with pytest.raises(ValueError):
        list(runner.stream_queries("synthetic.mp4", query()))
    assert closed


def test_stream_generator_close_releases_decoder(monkeypatch):
    closed = capture(monkeypatch)
    stream = runner.stream_queries("synthetic.mp4", query())
    assert next(stream)[0] == 0
    stream.close()
    assert closed


@pytest.mark.parametrize("shape", [(4, 10, 3), (10, 4, 3), (5, 5, 3), (4, 7, 3)])
def test_every_wide_tall_square_view_matches_the_retrieval_native_crop(shape):
    from monocular_osm.kartaview_vpr import _square_query_crop
    pixels = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    for view in range(3):
        actual = runner.square_view(pixels, view)
        np.testing.assert_array_equal(actual, _square_query_crop(pixels, view))
        assert actual.shape == (min(shape[:2]), min(shape[:2]), 3)


def test_checked_frozen_file_accepts_only_within_root(tmp_path):
    root = tmp_path/"frozen"; root.mkdir()
    target = root/"sub/input.json"; target.parent.mkdir(); target.write_text("{}")
    assert runner.checked_file(root, "sub/input.json") == target.resolve()
    assert runner.checked_file(root, "sub\\input.json") == target.resolve()


@pytest.mark.parametrize("relative", ["scores.json", "SCORES.JSON"])
def test_score_artifacts_are_never_opened_even_with_case_variant(tmp_path, relative):
    root = tmp_path/"frozen"; root.mkdir(); (root/relative).write_text("synthetic forbidden marker")
    with pytest.raises(ValueError): runner.checked_file(root, relative)


@pytest.mark.parametrize("absolute", [False, True])
def test_checked_file_refuses_escape_before_hashing_or_reading(tmp_path, absolute):
    root = tmp_path/"frozen"; root.mkdir()
    outside = tmp_path/"outside.json"; outside.write_text("synthetic forbidden marker")
    relative = str(outside) if absolute else "../outside.json"
    with pytest.raises(ValueError): runner.checked_file(root, relative)
