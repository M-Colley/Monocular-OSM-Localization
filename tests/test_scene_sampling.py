"""OCR acquisition stays streaming and distinguishes its sampling grids."""
import weakref

import cv2
import numpy as np
import pytest

from monocular_osm.scene_text import _sample_frames, extract_scene_text


class Capture:
    def __init__(self, total=600):
        self.total = total
        self.position = 0
        self.read_indices = []
        self.images = []
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_COUNT: self.total,
                cv2.CAP_PROP_POS_MSEC: (self.position - .75) * 100}.get(key, 0)

    def set(self, key, value):
        self.position = int(value)
        return True

    def read(self):
        if self.position >= self.total:
            return False, None
        self.read_indices.append(self.position)
        image = np.full((32, 48, 3), self.position % 255, np.uint8)
        self.images.append(weakref.ref(image))
        self.position += 1
        return True, image

    def release(self):
        self.released = True


def test_default_ocr_streams_native_frames_with_unchanged_interval_timestamps(tmp_path, monkeypatch):
    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    class Reader:
        times = 0
        def readtext(self, image):
            assert sum(ref() is not None for ref in capture.images) <= 2
            assert image.shape == (32, 48, 3)
            self.times += 1
            return [([], "Example", .9)]

    reader = Reader()
    result = extract_scene_text(tmp_path / "video.mp4", sample_interval_sec=1, ocr_reader=reader)
    assert [d.t_sec for d in result] == list(range(60))
    assert reader.times == 60 and capture.released


def test_fixed_count_uses_same_linspace_grid_as_probe_and_actual_pts(monkeypatch):
    capture = Capture(total=60)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    frames = list(_sample_frames("video.mp4", 0, 10, 6, sample_count=4))
    assert capture.read_indices == [0, 19, 39, 59]
    assert [t for t, _ in frames] == pytest.approx([.025, 1.925, 3.925, 5.925])
    assert capture.released


def test_ocr_error_closes_stream_capture(tmp_path, monkeypatch):
    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    class Reader:
        def readtext(self, image):
            raise RuntimeError("OCR failed")

    with pytest.raises(RuntimeError, match="OCR failed"):
        extract_scene_text(tmp_path / "video.mp4", ocr_reader=Reader())
    assert capture.released


def test_fixed_sample_count_invalidates_ocr_cache(tmp_path):
    cache = tmp_path / "ocr.json"
    class Reader:
        def __init__(self, text):
            self.text = text
        def readtext(self, image):
            return [([], self.text, .9)]
    kwargs = {"cache_path": cache, "frame_reader": lambda *args: [(0, object())]}
    first = extract_scene_text(tmp_path / "video.mp4", sample_count=2, ocr_reader=Reader("First"), **kwargs)
    second = extract_scene_text(tmp_path / "video.mp4", sample_count=3, ocr_reader=Reader("Second"), **kwargs)
    assert first[0].text == "First" and second[0].text == "Second"


@pytest.mark.parametrize("count", [0, -1, True, 1.5, float("nan")])
def test_invalid_fixed_count_is_refused_before_video_access(count):
    with pytest.raises(ValueError, match="sample count"):
        extract_scene_text("missing.mp4", sample_count=count)


def test_fixed_count_starts_on_the_vo_frame_it_was_given(monkeypatch):
    # Detroit g5lnpYCk1Ec: the recorded VO horizon starts at frame 119, stamped
    # 3.986065518858444 s (one ulp below 119 / 29.854); int(that * 29.854) is
    # 118, one frame before the horizon, so every assisted run of the clip
    # refused its OCR cue.
    class Capture29(Capture):
        def get(self, key):
            return {cv2.CAP_PROP_FPS: 29.854, cv2.CAP_PROP_FRAME_COUNT: self.total,
                    cv2.CAP_PROP_POS_MSEC: self.position / 29.854 * 1000}.get(key, 0)

    capture = Capture29(total=18000)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    start = 3.986065518858444
    assert int(start * 29.854) == 118                       # the float artefact itself
    frames = list(_sample_frames("video.mp4", start, 17879 / 29.854, 26, sample_count=24))
    assert capture.read_indices[0] == 119 and frames[0][0] >= start
