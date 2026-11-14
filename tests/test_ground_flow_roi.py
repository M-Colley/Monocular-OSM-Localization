"""Geometry/compatibility checks for an explicitly requested flow seed ROI."""
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from monocular_osm import ground_flow_scale as flow


def intrinsics(width=960, height=540):
    focal = width/(2*math.tan(math.radians(35)))
    return np.array([[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]])


def projected_row(depth, height_m, pitch, K):
    """Independent scalar vehicle-to-camera projection, no production helper."""
    c, s = math.cos(pitch), math.sin(pitch)
    camera_y = c*height_m-s*depth
    camera_z = s*height_m+c*depth
    return K[1, 2]+K[1, 1]*camera_y/camera_z


def test_fixed_generic_geometry_gives_row247_without_changing_globals():
    before = (flow._ROI_TOP, flow._ROI_BOT, flow._ROI_LEFT, flow._ROI_RIGHT)
    K = intrinsics()
    row = flow.road_seed_top_row(K, (960, 540))
    assert type(row) is int and row == 247
    assert int(540*(row/540)) == row
    assert before == (flow._ROI_TOP, flow._ROI_BOT, flow._ROI_LEFT, flow._ROI_RIGHT)
    assert before == (.55, .97, .10, .90)


@pytest.mark.parametrize("width,height", [(960, 540), (1280, 720), (640, 360)])
def test_seed_envelope_covers_visible_near_far_points_across_horizon_search(width, height):
    K = intrinsics(width, height)
    top = flow.road_seed_top_row(K, (width, height))
    visible, missed_by_legacy = 0, 0
    for horizon in np.linspace(.37*height, .85*height, 41):
        pitch = math.atan((K[1, 2]-horizon)/K[0, 0])
        for depth in np.linspace(10., 20., 21):
            v = projected_row(depth, 1.4, pitch, K)
            # Existing bottom exclusion is deliberate; this change only widens top.
            if 0 <= v < int(height*.97):
                visible += 1
                assert top <= math.floor(v)
                missed_by_legacy += math.floor(v) < int(height*.55)
    assert visible > 100 and missed_by_legacy > 0


def test_geometry_exposes_the_legacy_empty_depth_band_without_real_footage():
    K = intrinsics()
    pitch = math.atan((270.-.37*540)/K[0, 0])
    near = projected_row(10., 1.4, pitch, K)
    far = projected_row(20., 1.4, pitch, K)
    assert far < near < int(540*.55)
    assert flow.road_seed_top_row(K, (960, 540)) <= far
    assert near == pytest.approx(295.40728813159524)


def test_envelope_unions_legacy_band_and_clips_to_visible_image():
    K = intrinsics()
    assert flow.road_seed_top_row(K, (960, 540), camera_height_m=10.) == int(540*.55)
    tall_focal = K.copy()
    tall_focal[1, 1] = 10000.
    assert flow.road_seed_top_row(tall_focal, (960, 540)) == 0


@pytest.mark.parametrize("parameter", ["camera_height_m", "far_m"])
@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf, -np.inf, True, "1.4"])
def test_nonphysical_geometry_parameters_decline(parameter, bad):
    with pytest.raises(ValueError, match="finite and positive"):
        flow.road_seed_top_row(intrinsics(), (960, 540), **{parameter: bad})


@pytest.mark.parametrize("bad", [(0, 540), (960, -1), (960., 540), (True, 540), (960,), "960x540"])
def test_bad_image_size_declines(bad):
    with pytest.raises(ValueError, match="positive integer"):
        flow.road_seed_top_row(intrinsics(), bad)


@pytest.mark.parametrize("bad", [np.eye(2), np.full((3, 3), np.nan), np.full((3, 3), np.inf),
    np.diag([0., 2., 1.]), np.diag([2., -1., 1.]), np.diag([2., 2., 2.])])
def test_bad_intrinsics_decline(bad):
    with pytest.raises(ValueError, match="pinhole"):
        flow.road_seed_top_row(bad, (960, 540))


@pytest.mark.parametrize("top", [-.001, .97, 1., np.nan, np.inf, -np.inf, True, "0.4", np.array([.4])])
def test_bad_roi_declines_before_filesystem_or_video_io(monkeypatch, top):
    monkeypatch.setattr(Path, "exists", lambda *_: pytest.fail("invalid ROI touched filesystem"))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: pytest.fail("invalid ROI opened video"))
    with pytest.raises(ValueError, match="roi_top_frac"):
        flow.track_road_features(Path("absent.mp4"), stride=6, roi_top_frac=top)


def tracking_fixture(tmp_path, monkeypatch):
    video = tmp_path/"synthetic.mp4"
    video.write_bytes(b"fake capture; no encoded video needed")
    frames = [np.full((54, 96, 3), i, np.uint8) for i in range(7)]
    masks, captures = [], []
    class Capture:
        def __init__(self, *_): self.index = 0; self.released = False; captures.append(self)
        def isOpened(self): return True
        def get(self, *_): return 30.
        def read(self):
            if self.index == len(frames): return False, None
            frame = frames[self.index].copy(); self.index += 1
            return True, frame
        def release(self): self.released = True
    def correspondences(buf, roi, corners, size):
        masks.append(roi.copy())
        if not roi.any(): return np.zeros((0, 4), np.float32)
        row, col = np.argwhere(roi > 0)[0]
        return np.array([[col, row, col+1, row+1]], np.float32)
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr(flow, "_step_correspondences", correspondences)
    return video, masks, captures


def test_none_and_explicit_legacy_roi_have_identical_mask_and_tracks(tmp_path, monkeypatch):
    video, masks, captures = tracking_fixture(tmp_path, monkeypatch)
    a = flow.track_road_features(video, stride=2, size=(96, 54))
    b = flow.track_road_features(video, stride=2, size=(96, 54), roi_top_frac=.55)
    expected = np.zeros((54, 96), np.uint8)
    expected[int(54*.55):int(54*.97), int(96*.10):int(96*.90)] = 255
    assert len(masks) == 6 and all(mask.tobytes() == expected.tobytes() for mask in masks)
    assert a.pts.tobytes() == b.pts.tobytes() and a.offsets.tobytes() == b.offsets.tobytes()
    assert a.fps == b.fps == 30. and a.stride == b.stride == 2 and a.size == b.size == (96, 54)
    assert all(cap.released for cap in captures)


def test_requested_geometry_top_only_widens_seed_mask(tmp_path, monkeypatch):
    video, masks, _ = tracking_fixture(tmp_path, monkeypatch)
    size = (960, 540)
    top = flow.road_seed_top_row(intrinsics(), size)
    flow.track_road_features(video, stride=6, size=size, roi_top_frac=top/size[1])
    assert len(masks) == 1
    mask = masks[0]
    assert np.flatnonzero(mask.any(axis=1))[0] == 247
    assert np.flatnonzero(mask.any(axis=1))[-1] == int(540*.97)-1
    assert not mask[:247].any() and not mask[:, :96].any() and not mask[:, int(960*.90):].any()
    legacy = np.zeros_like(mask); legacy[297:int(540*.97), 96:int(960*.90)] = 255
    assert np.all(mask[legacy > 0] == 255) and mask.sum() > legacy.sum()


@pytest.mark.parametrize("top", [0, np.float64(.4), np.float32(.6), .969])
def test_valid_explicit_fraction_is_used_and_capture_released(tmp_path, monkeypatch, top):
    video, masks, captures = tracking_fixture(tmp_path, monkeypatch)
    tracks = flow.track_road_features(video, stride=6, size=(96, 54), roi_top_frac=top)
    if int(54*top) < int(54*.97):
        assert np.flatnonzero(masks[0].any(axis=1))[0] == int(54*top)
    else:
        assert not masks[0].any() and len(tracks.pts) == 0
    assert captures[0].released
