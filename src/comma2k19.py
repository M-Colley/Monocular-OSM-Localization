"""Adapter: comma2k19 segments → our video + ground-truth format.

comma2k19 (comma.ai) is ~33 h of California CA-280 commute driving in
2019 segments of 1 minute each. Its distinguishing feature for us is the
**ground-truth quality**: each segment ships a global pose computed by a
tightly-coupled INS/GNSS/Vision optimiser — far better than a bare GPS
fix. That makes it the gold reference for validating our monocular
localization, complementing KITTI raw (cleaner GPS, urban Karlsruhe).

A segment directory looks like::

    <route>/<seg>/
        video.hevc                     # road camera, 1164x874 @ 20 Hz
        global_pose/frame_positions    # (N,3) ECEF metres, one per frame
        global_pose/frame_orientations # (N,4) quaternions  (unused here)
        global_pose/frame_times        # (N,) device boot time, seconds

The poses are in ECEF (Earth-Centred Earth-Fixed) metres; we convert to
WGS84 lat/lon with pyproj. Because the pose is global and continuous,
*consecutive* segments of one route concatenate into a longer drive —
which is what gives a highway clip enough trajectory shape to localize.
This module reads a route's poses into our GT schema and transcodes its
``video.hevc`` files into one mp4 the pipeline can consume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .gps_overlay import GpsFix, osm_around_for_track, track_to_ground_truth

__all__ = [
    "COMMA_FPS", "ecef_to_latlon", "load_segment_track", "load_route_track",
    "comma_ground_truth", "render_route_to_video", "osm_around_for_track",
]

COMMA_FPS = 20.0  # comma2k19 road camera is 20 Hz


def _load_npy(path: Path) -> np.ndarray:
    """Load a comma2k19 numpy array (stored without a .npy extension)."""
    with open(path, "rb") as fh:
        return np.load(fh, allow_pickle=False)


def ecef_to_latlon(positions: np.ndarray) -> np.ndarray:
    """Convert (N,3) ECEF metres to (N,2) WGS84 ``[lat, lon]`` degrees."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    xyz = np.atleast_2d(positions)
    lon, lat, _alt = t.transform(
        xyz[:, 0].tolist(), xyz[:, 1].tolist(), xyz[:, 2].tolist())
    return np.column_stack([np.asarray(lat), np.asarray(lon)])


def load_segment_track(seg_dir: Path) -> list[GpsFix]:
    """Read one segment's global pose into a lat/lon track.

    Times are referenced to the segment's first frame; downsampling /
    waypoint selection happens later in :func:`track_to_ground_truth`.
    """
    seg_dir = Path(seg_dir)
    pose = seg_dir / "global_pose"
    positions = _load_npy(pose / "frame_positions")
    latlon = ecef_to_latlon(positions)
    times_path = pose / "frame_times"
    if times_path.exists():
        times = _load_npy(times_path).astype(float)
        times = times - times[0]
    else:
        times = np.arange(len(latlon)) / COMMA_FPS
    n = min(len(latlon), len(times))
    return [GpsFix(float(times[i]), float(latlon[i, 0]), float(latlon[i, 1]))
            for i in range(n)]


def load_route_track(seg_dirs: list[Path]) -> list[GpsFix]:
    """Concatenate consecutive segments into one continuous track.

    Each segment's frame times restart near zero, so we lay segments
    end-to-end on a synthetic clock: every segment after the first starts
    one frame-interval after the previous one ended.
    """
    track: list[GpsFix] = []
    t_offset = 0.0
    for seg in seg_dirs:
        seg_track = load_segment_track(seg)
        if not seg_track:
            continue
        for f in seg_track:
            track.append(GpsFix(f.t_sec + t_offset, f.lat, f.lon))
        t_offset = track[-1].t_sec + 1.0 / COMMA_FPS
    return track


def comma_ground_truth(seg_dirs: list[Path], *, n_waypoints: int = 12) -> dict:
    """Build the project's ground-truth JSON from a comma2k19 route."""
    fixes = load_route_track(seg_dirs)
    if not fixes:
        raise ValueError("no poses found in the given segments")
    route_id = Path(seg_dirs[0]).parent.name or Path(seg_dirs[0]).name
    return track_to_ground_truth(
        fixes, video_id=f"comma2k19_{route_id}",
        video_url="https://huggingface.co/datasets/commaai/comma2k19",
        city="San Francisco Peninsula, California, USA",
        n_waypoints=n_waypoints,
    ) | {"source": "comma2k19_global_pose"}


def render_route_to_video(
    seg_dirs: list[Path], out_path: Path, *, fps: float = COMMA_FPS
) -> Path:
    """Transcode a route's ``video.hevc`` files into one mp4 in order."""
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for seg in seg_dirs:
            hevc = Path(seg) / "video.hevc"
            if not hevc.exists():
                raise FileNotFoundError(f"missing {hevc}")
            cap = cv2.VideoCapture(str(hevc))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"VideoWriter failed to open {out_path}")
                writer.write(frame)
            cap.release()
    finally:
        if writer is not None:
            writer.release()
    if writer is None:
        raise RuntimeError("no frames decoded from the given segments")
    return out_path
