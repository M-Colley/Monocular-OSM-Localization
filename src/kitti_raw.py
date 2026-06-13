"""Adapter: KITTI raw drives → our video + ground-truth format.

KITTI raw (cvlibs.net) is the cleanest fit for this pipeline's data
shape: urban/residential Karlsruhe driving, a forward colour camera, and
— crucially — an OXTS INS/GNSS unit logging **global latitude/longitude
per frame**. So unlike most AV benchmarks (local map frames), KITTI raw
gives us true geographic ground truth with no georeferencing puzzle.

A synced drive directory looks like::

    2011_09_26_drive_0009_sync/
        image_02/data/0000000000.png ...   # left colour camera (forward)
        image_02/timestamps.txt
        oxts/data/0000000000.txt ...        # 30 values; [0]=lat [1]=lon ...
        oxts/timestamps.txt

This module reads the OXTS track into our :class:`GpsFix` /
``ground_truth`` schema and renders ``image_02`` into an mp4 the existing
pipeline can consume — so a KITTI drive drops straight into
``main.py --video ... --ground-truth-waypoints ...``.
"""

from __future__ import annotations

from pathlib import Path

from .gps_overlay import GpsFix, osm_around_for_track, track_to_ground_truth

__all__ = [
    "KITTI_FPS", "load_oxts_track", "kitti_ground_truth",
    "render_images_to_video", "osm_around_for_track",
]

KITTI_FPS = 10.0  # KITTI raw is captured/synced at 10 Hz


def _read_timestamps(path: Path) -> list[float] | None:
    """Parse a KITTI ``timestamps.txt`` into seconds-from-start, or None."""
    if not path.exists():
        return None
    import datetime as _dt

    secs: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: '2011-09-26 13:02:25.援...' (nanosecond fraction).
        stamp = line[:26]  # trim to microseconds for fromisoformat
        try:
            dt = _dt.datetime.fromisoformat(stamp)
        except ValueError:
            return None
        secs.append(dt.timestamp())
    if not secs:
        return None
    t0 = secs[0]
    return [s - t0 for s in secs]


def load_oxts_track(drive_dir: Path) -> list[GpsFix]:
    """Read the OXTS GPS track of a synced KITTI raw drive.

    Each ``oxts/data/NNN.txt`` line begins with ``lat lon alt roll ...``;
    we take the first two as WGS84 latitude/longitude. Timestamps come
    from ``oxts/timestamps.txt`` when present, else frame-index / 10 Hz.
    """
    drive_dir = Path(drive_dir)
    oxts_dir = drive_dir / "oxts" / "data"
    files = sorted(oxts_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no OXTS files under {oxts_dir}")
    ts = _read_timestamps(drive_dir / "oxts" / "timestamps.txt")
    fixes: list[GpsFix] = []
    for i, f in enumerate(files):
        parts = f.read_text(encoding="utf-8").split()
        if len(parts) < 2:
            continue
        lat, lon = float(parts[0]), float(parts[1])
        t_sec = ts[i] if (ts is not None and i < len(ts)) else i / KITTI_FPS
        fixes.append(GpsFix(t_sec=float(t_sec), lat=lat, lon=lon))
    return fixes


def kitti_ground_truth(drive_dir: Path, *, n_waypoints: int = 12) -> dict:
    """Build the project's ground-truth JSON from a KITTI drive's OXTS."""
    fixes = load_oxts_track(drive_dir)
    drive = Path(drive_dir).name
    return track_to_ground_truth(
        fixes, video_id=drive, video_url="https://www.cvlibs.net/datasets/kitti/raw_data.php",
        city="Karlsruhe, Germany", n_waypoints=n_waypoints,
    ) | {"source": "kitti_raw_oxts"}


def render_images_to_video(
    drive_dir: Path, out_path: Path, *, camera: str = "image_02", fps: float = KITTI_FPS
) -> Path:
    """Encode a KITTI camera's PNG sequence into an mp4 for the pipeline."""
    import cv2

    img_dir = Path(drive_dir) / camera / "data"
    frames = sorted(img_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no PNG frames under {img_dir}")
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"cv2 VideoWriter failed to open {out_path}")
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is not None:
            vw.write(img)
    vw.release()
    return out_path
