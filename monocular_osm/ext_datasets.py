"""Adapters for the 2026-07 fleet-extension datasets → our video + GT format.

Five new evaluation datasets were added to broaden the fleet beyond the
original KITTI/comma/YouTube clips (see memory dataset-shortlist-2026-07):

* **Málaga Urban** (Spain)      — consumer-GPS log + stereo image folder
* **Boreas** (Toronto)          — Applanix post-processed pose + PNG frames
* **CARD** (Italy/Germany)      — georeferenced poses + camera frames/video
* **Brno Urban** (Czechia)      — RTK GNSS csv + h265 front camera
* **ZOD Drives** (14 EU states) — OxTS RTK GNSS + front camera frames

Each loader returns a ``list[GpsFix]`` (t_sec-from-clip-start, lat, lon),
exactly like :mod:`kitti_raw` / :mod:`comma2k19`, so
:func:`gps_overlay.track_to_ground_truth` emits the standard
``ground_truth/*.json`` and the clip drops into
``main.py --video ... --ground-truth-waypoints ...``. These datasets carry
real (mostly RTK) per-frame GPS, so the GT is higher quality than the
hand-labelled YouTube clips.

The pipeline consumes an mp4, so image-sequence datasets are transcoded via
:func:`frames_to_video` (shared, format-independent); ENU-referenced poses
(Boreas) are converted with :func:`enu_to_latlon`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .gps_overlay import GpsFix, osm_around_for_track, track_to_ground_truth

__all__ = [
    "frames_to_video", "enu_to_latlon", "subsample_fixes",
    "osm_around_for_track", "track_to_ground_truth", "GpsFix",
]


# --- shared, format-independent helpers ------------------------------------


def frames_to_video(
    frame_paths: list[Path], out_path: Path, *, fps: float,
    max_frames: int | None = None,
) -> Path:
    """Encode an ordered list of image files into an mp4 for the pipeline.

    Generic version of :func:`kitti_raw.render_images_to_video` that takes an
    explicit ordered path list (datasets name frames differently). Unreadable
    frames are dropped with a loud warning — every drop shifts later frames
    earlier vs the GPS-derived GT timestamps, so silent misalignment must be
    visible. ``max_frames`` caps the clip length (evaluation clips are 1-5
    min; a full 10-min drive is unnecessary and slow).
    """
    import cv2

    frame_paths = list(frame_paths)
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        raise FileNotFoundError("no frames given to frames_to_video")
    first = None
    for fp in frame_paths:
        first = cv2.imread(str(fp))
        if first is not None:
            break
    if first is None:
        raise RuntimeError("no readable frames in the given list")
    h, w = first.shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"cv2 VideoWriter failed to open {out_path}")
    dropped = 0
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            dropped += 1
            continue
        if (img.shape[0], img.shape[1]) != (h, w):
            img = cv2.resize(img, (w, h))
        vw.write(img)
    vw.release()
    if dropped:
        warnings.warn(
            f"{dropped}/{len(frame_paths)} frames were unreadable and "
            f"dropped; the mp4 is ~{dropped / fps:.1f}s shorter than the "
            f"GPS ground truth (video/GT misalignment)",
            RuntimeWarning, stacklevel=2)
    return out_path


def enu_to_latlon(
    east_m: np.ndarray, north_m: np.ndarray, lat0: float, lon0: float,
) -> np.ndarray:
    """Convert local ENU metres (relative to a WGS84 origin) to ``[lat,lon]``.

    A local-tangent-plane inverse using the LATITUDE-DEPENDENT WGS84 metre-
    per-degree scales, not the spherical 111320 constant: at Toronto's 43.8°N
    the meridian scale is 111108 m/deg (0.19% off the constant), which is a
    2.3-3.6 m systematic error 1.8 km from the origin — the GT would be worse
    than the Applanix data it comes from (audit round-4 EXT-1; with these
    scales the in-window error vs the reference lat/lon columns is <0.3 m).
    Boreas poses are ENU relative to a per-sequence reference lat/lon.
    """
    east_m = np.asarray(east_m, float)
    north_m = np.asarray(north_m, float)
    la = np.radians(lat0)
    m_per_deg_lat = (111132.954 - 559.822 * np.cos(2 * la)
                     + 1.175 * np.cos(4 * la))
    m_per_deg_lon = (np.pi / 180.0) * 6378137.0 * np.cos(la) / np.sqrt(
        1.0 - 0.00669437999014 * np.sin(la) ** 2)
    lat = lat0 + (north_m / m_per_deg_lat)
    lon = lon0 + (east_m / m_per_deg_lon)
    return np.column_stack([lat, lon])


def subsample_fixes(fixes: list[GpsFix], every_n: int) -> list[GpsFix]:
    """Keep every ``every_n``-th fix (datasets log GPS at 100+ Hz; the GT
    schema only needs a sparse track). Endpoints preserved."""
    if every_n <= 1 or len(fixes) <= 2:
        return fixes
    keep = list(range(0, len(fixes), every_n))
    if keep[-1] != len(fixes) - 1:
        keep.append(len(fixes) - 1)
    return [fixes[i] for i in keep]


# --- Málaga Urban (Spain) --------------------------------------------------

MALAGA_FPS = 20.0  # rectified camera is 20 Hz


def malaga_track(gps_txt: Path, *, t0_abs: float | None = None) -> list[GpsFix]:
    """Read a Málaga ``*_all-sensors_GPS.txt`` into a GpsFix track.

    Columns: ``Time Lat Lon Alt fix #sats ...`` — **Lat/Lon are in RADIANS**
    (WGS84), Time is UNIX seconds. A ``%`` header line is skipped.

    ``t0_abs`` is the absolute UNIX time that becomes t_sec=0. Pass the FIRST
    VIDEO FRAME's timestamp (from the image filename): the GPS log starts a
    fraction of a second after the camera (verified 0.42 s on extract-07 ≈
    3.5 m at urban speed), so rebasing to the GPS row 0 — the default — puts
    a systematic along-track bias on every waypoint (audit round-4 EXT-2).
    """
    import math
    rows = []
    for line in Path(gps_txt).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        p = line.split()
        if len(p) < 3:
            continue
        try:
            t = float(p[0])
            lat = math.degrees(float(p[1]))
            lon = math.degrees(float(p[2]))
        except ValueError:
            continue
        rows.append((t, lat, lon))
    if not rows:
        raise ValueError(f"no GPS rows in {gps_txt}")
    t0 = rows[0][0] if t0_abs is None else float(t0_abs)
    return [GpsFix(t - t0, la, lo) for (t, la, lo) in rows]


# --- Boreas (Toronto) ------------------------------------------------------

BOREAS_FPS = 20.0  # forward camera is 20 Hz


def boreas_pose_track(applanix_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(t_seconds_from_first_frame[N], latlon[N,2])`` per camera frame.

    ``camera_poses.csv`` has one ENU pose per image (GPSTime in microseconds
    == the PNG filename / video frame order); ``gps_post_process.csv`` gives
    the WGS84 origin (its lat/lon columns are in RADIANS). We convert each
    frame's ENU (easting, northing) to lat/lon about that origin, so the GT
    is aligned frame-for-frame with the camera video. The origin row must
    itself sit at ENU (0,0) — guarded, because a sequence whose reference
    origin is defined elsewhere would silently shift the whole track.
    """
    import csv
    import math
    ad = Path(applanix_dir)
    with open(ad / "gps_post_process.csv") as fh:
        r0 = next(csv.DictReader(fh))
    if abs(float(r0["easting"])) > 1.0 or abs(float(r0["northing"])) > 1.0:
        raise ValueError(
            f"gps_post_process.csv row 0 is not the ENU origin "
            f"(easting={r0['easting']}, northing={r0['northing']}); this "
            f"sequence's reference frame is defined elsewhere — refusing to "
            f"emit shifted ground truth")
    origin_lat = math.degrees(float(r0["latitude"]))
    origin_lon = math.degrees(float(r0["longitude"]))
    ts, easts, norths = [], [], []
    with open(ad / "camera_poses.csv") as fh:
        for row in csv.DictReader(fh):
            ts.append(float(row["GPSTime"]))
            easts.append(float(row["easting"]))
            norths.append(float(row["northing"]))
    if not ts:
        raise ValueError(f"no camera poses in {ad/'camera_poses.csv'}")
    latlon = enu_to_latlon(np.array(easts), np.array(norths), origin_lat, origin_lon)
    tp = (np.asarray(ts, float) - ts[0]) / 1e6
    return tp, latlon


def boreas_track(applanix_dir: Path) -> list[GpsFix]:
    """:func:`boreas_pose_track` as the project's GpsFix list."""
    tp, latlon = boreas_pose_track(applanix_dir)
    return [GpsFix(float(tp[i]), float(latlon[i, 0]), float(latlon[i, 1]))
            for i in range(len(tp))]


# --- Brno Urban (Czechia) --------------------------------------------------

BRNO_FPS = 10.0  # RGB cameras are 10 Hz


def brno_track(gnss_pose_txt: Path, *, ts_scale: float = 1e9) -> list[GpsFix]:
    """Read Brno ``gnss/pose.txt`` (RTK, WGS84 degrees) into a GpsFix track.

    Columns: ``system_timestamp latitude longitude altitude heading...``.
    ``ts_scale`` converts the integer system timestamp to seconds (Brno's
    clock is nanoseconds → 1e9); verified/overridden against the real file.
    """
    rows = []
    for line in Path(gnss_pose_txt).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line[0] in "#%":
            continue
        p = line.replace(",", " ").split()
        if len(p) < 3:
            continue
        try:
            t = float(p[0]); lat = float(p[1]); lon = float(p[2])
        except ValueError:
            continue
        if abs(lat) > 90 or abs(lon) > 180:
            continue
        rows.append((t, lat, lon))
    if not rows:
        raise ValueError(f"no GNSS rows in {gnss_pose_txt}")
    t0 = rows[0][0]
    return [GpsFix((t - t0) / ts_scale, la, lo) for (t, la, lo) in rows]
