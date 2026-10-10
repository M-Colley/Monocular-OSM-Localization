"""Fetch only what the pipeline needs from a KITTI raw synced drive.

The public archives bundle four cameras and the Velodyne scans
(drive 0033: 6.6 GB). The pipeline reads the forward colour camera
(``image_02``) and the OXTS GPS/IMU; the camera intrinsics come from the
per-date calibration archive. ``remotezip`` reads the archive's central
directory with HTTP range requests and downloads just those members — about
a fifth of the bytes.

    python scripts/fetch_kitti_raw.py 2011_09_26_drive_0009 2011_09_30_drive_0033
    python scripts/fetch_kitti_raw.py ... --render   # also write data/kitti/drive_NNNN.mp4

Output layout (what ``monocular_osm.kitti_raw`` expects):

    data/kitti/raw/<date>/<drive>_sync/image_02/...
    data/kitti/raw/<date>/<drive>_sync/oxts/...
    data/kitti/raw/<date>/calib_cam_to_cam.txt ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
WANTED_DIRS = ("image_02/", "oxts/")


def _extract(url: str, out_root: Path, keep) -> int:
    n = 0
    with RemoteZip(url) as z:
        members = [m for m in z.infolist() if not m.is_dir() and keep(m.filename)]
        total = sum(m.file_size for m in members)
        print(f"  {url.rsplit('/', 1)[-1]}: {len(members)} files, "
              f"{total / 1e6:.0f} MB", flush=True)
        for i, m in enumerate(members, 1):
            target = out_root / m.filename
            if target.exists() and target.stat().st_size == m.file_size:
                continue
            z.extract(m, out_root)
            n += 1
            if i % 200 == 0:
                print(f"    {i}/{len(members)}", flush=True)
    return n


def fetch_drive(drive: str, raw_root: Path) -> Path:
    date = drive[:10]
    _extract(f"{BASE_URL}/{date}_calib.zip", raw_root, lambda f: f.endswith(".txt"))
    _extract(f"{BASE_URL}/{drive}/{drive}_sync.zip", raw_root,
             lambda f: any(f"/{d}" in f for d in WANTED_DIRS))
    return raw_root / date / f"{drive}_sync"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("drives", nargs="+", help="e.g. 2011_09_26_drive_0009")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "kitti" / "raw")
    p.add_argument("--render", action="store_true",
                   help="also render image_02 to data/kitti/drive_<NNNN>.mp4")
    args = p.parse_args(argv)
    for drive in args.drives:
        print(f"[kitti] {drive}", flush=True)
        drive_dir = fetch_drive(drive, args.out)
        if args.render:
            sys.path.insert(0, str(ROOT))
            from monocular_osm.kitti_raw import render_images_to_video

            video = args.out.parent / f"drive_{drive[-4:]}.mp4"
            render_images_to_video(drive_dir, video)
            print(f"  -> {video}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
