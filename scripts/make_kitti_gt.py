"""Turn a KITTI raw synced drive into a pipeline-ready clip + ground truth.

    python scripts/make_kitti_gt.py path/to/2011_09_26_drive_0009_sync \
        --out-video data/kitti/drive_0009.mp4 \
        --out-gt ground_truth/kitti_drive_0009.json

Reads the OXTS global lat/lon track into our ground_truth schema and
renders the forward colour camera (image_02) into an mp4 — then prints a
ready-to-run ``main.py`` command (with the OXTS-derived ``--osm-around``
region prior). This is how a KITTI drive becomes a third validation clip
with real geographic ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kitti_raw import (  # noqa: E402
    kitti_ground_truth,
    load_oxts_track,
    osm_around_for_track,
    render_images_to_video,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("drive_dir", type=Path, help="synced KITTI drive directory")
    p.add_argument("--out-video", type=Path, required=True)
    p.add_argument("--out-gt", type=Path, required=True)
    p.add_argument("--waypoints", type=int, default=12)
    p.add_argument("--skip-video", action="store_true",
                   help="only (re)build the ground truth, reuse existing mp4")
    args = p.parse_args()

    fixes = load_oxts_track(args.drive_dir)
    dur = fixes[-1].t_sec - fixes[0].t_sec
    print(f"OXTS: {len(fixes)} fixes over {dur:.1f}s "
          f"(lat[{min(f.lat for f in fixes):.5f},{max(f.lat for f in fixes):.5f}] "
          f"lon[{min(f.lon for f in fixes):.5f},{max(f.lon for f in fixes):.5f}])")

    gt = kitti_ground_truth(args.drive_dir, n_waypoints=args.waypoints)
    args.out_gt.parent.mkdir(parents=True, exist_ok=True)
    args.out_gt.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> wrote {args.out_gt} ({len(gt['waypoints'])} waypoints)")

    if not args.skip_video:
        print(f"Encoding image_02 -> {args.out_video} ...")
        render_images_to_video(args.drive_dir, args.out_video)
        print(f"  -> wrote {args.out_video}")

    clat, clon, radius = osm_around_for_track(fixes)
    print("\nRun the pipeline with:")
    print(
        f'  python main.py --video "{args.out_video}" --city "Karlsruhe, Germany" \\\n'
        f'      --osm-around {clat:.6f},{clon:.6f},{int(radius)} \\\n'
        f'      --ground-truth-waypoints "{args.out_gt}" --scale-lock'
    )


if __name__ == "__main__":
    main()
