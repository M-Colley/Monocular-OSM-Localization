"""Turn a comma2k19 route into a pipeline-ready clip + ground truth.

    python scripts/make_comma_gt.py path/to/<route_id>/<datetime> \
        --segments 0-9 \
        --out-video data/comma/route.mp4 \
        --out-gt ground_truth/comma_route.json

Reads the INS/GNSS/Vision global pose (ECEF) of the chosen consecutive
segments into our ground_truth schema and transcodes their video.hevc
into one mp4 — then prints a ready-to-run main.py command with the
pose-derived --osm-around region prior. Concatenating several 1-minute
segments is what gives a highway clip enough trajectory shape to match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.comma2k19 import (  # noqa: E402
    comma_ground_truth,
    load_route_track,
    osm_around_for_track,
    render_route_to_video,
)


def _parse_segments(spec: str | None, route_dir: Path) -> list[Path]:
    """Resolve a segment spec ('0-9', '0,3,4', or None=all) to seg dirs."""
    subdirs = sorted(
        (p for p in route_dir.iterdir() if p.is_dir() and (p / "global_pose").exists()),
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )
    if spec is None:
        return subdirs
    by_name = {p.name: p for p in subdirs}
    wanted: list[str] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            wanted.extend(str(i) for i in range(int(a), int(b) + 1))
        else:
            wanted.append(part.strip())
    segs = [by_name[w] for w in wanted if w in by_name]
    if not segs:
        raise SystemExit(f"no matching segments for '{spec}' under {route_dir}")
    return segs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("route_dir", type=Path, help="a comma2k19 route directory")
    p.add_argument("--segments", default=None,
                   help="segments to use: '0-9', '0,3,4', or omit for all")
    p.add_argument("--out-video", type=Path, required=True)
    p.add_argument("--out-gt", type=Path, required=True)
    p.add_argument("--waypoints", type=int, default=12)
    p.add_argument("--skip-video", action="store_true")
    args = p.parse_args()

    segs = _parse_segments(args.segments, args.route_dir)
    print(f"Using {len(segs)} segment(s): {[s.name for s in segs]}")

    fixes = load_route_track(segs)
    dur = fixes[-1].t_sec - fixes[0].t_sec
    print(f"Pose: {len(fixes)} frames over {dur:.1f}s "
          f"(lat[{min(f.lat for f in fixes):.5f},{max(f.lat for f in fixes):.5f}] "
          f"lon[{min(f.lon for f in fixes):.5f},{max(f.lon for f in fixes):.5f}])")

    gt = comma_ground_truth(segs, n_waypoints=args.waypoints)
    args.out_gt.parent.mkdir(parents=True, exist_ok=True)
    args.out_gt.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> wrote {args.out_gt} ({len(gt['waypoints'])} waypoints)")

    if not args.skip_video:
        print(f"Transcoding {len(segs)} video.hevc -> {args.out_video} ...")
        render_route_to_video(segs, args.out_video)
        print(f"  -> wrote {args.out_video}")

    clat, clon, radius = osm_around_for_track(fixes)
    print("\nRun the pipeline with:")
    print(
        f'  python main.py --video "{args.out_video}" '
        f'--city "San Francisco, California, USA" \\\n'
        f'      --osm-around {clat:.6f},{clon:.6f},{int(radius)} \\\n'
        f'      --vo-segment 0:{int(dur) + 1} \\\n'
        f'      --ground-truth-waypoints "{args.out_gt}" --scale-lock'
    )


if __name__ == "__main__":
    main()
