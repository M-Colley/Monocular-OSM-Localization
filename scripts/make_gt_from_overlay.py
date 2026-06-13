"""Turn a dashcam clip with a burned-in GPS overlay into a ground-truth file.

    python scripts/make_gt_from_overlay.py VIDEO --city "London, UK" \
        --out ground_truth/<name>.json

OCRs the overlay band of sampled frames, parses the coordinates, sanity-
filters the track, and writes the project's standard ground_truth schema —
so the clip can be evaluated with ``main.py --ground-truth-waypoints``.
This is how we scale validation: every overlay clip becomes free GT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gps_overlay import extract_gps_track, track_to_ground_truth  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path, help="local video file with a GPS overlay")
    p.add_argument("--city", required=True, help="city/region, e.g. 'London, UK'")
    p.add_argument("--out", type=Path, required=True, help="output ground_truth JSON")
    p.add_argument("--video-id", default=None)
    p.add_argument("--video-url", default="")
    p.add_argument("--interval", type=float, default=2.0,
                   help="seconds between sampled frames (default 2)")
    p.add_argument("--region", choices=("bottom", "top", "full"), default="bottom",
                   help="where the overlay sits in the frame (default bottom)")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--waypoints", type=int, default=10)
    args = p.parse_args()

    print(f"OCR-ing GPS overlay from {args.video} (region={args.region}, "
          f"every {args.interval}s)...")
    track = extract_gps_track(
        args.video, sample_interval_sec=args.interval, region=args.region,
        start_sec=args.start, end_sec=args.end,
    )
    print(f"  -> {len(track)} GPS fixes recovered")
    if not track:
        raise SystemExit(
            "No coordinates parsed. Check --region (overlay band), the clip "
            "actually has a burned-in lat/lon overlay, and resolution is high "
            "enough for OCR.")
    lats = [f.lat for f in track]
    lons = [f.lon for f in track]
    print(f"  -> bbox lat[{min(lats):.5f},{max(lats):.5f}] "
          f"lon[{min(lons):.5f},{max(lons):.5f}]")
    gt = track_to_ground_truth(
        track,
        video_id=args.video_id or args.video.stem,
        video_url=args.video_url,
        city=args.city,
        n_waypoints=args.waypoints,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> wrote {args.out} ({len(gt['waypoints'])} waypoints)")


if __name__ == "__main__":
    main()
