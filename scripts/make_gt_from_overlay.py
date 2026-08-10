"""Turn a dashcam clip with a burned-in GPS overlay into a ground-truth file.

    python scripts/make_gt_from_overlay.py VIDEO --city "London, UK" \
        --out ground_truth/<name>.json
    python scripts/make_gt_from_overlay.py --url https://youtu.be/<id> \
        --section 0:600 --out ground_truth/<name>.json

OCRs the overlay band of sampled frames, parses the coordinates, sanity-
filters the track, and writes the project's standard ground_truth schema —
so the clip can be evaluated with ``main.py --ground-truth-waypoints``.
This is how we scale validation: every overlay clip becomes free GT.

Without ``--city`` the midpoint of the extracted track is reverse-geocoded,
so a URL plus an output path is enough for a fully unattended run. For a
whole fleet of clips at once, see ``scripts/build_overlay_fleet.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monocular_osm.city_inference import slugify_submission  # noqa: E402
from monocular_osm.download import cached_video_metadata, download_video  # noqa: E402
from monocular_osm.gps_overlay import (  # noqa: E402
    city_for_track,
    extract_gps_track,
    track_to_ground_truth,
)

ROOT = Path(__file__).resolve().parents[1]


def _parse_section(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    start, _, end = text.partition(":")
    return float(start), float(end)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("video", nargs="?", type=Path,
                     help="local video file with a GPS overlay")
    src.add_argument("--url", help="YouTube URL to download instead")
    p.add_argument("--city", default=None,
                   help="city/region, e.g. 'London, UK'. Omit to reverse-geocode "
                        "the extracted track.")
    p.add_argument("--out", type=Path, required=True, help="output ground_truth JSON")
    p.add_argument("--video-id", default=None)
    p.add_argument("--video-url", default="")
    p.add_argument("--section", default=None, metavar="START:END",
                   help="with --url, download only these seconds (e.g. 0:600)")
    p.add_argument("--data-dir", default="data", help="where --url downloads land")
    p.add_argument("--max-height", type=int, default=1080)
    p.add_argument("--interval", type=float, default=2.0,
                   help="seconds between sampled frames (default 2)")
    p.add_argument("--region", choices=("bottom", "top", "full"), default="bottom",
                   help="where the overlay sits in the frame (default bottom)")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--waypoints", type=int, default=10)
    args = p.parse_args()

    video, video_id, video_url = args.video, args.video_id, args.video_url
    if args.url:
        meta = cached_video_metadata(args.url, Path(args.data_dir))
        section = _parse_section(args.section)
        slug = slugify_submission(meta.video_id, meta.title, fallback_seed=args.url)
        stem = "input" if section is None else f"input_{int(section[0])}-{int(section[1])}"
        print(f"Downloading {meta.title!r} -> {args.data_dir}/{slug}/{stem}.*")
        video = download_video(args.url, Path(args.data_dir) / slug,
                               filename_stem=stem, max_height=args.max_height,
                               section=section)
        video_id = video_id or meta.video_id
        video_url = video_url or meta.url

    print(f"OCR-ing GPS overlay from {video} (region={args.region}, "
          f"every {args.interval}s)...")
    track = extract_gps_track(
        video, sample_interval_sec=args.interval, region=args.region,
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

    city = args.city or city_for_track(
        track, cache_path=ROOT / "data" / "reverse_geocode_cache.json")
    if not city:
        raise SystemExit("Reverse geocoding failed; pass --city explicitly.")
    print(f"  -> city {city}")

    gt = track_to_ground_truth(
        track,
        video_id=video_id or Path(video).stem,
        video_url=video_url,
        city=city,
        n_waypoints=args.waypoints,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> wrote {args.out} ({len(gt['waypoints'])} waypoints)")


if __name__ == "__main__":
    main()
