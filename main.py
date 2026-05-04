"""CLI entrypoint for the video → place mapping PoC.

    python main.py                         # use the default Ulm dashcam clip
    python main.py --url ... --city ...    # any other ego-driving clip
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import PipelineConfig, run_pipeline


DEFAULT_URL = "https://www.youtube.com/watch?v=ULl8s4qydrk"
DEFAULT_CITY = "Ulm, Germany"


def _parse_segment(s: str) -> tuple[float, float | None]:
    a, _, b = s.partition(":")
    start = float(a) if a else 0.0
    end = float(b) if b else None
    return start, end


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=DEFAULT_URL, help="YouTube URL")
    p.add_argument("--city", default=DEFAULT_CITY, help="OSM place name for candidate graph")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--max-frames", type=int, default=1500)
    p.add_argument("--frame-stride", type=int, default=6)
    p.add_argument(
        "--vo-segment",
        default="0:300",
        help="Seconds 'start:end' of video to use for VO (default 0:300). "
             "Pick a window long enough to contain at least one real turn; "
             "a straight-line trajectory has no shape and cannot be localized.",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--estimated-length-m",
        type=float,
        default=4000.0,
        help="Approximate driven distance in meters; tunes OSM walk enumeration.",
    )
    p.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Subsample candidate start nodes (>=2 speeds up at cost of recall).",
    )
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--no-splat", action="store_true",
                   help="skip building/rendering the sparse splat point cloud")
    p.add_argument("--no-aerial", action="store_true",
                   help="skip the OSM-patch ORB feature match channel")
    p.add_argument("--splat-max-pairs", type=int, default=80,
                   help="cap on frame pairs used to triangulate the splat")
    p.add_argument("--use-da3", action="store_true",
                   help="run Depth Anything 3 (CUDA) for a dense reconstruction")
    p.add_argument("--da3-keyframes", type=int, default=32,
                   help="number of keyframes to feed DA3 (must fit in GPU memory)")
    p.add_argument("--enable-ipm", action="store_true",
                   help="render an inverse-perspective-mapped road-plane BEV")
    p.add_argument("--ipm-height", type=float, default=1.4,
                   help="dashcam height above road (meters)")
    p.add_argument("--ipm-pitch", type=float, default=6.0,
                   help="dashcam downward tilt (degrees)")
    p.add_argument("--ground-truth", nargs="*", default=[],
                   help="known street names traversed by the video (e.g. Neutorstrasse Olgastrasse)")
    args = p.parse_args()

    start, end = _parse_segment(args.vo_segment)
    cfg = PipelineConfig(
        url=args.url,
        city=args.city,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        vo_start_sec=start,
        vo_end_sec=end,
        top_k=args.top_k,
        estimated_length_m=args.estimated_length_m,
        skip_download=args.skip_download,
        sample_every=args.sample_every,
        enable_splat=not args.no_splat,
        enable_aerial_match=not args.no_aerial,
        splat_max_pairs=args.splat_max_pairs,
        enable_da3=args.use_da3,
        da3_keyframes=args.da3_keyframes,
        enable_ipm=args.enable_ipm,
        ipm_camera_height_m=args.ipm_height,
        ipm_pitch_deg=args.ipm_pitch,
        ground_truth_streets=tuple(args.ground_truth),
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
