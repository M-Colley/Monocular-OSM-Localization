"""CLI entrypoint for the video → place mapping PoC.

    python main.py                         # use the default Ulm dashcam clip
    python main.py --url ... --city ...    # any other ego-driving clip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure stdout/stderr use UTF-8 on Windows (avoids cp1252 crash on emoji/umlauts)
# and line-buffering so every print() is immediately visible in log files.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src.city_inference import guess_city_from_title, slugify_submission
from src.download import DownloadError, fetch_video_metadata
from src.pipeline import PipelineConfig, run_pipeline


DEFAULT_URL = "https://www.youtube.com/watch?v=ULl8s4qydrk"
DEFAULT_CITY = "Ulm, Germany"


def _parse_segment(s: str) -> tuple[float, float | None]:
    a, _, b = s.partition(":")
    start = float(a) if a else 0.0
    end = float(b) if b else None
    return start, end


def _resolve_city(explicit_city: str | None, url: str, title: str | None) -> str:
    if explicit_city:
        return explicit_city
    guessed = guess_city_from_title(title)
    if guessed:
        return guessed
    if url == DEFAULT_URL:
        return DEFAULT_CITY
    raise ValueError(
        "Could not infer a city from the video title; pass --city explicitly."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", nargs="+", default=[DEFAULT_URL], help="One or more YouTube URLs")
    p.add_argument("--city", default=None, help="OSM place name for candidate graph")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--max-frames", type=int, default=4200)
    p.add_argument("--frame-stride", type=int, default=3)
    p.add_argument(
        "--vo-segment",
        default="0:420",
        help="Seconds 'start:end' of video to use for VO (default 0:420 = 7 min). "
             "~400s is the sweet spot before monocular VO drift degrades shape quality; "
             "a straight-line trajectory has no shape and cannot be localized.",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--estimated-length-m",
        type=float,
        default=8000.0,
        help="Approximate driven distance in meters; tunes OSM walk enumeration. "
             "Default 8000 generates long walks that span approach roads + city center.",
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
    p.add_argument("--full-splat", action="store_true",
                   help="render the splat (sparse and/or DA3) as anisotropic "
                        "alpha-blended Gaussians instead of isotropic disks. "
                        "Pure CPU; adds a few seconds. Produces *_topdown_hq.png.")
    p.add_argument("--full-splat-scale", type=float, default=1.4,
                   help="size multiplier for anisotropic Gaussians (raise for sparse clouds)")
    p.add_argument("--full-splat-opacity", type=float, default=0.55,
                   help="per-Gaussian opacity in the anisotropic top-down render")
    p.add_argument("--train-3dgs", action="store_true",
                   help="run a full gradient-descent 3DGS fit on top of the "
                        "DA3 reconstruction (requires --use-da3, CUDA, and "
                        "`pip install gsplat`). Produces splat_3dgs.ply.")
    p.add_argument("--train-3dgs-iters", type=int, default=2000,
                   help="number of optimization iterations for --train-3dgs")
    p.add_argument("--enable-ipm", action="store_true",
                   help="render an inverse-perspective-mapped road-plane BEV")
    p.add_argument("--ipm-height", type=float, default=1.4,
                   help="dashcam height above road (meters)")
    p.add_argument("--ipm-pitch", type=float, default=6.0,
                   help="dashcam downward tilt (degrees)")
    p.add_argument("--enable-sliding-window", action="store_true",
                   help="re-score full-route candidates by support across trajectory windows")
    p.add_argument("--sliding-window-size", type=int, default=64,
                   help="window size in resampled trajectory points for sliding-window matching")
    p.add_argument("--sliding-window-step", type=int, default=32,
                   help="step size in resampled trajectory points for sliding-window matching")
    p.add_argument("--embedding-sources", nargs="*", choices=("osm", "geotessera"), default=[],
                   help="optional deep embedding retrieval sources to compare against the top-down query")
    p.add_argument("--embedding-model", default="resnet18",
                   help="deep image embedding model used for retrieval (default: resnet18)")
    p.add_argument("--geotessera-year", type=int, default=2024,
                   help="GeoTessera embedding year when geotessera retrieval is enabled")
    p.add_argument("--ground-truth", nargs="*", default=[],
                   help="known street names traversed by the video (e.g. Neutorstrasse Olgastrasse)")
    p.add_argument("--vo-workers", type=int, default=None,
                   help="Threads for parallel VO pose estimation. Defaults to "
                        "min(cpu_count, 12). Pass 1 to force sequential.")
    p.add_argument("--enable-bev-splat", action="store_true",
                   help="run the BevSplat (NeurIPS'26) cross-view localization channel "
                        "as an additional aerial matcher. Requires the upstream package "
                        "+ weights from https://github.com/wangqww/BevSplat (not yet released); "
                        "without them the channel renders the satellite tiles only.")
    p.add_argument("--bev-splat-weights", type=Path, default=None,
                   help="Path to BevSplat checkpoint (.pth). Weights live at the "
                        "authors' OneDrive share — see README BevSplat section.")
    p.add_argument("--bev-splat-repo-path", type=Path, default=None,
                   help="Path to a local clone of https://github.com/wangqww/BevSplat "
                        "with its CUDA extensions built (pano_feature_gaussian/ and "
                        "feature_gaussian/ both pip install -e .).")
    p.add_argument("--bev-splat-model-module", default="models.models_kitti_nips",
                   help="Which Python module inside the BevSplat repo provides the "
                        "`Model` class. Default models.models_kitti_nips matches the "
                        "KITTI_*.pth checkpoints; use models.models_kitti_vfa for an "
                        "extension-free import smoke test.")
    p.add_argument("--bev-splat-source", choices=("geotessera", "osm"), default="geotessera",
                   help="Satellite tile source for BevSplat. Defaults to geotessera "
                        "(real satellite-derived embedding); 'osm' uses the schematic "
                        "raster (domain-mismatched but offline).")
    p.add_argument("--bev-splat-tile-size", type=int, default=512,
                   help="BevSplat satellite tile side length in pixels (KITTI default: 512).")
    p.add_argument("--bev-splat-half-extent-m", type=float, default=60.0,
                   help="BevSplat satellite tile half-side in metres.")
    return p


def main() -> None:
    p = build_arg_parser()
    args = p.parse_args()

    start, end = _parse_segment(args.vo_segment)
    results = []
    metadata_errors: list[str] = []

    for url in args.url:
        try:
            metadata = fetch_video_metadata(url)
        except DownloadError as e:
            metadata_errors.append(f"{url}: {e}")
            continue

        city = _resolve_city(args.city, metadata.url, metadata.title)
        submission_slug = slugify_submission(
            metadata.video_id,
            metadata.title,
            city,
            fallback_seed=metadata.url,
        )

        cfg = PipelineConfig(
            url=metadata.url,
            city=city,
            data_dir=args.data_dir / submission_slug,
            output_dir=args.output_dir / submission_slug,
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
            enable_full_splat=args.full_splat,
            full_splat_scale=args.full_splat_scale,
            full_splat_opacity=args.full_splat_opacity,
            enable_train_3dgs=args.train_3dgs,
            train_3dgs_iters=args.train_3dgs_iters,
            enable_ipm=args.enable_ipm,
            ipm_camera_height_m=args.ipm_height,
            ipm_pitch_deg=args.ipm_pitch,
            enable_sliding_window=args.enable_sliding_window,
            sliding_window_size=args.sliding_window_size,
            sliding_window_step=args.sliding_window_step,
            embedding_sources=tuple(args.embedding_sources),
            embedding_model=args.embedding_model,
            geotessera_year=args.geotessera_year,
            ground_truth_streets=tuple(args.ground_truth),
            enable_bev_splat=args.enable_bev_splat,
            bev_splat_weights=args.bev_splat_weights,
            bev_splat_repo_path=args.bev_splat_repo_path,
            bev_splat_model_module=args.bev_splat_model_module,
            bev_splat_source=args.bev_splat_source,
            bev_splat_tile_size=args.bev_splat_tile_size,
            bev_splat_half_extent_m=args.bev_splat_half_extent_m,
            vo_workers=args.vo_workers,
        )
        print(f"\n=== Submission: {metadata.title or metadata.url} ===")
        print(f"    city={city!r}  data_dir={cfg.data_dir}  output_dir={cfg.output_dir}")
        result = run_pipeline(cfg)
        result["video_title"] = metadata.title
        result["submission_slug"] = submission_slug
        results.append(result)

    if metadata_errors:
        raise SystemExit(
            "Failed to inspect one or more videos:\n- " + "\n- ".join(metadata_errors)
        )

    if len(results) > 1:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        batch_result = args.output_dir / "batch_results.json"
        with batch_result.open("w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"\nWrote batch summary to {batch_result}")


if __name__ == "__main__":
    main()
