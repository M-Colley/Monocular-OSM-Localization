"""CLI entrypoint: video + city -> WGS84 position of the video.

    python main.py --video clip.mp4 --city "Ulm, Germany"   # local video file
    python main.py --url ... --city ...                     # YouTube clip
    python main.py                                          # default Ulm demo clip

The estimated position (lat/lon, route, street names, map links) is
printed at the end of the run and written to output/<slug>/result.json
under the "position" key.
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
from src.download import (
    DownloadError,
    VideoMetadata,
    fetch_video_metadata,
    local_video_metadata,
)
from src.pipeline import PipelineConfig, run_pipeline


DEFAULT_URL = "https://www.youtube.com/watch?v=ULl8s4qydrk"
DEFAULT_CITY = "Ulm, Germany"


def _parse_segment(s: str) -> tuple[float, float | None]:
    a, _, b = s.partition(":")
    start = float(a) if a else 0.0
    end = float(b) if b else None
    return start, end


# Frame budget that auto-stride aims for. ~4800 frames of 720p BGR is
# ~13 GB resident — comfortable on this 64 GB machine, and a frame count
# the VO stage is known to handle. Auto-stride scales the temporal
# sampling with the analyzed duration instead of silently truncating it.
_TARGET_FRAME_BUDGET = 4800
_NOMINAL_FPS = 30.0


def _auto_frame_stride(duration_sec: float | None, fps: float | None = None) -> int:
    """Pick a frame stride that keeps ~_TARGET_FRAME_BUDGET frames.

    7 min @30fps -> 3 (the historical default), 10 min -> 4, 15 min -> 6.
    ``fps`` is the source's real frame rate when known — a 60 fps upload
    at the nominal-30 assumption would get double the intended frame
    budget (~26 GB resident at 720p) and a 2x-denser VO baseline than
    the tuned default. Open-ended segments (duration None) fall back to
    the historical 3. Never below 3: smaller strides give a too-short VO
    baseline at urban speed and triple memory for no shape benefit.
    """
    if duration_sec is None or duration_sec <= 0:
        return 3
    if fps is None or fps <= 0:
        fps = _NOMINAL_FPS
    import math
    return max(3, math.ceil(duration_sec * fps / _TARGET_FRAME_BUDGET))


def _probe_video_fps(path: Path) -> float | None:
    """The container's frame rate, or None when unreadable."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
        finally:
            cap.release()
        return fps if fps > 0 else None
    except Exception:
        return None


# --- Metadata cache: lets --skip-download re-runs work fully offline ------
# fetch_video_metadata is a yt-dlp network call; without this cache an
# offline re-run of a fully cached clip died before touching the cached
# video. Keyed on the exact submitted URL, stored under the data dir.


def _metadata_cache_path(data_dir: Path, url: str) -> Path:
    from hashlib import sha256
    digest = sha256(url.encode("utf-8")).hexdigest()[:16]
    return data_dir / "metadata_cache" / f"{digest}.json"


def _load_cached_metadata(data_dir: Path, url: str) -> VideoMetadata | None:
    path = _metadata_cache_path(data_dir, url)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return VideoMetadata(
        url=d.get("url") or url,
        title=d.get("title"),
        video_id=d.get("video_id"),
    )


def _write_cached_metadata(data_dir: Path, url: str, metadata: VideoMetadata) -> None:
    path = _metadata_cache_path(data_dir, url)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "input_url": url,
                "url": metadata.url,
                "title": metadata.title,
                "video_id": metadata.video_id,
            }, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass  # cache is best-effort; never fail the run over it


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


def _validate_input_args(
    videos: list[Path] | None,
    urls: list[str] | None,
    city: str | None,
) -> None:
    """Reject invalid --video / --url / --city combinations.

    Local files have no title metadata worth trusting for city
    inference, so --video makes --city mandatory: the contract is
    "video + city in, position out".
    """
    if videos and urls:
        raise ValueError("Pass either --video or --url, not both.")
    if videos and not city:
        raise ValueError(
            "--city is required when using --video "
            "(e.g. --city 'Ulm, Germany')."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", nargs="+", type=Path, default=None,
                   help="One or more local video files (mutually exclusive with "
                        "--url; requires --city)")
    p.add_argument("--url", nargs="+", default=None,
                   help="One or more YouTube URLs (default: the Ulm demo clip)")
    p.add_argument("--city", default=None,
                   help="City the video was filmed in, as 'City, Country' "
                        "(e.g. 'Ulm, Germany'). Required with --video; inferred "
                        "from the video title for URLs when omitted.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap on frames sampled from the video. Default: no cap — "
                        "the analyzed segment bounds the count.")
    p.add_argument("--frame-stride", type=int, default=None,
                   help="Take every Nth frame. Default: auto — picked so the "
                        "analyzed segment yields ~4800 frames (7 min -> 3, "
                        "10 min -> 4, 15 min -> 6).")
    p.add_argument(
        "--analyze-minutes",
        type=float,
        default=None,
        help="Analyze the first N minutes of the video (shorthand for "
             "--vo-segment 0:N*60; takes precedence over it). Longer windows "
             "cover more turns but accumulate more monocular VO drift.",
    )
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
        default=None,
        help="Approximate driven distance in meters; tunes OSM walk enumeration. "
             "Default: derived from the analyzed segment duration at urban "
             "average speed (~20 km/h). A prior far from the true route length "
             "badly distorts the shape match — only override when you know "
             "the actual distance.",
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
    p.add_argument("--use-da3-trajectory", action="store_true",
                   help="use Depth Anything 3's globally-consistent camera path as the "
                        "trajectory fed to the shape matcher, instead of monocular VO "
                        "(needs CUDA). DA3 is metric and multi-frame-consistent, so it "
                        "has far less accumulated drift than frame-to-frame VO — the "
                        "dominant error source on long clips. VO is still used for the "
                        "splat/IPM renders.")
    p.add_argument("--da3-keyframes", type=int, default=32,
                   help="number of keyframes to feed DA3 (must fit in GPU memory)")
    p.add_argument("--use-mapanything-trajectory", action="store_true",
                   help="EXPERIMENTAL (does NOT improve accuracy — see below). Use "
                        "MapAnything (3DV'26) as the matcher's trajectory via "
                        "submap-stitching (src/mapanything_trajectory.py): short "
                        "high-overlap windows reconstructed feed-forward then chained "
                        "by a scale-guarded Sim(3). It recovers the metric scale a "
                        "single pass collapses and fits the trajectory SHAPE better "
                        "than VO (219 vs 258 m global-fit RMS on Ulm), but end-to-end "
                        "it localized WORSE (final start err 1400 vs 664 m on the Ulm "
                        "GT clip): a lower-drift path doesn't fix candidate SELECTION, "
                        "which is the real bottleneck. Kept as a research toggle. Needs "
                        "the mapanything package + GPU + ~9GB weights; no-op if missing.")
    p.add_argument("--openvo-trajectory", type=Path, default=None,
                   help="path to a precomputed OpenVO (CVPR'26) trajectory (KITTI 3x4 "
                        "poses .txt) to feed the shape matcher instead of VO. OpenVO is "
                        "intrinsic-free metric dashcam VO with lower drift than our VO "
                        "(196 vs 241 m global-fit RMS on Ulm). Tests whether a better "
                        "trajectory improves end-to-end OSM selection.")
    p.add_argument("--vggt-long-trajectory", type=Path, default=None,
                   help="Use a VGGT-Long camera_poses.txt (flattened 4x4 C2W "
                        "rows; 12-col KITTI rows also accepted) as the matcher "
                        "trajectory. ~30%% lower shape RMS than OpenVO on KITTI "
                        "0033 (148.6 vs 211.0 m) but flag-gated: better shape "
                        "has not implied better end-to-end before "
                        "(MapAnything). Takes precedence over staged OpenVO.")
    p.add_argument("--no-openvo-default", action="store_true",
                   help="disable the default of auto-using a staged OpenVO trajectory "
                        "(<data_dir>/openvo_trajectory.txt) as the matcher input; force "
                        "the built-in monocular VO instead.")
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
    p.add_argument("--embedding-sources", nargs="*",
                   choices=("osm", "geotessera", "esri", "satellite"), default=[],
                   help="optional deep embedding retrieval sources to compare against the "
                        "top-down query. 'esri'/'satellite' = real RGB orthoimagery "
                        "(recommended for IPM↔satellite comparison).")
    p.add_argument("--embedding-model", default="resnet18",
                   help="deep image embedding model used for retrieval: 'resnet18' (offline, "
                        "ImageNet) or 'dinov2_vits14'/'dinov2_vitb14'/'dinov2_vitl14' "
                        "(cross-domain VPR backbone, downloads weights on first use).")
    p.add_argument("--geotessera-year", type=int, default=2024,
                   help="GeoTessera embedding year when geotessera retrieval is enabled")
    p.add_argument("--enable-ocr-anchor", action="store_true",
                   help="OCR scene text (signs/POIs) and geocode it into absolute "
                        "position anchors that seed enumeration and re-rank "
                        "candidates. The one channel that injects absolute "
                        "geographic info from the video. Needs easyocr + network "
                        "geocoding (both cached after first run).")
    p.add_argument("--ocr-super-res", action="store_true",
                   help="Upscale + sharpen frames before OCR (src/scene_text._upscale_sharpen) "
                        "to recover legible street/place names from low-res signage. On London "
                        "720p it ~doubled high-confidence detections and recovered geocodable "
                        "names (Holborn, Bloomsbury, Euston) the original missed. Pair with "
                        "--enable-ocr-anchor.")
    p.add_argument("--ocr-sample-interval-sec", type=float, default=6.0,
                   help="Seconds between frames sampled for OCR (default 6).")
    p.add_argument("--no-osm-gazetteer", dest="use_osm_gazetteer",
                   action="store_false",
                   help="Disable the local OSM gazetteer anchor source (on by "
                        "default): fuzzy-matches OCR text against named OSM "
                        "features (POIs, transit stops) in the graph area — "
                        "offline, free, additive. It recovered 2 sub-300 m "
                        "anchors on London where Nominatim found none.")
    p.set_defaults(use_osm_gazetteer=True)
    p.add_argument("--classify-signs", action="store_true",
                   help="Classify each OCR anchor's sign as here vs direction "
                        "(Gemma 4) and drop directional signs, which name "
                        "places elsewhere and geocode off-route (the London "
                        "'Holborn' failure). GPU; needs ~9.5 GB free VRAM.")
    p.add_argument("--ocr-min-confidence", type=float, default=0.5,
                   help="Min OCR confidence for a detection to be geocoded (default 0.5).")
    p.add_argument("--ocr-video", type=Path, default=None,
                   help="Separate (higher-res, e.g. 4K) video used for OCR only; "
                        "VO/matching stay on the main video. Lets a 4K source feed "
                        "street-plate OCR without re-running VO at 4K.")
    p.add_argument("--no-scale-recovery", action="store_true",
                   help="Disable anchor-based metric scale recovery / georeferencing "
                        "(ideas 1+2). On by default; auto-declines when anchors are "
                        "too sparse/noisy for a reliable fit.")
    p.add_argument("--use-ipm-scale", action="store_true",
                   help="Estimate route length from ground-plane optical flow (idea 3). "
                        "Off by default — needs camera calibration to be reliable.")
    p.add_argument("--scale-lock", action="store_true",
                   help="Lock the matcher's alignment scale to the metric length prior "
                        "instead of a free Procrustes scale, so the localized route "
                        "spans the true extent (fixes route compression / the far-end "
                        "tail error).")
    p.add_argument("--osm-around", default=None,
                   help="Bound the OSM graph to a disc 'lat,lon,radius_m' instead of "
                        "the whole named city. Required for mega-cities (e.g. London) "
                        "where fetching the full place is infeasible.")
    p.add_argument("--use-vpr-prior", action="store_true",
                   help="Blind coarse-location prior via Visual Place Recognition on "
                        "KartaView street imagery (open API, no token) + EigenPlaces "
                        "(src/kartaview_vpr.py). Shape-INDEPENDENT — the fix for the "
                        "selection wall; ~53 m prior on Ulm 4K vs ~530 m for chance. Used "
                        "as a re-rank centre + the anchor-primary placement prior; runs "
                        "with or without --osm-around. (Gating the OSM graph to the prior "
                        "was refuted by experiment, so there is no gate knob.) Needs "
                        "requests + EigenPlaces weights; no-op if unavailable.")
    p.add_argument("--vpr-search-radius", type=float, default=3000.0,
                   help="Radius (m) around the city centre to fetch VPR reference "
                        "photos for --use-vpr-prior (default 3000).")
    p.add_argument("--vpr-cap", type=int, default=1500,
                   help="Max VPR reference photos fetched+embedded per clip "
                        "(default 1500). The fetch uniform-subsamples to this "
                        "cap, so a low cap thins dense areas; raise it to "
                        "densify retrieval-bound starts (cold-cache refetch).")
    p.add_argument("--vpr-source", choices=["kartaview", "mapillary", "panoramax"],
                   default="kartaview",
                   help="VPR reference imagery source. 'kartaview' is open and "
                        "tokenless. 'mapillary' is much denser (needs a free "
                        "MLY_TOKEN env var) and gave a 3-31 m prior on every GT "
                        "clip, including London/comma/KITTI that KartaView could "
                        "not cover. 'panoramax' is the federated open network "
                        "(tokenless, 105M+ images 2026, strongest in EU) — a "
                        "coverage complement where Mapillary is thin.")
    p.add_argument("--vpr-two-pass", action="store_true",
                   help="Match at both the VO scale and a scale pinned to the "
                        "VPR track extent, keeping whichever candidates better "
                        "explain the full VPR track. Fixes candidate SHAPE "
                        "where the VO scale is wrong (London mean 355->76 m), "
                        "but the anchor placement's own scale retry + "
                        "orientation refine now reach the same headline "
                        "without the second matching pass (A/B 2026-07-04: "
                        "identical on London/Ulm/comma) — so off by default. "
                        "Needs --use-vpr-prior.")
    p.add_argument("--no-vpr-viterbi", action="store_true",
                   help="Disable the Viterbi sequence decode of the VPR track "
                        "(fall back to per-frame argmax retrieval). The decode "
                        "is on by default: it transformed the weak clips "
                        "(0009 mean 80->36 m, 0033 pins unlocked) for a small "
                        "London cost (start 31->51 m); fleet mean 109.6->90.2 m.")
    p.add_argument("--use-vpr-sequence", action="store_true",
                   help="EXPERIMENTAL: score candidates against the per-frame VPR track "
                        "(sequence-median distance at matched arc fractions) instead of "
                        "the centroid-only distance. Untested on GT; off by default.")
    p.add_argument("--use-plate-anchor", action="store_true",
                   help="Blind coarse-location prior from license-plate REGISTRATION "
                        "DISTRICTS (src/plate_anchor.py): read the EU plate region prefix "
                        "(UL, M, B...) across the clip, vote, geocode the modal district, "
                        "and re-rank candidates by proximity to it (a wrong-district "
                        "guard, not a hard gate). Shape-INDEPENDENT; 0.4 km from GT on "
                        "Ulm. Privacy: only the district code (shared by 1000s of cars) is "
                        "used, never full plates. Needs fast-alpr; no-op if no district "
                        "emerges.")
    p.add_argument("--use-vlm-anchor", action="store_true",
                   help="VLM (Gemma 4, multimodal) district/landmark prior (src/vlm_anchor.py): "
                        "reads frames -> infers district + reads street/shop/landmark names -> "
                        "geocodes -> feeds the anchor-primary path. A COVERAGE FALLBACK, used "
                        "only when VPR finds no references. Needs the gemma-4-E4B-it weights.")
    p.add_argument("--use-sun-heading", action="store_true",
                   help="Recover an ABSOLUTE camera heading from the sun (src/sun_heading.py) "
                        "to pin the matcher's free rotation. Activates only if the clip carries "
                        "a capture time (container metadata or a burned-in dashcam clock) AND "
                        "the sun is in view; otherwise a graceful no-op. The astronomy is exact; "
                        "needs pysolar+timezonefinder. Reported in result.json['sun_heading'].")
    p.add_argument("--ground-truth", nargs="*", default=[],
                   help="known street names traversed by the video (e.g. Neutorstrasse Olgastrasse)")
    p.add_argument("--ground-truth-waypoints", type=Path, default=None,
                   help="JSON file with timestamped GPS fixes along the true route "
                        "(see ground_truth/ for the schema). Enables metric "
                        "start/route error reporting per candidate.")
    p.add_argument("--vo-workers", type=int, default=None,
                   help="Threads for parallel VO pose estimation. Defaults to "
                        "min(cpu_count, 12). Pass 1 to force sequential.")
    p.add_argument("--enable-bev-splat", action="store_true",
                   help="run the BevSplat (NeurIPS'25) cross-view localization channel "
                        "as an additional aerial matcher. Requires a local clone of "
                        "https://github.com/wangqww/BevSplat with built CUDA extensions "
                        "plus the published checkpoints; without them the channel "
                        "renders the satellite tiles only.")
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
    p.add_argument("--bev-splat-source", choices=("esri", "satellite", "geotessera", "osm"),
                   default="esri",
                   help="Satellite tile source for BevSplat. Defaults to 'esri' "
                        "(real RGB orthoimagery — matches BevSplat's KITTI training "
                        "domain). 'geotessera' uses satellite-derived PCA false-colour "
                        "(non-discriminative across inner-city tiles); 'osm' uses the "
                        "schematic raster (domain-mismatched but offline).")
    p.add_argument("--bev-splat-tile-size", type=int, default=512,
                   help="BevSplat satellite tile side length in pixels (KITTI default: 512).")
    p.add_argument("--bev-splat-half-extent-m", type=float, default=60.0,
                   help="BevSplat satellite tile half-side in metres.")
    p.add_argument("--ipm-scale-height", type=float, default=1.4,
                   help="Camera height (m) for --use-ipm-scale metric scale (default 1.4).")
    p.add_argument("--ipm-scale-pitch", type=float, default=1.5,
                   help="Camera downward pitch (deg) for --use-ipm-scale. The sensitive "
                        "knob: ~1-2 deg for a near-horizontal dashcam (default 1.5).")
    p.add_argument("--enable-loop-closure", action="store_true",
                   help="Detect a route that returns near its start and redistribute "
                        "VO drift so the loop closes (src/loop_closure.py). Pair with "
                        "--use-ipm-scale; closing at a wrong scale doesn't help.")
    p.add_argument("--use-vggt-gating", action="store_true",
                   help="Run VGGT (feed-forward, drift-free poses) to gate enumeration "
                        "to the area its trajectory selects, then let the loop-closed VO "
                        "geometry pick within it. Needs the vggt package + GPU + ~5GB "
                        "weights; no-op if unavailable. Best with --enable-loop-closure "
                        "--use-ipm-scale.")
    p.add_argument("--vggt-keyframes", type=int, default=64,
                   help="Keyframes fed to VGGT (default 64; wider baselines = cleaner poses).")
    p.add_argument("--use-orienternet", action="store_true",
                   help="Refine the shape-matched position with OrienterNet (neural "
                        "BEV->OSM matching + sequential fusion) — the metric localization "
                        "head, ~2 m on KITTI. Needs third_party/OrienterNet + GPU + weights.")
    p.add_argument("--orienternet-keyframes", type=int, default=10)
    p.add_argument("--orienternet-tile-m", type=float, default=160.0,
                   help="OrienterNet OSM tile half-extent (m); size to the coarse error.")
    p.add_argument("--bev-fusion-cap", type=int, default=5,
                   help="How many top-by-geometry candidates the BevSplat "
                        "appearance rank may reorder (default 5). Raise it when "
                        "geometry ranks the true route deep (see bench_matching).")
    return p


def main() -> None:
    p = build_arg_parser()
    args = p.parse_args()
    try:
        _validate_input_args(args.video, args.url, args.city)
    except ValueError as e:
        p.error(str(e))

    osm_around = None
    if args.osm_around:
        try:
            la, lo, rad = (float(x) for x in args.osm_around.split(","))
            osm_around = (la, lo, rad)
        except ValueError:
            p.error("--osm-around must be 'lat,lon,radius_m' (e.g. 51.52,-0.13,2500)")

    if args.analyze_minutes is not None:
        start, end = 0.0, args.analyze_minutes * 60.0
    else:
        start, end = _parse_segment(args.vo_segment)
    duration = (end - start) if end is not None else None
    results = []
    metadata_errors: list[str] = []

    # Build the submission list from either local files or URLs. Each
    # entry is (metadata, local_path); local_path is None for URLs.
    submissions: list[tuple[VideoMetadata, Path | None]] = []
    if args.video:
        for path in args.video:
            try:
                submissions.append((local_video_metadata(path), Path(path)))
            except DownloadError as e:
                metadata_errors.append(f"{path}: {e}")
    else:
        for url in (args.url or [DEFAULT_URL]):
            # --skip-download promises an offline re-run: prefer cached
            # metadata over the yt-dlp network fetch when available.
            metadata = (
                _load_cached_metadata(args.data_dir, url)
                if args.skip_download else None
            )
            if metadata is not None:
                print(f"Using cached metadata for {url} (--skip-download)")
                submissions.append((metadata, None))
                continue
            try:
                metadata = fetch_video_metadata(url)
            except DownloadError as e:
                metadata_errors.append(f"{url}: {e}")
                continue
            _write_cached_metadata(args.data_dir, url, metadata)
            submissions.append((metadata, None))

    for metadata, local_path in submissions:
        if local_path is not None:
            city = args.city  # guaranteed by _validate_input_args
        else:
            city = _resolve_city(args.city, metadata.url, metadata.title)
        submission_slug = slugify_submission(
            metadata.video_id,
            metadata.title,
            city,
            fallback_seed=metadata.url,
        )

        # Auto stride uses the source's REAL fps when a local/cached file
        # can be probed (a 60 fps upload must not get double the frame
        # budget); falls back to the nominal 30.
        frame_stride = args.frame_stride
        if frame_stride is None:
            probe = local_path
            if probe is None:
                # Same ext-ranked pick the pipeline uses to choose the analyzed
                # file, so the probed fps matches the file actually analyzed
                # (a bare glob picks input.download.json first).
                from src.pipeline import rank_cached_inputs
                cached = rank_cached_inputs(
                    (args.data_dir / submission_slug).glob("input.*"))
                probe = cached[0] if cached else None
            fps = _probe_video_fps(probe) if probe is not None else None
            frame_stride = _auto_frame_stride(duration, fps=fps)

        cfg = PipelineConfig(
            url=metadata.url,
            city=city,
            data_dir=args.data_dir / submission_slug,
            output_dir=args.output_dir / submission_slug,
            max_frames=args.max_frames,
            frame_stride=frame_stride,
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
            use_da3_trajectory=args.use_da3_trajectory,
            use_mapanything_trajectory=args.use_mapanything_trajectory,
            openvo_trajectory_path=args.openvo_trajectory,
            prefer_openvo_trajectory=not args.no_openvo_default,
            vggt_long_trajectory_path=args.vggt_long_trajectory,
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
            enable_ocr_anchor=args.enable_ocr_anchor,
            use_ocr_super_res=args.ocr_super_res,
            ocr_sample_interval_sec=args.ocr_sample_interval_sec,
            ocr_min_confidence=args.ocr_min_confidence,
            ocr_video_path=args.ocr_video,
            use_osm_gazetteer=args.use_osm_gazetteer,
            classify_signs=args.classify_signs,
            enable_scale_recovery=not args.no_scale_recovery,
            use_ipm_scale=args.use_ipm_scale,
            scale_lock=args.scale_lock,
            osm_around=osm_around,
            use_vpr_prior=args.use_vpr_prior,
            vpr_source=args.vpr_source,
            use_vpr_sequence=args.use_vpr_sequence,
            vpr_viterbi=not args.no_vpr_viterbi,
            vpr_two_pass_scale=args.vpr_two_pass,
            use_plate_anchor=args.use_plate_anchor,
            use_vlm_anchor=args.use_vlm_anchor,
            vpr_search_radius_m=args.vpr_search_radius,
            vpr_ref_cap=args.vpr_cap,
            use_sun_heading=args.use_sun_heading,
            ground_truth_streets=tuple(args.ground_truth),
            ground_truth_waypoints=args.ground_truth_waypoints,
            enable_bev_splat=args.enable_bev_splat,
            bev_splat_weights=args.bev_splat_weights,
            bev_splat_repo_path=args.bev_splat_repo_path,
            bev_splat_model_module=args.bev_splat_model_module,
            bev_splat_source=args.bev_splat_source,
            bev_splat_tile_size=args.bev_splat_tile_size,
            bev_splat_half_extent_m=args.bev_splat_half_extent_m,
            bev_fusion_cap=args.bev_fusion_cap,
            enable_loop_closure=args.enable_loop_closure,
            use_vggt_gating=args.use_vggt_gating,
            vggt_keyframes=args.vggt_keyframes,
            use_orienternet=args.use_orienternet,
            orienternet_keyframes=args.orienternet_keyframes,
            orienternet_tile_m=args.orienternet_tile_m,
            ipm_scale_camera_height_m=args.ipm_scale_height,
            ipm_scale_pitch_deg=args.ipm_scale_pitch,
            vo_workers=args.vo_workers,
            video_path=local_path,
        )
        print(f"\n=== Submission: {metadata.title or metadata.url} ===")
        print(f"    city={city!r}  data_dir={cfg.data_dir}  output_dir={cfg.output_dir}")
        result = run_pipeline(cfg)
        result["video_title"] = metadata.title
        result["submission_slug"] = submission_slug
        results.append(result)

    # Write the batch summary BEFORE raising on metadata errors: one
    # bad/offline URL must not suppress the summary of the runs that
    # succeeded.
    if len(results) > 1:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        batch_result = args.output_dir / "batch_results.json"
        with batch_result.open("w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"\nWrote batch summary to {batch_result}")

    if metadata_errors:
        raise SystemExit(
            "Failed to inspect one or more videos:\n- " + "\n- ".join(metadata_errors)
        )


if __name__ == "__main__":
    main()
