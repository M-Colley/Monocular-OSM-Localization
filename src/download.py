"""Acquire input videos: download from YouTube, or describe a local file."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import yt_dlp

# yt-dlp intermediates: per-format streams ("input.f399.mp4") and the FFmpeg
# merge temp file ("input.temp.mp4"). Neither is a finished download.
_INTERMEDIATE = re.compile(r"\.(?:temp|f\d+)\.[^.]+$", re.IGNORECASE)
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm"}


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    url: str
    title: str | None
    video_id: str | None
    # Source frame rate reported by the extractor, when known. Lets the
    # caller pick a frame stride tuned to a 60 fps upload BEFORE the file
    # is downloaded (a first --url run has no local file to probe, so
    # without this it assumes 30 fps and picks half the intended stride).
    fps: float | None = None
    # Uploader description. Often names the route's districts / landmarks /
    # streets — a strong GPS-free COARSE location prior (see
    # src/location_prior.py), tighter than the city centroid.
    description: str | None = None


def local_video_metadata(path: Path) -> VideoMetadata:
    """Build :class:`VideoMetadata` for a local video file.

    Mirrors :func:`fetch_video_metadata` so the rest of the pipeline
    (slug generation, result JSON) treats local files and URLs the same:
    the title is the filename stem and the id is a stable digest of the
    absolute path, so re-running on the same file reuses the same
    data/output directories (and therefore the VO cache).
    """
    path = Path(path)
    if not path.exists():
        raise DownloadError(f"local video not found: {path}")
    resolved = path.resolve()
    digest = sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    return VideoMetadata(
        url=resolved.as_uri(),
        title=path.stem,
        video_id=f"local-{digest}",
    )


def fetch_video_metadata(url: str) -> VideoMetadata:
    """Fetch lightweight metadata without downloading the video."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        raw_id = info.get("id")
        # YouTube encodes every rendition at the source frame rate, so the
        # top-level fps matches the <=max_height stream we later download —
        # good enough to choose a stride before the file exists.
        raw_fps = info.get("fps")
        try:
            fps = float(raw_fps) if raw_fps else None
        except (TypeError, ValueError):
            fps = None
        desc = info.get("description")
        return VideoMetadata(
            url=str(info.get("webpage_url") or info.get("original_url") or url),
            title=info.get("title"),
            video_id=str(raw_id) if raw_id is not None else None,
            fps=fps,
            description=str(desc)[:4000] if desc else None,
        )
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(str(e)) from e


def _format_selector(max_height: int) -> str:
    """Video-only format chain with a last-resort '/best' fallback.

    Audio is never used by any pipeline stage, so we don't request it.
    The trailing '/best' keeps sources with unreported heights (some HLS)
    or no <=max_height rendition from hard-failing.
    """
    return f"bestvideo[height<={max_height}]/best[height<={max_height}]/best"


def _marker_path(out_dir: Path, filename_stem: str) -> Path:
    return Path(out_dir) / f"{filename_stem}.download.json"


def _existing_download(out_dir: Path, filename_stem: str) -> Path | None:
    """Return a previously completed download, or None.

    Prefers the marker written by :func:`download_video` (deterministic
    resume); falls back to globbing, but never picks up yt-dlp
    intermediates (``input.temp.mp4`` / ``input.fNNN.mp4``).
    """
    out_dir = Path(out_dir)
    marker = _marker_path(out_dir, filename_stem)
    if marker.exists():
        try:
            recorded = out_dir / json.load(open(marker, encoding="utf-8"))["file"]
            if recorded.exists():
                return recorded
        except (OSError, ValueError, KeyError):
            pass  # corrupt marker: fall through to the glob
    candidates = [
        p for p in sorted(out_dir.glob(f"{filename_stem}.*"))
        if p.suffix.lower() in _VIDEO_SUFFIXES and not _INTERMEDIATE.search(p.name)
    ]
    # Prefer the canonical merged name over other suffixes.
    for p in candidates:
        if p.name.lower() == f"{filename_stem}.mp4".lower():
            return p
    return candidates[0] if candidates else None


def download_video(
    url: str,
    out_dir: Path,
    *,
    filename_stem: str = "input",
    max_height: int = 720,
) -> Path:
    """Download `url` into `out_dir`. Returns the path to the resulting file.

    If a file matching the stem already exists, return it without re-downloading.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = _existing_download(out_dir, filename_stem)
    if existing is not None:
        return existing

    outtmpl = str(out_dir / f"{filename_stem}.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": _format_selector(max_height),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
            if path.suffix != ".mp4":
                merged = path.with_suffix(".mp4")
                if merged.exists():
                    path = merged
            if not path.exists():
                # yt-dlp sometimes renames during merge; fall back to globbing
                # (same intermediate/suffix filter as the resume check).
                path = _existing_download(out_dir, filename_stem)
                if path is None:
                    raise DownloadError(
                        f"yt-dlp finished but no completed file for "
                        f"{filename_stem}.* in {out_dir}")
            # Persist the resolved name so later runs resume deterministically.
            json.dump({"file": path.name, "url": url},
                      open(_marker_path(out_dir, filename_stem), "w", encoding="utf-8"))
            return path
    except yt_dlp.utils.DownloadError as e:  # network / unavailable
        raise DownloadError(str(e)) from e
