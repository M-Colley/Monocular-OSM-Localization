"""Acquire input videos: download from YouTube, or describe a local file."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import yt_dlp


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    url: str
    title: str | None
    video_id: str | None


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
        return VideoMetadata(
            url=str(info.get("webpage_url") or info.get("original_url") or url),
            title=info.get("title"),
            video_id=str(raw_id) if raw_id is not None else None,
        )
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(str(e)) from e


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

    for existing in out_dir.glob(f"{filename_stem}.*"):
        if existing.suffix.lower() in {".mp4", ".mkv", ".webm"}:
            return existing

    outtmpl = str(out_dir / f"{filename_stem}.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
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
                candidates = list(out_dir.glob(f"{filename_stem}.*"))
                if not candidates:
                    raise DownloadError(f"yt-dlp finished but no file at {path}")
                path = candidates[0]
            return path
    except yt_dlp.utils.DownloadError as e:  # network / unavailable
        raise DownloadError(str(e)) from e
