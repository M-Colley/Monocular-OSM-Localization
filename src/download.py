"""Download a YouTube video to a local file using yt-dlp."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yt_dlp


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    url: str
    title: str | None
    video_id: str | None


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
        return VideoMetadata(
            url=str(info.get("webpage_url") or info.get("original_url") or url),
            title=info.get("title"),
            video_id=str(info["id"]) if info.get("id") else None,
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
