"""Stage 1-2: get the video and its frames.

Resolution order for the video: a local file the user supplied, a cached
download, or a fresh yt-dlp download.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..download import download_video
from ..frame_extraction import extract_frames
from .context import RunContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..pipeline import PipelineConfig


_VIDEO_EXT_RANK = {".mp4": 0, ".mkv": 1, ".webm": 2}
_NON_VIDEO_SUFFIXES = {".json", ".part", ".ytdl", ".txt"}


def rank_cached_inputs(paths) -> list:
    """Deterministic order for cached ``input.*`` video files: fixed extension
    preference (mp4 < mkv < webm, other suffixes last), then newest, then name.

    yt-dlp intermediates (``input.temp.mp4``, per-format ``input.f399.mp4``)
    and sidecars (``input.download.json``, ``.part``, ``.ytdl``) are not
    finished videos and are dropped — an intermediate shares the ``.mp4``
    suffix and, being newer, used to outrank the real download.
    """
    from ..download import _INTERMEDIATE

    finished = [
        p for p in paths
        if p.suffix.lower() not in _NON_VIDEO_SUFFIXES
        and not _INTERMEDIATE.search(p.name)
    ]
    return sorted(
        finished,
        key=lambda p: (_VIDEO_EXT_RANK.get(p.suffix.lower(), len(_VIDEO_EXT_RANK)),
                       -p.stat().st_mtime, p.name),
    )


def pick_cached_input(data_dir: Path) -> Path | None:
    """The cached download to analyse, or None.

    The marker :func:`download.download_video` writes names the finished
    file and wins; otherwise the best of :func:`rank_cached_inputs`. Shared by
    :func:`_resolve_input_video` and the CLI's fps probe so the file whose fps
    picks the stride is the file the pipeline analyses.
    """
    from ..download import _existing_download

    marked = _existing_download(Path(data_dir), "input")
    if marked is not None:
        return marked
    ranked = rank_cached_inputs(Path(data_dir).glob("input.*"))
    return ranked[0] if ranked else None


def _resolve_input_video(cfg: PipelineConfig) -> Path:
    """Return the path of the video to analyze.

    Resolution order:
    1. ``cfg.video_path`` — a local file supplied by the user. Used as-is
       (never copied; multi-GB dashcam files shouldn't be duplicated).
    2. ``cfg.skip_download`` — a previously-downloaded ``input.*`` in
       ``cfg.data_dir``.
    3. Otherwise download ``cfg.url`` with yt-dlp.
    """
    if cfg.video_path is not None:
        video_path = Path(cfg.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"local video not found: {video_path}")
        print(f"[1/5] Using local video: {video_path}")
        return video_path
    if cfg.skip_download:
        picked = pick_cached_input(cfg.data_dir)
        if picked is None:
            raise FileNotFoundError("--skip-download but no cached video in data/")
        print(f"[1/5] Using cached video: {picked}")
        return picked
    print(f"[1/5] Downloading video from {cfg.url}")
    video_path = download_video(cfg.url, cfg.data_dir)
    print(f"      -> {video_path}")
    return video_path


def acquire_video(ctx: RunContext) -> None:
    """Resolve the input video (local file / cached download / yt-dlp) -> ``ctx.video_path``."""
    cfg = ctx.cfg
    # 1. Acquire the input video (local file / cache / download).
    ctx.video_path = _resolve_input_video(cfg)


def extract_video_frames(ctx: RunContext) -> None:
    """Decode the analysed window of the video -> ``ctx.frames``."""
    cfg = ctx.cfg
    video_path = ctx.video_path
    # 2. Frames.
    print(f"[2/5] Extracting frames (stride={cfg.frame_stride}, "
          f"window={cfg.vo_start_sec}-{cfg.vo_end_sec}s)")
    ctx.frames = extract_frames(
        video_path,
        stride=cfg.frame_stride,
        max_frames=cfg.max_frames,
        start_sec=cfg.vo_start_sec,
        end_sec=cfg.vo_end_sec,
    )
    print(f"      -> {len(ctx.frames.frames)} frames @ {ctx.frames.fps:.1f} fps")
