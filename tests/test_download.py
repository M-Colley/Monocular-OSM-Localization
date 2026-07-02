"""Tests for local-video metadata and download resume logic (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.download import (
    DownloadError,
    _existing_download,
    _format_selector,
    download_video,
    local_video_metadata,
)


def test_local_metadata_fields(tmp_path: Path) -> None:
    video = tmp_path / "Driving in Ulm.mp4"
    video.write_bytes(b"\x00")
    meta = local_video_metadata(video)
    assert meta.title == "Driving in Ulm"
    assert meta.video_id is not None and meta.video_id.startswith("local-")
    assert len(meta.video_id) == len("local-") + 12
    assert meta.url.startswith("file://")


def test_local_metadata_id_is_stable_and_path_normalized(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    # Same file through a non-canonical spelling must produce the same id,
    # so re-runs reuse the same data/output dirs (and the VO cache).
    alias = video.parent / "." / video.name
    assert local_video_metadata(video) == local_video_metadata(alias)


def test_local_metadata_differs_per_file(tmp_path: Path) -> None:
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"\x00")
    b.write_bytes(b"\x00")
    assert local_video_metadata(a).video_id != local_video_metadata(b).video_id


def test_local_metadata_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="not found"):
        local_video_metadata(tmp_path / "missing.mp4")


# ---------------------------------------------------------------------------
# resume check — must never mistake yt-dlp intermediates for a finished file
# ---------------------------------------------------------------------------


def test_resume_ignores_ytdlp_intermediates(tmp_path: Path) -> None:
    # A run killed mid-merge leaves the FFmpeg temp file and per-format
    # streams behind; neither is a completed download.
    (tmp_path / "input.temp.mp4").write_bytes(b"\x00")
    (tmp_path / "input.f399.mp4").write_bytes(b"\x00")
    (tmp_path / "input.f140.m4a").write_bytes(b"\x00")
    assert _existing_download(tmp_path, "input") is None


def test_resume_prefers_final_mp4_over_leftovers(tmp_path: Path) -> None:
    (tmp_path / "input.f399.mp4").write_bytes(b"\x00")
    (tmp_path / "input.mp4").write_bytes(b"\x00")
    (tmp_path / "input.temp.mp4").write_bytes(b"\x00")
    found = _existing_download(tmp_path, "input")
    assert found is not None and found.name == "input.mp4"


def test_resume_fallback_glob_filters_suffixes(tmp_path: Path) -> None:
    # An audio-only leftover sorts alphabetically before input.webm on
    # Windows globs — it must not be returned as the video.
    (tmp_path / "input.part").write_bytes(b"\x00")
    (tmp_path / "input.webm").write_bytes(b"\x00")
    found = _existing_download(tmp_path, "input")
    assert found is not None and found.name == "input.webm"


def test_resume_uses_persisted_marker(tmp_path: Path) -> None:
    # The marker records the resolved final filename; it wins over globbing.
    (tmp_path / "input.mkv").write_bytes(b"\x00")
    (tmp_path / "input.mp4").write_bytes(b"\x00")
    (tmp_path / "input.download.json").write_text(
        json.dumps({"file": "input.mkv", "url": "http://x"}), encoding="utf-8")
    found = _existing_download(tmp_path, "input")
    assert found is not None and found.name == "input.mkv"


def test_resume_marker_for_missing_file_falls_back(tmp_path: Path) -> None:
    (tmp_path / "input.download.json").write_text(
        json.dumps({"file": "gone.mp4", "url": "http://x"}), encoding="utf-8")
    (tmp_path / "input.mp4").write_bytes(b"\x00")
    found = _existing_download(tmp_path, "input")
    assert found is not None and found.name == "input.mp4"


def test_download_video_resumes_completed_file(tmp_path: Path) -> None:
    # No network: an existing completed file short-circuits the download.
    (tmp_path / "input.mp4").write_bytes(b"\x00")
    assert download_video("http://ignored", tmp_path).name == "input.mp4"


def test_format_selector_video_only_with_best_fallback() -> None:
    fmt = _format_selector(720)
    assert "bestaudio" not in fmt          # audio is never used by the pipeline
    assert fmt.endswith("/best")           # last-resort fallback, no hard fail
    assert "height<=720" in fmt
