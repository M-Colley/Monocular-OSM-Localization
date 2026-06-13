"""Tests for local-video metadata (no network involved)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.download import DownloadError, local_video_metadata


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
