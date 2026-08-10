"""Tests for local-video metadata and download resume logic (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monocular_osm.download import (
    DownloadError,
    VideoMetadata,
    _existing_download,
    _format_selector,
    cached_video_metadata,
    download_video,
    load_cached_metadata,
    local_video_metadata,
    metadata_cache_path,
    write_cached_metadata,
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


# ---------------------------------------------------------------------------
# metadata cache — a downloaded clip must not need the network again
# ---------------------------------------------------------------------------


def test_cached_video_metadata_serves_the_cache_without_a_fetch(tmp_path: Path, monkeypatch) -> None:
    url = "https://www.youtube.com/watch?v=abc123"
    write_cached_metadata(tmp_path, url, VideoMetadata(
        url=url, title="A Drive", video_id="abc123", fps=30.0, description="d"))

    def _boom(_url):    # any network call is a failure of the contract
        raise AssertionError("cached_video_metadata hit the network")

    monkeypatch.setattr("monocular_osm.download.fetch_video_metadata", _boom)
    meta = cached_video_metadata(url, tmp_path)
    assert meta.video_id == "abc123"
    assert meta.title == "A Drive"
    assert meta.fps == 30.0


def test_cached_video_metadata_writes_back_after_a_fetch(tmp_path: Path, monkeypatch) -> None:
    url = "https://www.youtube.com/watch?v=xyz789"
    fetched = VideoMetadata(url=url, title="T", video_id="xyz789", fps=50.0)
    calls = []

    def _fetch(u):
        calls.append(u)
        return fetched

    monkeypatch.setattr("monocular_osm.download.fetch_video_metadata", _fetch)
    assert cached_video_metadata(url, tmp_path).video_id == "xyz789"
    # Second call is served from the cache the first one wrote.
    assert cached_video_metadata(url, tmp_path).video_id == "xyz789"
    assert calls == [url]


def test_cached_video_metadata_propagates_a_cold_failure(tmp_path: Path, monkeypatch) -> None:
    # Nothing cached and the network refuses: the caller must be able to tell,
    # so it can fall back to whatever is already on disk.
    def _fetch(_u):
        raise DownloadError("Sign in to confirm you're not a bot")

    monkeypatch.setattr("monocular_osm.download.fetch_video_metadata", _fetch)
    with pytest.raises(DownloadError):
        cached_video_metadata("https://www.youtube.com/watch?v=cold", tmp_path)


def test_metadata_cache_is_keyed_per_url(tmp_path: Path) -> None:
    a = "https://www.youtube.com/watch?v=aaa"
    b = "https://www.youtube.com/watch?v=bbb"
    write_cached_metadata(tmp_path, a, VideoMetadata(url=a, title="A", video_id="aaa"))
    assert load_cached_metadata(tmp_path, a).title == "A"
    assert load_cached_metadata(tmp_path, b) is None
    assert metadata_cache_path(tmp_path, a) != metadata_cache_path(tmp_path, b)


def test_load_cached_metadata_survives_a_corrupt_file(tmp_path: Path) -> None:
    url = "https://www.youtube.com/watch?v=corrupt"
    path = metadata_cache_path(tmp_path, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_cached_metadata(tmp_path, url) is None


def test_format_selector_video_only_with_best_fallback() -> None:
    fmt = _format_selector(720)
    assert "bestaudio" not in fmt          # audio is never used by the pipeline
    assert fmt.endswith("/best")           # last-resort fallback, no hard fail
    assert "height<=720" in fmt


def test_format_selector_prefers_mp4_then_falls_back() -> None:
    # mp4/AVC first: YouTube serves it ~13x faster than the VP9 webm of the
    # same upload, and cv2's bundled FFmpeg decodes it most reliably. The
    # codec-agnostic alternative must still follow, so an upload with no
    # AVC rendition is not left undownloadable.
    fmt = _format_selector(1080)
    choices = fmt.split("/")
    assert choices[0] == "bestvideo[height<=1080][ext=mp4]"
    assert "bestvideo[height<=1080]" in choices[1:]
