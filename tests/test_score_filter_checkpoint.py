import json

import pytest

from monocular_osm.provenance import file_sha256
from scripts.score_filter_checkpoint import verify_checkpoint


def _checkpoint(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    source = out / "source_snapshot" / "filter.py"
    source.parent.mkdir()
    source.write_bytes(b"# frozen source\r\n")
    video, graph = tmp_path / "video.mp4", tmp_path / "city.graphml"
    video.write_bytes(b"actual video bytes")
    graph.write_bytes(b"named-city map bytes")
    artifact = out / "result.json"
    artifact.write_text("{}")
    inputs = {"source_sha256": {"filter.py": file_sha256(source)},
              "video": {"path": str(video), "sha256": file_sha256(video)},
              "map": {"cache_path": str(graph), "sha256": file_sha256(graph)}}
    manifest = {"prediction_available": True, "scoring_has_not_run": True,
                "artifacts": {"result.json": file_sha256(artifact)}, "inputs": inputs}
    (out / "frozen_prediction.json").write_text(json.dumps(manifest), encoding="utf-8")
    return out, {"artifact": artifact, "source": source, "video": video, "map": graph}


def test_post_freeze_scorer_checks_complete_prediction_and_inputs(tmp_path):
    out, _ = _checkpoint(tmp_path)
    assert verify_checkpoint(out)["prediction_available"]
    (out / "offline_score.json").write_text("must never be parsed")
    verify_checkpoint(out)


@pytest.mark.parametrize("target", ["artifact", "source", "video", "map"])
def test_post_freeze_scorer_rejects_changed_bytes_before_labels(tmp_path, target):
    out, paths = _checkpoint(tmp_path)
    paths[target].write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_checkpoint(out)


@pytest.mark.parametrize("change", [{"vo_start_sec": .5}, {"vo_end_sec": .9}, {"frame_stride": 2}])
def test_normal_scorer_rejects_unproven_window_before_label_access(tmp_path, monkeypatch, change):
    from pathlib import Path
    from types import SimpleNamespace
    import cv2
    import numpy as np
    from scripts import score_filter_checkpoint as scorer
    config = {"vo_start_sec": 0., "vo_end_sec": None, "frame_stride": 1, "max_frames": None, **change}
    (tmp_path / "result.json").write_text(json.dumps({"provenance": {"config": config}}))
    inputs = {"video": {"path": "unused.mp4"},
              "extraction": {"stride": 1, "max_frames": None, "n_frames": 11},
              "timing": {"fps": 10., "first_s": 0., "last_s": 1.}}
    monkeypatch.setattr(scorer, "verify_checkpoint", lambda path: {"inputs": inputs})
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: SimpleNamespace(
        get=lambda key: 11 if key == cv2.CAP_PROP_FRAME_COUNT else 10., release=lambda: None))
    np.savez(tmp_path / "filter_prediction.npz", frame_index=[5, 10], video_times_s=[.5, 1.])
    original = Path.resolve
    def guarded(path, *args, **kwargs):
        if path.name == "forbidden_labels":
            raise AssertionError("labels accessed before full-video proof")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", guarded)
    with pytest.raises(ValueError, match="full-video"):
        scorer.main(["--checkpoint", str(tmp_path), "--labels-dir", str(tmp_path / "forbidden_labels"),
                     "--waypoints", str(tmp_path / "waypoints.json")])
