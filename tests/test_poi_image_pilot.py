"""No-model checks for the image-only pilot's source and label barriers."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pilot_poi_image_track.py"
spec = importlib.util.spec_from_file_location("poi_image_pilot_tests", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def _freeze(run):
    snapshot = run / "source_snapshot"
    snapshot.mkdir()
    (snapshot / SCRIPT.name).write_bytes(SCRIPT.read_bytes())
    inputs = run / "inputs"
    inputs.mkdir()
    for name in ("video", "inventory"):
        (inputs / name).write_text(name)
    model = run / "model.bin"
    model.write_bytes(b"public model")
    references = run / "references"
    references.mkdir()
    (references / "reference.bin").write_bytes(b"public reference")
    pilot.write(run / "experiment.json", {
        "runner_sha256": pilot.sha(SCRIPT), "source_files": {}, "model_assets": [
            {"relative": "model.bin", "sha256": pilot.sha(model)}],
        **{key: {"path": str(inputs / key), "sha256": pilot.sha(inputs / key)}
           for key in ("video", "inventory")}})
    for name in ("query_input", "clues", "native_ocr", "resources"):
        pilot.write(run / f"{name}.json", {})
    pilot.write(run / "prediction.json", {"status": "unavailable"})
    pilot.freeze_outputs(run)


@pytest.mark.parametrize("amount,passes", [(7, False), (8, True)])
def test_real_commit_gate_precedes_inference(monkeypatch, amount, passes):
    monkeypatch.setattr(pilot, "available_commit", lambda: amount * 1024**3)
    if passes:
        assert pilot.require_memory() == 8 * 1024**3
    else:
        with pytest.raises(RuntimeError, match="8 GiB"):
            pilot.require_memory()


def test_token_values_and_url_encoded_spellings_are_not_persisted(monkeypatch):
    monkeypatch.setenv("MLY_TOKEN", "fake/a+ b")
    result = pilot.safe({"raw": "fake/a+ b", "encoded": "fake%2Fa%2B%20b", "plus": "fake%2Fa%2B+b"})
    assert set(result.values()) == {"[REDACTED]"}


@pytest.mark.parametrize("relative", ["prediction.json", "clues.json", "inputs/video", "inputs/inventory",
                                     "model.bin", "references/reference.bin"])
def test_all_frozen_inputs_verified_before_any_label_access(tmp_path, monkeypatch, relative):
    _freeze(tmp_path)
    (tmp_path / relative).write_text("changed")
    original = pilot.read
    def guarded(path):
        if Path(path).name == "labels.json":
            raise AssertionError("label barrier violated")
        return original(path)
    monkeypatch.setattr(pilot, "read", guarded)
    with pytest.raises(ValueError, match="changed"):
        pilot.score(SimpleNamespace(run=tmp_path, gt=tmp_path / "labels.json"))


def test_freeze_cannot_rebless_an_edited_prediction(tmp_path):
    _freeze(tmp_path)
    pilot.write(tmp_path / "prediction.json", {"status": "complete"})
    with pytest.raises(ValueError, match="cannot be rewritten"):
        pilot.freeze_outputs(tmp_path)
    with pytest.raises(ValueError, match="changed"):
        pilot.verify_frozen(tmp_path)


@pytest.mark.parametrize("branches", [0, 2])
def test_fresh_ambiguous_or_empty_ocr_refuses_before_reference_or_model_work(tmp_path, monkeypatch, branches):
    from monocular_osm import poi_image_track as image
    closed = []
    frames = SimpleNamespace(frames=[np.zeros((2, 3, 3), np.uint8)] * 2,
        timestamps=np.array([0., 1.]), fps=30., sampling={"sample_count": 2}, close=lambda: closed.append(True))
    def extract(*args, **kwargs):
        assert kwargs["sample_count"] == 80 and kwargs["stride"] == 1 and kwargs["max_frames"] is None
        return frames
    monkeypatch.setattr(pilot, "native_ocr", lambda *args: ([], {}))
    def forbidden(*args):
        pytest.fail("reference inference ran despite an unresolved native clue branch")
    monkeypatch.setattr(pilot, "acquire_and_decode", forbidden)
    args = SimpleNamespace(run=tmp_path, start=0., end=1.)
    pilot.compute_queries(args, image, {"video": {"path": "video.mp4", "sha256": "b" * 64}}, {}, extract,
                          lambda *args: {"candidate_clusters": [{}] * branches})
    assert closed and pilot.read(tmp_path / "prediction.json")["status"] == "unavailable"


def test_geodesic_scoring_interpolates_window_edges_without_extrapolation():
    labels = [{"t_sec": t, "lat": 0., "lon": t / 1000} for t in (0., 10., 20.)]
    times, route, at = pilot.scoring_truth(labels, [5., 15.])
    np.testing.assert_allclose(times, [5., 10., 15.])
    np.testing.assert_allclose(route[:, 1], [.005, .010, .015], atol=1e-12)
    assert at(7.5)[1] == pytest.approx(.0075)
    with pytest.raises(ValueError, match="covered"):
        pilot.scoring_truth(labels, [0., 21.])


def test_frozen_artifact_paths_cannot_escape_the_isolated_run(tmp_path):
    _freeze(tmp_path)
    manifest = pilot.read(tmp_path / "frozen_prediction.json")
    manifest["artifacts"]["../escaped"] = "irrelevant"
    pilot.write(tmp_path / "frozen_prediction.json", manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        pilot.verify_frozen(tmp_path)


@pytest.mark.parametrize("change", ["timestamp", "subset", "ordinal"])
def test_public_evidence_must_bind_every_actual_fresh_query_frame(change):
    times = np.array([0., 1., 3.])
    evidence = {"query_indices": np.arange(3), "query_times": times.copy()}
    pilot.validate_query_identity(evidence, times)
    if change == "timestamp":
        evidence["query_times"][1] += .01
    elif change == "subset":
        evidence = {"query_indices": np.array([0, 2]), "query_times": times[[0, 2]]}
    else:
        evidence["query_indices"][2] = 9
    with pytest.raises(ValueError, match="freshly sampled"):
        pilot.validate_query_identity(evidence, times)
