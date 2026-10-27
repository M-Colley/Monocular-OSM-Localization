import json
from dataclasses import asdict

import pytest

from scripts.probe_strict_video_filter import sha256
from scripts.replicate_strict_video_filter import fixed_seed_config, verify_frozen_inputs


def _frozen(tmp_path):
    base, source = tmp_path / "base", tmp_path / "source"
    base.mkdir()
    relative = "monocular_osm/graph_filter.py"
    for path in (base / "source_snapshot" / relative, source / relative):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen particle filter\n")
    video, graph = tmp_path / "video.mp4", tmp_path / "city.graphml"
    video.write_bytes(b"video pixels")
    graph.write_bytes(b"whole-city graph")
    config = {"video": str(video), "graph": str(graph), "video_sha256": sha256(video),
              "graph_sha256": sha256(graph), "input_policy": {key: False for key in
              ("calibration", "raw_images", "sensor_timestamps", "gps_or_oxts", "retrieval",
               "initial_position", "ground_truth", "network")}}
    measured = {"filter_config": {"init_disc": None}}
    for name, value in (("inference_config.json", config), ("measured_config.json", measured)):
        (base / name).write_text(json.dumps(value), encoding="utf-8")
    manifest = {"source_sha256": {relative: sha256(source / relative)},
                "artifacts": {name: sha256(base / name) for name in
                              ("inference_config.json", "measured_config.json")}}
    (base / "frozen_prediction.json").write_text(json.dumps(manifest), encoding="utf-8")
    return base, source


def test_frozen_replay_verifies_artifacts_sources_and_inputs(tmp_path):
    base, source = _frozen(tmp_path)
    manifest, config, measured = verify_frozen_inputs(base, source)
    assert measured["filter_config"]["init_disc"] is None
    # Existing score files are deliberately outside the frozen input set.
    (base / "offline_score.json").write_text("this must never be parsed")
    verify_frozen_inputs(base, source)
    (base / "measured_config.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_frozen_inputs(base, source)


@pytest.mark.parametrize("target", ["snapshot", "runtime", "video", "graph"])
def test_changed_input_or_pf_source_aborts(tmp_path, target):
    base, source = _frozen(tmp_path)
    paths = {"snapshot": base / "source_snapshot/monocular_osm/graph_filter.py",
             "runtime": source / "monocular_osm/graph_filter.py", "video": tmp_path / "video.mp4",
             "graph": tmp_path / "city.graphml"}
    paths[target].write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch|differs"):
        verify_frozen_inputs(base, source)


def test_only_seed_changes_from_frozen_experiment():
    from monocular_osm.graph_filter import FilterConfig

    frozen = asdict(FilterConfig(n_particles=600_000, seed=0, path_budget_mb=256,
                                metric_logk_sigma=.25, metric_speed_dev_sigma=.09737614682216433))
    for seed in (1, 2, 3, 4):
        actual = asdict(fixed_seed_config(frozen, seed))
        assert actual == {**frozen, "seed": seed}
        assert actual["init_disc"] is None
