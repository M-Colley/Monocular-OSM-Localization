"""Synthetic files/roads only: no cases, video decoding, OCR, network or GT."""
import copy
import hashlib
import json
import os
from pathlib import Path
import zipfile

import networkx as nx
import numpy as np
import pytest

from scripts import junction_sequence_probe as p
from monocular_osm import graph_filter as gf, junction_inputs as ji, junction_sequence as js
from monocular_osm import visual_odometry as vo
from monocular_osm.road_sign_clues import extract_road_sign_clues, _digest


def dump(path, value):
    p.write_json(path, value)
    return {"path": str(path), "sha256": p.sha(path)}


def sources():
    return {name: p.sha(p.ROOT / name) for name in p.SOURCE_PATHS}


def roads():
    graph = nx.MultiDiGraph(crs="EPSG:32632")
    for i in range(3):
        graph.add_node(i + 1, x=455000. + i * 100., y=5428000.)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    return graph, gf.RoadNetwork.from_graph(graph)


def motion(tmp_path):
    n = 6
    x = np.array([0., 1., 1., 2., 2., 3.])
    centers = np.column_stack([x, np.zeros(n), np.zeros(n)])
    times = 10. + np.arange(n, dtype=np.float64) * .1
    video = {"path": str(tmp_path / "never_opened_video.mp4"), "size": 123, "sha256": "a" * 64}
    signature = {"schema": 2, "vo_algo_version": 3,
                 "video": {"path": os.path.normcase(str(Path(video["path"]).resolve())), "size": 123},
                 "extraction": {"n_frames": n, "stride": 3, "fps": 30.,
                                "timestamps_sha256": hashlib.sha256(times.astype("<f8").tobytes()).hexdigest()}}
    arrays = {"centers": centers, "xz": np.column_stack([x, np.zeros(n)]),
              "valid": np.array([True, True, False, True, True, True]),
              "n_inliers": np.full(n, 15), "rotations": np.repeat(np.eye(3)[None], n, axis=0),
              "translations": -centers, "cache_signature": np.array(json.dumps(signature))}
    native = {"xz": arrays["xz"].copy(), "valid": arrays["valid"].copy(), "timestamps": times}
    return arrays, native, video


def prepared(tmp_path, monkeypatch):
    graph, net = roads()
    lonlat = net.local_to_latlon(np.array([[200., 0.]]))[0]
    plan = ji.plan_inputs("Synthetic", [1., 2., 1.2, 2.2], bbox_source={
        "city": "Synthetic", "scope": "named_city_bbox", "source_kind": "public_geocode", "source_sha256": "b" * 64})
    junctions = [{"id": "node/3", "lat": float(lonlat[0]), "lon": float(lonlat[1]),
                  "ref": "7A", "normalized_refs": ["7A"], "tags": {"highway": "motorway_junction", "ref": "7A"}}]
    inventory = {"status": "complete", "city": "Synthetic", "scope": "named_city_bbox", "plan": plan["inventory_plan"],
                 "junctions": junctions, "snapshot_sha256": _digest(junctions)}
    clues = extract_road_sign_clues([{"text": "EXIT 7A", "confidence": .99, "t_sec": 10.3,
                                     "bbox": [10., 10., 80., 30.]}], [480, 640])
    raw = tmp_path / "raw.graphml"
    raw.write_bytes(b"synthetic private graph loader fixture")
    import osmnx as ox
    monkeypatch.setattr(ox, "load_graphml", lambda path: graph)
    observed = {}
    def fake_prepare(raw_graph, inv, matched, actual_plan, *, copy_graph=True):
        observed.update(copy=copy_graph, matched=matched)
        assert raw_graph is graph and inv == inventory and actual_plan == plan
        return net, {"schema": "synthetic-topology-receipt", "copy_graph": copy_graph}
    monkeypatch.setattr(ji, "prepare_graph", fake_prepare)
    config = {"schema": "junction-sequence-probe-prepare-v1", "source_hashes": sources(),
              "inputs": {"raw_graphml": {"path": str(raw), "sha256": p.sha(raw)},
                         "junction_inventory": dump(tmp_path / "inventory.json", inventory),
                         "temporal_ocr": dump(tmp_path / "ocr.json", {"clues": clues, "origin_video_sha256": "a" * 64})},
              "output_dir": str(tmp_path / "prepared")}
    ref = dump(tmp_path / "prepare_config.json", config)
    result = p.prepare(ref["path"], ref["sha256"])
    assert observed["copy"] is False
    assert observed["matched"]["matches"][0]["correlation"]["power"] == 1.
    return result, tmp_path / "prepared"


def run_config(tmp_path, prepared_path):
    arrays, native, video = motion(tmp_path)
    p.save_npz(tmp_path / "vo.npz", arrays)
    p.save_npz(tmp_path / "native.npz", native)
    config = {"schema": "junction-sequence-probe-run-v1", "source_hashes": sources(), "video_binding": video,
              "inputs": {key: {"path": str(path), "sha256": p.sha(path)} for key, path in {
                  "prepared": prepared_path / "frozen_prepare.json", "native_vo": tmp_path / "vo.npz",
                  "native_clock": tmp_path / "native.npz"}.items()}, "output_dir": str(tmp_path / "run")}
    return config


def test_compact_network_roundtrip_preserves_all_numeric_arrays_and_keys(tmp_path):
    _, net = roads()
    p.save_network(tmp_path, net)
    restored = p.load_network(p.load_npz(tmp_path / "network.npz"), p.read_json(tmp_path / "network.json"), gf)
    assert restored.edge_keys == net.edge_keys and restored.crs == net.crs
    for key in p.NET_ARRAYS:
        np.testing.assert_array_equal(getattr(restored, key), getattr(net, key))


def test_prepare_consumes_disposable_graph_and_freezes_all_outputs(tmp_path, monkeypatch):
    result, directory = prepared(tmp_path, monkeypatch)
    assert result["preparation_worker_must_exit_before_run"] is True
    assert result["external_actor_closure_verified"] is False
    assert result["selected_answer"] is None
    assert result["origin_video_sha256"] == "a" * 64
    assert set(result["artifacts"]) == {"network.npz", "network.json", "matched.json", "topology_receipt.json", "plan.json"}
    for name, digest in result["artifacts"].items():
        assert p.sha(directory / name) == digest


def test_native_motion_uses_signed_nominal_interval_but_exact_pts_and_unknown_not_stop(tmp_path):
    arrays, native, video = motion(tmp_path)
    odometry, status, times = p.load_motion(arrays, native, "timestamps", video, gf, vo)
    assert odometry.unit_seconds == .1 and not odometry.metric
    np.testing.assert_array_equal(odometry.dt, np.diff(times))
    assert status.tolist() == ["measured", "unknown", "measured", "unknown", "measured"]
    assert not np.any(status == "stationary")
    np.testing.assert_array_equal(odometry.heading_var, gf.heading_increment_noise(odometry))
    assert not Path(video["path"]).exists()  # Root binds bytes; this helper never opens video.


@pytest.mark.parametrize("bad", ["clock", "valid", "xz", "signature", "fps", "video", "nonfinite"])
def test_motion_refuses_clock_cache_or_external_identity_mismatch(tmp_path, bad):
    arrays, native, video = motion(tmp_path)
    if bad == "clock": native["timestamps"][2] += .01
    if bad == "valid": native["valid"][1] = False
    if bad == "xz": native["xz"][1, 0] += 1.
    if bad in {"signature", "fps"}:
        sig = json.loads(str(arrays["cache_signature"]))
        if bad == "signature": sig["extraction"]["timestamps_sha256"] = "0" * 64
        else: sig["extraction"]["fps"] = 0.
        arrays["cache_signature"] = np.array(json.dumps(sig))
    if bad == "video": video["size"] += 1
    if bad == "nonfinite": arrays["rotations"][1, 0, 0] = np.nan
    with pytest.raises(ValueError):
        p.load_motion(arrays, native, "timestamps", video, gf, vo)


def test_run_keeps_all_result_arrays_and_frozen_profile(tmp_path, monkeypatch):
    _, directory = prepared(tmp_path, monkeypatch)
    config = run_config(tmp_path, directory)
    def fake_align(net, odometry, status, times, matched, *, config):
        assert config.particles == 50_000 and config.seed == 0 and config.inner_s == 1800.
        assert config.junction_prior == "legacy"
        assert len(times) == 6 and len(status) == 5
        assert matched["matches"][0]["candidates"][0]["id"] == "node/3"
        return {"status": "complete_diagnostic_only", "times": times, "route_edge": np.array([0, 1]),
                "route_latlon": np.array([[1., 2.], [1., 2.1]]), "other_array": np.array([np.nan]), "selection": None,
                "profile": {"method": "private-resolved-legacy-profile"}}
    monkeypatch.setattr(js, "align", fake_align)
    ref = dump(tmp_path / "run_config.json", config)
    result = p.run(ref["path"], ref["sha256"])
    stored = p.load_npz(tmp_path / "run" / "result.npz")
    assert {"times", "route_edge", "route_latlon", "other_array", "motion_status", "native_times"} <= set(stored)
    assert result["numerical_status"] == "complete_diagnostic_only" and result["accuracy_admitted"] is False
    assert result["profile"] == {"method": "private-resolved-legacy-profile"}
    assert p.read_json(tmp_path / "run" / "result.json")["video_bytes_opened_or_verified_by_probe"] is False


def test_full_window_unavailable_is_retained_not_relabelled_as_route(tmp_path, monkeypatch):
    _, directory = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(js, "align", lambda *a, **kw: {"status": "unavailable", "route_latlon": None,
                                                    "reason": "full window over cap", "profile": dict(js.PROFILE)})
    ref = dump(tmp_path / "run_config.json", run_config(tmp_path, directory))
    result = p.run(ref["path"], ref["sha256"])
    assert result["numerical_status"] == "unavailable"
    assert p.read_json(tmp_path / "run" / "result.json")["route_latlon"] is None


def test_explicit_conditional_uniform_prior_and_resolved_result_profile_are_recorded(tmp_path, monkeypatch):
    _, directory = prepared(tmp_path, monkeypatch)
    config = run_config(tmp_path, directory)
    config["junction_prior"] = "conditional-uniform-v3"
    resolved = {"method": "private-resolved-v3", "normalise_turn_prior": True}
    def fake_align(net, odometry, status, times, matched, *, config):
        assert config.junction_prior == "conditional-uniform-v3"
        assert config.particles == 50_000 and config.seed == 0
        return {"status": "unavailable", "route_latlon": None, "profile": resolved}
    monkeypatch.setattr(js, "align", fake_align)
    ref = dump(tmp_path / "run_config.json", config)
    result = p.run(ref["path"], ref["sha256"])
    assert result["profile"] == resolved and result["profile"] != js.PROFILE
    assert p.read_json(tmp_path / "run" / "result.json")["profile"] == resolved


@pytest.mark.parametrize("prior", ["uniform", "", None, True])
def test_unregistered_prior_refused_before_numeric_inputs(tmp_path, monkeypatch, prior):
    config = {"schema": "junction-sequence-probe-run-v1", "source_hashes": sources(), "junction_prior": prior}
    ref = dump(tmp_path / "invalid_prior.json", config)
    monkeypatch.setattr(p, "load_npz", lambda *a: pytest.fail("numeric values before prior validation"))
    with pytest.raises(ValueError):
        p.run(ref["path"], ref["sha256"])


def test_late_input_mutation_during_solver_refuses_final_master(tmp_path, monkeypatch):
    _, directory = prepared(tmp_path, monkeypatch)
    config = run_config(tmp_path, directory)
    def mutate(*args, **kwargs):
        target = (tmp_path / "native.npz").resolve()
        assert target.is_relative_to(tmp_path.resolve())
        target.write_bytes(b"changed private input")
        return {"status": "unavailable", "route_latlon": None, "profile": dict(js.PROFILE)}
    monkeypatch.setattr(js, "align", mutate)
    ref = dump(tmp_path / "run_config.json", config)
    with pytest.raises(ValueError, match="bytes changed"):
        p.run(ref["path"], ref["sha256"])
    assert not (tmp_path / "run" / "frozen_run.json").exists()


def test_config_sha_and_source_poison_refuse_before_any_graph_loader(tmp_path, monkeypatch):
    config = {"schema": "junction-sequence-probe-prepare-v1", "source_hashes": sources()}
    config["source_hashes"]["monocular_osm/junction_inputs.py"] = "0" * 64
    ref = dump(tmp_path / "bad.json", config)
    monkeypatch.setattr(p, "dependencies", lambda: pytest.fail("imports before source admission"))
    with pytest.raises(ValueError, match="configuration file SHA"):
        p.prepare(ref["path"], "0" * 64)
    with pytest.raises(ValueError, match="input bytes differ"):
        p.prepare(ref["path"], ref["sha256"])


def test_run_refuses_clues_from_different_video_before_network_or_motion_load(tmp_path, monkeypatch):
    _, directory = prepared(tmp_path, monkeypatch)
    config = run_config(tmp_path, directory)
    config["video_binding"]["sha256"] = "b" * 64
    ref = dump(tmp_path / "run_config.json", config)
    monkeypatch.setattr(p, "load_npz", lambda *a: pytest.fail("numeric values before video origin join"))
    with pytest.raises(ValueError, match="video origins differ"):
        p.run(ref["path"], ref["sha256"])


@pytest.mark.parametrize("direct", [False, True])
def test_missing_temporal_origin_refused_before_graph_loader(tmp_path, monkeypatch, direct):
    _, directory = prepared(tmp_path, monkeypatch)
    config = p.read_json(tmp_path / "prepare_config.json")
    source = p.read_json(tmp_path / "ocr.json")["clues"]
    config["inputs"]["temporal_ocr"] = dump(tmp_path / "unproven_ocr.json", source if direct else {"clues": source})
    if direct:
        config["clues_member"] = None
    config["output_dir"] = str(tmp_path / "must_not_publish")
    ref = dump(tmp_path / "missing_origin_config.json", config)
    import osmnx as ox
    monkeypatch.setattr(ox, "load_graphml", lambda *a: pytest.fail("graph before OCR origin proof"))
    with pytest.raises(ValueError, match="origin-video SHA required"):
        p.prepare(ref["path"], ref["sha256"])


def test_compact_typed_array_tamper_refused(tmp_path):
    _, net = roads()
    p.save_network(tmp_path, net)
    arrays = p.load_npz(tmp_path / "network.npz")
    arrays["edge_len"][0] += 1.
    with pytest.raises(ValueError, match="typed bytes differ"):
        p.load_network(arrays, p.read_json(tmp_path / "network.json"), gf)


def test_npz_traversal_and_object_arrays_forbidden(tmp_path):
    with zipfile.ZipFile(tmp_path / "bad.npz", "w") as archive:
        archive.writestr("../x.npy", b"bad")
    with pytest.raises(ValueError, match="member/storage"):
        p.load_npz(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="NPZ arrays only"):
        p.save_npz(tmp_path / "object.npz", {"x": np.array([object()], dtype=object)})
