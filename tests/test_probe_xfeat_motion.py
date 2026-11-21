"""Mocked protocol/streaming boundaries; never constructs a neural model."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import probe_xfeat_motion as r


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def test_import_has_no_torch_or_official_code():
    code = "import sys; from scripts import probe_xfeat_motion; assert 'torch' not in sys.modules; assert 'modules.xfeat' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=r.ROOT, check=True)


def test_profile_declares_exact_pair_storage_amendment_and_no_adoption():
    assert r.PROFILE["frames"] == 2950 and r.PROFILE["pairs_per_method"] == 2949
    assert r.PROFILE["stride"] == 3 and r.PROFILE["native_shape"] == [720, 1280, 3]
    assert r.PROFILE["XFeat"]["min_cossim"] == -1
    assert r.PROFILE["XFeat"]["top_k"] == 2500 and r.PROFILE["XFeat"]["detection_threshold"] == .05
    assert r.PROFILE["superseded_proposal_sha256"] == r.PRIOR_SHA
    assert "Do not duplicate" in r.PROFILE["storage_amendment"]
    assert r.PROFILE["selected_frontend"] is r.PROFILE["confidence"] is None
    assert r.PROFILE["policy_adopted"] is r.PROFILE["metric_scale"] is False


def capture_fixture(tmp_path, monkeypatch, *, truncate=False, wrong_pts=False, wrong_pixels=False):
    monkeypatch.setattr(r, "INPUT_SHAPE", (4, 6, 3))
    pixels = [np.full((4, 6, 3), i, np.uint8) for i in range(10)]
    frames = [{"ordinal": i, "t_sec": i/30., "pixels_sha256": hashlib.sha256(p.tobytes()).hexdigest()} for i, p in enumerate(pixels)]
    captures = []
    class Capture:
        def __init__(self, *_): self.index = 0; self.released = False; captures.append(self)
        def isOpened(self): return True
        def get(self, key):
            if key == cv2.CAP_PROP_POS_FRAMES: return self.index
            if key == cv2.CAP_PROP_POS_MSEC: return (self.index-1)/30.*1000+(1 if wrong_pts else 0)
            raise AssertionError(key)
        def read(self):
            if self.index >= (5 if truncate else len(pixels)): return False, None
            value = pixels[self.index].copy(); self.index += 1
            if wrong_pixels: value[0, 0, 0] += 1
            return True, value
        def release(self): self.released = True
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    return frames, captures


def test_fresh_native_stream_uses_actual_ordinals_pts_hashes_and_releases(tmp_path, monkeypatch):
    frames, captures = capture_fixture(tmp_path, monkeypatch)
    rows = list(r.stream_native(tmp_path, frames, 4))
    assert [row[2]["ordinal"] for row in rows] == [0, 3, 6, 9]
    assert [row[2]["t_sec"] for row in rows] == [frames[i]["t_sec"] for i in (0, 3, 6, 9)]
    assert all(row[2]["RGB_bytes_sha256"] == hashlib.sha256(r.released.rgb_view(row[1]).tobytes()).hexdigest() for row in rows)
    assert captures[0].released


@pytest.mark.parametrize("change", ["truncate", "wrong_pts", "wrong_pixels"])
def test_native_stream_refuses_changed_inputs_and_releases(tmp_path, monkeypatch, change):
    frames, captures = capture_fixture(tmp_path, monkeypatch, **{change: True})
    with pytest.raises(ValueError): list(r.stream_native(tmp_path, frames, 4))
    assert captures[0].released


def test_failed_native_open_releases_capture(tmp_path, monkeypatch):
    closed = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: SimpleNamespace(isOpened=lambda: False, release=lambda: closed.append(True)))
    with pytest.raises(ValueError, match="unavailable"): list(r.stream_native(tmp_path, [], 4))
    assert closed == [True]


def test_ORB_soft_feature_target_retains_all_actual_points():
    n = 3001
    points = np.column_stack([np.arange(n)%1280, np.arange(n)%720]).astype(np.float32)
    features = {"points": points, "descriptors": np.zeros((n, 32), np.uint8), "scores": np.ones(n, np.float32)}
    r.check_features(features, "orb")
    indices = np.column_stack([np.arange(n), np.arange(n)]).astype(np.int32)
    result = r.core.estimate_pair(points, points.copy(), indices, np.zeros(n, np.float32), r.default_intrinsics(1280, 720))
    assert result["match_count"] == n and result["reason"] == "low_parallax_unknown"
    assert result["n_features_first"] == n


class Tensor:
    """Small NumPy test double, not Torch or a network."""
    def __init__(self, value): self.value = np.asarray(value)
    @property
    def shape(self): return self.value.shape
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def t(self): return Tensor(self.value.T)
    def __matmul__(self, other): return Tensor(self.value@other.value)
    def __getitem__(self, index):
        if isinstance(index, tuple): index = tuple(i.value if isinstance(i, Tensor) else i for i in index)
        return Tensor(self.value[index])
    def __rsub__(self, scalar): return Tensor(scalar-self.value)


def test_indexed_official_match_preserves_negative_cosine_without_cutoff():
    wrapper = object.__new__(r.IndexedXFeat)
    wrapper.runtime = {"pair_match_requests": 0, "actual_mnn_calls": 0}
    wrapper.torch = SimpleNamespace(from_numpy=Tensor)
    calls = []
    def match(a, b, *, min_cossim):
        calls.append(min_cossim)
        return Tensor(np.array([0])), Tensor(np.array([0]))
    wrapper.model = SimpleNamespace(match=match)
    a, b = {"descriptors": np.array([[1., 0.]], np.float32)}, {"descriptors": np.array([[-1., 0.]], np.float32)}
    indices, distance = wrapper.indexed_matches(a, b)
    assert calls == [-1] and indices.tolist() == [[0, 0]] and distance.tolist() == [2.]
    assert wrapper.runtime == {"pair_match_requests": 1, "actual_mnn_calls": 1}


def test_empty_indexed_match_is_retained_without_fabricated_call():
    wrapper = object.__new__(r.IndexedXFeat)
    wrapper.runtime = {"pair_match_requests": 0, "actual_mnn_calls": 0}
    empty = {"descriptors": np.empty((0, 64), np.float32)}
    indices, distance = wrapper.indexed_matches(empty, empty)
    assert indices.shape == (0, 2) and distance.shape == (0,)
    assert wrapper.runtime == {"pair_match_requests": 1, "actual_mnn_calls": 0}


@pytest.mark.parametrize("change_input", [False, True])
def test_actual_pre_hook_binds_RGB_parse_and_network_tensor_without_extra_forwards(change_input):
    wrapper = object.__new__(r.IndexedXFeat)
    wrapper.runtime = {"feature_forward_calls": 0}
    hooks, seen_rgb = [], []
    class Hook:
        def remove(self): hooks.clear()
    def register(fn): hooks.append(fn); return Hook()
    def parse(rgb):
        seen_rgb.append(rgb.copy())
        return Tensor(rgb.transpose(2, 0, 1)[None].astype(np.float32)/255)
    def detect(tensor, *, top_k, detection_threshold):
        assert top_k == 2500 and detection_threshold == .05
        actual = Tensor(tensor.value+1) if change_input else tensor
        for hook in hooks: hook(None, (actual,))
        return [{"keypoints": Tensor(np.array([[1., 1.]], np.float32)),
            "descriptors": Tensor(np.ones((1, 64), np.float32)), "scores": Tensor(np.ones(1, np.float32))}]
    wrapper.model = SimpleNamespace(parse_input=parse, preprocess_tensor=lambda t: (t, 1, 1),
        net=SimpleNamespace(register_forward_pre_hook=register), detectAndCompute=detect)
    bgr = np.tile(np.array([1, 2, 3], np.uint8), (4, 6, 1))
    if change_input:
        with pytest.raises(ValueError, match="actual XFeat input"): wrapper.features(bgr)
    else:
        _, proof = wrapper.features(bgr)
        assert proof["parse_tensor_sha256"] == proof["actual_network_input_sha256"]
        assert proof["parse_shape"] == [1, 3, 4, 6]
    assert seen_rgb[0][0, 0].tolist() == [3, 2, 1]
    assert not hooks and wrapper.runtime["feature_forward_calls"] == 1


def controller_fixture(tmp_path, monkeypatch):
    family = tmp_path/"proof"; family.mkdir()
    frames, captures = capture_fixture(tmp_path, monkeypatch)
    (family/"inputs").mkdir(); (family/"inputs/video.mp4").write_bytes(b"fake decoder")
    put(family/"plan.json", {"synthetic_fixture": True})
    put(family/"inputs_frozen.json", {"artifacts": {"plan.json": r.sha(family/"plan.json"), "inputs/video.mp4": r.sha(family/"inputs/video.mp4")}})
    digest = r.sha(family/"inputs_frozen.json")
    assets = {"root": "fake_assets"}
    plan = {"source_sha256": {}, "source_commit": "pin"}
    bundle = {"frames": frames, "assets": assets, "K": r.default_intrinsics(6, 4)}
    def verify(f, pin, expected):
        r.utility.freeze(f/"inputs_frozen.json", expected)
        return plan, bundle
    monkeypatch.setattr(r, "verify_prepared", verify)
    monkeypatch.setattr(r, "resource_gate", lambda *_: {"synthetic": True})
    points = np.column_stack([np.linspace(.1, 5.8, 20), np.linspace(.1, 3.8, 20)]).astype(np.float32)
    descriptors = np.eye(20, 64, dtype=np.float32)
    constructors = []
    class Model:
        def __init__(self, *_):
            constructors.append(True); self.assets = assets
            self.state = {"sha256": r.STATE_SHA, "element_count": 1551625}
            self.runtime = {"cpu_threads": 2, "interop_threads": 1, "device": "cpu", "lighterglue_constructed": False,
                "feature_forward_calls": 0, "pair_match_requests": 0, "actual_mnn_calls": 0}
        def features(self, pixels):
            self.runtime["feature_forward_calls"] += 1
            return {"points": points.copy(), "descriptors": descriptors.copy(), "scores": np.ones(20, np.float32)}, {
                "parse_tensor_sha256": "a"*64, "actual_network_input_sha256": "b"*64}
        def indexed_matches(self, a, b):
            self.runtime["pair_match_requests"] += 1; self.runtime["actual_mnn_calls"] += 1
            return np.column_stack([np.arange(20), np.arange(20)]).astype(np.int32), np.zeros(20, np.float32)
        def verify_unchanged(self): pass
    class Detector:
        def __getattr__(self, name):
            values = {"getMaxFeatures": 2500, "getScaleFactor": 1.2, "getNLevels": 8, "getEdgeThreshold": 31,
                "getFirstLevel": 0, "getWTA_K": 2, "getScoreType": 0, "getPatchSize": 31, "getFastThreshold": 20}
            return lambda: values[name]
        def detectAndCompute(self, *_):
            return [cv2.KeyPoint(float(x), float(y), 1.) for x, y in points], np.eye(20, 32, dtype=np.uint8)
    class Matcher:
        def match(self, *_): return [cv2.DMatch(i, i, 0.) for i in range(20)]
    monkeypatch.setattr(r, "IndexedXFeat", Model)
    monkeypatch.setattr(cv2, "ORB_create", lambda **kw: Detector())
    monkeypatch.setattr(cv2, "BFMatcher", lambda *_a, **_kw: Matcher())
    return family, digest, constructors, captures


def test_runtime_smoke_freezes_both_full_evidence_packages_without_RGB_duplication(tmp_path, monkeypatch):
    family, digest, constructors, captures = controller_fixture(tmp_path, monkeypatch)
    result = r.execute(family, "pin", digest, smoke=True)
    frozen = r.utility.freeze(family/"smoke/frozen_measurement.json", result["freeze_sha256"])
    assert frozen["methods"] == ["orb", "xfeat"] and frozen["frames"] == 4 and frozen["runtime_only_smoke"] is True
    assert len(list((family/"smoke").glob("*/features/*.npz"))) == 8
    assert len(list((family/"smoke").glob("*/pairs/*.npz"))) == 6
    for method in r.METHODS:
        with np.load(family/f"smoke/motion_{method}.npz", allow_pickle=False) as z:
            assert z["relative_valid"].tolist() == [False]*3 and z["valid"].tolist() == [True, False, False, False]
            assert np.isnan(z["relative_t"]).all() and np.array_equal(z["centers"], np.zeros((4, 3)))
        with np.load(family/f"smoke/{method}/features/frame_0000.npz", allow_pickle=False) as z:
            assert set(z.files) == {"points", "descriptors", "scores"}
    assert len(constructors) == 1 and captures[0].released
    quality = r.load(family/"smoke/quality.json")
    assert quality["selected_frontend"] is None and quality["invalid_is_stop"] is False
    assert not list((family/"smoke").glob("*RGB*"))
    with pytest.raises(FileExistsError): r.execute(family, "pin", digest, smoke=True)


def test_full_requires_accepted_smoke_before_constructor(tmp_path, monkeypatch):
    family, digest, constructors, _ = controller_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="accepted smoke"): r.execute(family, "pin", digest)
    assert not constructors and not (family/"measurement").exists()


def test_small_full_replays_exact_accepted_smoke_and_both_methods(tmp_path, monkeypatch):
    family, digest, constructors, _ = controller_fixture(tmp_path, monkeypatch)
    smoke = r.execute(family, "pin", digest, smoke=True)
    monkeypatch.setattr(r, "FRAME_COUNT", 4)  # Synthetic bounded stand-in for the fixed2950-frame loop.
    full = r.execute(family, "pin", digest, accepted_smoke_sha256=smoke["freeze_sha256"])
    assert full["status"] == "complete" and len(constructors) == 2
    r.utility.freeze(family/"measurement/frozen_measurement.json", full["freeze_sha256"])


def test_input_reblessing_during_execution_cannot_write_master(tmp_path, monkeypatch):
    family, digest, _, _ = controller_fixture(tmp_path, monkeypatch)
    original = r.features_orb; changed = []
    def alter(*args):
        if not changed:
            changed.append(True); put(family/"plan.json", {"modified": True})
            frozen = r.load(family/"inputs_frozen.json"); frozen["artifacts"]["plan.json"] = r.sha(family/"plan.json")
            put(family/"inputs_frozen.json", frozen)
        return original(*args)
    monkeypatch.setattr(r, "features_orb", alter)
    with pytest.raises(ValueError, match="freeze changed"): r.execute(family, "pin", digest, smoke=True)
    assert not (family/"smoke/frozen_measurement.json").exists() and (family/"smoke/failure.json").exists()


def test_resource_refusal_occurs_before_constructor_and_leaves_no_master(tmp_path, monkeypatch):
    family, digest, constructors, _ = controller_fixture(tmp_path, monkeypatch)
    def refuse(*_): raise ValueError("8GiB physical/actual commit refusal")
    monkeypatch.setattr(r, "resource_gate", refuse)
    with pytest.raises(ValueError, match="8GiB"): r.execute(family, "pin", digest, smoke=True)
    assert not constructors and not (family/"smoke/frozen_measurement.json").exists()


def test_missing_second_method_preserves_partial_without_master(tmp_path, monkeypatch):
    family, digest, _, _ = controller_fixture(tmp_path, monkeypatch)
    def fail(*_): raise RuntimeError("synthetic xfeat failure")
    monkeypatch.setattr(r.IndexedXFeat, "features", fail)
    with pytest.raises(RuntimeError, match="xfeat failure"): r.execute(family, "pin", digest, smoke=True)
    assert (family/"smoke/orb/features/frame_0000.npz").exists()
    assert not (family/"smoke/frozen_measurement.json").exists()


def test_quality_reports_unknown_time_span_separately_from_identity_bootstrap():
    summary = r.quality_summary(np.array([False, False, True]), np.array([0, 0, 15]), np.array([0., .1, 4., 4.1]), ["unknown", "unknown", "ok"])
    assert summary["accepted_edge_fraction"] == 1/3 and summary["frame_valid_fraction_including_identity"] == .5
    assert summary["longest_unknown_span_s"] == 4. and summary["unknown_is_not_stop"] is True


@pytest.mark.parametrize("case", ["NaN", "descriptor_dtype", "coordinate", "scores"])
def test_feature_packages_fail_closed(case):
    value = {"points": np.ones((2, 2), np.float32), "descriptors": np.ones((2, 64), np.float32), "scores": np.ones(2, np.float32)}
    if case == "NaN": value["descriptors"][0, 0] = np.nan
    if case == "descriptor_dtype": value["descriptors"] = value["descriptors"].astype(np.float64)
    if case == "coordinate": value["points"][0, 0] = 1280.
    if case == "scores": value["scores"] = value["scores"][:-1]
    with pytest.raises(ValueError): r.check_features(value, "xfeat")


def test_resource_wall_and_resident_memory_limits(monkeypatch):
    monkeypatch.setattr(r.time, "monotonic", lambda: 8000)
    with pytest.raises(ValueError, match="wall"): r.resource_gate(0)
    monkeypatch.setattr(r.time, "monotonic", lambda: 1)
    monkeypatch.setattr(r.utility, "check_headroom", lambda: {})
    monkeypatch.setattr(r.psutil, "Process", lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=4*1024**3, vms=5*1024**3)))
    with pytest.raises(ValueError, match="resident"): r.resource_gate(0)


def prepare_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "NATIVE_COUNT", 10)
    monkeypatch.setattr(r, "source_identity", lambda *_: {})
    monkeypatch.setattr(r.utility, "check_headroom", lambda: {})
    assets = tmp_path/"assets"; assets.mkdir()
    monkeypatch.setattr(r.released, "inspect_assets", lambda *_: {"root": str(assets), "metadata_only": True})
    video = tmp_path/"video"/"input.mp4"; video.parent.mkdir(); video.write_bytes(b"immutable synthetic source")
    native_root = tmp_path/"native"; measurement = native_root/"measurement"; measurement.mkdir(parents=True)
    plan = native_root/"plan.json"; put(plan, {"video_sha256": r.sha(video)})
    protected = {str(p.resolve()): r.sha(p) for p in (video, plan)}
    put(native_root/"inputs_frozen.json", {"protected_sha256": protected})
    put(measurement/"measurement.json", {"inputs_sha256": r.sha(native_root/"inputs_frozen.json")})
    frames = [{"ordinal": i, "t_sec": i/30., "pixels_sha256": "a"*64} for i in range(10)]
    put(measurement/"native_frames.json", frames)
    master = measurement/"measurement_frozen.json"
    put(master, {"status": "complete", "ground_truth_opened": False,
        "artifacts": {p.name: r.sha(p) for p in measurement.iterdir()}})
    proposal = tmp_path/"proposal"/"original.json"; put(proposal, {"storage": "old8GBproposal"})
    monkeypatch.setattr(r, "PRIOR_SHA", r.sha(proposal))
    declared = {"video": {"path": str(video), "sha256": r.sha(video)},
        "native_control": {"freeze": str(master), "sha256": r.sha(master)}, "assets": str(assets),
        "superseded_proposal": {"path": str(proposal), "sha256": r.sha(proposal)}}
    spec = tmp_path/"spec.json"; put(spec, declared)
    return spec, declared, native_root


def test_prepare_is_metadata_only_and_freezes_video_native_identity_and_original_proposal(tmp_path, monkeypatch):
    spec, _, _ = prepare_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(r, "IndexedXFeat", lambda *_: pytest.fail("prepare constructed model"))
    family = tmp_path/"new"
    result = r.prepare(spec, family, "pin")
    frozen = r.utility.freeze(family/"inputs_frozen.json", result["input_freeze_sha256"])
    assert frozen["models_have_not_run"] is True and frozen["ground_truth_or_localization"] is False
    assert {"plan.json", "inputs/video.mp4", "inputs/native_frames.json", "inputs/original_proposal.json"} <= set(frozen["artifacts"])
    assert not (family/"smoke").exists() and not (family/"measurement").exists()
    assert (family/"inputs/original_proposal.json").read_bytes() == Path(r.specification(spec)["superseded_proposal"]["path"]).read_bytes()


@pytest.mark.parametrize("case", ["bad_video", "unregistered_plan", "changed_source", "bad_native_time", "bad_native_ordinal"])
def test_native_input_provenance_and_grid_fail_closed(tmp_path, monkeypatch, case):
    _, declared, native = prepare_fixture(tmp_path, monkeypatch)
    if case == "bad_video": declared["video"]["sha256"] = "f"*64
    elif case == "unregistered_plan":
        frozen = r.load(native/"inputs_frozen.json"); frozen["protected_sha256"].pop(str((native/"plan.json").resolve()))
        put(native/"inputs_frozen.json", frozen)
    elif case == "changed_source": Path(declared["video"]["path"]).write_bytes(b"changed")
    else:
        path = native/"measurement/native_frames.json"; rows = r.load(path)
        if case == "bad_native_time": rows[1]["t_sec"] += .01
        else: rows[1]["ordinal"] = 4
        put(path, rows)
    # Rebind only the synthetic control manifests, so semantic validation is exercised.
    if case in ("unregistered_plan", "bad_native_time", "bad_native_ordinal"):
        summary = native/"measurement/measurement.json"; put(summary, {"inputs_sha256": r.sha(native/"inputs_frozen.json")})
        master = native/"measurement/measurement_frozen.json"; value = r.load(master)
        value["artifacts"] = {p.name: r.sha(p) for p in master.parent.iterdir() if p != master}
        put(master, value); declared["native_control"]["sha256"] = r.sha(master)
    with pytest.raises(ValueError): r.verified_inputs(declared)


@pytest.mark.parametrize("case", ["extra_role", "saved_score", "missing_hash"])
def test_exact_spec_and_label_guards(tmp_path, monkeypatch, case):
    spec, declared, _ = prepare_fixture(tmp_path, monkeypatch)
    if case == "extra_role": declared["unknown"] = {}
    if case == "saved_score": declared["video"]["path"] = "offline_scores.json"
    if case == "missing_hash": declared["native_control"].pop("sha256")
    put(spec, declared)
    with pytest.raises(ValueError): r.specification(spec)
