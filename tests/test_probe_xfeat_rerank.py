import hashlib
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
import probe_xfeat_rerank as p


def test_only_local_extractor_matcher_change_from_frozen_orb_profile():
    for key in ("top_k", "candidate_order", "query_views", "image_size_wh", "resize",
                "geometry", "coverage", "score", "view_pooling", "tie_breaks", "zero_evidence", "temporal_decoder"):
        assert p.PROFILE[key] == p.ORB_PROFILE[key]
    assert p.PROFILE["orb"] is None
    assert p.PROFILE["xfeat"]["top_k"] == 2500
    assert p.PROFILE["xfeat"]["detection_threshold"] == .05
    assert p.PROFILE["matching"]["min_cossim"] == -1
    assert p.PROFILE["parent_geometry_profile_sha256"] == p.digest(p.ORB_PROFILE)


def test_correspondence_digest_binds_order_values_and_shape():
    a = np.array([[1, 2], [3, 4]], np.float32)
    b = a+1
    assert p.points_sha(a, b) != p.points_sha(a[::-1], b[::-1])
    assert p.points_sha(a, b) != p.points_sha(a, b+1)
    assert len(p.points_sha(np.empty((0, 2)), np.empty((0, 2)))) == 64


@pytest.fixture
def job(tmp_path, monkeypatch):
    output = tmp_path/"output"
    output.mkdir()
    for name in p.SOURCES:
        path = output/"source_snapshot"/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p.ROOT/name, path)
    p.write(output/"preregistration.json", {"source_commit": "synthetic-pin"})
    scores = np.arange(16, dtype=np.float32).reshape(2, 8)
    ids = np.array([f"mapillary:{i}" for i in range(8)])
    xy = np.column_stack((np.arange(8), np.arange(8))).astype(float)
    evidence = {"similarities": scores, "ref_ids": ids, "ref_latlon": xy,
                "query_times": np.array([0., 1.])}
    matrix = tmp_path/"matrix.npz"
    np.savez(matrix, **evidence)
    raw_store = tmp_path/"raw.npz"
    raw = np.stack([np.full((4, 4, 3), i, np.uint8) for i in range(8)])
    np.savez(raw_store, raw=raw, ids=np.array([str(i) for i in range(8)]), xy=xy)
    registration = {}
    for name in ("frozen_prediction", "recovery_lineage", "video"):
        path = tmp_path/name
        path.write_bytes(b"frozen input")
        registration[name] = {"path": str(path), "sha256": p.file_sha256(path)}
    registration["matrix"] = {"path": str(matrix), "sha256": p.file_sha256(matrix)}
    registration["raw_store"] = {"path": str(raw_store), "sha256": p.file_sha256(raw_store)}
    frames = [np.full((4, 6, 3), 100+i, np.uint8) for i in range(2)]
    digest = hashlib.sha256()
    for image in frames:
        digest.update(str((image.shape, image.dtype.str)).encode())
        digest.update(memoryview(image).cast("B"))
    query = {"pixels_sha256": digest.hexdigest()}
    monkeypatch.setattr(p, "stream_queries", lambda *args: iter(enumerate(frames)))
    models = []
    class Backend:
        def __init__(self, root):
            self.assets = {"root": root}
            self.state = {"sha256": "synthetic"}
            self.runtime = {"feature_forward_calls": 0, "pair_match_requests": 0}
            models.append(self)
        def features(self, pixels):
            self.runtime["feature_forward_calls"] += 1
            points = np.array([[30+i*20, 40+i*20] for i in range(8)], np.float32)
            return points, np.full((8, 64), float(pixels.mean()), np.float32)
        def correspondences(self, a, b):
            self.runtime["pair_match_requests"] += 1
            return a[0], b[0]+float(b[1][0, 0])
        def verify_unchanged(self): pass
    monkeypatch.setattr(p, "ReleasedXFeat", Backend)
    def verify(pa, pb, shape_a, shape_b):
        assert shape_a == shape_b == (512, 512)
        score = float(pb[0, 0]-30)
        return {"status": "ok", "match_count": 8, "inlier_count": 8,
                "query_coverage": score/8, "reference_coverage": score/8, "score": score}
    monkeypatch.setattr(p, "verify_geometry", verify)
    monkeypatch.setattr(p.subprocess, "check_output", lambda command, **kwargs:
        "synthetic-pin\n" if "rev-parse" in command else "")
    return output, registration, evidence, query, models, frames


def test_mocked_released_boundary_freezes_every_pair_control_and_bounded_feature_reuse(job):
    output, registration, evidence, query, models, _ = job
    p.execute(output, registration, evidence, query, {"root": "synthetic-upstream"})
    prediction = p.read(output/"diagnostic_prediction.json")
    control = p.read(output/"centered_top1_control.json")
    records = p.read(output/"pair_metrics.json")
    assert prediction["point_count"] == 2 and prediction["pair_view_count"] == len(records) == 48
    assert [item["selected_ref_index"] for item in prediction["selections"]] == [7, 7]
    assert control["chosen_ref_ids"] == ["mapillary:0", "mapillary:0"]
    assert models[0].runtime == {"feature_forward_calls": 14, "pair_match_requests": 48}
    assert not (output/"staging/raw.npy").exists()
    freeze = p.read(output/"frozen_prediction.json")
    assert freeze["ground_truth_opened"] is False and freeze["policy_adopted"] is False
    for name, expected in freeze["artifacts"].items():
        assert p.file_sha256(output/name) == expected


def test_default_preflight_does_not_construct_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(p, "inspect_inputs", lambda root: ({}, {}, {}))
    monkeypatch.setattr(p, "inspect_assets", lambda root: {})
    monkeypatch.setattr(p, "ReleasedXFeat", lambda *args: pytest.fail("no preflight model"))
    monkeypatch.setattr(sys, "argv", ["runner", "--input-run", str(tmp_path), "--assets", str(tmp_path)])
    assert p.main() == 0
    assert '"profile_sha256"' in capsys.readouterr().out


def test_actual_execution_requires_reviewed_pin_before_constructor(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "inspect_inputs", lambda root: ({}, {}, {}))
    monkeypatch.setattr(p, "inspect_assets", lambda root: {})
    monkeypatch.setattr(p.subprocess, "check_output", lambda command, **kwargs:
        "actual-pin\n" if "rev-parse" in command else "")
    monkeypatch.setattr(p, "ReleasedXFeat", lambda *args: pytest.fail("constructor must stay blocked"))
    monkeypatch.setattr(sys, "argv", ["runner", "--input-run", str(tmp_path), "--assets", str(tmp_path),
        "--execute", "--source-commit", "wrong", "--profile-sha256", p.digest(p.PROFILE)])
    with pytest.raises(ValueError, match="clean explicit source/profile"):
        p.main()


def test_foreign_driver_cannot_borrow_helper_checkout_identity(tmp_path, monkeypatch):
    foreign = tmp_path/"copied.py"
    foreign.write_text("# foreign", encoding="utf8")
    monkeypatch.setattr(p, "__file__", str(foreign))
    monkeypatch.setattr(p, "inspect_inputs", lambda *args: pytest.fail("must reject before input work"))
    with pytest.raises(ValueError, match="outside its recorded source"):
        p.main()


@pytest.mark.parametrize("kind", ["partial", "wrong_order", "pixels", "changed_input"])
def test_incomplete_or_changed_inputs_never_produce_a_final_freeze(job, monkeypatch, kind):
    output, registration, evidence, query, models, frames = job
    if kind == "partial": monkeypatch.setattr(p, "stream_queries", lambda *args: iter([(0, frames[0])]))
    if kind == "wrong_order": monkeypatch.setattr(p, "stream_queries", lambda *args: iter([(1, frames[0]), (0, frames[1])]))
    if kind == "pixels": query["pixels_sha256"] = "0"*64
    if kind == "changed_input":
        old = p.ReleasedXFeat
        class Changed(old):
            def verify_unchanged(self): Path(registration["matrix"]["path"]).write_bytes(b"changed")
        monkeypatch.setattr(p, "ReleasedXFeat", Changed)
    with pytest.raises(ValueError):
        p.execute(output, registration, evidence, query, {"root": "synthetic-upstream"})
    assert not (output/"frozen_prediction.json").exists()
    assert not (output/"staging/raw.npy").exists()


def test_main_disables_cuda_before_any_constructor_even_in_default_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(p, "inspect_inputs", lambda root: ({}, {}, {}))
    monkeypatch.setattr(p, "inspect_assets", lambda root: {})
    monkeypatch.setattr(p.subprocess, "check_output", lambda command, **kwargs:
        "pin\n" if "rev-parse" in command else "")
    monkeypatch.setattr(p, "ReleasedXFeat", lambda *args: pytest.fail("no constructor before valid output"))
    monkeypatch.setattr(sys, "argv", ["runner", "--input-run", str(tmp_path), "--assets", str(tmp_path),
        "--output", str(tmp_path/"out"), "--execute", "--source-commit", "pin", "--profile-sha256", p.digest(p.PROFILE)])
    with pytest.raises(ValueError, match="independent owned E-drive"):
        p.main()
    assert p.os.environ["CUDA_VISIBLE_DEVICES"] == "-1"


def test_changed_head_after_execution_prevents_final_freeze(job, monkeypatch):
    output, registration, evidence, query, _, _ = job
    monkeypatch.setattr(p.subprocess, "check_output", lambda command, **kwargs:
        "different-pin\n" if "rev-parse" in command else "")
    with pytest.raises(ValueError, match="HEAD changed"):
        p.execute(output, registration, evidence, query, {"root": "synthetic-upstream"})
    assert not (output/"frozen_prediction.json").exists()
