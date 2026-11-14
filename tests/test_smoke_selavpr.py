"""Model-free boundaries for the runtime-only 44-image SelaVPR++ probe."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import smoke_selavpr as smoke


def test_import_has_no_torch_or_upstream_model():
    code = ("import sys; from scripts import smoke_selavpr; assert 'torch' not in sys.modules; "
            "assert not any(n == 'model' or n.startswith('model.') for n in sys.modules)")
    subprocess.run([sys.executable, "-c", code], cwd=Path(smoke.__file__).parents[1], check=True)


def test_fixed_profile_has_first_four_all_views_and_no_accuracy_stage():
    assert smoke.PROFILE["query_positions"] == [0, 1, 2, 3]
    assert smoke.PROFILE["query_views"] == [0, 1, 2]
    assert smoke.PROFILE["queries"] + smoke.PROFILE["references"] == 44
    assert smoke.PROFILE["route_decoding"] is smoke.PROFILE["accuracy_or_label_reads"] is False


def test_selection_is_ID_hash_only_and_then_canonical_column_order():
    rows = [{"column": i, "id": f"mapillary:{i}", "arbitrary_appearance": -i} for i in range(80)]
    expected_ids = {r["id"] for r in sorted(rows, key=lambda r:
        (hashlib.sha256(r["id"].encode()).hexdigest(), r["id"]))[:32]}
    chosen = smoke.selected_columns(rows)
    assert chosen == sorted(chosen) and {rows[i]["id"] for i in chosen} == expected_ids
    for row in rows:
        row["arbitrary_appearance"] = 100000 - row["arbitrary_appearance"]
    assert smoke.selected_columns(rows) == chosen


@pytest.mark.parametrize("rows,count", [([], 32), ([{"column": 1, "id": "a"}], 1),
    ([{"column": 0, "id": "a"}, {"column": 1, "id": "a"}], 1),
    ([{"column": 0, "id": "a"}], True)])
def test_selection_refuses_missing_duplicate_or_unbounded_identity(rows, count):
    with pytest.raises(ValueError):
        smoke.selected_columns(rows, count)


def test_original_freeze_SHA_checked_before_any_member_read(monkeypatch, tmp_path):
    (tmp_path / "inputs_frozen.json").write_text("changed")
    monkeypatch.setattr(smoke.gallery, "verify_freeze", lambda *a: pytest.fail("read members before expected SHA"))
    with pytest.raises(ValueError, match="identity"):
        smoke.verify_gallery(tmp_path, "0" * 64)


def test_query_proof_exact_first_four_crop_grid_and_pixels(tmp_path):
    plan = {"query": {"planned_frame_ordinals": list(range(80)), "native_shape_hw": [2, 3]},
            "per_query_native_pixel_sha256": [f"native{i}" for i in range(80)]}
    rows = []
    for position in range(4):
        for view in range(3):
            pixels = np.full((2, 2, 3), position * 3 + view, np.uint8)
            name = f"query_{position:02d}_view_{view}.npy"
            np.save(tmp_path / name, pixels)
            rows.append({"position": position, "view": view, "file": name,
                "frame_ordinal": position, "native_sha256": f"native{position}", "shape": [2, 2, 3],
                "timestamp_s": float(position),
                "npy_sha256": smoke.sha(tmp_path / name),
                "crop_plain_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
    smoke._validate_query_proof(tmp_path, plan, {"queries": rows}, np.arange(80, dtype=float))
    rows[0]["frame_ordinal"] = 33
    with pytest.raises(ValueError, match="source"):
        smoke._validate_query_proof(tmp_path, plan, {"queries": rows}, np.arange(80, dtype=float))


def test_missing_query_view_declines_before_pixel_load(tmp_path):
    with pytest.raises(ValueError, match="all12"):
        smoke._validate_query_proof(tmp_path, {}, {"queries": []}, [])


def test_main_has_no_route_or_label_arguments(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(smoke.gallery.sparse, "offline", lambda: nullcontext())
    monkeypatch.setattr(smoke, "infer", lambda *a: calls.append(a) or {"status": "fake"})
    assert smoke.main(["infer", "--output", str(tmp_path), "--input-freeze-sha256", "a" * 64,
                       "--source-commit", "pin"]) == 0
    assert calls == [(tmp_path, "a" * 64, "pin")]
    with pytest.raises(SystemExit):
        smoke.main(["infer", "--output", str(tmp_path), "--input-freeze-sha256", "a" * 64,
                    "--source-commit", "pin", "--ground-truth", "labels"])


def _fake_infer(monkeypatch, tmp_path, *, mutate=False):
    output = tmp_path / "family"
    output.mkdir()
    (output / "inputs_frozen.json").write_text("frozen")
    expected = smoke.sha(output / "inputs_frozen.json")
    registration = {"profile": smoke.PROFILE, "profile_sha256": smoke.PROFILE_SHA256,
        "source_commit": "pin", "source_sha256": {"script": "sha"}, "gallery": "gallery",
        "gallery_freeze_sha256": "old", "assets": {"root": "assets"},
        "reference_columns": list(range(32)), "reference_ids": [f"id{i}" for i in range(32)]}
    (output / "registration.json").write_text(json.dumps(registration))
    pixels = np.zeros((2, 2, 3), np.uint8)
    queries = []
    for i in range(12):
        np.save(output / f"q{i}.npy", pixels)
        queries.append({"file": f"q{i}.npy"})
    refs = np.zeros((32, 2, 2, 3), np.uint8)
    events = []
    monkeypatch.setattr(smoke.gallery, "verify_freeze", lambda *a: events.append("freeze"))
    monkeypatch.setattr(smoke, "source_binding", lambda *a: {"script": "sha"})
    monkeypatch.setattr(smoke, "verify_gallery", lambda *a: ({}, None))
    monkeypatch.setattr(smoke.backend, "inspect_assets", lambda *a: {"root": "assets"})
    monkeypatch.setattr(smoke, "verify_samples", lambda *a: (refs, {"queries": queries}))
    monkeypatch.setattr(smoke, "verify_preprocessing", lambda *a: {"canonical_bitwise_equal": True})
    monkeypatch.setattr(smoke.gallery, "close_map", lambda a: events.append("close"))
    monkeypatch.setattr(smoke.gallery.sparse, "check_headroom", lambda: {
        "available_physical_bytes": 9 * 1024**3, "available_commit_bytes": 9 * 1024**3})
    class FakeModel:
        def __init__(self, root):
            self.runtime, self.state = {"forward_calls": 0}, {"sha256": "state"}
        def encode(self, pixels):
            self.runtime["forward_calls"] += 1
            result = np.zeros(2048, np.float32)
            result[0] = 1
            return result
        def verify_unchanged(self):
            events.append("verified-model")
            if mutate:
                (output / "inputs_frozen.json").write_text("reblessed")
    monkeypatch.setattr(smoke.backend, "ReleasedSelaBase", FakeModel)
    return output, expected, events


def test_mocked_infer_retains_all44_descriptors_and_freezes_after_cleanup(monkeypatch, tmp_path):
    output, expected, events = _fake_infer(monkeypatch, tmp_path)
    result = smoke.infer(output, expected, "pin")
    assert result["forwards"] == 44
    assert np.load(output / "inference/descriptors.npy").shape == (44, 2048)
    frozen = json.loads((output / "inference/runtime_frozen.json").read_text())
    assert frozen["no_accuracy_result"] is True and frozen["route_decoding"] is False
    assert events.index("close") < len(events) - 1


def test_in_run_reblessing_cannot_create_master_smoke_freeze(monkeypatch, tmp_path):
    output, expected, events = _fake_infer(monkeypatch, tmp_path, mutate=True)
    with pytest.raises(ValueError, match="changed during"):
        smoke.infer(output, expected, "pin")
    assert "close" in events
    assert not (output / "inference/runtime_frozen.json").exists()


@pytest.mark.parametrize("changed", [False, True])
def test_independent_canonical_preprocessing_check_without_Torch(monkeypatch, changed):
    class Tensor:
        def __init__(self, values): self.values = values
        def unsqueeze(self, axis): return Tensor(np.expand_dims(self.values, axis))
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.values
    canonical = np.ones((3, 2, 2), np.float32)
    monkeypatch.setattr(smoke.gallery.image, "canonical_tensor", lambda rgb: Tensor(canonical))
    actual = np.expand_dims(canonical + int(changed), 0)
    model = SimpleNamespace(input_tensor=lambda bgr: Tensor(actual),
        torch=SimpleNamespace(equal=lambda a, b: np.array_equal(a.values, b.values)))
    if changed:
        with pytest.raises(ValueError, match="preprocessing differs"):
            smoke.verify_preprocessing(model, np.zeros((2, 2, 3), np.uint8))
    else:
        record = smoke.verify_preprocessing(model, np.zeros((2, 2, 3), np.uint8))
        assert record["canonical_bitwise_equal"] is True
        assert record["input_tensor_sha256"] == hashlib.sha256(actual.tobytes()).hexdigest()
