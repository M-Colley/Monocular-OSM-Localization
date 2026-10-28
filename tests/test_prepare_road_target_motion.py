import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import prepare_road_target_motion as runner
from scripts.prepare_road_target_motion import require_headroom, sha, verified_inputs, verify_family


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def package(tmp_path):
    video = tmp_path/"video.mp4"; video.write_bytes(b"same video input only")
    family = tmp_path/"public_map"; family.mkdir()
    write(family/"original_ocr.json", {"window": [4., 60.], "detections": []})
    write(family/"all_junction_targets.json", {"selected_answer": None, "matches": []})
    write(family/"acquisition_plan.json", {"public_city": "Named City, State", "input_manifest": {
        "ocr": {"sha256": sha(family/"original_ocr.json")}}})
    freeze = family/"frozen_pbf_probe.json"
    write(freeze, {"artifacts": {p.name: sha(p) for p in family.iterdir()}})
    plan = tmp_path/"input_plan.json"
    write(plan, {"clips": [{"video": {"sha256": sha(video), "bytes": video.stat().st_size},
        "window": [4., 60.], "city": "Named City, State", "source_result": "DO_NOT_READ_LABELS_OR_OLD_ROUTES.json"}]})
    return family, sha(freeze), plan, sha(plan), video


def test_same_video_public_scope_and_registered_window_bound_without_old_routes(package):
    result = verified_inputs(*package)
    assert result["window"] == [4., 60.]
    assert result["city"] == "Named City, State"
    assert result["all_targets_sha256"] and result["ocr_sha256"]


@pytest.mark.parametrize("which", ["video", "plan", "targets", "ocr"])
def test_changed_input_bytes_refused(package, which):
    family, frozen_sha, plan, plan_sha, video = package
    target = {"video": video, "plan": plan, "targets": family/"all_junction_targets.json",
              "ocr": family/"original_ocr.json"}[which]
    target.write_bytes(b"changed")
    with pytest.raises(ValueError):
        verified_inputs(family, frozen_sha, plan, plan_sha, video)


def test_different_registered_window_declined_even_with_same_video(package):
    family, frozen_sha, plan, _, video = package
    value = json.loads(plan.read_text()); value["clips"][0]["window"] = [10., 60.]
    write(plan, value)
    with pytest.raises(ValueError, match="identity"):
        verified_inputs(family, frozen_sha, plan, sha(plan), video)


def test_duplicate_video_registration_is_ambiguous(package):
    family, frozen_sha, plan, _, video = package
    value = json.loads(plan.read_text()); value["clips"] *= 2; write(plan, value)
    with pytest.raises(ValueError, match="identity"):
        verified_inputs(family, frozen_sha, plan, sha(plan), video)


def test_frozen_path_cannot_escape_public_family(tmp_path):
    family = tmp_path/"family"; family.mkdir()
    outside = tmp_path/"outside.json"; outside.write_text("{}")
    freeze = family/"frozen.json"; write(freeze, {"artifacts": {"../outside.json": sha(outside)}})
    with pytest.raises(ValueError, match="artifact"):
        verify_family(freeze, sha(freeze))


@pytest.mark.parametrize("physical,commit", [(7.99, 10), (10, 7.99), (float("nan"), 10), (10, float("inf"))])
def test_both_actual_physical_and_commit_headroom_required(physical, commit):
    with pytest.raises(MemoryError):
        require_headroom({"available_physical_gib": physical, "available_commit_gib": commit})


def test_inclusive_eight_gib_gate():
    require_headroom({"available_physical_gib": 8., "available_commit_gib": 8.})


@pytest.mark.parametrize("fault", [None, "pixels", "pts", "ordinal", "seek", "count"])
def test_independent_native_ordinals_do_not_use_vfr_pts_times_fps(monkeypatch, fault):
    import cv2
    # VFR times deliberately do not round to native ordinals2/5/8 at10fps.
    ordinals = [2, 5, 8]
    pts = {2: .27, 5: .73, 8: 1.19}
    images = [np.full((2, 3, 3), i, np.uint8) for i in ordinals]
    frames = SimpleNamespace(frames=images, timestamps=[pts[i] for i in ordinals])
    closed = []
    class Capture:
        current = 0
        def isOpened(self): return True
        def set(self, key, value): self.current = int(value); return fault != "seek"
        def grab(self): self.current += 1; return True
        def read(self):
            im = np.full((2, 3, 3), self.current, np.uint8); self.current += 1
            return True, im + (1 if fault == "pixels" else 0)
        def get(self, key):
            if key == cv2.CAP_PROP_POS_FRAMES: return self.current + (1 if fault == "ordinal" else 0)
            return pts[self.current-1]*1000+(1 if fault == "pts" else 0)
        def release(self): closed.append(True)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: Capture())
    if fault == "count":
        frames.frames = images[:-1]
    kwargs = dict(fps=10., start=.2, end=1., stride=3, total_frames=10)
    if fault:
        with pytest.raises(ValueError):
            runner.verify_native_grid(frames, "mock.mp4", **kwargs)
    else:
        actual, hashes = runner.verify_native_grid(frames, "mock.mp4", **kwargs)
        assert actual.tolist() == ordinals
        assert not np.array_equal(np.rint(np.array(frames.timestamps)*10).astype(int), actual)
        assert len(hashes) == 3
    assert closed or fault == "count"


def test_executed_script_must_be_the_declared_pinned_runner(tmp_path):
    source = tmp_path/"source"; script = source/"scripts/prepare_road_target_motion.py"
    script.parent.mkdir(parents=True); script.write_text("# distinct script")
    with pytest.raises(ValueError, match="executed motion runner"):
        runner.verify_source(source, "pin")


def test_final_source_digest_and_clean_commit_are_rechecked(monkeypatch):
    source = Path(runner.__file__).resolve().parents[1]
    def command(argv, **kwargs):
        return "pin\n" if "rev-parse" in argv else ""
    monkeypatch.setattr(runner.subprocess, "check_output", command)
    relative = "scripts/prepare_road_target_motion.py"
    runner.verify_source(source, "pin", {relative: sha(source/relative)})
    with pytest.raises(ValueError, match="source changed"):
        runner.verify_source(source, "pin", {relative: "0"*64})
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "dirty\n")
    with pytest.raises(ValueError, match="clean"):
        runner.verify_source(source, "pin")
