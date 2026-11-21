"""Native synthetic pixels/injected readers only; no video, model or labels."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from monocular_osm.road_sign_context import ContextConfig, plan_context, recover_context
from monocular_osm.road_sign_clues import extract_road_sign_clues
from monocular_osm.scene_text import SceneText


SHAPE = [120, 200]


def seed(text="186B", *, conf=.95, t=2., box=(80., 50., 100., 60.)):
    return {"text": text, "confidence": conf, "t_sec": t, "bbox": list(box) if box else None}


def detection(text="EXIT", *, box=(60., 20., 100., 30.), confidence=.99):
    x0, y0, x1, y1 = box
    return ([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text, confidence)


def pixels():
    return np.arange(np.prod([*SHAPE, 3]), dtype=np.uint8).reshape(*SHAPE, 3)


def typed(image):
    return hashlib.sha256(str((image.shape, image.dtype.str)).encode() + image.tobytes()).hexdigest()


def run(rows=None, outputs=None, *, config=ContextConfig(), loader=None, reader=None, **kwargs):
    frame = pixels()
    calls = []

    def read(crop):
        calls.append(crop.copy())
        return outputs if outputs is not None else [detection()]

    result = recover_context(rows if rows is not None else [seed()], SHAPE,
        loader or (lambda t: (t, frame)), reader or SimpleNamespace(readtext=read), config=config, **kwargs)
    return result, frame, calls


def test_native_crop_offsets_hashes_time_and_literal_exit_are_preserved():
    original = [seed()]
    before = deepcopy(original)
    result, frame, calls = run(original)
    assert original == before and result["status"] == "complete"
    assert result["plan"]["crops"][0]["rect_xyxy"] == [20, 10, 160, 100]
    assert len(calls) == 1 and calls[0].shape == (90, 140, 3)
    assert np.array_equal(calls[0], frame[10:100, 20:160])
    assert result["crop_records"][0]["crop_to_native"] == {"scale_xy": [1., 1.], "offset_xy": [20, 10]}
    assert result["crop_records"][0]["crop_pixels_sha256"] == typed(calls[0])
    assert result["frame_records"][0]["full_frame_pixels_sha256"] == typed(frame)
    assert result["restored_detections"] == [{"text": "EXIT", "confidence": .99,
        "t_sec": 2., "bbox": [80., 30., 120., 40.]}]
    assert [r["ref"] for r in result["clues"]["observations"]] == ["186B"]
    assert result["clues"] == extract_road_sign_clues(result["merged_detections"], SHAPE)
    assert result["selected_answer"] is None and result["camera_fix"] is None
    assert not result["native_video_sampling_or_origin_verified"]


@pytest.mark.parametrize("outputs", [[detection("186B")], [detection("EXIT", confidence=.899)], []])
def test_bare_numbers_and_low_confidence_context_never_become_verified_exits(outputs):
    result, _, _ = run(outputs=outputs)
    assert result["status"] == "complete" and result["clues"]["observations"] == []


def test_all_literal_alternatives_are_retained_without_ref_selection():
    result, _, _ = run(outputs=[detection(), detection("188", box=(65., 40., 85., 50.))])
    assert result["status"] == "complete"
    assert {r["ref"] for r in result["clues"]["observations"]} == {"186B", "188"}
    assert result["clues"]["selected_answer"] is None


def test_edge_crops_clip_to_native_without_changing_reader_pixels():
    result, frame, calls = run([seed(box=(1., 2., 4., 7.))],
        outputs=[detection(box=(2., 3., 12., 8.))])
    assert result["plan"]["crops"][0]["rect_xyxy"] == [0, 0, 34, 27]
    assert np.array_equal(calls[0], frame[:27, :34])
    assert result["restored_detections"][0]["bbox"] == [2., 3., 12., 8.]


def test_stamp_filter_applies_to_original_and_context_detections():
    result, _, calls = run([seed(), seed("N:42.1234 W:83.4567")],
        outputs=[detection(), detection("60 MPH", box=(1., 1., 10., 8.))])
    assert result["status"] == "complete" and len(calls) == 1
    assert result["plan"]["filtered_stamp_indices"] == [1]
    assert result["crop_records"][0]["filtered_stamp_outputs"] == [1]
    assert not any("MPH" in r["text"] for r in result["merged_detections"])


def test_compact_stamp_exclusion_is_shared_by_original_and_restored_context():
    fabricated = "4025'32\"N7400'11\"W"
    result, _, calls = run([seed(), seed(fabricated)],
        outputs=[detection(), detection(fabricated, box=(1., 1., 10., 8.))])
    assert result["status"] == "complete" and len(calls) == 1
    assert result["plan"]["filtered_stamp_indices"] == [1]
    assert result["crop_records"][0]["filtered_stamp_outputs"] == [1]
    assert not any(row["text"] == fabricated for row in result["merged_detections"])
    assert [row["ref"] for row in result["clues"]["observations"]] == ["186B"]


def test_no_eligible_seed_never_calls_a_reader_or_loader_and_keeps_existing_clues():
    def forbid(*args):
        pytest.fail("unplanned pixels or OCR")

    rows = [seed("186B", conf=.899), seed("I-75"), seed("12AB"), seed("EXIT 7A")]
    result = recover_context(rows, SHAPE, forbid, SimpleNamespace(readtext=forbid))
    assert result["status"] == "complete" and not result["plan"]["crops"]
    assert [r["ref"] for r in result["clues"]["observations"]] == ["7A"]


def test_missing_native_box_is_explicit_and_never_invents_a_crop():
    result, _, calls = run([seed(box=None)])
    assert result["status"] == "complete" and not calls
    assert result["plan"]["unplannable_seed_indices"] == [0]
    assert result["clues"]["observations"] == []


@pytest.mark.parametrize("config,rows", [
    (ContextConfig(max_detections=1), [seed(), seed("188")]),
    (ContextConfig(max_seeds=1), [seed(), seed("188")]),
    (ContextConfig(max_crop_pixels=10), [seed()]),
    (ContextConfig(max_total_crop_pixels=20000), [seed(), seed("188")]),
    (ContextConfig(max_frame_pixels=1000), [seed()]),
    (ContextConfig(max_total_frame_pixels=24000), [seed(), seed("188", t=3.)]),
])
def test_plan_overflow_refuses_before_any_pixel_callback(config, rows):
    def forbid(*args):
        pytest.fail("overflow attempted a selected subset")

    result = recover_context(rows, SHAPE, forbid, SimpleNamespace(readtext=forbid), config=config)
    assert result["status"] == "unavailable" and result["clues"] is None


@pytest.mark.parametrize("config", [ContextConfig(max_outputs_per_crop=1), ContextConfig(max_reader_outputs=1)])
def test_output_overflow_discards_partial_clue_admission(config):
    result, _, _ = run(outputs=iter([detection(), detection("188")]), config=config)
    assert result["status"] == "unavailable" and result["clues"] is None


@pytest.mark.parametrize("fault", ["time", "shape", "dtype", "layout"])
def test_loader_cannot_substitute_times_resize_or_non_native_values(fault):
    frame = pixels()

    def load(t):
        if fault == "time": return t + .001, frame
        if fault == "shape": return t, frame[::2]
        if fault == "dtype": return t, frame.astype(np.float32)
        return t, np.asfortranarray(frame)

    result, _, calls = run(loader=load)
    assert result["status"] == "unavailable" and not calls and result["clues"] is None


@pytest.mark.parametrize("outputs", [[detection(box=(-1., 0., 10., 5.))],
    [detection(box=(1., 1., 2000., 5.))], [detection(confidence=float("nan"))],
    [detection(confidence=".99")], [([], "EXIT", .99)]])
def test_invalid_reader_geometry_or_confidence_cannot_be_repaired_into_evidence(outputs):
    result, _, _ = run(outputs=outputs)
    assert result["status"] == "unavailable" and result["clues"] is None


@pytest.mark.parametrize("fault", ["crop", "frame"])
def test_reader_mutation_cannot_publish_old_pixel_hashes(fault):
    frame = pixels()

    def read(crop):
        if fault == "crop": crop[0, 0, 0] ^= 1
        else: frame[0, 0, 0] ^= 1
        return [detection()]

    result = recover_context([seed()], SHAPE, lambda t: (t, frame), SimpleNamespace(readtext=read))
    assert result["status"] == "unavailable" and result["clues"] is None


def test_deadline_after_reader_and_reader_failure_are_unavailable():
    now = [0.]

    def read(crop):
        now[0] = 1.
        return [detection()]

    result, _, _ = run(reader=SimpleNamespace(readtext=read), config=ContextConfig(max_work_s=.5), clock=lambda: now[0])
    assert result["status"] == "unavailable" and result["clues"] is None

    def fail(crop):
        raise RuntimeError("fake OCR failed")

    result, _, _ = run(reader=SimpleNamespace(readtext=fail))
    assert result["status"] == "unavailable" and result["clues"] is None


def test_scene_text_rows_and_all_seed_order_are_preserved_in_plan():
    rows = [SceneText("188", .99, 3., (80., 50., 100., 60.)), seed("12 A", t=2.)]
    plan = plan_context(rows, SHAPE)
    assert plan["status"] == "planned" and [r["source_index"] for r in plan["crops"]] == [0, 1]
    assert plan["frame_times"] == [2., 3.]


@pytest.mark.parametrize("setting", [{"max_seeds": 65}, {"max_reader_outputs": 4097},
    {"max_work_s": 181.}, {"max_seeds": True}, {"max_work_s": float("nan")}])
def test_config_cannot_raise_fixed_work_bounds(setting):
    with pytest.raises(ValueError):
        ContextConfig(**setting)
