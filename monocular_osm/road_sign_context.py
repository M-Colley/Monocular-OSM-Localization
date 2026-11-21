"""Bounded native crops for literal road-sign OCR; no model or IO factory.

Numeric text only plans another OCR read. The unchanged road_sign_clues parser
still requires a literal EXIT witness. Overlapping crops are correlated reads,
not independent position observations or proof of native-video provenance.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import time

import numpy as np

from .road_sign_clues import extract_road_sign_clues
from .scene_text import SceneText, _polygon_to_bbox, is_position_stamp


_SEED = re.compile(r"[0-9]{1,4}(?:\s*[A-Z])?", re.I)
_LIMITS = {"max_detections": 4096, "max_seeds": 64, "max_crop_pixels": 4 << 20,
           "max_total_crop_pixels": 32 << 20, "max_frame_pixels": 16 << 20,
           "max_total_frame_pixels": 128 << 20, "max_outputs_per_crop": 256,
           "max_reader_outputs": 4096, "max_polygon_points": 8}


@dataclass(frozen=True)
class ContextConfig:
    max_detections: int = 4096
    max_seeds: int = 64
    max_crop_pixels: int = 4 << 20
    max_total_crop_pixels: int = 32 << 20
    max_frame_pixels: int = 16 << 20
    max_total_frame_pixels: int = 128 << 20
    max_outputs_per_crop: int = 256
    max_reader_outputs: int = 4096
    max_polygon_points: int = 8
    max_work_s: float = 180.

    def __post_init__(self):
        for name, maximum in _LIMITS.items():
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= maximum:
                raise ValueError(f"{name} must be a positive integer <= {maximum}")
        if (isinstance(self.max_work_s, bool) or not isinstance(self.max_work_s, (float, int))
                or not math.isfinite(self.max_work_s) or not 0 < self.max_work_s <= 180.):
            raise ValueError("max_work_s must be finite, positive and <=180")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _pixels(image):
    # All full frames and materialized crops below are C-contiguous uint8 BGR.
    h = hashlib.sha256(str((image.shape, image.dtype.str)).encode())
    h.update(memoryview(image).cast("B"))
    return h.hexdigest()


def _number(value):
    return (isinstance(value, (float, int, np.floating, np.integer))
            and not isinstance(value, (bool, np.bool_)) and math.isfinite(value))


def _box(value, shape):
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(map(_number, value)):
        return None
    x0, y0, x1, y1 = map(float, value)
    return [x0, y0, x1, y1] if 0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0] else None


def _unavailable(result, error):
    return {**result, "status": "unavailable", "clues": None,
            "reason": f"{type(error).__name__}: {error}"[:512]}


def plan_context(detections, frame_shape_hw, *, config=ContextConfig()):
    """Plan every eligible seed before frame loading or reader invocation."""
    result = {"schema": "native-road-sign-context-plan-v1", "config": asdict(config),
        "seed_confidence_min": .9, "horizontal_padding_bbox_heights": 6.,
        "vertical_padding_bbox_heights": 4., "detections": [], "source_indices": [],
        "filtered_stamp_indices": [], "unplannable_seed_indices": [], "crops": []}
    try:
        shape = list(frame_shape_hw)
        if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("actual native height/width required")
        result["frame_shape_hw"] = shape
        if shape[0] * shape[1] > config.max_frame_pixels:
            raise ValueError("native frame pixel budget exceeded")
        total = 0
        for index, supplied in enumerate(detections):
            if index >= config.max_detections:
                raise ValueError("input detection budget exceeded")
            row = asdict(supplied) if isinstance(supplied, SceneText) else dict(supplied)
            text, conf, stamp = row.get("text"), row.get("confidence"), row.get("t_sec")
            if (not isinstance(text, str) or len(text) > 512 or not _number(conf) or not 0 <= conf <= 1
                    or not _number(stamp) or stamp < 0):
                raise ValueError("finite typed OCR text/confidence/time required")
            if is_position_stamp(text):
                result["filtered_stamp_indices"].append(index)
                continue
            box = _box(row.get("bbox"), shape)
            normalized = {"text": text, "confidence": float(conf), "t_sec": float(stamp), "bbox": box}
            result["detections"].append(normalized)
            result["source_indices"].append(index)
            if conf < .9 or not _SEED.fullmatch(text.strip()):
                continue
            if box is None:
                result["unplannable_seed_indices"].append(index)
                continue
            if len(result["crops"]) >= config.max_seeds:
                raise ValueError("seed budget exceeded; no subset read")
            x0, y0, x1, y1 = box
            height = y1 - y0
            rect = [max(0, math.floor(x0 - 6 * height)), max(0, math.floor(y0 - 4 * height)),
                    min(shape[1], math.ceil(x1 + 6 * height)), min(shape[0], math.ceil(y1 + 4 * height))]
            pixels = (rect[2] - rect[0]) * (rect[3] - rect[1])
            total += pixels
            if pixels > config.max_crop_pixels or total > config.max_total_crop_pixels:
                raise ValueError("crop pixel budget exceeded; no subset read")
            result["crops"].append({"ordinal": len(result["crops"]), "source_index": index,
                "seed": deepcopy(normalized), "t_sec": float(stamp), "rect_xyxy": rect,
                "crop_shape_hwc": [rect[3] - rect[1], rect[2] - rect[0], 3],
                "crop_to_native": {"scale_xy": [1., 1.], "offset_xy": rect[:2]}, "pixels": pixels})
        times = sorted({row["t_sec"] for row in result["crops"]})
        if len(times) * shape[0] * shape[1] > config.max_total_frame_pixels:
            raise ValueError("total full-frame hash work budget exceeded")
        result.update(status="planned", frame_times=times, total_crop_pixels=total)
        result["plan_sha256"] = _digest(result)
        return result
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return _unavailable(result, error)


def recover_context(detections, frame_shape_hw, frame_loader, reader, *,
                    config=ContextConfig(), check=lambda: None, clock=time.monotonic):
    """Read planned native crops, restore OCR, and run the unchanged EXIT parser.

    frame_loader(t) returns (actual_t, BGR uint8 C-contiguous frame). The time
    must equal the seed's time. check/clock bounds are cooperative: a blocking
    injected loader/reader still needs the caller's external worker deadline.
    """
    plan = plan_context(detections, frame_shape_hw, config=config)
    result = {"schema": "native-road-sign-context-v1", "status": "unavailable", "plan": plan,
        "frame_records": [], "crop_records": [], "restored_detections": [],
        "selected_answer": None, "camera_fix": None, "confidence_calibrated": False,
        "external_deadline_required_for_blocking_callbacks": True,
        "native_video_sampling_or_origin_verified": False, "correlated_crop_reads": True,
        "clues": None}
    if plan["status"] != "planned":
        return _unavailable(result, ValueError(plan["reason"]))
    start = clock()

    def work():
        check()
        elapsed = clock() - start
        if not math.isfinite(elapsed) or not 0 <= elapsed <= config.max_work_s:
            raise TimeoutError("context OCR work deadline exceeded")

    try:
        groups = defaultdict(list)
        for crop in plan["crops"]:
            groups[crop["t_sec"]].append(crop)
        count = 0
        for stamp in plan["frame_times"]:
            work()
            actual_time, frame = frame_loader(stamp)
            work()
            if (not _number(actual_time) or float(actual_time) != stamp or type(frame) is not np.ndarray
                    or frame.dtype != np.uint8 or list(frame.shape) != [*plan["frame_shape_hw"], 3]
                    or not frame.flags.c_contiguous):
                raise ValueError("loader must return exact-time native C uint8 BGR pixels")
            frame_sha = _pixels(frame)
            result["frame_records"].append({"t_sec": stamp, "shape": list(frame.shape),
                "dtype": frame.dtype.str, "full_frame_pixels_sha256": frame_sha})
            for crop in groups[stamp]:
                work()
                x0, y0, x1, y1 = crop["rect_xyxy"]
                pixels = np.ascontiguousarray(frame[y0:y1, x0:x1])
                before = _pixels(pixels)
                record = {**deepcopy(crop), "full_frame_pixels_sha256": frame_sha,
                          "crop_pixels_sha256": before, "restored_indices": [], "filtered_stamp_outputs": []}
                result["crop_records"].append(record)
                outputs = reader.readtext(pixels)
                work()
                for offset, item in enumerate(outputs):
                    if offset >= config.max_outputs_per_crop or count >= config.max_reader_outputs:
                        raise ValueError("reader output budget exceeded; no partial clues")
                    count += 1
                    polygon, text, confidence = item
                    if (not isinstance(text, str) or len(text) > 512 or not _number(confidence)
                            or not 0 <= confidence <= 1 or not isinstance(polygon, (list, tuple, np.ndarray))
                            or not 2 <= len(polygon) <= config.max_polygon_points):
                        raise ValueError("invalid bounded reader text/confidence/polygon")
                    if any(not isinstance(point, (list, tuple, np.ndarray)) or len(point) != 2
                           or not all(map(_number, point)) for point in polygon):
                        raise ValueError("reader polygon must contain finite pixel pairs")
                    box = _box(_polygon_to_bbox(polygon), pixels.shape[:2])
                    if box is None:
                        raise ValueError("reader box escaped or degenerated within crop")
                    row = {"text": text.strip(), "confidence": float(confidence), "t_sec": stamp,
                           "bbox": [box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0]}
                    if is_position_stamp(row["text"]):
                        record["filtered_stamp_outputs"].append(offset)
                    else:
                        record["restored_indices"].append(len(result["restored_detections"]))
                        result["restored_detections"].append(row)
                work()
                if _pixels(pixels) != before:
                    raise ValueError("reader mutated its native crop pixels")
            work()
            if _pixels(frame) != frame_sha:
                raise ValueError("native frame pixels changed during crop reads")
        merged = plan["detections"] + result["restored_detections"]
        work()
        clues = extract_road_sign_clues(merged, plan["frame_shape_hw"])
        if _digest({key: value for key, value in plan.items() if key != "plan_sha256"}) != plan["plan_sha256"]:
            raise ValueError("context plan changed during callbacks")
        result.update(status="complete", reason=None, reader_output_count=count, merged_detections=merged,
            clues=clues, elapsed_s=clock() - start,
            pixel_hash_schema="sha256(repr((shape,dtype.str))+C_bytes)")
        if not math.isfinite(result["elapsed_s"]) or result["elapsed_s"] > config.max_work_s:
            raise TimeoutError("context OCR work deadline exceeded")
        return result
    except (ValueError, TypeError, KeyError, IndexError, OSError, RuntimeError) as error:
        return _unavailable(result, error)
