"""Pure native-grid masking proposal. No decoder, detector or model imports.

The caller must independently prove one completed detector forward on the exact
native frame. Only then may a completed detector's empty output become an
explicit empty (0, 4, 2) array. This function cannot certify detector recall.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np


_PROFILE_JSON = json.dumps({
    "schema": "native-pixel-mask-v1",
    "bands": "ceil(height/4) rows at each top and bottom edge",
    "text": "all returned native-coordinate quadrilaterals; no score selection",
    "text_padding_native_pixels": 4,
    "text_fill": "axis-aligned enclosing rectangle, outward-rounded, inclusive",
    "fill_BGR": [0, 0, 0],
    "max_native_pixels": 16000000,
    "max_detector_boxes": 1000,
    "pixel_coordinate_transform": "identity; no crop/resize/warp/channel conversion",
    "intrinsics": "caller retains unchanged native K; K is not an input or output",
    "scope": "mask proposal; no claim all location text was detected or removed",
}, sort_keys=True, separators=(",", ":"), allow_nan=False)
PROFILE_SHA256 = hashlib.sha256(_PROFILE_JSON.encode("utf-8")).hexdigest()


def profile() -> dict:
    """Return an independent copy of the fixed, shared policy."""
    return json.loads(_PROFILE_JSON)


def mask_native_bgr(native_bgr: np.ndarray, boxes: np.ndarray):
    """Return (new uint8 BGR frame, bool mask, pixel receipt), preserving grid.

    Box input is the completed detector's full N x 4 x 2 output, in native
    (x,y) coordinates. None, nonfinite/out-of-frame coordinates and over-budget
    outputs fail; none are silently treated as empty, rescaled or truncated.
    Rectangle bounds in the receipt are [left, top, right, bottom], inclusive.
    """
    if not isinstance(native_bgr, np.ndarray):
        raise ValueError("native frame must be an ndarray")
    if native_bgr.dtype != np.uint8 or native_bgr.ndim != 3 or native_bgr.shape[2] != 3:
        raise ValueError("native frame must be uint8 H x W x 3 BGR")
    height, width, _ = native_bgr.shape
    if min(height, width) < 1 or height * width > 16000000:
        raise ValueError("native frame pixel budget")
    if not isinstance(boxes, np.ndarray) or boxes.ndim != 3 or boxes.shape[1:] != (4, 2):
        raise ValueError("completed detector boxes must be ndarray N x 4 x 2")
    if len(boxes) > 1000 or boxes.dtype.kind not in "iuf":
        raise ValueError("detector box type/count budget")
    if not np.isfinite(boxes).all():
        raise ValueError("nonfinite detector coordinates")
    if len(boxes) and (
        (boxes[:, :, 0] < 0).any() or (boxes[:, :, 0] > width - 1).any()
        or (boxes[:, :, 1] < 0).any() or (boxes[:, :, 1] > height - 1).any()
    ):
        raise ValueError("detector coordinates must already use the native grid")

    band_rows = (height + 3) // 4
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[:band_rows, :] = True
    mask[height - band_rows:, :] = True
    rectangles = []
    for box in boxes:
        left = max(0, math.floor(float(np.min(box[:, 0]))) - 4)
        right = min(width - 1, math.ceil(float(np.max(box[:, 0]))) + 4)
        top = max(0, math.floor(float(np.min(box[:, 1]))) - 4)
        bottom = min(height - 1, math.ceil(float(np.max(box[:, 1]))) + 4)
        mask[top:bottom + 1, left:right + 1] = True
        rectangles.append([left, top, right, bottom])

    masked = native_bgr.copy(order="C")
    masked[mask] = 0
    receipt = {
        "schema": "native-pixel-mask-receipt-v1",
        "profile_sha256": PROFILE_SHA256,
        "native_shape": list(native_bgr.shape),
        "native_dtype": native_bgr.dtype.str,
        "native_BGR_plain_C_bytes_sha256": hashlib.sha256(native_bgr.tobytes(order="C")).hexdigest(),
        "masked_BGR_plain_C_bytes_sha256": hashlib.sha256(masked.tobytes(order="C")).hexdigest(),
        "mask_bool_plain_C_bytes_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
        "band_rows_each": band_rows,
        "detector_box_count": len(boxes),
        "expanded_inclusive_rectangles": sorted(rectangles),
        "masked_pixel_count": int(np.count_nonzero(mask)),
        "pixel_coordinate_transform": "identity",
        "detector_execution_and_recall_certified_by_this_function": False,
    }
    return masked, mask, receipt
