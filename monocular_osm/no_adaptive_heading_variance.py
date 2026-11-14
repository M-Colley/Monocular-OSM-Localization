"""Pure no-adaptive-extra-Q ablation; this is NOT measured VO covariance.

Only the optional native heading-increment variance is replaced. The genuine
filter's fixed bias random walk and coarsener's unknown-motion contribution
are outside this helper and remain required. No cache/source admission, IO,
pose estimation, time resampling, integration or default change occurs here.
"""
from copy import copy as _copy
import hashlib
import json
import math
from types import MappingProxyType
import numpy as np
from .graph_filter import Odometry


PROFILE = MappingProxyType({
    "method": "no-adaptive-native-heading-extra-Q-ablation-v1",
    "maximum_native_increments": 100_000,
    "replacement": "heading_var only: independent Float64 positive zeros",
    "preserved_arrays": ("step", "heading", "dt", "frame_index"),
    "preserved_scalars": ("metric", "unit_seconds"),
    "copy": "genuine Odometry object, independent copies of every declared array; no constructor clipping",
    "fixed_bias_random_walk_deg_per_sqrt_s_untouched": 2.,
    "unknown_motion_sigma_deg_per_sqrt_s_untouched": 5.,
    "uncertainty_source": "NO adaptive extra-Q; not an estimate of geometric precision",
    "measured_covariance": False, "calibrated_covariance": False,
    "native_clock_or_input_admission": False, "historical_source_upgraded": False,
    "original_stats_and_clock_provenance": "future caller must preserve and authenticate original inputs",
    "new_registered_target_required_before_integration": True,
    "policy_admission": False,
})
PROFILE_SHA256 = hashlib.sha256(json.dumps(dict(PROFILE), sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_scalar(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (float, int, np.floating, np.integer))):
        return False
    try:
        return math.isfinite(value) and value > 0
    except (OverflowError, ValueError):
        return False


def _validate(odometry):
    _need(type(odometry) is Odometry, "genuine graph_filter.Odometry required")
    step = odometry.step
    _need(isinstance(step, np.ndarray) and step.dtype == np.float64 and step.ndim == 1
        and 1 <= step.size <= 100_000, "bounded native F64 increment array required")
    count = step.size
    for name in ("heading", "dt"):
        value = getattr(odometry, name)
        _need(isinstance(value, np.ndarray) and value.dtype == np.float64 and value.shape == (count,),
              "same-shaped original F64 means and clock required")
    frames, variance = odometry.frame_index, odometry.heading_var
    _need(isinstance(frames, np.ndarray) and frames.dtype.kind in "iu" and frames.dtype.itemsize <= 8
        and frames.shape == (count,),
          "original integer frame-index array required")
    _need(variance is None or (isinstance(variance, np.ndarray) and variance.dtype == np.float64
        and variance.shape == (count,)), "original variance must be None or same-shaped F64")
    _need(isinstance(odometry.metric, (bool, np.bool_))
        and (odometry.unit_seconds is None or _positive_scalar(odometry.unit_seconds))
        and (bool(odometry.metric) or odometry.unit_seconds is not None),
        "preserve genuine metric flag and declared positive nonmetric unit_seconds")
    # Count/type/shape gates above precede bounded numerical-validation scratch
    # and ALL output allocations. Unknown headings remain NaN, never zero.
    _need(np.isfinite(step).all() and np.all(step >= 0)
        and not np.isinf(odometry.heading).any()
        and np.isfinite(odometry.dt).all() and np.all(odometry.dt > 0)
        and np.all(frames >= 0) and np.all(frames[1:] > frames[:-1]),
        "finite nonnegative steps, positive actual dt and ordered frame indices required")
    _need(variance is None or (np.isfinite(variance).all() and np.all(variance >= 0)),
          "invalid source variance cannot be silently upgraded")
    return count


def without_adaptive_heading_variance(odometry):
    """Return independent declared-array copies with only ``heading_var=0``.

    Validate the complete source before cloning or allocating output arrays.
    Avoid the Odometry constructor: its step clipping can change a signed zero.
    The real class/origin and exact scalar values are preserved by stdlib copy;
    no module origins, source labels or historical statistics are rewritten.
    Additional caller attributes are retained by shallow object copy, but are
    not admitted or mutated here. All declared arrays are independently owned.
    Maximum declared output payload is4MB for100k F64/integer64 increments.
    """
    count = _validate(odometry)
    result = _copy(odometry)
    for name in ("step", "heading", "dt", "frame_index"):
        setattr(result, name, getattr(odometry, name).copy(order="K"))
    result.heading_var = np.zeros(count, np.float64)
    return result
