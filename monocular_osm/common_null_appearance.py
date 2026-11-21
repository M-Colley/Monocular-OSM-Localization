"""A NEW pure spatial density-ratio target, not calibrated image likelihood.

No IO, native/evidence admission, filter state, or inference occurs here. The
caller supplies the complete grid and all authenticated raw columns/weights.
Sample returned logpsi DIRECTLY before temporal powering; the old row-max/.1
compatibility wrapper would define a different target.
"""
from types import MappingProxyType
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from . import poi_alignment as poi


PROFILE = MappingProxyType({
    "method": "common-null-globally-normalized-appearance-density-ratio-v1",
    "new_target": True,
    "query_rows": 80, "max_reference_slots": 96_000,
    "max_field_values": 16_000_000, "max_grid_cells": 200_000,
    "field_dtype": "float32", "max_field_bytes": 64_000_000,
    "statistics_dtype": "float64", "histogram_dtype": "float64",
    "row_mean_std": "weighted Float64 population mean/std over ALL original columns",
    "row_std_floor": .005, "z_clip": 6.,
    "field_cell_m": 100., "appearance_sigma_m": 150.,
    "gaussian_mode": "constant", "gaussian_truncate": 3.,
    "reference_binning": "nearest-even np.rint on the complete caller grid",
    "N": "Gaussian(sum_s weight_s*exp(z_qs))",
    "D": "Gaussian(sum_s weight_s)",
    "normalization": "actual Float64 Gaussian GRID sums, including finite boundary loss",
    "uniform_background": ".1/M shared by query and null, on same entire M-cell grid",
    "log_factor": "log(.1/M+.9*N/sumN)-log(.1/M+.9*D/sumD)",
    "outside_log_factor": 0.,
    "sampling": "DIRECT logpsi then caller's original temporal power; no second mixture/max wrapper",
    "all_one_weights": "same NEW Float64 math; no old Float32 exact-equivalence claim",
    "weight_semantics": "caller-supplied original positive snapshot weights; never renormalized or mutated as data",
    "imagelikelihood_calibrated": False, "native_admission_provided": False,
    "historical_encoder_verified": False, "policy_admission": False,
})
PROFILE_SHA256 = poi._json_digest(dict(PROFILE))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _mass_grid(value, name):
    array = np.asarray(value)
    _need(array.ndim <= 2 and 1 <= array.size <= PROFILE["max_grid_cells"]
          and array.dtype.kind in "fiu", name + " must be a bounded real scalar/1D/2D grid")
    array = np.asarray(array, np.float64)
    _need(np.isfinite(array).all() and np.all(array >= 0), name + " must be finite nonnegative mass")
    return array


def _log_mixture_inplace(values):
    """Convert owned F64 nonnegative grid mass to log(.1/M+.9*f)."""
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.sum(values, dtype=np.float64))
    _need(math.isfinite(total) and total > 0, "positive finite actual Gaussian grid sum required")
    values /= total
    values *= .9
    values += .1 / values.size
    np.log(values, out=values)
    return total


def common_null_log_ratio(numerator, density):
    """F64 log ratio of two globally normalized common-background mixtures.

    Both input mass grids remain untouched. Scalar/1D grids are useful for
    private quadrature/probability checks; the field builder uses 2D grids.
    M is the ENTIRE grid size, not the supported-cell count. Real-arithmetic
    ratio bounds are [1/(1+9M),1+9M], with ordinary F64 rounding tolerance.
    N=cD is neutral; no empirical image-likelihood calibration is asserted.
    """
    n, d = _mass_grid(numerator, "numerator"), _mass_grid(density, "density")
    _need(n.shape == d.shape, "query and null must use the same complete grid")
    result, null = n.copy(order="C"), d.copy(order="C")
    _log_mixture_inplace(result)
    _log_mixture_inplace(null)
    result -= null
    return result


def _inputs(reference_xy, scores, weights, times, origin_xy, shape, check):
    _need(isinstance(shape, (tuple, list)) and len(shape) == 2
        and all(not isinstance(v, (bool, np.bool_)) and isinstance(v, (int, np.integer)) and v >= 2 for v in shape),
        "two positive integer full-grid dimensions required")
    shape = tuple(int(v) for v in shape)
    _need(80 * math.prod(shape) <= PROFILE["max_field_values"], "complete80 field exceeds16M values; no crop")
    _need(isinstance(scores, np.ndarray) and scores.dtype == np.float32
        and scores.ndim == 2 and scores.shape[0] == 80 and scores.flags.c_contiguous
        and 1 <= scores.shape[1] <= PROFILE["max_reference_slots"],
        "ALL bounded C-F32 original80 raw columns required")
    count = scores.shape[1]
    _need(isinstance(reference_xy, np.ndarray) and reference_xy.dtype == np.float64
        and reference_xy.shape == (count, 2) and np.isfinite(reference_xy).all()
        and isinstance(weights, np.ndarray) and weights.dtype == np.float64
        and weights.shape == (count,) and np.isfinite(weights).all() and np.all(weights > 0),
        "finite original F64 reference positions and positive snapshot weights required")
    _need(isinstance(times, np.ndarray) and times.dtype == np.float64 and times.shape == (80,)
        and np.isfinite(times).all() and np.all(np.diff(times) > 0), "exact ordered F64 native80 times required")
    origin = np.asarray(origin_xy)
    _need(origin.shape == (2,) and origin.dtype.kind in "fiu" and np.isfinite(origin).all(),
          "finite full-grid origin required")
    origin = np.asarray(origin, np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        mass = float(np.sum(weights, dtype=np.float64))
        rounded = np.rint((reference_xy - origin) / 100.)
    _need(math.isfinite(mass) and 0 < mass < np.finfo(np.float64).max/(8*math.exp(6)),
          "positive finite numerically representable complete weight mass required")
    _need(np.isfinite(rounded).all() and np.all(rounded >= 0)
        and np.all(rounded[:, 0] < shape[1]) and np.all(rounded[:, 1] < shape[0]),
        "every original reference must be inside the complete grid")
    # Row validation never allocates a full80-by-reference F64/bool cube.
    for q in range(80):
        check()
        _need(np.isfinite(scores[q]).all() and np.max(np.abs(scores[q])) <= 1.0001,
              "finite original cosine scores required")
    return shape, origin, rounded.astype(np.int64), mass


def build_common_null_field(reference_xy, raw_similarities, snapshot_weights, query_times, *,
                            origin_xy, shape, check=lambda: None, provenance=None):
    """Return genuine RasterField(F32 logpsi, outside0), using ALL raw inputs.

    Every row, including all-one weights, uses weighted F64 population moments.
    The new builder has no native/evidence admission or complete-road extent
    authority: the caller supplies/authenticates the full grid. Any reference
    outside it refuses the WHOLE field. No subset, crop or input writes occur.

    Owned numeric payload at the16M cap:64MB F32 output; one1.6MB F64 shared
    null grid (D reused), one1.6MB F64 query grid (N reused), <=.768MB score-row
    scratch, <=1.536MB integer reference bins, plus small validation/kernel
    temporaries. No F64 query cube is constructed. Caller owns input storage
    and enforces process resources/cooperative callback deadlines.
    """
    check()
    scores, weights, times = raw_similarities, snapshot_weights, query_times
    shape, origin, indices, mass = _inputs(reference_xy, scores, weights, times, origin_xy, shape, check)
    input_sha256 = {"reference_xy": poi._digest(reference_xy), "raw_similarities": poi._digest(scores),
                    "snapshot_weights": poi._digest(weights), "query_times": poi._digest(times)}
    density = np.zeros(shape, np.float64)
    np.add.at(density, (indices[:, 1], indices[:, 0]), weights)
    gaussian_filter(density, sigma=1.5, output=density, mode="constant", truncate=3.)
    density_sha256 = poi._digest(density)
    # D is owned scratch. Reuse its storage for the fixed shared log-null.
    density_sum = _log_mixture_inplace(density)
    values = np.empty((80, *shape), np.float32)
    row = np.empty(len(weights), np.float64)
    numerator_sums = []
    for q in range(80):
        check()
        np.copyto(row, scores[q])
        mean = float(np.dot(weights, row)/mass)
        row -= mean
        np.square(row, out=row)
        std = math.sqrt(float(np.dot(weights, row)/mass))
        np.copyto(row, scores[q])
        row -= mean
        row /= max(std, .005)
        np.clip(row, -6., 6., out=row)
        np.exp(row, out=row)
        row *= weights
        numerator = np.zeros(shape, np.float64)
        np.add.at(numerator, (indices[:, 1], indices[:, 0]), row)
        gaussian_filter(numerator, sigma=1.5, output=numerator, mode="constant", truncate=3.)
        numerator_sums.append(_log_mixture_inplace(numerator))
        numerator -= density
        values[q] = numerator
        del numerator
    source = {"kind": "common-null globally normalized Gaussian density-ratio target",
        "profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256,
        "input_sha256": input_sha256, "reference_slots": len(weights), "weight_mass": mass,
        "actual_gaussian_density_sum": density_sum, "actual_gaussian_numerator_sums": numerator_sums,
        "Gaussian_D_sha256_before_shared_null": density_sha256,
        "shared_log_null_sha256": poi._digest(density), "all_one_weights_also_F64": True,
        "caller": dict(provenance or {}), "imagelikelihood_calibrated": False,
        "native_admission_provided": False, "policy_admission": False}
    check()
    return poi.RasterField(values, tuple(origin), 100., 0., times.copy(), source)
