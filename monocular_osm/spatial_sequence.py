"""Experimental full-support spatial smoothing of complete retrieval scores.

Training-free, posthoc DEV; not a TRAIL reproduction. No models, files,
network, VO, map, labels or user spatial prior. Internal posterior masses
are uncalibrated. The primary route follows posterior peaks, never means
between modes. All supplied references contribute to the appearance field.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import numpy as np
from .geo import LocalFrame
from .poi_alignment import MAX_FIELD_VALUES, POIAlignmentConfig, build_appearance_field

ALGORITHM_VERSION = "full-support-spatial-gaussian-v1"
VARIANTS = ("raw", "reference_mean")


@dataclass(frozen=True)
class SpatialSequenceConfig:
    field_cell_m: float = 100.0
    padding_m: float = 2000.0
    appearance_sigma_m: float = 150.0
    row_std_floor: float = 0.005
    z_clip: float = 6.0
    empty_appearance: float = -6.0
    emission_log_scale: float = 1.0
    motion_three_sigma_speed_mps: float = 40.0
    motion_truncate_sigma: float = 4.0
    recovery_time_s: float = 120.0
    mode_separation_m: float = 500.0
    retained_modes: int = 8
    max_queries: int = 80
    max_references: int = 100000
    max_matrix_values: int = 8000000
    max_field_values: int = MAX_FIELD_VALUES
    max_kernel_radius: int = 4096
    max_interval_s: float = 3600.0
    max_cell_m: float = 6400.0

    def __post_init__(self):
        integers = {"retained_modes", "max_queries", "max_references", "max_matrix_values", "max_field_values", "max_kernel_radius"}
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and real")
            if name in integers:
                if int(value) != value or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
                object.__setattr__(self, name, int(value))
            else:
                if name != "empty_appearance" and value <= 0:
                    raise ValueError(f"{name} must be positive")
                object.__setattr__(self, name, float(value))
        if (self.max_queries > 80 or self.max_references > 100000 or self.max_matrix_values > 8000000
                or self.max_field_values > MAX_FIELD_VALUES or self.max_queries < 2
                or self.max_field_values < 8 or self.retained_modes > 32 or self.max_kernel_radius > 4096
                or self.max_cell_m < self.field_cell_m or self.empty_appearance > 0
                or self.emission_log_scale * max(self.z_clip, abs(self.empty_appearance)) > 300):
            raise ValueError("configuration exceeds fixed allocation/numeric bounds")

    def record(self):
        return {"algorithm": ALGORITHM_VERSION, "validation_status": "posthoc_dev_unvalidated", "config": asdict(self),
            "variants": list(VARIANTS), "primary_route": "row-major argmax of saved float32 posterior; no mean or smoothing",
            "motion": "zero-mean isotropic Gaussian; sigma_m=(three_sigma_speed/3)*actual_dt; stops allowed",
            "boundary": "constant-zero convolution normalized at each source state; no wrap/reflection",
            "kernel_radius": "per axis min(size-1,ceil(truncate_sigma*sigma_cells)); counted before allocation",
            "recovery": "r=-expm1(-actual_dt/recovery_time_s); transition=(1-r)*motion+r*uniform_grid",
            "prior": "uniform over input-derived reference extent plus padding",
            "appearance": "existing all-reference Gaussian log-mean-exp; canonical ID order; no pruning",
            "bounds": "all reference extent+padding; cell doubled only for field cap; fail above max_cell_m",
            "inference": "sum-product forward/backward, all spatial states retained",
            "confidence": None, "confidence_calibrated": False, "unlocalized_probability": None,
            "recovery_is_calibrated_abstention": False}


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class GaussianTransition:
    """Finite-grid row-stochastic motion and its exact backward operator.

    Boundary conditioning makes the transition asymmetric despite a
    symmetric Gaussian kernel. Forward convolves mass/source_normalizer;
    backward divides convolved likelihood by source_normalizer.
    """
    def __init__(self, shape, cell_m, interval_s, config=None):
        from scipy.ndimage import gaussian_filter
        config = config or SpatialSequenceConfig()
        if (len(shape) != 2 or any(isinstance(v, bool) or int(v) != v or v < 2 for v in shape)
                or math.prod(shape) > config.max_field_values or not np.isfinite(cell_m) or cell_m <= 0
                or not np.isfinite(interval_s) or not 0 < interval_s <= config.max_interval_s):
            raise ValueError("bounded grid, positive cell size and actual interval required")
        self.shape = tuple(int(v) for v in shape)
        self.interval_s = float(interval_s)
        self.sigma_cells = (config.motion_three_sigma_speed_mps / 3) * (float(interval_s) / float(cell_m))
        if not np.isfinite(self.sigma_cells) or self.sigma_cells <= 0:
            raise ValueError("motion sigma must be finite and positive")
        extent = config.motion_truncate_sigma * self.sigma_cells
        if not np.isfinite(extent):
            raise ValueError("motion kernel extent overflow")
        # Count before allocation; offsets beyond finite extent connect no states.
        self.radius = tuple(min(size - 1, math.ceil(extent)) for size in self.shape)
        if max(self.radius) > config.max_kernel_radius:
            raise ValueError("motion kernel exceeds fixed radius budget")
        self.recovery = float(-np.expm1(-float(interval_s) / config.recovery_time_s))
        self._filter = gaussian_filter
        self.source_normalizer = self._convolve(np.ones(self.shape, dtype=np.float64))
        if not np.isfinite(self.source_normalizer).all() or np.any(self.source_normalizer <= 0):
            raise ValueError("invalid transition normalization")

    def _convolve(self, values):
        return self._filter(values, sigma=self.sigma_cells, radius=self.radius,
                            mode="constant", cval=0., output=np.float64)

    def _values(self, values):
        value = np.asarray(values, dtype=np.float64)
        if value.shape != self.shape or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("transition values must be finite nonnegative grid masses")
        return value

    def forward(self, values):
        value = self._values(values)
        return ((1 - self.recovery) * self._convolve(value / self.source_normalizer)
                + self.recovery * value.sum() / value.size)

    def backward(self, values):
        value = self._values(values)
        return ((1 - self.recovery) * self._convolve(value) / self.source_normalizer
                + self.recovery * value.mean())

    def record(self):
        return {"interval_s": self.interval_s, "sigma_cells": self.sigma_cells,
                "radius": list(self.radius), "recovery_mass": self.recovery}


def _normalize(values):
    total = float(values.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("spatial evidence has no finite positive mass")
    return values / total, total


def _modes(posterior, cell_m, config):
    """Separated diagnostic peaks; all mass remains in the saved posterior."""
    height, width = posterior.shape
    work = posterior.astype(np.float64, copy=True)
    radius = min(math.hypot(height, width), config.mode_separation_m / cell_m)
    result = []
    for _ in range(config.retained_modes):
        flat = int(work.argmax())
        if work.flat[flat] < 0:
            break
        row, col = divmod(flat, width)
        result.append({"row": row, "col": col, "posterior_mass": float(posterior[row, col])})
        reach = min(max(height, width), math.ceil(radius))
        y0, y1, x0, x1 = max(0, row-reach), min(height, row+reach+1), max(0, col-reach), min(width, col+reach+1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy-row)**2 + (xx-col)**2 <= radius**2
        work[y0:y1, x0:x1][mask] = -1.
    return result


def infer_spatial_posterior(log_appearance, query_times, cell_m, *, config=None):
    """Normalized smoothed rows and fixed peak route for [query,row,col].

    Recovery resets transition mass; it is not a calibrated null state.
    """
    config = config or SpatialSequenceConfig()
    field = np.asarray(log_appearance)
    times = np.asarray(query_times, dtype=np.float64)
    if (field.ndim != 3 or field.dtype.kind not in "fiu" or not 2 <= len(field) <= config.max_queries
            or min(field.shape[1:]) < 2 or field.size > config.max_field_values
            or not np.isfinite(field).all() or times.shape != (len(field),)
            or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0)
            or np.max(np.diff(times)) > config.max_interval_s or not np.isfinite(cell_m) or cell_m <= 0):
        raise ValueError("bounded finite complete field and strictly increasing actual query times required")
    if max(abs(float(field.max())), abs(float(field.min()))) * config.emission_log_scale > 300:
        raise ValueError("appearance dynamic range exceeds fixed numeric budget")
    shape = field.shape[1:]
    forward = np.empty(field.shape, dtype=np.float64)
    shifts, normalizers, transitions = [], [], []
    prior = np.full(shape, 1 / math.prod(shape), dtype=np.float64)
    max_mass_error = 0.
    for q in range(len(field)):
        row = np.asarray(field[q], dtype=np.float64)
        shift = float(row.max())
        emission = np.exp(config.emission_log_scale * (row - shift))
        if q:
            transition = GaussianTransition(shape, cell_m, float(times[q]-times[q-1]), config)
            prior = transition.forward(forward[q-1])
            max_mass_error = max(max_mass_error, abs(float(prior.sum()) - 1.))
            transitions.append(transition.record())
        forward[q], normalizer = _normalize(prior * emission)
        shifts.append(config.emission_log_scale * shift)
        normalizers.append(normalizer)
    posterior = np.empty(field.shape, dtype=np.float32)
    backward = np.ones(shape, dtype=np.float64)
    for q in range(len(field)-1, -1, -1):
        posterior[q] = _normalize(forward[q] * backward)[0]
        if q:
            row = np.asarray(field[q], dtype=np.float64)
            emission = np.exp(config.emission_log_scale * row - shifts[q])
            transition = GaussianTransition(shape, cell_m, float(times[q]-times[q-1]), config)
            backward = transition.backward(emission * backward)
            maximum = float(backward.max())
            if not np.isfinite(maximum) or maximum <= 0:
                raise ValueError("invalid backward message")
            backward /= maximum  # arbitrary common scale, preserving the posterior
    flat = posterior.reshape(len(field), -1)
    peaks = flat.argmax(axis=1)
    route_rc = np.column_stack(np.divmod(peaks, shape[1]))
    entropy = np.empty(len(field), dtype=np.float64)
    for q, values in enumerate(flat):
        positive = values[values > 0].astype(np.float64)
        entropy[q] = -np.sum(positive * np.log(positive))
    return {"posterior": posterior, "posterior_sha256": array_sha(posterior),
        "route_rc": route_rc, "peak_mass": flat[np.arange(len(field)), peaks].astype(float),
        "entropy_nats": entropy, "normalized_entropy": entropy / math.log(math.prod(shape)),
        "modes": [_modes(row, cell_m, config) for row in posterior],
        "transitions": transitions, "maximum_transition_mass_error": max_mass_error,
        "log_evidence": float(np.log(normalizers).sum() + np.sum(shifts)),
        "confidence": None, "confidence_calibrated": False, "unlocalized_probability": None,
        "primary_route": "posterior_peak_row_major", "all_query_rows_retained": True}


def _inputs(raw_cosines, ref_ids, ref_latlon, query_times, config):
    scores, ids, refs = np.asarray(raw_cosines), np.asarray(ref_ids), np.asarray(ref_latlon)
    times = np.asarray(query_times)
    if (scores.ndim != 2 or scores.dtype.kind != "f" or scores.dtype.itemsize not in (4, 8)
            or not 2 <= scores.shape[0] <= config.max_queries or not 1 <= scores.shape[1] <= config.max_references
            or scores.size > config.max_matrix_values or not np.isfinite(scores).all() or np.any(np.abs(scores) > 1.00001)
            or ids.shape != (scores.shape[1],) or ids.dtype.kind not in "US" or ids.dtype.itemsize > 512
            or refs.shape != (scores.shape[1], 2) or refs.dtype.kind not in "fiu" or not np.isfinite(refs).all()
            or np.any(np.abs(refs[:, 0]) > 90) or np.any(np.abs(refs[:, 1]) > 180)
            or times.shape != (scores.shape[0],) or times.dtype.kind not in "fiu" or not np.isfinite(times).all()
            or times[0] < 0 or np.any(np.diff(times.astype(float)) <= 0)
            or np.max(np.diff(times.astype(float))) > config.max_interval_s):
        raise ValueError("complete bounded raw cosines, unique IDs, public geotags and actual query times required")
    ids = ids.astype(str)
    if len(set(ids.tolist())) != len(ids) or any(not value for value in ids):
        raise ValueError("reference IDs must be nonempty and unique")
    order = np.argsort(ids, kind="stable")
    references = refs[order].astype(np.float64)
    angles = np.radians(references[:, 1])
    sine, cosine = float(np.sin(angles).mean()), float(np.cos(angles).mean())
    if np.hypot(sine, cosine) <= 1e-8:
        raise ValueError("reference longitudes have no stable metric origin")
    frame = LocalFrame(float(references[:, 0].mean()), float(np.degrees(np.arctan2(sine, cosine))))
    xy = np.column_stack(frame.to_xy(references[:, 0], references[:, 1]))
    if not np.isfinite(xy).all():
        raise ValueError("public references cannot be projected into a finite metric frame")
    cell = config.field_cell_m
    while True:
        low = np.floor((xy.min(axis=0) - config.padding_m) / cell) * cell
        high = np.ceil((xy.max(axis=0) + config.padding_m) / cell) * cell
        dimensions = (high - low) / cell
        if not np.isfinite(dimensions).all():
            raise ValueError("reference extent overflow")
        width, height = (int(round(value)) + 1 for value in dimensions)
        if len(times) * width * height <= config.max_field_values:
            break
        cell *= 2
        if cell > config.max_cell_m:
            raise ValueError("all-reference extent exceeds fixed field budget")
    return np.ascontiguousarray(scores[:, order]), times.astype(np.float64), frame, xy, low, (height, width), cell, ids[order]


def decode_spatial_tracks(raw_cosines, ref_ids, ref_latlon, query_times, *, config=None, provenance=None):
    """Use all supplied queries/references; return both registered variants.

    Posterior arrays are returned for the caller to freeze as typed artifacts.
    No winner is chosen between raw and reference-mean variants.
    """
    config = config or SpatialSequenceConfig()
    scores, times, frame, xy, low, shape, cell, ordered_ids = _inputs(raw_cosines, ref_ids, ref_latlon, query_times, config)
    profile = config.record()
    source = {"raw_cosines_sha256": array_sha(raw_cosines), "reference_ids_sha256": array_sha(ref_ids),
        "reference_latlon_sha256": array_sha(ref_latlon), "query_times_sha256": array_sha(query_times),
        "canonical_reference_ids_sha256": array_sha(ordered_ids), "caller": dict(provenance or {})}
    result = {"status": "complete", "profile": profile, "profile_sha256": json_sha(profile), "source": source,
        "query_count": len(times), "reference_count": len(xy), "selected_variant": None,
        "query_times": times.tolist(), "map_crs": frame.crs.to_string(), "origin_xy": low.tolist(),
        "shape": list(shape), "cell_m": cell, "variants": {}}
    field_config = replace(POIAlignmentConfig(), field_cell_m=cell, appearance_sigma_m=config.appearance_sigma_m,
        row_std_floor=config.row_std_floor, z_clip=config.z_clip, empty_appearance=config.empty_appearance)
    for variant in VARIANTS:
        field = build_appearance_field(xy, scores, times, origin_xy=low, shape=shape,
            reference_mean=variant == "reference_mean", config=field_config,
            provenance={"spatial_profile_sha256": result["profile_sha256"], **source})
        output = infer_spatial_posterior(field.values, times, cell, config=config)
        route = low + output["route_rc"][:, ::-1] * cell
        latlon = np.column_stack(frame.to_latlon(route[:, 0], route[:, 1]))
        output.update(variant=variant, route_xy=route, route_latlon=latlon,
            route_timestamps=times, appearance=field.source_record())
        result["variants"][variant] = output
    return result
