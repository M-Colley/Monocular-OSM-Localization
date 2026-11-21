"""Proposal-only backward appearance messages over the complete snapshot grid.

This is a spatial diffusion heuristic, not an additional observation likelihood
or a road-transition model. The caller retains every original appearance/OCR
factor and the genuine initial, turn and auxiliary importance corrections.
There is no IO, random sampling, model execution or evidence admission here.
"""
from types import MappingProxyType
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from . import reference_snapshot_appearance as snapshot

strict = snapshot.strict

PROFILE = MappingProxyType({
    "method": "snapshot-backward-future-proposal-v1",
    "study_contract_sha256": "da673b3f0316417c53cccb9e8f8868971c236910e3b8ad759ad41c5c58dd87fd",
    "snapshot_appearance_profile_sha256": snapshot.PROFILE_SHA256,
    "query_rows": 80, "cell_m": 100., "diffusion_m_per_s": 25.,
    "gaussian_mode": "nearest", "gaussian_truncate": 3.,
    "maximum_gaussian_sigma_cells": 8., "maximum_gaussian_radius": 24,
    "maximum_axis_value_visits_per_pass": 19_600_000,
    "oversize_diffusion": "whole-control refusal before Gaussian allocation; never clip or crop",
    "diffusion_limitations": "query-spacing-dependent heuristic; sigma proportional to time is not a semigroup",
    "G": "(.1+.9*exp(A_q-max_complete_grid(A_q)))**original_group_power",
    "terminal_H": "H_79=1",
    "backward": "H_q=maxnormalize(Gaussian(25*delta_t/100,G_(q+1)*H_(q+1)))",
    "active": "maxnormalize(Gaussian(25*remaining_time/100,G_next*H_next)); floor1e-6",
    "future": "first eligible scheduled_step strictly greater than current PF index",
    "eligible_start_query": 1,
    "no_future": "ones, including outside the raster",
    "outside_active": 1e-6, "active_floor": 1e-6,
    "max_field_values": 16_000_000, "backward_dtype": "float32",
    "max_backward_bytes": 64_000_000, "max_composed_endpoints": 1201,
    "max_sample_positions": 200_000, "sample_chunk": 4096,
    "cache": "one active float32 layer for current monotone PF index",
    "auxiliary_defense": ".25+.75*rawg/(normalized_w@rawg); only auxiliary caller",
    "target_changed": False, "historical_encoder_verified": False,
    "calibrated_confidence": False, "policy_admission": False,
})
PROFILE_SHA256 = strict.poi._json_digest(dict(PROFILE))


def _normalise(values):
    maximum = float(values.max())
    strict.need(math.isfinite(maximum) and maximum > 0, "positive finite full guide maximum required")
    values /= maximum


def _factor(appearance, q, scratch):
    """One F64 grid scratch; the complete original observation stays untouched."""
    np.copyto(scratch, appearance.field.values[q])
    scratch -= appearance.maxima[q]
    np.exp(scratch, out=scratch)
    scratch *= .9
    scratch += .1
    np.power(scratch, appearance.powers[q], out=scratch)


def _next_query(steps, index):
    query = int(np.searchsorted(steps, index, side="right"))
    return max(1, query)  # legacy initializer already uses G0; target G0 remains unchanged


def _widths(clock, times, steps):
    """Check every possible active width BEFORE any backward/Gaussian work."""
    backward = .25 * np.diff(times)
    active = np.zeros(len(clock), np.float64)
    for index in range(len(clock)):
        q = _next_query(steps, index)
        if q < 80:
            active[index] = .25 * float(times[q] - clock[index])
            strict.need(active[index] > 0, "next scheduled native query must be strictly future")
    strict.need(np.isfinite(backward).all() and np.all(backward > 0)
        and np.all(backward <= 8.) and np.isfinite(active).all() and np.all(active <= 8.),
        "Gaussian sigma exceeds registered 8-cell whole-control bound; never clip/crop")
    return backward, active


def defensive_auxiliary_guide(weights, raw_guide):
    """Return effective g for q=.25*w+.75*w*rawg/(w@rawg).

    The existing auxiliary ``1/g`` correction must use THIS returned value.
    Initial/turn proposals already contain their own defense and use raw g.
    """
    w, g = np.asarray(weights), np.asarray(raw_guide)
    strict.need(w.dtype == np.float64 and w.ndim == 1 and 1 <= len(w) <= 50_000
        and g.dtype == np.float64 and g.shape == w.shape
        and np.isfinite(w).all() and np.all(w >= 0)
        and np.isfinite(g).all() and np.all(g > 0)
        and math.isclose(float(w.sum(dtype=np.float64)), 1., rel_tol=0., abs_tol=1e-12),
        "normalized finite F64 weights and positive bounded raw guides required")
    mean = float(np.dot(w, g))
    strict.need(math.isfinite(mean) and mean > 0, "positive finite weighted guide mean required")
    effective = g / mean
    effective *= .75
    effective += .25
    strict.need(np.isfinite(effective).all(), "auxiliary effective guide overflow")
    return effective


def _validate(appearance, composed_times):
    strict.need(type(appearance) is strict.Appearance
        and type(appearance.field) is strict.poi.RasterField
        and appearance.receipt.get("profile") == dict(snapshot.PROFILE)
        and appearance.receipt.get("profile_sha256") == snapshot.PROFILE_SHA256
        and appearance.receipt.get("all_queries_once") is True
        and appearance.receipt.get("historical_encoder_verified") is False,
        "genuine complete snapshot appearance required; source admission remains with caller")
    field = appearance.field
    values = field.values
    strict.need(isinstance(values, np.ndarray) and values.dtype == np.float32
        and values.ndim == 3 and values.shape[0] == 80 and min(values.shape[1:]) >= 2
        and values.flags.c_contiguous and values.size <= 16_000_000
        and field.cell_m == 100. and len(field.origin_xy) == 2
        and np.isfinite(field.origin_xy).all(), "complete bounded C-F32 100m snapshot field required")
    times, clock = np.asarray(field.query_times), np.asarray(composed_times)
    strict.need(clock.dtype == np.float64 and clock.ndim == 1 and 2 <= len(clock) <= 1201
        and np.isfinite(clock).all() and np.all(np.diff(clock) > 0)
        and times.dtype == np.float64 and times.shape == (80,) and np.isfinite(times).all()
        and np.all(np.diff(times) > 0) and clock[0] <= times[0] and times[-1] <= clock[-1],
        "full native80 query and bounded genuine composed clocks required")
    groups, powers = strict.temporal_groups(times)
    steps = np.searchsorted(clock, times, side="left").astype(np.int64)
    strict.need(isinstance(appearance.groups, np.ndarray) and appearance.groups.dtype == np.int64
        and np.array_equal(appearance.groups, groups)
        and isinstance(appearance.powers, np.ndarray) and appearance.powers.dtype == np.float64
        and np.array_equal(appearance.powers, powers)
        and isinstance(appearance.steps, np.ndarray) and appearance.steps.dtype == np.int64
        and np.array_equal(appearance.steps, steps)
        and np.all(clock[steps] - times <= 1.)
        and isinstance(appearance.maxima, np.ndarray) and appearance.maxima.dtype == np.float64
        and appearance.maxima.shape == (80,), "original groups/powers and exact scheduled endpoints differ")
    for q in range(80):
        strict.need(np.isfinite(values[q]).all() and appearance.maxima[q] == float(values[q].max()),
                    "finite original complete field and exact maxima required")
    receipt = appearance.receipt
    strict.need(receipt.get("field") == field.source_record()
        and receipt.get("query_times") == times.tolist()
        and receipt.get("group_ids") == groups.tolist()
        and receipt.get("group_powers") == powers.tolist()
        and receipt.get("scheduled_steps") == steps.tolist()
        and receipt.get("row_maxima_over_complete_grid") == appearance.maxima.tolist(),
        "snapshot field receipt/clock/group identity differs")
    return clock.copy(), times.copy(), steps.copy()


class FutureGuide:
    """A backward volume and one cached active raster; no particle RNG state."""
    def __init__(self, appearance, clock, times, steps, backward, active_sigmas, scratch, check):
        self._appearance = appearance
        self._clock, self._times, self._steps = clock, times, steps
        self._active_sigmas = active_sigmas
        self.backward_H = backward
        self.backward_H.flags.writeable = False
        self._scratch = scratch
        self._active = np.empty(backward.shape[1:], np.float32)
        self._index, self._next_query = -1, None
        self._raster = None
        self._check = check
        self.gaussian_calls = 0
        self.receipt = {"profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256,
            "snapshot_appearance_profile_sha256": snapshot.PROFILE_SHA256,
            "input_field": appearance.field.source_record(),
            "composed_times_sha256": strict.poi._digest(clock),
            "scheduled_steps": steps.tolist(), "group_powers": appearance.powers.tolist(),
            "eligible_start_query": 1,
            "backward_shape": list(backward.shape), "backward_bytes": backward.nbytes,
            "backward_sha256": strict.poi._digest(backward),
            "maximum_active_sigma_cells": float(active_sigmas.max()),
            "active_layer_bytes": self._active.nbytes, "F64_grid_scratch_bytes": scratch.nbytes,
            "target_changed": False, "no_observations_added_or_removed": True,
            "caller_must_preserve_importance_corrections": True}

    def _at(self, index):
        strict.need(type(index) is int and 0 <= index < len(self._clock) and self._index <= index,
                    "monotone genuine PF index within complete composed clock required")
        if index == self._index:
            return self._raster
        self._check()
        query = _next_query(self._steps, index)
        if query == 80:
            self._raster = None
            self._index, self._next_query = index, None
            return None
        _factor(self._appearance, query, self._scratch)
        self._scratch *= self.backward_H[query]
        gaussian_filter(self._scratch, sigma=float(self._active_sigmas[index]), output=self._scratch,
                        mode="nearest", truncate=3.)
        self.gaussian_calls += 1
        _normalise(self._scratch)
        np.maximum(self._scratch, 1e-6, out=self._scratch)
        np.copyto(self._active, self._scratch)
        self._raster = strict.poi.RasterField(self._active, self._appearance.field.origin_xy,
                                             100., 1e-6)
        self._index, self._next_query = index, query
        return self._raster

    def sample(self, index, net, edge, offset):
        """Sample fresh road positions using genuine CRS conversion/bilinear math.

        Initial partition centres, 2-D turn alternatives and particle vectors
        share the same cached raster. Small blocks bound sampling temporaries.
        """
        e, s = np.asarray(edge), np.asarray(offset)
        strict.need(e.dtype.kind in "iu" and s.dtype.kind in "fiu"
            and math.prod(np.broadcast_shapes(e.shape, s.shape)) <= 200_000
            and str(net.crs) == str(self._appearance.receipt["crs"]),
            "bounded road candidates in the exact snapshot CRS required")
        e, s = np.broadcast_arrays(e, s)
        strict.need(e.size > 0, "nonempty road candidate array required")
        flat_e, flat_s = e.ravel(), s.ravel()
        out = np.empty(e.size, np.float64)
        raster = self._at(index)
        for lo in range(0, e.size, 4096):
            self._check()
            hi = min(e.size, lo + 4096)
            edges, offsets = flat_e[lo:hi], flat_s[lo:hi]
            strict.need(np.all(edges < net.n_edges) and np.all(edges >= 0)
                and np.isfinite(offsets).all() and np.all(offsets >= 0)
                and np.all(offsets <= net.edge_len[edges]), "finite actual road edge/offset candidates required")
            if raster is None:
                out[lo:hi] = 1.
            else:
                x, y, _, _ = net.locate(edges, offsets)
                absolute = net.local_to_crs(np.column_stack([x, y]))
                strict.need(np.isfinite(absolute).all(), "finite absolute road coordinates required")
                out[lo:hi] = raster.sample(absolute, np.empty(0, np.float64))
        return out.reshape(e.shape)


def build(appearance, composed_times, *, check=lambda: None):
    """Prepare all 80 messages without modifying or duplicating the target field.

    Explicit array payload: <=64MB backward F32 volume, one <=.8MB active
    layer, one <=1.6MB F64 grid scratch, and tiny immutable clocks. SciPy's
    one-dimensional kernel/line buffers and sampling chunks are additional;
    external process memory/time enforcement remains the caller's job.
    """
    check()
    clock, times, steps = _validate(appearance, composed_times)
    # Preflight ALL diffusion widths before allocating a backward volume.
    sigma, active_sigmas = _widths(clock, times, steps)
    backward = np.empty(appearance.field.values.shape, np.float32)
    scratch = np.empty(backward.shape[1:], np.float64)
    backward[79].fill(1.)
    for q in range(78, -1, -1):
        check()
        _factor(appearance, q + 1, scratch)
        scratch *= backward[q + 1]
        gaussian_filter(scratch, sigma=float(sigma[q]), output=scratch,
                        mode="nearest", truncate=3.)
        _normalise(scratch)
        np.copyto(backward[q], scratch)
    result = FutureGuide(appearance, clock, times, steps, backward, active_sigmas, scratch, check)
    check()
    return result
