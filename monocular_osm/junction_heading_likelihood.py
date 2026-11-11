"""Predictive-kernel normalization for a FUTURE junction-heading target.

Pure geometry/logdensity API: no IO, filter-state updates, or accuracy/policy
admission. Normalizing a nearest-arc kernel is not exact convolution of a latent
arc heading with Gaussian bias/noise. Reusing the
robust scalar Kalman bias update would remain an explicitly approximate update.
Width determines the squared kernel, not signed innovation tie ordering.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy.special import erf

from .graph_filter import wrap_angle


PROFILE = {
    "method": "circle-normalized-nearest-junction-arc-predictive-kernel-v1",
    "scope": "pure geometry/logdensity API; no IO/filter-state updates or accuracy/policy admission",
    "max_particles": 50_000, "max_edges": 100_000,
    "max_segments": 500_000, "max_successors": 32,
    "geometry": "all active shorter arcs share current road tangent; include tangent point",
    "before_gate": "edge_length-offset < zone",
    "after_gate": "offset < zone and finite previous heading",
    "before_extent": "wrap(successor_heading-current_heading)",
    "after_extent": "-wrap(current_heading-previous_heading); oriented antipodal tie retained",
    "normalizer": "W+sqrt(2*pi*S)*erf((2*pi-W)/(2*sqrt(2*S)))",
    "outlier": "normalized uniform on circle",
    "signed_innovation_ties_preserved_by_width": False,
    "exact_marginalized_arc_convolution": False,
    "robust_KF_bias_update_provided": False,
    "accuracy_admitted": False,
}
PROFILE_SHA256 = hashlib.sha256(json.dumps(PROFILE, sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
_TWO_PI = 2 * math.pi


def _broadcast_real(*values):
    arrays = [np.asarray(value) for value in values]
    if any(a.ndim > 1 or a.dtype.kind not in "fiu" for a in arrays):
        raise ValueError("real scalar or one-dimensional arrays required")
    arrays = np.broadcast_arrays(*arrays)
    if arrays[0].size > PROFILE["max_particles"]:
        raise ValueError("particle array exceeds fixed primitive bound")
    return tuple(np.asarray(a, dtype=np.float64) for a in arrays)


def arc_union_width(net, edge, offset, prev_head, zone):
    """F64 union width with broadcast scalar/1D particle shape (maximum50k).

    Genuine ``net.locate`` supplies each current tangent. All legal successor
    arcs before a node and the previous arc after it contain that tangent.
    Their union is [min(0,extents),max(0,extents)] in its relative frame. The
    incoming extent reverses the ORIENTED shorter arc: at an exact antipode
    it is +pi whereas an outgoing antipodal extent is -pi. Together these
    cover the full circle. This does not return a signed innovation/bias update.
    Scratch is O(particles), with at most32 successor passes, no graph copy.
    """
    z = np.asarray(zone)
    if z.shape or z.dtype.kind not in "fiu" or not np.isfinite(z) or z < 0:
        raise ValueError("finite nonnegative scalar junction zone required")
    e = np.asarray(edge)
    if e.ndim > 1 or e.dtype.kind not in "iu":
        raise ValueError("integer scalar or one-dimensional edge indices required")
    off, prev, edges = _broadcast_real(offset, prev_head, e)
    shape = edges.shape
    # Check original integer values before int64 conversion; do not let an
    # unsigned overflow or float rounding silently select a different edge.
    count = len(net.edge_len)
    table, successor_heading = np.asarray(net.succ_table), np.asarray(net.succ_head)
    if (not 1 <= count <= PROFILE["max_edges"]
            or len(net.seg_heading) > PROFILE["max_segments"]
            or table.ndim != 2 or table.shape[0] != count
            or not 1 <= table.shape[1] <= PROFILE["max_successors"]
            or table.dtype.kind not in "iu" or successor_heading.shape != table.shape
            or np.any(e < 0) or np.any(e >= count)):
        raise ValueError("network or edge indices exceed primitive contract")
    ei = np.broadcast_to(e, shape).reshape(-1).astype(np.int64)
    off, prev = off.reshape(-1), prev.reshape(-1)
    lengths = np.asarray(net.edge_len)[ei]
    if (not np.isfinite(off).all() or np.isinf(prev).any()
            or not np.isfinite(lengths).all() or np.any(lengths <= 0)
            or np.any(off < 0) or np.any(off > lengths)):
        raise ValueError("valid road offsets, lengths and previous headings required")
    current = np.asarray(net.locate(ei, off)[2], dtype=np.float64)
    if current.shape != off.shape or not np.isfinite(current).all():
        raise ValueError("finite current road tangent required")
    lower, upper = np.zeros(len(ei)), np.zeros(len(ei))
    after = (off < float(z)) & np.isfinite(prev)
    extent = -wrap_angle(current[after] - prev[after])
    lower[after], upper[after] = np.minimum(0., extent), np.maximum(0., extent)
    before = np.flatnonzero(lengths - off < float(z))
    for column in range(table.shape[1]):
        candidates = table[ei[before], column]
        if np.any(candidates < -1) or np.any(candidates >= count):
            raise ValueError("invalid successor index")
        active = before[candidates >= 0]
        headings = successor_heading[ei[active], column]
        if not np.isfinite(headings).all():
            raise ValueError("finite legal successor tangent required")
        extent = wrap_angle(headings - current[active])
        lower[active] = np.minimum(lower[active], extent)
        upper[active] = np.maximum(upper[active], extent)
    return (upper - lower).reshape(shape)


def normalized_loglike(residual, variance, width, outlier):
    """Circle-normalized nearest-union kernel plus uniform outlier, in logspace.

    Inputs broadcast as real scalars/1D arrays (maximum50k). Residual must be
    the genuine nearest-union signed innovation in [-pi,pi]; only its square
    enters here. Variance is S=R+P>0; width lies in [0,2*pi]. Width alone cannot
    establish union membership or signed tie convention. No bias moments or
    Kalman gain are returned, and this is not an exact latent-arc convolution.
    """
    r, s, w = _broadcast_real(residual, variance, width)
    out = np.asarray(outlier)
    if (out.shape or out.dtype.kind not in "fiu" or not np.isfinite(out)
            or not 0 <= out <= 1 or not np.isfinite(r).all()
            or np.any(np.abs(r) > math.pi) or not np.isfinite(s).all()
            or np.any(s <= 0) or not np.isfinite(w).all()
            or np.any(w < 0) or np.any(w > _TWO_PI)):
        raise ValueError("valid circular residual, positive variance, width and outlier required")
    root_s = np.sqrt(s)
    # sqrt(S) products remain finite for EVERY positive finite F64 S. Avoid
    # sqrt(2*pi*S), which can overflow before its small erf factor is applied.
    z = w + (math.sqrt(_TWO_PI) * root_s) * erf(
        (_TWO_PI - w) / (2 * math.sqrt(2.) * root_s))
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        log_inlier = -.5 * np.square(r / root_s) - np.log(z)
    epsilon = float(out)
    if epsilon == 0:
        return log_inlier
    if epsilon == 1:
        return np.full(r.shape, -math.log(_TWO_PI), dtype=np.float64)
    return np.logaddexp(math.log1p(-epsilon) + log_inlier,
                        math.log(epsilon) - math.log(_TWO_PI))
