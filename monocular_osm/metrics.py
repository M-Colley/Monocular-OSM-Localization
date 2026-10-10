"""Route-level accuracy metrics for a predicted lat/lon route vs GT waypoints.

The project's historical headline number, "mean route error", is the mean
over ground-truth waypoints of the distance to the nearest point of the
predicted route (here ``gt_to_pred_mean_m``). It is one-directional: a
prediction that wanders, overshoots or is far too long is never penalised
as long as SOME part of it passes near every GT waypoint. This module keeps
that number for continuity and adds the complementary direction plus the
standard curve-to-curve distances, so a benchmark row can no longer be
gamed by a long scribble.

All geometry runs in ONE local metric frame per clip: an azimuthal
equidistant projection (pyproj ``aeqd``, WGS84) centred on the GT
centroid. Distances from the centre are exact and the scale error at 10 km
from it is below 1e-6, so there are no flat-earth constants anywhere.
Point/segment geometry is done with shapely (exact point-to-segment
distances, not vertex sampling).

Definitions (``pred`` = predicted route, ``gt`` = GT waypoint polyline in
time order)::

    start_error_m       |pred[0]  - gt[0]|   (GT waypoint at the analysis
                                              window's first instant)
    end_error_m         |pred[-1] - gt[-1]|
    gt_to_pred_mean_m   mean_i dist(gt_i, pred)          -- the legacy
                        "mean route error" (per GT waypoint, time-sampled)
    gt_to_pred_max_m    max_i  dist(gt_i, pred)
    pred_to_gt_mean_m   mean over pred resampled every ``densify_m`` metres
                        of ARC LENGTH of dist(p, gt polyline)
    symmetric_mean_m    (gt_to_pred_mean_m + pred_to_gt_mean_m) / 2
    hausdorff_m         Hausdorff distance between the two curves
    frechet_m           discrete Fréchet distance (order-aware: a route
                        driven the wrong way round scores badly)

Hausdorff and Fréchet are computed on both curves segmentized to at most
``densify_m`` (default 10 m) between vertices, so their discretisation
error is bounded by ~``densify_m``.

**Single-point predictions** (baselines such as the city centroid or the
VPR retrieval centre) are scored as a *degenerate route*: a route of
length zero sitting at that point. Then ``start_error_m`` and
``end_error_m`` are the point's distances to the first/last GT waypoint,
``gt_to_pred_mean_m`` is the mean waypoint distance to the point,
``pred_to_gt_mean_m`` is the point's distance to the GT polyline, and
Hausdorff = Fréchet = the largest distance from the point to the GT curve.
This puts point baselines and route predictions on one scale; the
``degenerate`` flag records which kind was scored.

**Ground-truth sparsity caveat.** GT files carry 8-27 waypoints per clip,
joined here by straight chords. Where the true road curves between two
waypoints the chord cuts the corner, so every curve-based number has a
noise floor of the chord-to-road deviation (typically metres, up to tens
of metres on long gaps). Compare methods on the same GT, not against zero.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np
import shapely
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point

#: Success thresholds (metres) reported for start and symmetric-mean error.
THRESHOLDS_M: tuple[int, ...] = (25, 50, 100, 250, 500, 1000)

#: Resampling step (metres) for pred->GT and the curve distances.
DENSIFY_M: float = 10.0

#: Metrics that get success flags, in report order.
SUCCESS_METRICS: tuple[str, ...] = ("start_error_m", "symmetric_mean_m")


def local_transformer(lat0: float, lon0: float) -> Transformer:
    """WGS84 -> azimuthal-equidistant metres centred on (lat0, lon0)."""
    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.9f} +lon_0={lon0:.9f} "
        "+datum=WGS84 +units=m +no_defs")
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def _as_latlon(arr: Sequence | np.ndarray, what: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1 and a.shape == (2,):
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != 2 or len(a) == 0:
        raise ValueError(f"{what}: expected a non-empty Nx2 [lat, lon] array, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{what}: non-finite coordinates")
    if (np.abs(a[:, 0]) > 90).any() or (np.abs(a[:, 1]) > 180).any():
        raise ValueError(f"{what}: coordinates outside WGS84 range")
    return a


def project(latlon: np.ndarray, tf: Transformer) -> np.ndarray:
    """Nx2 [lat, lon] -> Nx2 [x, y] metres in the frame of ``tf``."""
    x, y = tf.transform(latlon[:, 1], latlon[:, 0])
    return np.column_stack([np.asarray(x, dtype=np.float64),
                            np.asarray(y, dtype=np.float64)])


def _geometry(xy: np.ndarray):
    """LineString for >=2 distinct vertices, else a Point (degenerate route)."""
    if len(xy) >= 2:
        # Drop consecutive duplicates (stationary samples) — they carry no
        # geometry and a 2-vertex zero-length LineString is invalid.
        keep = np.ones(len(xy), dtype=bool)
        keep[1:] = np.any(np.abs(np.diff(xy, axis=0)) > 1e-9, axis=1)
        xy = xy[keep]
    if len(xy) >= 2:
        return LineString(xy)
    return Point(xy[0])


def _resample_by_arc_length(geom, step_m: float) -> np.ndarray:
    """Points every ``step_m`` metres along ``geom`` (endpoints included)."""
    if isinstance(geom, Point):
        return np.array([[geom.x, geom.y]])
    length = geom.length
    n = max(1, int(math.ceil(length / step_m)))
    d = np.linspace(0.0, length, n + 1)
    pts = shapely.line_interpolate_point(geom, d)
    return shapely.get_coordinates(pts)


def _segmentized(geom, step_m: float):
    if isinstance(geom, Point):
        return geom
    return shapely.segmentize(geom, step_m)


@dataclass
class RouteScore:
    """Every metric for one prediction against one GT track (metres)."""

    start_error_m: float
    end_error_m: float
    gt_to_pred_mean_m: float
    gt_to_pred_max_m: float
    pred_to_gt_mean_m: float
    symmetric_mean_m: float
    hausdorff_m: float
    frechet_m: float
    pred_length_m: float
    gt_length_m: float
    n_pred_points: int
    n_gt_points: int
    degenerate: bool
    success: dict[str, dict[str, bool]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        for k, v in list(out.items()):
            if isinstance(v, float):
                out[k] = round(v, 2)
        return out


def success_flags(values: dict[str, float],
                  thresholds: Iterable[float] = THRESHOLDS_M) -> dict[str, dict[str, bool]]:
    """``{metric: {str(threshold): value <= threshold}}`` (NaN/inf -> False)."""
    out: dict[str, dict[str, bool]] = {}
    for metric in SUCCESS_METRICS:
        v = values.get(metric)
        ok = v is not None and np.isfinite(v)
        out[metric] = {str(int(t)): bool(ok and v <= t) for t in thresholds}
    return out


def score_route(
    pred_latlon: Sequence | np.ndarray,
    gt_latlon: Sequence | np.ndarray,
    *,
    densify_m: float = DENSIFY_M,
    thresholds: Iterable[float] = THRESHOLDS_M,
) -> RouteScore:
    """Score a predicted route (or single point) against GT waypoints.

    ``pred_latlon``: Nx2 [lat, lon] in route order (N >= 1; N == 1 or all
    points identical is a degenerate route — see the module docstring).
    ``gt_latlon``: Mx2 [lat, lon] GT waypoints in time order (M >= 1).
    """
    pred = _as_latlon(pred_latlon, "prediction")
    gt = _as_latlon(gt_latlon, "ground truth")

    lat0, lon0 = gt.mean(axis=0)
    tf = local_transformer(float(lat0), float(lon0))
    pred_xy = project(pred, tf)
    gt_xy = project(gt, tf)

    pred_geom = _geometry(pred_xy)
    gt_geom = _geometry(gt_xy)
    degenerate = isinstance(pred_geom, Point)

    start = float(np.hypot(*(pred_xy[0] - gt_xy[0])))
    end = float(np.hypot(*(pred_xy[-1] - gt_xy[-1])))

    gt_pts = shapely.points(gt_xy)
    g2p = np.asarray(shapely.distance(gt_pts, pred_geom), dtype=np.float64)

    samples = _resample_by_arc_length(pred_geom, densify_m)
    p2g = np.asarray(shapely.distance(shapely.points(samples), gt_geom), dtype=np.float64)

    pred_d = _segmentized(pred_geom, densify_m)
    gt_d = _segmentized(gt_geom, densify_m)
    hausdorff = float(shapely.hausdorff_distance(pred_d, gt_d))
    frechet = float(shapely.frechet_distance(pred_d, gt_d))

    g2p_mean = float(g2p.mean())
    p2g_mean = float(p2g.mean())
    values = {
        "start_error_m": start,
        "end_error_m": end,
        "gt_to_pred_mean_m": g2p_mean,
        "gt_to_pred_max_m": float(g2p.max()),
        "pred_to_gt_mean_m": p2g_mean,
        "symmetric_mean_m": 0.5 * (g2p_mean + p2g_mean),
        "hausdorff_m": hausdorff,
        "frechet_m": frechet,
    }
    return RouteScore(
        **values,
        pred_length_m=0.0 if degenerate else float(pred_geom.length),
        gt_length_m=0.0 if isinstance(gt_geom, Point) else float(gt_geom.length),
        n_pred_points=int(len(pred)),
        n_gt_points=int(len(gt)),
        degenerate=degenerate,
        success=success_flags(values, thresholds),
    )


def failed_score(n_gt_points: int, thresholds: Iterable[float] = THRESHOLDS_M) -> dict:
    """Score dict for a run that produced no prediction: every error is
    +inf and every success flag False, so failures count against a method
    instead of silently dropping out of its averages."""
    inf = float("inf")
    values = {k: inf for k in (
        "start_error_m", "end_error_m", "gt_to_pred_mean_m", "gt_to_pred_max_m",
        "pred_to_gt_mean_m", "symmetric_mean_m", "hausdorff_m", "frechet_m")}
    return {
        **values,
        "pred_length_m": None, "gt_length_m": None,
        "n_pred_points": 0, "n_gt_points": int(n_gt_points),
        "degenerate": None,
        "success": success_flags(values, thresholds),
        "failed": True,
    }
