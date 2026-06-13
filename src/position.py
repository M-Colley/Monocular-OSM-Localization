"""Turn the winning route match into a WGS84 position answer.

The matcher works entirely in a projected metric CRS (UTM for the
city), which is right for geometry but useless as a final answer: the
user asked "where is this video?" and wants coordinates they can paste
into a map. This module converts the consensus candidate's *aligned
camera path* (``MatchCandidate.aligned_traj_xy`` — the VO trajectory
after Procrustes alignment onto the matched OSM walk) into:

* a headline ``(latitude, longitude)`` — where the video segment starts,
* start / end / center points of the driven route,
* a subsampled route polyline in lat/lon,
* the street names traversed,
* shareable Google Maps / OpenStreetMap links,
* a coarse confidence classification with the raw signals attached.

Everything here is a pure function over (candidate, road graph,
result-matches), so it is unit-testable without network access or a
real pipeline run.
"""

from __future__ import annotations

import numpy as np

from .osm_data import RoadGraph
from .trajectory_matching import MatchCandidate, candidate_geographic_summary


# Confidence thresholds. Heuristic, derived from GT-evaluated runs on
# the Ulm clip: the correct candidate consistently had bearing_corr
# >= ~0.3 and RMS in the low hundreds; wildly-wrong candidates pair a
# low correlation with a large residual. "high" requires BOTH a strong
# turn-pattern agreement and a tight geometric fit; "low" fires when
# EITHER signal is bad. The two classes are mutually exclusive by
# construction (high needs corr >= 0.5 and rms <= 150; low needs
# corr < 0.25 or rms > 400).
_HIGH_MIN_CORR = 0.5
_HIGH_MAX_RMS_M = 150.0
_LOW_MAX_CORR = 0.25
_LOW_MIN_RMS_M = 400.0


def _usable_crs(crs: str | None) -> str | None:
    """Return a non-empty CRS string, or None.

    ``RoadGraph.crs`` is built via ``str(graph.graph.get("crs", ""))``,
    so a missing CRS shows up as ``""`` or the literal string ``"None"``.
    """
    if crs is None:
        return None
    crs = crs.strip()
    if not crs or crs == "None":
        return None
    return crs


def xy_to_latlon(xy: np.ndarray, crs: str) -> np.ndarray:
    """Convert an Nx2 array of projected (x, y) to Nx2 (lat, lon) WGS84.

    Raises whatever pyproj raises for an invalid CRS — callers that
    need graceful degradation should use :func:`build_position_report`.
    """
    from pyproj import Transformer

    xy = np.asarray(xy, dtype=np.float64)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    # Pass lists, not arrays: pyproj routes 1-element arrays through its
    # scalar path, which trips NumPy's deprecated array->scalar
    # conversion (an error in future NumPy versions).
    lon, lat = transformer.transform(xy[:, 0].tolist(), xy[:, 1].tolist())
    return np.column_stack([lat, lon])


def latlon_to_xy(latlon: np.ndarray, crs: str) -> np.ndarray:
    """Inverse of :func:`xy_to_latlon`: Nx2 (lat, lon) -> Nx2 projected (x, y)."""
    from pyproj import Transformer

    latlon = np.asarray(latlon, dtype=np.float64)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(latlon[:, 1].tolist(), latlon[:, 0].tolist())
    return np.column_stack([x, y])


def candidate_center_latlon(
    cand: MatchCandidate, road: RoadGraph
) -> tuple[float, float] | None:
    """(lat, lon) of the candidate walk's centroid, or None when the
    graph has no usable CRS."""
    crs = _usable_crs(road.crs)
    if crs is None:
        return None
    try:
        center = np.asarray(cand.walk_xy, dtype=np.float64).mean(axis=0, keepdims=True)
        latlon = xy_to_latlon(center, crs)
    except Exception:
        return None
    lat, lon = float(latlon[0, 0]), float(latlon[0, 1])
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return None
    return lat, lon


def _subsample_route(route: np.ndarray, max_points: int) -> np.ndarray:
    """Keep at most `max_points` points, always including the endpoints."""
    if len(route) <= max_points:
        return route
    idx = np.unique(np.linspace(0, len(route) - 1, max_points).round().astype(int))
    return route[idx]


def _point_dict(lat: float, lon: float) -> dict:
    return {"latitude": round(float(lat), 6), "longitude": round(float(lon), 6)}


def google_maps_url(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"


def openstreetmap_url(lat: float, lon: float) -> str:
    return (
        f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}"
        f"#map=16/{lat:.6f}/{lon:.6f}"
    )


def classify_confidence(
    cand: MatchCandidate, matches: list[dict] | None = None
) -> dict:
    """Coarse high/medium/low confidence plus the raw signals behind it.

    The classification only uses the winner's own shape signals (RMS
    residual + bearing correlation). When the ranked ``matches`` list is
    available we also attach the consensus margin to runner-up and the
    sliding-window support ratio — informative for a human reading the
    JSON, but deliberately not folded into the class boundaries (their
    scales vary too much across runs to threshold robustly).
    """
    rms = float(cand.score)
    corr = float(cand.bearing_corr)

    if corr >= _HIGH_MIN_CORR and rms <= _HIGH_MAX_RMS_M:
        level = "high"
    elif corr < _LOW_MAX_CORR or rms > _LOW_MIN_RMS_M:
        level = "low"
    else:
        level = "medium"

    out: dict = {
        "level": level,
        "score_rms_m": round(rms, 1),
        "bearing_corr": round(corr, 3),
    }

    if matches and len(matches) >= 2:
        first, second = matches[0], matches[1]
        if "consensus_score" in first and "consensus_score" in second:
            # Lower fused score = better; positive margin = clear winner.
            out["consensus_margin"] = round(
                float(second["consensus_score"]) - float(first["consensus_score"]), 2
            )
        if "sliding_window_support_ratio" in first:
            out["sliding_window_support_ratio"] = round(
                float(first["sliding_window_support_ratio"]), 3
            )
    return out


def build_position_report(
    cand: MatchCandidate,
    road: RoadGraph,
    *,
    matches: list[dict] | None = None,
    ranking: str = "shape",
    max_route_points: int = 50,
) -> dict | None:
    """Build the JSON-safe ``position`` block for the winning candidate.

    Returns ``None`` when the road graph has no usable CRS or the
    aligned trajectory cannot be converted to finite WGS84 coordinates
    — the caller should record an error rather than crash, since by
    this point the pipeline has done all the expensive work and the
    rest of the result is still valuable.

    The headline ``latitude``/``longitude`` is the **start of the
    aligned camera path**: where the camera was at the first frame of
    the analyzed segment. That is the most literal answer to "where is
    this video?". ``start``/``end``/``center`` and ``route_latlon``
    describe the whole driven route.
    """
    crs = _usable_crs(road.crs)
    if crs is None:
        return None

    traj = np.asarray(cand.aligned_traj_xy, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[0] < 1 or traj.shape[1] != 2:
        return None
    if not np.isfinite(traj).all():
        return None

    try:
        route = xy_to_latlon(traj, crs)
    except Exception:
        return None
    if not np.isfinite(route).all():
        return None
    # Reject results outside valid WGS84 ranges (symptom of a bogus CRS
    # that pyproj nonetheless accepted).
    if (np.abs(route[:, 0]) > 90.0).any() or (np.abs(route[:, 1]) > 180.0).any():
        return None

    start = route[0]
    end = route[-1]
    center = route.mean(axis=0)
    route_sub = _subsample_route(route, max_route_points)

    summary = candidate_geographic_summary(cand, road.graph)

    lat, lon = float(start[0]), float(start[1])
    return {
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "start": _point_dict(*start),
        "end": _point_dict(*end),
        "center": _point_dict(*center),
        "route_latlon": [
            [round(float(la), 6), round(float(lo), 6)] for la, lo in route_sub
        ],
        "street_names": summary["street_names"],
        "google_maps_url": google_maps_url(lat, lon),
        "openstreetmap_url": openstreetmap_url(lat, lon),
        "ranking": ranking,
        "n_candidates": len(matches) if matches else 1,
        "confidence": classify_confidence(cand, matches),
    }


def format_position_summary(position: dict) -> str:
    """Human-readable block printed at the end of a pipeline run."""
    conf = position.get("confidence", {})
    streets = ", ".join(position.get("street_names", [])[:5]) or "(unnamed roads)"
    center = position.get("center", {})
    lines = [
        "=" * 64,
        f"ESTIMATED POSITION  (ranking={position.get('ranking', '?')}, "
        f"{position.get('n_candidates', '?')} candidates considered)",
        "=" * 64,
        f"  Video starts at:  {position['latitude']:.6f}, {position['longitude']:.6f}",
        f"  Route center:     {center.get('latitude', float('nan')):.6f}, "
        f"{center.get('longitude', float('nan')):.6f}",
        f"  Streets:          {streets}",
        f"  Confidence:       {conf.get('level', 'unknown')} "
        f"(RMS {conf.get('score_rms_m', '?')} m, "
        f"bearing corr {conf.get('bearing_corr', '?')})",
        f"  Google Maps:      {position['google_maps_url']}",
        f"  OpenStreetMap:    {position['openstreetmap_url']}",
        "=" * 64,
    ]
    return "\n".join(lines)
