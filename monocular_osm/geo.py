"""One home for geodesy, built on pyproj (PROJ + GeographicLib).

Before this module the package carried three hand-written haversines (a
6371 km sphere) and a dozen flat-earth ``111320 m/deg`` conversions. Those
are 0.1-0.5 % off on the WGS84 ellipsoid at the latitudes we work at
(e.g. at 48 N one degree of latitude is 111.2 km, one of longitude
74.6 km, vs 111.32 km and 74.5 km on the sphere), and every call site
re-derived the same arithmetic. Everything here is WGS84:

* :func:`geodesic_distance_m` - ellipsoidal great-circle (geodesic)
  distance via :class:`pyproj.Geod` (Karney's algorithm, nm-accurate).
* :class:`LocalFrame` - a local metric east/north frame around
  ``(lat0, lon0)``: an azimuthal-equidistant projection (``+proj=aeqd``)
  centred there, so distances and bearings *from the origin* are exact
  and distortion elsewhere is ~1e-7 at the few-km scale of a drive.
* :func:`metres_per_degree` - local WGS84 scale of one degree of latitude
  and longitude (for converting raster pixel sizes and degree steps).
* :func:`degree_half_extent` - the half-width in degrees of a metric
  radius around a point (for building lat/lon bounding boxes).

Thresholds expressed in metres keep their meaning at every call site;
only the metres<->degrees conversion becomes ellipsoidal.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyproj import CRS, Geod, Transformer

__all__ = [
    "WGS84",
    "LocalFrame",
    "degree_half_extent",
    "geodesic_distance_m",
    "metres_per_degree",
]

#: Shared WGS84 ellipsoid (thread-safe for ``inv``/``fwd``).
WGS84 = Geod(ellps="WGS84")


def _is_scalar(*xs) -> bool:
    return all(np.ndim(x) == 0 for x in xs)


def geodesic_distance_m(lat1, lon1, lat2, lon2):
    """WGS84 geodesic distance in metres between ``(lat1, lon1)`` and
    ``(lat2, lon2)`` (degrees).

    Vectorised: inputs may be scalars or array-likes that broadcast
    against each other. Returns a ``float`` when every input is a scalar,
    otherwise an ``ndarray`` of the broadcast shape.
    """
    if _is_scalar(lat1, lon1, lat2, lon2):
        _, _, d = WGS84.inv(float(lon1), float(lat1), float(lon2), float(lat2))
        return float(d)
    la1, lo1, la2, lo2 = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (lat1, lon1, lat2, lon2)))
    shape = la1.shape
    if la1.size == 0:
        return np.zeros(shape, dtype=np.float64)
    _, _, d = WGS84.inv(lo1.ravel(), la1.ravel(), lo2.ravel(), la2.ravel())
    return np.asarray(d, dtype=np.float64).reshape(shape)


@lru_cache(maxsize=256)
def _aeqd_transformers(lat0: float, lon0: float) -> tuple[Transformer, Transformer]:
    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0!r} +lon_0={lon0!r} +x_0=0 +y_0=0 "
        "+ellps=WGS84 +units=m +no_defs")
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return fwd, inv


class LocalFrame:
    """Local metric frame (x = east, y = north, metres) around a WGS84
    origin, implemented as a pyproj azimuthal-equidistant projection.

    ``to_xy`` / ``to_latlon`` accept scalars or arrays and return the same
    kind (floats for scalar input, arrays otherwise).
    """

    __slots__ = ("lat0", "lon0", "_fwd", "_inv")

    def __init__(self, lat0: float, lon0: float) -> None:
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self._fwd, self._inv = _aeqd_transformers(self.lat0, self.lon0)

    def __repr__(self) -> str:
        return f"LocalFrame(lat0={self.lat0!r}, lon0={self.lon0!r})"

    @property
    def crs(self) -> CRS:
        return self._fwd.target_crs

    def to_xy(self, lat, lon):
        """(lat, lon) degrees -> (x east, y north) metres."""
        if _is_scalar(lat, lon):
            x, y = self._fwd.transform(float(lon), float(lat))
            return float(x), float(y)
        la, lo = np.broadcast_arrays(np.asarray(lat, dtype=np.float64),
                                     np.asarray(lon, dtype=np.float64))
        x, y = self._fwd.transform(lo, la)
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def to_latlon(self, x, y):
        """(x east, y north) metres -> (lat, lon) degrees."""
        if _is_scalar(x, y):
            lon, lat = self._inv.transform(float(x), float(y))
            return float(lat), float(lon)
        xx, yy = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                     np.asarray(y, dtype=np.float64))
        lon, lat = self._inv.transform(xx, yy)
        return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)


def metres_per_degree(lat) -> tuple:
    """Local WGS84 length of one degree of latitude and of longitude at
    latitude ``lat`` (degrees): ``(m_per_deg_lat, m_per_deg_lon)``.

    Uses the meridional (M) and prime-vertical (N) radii of curvature of
    pyproj's WGS84 ellipsoid: ``M = a(1-e^2)/(1-e^2 sin^2 phi)^1.5``,
    ``N = a/sqrt(1-e^2 sin^2 phi)``; one degree is ``pi/180 * M`` north
    and ``pi/180 * N cos(phi)`` east. Scalars in, floats out; arrays in,
    arrays out.
    """
    phi = np.radians(np.asarray(lat, dtype=np.float64))
    a, es = WGS84.a, WGS84.es
    w = np.sqrt(1.0 - es * np.sin(phi) ** 2)
    m_lat = np.pi / 180.0 * a * (1.0 - es) / w ** 3
    m_lon = np.pi / 180.0 * a * np.cos(phi) / w
    if np.ndim(lat) == 0:
        return float(m_lat), float(m_lon)
    return m_lat, m_lon


def degree_half_extent(lat0: float, lon0: float, radius_m: float) -> tuple[float, float]:
    """Half-extent ``(dlat, dlon)`` in degrees of a disc of ``radius_m``
    metres centred on ``(lat0, lon0)``: the lat/lon offsets of the points
    ``radius_m`` due north/south and east/west of the centre (WGS84
    geodesics), so ``[lat0 - dlat, lat0 + dlat] x [lon0 - dlon, lon0 +
    dlon]`` is the bounding box the disc spans along its centre lines.
    """
    lons, lats, _ = WGS84.fwd([lon0] * 4, [lat0] * 4, [0.0, 180.0, 90.0, 270.0],
                              [float(radius_m)] * 4)
    dlat = 0.5 * (lats[0] - lats[1])
    dlon = 0.5 * ((lons[2] - lons[3]) % 360.0)   # unwrap across the antimeridian
    return float(dlat), float(dlon)
