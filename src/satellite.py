"""Real RGB satellite tile fetching for cross-view matching.

Both the deep-embedding retrieval channel (:mod:`embedding_retrieval`) and
the BevSplat channel (:mod:`bev_splat_match`) originally had only two
satellite-tile sources, *neither of which is a real RGB satellite photo*:

* **OSM schematic raster** — a black-on-white line drawing.  A huge domain
  gap from any photographic ground / IPM image, which is why ORB and
  ResNet-embedding comparison against it ranked wrong candidates to the top.
* **GeoTessera DINOv2-PCA** — real *satellite-derived* but rendered as a
  false-colour PCA of 128 embedding channels, not an RGB photo.  Visually
  near-identical for all inner-city tiles → non-discriminative (BevSplat
  scored 0.5947 for 9/10 candidates).

This module fetches **real RGB orthoimagery** (Esri World Imagery by
default) for a candidate location via ``contextily``.  That is exactly the
domain BevSplat's KITTI checkpoints were trained on, and the right domain
to compare a photographic IPM stitch against — closing the domain gap that
made those two channels anti-correlated with ground truth.

Network access is required for the first fetch of a tile; ``contextily``
caches tiles on disk afterwards.
"""

from __future__ import annotations

import cv2
import numpy as np
from pyproj import Transformer

from .osm_data import RoadGraph
from .trajectory_matching import MatchCandidate

# provider alias -> (xyzservices family attr, layer attr)
_PROVIDERS: dict[str, tuple[str, str]] = {
    "esri": ("Esri", "WorldImagery"),
    "satellite": ("Esri", "WorldImagery"),
    "esri_worldimagery": ("Esri", "WorldImagery"),
}


def _provider_source(provider: str):
    import contextily as cx

    key = provider.lower()
    if key not in _PROVIDERS:
        raise ValueError(
            f"unsupported satellite provider {provider!r}; "
            f"choose from {sorted(_PROVIDERS)}"
        )
    family, layer = _PROVIDERS[key]
    return getattr(getattr(cx.providers, family), layer)


def fetch_satellite_tile(
    lon: float,
    lat: float,
    *,
    half_extent_m: float = 60.0,
    size: int = 512,
    provider: str = "esri",
) -> np.ndarray:
    """Fetch a real RGB satellite tile centred on ``(lon, lat)``.

    Returns an ``(size, size, 3)`` uint8 RGB image covering a square of
    side ``2 * half_extent_m`` on the ground.  The square is defined in a
    local azimuthal-equidistant projection (true metres, no latitude
    distortion), the mosaic returned by ``contextily`` is cropped exactly
    to that bbox in Web-Mercator pixels, then resized to ``size``.

    Requires network access on first call (tiles are cached afterwards).
    """
    import contextily as cx

    # Square bbox of side 2*half_extent_m around the centre, defined in a
    # local AEQD frame (true metres) and reprojected to lon/lat corners.
    aeqd = (
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    to_ll = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)
    H = float(half_extent_m)
    corners_m = [(-H, -H), (H, -H), (H, H), (-H, H)]
    lons, lats = [], []
    for x, y in corners_m:
        clon, clat = to_ll.transform(x, y)
        lons.append(clon)
        lats.append(clat)
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)

    img, extent = cx.bounds2img(
        west, south, east, north, ll=True, source=_provider_source(provider),
    )
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # contextily returns whole tiles covering (and overflowing) the bbox.
    # Crop precisely to the requested bbox using the Web-Mercator extent.
    cropped = _crop_to_bbox_3857(img, extent, west, south, east, north)
    if cropped.shape[0] >= 2 and cropped.shape[1] >= 2:
        img = cropped

    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def _crop_to_bbox_3857(
    img: np.ndarray,
    extent: tuple[float, float, float, float],
    west: float,
    south: float,
    east: float,
    north: float,
) -> np.ndarray:
    """Crop the mosaic ``img`` (whose Web-Mercator extent is ``extent``,
    in matplotlib ``(left, right, bottom, top)`` order) to the lon/lat
    bbox ``(west, south, east, north)``."""
    left, right, bottom, top = extent
    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    wx, sy = to_merc.transform(west, south)
    ex, ny = to_merc.transform(east, north)
    h_img, w_img = img.shape[:2]
    if right == left or top == bottom:
        return img

    def _col(x: float) -> int:
        return int(round((x - left) / (right - left) * w_img))

    def _row(y: float) -> int:
        return int(round((top - y) / (top - bottom) * h_img))

    col0, col1 = sorted((_col(wx), _col(ex)))
    row0, row1 = sorted((_row(ny), _row(sy)))
    col0 = max(0, min(col0, w_img - 1))
    col1 = max(col0 + 1, min(col1, w_img))
    row0 = max(0, min(row0, h_img - 1))
    row1 = max(row0 + 1, min(row1, h_img))
    return img[row0:row1, col0:col1]


def satellite_tile_for_candidate(
    road: RoadGraph,
    cand: MatchCandidate,
    *,
    half_extent_m: float,
    size: int,
    provider: str = "esri",
) -> np.ndarray:
    """Real RGB satellite tile centred on a candidate's walk centroid."""
    center_xy = cand.walk_xy.mean(axis=0)
    transformer = Transformer.from_crs(road.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(float(center_xy[0]), float(center_xy[1]))
    return fetch_satellite_tile(
        lon, lat, half_extent_m=half_extent_m, size=size, provider=provider,
    )
