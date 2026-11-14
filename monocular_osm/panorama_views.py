"""Perspective views of complete equirectangular reference rasters.

This experimental transform preserves IDs and image channels. It needs public
camera-type metadata; it never infers projection from location or similarity.
Four horizontal views cover the azimuth around the panorama's own coordinate
frame. Their yaw offsets are not world headings or vehicle travel directions.
Stitched panoramas are only approximately central cameras.
"""
from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType

import cv2
import numpy as np


PROFILE = MappingProxyType({
    "version": "reference-panorama-perspective-v1",
    "experimental": True,
    "panorama_camera_types": ("spherical", "equirectangular"),
    "source_contract": "complete equirectangular raster; normalized angular extent survives square resize",
    "yaw_offsets_deg": (0., 90., 180., 270.),
    "horizontal_fov_deg": 90.,
    "output_size": 512,
    "pitch_deg": 0.,
    "interpolation": "OpenCV_INTER_LINEAR",
    "seam": "periodic horizontal; vertical samples remain inside the sphere",
    "pixel_coordinates": "centres; longitude=atan2(x,z), latitude=atan2(-y,hypot(x,z))",
    "unknown_or_other_camera_type": "one unmodified reference; no projection assumption",
    "heading_claim": False,
    "central_camera_assumption": "approximate for stitched panoramas",
    "default_policy_adopted": False,
})


def _image(value):
    array = np.asarray(value)
    if (array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3
            or min(array.shape[:2]) < 2 or array.size > 128 * 1024**2):
        raise ValueError("bounded nonempty uint8 H,W,3 reference pixels required")
    return array


def perspective_maps(height, width, yaw_deg, *, size=512, fov_deg=90.):
    """Return bounded read-only inverse maps; input aspect ratio is arbitrary.

    The source must contain the entire 360 x 180 degree angular raster, even
    when a previous image resize changed its aspect ratio. This function does
    not repair a cropped, fisheye or non-equirectangular input. Very small
    source grids can be refused when a ray lies beyond their polar pixel
    centres; no sample is silently wrapped to the opposite pole.
    """
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
            for v in (height, width, size))
            or isinstance(yaw_deg, (bool, np.bool_))
            or not isinstance(yaw_deg, (int, float, np.integer, np.floating))
            or isinstance(fov_deg, (bool, np.bool_))
            or not isinstance(fov_deg, (int, float, np.integer, np.floating))):
        raise ValueError("bounded integer source/output grids and scalar angles required")
    height, width, size = int(height), int(width), int(size)
    if (not 2 <= min(height, width) or max(height, width) >= 32767
            or height * width * 3 > 128 * 1024**2 or not 1 <= size <= 1024):
        raise ValueError("source/output grid exceeds image or OpenCV remap bounds")
    try:
        yaw, fov = float(yaw_deg), float(fov_deg)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("finite numeric angles required") from error
    if not np.isfinite(yaw) or not np.isfinite(fov) or not 0 < fov <= 150.:
        raise ValueError("finite yaw and FOV in (0,150] required")
    # Validate before cache lookup: Python numeric equality must never let an
    # invalid bool/float dimension hit a previously valid integer cache key.
    return _cached_maps(height, width, yaw % 360., size, fov)


@lru_cache(maxsize=4)
def _cached_maps(height, width, yaw, size, fov):
    # Four views have an arbitrary panorama origin, without world heading.
    yaw = np.deg2rad(yaw)
    axis = ((np.arange(size, dtype=np.float64) + .5) * (2. / size) - 1.)
    axis *= np.tan(np.deg2rad(fov) / 2.)
    x, y = np.meshgrid(axis, axis)
    longitude = np.arctan2(x, 1.) + yaw
    latitude = np.arctan2(-y, np.hypot(x, 1.))
    map_x = np.ascontiguousarray(((longitude / (2. * np.pi) + .5) * width - .5) % width, dtype=np.float32)
    map_y = np.ascontiguousarray((.5 - latitude / np.pi) * height - .5, dtype=np.float32)
    if not (np.isfinite(map_x).all() and np.isfinite(map_y).all()
            and np.all((map_y >= 0.) & (map_y <= height - 1.))):
        raise ValueError("perspective grid left the vertical source domain")
    # Back with immutable bytes. Merely clearing writeable on an owning array
    # lets callers re-enable it and poison all subsequent cache consumers.
    map_x = np.frombuffer(map_x.tobytes(), dtype=np.float32).reshape(size, size)
    map_y = np.frombuffer(map_y.tobytes(), dtype=np.float32).reshape(size, size)
    return map_x, map_y


def rectify_reference(image, yaw_deg, *, size=512, fov_deg=90.):
    """Project one virtual pinhole view, retaining the original channel order."""
    pixels = _image(image)
    map_x, map_y = perspective_maps(*pixels.shape[:2], yaw_deg, size=size, fov_deg=fov_deg)
    # BORDER_WRAP provides seam interpolation. Vertical coordinates are checked
    # above and cannot wrap to the opposite pole.
    return cv2.remap(pixels, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def reference_views(image, camera_type):
    """Return (panorama-yaw-offset, pixels) pairs under the fixed profile.

    A None offset means an unmodified source, including unknown/fisheye types.
    Consumers must not label that case a verified pinhole camera.
    """
    pixels = _image(image)
    if camera_type not in PROFILE["panorama_camera_types"]:
        return [(None, pixels)]
    return [(yaw, rectify_reference(pixels, yaw, size=PROFILE["output_size"],
                                   fov_deg=PROFILE["horizontal_fov_deg"]))
            for yaw in PROFILE["yaw_offsets_deg"]]
