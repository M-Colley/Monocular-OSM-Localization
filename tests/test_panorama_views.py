import numpy as np
import pytest

from monocular_osm.panorama_views import PROFILE, perspective_maps, rectify_reference, reference_views


def test_odd_grid_centre_is_panorama_equator_and_declared_yaw():
    for yaw in (0., 90., 180., 270.):
        x, y = perspective_maps(41, 80, yaw, size=5)
        assert x[2, 2] == pytest.approx(((yaw / 360. + .5) * 80 - .5) % 80)
        assert y[2, 2] == pytest.approx(20.)
        assert not x.flags.writeable and not y.flags.writeable
        assert x.dtype == y.dtype == np.float32


def test_inverse_map_implements_analytic_spherical_rays_without_vertical_wrap():
    x, y = perspective_maps(101, 202, 0., size=3)
    ray_x, ray_y = 2. / 3., -2. / 3.
    expected_lon = np.arctan2(ray_x, 1.)
    expected_lat = np.arctan2(-ray_y, np.hypot(ray_x, 1.))
    assert x[0, 2] == pytest.approx((expected_lon / (2*np.pi) + .5) * 202 - .5, abs=1e-5)
    assert y[0, 2] == pytest.approx((.5 - expected_lat / np.pi) * 101 - .5, abs=1e-5)
    assert y[0, 2] < 50 < y[2, 2]
    assert np.allclose(y[1], 50)
    assert 0 < y.min() < y.max() < 100


def test_periodic_yaw_and_complete_square_warp_share_normalized_angular_coordinates():
    x, y = perspective_maps(512, 512, -90., size=9)
    xx, yy = perspective_maps(256, 1024, 270., size=9)
    np.testing.assert_allclose((x+.5)/512 % 1, (xx+.5)/1024 % 1, atol=1e-7)
    np.testing.assert_allclose((y+.5)/512, (yy+.5)/256, atol=1e-7)


def test_seam_is_interpolated_periodically_and_channels_are_preserved():
    image = np.zeros((7, 8, 3), np.uint8)
    image[:, 0] = [20, 40, 60]
    image[:, -1] = [180, 200, 220]
    result = rectify_reference(image, 180., size=3)
    np.testing.assert_array_equal(result[1, 1], [100, 120, 140])
    assert result.dtype == np.uint8 and result.shape == (3, 3, 3)


@pytest.mark.parametrize("camera_type", [None, "perspective", "fisheye", "unknown", "", "SPHERICAL"])
def test_unknown_or_nonpanorama_is_retained_without_projection_assumption(camera_type):
    pixels = np.ones((8, 9, 3), np.uint8)
    result = reference_views(pixels, camera_type)
    assert len(result) == 1 and result[0][0] is None and result[0][1] is pixels


@pytest.mark.parametrize("camera_type", ["spherical", "equirectangular"])
def test_both_official_panorama_types_produce_all_four_fixed_views(camera_type):
    pixels = np.full((8, 16, 3), [10, 20, 30], np.uint8)
    result = reference_views(pixels, camera_type)
    assert [yaw for yaw, _ in result] == list(PROFILE["yaw_offsets_deg"])
    assert all(view.shape == (512, 512, 3) and view.dtype == np.uint8 for _, view in result)
    for _, view in result:
        np.testing.assert_array_equal(view, np.broadcast_to([10, 20, 30], view.shape))


@pytest.mark.parametrize("kw", [dict(height=True), dict(width=0), dict(size=1025), dict(size=2.5),
    dict(yaw_deg=float("nan")), dict(yaw_deg=True), dict(yaw_deg="bad"), dict(fov_deg=0),
    dict(fov_deg=151), dict(fov_deg=float("inf"))])
def test_invalid_or_unbounded_projection_request_refused(kw):
    args = dict(height=512, width=512, yaw_deg=0., size=5, fov_deg=90.)
    args.update(kw)
    with pytest.raises(ValueError):
        perspective_maps(**args)


@pytest.mark.parametrize("pixels", [np.zeros((5, 5), np.uint8), np.zeros((5, 5, 4), np.uint8),
    np.zeros((5, 5, 3), np.float32), np.zeros((0, 5, 3), np.uint8)])
def test_invalid_image_refused_before_remapping(pixels):
    with pytest.raises(ValueError):
        rectify_reference(pixels, 0., size=5)


@pytest.mark.parametrize("kw", [dict(height=512.), dict(yaw_deg=False), dict(size=True)])
def test_invalid_equal_numeric_keys_cannot_bypass_validation_after_cache_warmup(kw):
    perspective_maps(512, 512, 0., size=1)
    args = dict(height=512, width=512, yaw_deg=0., size=1)
    args.update(kw)
    with pytest.raises(ValueError):
        perspective_maps(**args)


def test_cached_maps_cannot_be_mutated_by_reenabling_numpy_write_flags():
    x, y = perspective_maps(32, 64, 0., size=5)
    saved = x.copy()
    for array in (x, y, x.base, y.base):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    np.testing.assert_array_equal(perspective_maps(32, 64, 0., size=5)[0], saved)


@pytest.mark.parametrize("angle", [complex(1, 2), np.complex128(1+2j), [0.], np.array(0.)])
def test_nonreal_or_nonscalar_angles_refused_before_cache_lookup(angle):
    with pytest.raises(ValueError):
        perspective_maps(32, 64, angle, size=5)


def test_opencv_dimension_limit_refused_before_remapping_and_integer_products_cannot_overflow():
    with pytest.raises(ValueError, match="OpenCV"):
        rectify_reference(np.zeros((2, 40000, 3), np.uint8), 0., size=5)
    with pytest.raises(ValueError):
        perspective_maps(np.int64(2**62), np.int64(4), 0., size=5)


def test_undersampled_polar_pixel_centres_are_refused_without_vertical_wrap():
    with pytest.raises(ValueError, match="vertical source domain"):
        perspective_maps(2, 8, 0., size=512, fov_deg=150.)
