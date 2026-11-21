"""Invented pixels/cameras only; no models, actual files, videos, maps or GT."""
from dataclasses import replace

import cv2
import numpy as np
import pytest

from monocular_osm import dvlt_native_geometry as g


@pytest.fixture(autouse=True)
def cpu_two():
    old = cv2.getNumThreads()
    cv2.setNumThreads(2)
    try:
        yield
    finally:
        cv2.setNumThreads(old)


def scene(noise=0., plane=False, wide=False):
    rng = np.random.default_rng(12)
    n = 80
    xyz = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1, 1, n),
                           np.full(n, 10.) if plane else rng.uniform(8, 15, n)])
    if wide:
        xyz[:, :2] *= 2.
    c2w = np.repeat(np.column_stack([np.eye(3), np.zeros(3)])[None], 7, axis=0)
    c2w[:, 0, 3] = np.arange(7)*.12
    nativeK = np.array([[500., 0., 320.], [0., 490., 240.], [0., 0., 1.]])
    if wide:
        nativeK[0, 0], nativeK[1, 1] = 400., 390.
    transform = g.make_transform((480, 640))
    rh, rw = transform.resized_hw
    top, left = transform.crop_top_left
    pt, pl = transform.pad_top_left
    kp = nativeK.copy()
    kp[0, 0] *= rw/640
    kp[1, 1] *= rh/480
    kp[0, 2] = (nativeK[0, 2]+.5)*rw/640-left+pl
    kp[1, 2] = (nativeK[1, 2]+.5)*rh/480-top+pt
    K = np.repeat(kp[None], 7, axis=0)
    pixels = []
    for i in range(7):
        q = (xyz-c2w[i, :, 3]) @ nativeK.T
        pixels.append(q[:, :2]/q[:, 2:])
    points = np.array(pixels)+rng.normal(0, noise, (7, n, 2))
    ids = np.arange(n, dtype=np.int32)
    tracks = dict(points_px=points, valid=np.ones((7, n), bool), seed_ids=ids,
                  fit_seed=ids % 2 == 0, check_seed=ids % 2 == 1,
                  ordinals=np.arange(7, dtype=np.int64)*4,
                  timestamps=np.arange(7, dtype=np.float64)/5., native_shape=(480, 640, 3),
                  profile_sha256=g.PROFILE_SHA256)
    return tracks, c2w, K, transform


def test_correct_fixed_poses_validate_without_any_pose_or_scale_fit(monkeypatch):
    for name in ('solvePnP', 'solvePnPRansac', 'findEssentialMat', 'findFundamentalMat'):
        monkeypatch.setattr(cv2, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError('fit called')))
    tracks, c2w, K, t = scene()
    before = [a.tobytes() for a in (tracks['points_px'], c2w, K)]
    result = g.validate_geometry(tracks, c2w, K, t)
    p = result['pose_only']
    assert p['passes_candidate_defaults'] and p['passes_proposed_robust_full_check']
    assert np.nanmax(p['errors_native_px']) < 1e-9
    assert p['errors_native_px'].shape == (7, 80)
    assert [r['frame'] for r in p['later_view_metrics']] == [1, 2, 4, 5, 6]
    assert all(r['original_check_denominator'] == 40 for r in p['later_view_metrics'])
    assert [a.tobytes() for a in (tracks['points_px'], c2w, K)] == before
    assert not result['summary']['metric_scale_recovered']
    assert not result['summary']['global_or_cross_window_gauge']


def test_wrong_later_pose_is_independently_falsified():
    tracks, c2w, K, t = scene()
    c2w[4, 1, 3] = 1.
    p = g.validate_geometry(tracks, c2w, K, t)['pose_only']
    assert not p['passes_candidate_defaults'] and not p['passes_proposed_robust_full_check']
    assert np.median(p['errors_native_px'][4, 1::2]) > 20


def test_small_noise_and_five_percent_outliers_are_separate_robust_scope():
    tracks, c2w, K, t = scene(.1)
    tracks['points_px'][4, [1, 3], 0] += 30.
    p = g.validate_geometry(tracks, c2w, K, t)['pose_only']
    assert not p['passes_candidate_defaults']  # Original all-supported RMSE gate.
    assert p['passes_proposed_robust_full_check']
    assert p['robust_criterion_is_prospective_not_admitted']
    assert p['projection_valid'][:, [1, 3]].all()  # Outliers were not removed.


def test_invalid_tracks_remain_in_full_check_denominator():
    tracks, c2w, K, t = scene()
    tracks['valid'][2, [1, 3]] = False
    tracks['points_px'][2, [1, 3]] = np.nan
    p = g.validate_geometry(tracks, c2w, K, t)['pose_only']
    assert np.isinf(p['errors_with_invalid_inf'][:, [1, 3]]).all()
    assert np.all(p['check_coverage_by_view'] == .95)
    assert p['passes_proposed_robust_full_check']
    assert all(r['original_check_denominator'] == 40 and r['eligible_check_denominator'] == 38
               and r['excluded_check_seed_ids'] == [1, 3] for r in p['later_view_metrics'])
    tracks['valid'][:, 1::4] = False
    assert not g.validate_geometry(tracks, c2w, K, t)['pose_only']['passes_proposed_robust_full_check']


@pytest.mark.parametrize('factor', [0., 1e-6])
def test_rotation_only_and_low_parallax_never_claim_stop(factor):
    tracks, c2w, K, t = scene()
    if factor == 0:
        c2w[:, :, 3] = 0.
    else:
        # Scale translations and observed parallax, not just the arbitrary gauge.
        p0 = tracks['points_px'][0].copy()
        tracks['points_px'] = p0[None]+factor*(tracks['points_px']-p0[None])
        c2w[:, :, 3] *= factor
    r = g.validate_geometry(tracks, c2w, K, t)
    assert not r['pose_only']['passes_candidate_defaults']
    assert not r['pose_only']['passes_proposed_robust_full_check']
    assert not r['summary']['measured_stop']


def test_global_similarity_gauge_changes_no_image_evidence():
    tracks, c2w, K, t = scene()
    original = g.validate_geometry(tracks, c2w, K, t)['pose_only']
    Q = cv2.Rodrigues(np.array([.2, -.4, .1]))[0]
    transformed = c2w.copy()
    transformed[:, :, :3] = Q @ c2w[:, :, :3]
    transformed[:, :, 3] = 17.*(c2w[:, :, 3] @ Q.T)+[3., -5., 8.]
    changed = g.validate_geometry(tracks, transformed, K, t)['pose_only']
    np.testing.assert_allclose(changed['predicted_native_px'], original['predicted_native_px'], atol=1e-9)
    assert changed['passes_candidate_defaults']


def test_model_point_check_is_separate_and_not_same_view_certification():
    tracks, c2w, K, t = scene(plane=True, wide=True)
    ph, pw = t.processed_hw
    yy, xx = np.indices((ph, pw))
    rays = np.stack([xx+.5, yy+.5, np.ones_like(xx)], axis=-1) @ np.linalg.inv(K[0]).T
    world = np.empty((7, ph, pw, 3), np.float32)
    for i in range(7):
        world[i] = rays*10.+c2w[i, :, 3]
    world.flags.writeable = False
    r = g.validate_geometry(tracks, c2w, K, t, world_points=world)
    assert r['pose_only']['passes_candidate_defaults']
    assert r['model_depth_pose']['passes_candidate_defaults']
    assert not r['model_depth_pose']['same_view_consistency_is_independent_evidence']
    bad = world.copy()
    bad[0, :, :, 2] *= .2
    r = g.validate_geometry(tracks, c2w, K, t, world_points=bad)
    assert r['pose_only']['passes_candidate_defaults']
    assert not r['model_depth_pose']['passes_candidate_defaults']


@pytest.mark.parametrize('native_hw, padded_hw', [((480, 854), None), ((481, 641), (504, 504)), ((1080, 1920), None)])
def test_affine_exact_rounding_pixel_centers_crop_pad_and_inverse(native_hw, padded_hw):
    t = g.make_transform(native_hw, padded_hw)
    h, w = native_hw
    p = np.array([[0., 0.], [w-1., h-1.], [w/2-.5, h/2-.5]])
    q = t.native_to_processed(p)
    rh, rw = round(h*504/max(h, w)), round(w*504/max(h, w))
    ch, cw = rh//14*14, rw//14*14
    ph, pw = t.processed_hw
    expected = (p+.5)*[rw/w, rh/h]-[(rw-cw)//2, (rh-ch)//2]+[(pw-cw)//2, (ph-ch)//2]
    np.testing.assert_allclose(q, expected, atol=1e-12)
    np.testing.assert_allclose(t.processed_to_native(q), p, atol=1e-12)
    # A discarded odd crop pixel stays on bottom/right, not a fabricated recenter.
    assert q[2, 0] == pw/2+((rw-cw) % 2)/2
    assert q[2, 1] == ph/2+((rh-ch) % 2)/2


def test_support_full_pyramid_border_and_swept_rectangle():
    mask = np.zeros((512, 800), bool)
    point = np.array([[250., 250.], [109., 250.], [110., 250.]])
    assert g.support_valid(g._integral(mask), point, point).tolist() == [True, False, True]
    mask[250, 360] = True  # Exactly the conservative dependency boundary.
    assert not g.support_valid(g._integral(mask), point[:1], point[:1])[0]
    mask[:] = False
    mask[250, 375] = True
    left, right = np.array([[250., 250.]]), np.array([[500., 250.]])
    assert g.support_valid(g._integral(mask), left, left)[0]
    assert g.support_valid(g._integral(mask), right, right)[0]
    assert not g.support_valid(g._integral(mask), left, right)[0]


def moving_frames():
    rng = np.random.default_rng(5)
    # Genuine textured pixels, not individually transported point patches.
    image = rng.integers(0, 256, (512, 640, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    frames = [cv2.warpAffine(image, np.array([[1., 0., float(t)], [0., 1., 0.]]), (640, 512)) for t in range(7)]
    masks = [np.zeros((512, 640), bool) for _ in range(7)]
    for frame, mask in zip(frames, masks):
        mask[:50] = mask[-50:] = True
        frame[mask] = 0
    return frames, masks


def test_real_masked_tracker_has_usable_support_and_no_synthetic_edge_seeds():
    frames, masks = moving_frames()
    before = [(f.tobytes(), m.tobytes()) for f, m in zip(frames, masks)]
    for a in frames+masks:
        a.flags.writeable = False
    tracks = g.track_masked_window(frames, masks, np.arange(7, dtype=np.int64), np.arange(7, dtype=np.float64))
    assert tracks['persistent_count'] >= 40
    assert not tracks['hidden_lk_iteration_paths_certified']
    assert np.all((tracks['points_px'][0, :, 1] >= 160) & (tracks['points_px'][0, :, 1] <= 351))
    assert np.array_equal(tracks['check_seed'], tracks['seed_ids'] % 2 == 1)
    assert [(f.tobytes(), m.tobytes()) for f, m in zip(frames, masks)] == before
    good = tracks['valid'].all(axis=0)
    np.testing.assert_allclose(tracks['points_px'][6, good]-tracks['points_px'][0, good],
                               np.tile([6., 0.], (int(good.sum()), 1)), atol=.1)


def test_arbitrary_tiny_model_scale_is_not_a_physical_baseline_threshold():
    tracks, c2w, K, t = scene()
    c2w[:, :, 3] *= 1e-100
    assert g.validate_geometry(tracks, c2w, K, t)['pose_only']['passes_candidate_defaults']


@pytest.mark.parametrize('bad', ['Rreflection', 'Rnonorthogonal', 'Knegative', 'Kskew', 'Krow', 'parity', 'clock', 'forged_affine'])
def test_camera_clock_identity_and_affine_contracts_refuse(bad):
    tracks, c2w, K, t = scene()
    if bad == 'Rreflection': c2w[1, 0, 0] = -1
    if bad == 'Rnonorthogonal': c2w[1, 0, 0] = 1.1
    if bad == 'Knegative': K[2, 0, 0] = -1
    if bad == 'Kskew': K[2, 0, 1] = .01
    if bad == 'Krow': K[2, 2, 0] = .01
    if bad == 'parity': tracks['check_seed'] = tracks['fit_seed'].copy()
    if bad == 'clock': tracks['timestamps'][3] = tracks['timestamps'][2]
    if bad == 'forged_affine': t = replace(t, crop_top_left=(2, 2))
    with pytest.raises(ValueError):
        g.validate_geometry(tracks, c2w, K, t)


@pytest.mark.parametrize('shape', [(True, 640), (0, 640), (1, 8_500_001), (8_500_001, 1)])
def test_native_shape_caps_before_body(shape):
    with pytest.raises(ValueError):
        g.make_transform(shape)


def test_no_seeds_and_all_masked_are_honest_unavailable():
    frames = [np.zeros((256, 256, 3), np.uint8) for _ in range(7)]
    masks = [np.ones((256, 256), bool) for _ in range(7)]
    tracks = g.track_masked_window(frames, masks, np.arange(7, dtype=np.int64), np.arange(7, dtype=np.float64))
    assert tracks['status'] == 'unavailable_no_seeds' and tracks['points_px'].shape == (7, 0, 2)
    assert tracks['rejection_reason'].shape == (7, 0)
    _, c2w, K, _ = scene()
    r = g.validate_geometry(tracks, c2w, K, g.make_transform((256, 256)))
    assert not r['pose_only']['passes_candidate_defaults']


def test_model_point_nonfinite_check_is_retained_not_confidence_filtered():
    tracks, c2w, K, t = scene(plane=True, wide=True)
    ph, pw = t.processed_hw
    yy, xx = np.indices((ph, pw))
    rays = np.stack([xx+.5, yy+.5, np.ones_like(xx)], axis=-1) @ np.linalg.inv(K[0]).T
    world = np.repeat((rays*10.)[None], 7, axis=0)
    q = t.native_to_processed(tracks['points_px'][0, 1:2])-.5
    x, y = np.floor(q[0]).astype(int)
    world[0, y:y+2, x:x+2] = np.nan
    r = g.validate_geometry(tracks, c2w, K, t, world_points=world)
    assert r['pose_only']['passes_candidate_defaults']
    report = r['model_depth_pose']
    assert not report['passes_candidate_defaults'] and report['passes_proposed_robust_full_check']
    assert 1 in report['later_view_metrics'][0]['invalid_check_seed_ids']
    assert np.isinf(report['errors_with_invalid_inf'][:, 1]).all()


def test_unmasked_pixel_or_missing_mask_is_not_sanitized_input():
    frames, masks = moving_frames()
    frames[0][0, 0] = 1
    with pytest.raises(ValueError, match='black'):
        g.track_masked_window(frames, masks, np.arange(7, dtype=np.int64), np.arange(7, dtype=np.float64))
    with pytest.raises(ValueError):
        g.track_masked_window(frames, masks[:6], np.arange(7, dtype=np.int64), np.arange(7, dtype=np.float64))


def test_callbacks_can_refuse_before_geometry_and_tracking_continues():
    def refuse():
        raise TimeoutError('private deadline')
    tracks, c2w, K, t = scene()
    with pytest.raises(TimeoutError):
        g.validate_geometry(tracks, c2w, K, t, check=refuse)
    frames, masks = moving_frames()
    with pytest.raises(TimeoutError):
        g.track_masked_window(frames, masks, tracks['ordinals'], tracks['timestamps'], check=refuse)


def test_late_mask_contamination_retains_all_ids_attempts_and_reasons():
    frames, masks = moving_frames()
    masks[2][220:230, 310:320] = True
    frames[2][masks[2]] = 0
    tracks = g.track_masked_window(frames, masks, np.arange(7, dtype=np.int64), np.arange(7, dtype=np.float64))
    n = len(tracks['seed_ids'])
    rejected = (tracks['rejection_reason'][2] == 'mask_or_pyramid_support')
    assert rejected.any() and np.isfinite(tracks['proposed_points_px'][2, rejected]).all()
    assert tracks['raw_forward_valid'][2, rejected].all() and not tracks['valid'][2, rejected].any()
    assert np.array_equal(tracks['seed_ids'], np.arange(n, dtype=np.int32))
    assert tracks['forward_status'].shape == (6, n)
    assert (tracks['rejection_reason'][3, rejected] == 'not_active').all()


def test_insufficient_count_still_exports_observable_continuous_residuals():
    tracks, c2w, K, t = scene()
    tracks['valid'][:, 40:] = False  # Twenty original CHECK remain observable.
    tracks['valid'][:, 39] = False   # Nineteen is below the inherited minimum.
    p = g.validate_geometry(tracks, c2w, K, t)['pose_only']
    assert not p['passes_candidate_defaults'] and not p['passes_proposed_robust_full_check']
    assert np.isfinite(p['residual_native_px'][:, 1:38:2]).all()
    assert np.isinf(p['errors_with_invalid_inf'][:, 39:]).all()


def test_rotation_only_can_have_tiny_model_reprojection_but_translation_unknown():
    tracks, c2w, K, t = scene(plane=True)
    tracks['points_px'][:] = tracks['points_px'][0]
    c2w[:, :, 3] = 0.
    ph, pw = t.processed_hw
    yy, xx = np.indices((ph, pw))
    rays = np.stack([xx+.5, yy+.5, np.ones_like(xx)], axis=-1) @ np.linalg.inv(K[0]).T
    world = np.repeat((rays*10.)[None], 7, axis=0)
    r = g.validate_geometry(tracks, c2w, K, t, world_points=world)
    report = r['model_depth_pose']
    assert np.nanmax(report['errors_native_px']) < 1e-9
    assert not report['passes_candidate_defaults'] and not report['passes_proposed_robust_full_check']
    assert 'unknown' in report['reason'] and not r['summary']['observable_anchor_for_candidate_defaults']


def test_inf_quantiles_preserve_full_denominator_without_nan_or_dropping():
    assert g._quantile_with_inf(np.array([0., 0., np.inf, np.inf]), .9) == np.inf
    assert g._quantile_with_inf(np.array([0., np.inf]), .5) == np.inf
    finite = np.array([3., 1., 2., 7., 4.])
    assert g._quantile_with_inf(finite, .9) == np.percentile(finite, 90)
    assert g._quantile_with_inf(np.array([], np.float64), .5) is None


def test_maximum_shape_refused_before_copy_or_full_pixel_scan():
    frame = np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (8_500_001, 1, 3))
    mask = np.broadcast_to(np.zeros((1, 1), bool), (8_500_001, 1))
    with pytest.raises(ValueError, match='bounded native'):
        g.track_masked_window([frame]*7, [mask]*7, np.arange(7, dtype=np.int64), np.arange(7, dtype=np.float64))


def test_narrow_translating_plane_can_be_rotation_ambiguous_within_noise_floor():
    tracks, c2w, K, t = scene(plane=True)
    r = g.validate_geometry(tracks, c2w, K, t)
    assert r['rotation_only']['explains_with_existing_rotation_floor']
    assert not r['pose_only']['passes_candidate_defaults']
    assert np.nanmax(r['pose_only']['errors_native_px']) < 1e-9


def test_wrong_model_rotation_cannot_manufacture_observable_parallax():
    tracks, c2w, K, t = scene()
    first = t.native_to_processed(tracks['points_px'][0])
    rays = np.column_stack([first, np.ones(len(first))]) @ np.linalg.inv(K[0]).T
    for i in range(7):
        true_w2c_R = cv2.Rodrigues(np.array([0., .015*i, 0.]))[0]
        q = (rays @ true_w2c_R.T) @ K[i].T
        tracks['points_px'][i] = t.processed_to_native(q[:, :2]/q[:, 2:])
    # Model rotations remain wrong(identity), translations nonzero. No true pose
    # is passed separately to the validator, only the falsifiable model arrays.
    r = g.validate_geometry(tracks, c2w, K, t)
    assert r['summary']['anchor_median_derotated_parallax_rad'] > r['summary']['anchor_parallax_limit_rad']
    assert r['rotation_only']['explains_with_existing_rotation_floor']
    assert not r['pose_only']['passes_candidate_defaults']
    assert 'rotation-only' in r['pose_only']['reason']
