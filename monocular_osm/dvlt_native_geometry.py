"""Pure image-supported local geometry; no model, file IO or metric authority.

CHECK is held out of post-fit geometry, not unseen by the seven-image model.
Masked/pyramid support is conservative observation support, not certification of
hidden LK iteration paths, detector recall, static scenes or correct tracks.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

import cv2
import numpy as np

_PROFILE = {
    'schema': 'dvlt-native-image-geometry-v1', 'frames': 7,
    'max_native_pixels': 8_500_000, 'max_features': 800,
    'gftt_quality': .01, 'gftt_min_distance_px': 8., 'gftt_block': 7,
    'lk_window': 21, 'lk_levels': 3, 'lk_fb_px': .75,
    'support_radius_native_px': 110,
    'support_derivation': '(10+bilinear1+Scharr1)*2^3+2*(2^3-1)',
    'support_scope': 'full conservative pyramid patch and swept endpoint rectangle; not hidden LK iterates',
    'seed_extraction': 'GFTT valid-support mask, then fixed original ID parity before later rejection',
    'anchor': (0, 3), 'later_views': (1, 2, 4, 5, 6),
    'minimum_persistent_check': 20, 'assumed_native_noise_floor_px': 1.,
    'parallax_noise_multiple': 3., 'median_limit_px': 2.,
    'p90_limit_px': 4., 'rmse_limit_px': 4.,
    'rotation_alternative': 'FIT-only Wahba; all6 later CHECK median<=1px/p90<=2px; >=20 FIT/CHECK',
    'rotation_matrix_abs_tolerance': 1e-4, 'positive_unit_gauge_depth': 1e-12,
    'raw_model_baseline_rule': 'positive finite hypot; no absolute model-unit scale threshold',
    'robust_candidate_scope': 'prospective only; full extracted CHECK denominator, invalid=inf',
    'robust_minimum_valid_check': 20, 'robust_minimum_coverage': .90,
    'robust_median_limit_px': 2., 'robust_p90_limit_px': 4.,
    'model_long_side': 504, 'patch_size': 14,
    'model_point_sampling': 'bilinear original WORLD_POINTS0, requiring full finite cell; no confidence filter',
    'candidate_default_differences': (
        'model poses supplied: no FIT E/PnP support or pose-cycle gate',
        'valid-support GFTT extraction replaces unmasked comparator seed extraction',
        'native mask/pyramid support and processed-center affine added',
        'proper-rotation numerical tolerance added; no rotation projection',
        'all extracted seeds retained; tracking eligibility frozen before geometry'),
    'metric_scale_covariance_or_policy_admission': False,
}
PROFILE = MappingProxyType(_PROFILE)
PROFILE_SHA256 = hashlib.sha256(json.dumps(
    _PROFILE, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _hw(value):
    _require(isinstance(value, (tuple, list)) and len(value) == 2
             and all(type(v) is int and v > 0 for v in value)
             and value[0]*value[1] <= 8_500_000, 'bounded integer H,W required')
    return tuple(value)


def _clock(ordinals, times):
    _require(isinstance(ordinals, np.ndarray) and ordinals.dtype == np.int64
             and ordinals.shape == (7,) and np.all(ordinals >= 0)
             and np.all(ordinals[1:] > ordinals[:-1])
             and isinstance(times, np.ndarray) and times.dtype == np.float64
             and times.shape == (7,) and np.isfinite(times).all()
             and np.all(times[1:] > times[:-1]), 'exact increasing ordinals/F64 PTS required')


@dataclass(frozen=True)
class PixelTransform:
    """Native OpenCV indices <-> processed DVLT centers (index+.5)."""
    native_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    crop_top_left: tuple[int, int]
    crop_hw: tuple[int, int]
    pad_top_left: tuple[int, int]
    processed_hw: tuple[int, int]

    def _points(self, points):
        _require(isinstance(points, np.ndarray), 'pixel pairs must already be an ndarray')
        q = points
        _require(q.dtype in (np.dtype('float32'), np.dtype('float64'))
                 and q.ndim >= 1 and q.shape[-1] == 2 and q.size <= 2*7*800
                 and np.isfinite(q).all(), 'bounded finite pixel pairs required')
        return q.astype(np.float64, copy=True)

    def native_to_processed(self, points):
        q = self._points(points)
        h, w = self.native_hw
        rh, rw = self.resized_hw
        top, left = self.crop_top_left
        pt, pl = self.pad_top_left
        q[..., 0] = (q[..., 0]+.5)*(rw/w)-left+pl
        q[..., 1] = (q[..., 1]+.5)*(rh/h)-top+pt
        return q

    def processed_to_native(self, points):
        q = self._points(points)
        h, w = self.native_hw
        rh, rw = self.resized_hw
        top, left = self.crop_top_left
        pt, pl = self.pad_top_left
        q[..., 0] = (q[..., 0]+left-pl)*(w/rw)-.5
        q[..., 1] = (q[..., 1]+top-pt)*(h/rh)-.5
        return q

    def receipt(self):
        return {name: list(getattr(self, name)) for name in (
            'native_hw', 'resized_hw', 'crop_top_left', 'crop_hw',
            'pad_top_left', 'processed_hw')}


def make_transform(native_hw, processed_hw=None):
    """Match official rounding/crop; optional common batch center pad."""
    h, w = _hw(native_hw)
    scale = 504/max(h, w)
    rh, rw = round(h*scale), round(w*scale)
    _require(min(rh, rw) >= 14, 'processed short side cannot hold one patch')
    ch, cw = max(14, rh//14*14), max(14, rw//14*14)
    ph, pw = (ch, cw) if processed_hw is None else _hw(processed_hw)
    _require(ch <= ph <= 504 and cw <= pw <= 504 and ph % 14 == pw % 14 == 0,
             'official bounded divisible common batch shape required')
    return PixelTransform((h, w), (rh, rw), ((rh-ch)//2, (rw-cw)//2),
                          (ch, cw), ((ph-ch)//2, (pw-cw)//2), (ph, pw))


def _integral(mask):
    return cv2.integral(mask.astype(np.uint8), sdepth=cv2.CV_32S)


def support_valid(mask_integral, first, last):
    """Conservative swept patch rectangle must be native and unmasked."""
    _require(isinstance(mask_integral, np.ndarray) and mask_integral.dtype == np.int32
             and mask_integral.ndim == 2, 'int32 mask integral required')
    _require(isinstance(first, np.ndarray) and isinstance(last, np.ndarray), 'endpoint arrays required')
    a, b = first, last
    _require(a.dtype == b.dtype == np.float64 and a.shape == b.shape
             and a.ndim == 2 and a.shape[1] == 2 and len(a) <= 800,
             'bounded F64 endpoint pairs required')
    result = np.zeros(len(a), bool)
    ids = np.flatnonzero(np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1))
    if not len(ids):
        return result
    low = np.floor(np.minimum(a[ids], b[ids]))-110
    high = np.ceil(np.maximum(a[ids], b[ids]))+110
    h, w = np.array(mask_integral.shape)-1
    inside = (low >= 0).all(axis=1) & (high[:, 0] < w) & (high[:, 1] < h)
    ids, low, high = ids[inside], low[inside].astype(np.int64), high[inside].astype(np.int64)
    if len(ids):
        x0, y0 = low.T
        x1, y1 = (high+1).T
        count = mask_integral[y1, x1]-mask_integral[y0, x1]-mask_integral[y1, x0]+mask_integral[y0, x0]
        result[ids] = count == 0
    return result


def track_masked_window(frames_bgr, masks_bool, ordinals, timestamps, *, check=lambda: None):
    """Support-masked seed extraction; IDs/parity frozen before later rejection."""
    _clock(ordinals, timestamps)
    _require(isinstance(frames_bgr, (list, tuple, np.ndarray)) and len(frames_bgr) == 7
             and isinstance(masks_bool, (list, tuple, np.ndarray)) and len(masks_bool) == 7,
             'exactly seven sanitized frames/masks required')
    shape = frames_bgr[0].shape if isinstance(frames_bgr[0], np.ndarray) else ()
    _require(len(shape) == 3 and shape[2] == 3 and min(shape[:2]) >= 32
             and shape[0]*shape[1] <= 8_500_000, 'bounded native BGR shape required')
    check()
    for frame, mask in zip(frames_bgr, masks_bool):
        _require(isinstance(frame, np.ndarray) and frame.dtype == np.uint8
                 and frame.shape == shape and frame.flags.c_contiguous
                 and isinstance(mask, np.ndarray) and mask.dtype == np.bool_
                 and mask.shape == shape[:2] and mask.flags.c_contiguous,
                 'native C BGR/mask dtype/shape differs')
        _require(not np.any(frame[mask]), 'masked pixels must already be black')
        check()
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    check()
    seed_support = cv2.erode((~masks_bool[0]).astype(np.uint8),
        np.ones((221, 221), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    seed = cv2.goodFeaturesToTrack(gray[0], mask=seed_support, maxCorners=800,
        qualityLevel=.01, minDistance=8., blockSize=7, useHarrisDetector=False)
    seed = np.empty((0, 2), np.float64) if seed is None else seed.reshape(-1, 2).astype(np.float64)
    _require(len(seed) <= 800 and np.isfinite(seed).all(), 'invalid GFTT result')
    seed = seed[np.lexsort((seed[:, 1], seed[:, 0]))]
    n = len(seed)
    ids = np.arange(n, dtype=np.int32)
    points = np.full((7, n, 2), np.nan)
    proposed = points.copy()
    back = np.full((6, n, 2), np.nan)
    valid, raw_valid = np.zeros((7, n), bool), np.zeros((7, n), bool)
    fs, bs = np.full((6, n), -1, np.int8), np.full((6, n), -1, np.int8)
    fb, error = np.full((6, n), np.nan), np.full((6, n), np.nan)
    reason = np.full((7, n), 'not_active', dtype='<U40')
    points[0] = proposed[0] = seed
    raw_valid[0] = True
    previous_integral = _integral(masks_bool[0])
    valid[0] = support_valid(previous_integral, seed, seed)
    reason[0] = np.where(valid[0], 'accepted', 'mask_or_pyramid_support')
    h, w = shape[:2]
    inside = lambda q: np.isfinite(q).all(axis=1) & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01), flags=0)
    for t in range(1, 7):
        check()
        current_integral = _integral(masks_bool[t])
        active = np.flatnonzero(valid[t-1])
        if not len(active):
            previous_integral = current_integral
            continue
        old = points[t-1, active]
        prev = old.astype(np.float32).reshape(-1, 1, 2)
        forward, status, patch = cv2.calcOpticalFlowPyrLK(gray[t-1], gray[t], prev, None, **lk)
        reason[t, active] = 'forward_failure'
        if forward is None or status is None or forward.shape != prev.shape or status.size != len(active):
            previous_integral = current_integral
            continue
        q = forward.reshape(-1, 2).astype(np.float64)
        proposed[t, active] = q
        fs[t-1, active] = status.reshape(-1).astype(np.int8)
        if patch is not None and patch.size == len(active):
            error[t-1, active] = patch.reshape(-1)
        good = status.reshape(-1).astype(bool) & inside(q)
        raw_valid[t, active] = good
        reason[t, active[good]] = 'mask_or_pyramid_support'
        good &= support_valid(previous_integral, old, q) & support_valid(current_integral, old, q)
        choose = active[good]
        if len(choose):
            backward, bstatus, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t-1], forward[good], None, **lk)
            reason[t, choose] = 'backward_failure'
            if backward is not None and bstatus is not None and backward.shape == forward[good].shape and bstatus.size == len(choose):
                bq = backward.reshape(-1, 2).astype(np.float64)
                back[t-1, choose] = bq
                bs[t-1, choose] = bstatus.reshape(-1).astype(np.int8)
                distance = np.linalg.norm(bq-points[t-1, choose], axis=1)
                fb[t-1, choose] = distance
                supported = support_valid(previous_integral, q[good], bq) & support_valid(current_integral, q[good], bq)
                accepted = bstatus.reshape(-1).astype(bool) & inside(bq) & supported & (distance <= .75)
                reason[t, choose] = 'forward_backward_or_support'
                valid[t, choose[accepted]] = True
                points[t, choose[accepted]] = q[good][accepted]
                reason[t, choose[accepted]] = 'accepted'
        previous_integral = current_integral
    return {'status': 'tracked' if n else 'unavailable_no_seeds', 'seed_ids': ids,
            'fit_seed': ids % 2 == 0, 'check_seed': ids % 2 == 1,
            'points_px': points, 'proposed_points_px': proposed, 'backward_points_px': back,
            'valid': valid, 'raw_forward_valid': raw_valid, 'forward_status': fs, 'backward_status': bs,
            'forward_backward_error_px': fb, 'forward_patch_error': error, 'rejection_reason': reason,
            'ordinals': ordinals.copy(), 'timestamps': timestamps.copy(), 'native_shape': shape,
            'persistent_count': int(valid.all(axis=0).sum()), 'all_original_seed_ids_retained': True,
            'split_after_support_masked_extraction_before_later_validity': True,
            'profile_sha256': PROFILE_SHA256, 'hidden_lk_iteration_paths_certified': False,
            'LK_error_is_covariance': False}


def _track_contract(tracks, transform):
    _require(type(transform) is PixelTransform and transform == make_transform(
        transform.native_hw, transform.processed_hw), 'canonical public preprocessing transform required')
    _require(isinstance(tracks, dict), 'track dictionary required')
    _clock(tracks['ordinals'], tracks['timestamps'])
    p, valid, ids = tracks['points_px'], tracks['valid'], tracks['seed_ids']
    _require(isinstance(p, np.ndarray) and p.dtype == np.float64 and p.ndim == 3
             and p.shape[0] == 7 and p.shape[2] == 2 and p.shape[1] <= 800
             and isinstance(valid, np.ndarray) and valid.dtype == np.bool_ and valid.shape == p.shape[:2]
             and isinstance(ids, np.ndarray) and ids.dtype == np.int32 and ids.shape == (p.shape[1],)
             and np.array_equal(ids, np.arange(len(ids), dtype=np.int32))
             and np.array_equal(tracks['fit_seed'], ids % 2 == 0)
             and np.array_equal(tracks['check_seed'], ids % 2 == 1)
             and tracks['profile_sha256'] == PROFILE_SHA256, 'full bounded original ID/parity tracks required')
    h, w = transform.native_hw
    _require(tuple(tracks['native_shape']) == (h, w, 3) and np.isfinite(p[valid]).all()
             and np.all(p[valid] >= 0) and np.all(p[valid][:, 0] < w)
             and np.all(p[valid][:, 1] < h), 'valid native observations differ')
    processed = np.full(p.shape, np.nan)
    processed[valid] = transform.native_to_processed(p[valid])
    pt, pl = transform.pad_top_left
    ch, cw = transform.crop_hw
    in_crop = valid & (processed[..., 0] >= pl+.5) & (processed[..., 0] <= pl+cw-.5)
    in_crop &= (processed[..., 1] >= pt+.5) & (processed[..., 1] <= pt+ch-.5)
    return p, processed, valid.all(axis=0), in_crop.all(axis=0), ids


def _cameras(c2w, intrinsics):
    for a, shape in ((c2w, (7, 3, 4)), (intrinsics, (7, 3, 3))):
        _require(isinstance(a, np.ndarray) and a.dtype in (np.dtype('float32'), np.dtype('float64'))
                 and a.shape == shape and np.isfinite(a).all(), 'finite bounded camera arrays required')
    C = c2w.astype(np.float64)
    K = intrinsics.astype(np.float64)
    R, centres = C[:, :, :3], C[:, :, 3]
    _require(np.allclose(np.swapaxes(R, 1, 2) @ R, np.eye(3), rtol=0, atol=1e-4)
             and np.all(np.abs(np.linalg.det(R)-1) <= 1e-4), 'proper rotations required; no silent projection')
    _require(np.all(K[:, (0, 1), (0, 1)] > 0) and np.all(K[:, 0, 1] == 0)
             and np.all(K[:, 1, 0] == 0) and np.array_equal(K[:, 2], np.tile([0., 0., 1.], (7, 1))),
             'positive zero-skew pinhole intrinsics required')
    return R, centres, K


def _project(points, Rwc, twc, K, transform):
    camera = points @ Rwc.T + twc
    finite_camera = np.isfinite(camera).all(axis=1)
    good = finite_camera & (camera[:, 2] > 1e-12)
    reason = np.full(len(points), 'nonfinite_point_or_camera', dtype='<U48')
    reason[finite_camera] = 'nonpositive_camera_depth'
    reason[good] = 'nonfinite_projection'
    predicted = np.full((len(points), 2), np.nan)
    ids = np.flatnonzero(good)
    if len(ids):
        homogeneous = camera[ids] @ K.T
        centres = homogeneous[:, :2]/homogeneous[:, 2:]
        finite = np.isfinite(centres).all(axis=1)
        ids, centres = ids[finite], centres[finite]
        good[:] = False
        predicted[ids] = transform.processed_to_native(centres)
        reason[ids] = 'outside_real_processed_crop'
        pt, pl = transform.pad_top_left
        ch, cw = transform.crop_hw
        good[ids] = ((centres[:, 0] >= pl+.5) & (centres[:, 0] <= pl+cw-.5)
                     & (centres[:, 1] >= pt+.5) & (centres[:, 1] <= pt+ch-.5))
        reason[good] = 'accepted'
    return predicted, good, camera[:, 2].copy(), reason


def _empty_result(n, reason):
    return {'status': 'unavailable', 'reason': reason,
            'predicted_native_px': np.full((7, n, 2), np.nan),
            'errors_native_px': np.full((7, n), np.nan),
            'residual_native_px': np.full((7, n, 2), np.nan),
            'errors_with_invalid_inf': np.full((7, n), np.inf),
            'camera_depth_validation_gauge': np.full((7, n), np.nan),
            'projection_valid': np.zeros((7, n), bool),
            'rejection_reason': np.full((7, n), reason, dtype='<U48'),
            'check_coverage_by_view': np.zeros(7),
            'later_view_metrics': [], 'passes_candidate_defaults': False,
            'passes_proposed_robust_full_check': False}


def _quantile_with_inf(values, fraction):
    if not len(values):
        return None
    ordered = np.sort(values)
    position = (len(ordered)-1)*fraction
    lo, hi = int(np.floor(position)), int(np.ceil(position))
    if lo == hi:
        return float(ordered[lo])
    if np.isinf(ordered[hi]):
        return float('inf')
    return float(ordered[lo]+(position-lo)*(ordered[hi]-ordered[lo]))


def _evaluate_points(xyz, eligible, native, ids, Rwc, twc, K, transform, check):
    n = len(ids)
    result = _empty_result(n, 'track_or_crop_unavailable')
    checker = ids % 2 == 1
    eligible_ids = ids[checker & eligible]
    result['eligible_check_seed_ids'] = eligible_ids.copy()
    for t in range(7):
        check()
        predicted, good, depth, why = _project(xyz, Rwc[t], twc[t], K[t], transform)
        good &= eligible
        result['predicted_native_px'][t] = predicted
        result['projection_valid'][t] = good
        result['camera_depth_validation_gauge'][t] = depth
        measured = np.isfinite(predicted).all(axis=1) & np.isfinite(native[t]).all(axis=1)
        result['residual_native_px'][t, measured] = predicted[measured]-native[t, measured]
        result['errors_native_px'][t, measured] = np.linalg.norm(result['residual_native_px'][t, measured], axis=1)
        result['errors_with_invalid_inf'][t, good] = result['errors_native_px'][t, good]
        result['check_coverage_by_view'][t] = float(good[checker].mean()) if checker.any() else 0.
        result['rejection_reason'][t, eligible] = why[eligible]
        result['rejection_reason'][t, good] = 'accepted'
        if t not in (1, 2, 4, 5, 6):
            continue
        all_good = len(eligible_ids) > 0 and bool(good[eligible_ids].all())
        errors = result['errors_native_px'][t, eligible_ids]
        row = {'frame': t, 'original_check_denominator': int(checker.sum()),
               'eligible_check_denominator': len(eligible_ids),
               'excluded_check_seed_ids': ids[checker & ~eligible].tolist(),
               'invalid_check_seed_ids': ids[checker & ~good].tolist(),
               'invalid_eligible_check_seed_ids': eligible_ids[~good[eligible_ids]].tolist(),
               'median_error_px': float(np.median(errors)) if all_good else None,
               'p90_error_px': float(np.percentile(errors, 90)) if all_good else None,
               'rmse_px': float(np.sqrt(np.mean(errors**2))) if all_good else None}
        row['passes_candidate_defaults'] = bool(all_good and len(eligible_ids) >= 20
            and row['median_error_px'] <= 2. and row['p90_error_px'] <= 4. and row['rmse_px'] <= 4.)
        full = result['errors_with_invalid_inf'][t, checker]
        row['full_check_median_including_invalid'] = _quantile_with_inf(full, .5)
        row['full_check_p90_including_invalid'] = _quantile_with_inf(full, .9)
        row['full_check_coverage'] = result['check_coverage_by_view'][t]
        row['passes_proposed_robust_full_check'] = bool(good[checker].sum() >= 20
            and row['full_check_coverage'] >= .9
            and row['full_check_median_including_invalid'] <= 2.
            and row['full_check_p90_including_invalid'] <= 4.)
        result['later_view_metrics'].append(row)
    passed = all(row['passes_candidate_defaults'] for row in result['later_view_metrics'])
    result.update(status='candidate_on_frozen_track_support' if passed else 'unavailable',
                  reason=None if passed else 'independent later-view support or projection failed',
                  passes_candidate_defaults=passed)
    result['passes_proposed_robust_full_check'] = all(
        row['passes_proposed_robust_full_check'] for row in result['later_view_metrics'])
    result['robust_criterion_is_prospective_not_admitted'] = True
    return result


def _rotation_alternative(pixels, native, eligible, ids, K, transform, check):
    fit = ids[eligible & (ids % 2 == 0)]
    held = ids[eligible & (ids % 2 == 1)]
    n = len(ids)
    if len(fit) < 20 or len(held) < 20:
        r = _empty_result(n, 'insufficient FIT/CHECK for rotation-only alternative')
        r.update(available=False, explains_with_existing_rotation_floor=False)
        return r
    basis = np.full((n, 3), np.nan)
    chosen = ids[eligible]
    basis[chosen] = np.column_stack([pixels[0, chosen], np.ones(len(chosen))]) @ np.linalg.inv(K[0]).T
    basis[chosen] /= np.linalg.norm(basis[chosen], axis=1, keepdims=True)
    rotations = np.repeat(np.eye(3)[None], 7, axis=0)
    for t in range(1, 7):
        check()
        later = np.column_stack([pixels[t, fit], np.ones(len(fit))]) @ np.linalg.inv(K[t]).T
        later /= np.linalg.norm(later, axis=1, keepdims=True)
        U, _, Vt = np.linalg.svd(later.T @ basis[fit])
        rotations[t] = U @ np.diag([1., 1., np.linalg.det(U @ Vt)]) @ Vt
    r = _evaluate_points(basis, eligible, native, ids, rotations, np.zeros((7, 3)), K, transform, check)
    rows = []
    for t in range(1, 7):
        good = bool(r['projection_valid'][t, held].all())
        errors = r['errors_native_px'][t, held]
        rows.append({'frame': t, 'eligible_check_denominator': len(held),
                     'median_error_px': float(np.median(errors)) if good else None,
                     'p90_error_px': float(np.percentile(errors, 90)) if good else None,
                     'explains': bool(good and np.median(errors) <= 1. and np.percentile(errors, 90) <= 2.)})
    r.update(available=True, fit_seed_ids=fit.copy(), camera0_to_camera_rotations=rotations,
             rotation_floor_metrics=rows,
             explains_with_existing_rotation_floor=all(row['explains'] for row in rows),
             scope='independent image-based FIT rotation alternative; CHECK never fitted; predicted intrinsics assumed')
    return r


def validate_geometry(tracks, c2w, intrinsics, transform, *, world_points=None, check=lambda: None):
    """Compare fixed model poses/points with observations; never fit CHECK.

    Optional official WORLD_POINTS is [7,H,W,3]. Its same-view agreement is
    internal by construction; only observed later-view correspondences test it.
    All report arrays keep the entire extracted seed dimension.
    """
    native, pixels, persistent, in_crop, ids = _track_contract(tracks, transform)
    R, centres, K = _cameras(c2w, intrinsics)
    n = len(ids)
    ph, pw = transform.processed_hw
    if world_points is not None:
        _require(isinstance(world_points, np.ndarray)
                 and world_points.dtype in (np.dtype('float32'), np.dtype('float64'))
                 and world_points.shape == (7, ph, pw, 3) and world_points.flags.c_contiguous,
                 'bounded C official world-point grid required')
    eligible = persistent & in_crop
    checker = ids % 2 == 1
    summary = {'status': 'local_geometry_diagnostic_only', 'profile_sha256': PROFILE_SHA256,
               'original_seed_count': n, 'original_check_count': int(checker.sum()),
               'persistent_count': int(persistent.sum()), 'eligible_check_count': int((eligible & checker).sum()),
               'eligible_seed_ids': ids[eligible].copy(), 'excluded_seed_ids': ids[~eligible].copy(),
               'eligibility_frozen_before_camera_geometry': True, 'first_gauge_views': [0, 3],
               'later_check_views': [1, 2, 4, 5, 6], 'transform': transform.receipt(),
               'model_images_held_out': False, 'CHECK_pose_or_scale_fitted': False,
               'all_original_seed_rows_retained': True, 'matrix_cycle_is_independent_evidence': False,
               'metric_scale_recovered': False, 'calibrated_covariance': False, 'measured_stop': False,
               'global_or_cross_window_gauge': False, 'policy_admitted': False}
    result = {'summary': summary, 'pose_only': _empty_result(n, 'insufficient persistent CHECK support'),
              'model_depth_pose': _empty_result(n, 'model points not supplied' if world_points is None
                                               else 'insufficient persistent CHECK support')}
    check()
    result['rotation_only'] = _rotation_alternative(pixels, native, eligible, ids, K, transform, check)
    summary['independent_rotation_alternative_available'] = result['rotation_only']['available']
    summary['rotation_only_explains_independent_tracks'] = result['rotation_only']['explains_with_existing_rotation_floor']
    baseline = float(math.hypot(*(centres[3]-centres[0])))
    summary['model_anchor_baseline_arbitrary_units'] = baseline
    if not np.isfinite(baseline):
        result['pose_only'] = _empty_result(n, 'nonfinite anchor baseline; translation unknown')
        if world_points is not None:
            result['model_depth_pose'] = _empty_result(n, 'nonfinite anchor baseline; translation unknown')
        return result
    # Normalize arbitrary translation gauge for conditioning, not metric scale.
    gauge = baseline if baseline > 0 else 1.
    normalized_centres = (centres-centres[0])/gauge
    summary['validation_gauge'] = ('unit fixed0/3 anchor' if baseline > 0
                                    else 'original model units; no anchor scale')
    Rwc = np.linalg.inv(R)
    twc = -np.einsum('nij,nj->ni', Rwc, normalized_centres)
    held = ids[eligible & checker]
    world_rays = []
    for t in (0, 3):
        rays = np.column_stack([pixels[t, held], np.ones(len(held))]) @ np.linalg.inv(K[t]).T
        rays = rays @ R[t].T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        world_rays.append(rays)
    angles = np.arctan2(np.linalg.norm(np.cross(*world_rays), axis=1),
                        np.sum(world_rays[0]*world_rays[1], axis=1))
    rh, rw = transform.resized_hw
    h, w = transform.native_hw
    native_focals = K[[0, 3]][:, (0, 1), (0, 1)]*np.array([w/rw, h/rh])
    parallax_limit = 3./float(native_focals.min())
    median_angle = float(np.median(angles)) if len(angles) else None
    anchor_good = bool(baseline > 0 and len(angles) and np.isfinite(angles).all()
                       and median_angle > parallax_limit
                       and not result['rotation_only']['explains_with_existing_rotation_floor'])
    summary.update(anchor_median_derotated_parallax_rad=median_angle,
                   anchor_parallax_limit_rad=parallax_limit,
                   observable_anchor_for_candidate_defaults=anchor_good,
                   normalized_centres=normalized_centres.copy())
    chosen = ids[eligible]
    P = K @ np.concatenate([Rwc, twc[:, :, None]], axis=2)
    xyz = np.full((n, 3), np.nan)
    if len(chosen) and baseline > 0:
        homogeneous = cv2.triangulatePoints(P[0], P[3], pixels[0, chosen].T, pixels[3, chosen].T)
        finite = np.isfinite(homogeneous).all(axis=0) & (np.abs(homogeneous[3]) > 1e-12)
        xyz[chosen[finite]] = (homogeneous[:3, finite]/homogeneous[3, finite]).T
    for t in (0, 3):
        camera = xyz @ Rwc[t].T + twc[t]
        bad = ~np.isfinite(camera).all(axis=1) | (camera[:, 2] <= 1e-12)
        xyz[bad] = np.nan
    result['pose_only'] = _evaluate_points(xyz, eligible, native, ids, Rwc, twc, K, transform, check)
    result['pose_only']['triangulated_points_unit_anchor_gauge'] = xyz
    result['pose_only']['observation_scope'] = 'CHECK0/3 condition landmarks; observed1/2/4/5/6 test fixed poses'
    if world_points is not None:
        result = _model_point_report(result, world_points, pixels, chosen, centres[0], gauge,
                                    eligible, native, ids, Rwc, twc, K, transform, check)
    # Keep continuous residuals even where translation is unobservable; the
    # candidate decision still refuses the same weak/zero-parallax cases.
    if not anchor_good:
        for name in ('pose_only', 'model_depth_pose'):
            if name == 'model_depth_pose' and world_points is None:
                continue
            result[name].update(status='unavailable', reason='zero or weak anchor parallax; translation unknown',
                                passes_candidate_defaults=False, passes_proposed_robust_full_check=False)
            if summary['rotation_only_explains_independent_tracks']:
                result[name]['reason'] = 'rotation-only explains independent tracks; translation unknown'
    return result


def _model_point_report(result, world_points, pixels, chosen, origin, baseline,
                        eligible, native, ids, Rwc, twc, K, transform, check):
    ph, pw = transform.processed_hw
    model_xyz = np.full((len(ids), 3), np.nan)
    # Array index (x,y) represents the DVLT center (x+.5,y+.5).
    indices = pixels[0, chosen]-.5
    inside = ((indices >= 0).all(axis=1) & (indices[:, 0] < pw-1) & (indices[:, 1] < ph-1))
    sample_ids, q = chosen[inside], indices[inside]
    if len(q):
        lo = np.floor(q).astype(np.int64)
        dx, dy = (q-lo).T
        x, y = lo.T
        values = (world_points[0, y, x]*(1-dx)[:, None]*(1-dy)[:, None]
                  + world_points[0, y, x+1]*dx[:, None]*(1-dy)[:, None]
                  + world_points[0, y+1, x]*(1-dx)[:, None]*dy[:, None]
                  + world_points[0, y+1, x+1]*dx[:, None]*dy[:, None])
        model_xyz[sample_ids] = (values-origin)/baseline
    result['model_depth_pose'] = _evaluate_points(model_xyz, eligible, native, ids, Rwc, twc, K, transform, check)
    result['model_depth_pose'].update(
        model_points_source='official WORLD_POINTS; normally derived from own depth/cameras',
        same_view_consistency_is_independent_evidence=False,
        later_view_observed_pixels_are_the_test=True, confidence_selection_used=False,
        anchor_model_point_sample_valid=np.isfinite(model_xyz).all(axis=1))
    return result
