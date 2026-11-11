"""Private seven-frame geometric frontend; no file/video/model/GT IO.
K is supplied, not calibrated here. solve_window estimates all poses from tracks.
Research candidates are not stops, covariance, metric speed or geographic routes.
"""
import cv2
import numpy as np

PROFILE = {'frames': 7, 'max_features': 800, 'max_frame_pixels': 8_500_000,
    'gftt_quality': .01, 'gftt_min_distance_px': 8., 'lk_window': (21, 21),
    'lk_levels': 3, 'lk_fb_px': .75, 'anchor': (0, 3),
    'split': 'stable original seed ID parity before rejection',
    'sampling': 'seven consecutive ORIGINAL VO READ rows; external ledger proves rows, supplied video ordinals may be stride-separated',
    'minimum_fit': 20, 'minimum_check': 20, 'assumed_noise_floor_px': 1.,
    'parallax_noise_multiple': 3., 'opencv_seed': 0,
    'E_threshold_px': 1., 'PnP_threshold_px': 2.,
    'E_refinement': 'normalized-eight-point on FIT consensus; equal-singular-value projection',
    'recoverPose_distance_in_unit_baselines': 10000.,
    'supplied_K_is_verified_calibration': False,
    'opencv_global_rng_reset_requires_dedicated_process': True}


def clock_and_K(K, ordinals, timestamps):
    if (not isinstance(K, np.ndarray) or K.dtype != np.float64 or K.shape != (3, 3)
            or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.array_equal(K[2], [0., 0., 1.]) or K[0, 1] != 0 or K[1, 0] != 0
            or not isinstance(ordinals, np.ndarray) or ordinals.dtype != np.int64
            or ordinals.shape != (7,) or np.any(ordinals < 0) or not np.all(np.diff(ordinals) > 0)
            or not isinstance(timestamps, np.ndarray) or timestamps.dtype != np.float64
            or timestamps.shape != (7,) or not np.isfinite(timestamps).all()
            or not np.all(np.diff(timestamps) > 0)):
        raise ValueError('exact increasing native ordinals/PTS and full F64 K required; caller proves READ-row lineage')


def track_window(frames, K, ordinals, timestamps, *, check=lambda: None):
    """GFTT once, persistent LK forward/back; preserve every seed and failure."""
    clock_and_K(K, ordinals, timestamps)
    if not isinstance(frames, (list, tuple)) or len(frames) != 7:
        raise ValueError('exactly seven native arrays required')
    shape = frames[0].shape if isinstance(frames[0], np.ndarray) else None
    if (shape is None or len(shape) not in (2, 3) or shape[0] < 32 or shape[1] < 32
            or (len(shape) == 3 and shape[2] != 3)
            or shape[0]*shape[1] > PROFILE['max_frame_pixels']
            or any(not isinstance(f, np.ndarray) or f.dtype != np.uint8
                   or f.shape != shape or not f.flags.c_contiguous for f in frames)):
        raise ValueError('same native C-uint8 gray/BGR arrays within pixel cap required')
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    check()
    seeds = cv2.goodFeaturesToTrack(gray[0], maxCorners=PROFILE['max_features'],
        qualityLevel=PROFILE['gftt_quality'], minDistance=PROFILE['gftt_min_distance_px'],
        blockSize=7, useHarrisDetector=False)
    seeds = np.empty((0, 2), np.float32) if seeds is None else seeds.reshape(-1, 2)
    seeds = seeds[np.lexsort((seeds[:, 1], seeds[:, 0]))]
    n = len(seeds)
    ids = np.arange(n, dtype=np.int32)
    points, valid = np.full((7, n, 2), np.nan), np.zeros((7, n), bool)
    fb, patch_error = np.full((6, n), np.nan), np.full((6, n), np.nan)
    reason = np.full((6, n), 'not_active', dtype='<U32')
    points[0], valid[0] = seeds, True
    lk = dict(winSize=PROFILE['lk_window'], maxLevel=PROFILE['lk_levels'],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01), flags=0)
    h, w = shape[:2]
    inside = lambda q: np.isfinite(q).all(axis=1) & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    for t in range(1, 7):
        check()
        active = np.flatnonzero(valid[t-1])
        if not len(active):
            break
        prev = points[t-1, active].astype(np.float32).reshape(-1, 1, 2)
        forward, status, error = cv2.calcOpticalFlowPyrLK(gray[t-1], gray[t], prev, None, **lk)
        reason[t-1, active] = 'forward_failure'
        if forward is None or status is None or forward.shape != prev.shape or status.size != len(active):
            continue
        proposed = forward.reshape(-1, 2)
        good = status.reshape(-1).astype(bool) & inside(proposed)
        if error is not None and error.size == len(active):
            patch_error[t-1, active[good]] = error.reshape(-1)[good]
        selected = active[good]
        if not len(selected):
            continue
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t-1], forward[good], None, **lk)
        reason[t-1, selected] = 'backward_failure'
        if backward is None or back_status is None or backward.shape != forward[good].shape or back_status.size != len(selected):
            continue
        back = backward.reshape(-1, 2)
        distance = np.linalg.norm(back-points[t-1, selected], axis=1)
        fb[t-1, selected] = distance
        back_good = back_status.reshape(-1).astype(bool) & inside(back)
        reason[t-1, selected[back_good]] = 'forward_backward_error'
        accepted = back_good & (distance <= PROFILE['lk_fb_px'])
        points[t, selected[accepted]] = proposed[good][accepted]
        valid[t, selected[accepted]] = True
        reason[t-1, selected[accepted]] = 'accepted'
    return {'status': 'tracked' if n else 'unavailable_no_seeds', 'seed_ids': ids,
        'fit_seed': ids % 2 == 0, 'check_seed': ids % 2 == 1,
        'points_px': points, 'valid': valid, 'forward_backward_error_px': fb,
        'forward_patch_error': patch_error, 'rejection_reason': reason,
        'ordinals': ordinals.copy(), 'timestamps': timestamps.copy(), 'K': K.copy(),
        'native_shape': shape, 'resized': False, 'persistent_count': int(valid.all(axis=0).sum()),
        'failed_seed_ids_retained': True, 'LK_error_is_covariance': False}


def bearing(points, K):
    rays = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(K).T
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def rotate_only(first, later):
    U, _, Vt = np.linalg.svd(later.T @ first)
    return U @ np.diag([1., 1., np.linalg.det(U @ Vt)]) @ Vt


def rotation_degrees(R):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(R)-1.)/2., -1., 1.))))


def essential(first, later, K):
    E, consensus = cv2.findEssentialMat(first, later, cameraMatrix=K,
        method=cv2.RANSAC, prob=.999, threshold=PROFILE['E_threshold_px'], maxIters=2000)
    if E is None or E.shape != (3, 3) or not np.isfinite(E).all() or consensus is None:
        return None
    if consensus.shape != (len(first), 1):
        return None
    initial_mask = consensus.copy()
    keep = consensus.reshape(-1).astype(bool)
    if keep.sum() < 8:
        return None
    first_normalized = cv2.undistortPoints(first.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    later_normalized = cv2.undistortPoints(later.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    refined, _ = cv2.findFundamentalMat(first_normalized[keep], later_normalized[keep], cv2.FM_8POINT)
    if refined is None or refined.shape != (3, 3) or not np.isfinite(refined).all():
        return None
    U, sv, Vt = np.linalg.svd(refined)
    s = (sv[0]+sv[1])/2.
    E = U @ np.diag([s, s, 0.]) @ Vt
    count, R, t, positive, _ = cv2.recoverPose(E, first, later, cameraMatrix=K,
        distanceThresh=PROFILE['recoverPose_distance_in_unit_baselines'], mask=consensus.copy())
    if not np.isfinite(R).all() or not np.isfinite(t).all() or positive is None:
        return None
    return {'R': R, 't': t.reshape(3), 'consensus': initial_mask.reshape(-1).astype(bool),
            'positive': positive.reshape(-1).astype(bool), 'count': int(count)}


def triangulate(points0, points3, R3, t3, K):
    P0 = K @ np.column_stack([np.eye(3), np.zeros(3)])
    P3 = K @ np.column_stack([R3, t3])
    h = cv2.triangulatePoints(P0, P3, points0.T, points3.T)
    good = np.isfinite(h).all(axis=0) & (np.abs(h[3]) > 1e-12)
    points = np.full((len(points0), 3), np.nan)
    points[good] = (h[:3, good]/h[3, good]).T
    return points


def reprojection(points, R, t, K):
    camera = points @ R.T + t
    good = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-12)
    image = np.full((len(points), 2), np.nan)
    h = camera[good] @ K.T
    image[good] = h[:, :2]/h[:, 2:]
    return image, good


def _solve_window(tracks, K, *, check=lambda: None):
    """Estimate poses from FIT pixels; CHECK later views never enter fitting.

    This resets OpenCV's global RNG for a reproducible dedicated-process probe.
    Translation gauge is fixed by the predeclared views0/3, never a best pair.
    """
    clock_and_K(K, tracks['ordinals'], tracks['timestamps'])
    pixels, valid, ids = tracks['points_px'], tracks['valid'], tracks['seed_ids']
    if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.float64
            or pixels.ndim != 3 or pixels.shape[0] != 7 or pixels.shape[2] != 2
            or not 0 <= pixels.shape[1] <= PROFILE['max_features']
            or not isinstance(valid, np.ndarray) or valid.dtype != np.bool_
            or valid.shape != pixels.shape[:2] or not np.isfinite(pixels[valid]).all()
            or not isinstance(ids, np.ndarray) or ids.dtype != np.int32
            or not np.array_equal(ids, np.arange(pixels.shape[1], dtype=np.int32))
            or not np.array_equal(tracks['fit_seed'], ids % 2 == 0)
            or not np.array_equal(tracks['check_seed'], ids % 2 == 1)
            or not np.array_equal(tracks['K'], K)):
        raise ValueError('bounded native tracks with original fixed ID split required')
    h, w = tracks['native_shape'][:2]
    if (type(h) is not int or type(w) is not int or min(h, w) < 32
            or h*w > PROFILE['max_frame_pixels'] or np.any(pixels[valid] < 0)
            or np.any(pixels[valid][:, 0] >= w) or np.any(pixels[valid][:, 1] >= h)):
        raise ValueError('valid tracks must remain in the declared full native raster')
    persistent = valid.all(axis=0)
    fit = np.flatnonzero(persistent & (ids % 2 == 0))
    held = np.flatnonzero(persistent & (ids % 2 == 1))
    base = {'metric_scale_recovered': False, 'measured_stop': False,
        'calibrated_covariance': False, 'static_world_verified': False,
        'split_before_geometry_filtering': True, 'first_gauge_views': [0, 3],
        'total_seed_count': len(ids), 'persistent_fit_count': len(fit),
        'persistent_check_count': len(held), 'discarded_seed_ids': ids[~persistent].tolist(),
        'all_fit_seed_ids': ids[ids % 2 == 0].tolist(), 'all_check_seed_ids': ids[ids % 2 == 1].tolist(),
        'translation_status': 'unknown', 'centres': None, 'rotations': None}
    def unavailable(reason):
        return {**base, 'status': 'unavailable', 'reason': reason}
    if len(fit) < 20 or len(held) < 20:
        return unavailable('insufficient persistent fit/check support; no replacement seeds')
    cv2.setRNGSeed(PROFILE['opencv_seed'])
    check()
    # Work in original ID indices when reporting denominators or failures.
    rotation_only, rotation_records = [np.eye(3)], []
    for i in range(1, 7):
        R = rotate_only(bearing(pixels[0, fit], K), bearing(pixels[i, fit], K))
        rotation_only.append(R)
        predicted, good = reprojection(bearing(pixels[0, held], K), R, np.zeros(3), K)
        errors = np.linalg.norm(predicted-pixels[i, held], axis=1)
        rotation_records.append({'frame': i, 'check_denominator': len(held),
            'invalid_check_count': int((~good).sum()),
            'median_error_px': float(np.median(errors)) if good.all() else None,
            'p90_error_px': float(np.percentile(errors, 90)) if good.all() else None})
    base['rotation_only_check'] = rotation_records
    R01 = rotation_only[1]
    R13 = rotate_only(bearing(pixels[1, fit], K), bearing(pixels[3, fit], K))
    base['rotation_only_independent_cycle_degrees'] = rotation_degrees(R13 @ R01 @ rotation_only[3].T)
    noise = PROFILE['assumed_noise_floor_px']
    if all(r['invalid_check_count'] == 0 and r['median_error_px'] <= noise
           and r['p90_error_px'] <= 2*noise for r in rotation_records):
        base['rotation_only_rotations'] = np.asarray(rotation_only)
        return unavailable('rotation-only explains pixels within declared floor; translation ambiguous, not stop')
    anchor = essential(pixels[0, fit], pixels[3, fit], K)
    if anchor is None:
        return unavailable('essential anchor absent or multiple; no forced model')
    base['essential_consensus_fit_ids'] = ids[fit][anchor['consensus']].tolist()
    base['essential_positive_fit_ids'] = ids[fit][anchor['positive']].tolist()
    if anchor['count'] < 20:
        return unavailable('insufficient anchor cheirality')
    R3, t3 = anchor['R'], anchor['t']
    world0 = bearing(pixels[0, held], K)
    world3 = bearing(pixels[3, held], K) @ R3
    angles = np.arctan2(np.linalg.norm(np.cross(world0, world3), axis=1), np.sum(world0*world3, axis=1))
    base['anchor_check_median_derotated_parallax_rad'] = float(np.median(angles))
    threshold = PROFILE['parallax_noise_multiple'] * noise / min(K[0, 0], K[1, 1])
    if np.median(angles) <= threshold:
        return unavailable('weak anchor parallax relative to declared floor')
    landmarks = triangulate(pixels[0, persistent], pixels[3, persistent], R3, t3, K)
    local_ids = ids[persistent]
    fit_local = np.flatnonzero(local_ids % 2 == 0)
    held_local = np.flatnonzero(local_ids % 2 == 1)
    anchor_camera3 = landmarks @ R3.T + t3
    usable = (np.isfinite(landmarks).all(axis=1) & (landmarks[:, 2] > 0)
              & np.isfinite(anchor_camera3).all(axis=1) & (anchor_camera3[:, 2] > 0))
    training = fit_local[usable[fit_local] & anchor['positive']]
    base['anchor_invalid_check_ids'] = local_ids[held_local[~usable[held_local]]].tolist()
    base['anchor_filtered_fit_ids'] = local_ids[fit_local[~usable[fit_local] | ~anchor['positive']]].tolist()
    # Preserve the entire CHECK denominator. Do not select a nice subset of it.
    if len(training) < 20 or not usable[held_local].all():
        return unavailable('triangulation failed fit support or any held-out cheirality')
    R_all = np.repeat(np.eye(3)[None], 7, axis=0)
    t_all = np.zeros((7, 3))
    R_all[3], t_all[3] = R3, t3
    pose_records = []
    for i in (1, 2, 4, 5, 6):
        check()
        target = pixels[i, local_ids[training]]
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(landmarks[training], target, K, None,
            iterationsCount=1000, reprojectionError=PROFILE['PnP_threshold_px'],
            confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < 20:
            return unavailable('PnP failed independent fit support at frame'+str(i))
        chosen = inliers.reshape(-1)
        ok, rvec, tvec = cv2.solvePnP(landmarks[training][chosen], target[chosen], K, None,
            rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return unavailable('PnP refinement unavailable')
        R_all[i], t_all[i] = cv2.Rodrigues(rvec)[0], tvec.reshape(3)
        predicted, good = reprojection(landmarks[held_local], R_all[i], t_all[i], K)
        errors = np.linalg.norm(predicted-pixels[i, held], axis=1)
        row = {'frame': i, 'check_denominator': len(held),
            'invalid_check_ids': ids[held[~good]].tolist(),
            'fit_inlier_ids': local_ids[training][chosen].tolist(),
            'median_error_px': float(np.median(errors)) if good.all() else None,
            'p90_error_px': float(np.percentile(errors, 90)) if good.all() else None,
            'rmse_px': float(np.sqrt(np.mean(errors**2))) if good.all() else None}
        pose_records.append(row)
        base['later_view_check_reprojection'] = pose_records
        if (not good.all() or row['median_error_px'] > 2*noise
                or row['p90_error_px'] > 4*noise or row['rmse_px'] > 4*noise):
            return unavailable('held-out later-view geometry failed; full denominator retained')
    independent = essential(pixels[1, fit], pixels[6, fit], K)
    if independent is None or independent['count'] < 20:
        return unavailable('independent pose/cycle check unavailable')
    predicted_R16 = R_all[6] @ R_all[1].T
    base['independent_rotation_cycle_degrees'] = rotation_degrees(independent['R'] @ predicted_R16.T)
    direction = t_all[6] - predicted_R16 @ t_all[1]
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        return unavailable('independent cycle translation direction unobservable')
    base['independent_translation_direction_degrees'] = float(np.rad2deg(np.arccos(
        np.clip(np.dot(direction/length, independent['t']), -1., 1.))))
    if (base['independent_rotation_cycle_degrees'] > 3.
            or base['independent_translation_direction_degrees'] > 15.):
        return unavailable('independent pose-cycle disagreement')
    centres = -np.einsum('nji,nj->ni', R_all, t_all)
    if not np.isfinite(centres).all():
        return unavailable('nonfinite shared-gauge centres')
    base.update(status='complete_relative_gauge_candidate', translation_status='candidate_only',
        centres=centres, rotations=R_all, adjacent_steps_gauge=np.linalg.norm(np.diff(centres, axis=0), axis=1),
        anchor_baseline_gauge=float(np.linalg.norm(centres[3])),
        approximate_poses_not_joint_bundle_adjustment=True,
        heldout_scope='views1,2,4,5,6 only; views0/3 condition CHECK landmarks',
        connected_gauge_scope='this one seven-frame window only; no cross-window stitching')
    return base


def solve_window(tracks, K, *, check=lambda: None):
    """Contract violations raise; valid-but-degenerate geometry is unavailable."""
    try:
        return _solve_window(tracks, K, check=check)
    except (cv2.error, np.linalg.LinAlgError) as error:
        return {'status': 'unavailable', 'reason': 'geometric solver failure: '+str(error),
            'translation_status': 'unknown', 'centres': None, 'rotations': None,
            'metric_scale_recovered': False, 'measured_stop': False,
            'calibrated_covariance': False, 'static_world_verified': False,
            'all_seed_ids_retained': tracks['seed_ids'].tolist()}
