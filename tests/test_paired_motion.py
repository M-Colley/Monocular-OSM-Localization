"""Synthetic geometry and legacy equivalence; no models, videos or labels."""
import numpy as np
import cv2
import pytest

from monocular_osm import paired_motion as p, visual_odometry as old


K = old.default_intrinsics(1280, 720)


def scene(count=120, static=0):
    rng = np.random.default_rng(53)
    world = np.column_stack([rng.uniform(-4, 4, count), rng.uniform(-2, 2, count), rng.uniform(8, 25, count)])
    angle = .035
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    t = np.array([.15, .02, -1.5])
    def project(value):
        uv = value@K.T
        return (uv[:, :2]/uv[:, 2:]).astype(np.float32)
    a, b = project(world), project(world@R.T+t)
    if static:
        stamps = rng.uniform([20, 20], [1200, 700], (static, 2)).astype(np.float32)
        a, b = np.vstack([a, stamps]), np.vstack([b, stamps])
    matches = np.column_stack([np.arange(len(a)), np.arange(len(a))]).astype(np.int32)
    distances = np.r_[np.arange(count)%17+1, np.zeros(static)].astype(np.float32)
    return a, b, matches, distances, R, t/np.linalg.norm(t)


def old_fixture(a, b, matches, distance):
    class Detector:
        def __init__(self): self.calls = 0
        def detectAndCompute(self, *_):
            points = (a, b)[self.calls]; self.calls += 1
            return [cv2.KeyPoint(float(x), float(y), 1.) for x, y in points], np.zeros((len(points), 32), np.uint8)
    class Matcher:
        def match(self, *_): return [cv2.DMatch(int(i), int(j), float(d)) for (i, j), d in zip(matches, distance)]
    return Detector(), Matcher()


def test_real_synthetic_SE3_matches_legacy_pose_and_known_direction():
    a, b, matches, distance, R, t = scene()
    result = p.estimate_pair(a, b, matches, distance, K)
    assert result["status"] == "complete"
    assert np.allclose(result["R"], R, atol=.002)
    assert result["t"]@t > .99
    detector, matcher = old_fixture(a, b, matches, distance)
    cv2.setRNGSeed(0)
    legacy = old._estimate_relative_pose(np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8), K, detector, matcher)
    assert legacy is not None
    assert np.array_equal(result["R"], legacy[0]) and np.array_equal(result["t"], legacy[1])
    assert result["n_inliers"] == legacy[2]


def test_static_zero_distance_matches_do_not_displace_real_moving_geometry():
    a, b, matches, distance, _, t = scene(count=45, static=200)
    result = p.estimate_pair(a, b, matches, distance, K)
    assert result["status"] == "complete" and result["t"]@t > .99
    assert not result["moving_mask"][-200:].any()
    assert np.all(result["selected_match_indices"] < 45)
    assert result["match_count"] == 245


def test_low_parallax_is_unknown_not_a_verified_stop(monkeypatch):
    a, _, matches, distance, _, _ = scene()
    monkeypatch.setattr(cv2, "findEssentialMat", lambda *_args, **_kw: pytest.fail("low parallax must not fit E"))
    result = p.estimate_pair(a, a.copy(), matches, distance, K)
    assert result["status"] == "unknown" and result["reason"] == "low_parallax_unknown"
    assert result["failure_is_stop"] is result["metric_scale"] is False


def mock_fit(monkeypatch, *, n=300, translation=None, rotation=None):
    essential = np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    masks = []
    def find(a, b, matrix, **kwargs):
        assert kwargs == {"method": cv2.RANSAC, "prob": .999, "threshold": 1.}
        mask = np.ones((len(a), 1), np.uint8); masks.append(mask)
        return essential.copy(), mask
    def recover(E, a, b, matrix, *, mask):
        mask[:2] = 0
        return n, np.eye(3) if rotation is None else rotation, np.array([[0.], [0.], [2.]]) if translation is None else translation, mask
    monkeypatch.setattr(cv2, "findEssentialMat", find)
    monkeypatch.setattr(cv2, "recoverPose", recover)
    return masks


def test_top300_stable_ties_and_mask_copy_before_recover_mutation(monkeypatch):
    rng = np.random.default_rng(12); a = rng.uniform([10, 10], [1200, 700], (400, 2)).astype(np.float32)
    b = a+np.array([10., 0.], np.float32)
    order = np.arange(399, -1, -1)
    matches = np.column_stack([order, order]).astype(np.int32)
    masks = mock_fit(monkeypatch)
    result = p.estimate_pair(a, b, matches, np.ones(400), K)
    assert result["status"] == "complete" and result["selected_match_indices"].tolist() == list(range(300))
    assert result["essential_mask"].tolist() == [1]*300
    assert result["pose_mask"][:2].tolist() == [0, 0] and masks[0][:2].tolist() == [[0], [0]]
    assert result["t_before_normalization"].tolist() == [0., 0., 2.]
    assert result["t"].tolist() == [0., 0., 1.]


@pytest.mark.parametrize("case", ["features", "matches", "moving"])
def test_existing_initial16_gates_preserved(monkeypatch, case):
    a, b, matches, distance, _, _ = scene()
    if case == "features": a, b, matches, distance = a[:15], b[:15], matches[:15], distance[:15]
    elif case == "matches": matches, distance = matches[:15], distance[:15]
    else: b[15:] = a[15:]
    monkeypatch.setattr(cv2, "findEssentialMat", lambda *_args, **_kw: pytest.fail("16 gate did not decline"))
    result = p.estimate_pair(a, b, matches, distance, K)
    expected = "insufficient_moving_matches" if case == "moving" else "insufficient_"+case
    assert result["status"] == "unknown" and result["reason"] == expected


@pytest.mark.parametrize("case", ["n7", "zero_t", "improper_R"])
def test_pose_rejections_keep_unknown_evidence(monkeypatch, case):
    a, b, matches, distance, _, _ = scene()
    mock_fit(monkeypatch, n=7 if case == "n7" else 20,
        translation=np.zeros((3, 1)) if case == "zero_t" else None,
        rotation=np.diag([1., 1., -1.]) if case == "improper_R" else None)
    result = p.estimate_pair(a, b, matches, distance, K)
    assert result["status"] == "unknown" and not result["failure_is_stop"]
    assert result["n_inliers"] == 0 and result["recover_pose_inliers"] in (7, 20)
    assert result["E"].shape == (3, 3) and len(result["essential_mask"]) > 0


def test_invalid_multi_essential_is_saved_and_rejected(monkeypatch):
    a, b, matches, distance, _, _ = scene()
    monkeypatch.setattr(cv2, "findEssentialMat", lambda *args, **kw: (np.ones((6, 3)), np.ones((len(args[0]), 1), np.uint8)))
    monkeypatch.setattr(cv2, "recoverPose", lambda *_a, **_k: pytest.fail("multi E must not recover"))
    result = p.estimate_pair(a, b, matches, distance, K)
    assert result["reason"] == "invalid_essential" and result["E"].shape == (6, 3)


@pytest.mark.parametrize("case", ["bad_index", "duplicates", "NaN", "oversize", "bad_K", "missing_distance"])
def test_bad_correspondence_packages_decline(case):
    a, b, matches, distance, _, _ = scene()
    matrix = K.copy()
    if case == "bad_index": matches[0, 0] = len(a)
    if case == "duplicates": matches[0] = matches[1]
    if case == "NaN": a[0, 0] = np.nan
    if case == "oversize": a = np.zeros((16385, 2), np.float32)
    if case == "bad_K": matrix[0, 0] = 0
    if case == "missing_distance": distance = distance[:-1]
    with pytest.raises(ValueError): p.estimate_pair(a, b, matches, distance, matrix)


def test_chain_matches_legacy_composition_and_plane_handedness(monkeypatch):
    angle = .2; R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    rotations = np.array([np.eye(3), np.full((3, 3), np.nan), R, R])
    translations = np.array([[0, 0, -1], [np.nan]*3, [.2, 0, -.9797958971132712], [0, 0, -1.]])
    valid = np.array([True, False, True, True]); counts = np.array([25, 0, 30, 50])
    new = p.chain_relative_poses(rotations, translations, valid, counts)
    relatives = iter([(rotations[i], translations[i], int(counts[i])) if valid[i] else None for i in range(4)])
    monkeypatch.setattr(old._ThreadLocalDetectors, "get", lambda *_: (None, None))
    monkeypatch.setattr(old, "_estimate_relative_pose", lambda *_: next(relatives))
    legacy = old.estimate_trajectory([np.zeros((8, 8), np.uint8)]*5, K, n_workers=1)
    for key in new: np.testing.assert_array_equal(new[key], getattr(legacy, key))
    assert not new["valid"][2] and np.array_equal(new["centers"][2], new["centers"][1])


def test_chain_rejects_nonunit_or_nonproper_accepted_poses():
    with pytest.raises(ValueError): p.chain_relative_poses(np.eye(3)[None], np.array([[0., 0., 2.]]), np.ones(1, bool), np.array([20]))
    with pytest.raises(ValueError): p.chain_relative_poses(np.zeros((1, 3, 3)), np.array([[0., 0., 1.]]), np.ones(1, bool), np.array([20]))
