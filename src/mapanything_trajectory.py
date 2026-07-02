"""MapAnything (3DV'26) submap-stitched metric trajectory (VGGT-SLAM-lite).

A single feed-forward MapAnything pass over a long drive collapses the metric
scale: a ~2.5 km route reconstructs as ~400 m because distant frames share no
co-visibility (global-fit RMS 421-437 m, worse than VO's 258 m on the Ulm GT
clip). This module instead runs MapAnything on short, HIGH-OVERLAP sliding
windows -- where its feed-forward metric reconstruction is reliable -- and chains
consecutive windows with a *scale-guarded* Sim(3) (Umeyama) fit over their shared
frames. That propagates orientation + scale along the route without a global
bundle/factor-graph solver (the VGGT-SLAM idea, minus the gtsam dependency that
has no Windows wheel). On the Ulm GT clip the best window config recovers the
route extent and beats VO on trajectory SHAPE: global-fit RMS 219 vs 258 m.

Heavy + optional: needs the ``mapanything`` package + a GPU + ~9 GB of weights
(model.safetensors + the dinov2_vitg14 backbone). NB installing MapAnything
downgrades torch to CPU -- restore with the cu128 wheel. Returns ``None`` if
anything is missing, so callers degrade gracefully.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) mapping ``src`` -> ``dst`` for 3D point sets (N,3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    H = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / max(var, 1e-9)
    t = mu_d - s * R @ mu_s
    return float(s), R, t


def _rigid(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Rigid (scale = 1) fit -- the fallback when the Umeyama scale is implausible."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (dst - mu_d).T @ (src - mu_s)
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return 1.0, R, mu_d - R @ mu_s


def _yaw_fit(
    src: np.ndarray,
    dst: np.ndarray,
    scale_guard: tuple[float, float],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Yaw-only (rotation about +y) similarity for rank-deficient overlaps.

    On straight driving the shared camera positions are collinear, so a
    full 3D fit leaves the rotation about the driving line unconstrained
    and the SVD returns an arbitrary roll that every later window
    inherits. Constraining the rotation to the ground plane (x-z, +y is
    the camera-down axis) removes that degree of freedom: 2D Procrustes
    for the yaw angle, spread ratio for the scale (guarded to 1 like the
    full fit — MapAnything is metric).
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    # Optimal in-plane rotation angle from the 2D cross/dot sums.
    x_s, z_s = sc[:, 0], sc[:, 2]
    x_d, z_d = dc[:, 0], dc[:, 2]
    a = np.arctan2(float((x_s * z_d - z_s * x_d).sum()),
                   float((x_s * x_d + z_s * z_d).sum()))
    c, s_a = np.cos(a), np.sin(a)
    R = np.array([[c, 0.0, -s_a], [0.0, 1.0, 0.0], [s_a, 0.0, c]])
    denom = float((sc ** 2).sum())
    s = float(np.sqrt((dc ** 2).sum() / denom)) if denom > 1e-12 else 1.0
    if not (scale_guard[0] <= s <= scale_guard[1]):
        s = 1.0
    return s, R, mu_d - s * R @ mu_s


def stitch_windows(
    windows: list[tuple[list[int], np.ndarray]],
    *,
    min_overlap: int = 3,
    scale_guard: tuple[float, float] = (0.6, 1.7),
) -> dict[int, np.ndarray]:
    """Chain per-window 3D camera positions into one global frame.

    ``windows`` is a list of ``(frame_ids, positions)`` where ``positions`` is
    (len(frame_ids), 3) in that window's own arbitrary metric frame and
    consecutive windows share some frame ids. Each new window is aligned to the
    accumulated global frame by a Sim(3) fit over the shared ids; a scale outside
    ``scale_guard`` (MapAnything is metric -> inter-window scale must be ~1)
    signals a degenerate/collinear overlap and falls back to a rigid fit.
    Returns ``{frame_id: global_xyz}``.
    """
    G: dict[int, np.ndarray] = {}
    for k, (ids, pos) in enumerate(windows):
        pos = np.asarray(pos, dtype=np.float64)
        if k == 0:
            for fid, p in zip(ids, pos):
                G[fid] = p
            continue
        ov = [(j, fid) for j, fid in enumerate(ids) if fid in G]
        if len(ov) < min_overlap:
            continue
        src = np.array([pos[j] for j, _ in ov])
        dst = np.array([G[fid] for _, fid in ov])
        # Straight-driving overlaps are collinear: the full 3D fit's
        # rotation about the line is unconstrained (arbitrary roll with a
        # plausible scale, so the scale guard can't catch it). Detect the
        # rank deficiency from the singular-value ratio of the centered
        # shared positions and fall back to a yaw-only in-plane fit.
        sv = np.linalg.svd(src - src.mean(0), compute_uv=False)
        if len(src) < 3 or sv[0] < 1e-9 or sv[1] / sv[0] < 0.05:
            s, R, t = _yaw_fit(src, dst, scale_guard)
        else:
            s, R, t = _umeyama(src, dst)
            if not (scale_guard[0] <= s <= scale_guard[1]):
                s, R, t = _rigid(src, dst)
        for j, fid in enumerate(ids):
            if fid not in G:
                G[fid] = s * R @ pos[j] + t
    return G


def _positions_to_xy(positions: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Project 3D camera positions onto their 2D driving plane (PCA top-2).

    numpy's SVD returns the axes with arbitrary signs, so the raw top-2
    projection can be a MIRROR of the true top-down path (turn signs
    inverted -- fatal for the chirality-sensitive matcher downstream).
    Enforce view-from-above handedness: the kept axes' normal must align
    with ``up`` (camera up = -y in the OpenCV convention MapAnything's
    first-camera world frame approximately shares); flip the second axis
    when it points the other way.
    """
    if up is None:
        up = np.array([0.0, -1.0, 0.0])
    c = positions - positions.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    axes = vt[:2].T.copy()
    if float(np.dot(np.cross(axes[:, 0], axes[:, 1]), up)) < 0.0:
        axes[:, 1] *= -1.0
    return c @ axes


def mapanything_trajectory_xy(
    video_path: str,
    t0: float,
    t1: float,
    *,
    dt: float = 2.0,
    window: int = 16,
    step: int = 8,
    device=None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Submap-stitched 2D MapAnything trajectory for ``video_path`` over [t0,t1].

    Returns ``(xy, times)`` with ``xy`` of shape (K,2) in metres on the driving
    plane and ``times`` the sampling timestamps (s), or ``None`` if MapAnything /
    its weights / a GPU are unavailable.
    """
    try:
        import cv2
        import torch
        from mapanything.models import MapAnything
        from mapanything.utils.image import load_images
    except Exception:
        return None
    cap = None
    try:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = MapAnything.from_pretrained("facebook/map-anything").to(device).eval()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        times = np.arange(t0, t1, dt)
        M = len(times)
        if M < window:
            return None

        def infer(ids: list[int]) -> tuple[list[int], np.ndarray]:
            # Return the ids that were actually decoded alongside their
            # positions: a failed cap.read() (e.g. a seek past EOF) must
            # not shift every later position onto the wrong frame id.
            kept: list[int] = []
            with tempfile.TemporaryDirectory() as folder:
                for fid in ids:
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(times[fid]) * 1000)
                    ok, bgr = cap.read()
                    if ok:
                        cv2.imwrite(os.path.join(folder, f"{len(kept):03d}.jpg"), bgr)
                        kept.append(fid)
                if not kept:
                    return [], np.zeros((0, 3))
                views = load_images(folder)
                with torch.no_grad():
                    preds = model.infer(views, memory_efficient_inference=True,
                                        use_amp=True, amp_dtype="bf16")
            out = []
            for p in preds:
                if "cam_trans" in p:
                    out.append(np.asarray(p["cam_trans"].detach().cpu().numpy()).ravel()[:3])
                else:
                    cp = np.asarray(p["camera_poses"].detach().cpu().numpy()).reshape(-1, 4, 4)[0]
                    out.append(cp[:3, 3])
            return kept, np.array(out)

        starts = list(range(0, max(1, M - window + 1), step))
        if starts and starts[-1] != M - window:
            starts.append(max(0, M - window))
        windows = []
        for st in starts:
            ids = list(range(st, min(st + window, M)))
            windows.append(infer(ids))

        G = stitch_windows(windows)
        if len(G) < window:
            return None
        fids = sorted(G)
        traj = np.array([G[f] for f in fids])
        return _positions_to_xy(traj), times[fids]
    except Exception:
        return None
    finally:
        if cap is not None:
            cap.release()
