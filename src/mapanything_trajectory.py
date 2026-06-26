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
        s, R, t = _umeyama(src, dst)
        if not (scale_guard[0] <= s <= scale_guard[1]):
            s, R, t = _rigid(src, dst)
        for j, fid in enumerate(ids):
            if fid not in G:
                G[fid] = s * R @ pos[j] + t
    return G


def _positions_to_xy(positions: np.ndarray) -> np.ndarray:
    """Project 3D camera positions onto their 2D driving plane (PCA top-2)."""
    c = positions - positions.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[:2].T


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

        def infer(ids: list[int]) -> np.ndarray:
            folder = tempfile.mkdtemp()
            for j, fid in enumerate(ids):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(times[fid]) * 1000)
                ok, bgr = cap.read()
                if ok:
                    cv2.imwrite(os.path.join(folder, f"{j:03d}.jpg"), bgr)
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
            return np.array(out)

        starts = list(range(0, max(1, M - window + 1), step))
        if starts and starts[-1] != M - window:
            starts.append(max(0, M - window))
        windows = []
        for st in starts:
            ids = list(range(st, min(st + window, M)))
            windows.append((ids, infer(ids)))
        cap.release()

        G = stitch_windows(windows)
        if len(G) < window:
            return None
        fids = sorted(G)
        traj = np.array([G[f] for f in fids])
        return _positions_to_xy(traj), times[fids]
    except Exception:
        return None
