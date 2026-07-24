"""VGGT feed-forward camera trajectory — drift-free shape for SELECTION.

VGGT (Visual Geometry Grounded Transformer) predicts camera extrinsics for
a set of frames in a single forward pass with global cross-frame attention.
Its top-down path is therefore globally **drift-free** — on a loop it closes
to ~1% where monocular VO leaves a 27% gap. That matters because the
selection wall (which OSM walk is the true street) is driven by drift:
VO's drifted shape fits parallel streets equally well (the true route ranks
~#14), but VGGT's drift-free bearing signature matches the TRUE street
(ranks #1-5).

VGGT is, however, locally noisy on forward-driving footage (small baselines
between adjacent frames), so its fine geometry is worse than VO. The
production use is therefore a **hybrid**: VGGT selects the *area* (gates
enumeration to its top start-nodes), and the loop-closed VO supplies the
precise geometry within it. See ``vggt_seed_nodes``.

Heavy + optional: needs the ``vggt`` package + a GPU + a ~5 GB weights
download. Every entry point degrades to ``None`` / ``[]`` if unavailable,
so the pipeline keeps running without it.
"""

from __future__ import annotations

import tempfile

import numpy as np


def vggt_camera_trajectory(
    frames: list,
    *,
    n_keyframes: int = 64,
    smooth: int = 7,
    device: str | None = None,
) -> np.ndarray | None:
    """Top-down ``(x, z)`` camera-centre path ``(N, 2)`` from VGGT, or None.

    Subsamples ``frames`` to ``n_keyframes`` (VGGT's memory scales with the
    count, and wider baselines give cleaner poses than dense ones), runs
    VGGT, extracts camera centres from the extrinsics, and applies a
    Savitzky-Golay smooth (``smooth`` window) to tame the local jitter.
    Returns ``None`` if VGGT/torch/CUDA is unavailable or inference fails.
    """
    try:
        import cv2
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except Exception:
        return None

    if not frames:
        return None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if len(frames) > n_keyframes:
        idx = np.linspace(0, len(frames) - 1, n_keyframes).round().astype(int)
        frames = [frames[i] for i in idx]

    try:
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, f in enumerate(frames):
                p = f"{td}/f{i:04d}.png"
                cv2.imwrite(p, f)
                paths.append(p)
            images = load_and_preprocess_images(paths).to(device)
            model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
            use_amp = device == "cuda"
            dtype = (torch.bfloat16
                     if use_amp and torch.cuda.get_device_capability()[0] >= 8
                     else torch.float16)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
                pred = model(images[None])
            extri, _intri = pose_encoding_to_extri_intri(
                pred["pose_enc"], images.shape[-2:])
            extri = extri[0].float().cpu().numpy()  # (N,3,4) world->cam (OpenCV)
    except Exception:
        return None

    # Camera centre C = -R^T t; ground plane is X-Z (Y is down in OpenCV).
    centres = np.array([-(e[:, :3].T @ e[:, 3]) for e in extri])
    xy = centres[:, [0, 2]]
    if smooth and len(xy) > smooth:
        try:
            from scipy.signal import savgol_filter
            w = smooth if smooth % 2 else smooth + 1
            xy = np.column_stack([savgol_filter(xy[:, 0], w, 2),
                                  savgol_filter(xy[:, 1], w, 2)])
        except Exception:
            pass
    return xy


def vggt_seed_nodes(
    vggt_xy: np.ndarray,
    road,
    *,
    estimated_length_m: float,
    top_k: int = 8,
) -> list:
    """Start-nodes of VGGT's top matches — the area to gate VO enumeration to.

    Matches the drift-free VGGT trajectory against the road graph and
    returns the distinct start-nodes of its ``top_k`` best candidates.
    Feeding these as ``extra_start_nodes`` + ``restrict_to_start_nodes``
    to the VO match confines the (precise but selection-ambiguous) VO pool
    to the place VGGT identified.
    """
    from .trajectory_matching import match_trajectory
    from .visual_odometry import trajectory_arc_length

    if vggt_xy is None or len(vggt_xy) < 2:
        return []
    arc = float(trajectory_arc_length(vggt_xy)[-1])
    if arc <= 1e-6:
        return []
    pool = match_trajectory(
        vggt_xy, road, final_top_k=max(top_k, 15), sample_every=1,
        estimated_length_m=estimated_length_m, locked_scale=estimated_length_m / arc,
        progress=False)
    return list(dict.fromkeys(c.start_node for c in pool[:top_k]))
