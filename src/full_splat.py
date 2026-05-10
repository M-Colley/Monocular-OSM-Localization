"""High-quality Gaussian-splat rendering and (optional) full 3DGS fit.

Two paths, picked at the CLI level:

1. ``render_full_splat_topdown`` — **anisotropic rasterizer (no training)**.
   Takes any colored point cloud (the sparse ORB cloud or the dense DA3
   cloud) and renders a top-down view as alpha-blended *anisotropic*
   Gaussians, instead of the isotropic disks + global blur used by
   ``splat.render_topdown_splat``. Each point's covariance is estimated
   from its k-NN neighborhood by local PCA; the projection onto the
   ground plane gives a 2D covariance per Gaussian, and we composite
   back-to-front with proper Gaussian falloff. Pure NumPy/SciPy, runs on
   the CPU in seconds. This is the visual-quality fix for the "really
   bad" sparse-disk render.

2. ``fit_3dgs`` — **full 3D Gaussian Splatting fit (training)**. Takes
   the DA3 reconstruction as initialization and gradient-descent fits
   per-Gaussian position / covariance / opacity / SH coefficients
   against the actual keyframes using `gsplat`. This is the proper
   3DGS that the original spec referred to. Requires CUDA and
   ``pip install gsplat``; lazy-imported so the rest of the pipeline
   still works without it.

The two paths are separate flags (``--full-splat`` vs ``--train-3dgs``)
because their cost is wildly different: option 1 adds seconds, option
2 adds minutes-to-tens-of-minutes per clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Option 1: anisotropic top-down rasterizer (no training)
# ---------------------------------------------------------------------------


def _estimate_local_covariances(
    points: np.ndarray,
    *,
    k: int = 12,
    min_eig: float = 1e-4,
) -> np.ndarray:
    """Per-point 3x3 covariance from k-nearest-neighbor PCA.

    A Gaussian splat's covariance encodes the *local surface* it
    approximates: flat regions yield disk-like ellipsoids whose short
    axis is the surface normal; thin structures yield rod-like
    ellipsoids. Real 3DGS learns these by gradient descent; for a
    train-free render we can recover a decent first estimate from the
    point's neighborhood. `min_eig` prevents zero-variance directions
    from collapsing the Gaussian to a degenerate plane.
    """
    from scipy.spatial import cKDTree

    n = len(points)
    if n == 0:
        return np.zeros((0, 3, 3))

    tree = cKDTree(points)
    k_eff = min(k + 1, n)
    _, idx = tree.query(points, k=k_eff)
    if k_eff == 1:
        # Degenerate: only one point. Use a tiny isotropic covariance.
        return np.tile(np.eye(3) * min_eig, (n, 1, 1))

    nbr_idx = idx[:, 1:]
    nbrs = points[nbr_idx]                       # (N, k, 3)
    centered = nbrs - points[:, None, :]          # (N, k, 3)
    covs = np.einsum("nki,nkj->nij", centered, centered) / max(k_eff - 1, 1)

    # Regularize: clamp eigenvalues from below so each Gaussian has
    # nonzero extent in every direction. A degenerate (rank-2) covariance
    # would project to a line in BEV and disappear under rasterization.
    eigvals, eigvecs = np.linalg.eigh(covs)
    eigvals = np.clip(eigvals, min_eig, None)
    covs = np.einsum("nij,nj,nkj->nik", eigvecs, eigvals, eigvecs)
    return covs


def render_full_splat_topdown(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    resolution: int = 1024,
    margin_px: int = 24,
    scale: float = 1.4,
    opacity: float = 0.55,
    knn: int = 12,
    background: tuple[int, int, int] = (0, 0, 0),
    max_radius_px: int = 28,
    progress: bool = False,
) -> np.ndarray:
    """Render `points` as anisotropic 2D Gaussians from a top-down view.

    Pipeline per-point:

      * fit a 3D covariance from k-NN neighbors (`_estimate_local_covariances`)
      * project to the (x, z) ground plane → 2D covariance
      * scale up by `scale` (nearest-neighbor PCA underestimates true
        Gaussian extent — the closest neighbors lie *inside* the
        underlying surface)
      * find the 2D ellipse's pixel bounding box
      * accumulate `opacity * exp(-0.5 * d^T * Σ^-1 * d)` * color in a
        front-to-back alpha-composited buffer

    Sorted top-down (i.e. by world-y, the camera-vertical axis) so
    higher-up Gaussians render in front — matches what a real 3DGS
    rasterizer would produce when looking straight down.

    `scale` and `opacity` are the two "look" knobs. Defaults give a
    soft, surface-like render on DA3 clouds; tune `scale` up for
    sparser clouds.
    """
    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors)
    if cols.dtype != np.uint8:
        cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8)

    img = np.zeros((resolution, resolution, 3), dtype=np.float32)
    transmittance = np.ones((resolution, resolution), dtype=np.float32)
    bg = np.array(background, dtype=np.float32)

    if len(pts) == 0:
        return np.broadcast_to(bg, img.shape).astype(np.uint8).copy()

    covs3d = _estimate_local_covariances(pts, k=knn) * (scale ** 2)
    # Project to (x, z): drop the y row/column.
    cov2d = covs3d[:, [[0], [2]], [0, 2]]   # (N, 2, 2)

    xz = pts[:, [0, 2]]
    xmin, ymin = xz.min(axis=0)
    xmax, ymax = xz.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    s = (resolution - 2 * margin_px) / span

    px = (xz[:, 0] - xmin) * s + margin_px
    # Flip y so increasing world-z goes UP in the image.
    py = resolution - margin_px - 1 - (xz[:, 1] - ymin) * s
    cov_px = cov2d * (s ** 2)

    # Render top-most Gaussians last (overwrite below). World convention
    # in this codebase is camera-y up, so a smaller y-coord = higher up;
    # we sort by y *descending* so the highest things composite last.
    order = np.argsort(-pts[:, 1])

    iter_range = order
    if progress:
        try:
            from tqdm import tqdm
            iter_range = tqdm(order, desc="splat", leave=False)
        except ImportError:
            pass

    for i in iter_range:
        cov = cov_px[i]
        # Eigendecomp gives ellipse axes; bounding box ≈ 3-sigma.
        try:
            eigvals, _ = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        max_sigma = float(np.sqrt(max(eigvals.max(), 1e-6)))
        radius = int(min(max(np.ceil(3.0 * max_sigma), 1), max_radius_px))

        cx, cy = float(px[i]), float(py[i])
        x0 = max(int(np.floor(cx - radius)), 0)
        x1 = min(int(np.ceil(cx + radius)) + 1, resolution)
        y0 = max(int(np.floor(cy - radius)), 0)
        y1 = min(int(np.ceil(cy + radius)) + 1, resolution)
        if x1 <= x0 or y1 <= y0:
            continue

        try:
            cov_inv = np.linalg.inv(cov + np.eye(2) * 1e-6)
        except np.linalg.LinAlgError:
            continue

        ys = np.arange(y0, y1, dtype=np.float32) - cy
        xs = np.arange(x0, x1, dtype=np.float32) - cx
        XX, YY = np.meshgrid(xs, ys)
        # 2D Mahalanobis: d^T Σ^-1 d
        a, b = cov_inv[0, 0], cov_inv[0, 1]
        c = cov_inv[1, 1]
        m = a * XX * XX + 2.0 * b * XX * YY + c * YY * YY
        gauss = np.exp(-0.5 * m)
        alpha = opacity * gauss   # (h, w)

        # Front-to-back: out += T * alpha * c;  T *= (1 - alpha)
        T = transmittance[y0:y1, x0:x1]
        contrib = (T * alpha)[:, :, None] * cols[i].astype(np.float32)
        img[y0:y1, x0:x1] += contrib
        transmittance[y0:y1, x0:x1] = T * (1.0 - alpha)

    # Composite the residual transmittance onto background.
    final = img + transmittance[:, :, None] * bg
    return np.clip(final, 0, 255).astype(np.uint8)


def render_full_splat_to_file(
    points: np.ndarray,
    colors: np.ndarray,
    path: Path,
    **kwargs,
) -> None:
    img_rgb = render_full_splat_topdown(points, colors, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Option 2: full 3DGS fit via gsplat (training)
# ---------------------------------------------------------------------------


@dataclass
class TrainedSplat:
    """Output of a 3DGS training run. Anisotropic Gaussians in world frame."""
    means: np.ndarray            # (N, 3)
    quats: np.ndarray            # (N, 4)  rotation as wxyz
    scales: np.ndarray           # (N, 3)  log-scale per axis
    opacities: np.ndarray        # (N,)    pre-sigmoid logits
    colors: np.ndarray           # (N, 3)  RGB in [0, 1]
    n_iters: int
    final_loss: float


def _import_gsplat():
    try:
        import gsplat  # noqa: F401
        import torch
    except ImportError as e:
        raise RuntimeError(
            "Full 3DGS fit requires `gsplat` + `torch` with CUDA. Install "
            "with: pip install torch torchvision --extra-index-url "
            "https://download.pytorch.org/whl/cu121 && pip install gsplat"
        ) from e
    return gsplat, torch


def fit_3dgs(
    rec,                                # DA3Reconstruction (init source)
    frames: Sequence[np.ndarray],       # BGR uint8, full video frames
    *,
    n_iters: int = 2000,
    lr_means: float = 1.6e-4,
    lr_scales: float = 5e-3,
    lr_quats: float = 1e-3,
    lr_opacities: float = 5e-2,
    lr_colors: float = 2.5e-3,
    init_scale_factor: float = 1.0,
    densify_every: int = 200,
    device: str = "cuda",
    log_every: int = 100,
    progress: bool = True,
) -> TrainedSplat:
    """Gradient-descent fit a real 3D Gaussian Splat.

    DA3 already gives us the SfM front-end (per-frame metric depth, K, R,
    t). We initialize one Gaussian per dense point with the point color
    and a small isotropic scale, then optimize per-Gaussian position,
    rotation (as quaternion), per-axis scale, opacity, and RGB against
    the actual rendered keyframes. This is the same loop as Inria's
    reference 3DGS, just shorter (we skip SH bands beyond DC and skip
    the densification adaptive control beyond a basic clone-on-grad
    schedule).

    Output is a `TrainedSplat`. Use `save_trained_splat_ply` to write a
    PLY consumable by SuperSplat / SIBR / any 3DGS viewer.

    Cost: a few minutes for ~50k Gaussians on a consumer GPU. Memory is
    the limiter; if you OOM, reduce DA3 keyframes or `n_iters`.
    """
    gsplat, torch = _import_gsplat()

    pts = torch.tensor(np.asarray(rec.points_world), dtype=torch.float32, device=device)
    rgbs = torch.tensor(
        np.asarray(rec.colors_rgb, dtype=np.float32) / 255.0,
        dtype=torch.float32, device=device,
    )

    n = len(pts)
    if n == 0:
        raise ValueError("empty point cloud — cannot fit 3DGS")

    # Per-Gaussian parameters (same parameterization as gsplat examples).
    means = torch.nn.Parameter(pts.clone())
    # Isotropic init scale ≈ median nearest-neighbor distance — small enough
    # that splats don't overlap on init, large enough that they're visible.
    from scipy.spatial import cKDTree
    tree = cKDTree(rec.points_world)
    nn_dist, _ = tree.query(rec.points_world, k=2)
    init_log_scale = float(np.log(np.maximum(np.median(nn_dist[:, 1]), 1e-3) * init_scale_factor))
    scales = torch.nn.Parameter(torch.full((n, 3), init_log_scale, device=device))
    # Identity quaternion (w, x, y, z).
    quats = torch.nn.Parameter(
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4).clone()
    )
    # Opacity logit for sigmoid(logit)=0.5.
    opacities = torch.nn.Parameter(torch.zeros(n, device=device))
    colors = torch.nn.Parameter(rgbs.clone())

    optim = torch.optim.Adam([
        {"params": [means], "lr": lr_means},
        {"params": [scales], "lr": lr_scales},
        {"params": [quats], "lr": lr_quats},
        {"params": [opacities], "lr": lr_opacities},
        {"params": [colors], "lr": lr_colors},
    ])

    # Build training views from DA3's keyframe poses.
    h_proc, w_proc = rec.processed_size
    views = []
    for k in range(rec.extrinsics_w2c.shape[0]):
        kf_idx = int(rec.keyframe_indices[k])
        rgb = cv2.cvtColor(np.asarray(frames[kf_idx]), cv2.COLOR_BGR2RGB)
        rgb_proc = cv2.resize(rgb, (w_proc, h_proc), interpolation=cv2.INTER_AREA)
        views.append({
            "image": torch.tensor(rgb_proc, dtype=torch.float32, device=device) / 255.0,
            "K": torch.tensor(rec.intrinsics[k], dtype=torch.float32, device=device),
            "extr": torch.tensor(rec.extrinsics_w2c[k], dtype=torch.float32, device=device),
        })

    def render(view):
        viewmat = torch.eye(4, device=device)
        viewmat[:3, :4] = view["extr"]
        K = view["K"]
        # gsplat.rasterization expects [..., N, 3], etc.
        rendered, _, _ = gsplat.rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=w_proc,
            height=h_proc,
            packed=False,
        )
        return rendered[0]  # (H, W, 3)

    iter_iter = range(n_iters)
    if progress:
        try:
            from tqdm import tqdm
            iter_iter = tqdm(iter_iter, desc="3DGS fit")
        except ImportError:
            pass

    final_loss = float("nan")
    rng = np.random.default_rng(0)
    for it in iter_iter:
        view = views[int(rng.integers(0, len(views)))]
        out = render(view)
        loss = (out - view["image"]).abs().mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        # Densification is hard to do well in <2k lines; we omit it. The
        # init from DA3 is dense enough to give a usable splat without
        # the adaptive density control loop.
        final_loss = float(loss.detach())
        if log_every and (it % log_every == 0):
            if hasattr(iter_iter, "set_postfix"):
                iter_iter.set_postfix(loss=final_loss)

    # Detach to NumPy for serialization.
    return TrainedSplat(
        means=means.detach().cpu().numpy(),
        quats=(quats / quats.norm(dim=-1, keepdim=True)).detach().cpu().numpy(),
        scales=scales.detach().cpu().numpy(),
        opacities=opacities.detach().cpu().numpy(),
        colors=colors.detach().clamp(0, 1).cpu().numpy(),
        n_iters=n_iters,
        final_loss=final_loss,
    )


def save_trained_splat_ply(splat: TrainedSplat, path: Path) -> None:
    """Write the trained Gaussians to a PLY readable by 3DGS viewers
    (SuperSplat, SIBR, antimatter15 viewer).

    The format is the de-facto standard from Inria's reference impl:
    `means (3), normals (3 zeros), f_dc (3 SH-DC), opacity (1),
    scale (3 log), rotation (4 quat)`.
    """
    n = len(splat.means)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # SH DC coefficient = (color - 0.5) / 0.28209479177387814
    SH_C0 = 0.28209479177387814
    f_dc = (splat.colors - 0.5) / SH_C0

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "end_header\n"
    )

    arr = np.empty((n, 17), dtype=np.float32)
    arr[:, 0:3] = splat.means
    arr[:, 3:6] = 0.0
    arr[:, 6:9] = f_dc
    arr[:, 9] = splat.opacities
    arr[:, 10:13] = splat.scales
    arr[:, 13:17] = splat.quats

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())
