"""Smoke tests for the anisotropic top-down splat renderer.

The gsplat training path needs a CUDA GPU and is skipped here — these
tests cover only the train-free `render_full_splat_topdown` rasterizer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.full_splat import (
    _estimate_local_covariances,
    render_full_splat_topdown,
)


def test_local_covariances_returns_psd_matrices():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(40, 3)).astype(np.float32)
    covs = _estimate_local_covariances(pts, k=8)
    assert covs.shape == (40, 3, 3)
    eigvals = np.linalg.eigvalsh(covs)
    # All eigenvalues must be positive (regularized covariance).
    assert (eigvals > 0).all()


def test_render_handles_empty_cloud():
    img = render_full_splat_topdown(
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.uint8),
        resolution=64,
        background=(7, 8, 9),
    )
    assert img.shape == (64, 64, 3)
    # Empty cloud → entire image is the background colour.
    assert (img[..., 0] == 7).all()
    assert (img[..., 1] == 8).all()
    assert (img[..., 2] == 9).all()


def test_render_produces_colored_output_on_simple_grid():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-5, 5, size=(200, 3)).astype(np.float32)
    cols = np.full((200, 3), [200, 50, 30], dtype=np.uint8)
    img = render_full_splat_topdown(
        pts, cols,
        resolution=128,
        scale=1.5,
        opacity=0.6,
        background=(0, 0, 0),
    )
    assert img.shape == (128, 128, 3)
    # Some pixels must be lit (red-ish) — i.e. splats actually rasterized.
    lit = (img[..., 0] > 30).sum()
    assert lit > 100, f"expected many red pixels, got {lit}"
    # Red channel should dominate green/blue on lit pixels.
    mask = img[..., 0] > 30
    assert img[..., 0][mask].mean() > img[..., 1][mask].mean()


def test_render_output_dtype_and_range():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    cols = np.array([[255, 255, 255]] * 3, dtype=np.uint8)
    img = render_full_splat_topdown(pts, cols, resolution=64, scale=2.0)
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255
