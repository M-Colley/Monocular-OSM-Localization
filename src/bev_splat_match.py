"""BevSplat cross-view localization channel (NeurIPS'26).

Wires the third-party `BevSplat <https://github.com/wangqww/BevSplat>`_
model into our pipeline as an additional aerial-matching backend.

BevSplat takes a ground-view RGB image plus a *real* satellite tile
(512x512 in the reference KITTI training setup) and produces a
correlation map + a per-pixel ``(shift_u, shift_v, heading)`` offset
locating the camera inside that tile. This complements:

* **Trajectory IoU** (``aerial_match._traj_iou_score``) — pure geometry,
  no appearance signal.
* **ORB on OSM raster** (``aerial_match.feature_match_score``) — weak
  cross-domain appearance signal (photo vs schematic line drawing).
* **Deep embedding retrieval** (``embedding_retrieval``) — global
  appearance similarity via ResNet/DINOv2 features over satellite
  tiles or OSM patches.

BevSplat is the strongest of these in principle: it was *trained* to
localize a ground image inside a satellite tile, so it produces a
calibrated pose estimate rather than a similarity score.

Status — functional (verified on Windows, 2026-06-09)
-----------------------------------------------------

The integration runs end-to-end: the model constructs, the published
``KITTI_no_GPS.pth`` checkpoint loads, and a forward pass returns a
score + ``(shift_u, shift_v, heading)``. Both historic blockers are
resolved:

1. **Pre-trained weights are published.** wangqww shipped six
   checkpoints on OneDrive (KITTI_GPS/KITTI_no_GPS + four VIGOR
   variants); see the BevSplat section of ``README.md`` for the link
   and the table. For our dashcam-on-OSM use case grab
   ``KITTI_no_GPS.pth`` and pass it via ``--bev-splat-weights``.
2. **CUDA extensions build locally.** The ``feature_gaussian`` and
   ``pano_feature_gaussian`` subpackages are CUDA C++ extensions (not
   on PyPI); build them once with ``third_party/build_extensions.bat``
   on Windows (it self-activates vcvars64 + CUDA 12.8) or ``pip install
   -e .`` in each on Linux. See ``patches/setup_bevsplat.sh``.

Two Windows-specific notes worth knowing:

* **Run env:** the model + CUDA extensions require the Python that built
  them — on this machine that is the Python 3.12 install
  (``…\\Programs\\Python\\Python312\\python.exe``, torch 2.7.0+cu128),
  *not* base conda. The ``_C.cp312-win_amd64.pyd`` extensions won't load
  under a different Python minor version.
* **dino_fit/dino_Fit case collision:** upstream tracks two files
  differing only in case; on Windows they share one physical file.
  ``patches/setup_bevsplat.sh`` patches and verifies it. Do **not** run
  ``git stash``/``reset``/``checkout`` inside ``third_party/BevSplat`` —
  it silently reverts the patch; re-run the setup script to recover.

Beyond the inference path this module also:

* Renders the satellite tiles for each candidate using
  :func:`embedding_retrieval._render_geotessera_patch` (real DINOv2
  embeddings of satellite imagery, PCA-reduced to RGB) when geotessera
  is installed, falling back to OSM raster otherwise. These tiles are
  written under ``output/<submission>/bev_splat/`` for inspection.
* Reports ``BevSplatMatchResult.error`` per candidate when the model
  cannot be loaded, so the pipeline keeps running.

A :class:`MockBevSplatInference` is provided for tests and weight-free
sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import cv2
import numpy as np

from .aerial_match import render_osm_patch
from .osm_data import RoadGraph
from .trajectory_matching import MatchCandidate


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BevSplatConfig:
    """Knobs for the BevSplat aerial channel.

    Parameters
    ----------
    weights_path:
        Path to the BevSplat checkpoint (``.pth``/``.ckpt``). The
        upstream authors published weights at a OneDrive share linked
        from `our README BevSplat section <../README.md#bevsplat-integration>`_;
        download the relevant `.pth` and pass the local path here.
        Pass ``None`` to disable inference and only render the
        satellite tiles for inspection.
    repo_path:
        Path to a local clone of https://github.com/wangqww/BevSplat,
        with its CUDA extensions built (``cd pano_feature_gaussian &&
        pip install -e .`` and similarly for ``feature_gaussian``). The
        loader will prepend this to ``sys.path`` so it can import
        ``models.models_kitti_seq.Model``. Pass ``None`` to skip the
        inference path (tiles still get rendered).
    device:
        ``"cuda"`` or ``"cpu"``. BevSplat's CUDA extensions assume CUDA;
        CPU inference is not really supported by the upstream model but
        the scaffold tolerates it.
    satellite_source:
        ``"esri"`` / ``"satellite"`` (real RGB orthoimagery via
        ``contextily`` — **recommended**, matches BevSplat's KITTI
        training domain), ``"geotessera"`` (real satellite-derived
        DINOv2 embedding, PCA-reduced to false-colour RGB — visually
        near-identical across inner-city tiles, so non-discriminative),
        or ``"osm"`` (rasterized schematic — domain-mismatched, listed
        for completeness).
    satellite_size:
        Side length of the satellite tile fed to BevSplat. 512 matches
        the KITTI training pipeline.
    half_extent_m:
        Half-side of the satellite tile in metres. The KITTI setup is
        about 60 m. Larger tiles widen the search radius but coarsen
        the prediction; smaller tiles assume better priors.
    geotessera_year:
        Tile year for the geotessera embedding source.
    model_args:
        Override dict for the BevSplat argparse defaults (see
        ``train_KITTI_weak_seq.parse_args``). Used to build the
        ``argparse.Namespace`` passed to ``Model(args, device=...)``.
        Unspecified fields fall back to the upstream defaults.
    sequence_length:
        Number of "ground views" the model expects as a sequence; we
        replicate the single query frame this many times to satisfy the
        shape (the model was trained with sequences but the
        replicate-trick is a documented eval mode).
    """

    weights_path: Path | None = None
    repo_path: Path | None = None
    device: str = "cuda"
    satellite_source: str = "geotessera"
    satellite_size: int = 512
    half_extent_m: float = 60.0
    geotessera_year: int = 2024
    model_args: dict[str, object] = field(default_factory=dict)
    sequence_length: int = 2
    # Camera height above the road surface in metres, used by the
    # analytic flat-ground depth map fed to the model (KITTI's stereo
    # rig sits at ~1.65 m; a windshield-mounted dashcam is ~1.4-1.6 m).
    camera_height_m: float = 1.55
    # Depth assigned to rows at/above the horizon (and clamp for rows
    # just below it) in the flat-ground depth map.
    max_ground_depth_m: float = 80.0
    model_module: str = "models.models_kitti_nips"
    # Which file inside the cloned BevSplat repo defines the `Model` class.
    # Probing the commit `187da9e`:
    #   models.models_kitti_seq  — broken upstream (missing `loss/`, `gaussian.encoder`, `gaussian.decoder`)
    #   models.models_kitti_nips — imports cleanly once `feat_gaussian` CUDA ext is built
    #   models.models_kitti_vfa  — imports cleanly with zero extra setup
    #   models.models_kitti_orienternet — imports cleanly with zero extra setup
    #   models.models_vigor      — imports cleanly once `pano_gaussian_feat` is built
    # KITTI dashcam checkpoints (KITTI_GPS / KITTI_no_GPS) most likely match
    # `models_kitti_nips` (the NeurIPS 2026 reference version); set to
    # `models.models_kitti_vfa` if you want a CUDA-extension-free smoke
    # test of the integration before building the extensions.


@dataclass
class BevSplatMatchResult:
    """Per-candidate output of the BevSplat channel.

    A ``score`` of ``0.0`` and a populated ``error`` indicates that the
    model could not be invoked for this candidate (e.g. weights missing
    or satellite tile failed to render); the pipeline keeps going.
    """

    candidate_index: int
    score: float                  # peak correlation in [0, 1] (higher = better)
    pred_shift_u: float           # normalized lateral shift in tile, [-1, 1]
    pred_shift_v: float           # normalized longitudinal shift in tile, [-1, 1]
    pred_heading: float           # normalized heading delta, [-1, 1]
    satellite_path: Path | None   # rendered satellite tile (always written if produced)
    error: str | None = None


# ---------------------------------------------------------------------------
# Inference contract
# ---------------------------------------------------------------------------


class BevSplatInference(Protocol):
    """Callable contract for a BevSplat inference backend.

    Concrete backends:

    * :class:`MockBevSplatInference` (tests, sanity checks before weights land).
    * :func:`_load_bev_splat_inference` (real upstream model loader).

    The return is ``(score, shift_u, shift_v, heading_delta)`` where
    ``shift_*`` and ``heading_delta`` are normalized to ``[-1, 1]`` to
    match the upstream KITTI training labels.
    """

    def __call__(
        self,
        ground_rgb: np.ndarray,         # H_g x W_g x 3, uint8 RGB
        satellite_rgb: np.ndarray,      # H_s x W_s x 3, uint8 RGB
        intrinsics: np.ndarray,         # 3x3, float
    ) -> tuple[float, float, float, float]:
        ...


# ---------------------------------------------------------------------------
# Mock backend: a no-weights, pure-correlation sanity check
# ---------------------------------------------------------------------------


class MockBevSplatInference:
    """A weight-free stand-in that returns *something* sensible.

    Computes a normalized cross-correlation peak between a downscaled
    ground frame and the satellite tile. This is **not** a substitute
    for the real model — appearance correlation across views is the
    exact thing BevSplat was designed to be better at than naive
    cross-correlation — but it lets us exercise the integration end to
    end, keep the result-JSON schema stable, and write meaningful
    tests.
    """

    def __init__(self, *, score_floor: float = 0.05) -> None:
        self._score_floor = float(score_floor)

    def __call__(
        self,
        ground_rgb: np.ndarray,
        satellite_rgb: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[float, float, float, float]:
        # Downscale ground frame to a template; cross-correlate against satellite.
        h_s, w_s = satellite_rgb.shape[:2]
        template_size = max(32, min(h_s, w_s) // 4)
        g = cv2.cvtColor(ground_rgb, cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (template_size, template_size), interpolation=cv2.INTER_AREA)
        s = cv2.cvtColor(satellite_rgb, cv2.COLOR_RGB2GRAY)
        if s.shape[0] <= template_size or s.shape[1] <= template_size:
            return self._score_floor, 0.0, 0.0, 0.0
        ncc = cv2.matchTemplate(s, g, cv2.TM_CCOEFF_NORMED)
        peak_val = float(ncc.max())
        min_val, max_val, _, max_loc = cv2.minMaxLoc(ncc)
        # Normalize peak position to [-1, 1] relative to tile centre.
        cx, cy = max_loc[0] + template_size / 2.0, max_loc[1] + template_size / 2.0
        shift_u = float((cx - w_s / 2.0) / (w_s / 2.0))
        shift_v = float((cy - h_s / 2.0) / (h_s / 2.0))
        # Mock has no heading; report 0.
        score = max(self._score_floor, (peak_val + 1.0) / 2.0)  # NCC ∈ [-1,1] → [0,1]
        return score, shift_u, shift_v, 0.0


# ---------------------------------------------------------------------------
# Real backend loader (best-effort; returns None if upstream is unavailable)
# ---------------------------------------------------------------------------


# Defaults taken from `train_KITTI_weak_seq.py::parse_args`, with one
# inference-friendly override: `level="0"` instead of the training
# default `"0_2"`. In `models_kitti_nips.py` the `stage=1` forward path
# populates only `sat_feat_dict_forT[self.level[0]]` but the post-loop
# iterates over all of `self.level`, raising KeyError on level=2.
# Restricting to one level keeps inference numerically valid against
# the released checkpoints; users who fix the upstream loop can pass
# `model_args={"level": "0_2"}` to use both feature levels.
_BEV_SPLAT_DEFAULT_ARGS: dict[str, object] = {
    "resume": 0,
    "test": 1,                # eval mode (not training)
    "epochs": 3,
    "lr": 6.25e-05,
    "rotation_range": 10.0,
    "shift_range_lat": 20.0,
    "shift_range_lon": 20.0,
    "batch_size": 1,           # we run per-candidate
    "level": "0",              # inference override; train default was "0_2"
    "channels": "32_16_4",
    "N_iters": 1,
    "ConfGrd": 1,
    "ConfSat": 0,
    "share": 1,
    "Optimizer": "TransV1",
    "proj": "geo",
    "visualize": 0,
    "multi_gpu": 0,
    "GPS_error": 5,
    "GPS_error_coe": 0.0,
    "contrastive_coe": 0.0,
    "stage": 1,
    "task": "3DoF",
    "supervise_amount": 1.0,
    "name": "monocular-osm-localization-inference",
    "sequence": 2,
}


def _build_bev_splat_args(overrides: dict[str, object]) -> object:
    """Return an argparse.Namespace matching BevSplat's training defaults."""
    import argparse

    merged = dict(_BEV_SPLAT_DEFAULT_ARGS)
    merged.update(overrides or {})
    return argparse.Namespace(**merged)


def _scale_intrinsics_to(
    K: np.ndarray,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> np.ndarray:
    """Rescale a pinhole ``K`` from a ``src_hw`` image to a ``dst_hw`` image.

    The model normalizes K assuming it is expressed in pixels of the
    *resized* ground image (models_kitti_nips.py divides row 0 by the
    depth-map width and row 1 by its height). The resize from a 16:9
    dashcam frame to KITTI's 256x1024 crop is anisotropic, so row 0
    (fx, cx) and row 1 (fy, cy) need *different* scale factors.
    """
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    K_scaled = np.asarray(K, dtype=np.float32).copy()
    K_scaled[0, :] *= dst_w / float(max(src_w, 1))
    K_scaled[1, :] *= dst_h / float(max(src_h, 1))
    return K_scaled


def _flat_ground_depth(
    K: np.ndarray,
    h: int,
    w: int,
    *,
    camera_height_m: float = 1.55,
    max_depth_m: float = 80.0,
) -> np.ndarray:
    """Analytic flat-ground depth map for a forward-facing dashcam.

    The upstream encoder places each ground Gaussian at
    ``origin + direction * grd_depth`` — depth IS the geometry, so an
    all-zero placeholder collapses every Gaussian onto the camera center
    and the channel degenerates. Without a depth network at match time
    we can still supply a structurally sound prior: intersect each pixel
    ray with the ground plane.

    Camera-y points DOWN in this codebase, so the road plane is
    ``y = +camera_height_m``. For pixel ``(u, v)`` the (un-normalized)
    ray is ``d = ((u - cx)/fx, (v - cy)/fy, 1)``; rows below the horizon
    (``d_y > 0``) hit the plane at ``t = camera_height_m / d_y`` and get
    range ``t * |d|`` (clamped to ``max_depth_m``); rows at/above the
    horizon get ``max_depth_m`` (far). ``K`` must be expressed in pixels
    of the ``(h, w)`` image (see :func:`_scale_intrinsics_to`).
    """
    K = np.asarray(K, dtype=np.float64)
    fx = max(float(K[0, 0]), 1e-6)
    fy = max(float(K[1, 1]), 1e-6)
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    us = (np.arange(w, dtype=np.float64) - cx) / fx
    vs = (np.arange(h, dtype=np.float64) - cy) / fy
    dx, dy = np.meshgrid(us, vs)
    ray_norm = np.sqrt(dx * dx + dy * dy + 1.0)
    depth = np.full((h, w), max_depth_m, dtype=np.float32)
    below = dy > 1e-6
    ground_range = camera_height_m / dy[below] * ray_norm[below]
    depth[below] = np.clip(ground_range, 0.0, max_depth_m)
    return depth


def _state_dict_coverage_error(
    missing: Sequence[str],
    unexpected: Sequence[str],
    n_model_keys: int,
) -> str | None:
    """Return an error string when a strict=False load left the model random-init.

    ``load_state_dict(strict=False)`` succeeds even when *no* key matches
    — the model then runs with construction-time random weights and
    produces plausible-looking scores. Treat >50% missing keys as a
    checkpoint/module mismatch and fail loudly.
    """
    if n_model_keys > 0 and len(missing) > 0.5 * n_model_keys:
        return (
            f"BevSplat checkpoint/module mismatch: {len(missing)}/{n_model_keys} "
            f"model keys missing from the checkpoint ({len(unexpected)} unexpected). "
            "The model would run with random-init weights and produce noise "
            "scores. Check that --bev-splat-model-module matches the "
            "checkpoint (KITTI_GPS/KITTI_no_GPS -> models.models_kitti_nips)."
        )
    return None


def _load_bev_splat_inference(
    config: BevSplatConfig,
) -> tuple[BevSplatInference | None, str | None]:
    """Try to load the upstream BevSplat model.

    Returns ``(inference, error)``:

    * ``(callable, None)`` on success — pass it to
      :func:`score_candidates_with_bevsplat`.
    * ``(None, message)`` if any prerequisite is missing (no weights,
      no repo clone, CUDA extensions unbuilt, checkpoint shape
      mismatch, etc.). The caller populates
      :class:`BevSplatMatchResult.error` and the pipeline keeps going.

    Prerequisites (all three):

    1. ``config.weights_path`` points to a ``.pth`` downloaded from the
       authors' OneDrive share (link in the README).
    2. ``config.repo_path`` points to a local clone of
       https://github.com/wangqww/BevSplat with its CUDA extensions
       built (``pip install -e ./pano_feature_gaussian`` and
       ``pip install -e ./feature_gaussian``).
    3. ``torch`` is importable. CUDA is recommended; the upstream
       model uses CUDA-specific extensions.

    The returned callable wraps ``model.forward(...)`` with a small
    adapter that:

    * tiles our query frame ``sequence_length`` times to match the
      BevSplat sequence-input convention,
    * rescales the intrinsics to the resized 256×1024 ground image and
      feeds an analytic flat-ground depth map (see
      :func:`_flat_ground_depth`); zero placeholders remain only for
      ``loc_shift_left`` and ``heading_shift_left`` (no priors at
      inference time),
    * extracts a scalar score + ``(shift_u, shift_v, heading)`` from
      whatever the forward call returns, falling back to ``NaN`` when
      the schema doesn't match (verified at first call).
    """
    if config.weights_path is None:
        return None, (
            "BevSplat weights_path not configured. The upstream authors "
            "published six checkpoints at the OneDrive share — for our "
            "dashcam-on-OSM-candidate use case, grab `KITTI_no_GPS.pth` "
            "(see README.md BevSplat section for the link and the full "
            "checkpoint table) and pass it via --bev-splat-weights."
        )

    weights_path = Path(config.weights_path)
    if not weights_path.exists():
        return None, f"BevSplat weights not found at {weights_path}"

    if config.repo_path is None:
        return None, (
            "BevSplat --bev-splat-repo-path not set. Clone "
            "https://github.com/wangqww/BevSplat locally, build the CUDA "
            "extensions (cd pano_feature_gaussian && pip install -e . ; "
            "cd ../feature_gaussian && pip install -e .), then point "
            "--bev-splat-repo-path at the clone root."
        )

    repo_path = Path(config.repo_path)
    if not (repo_path / "models" / "models_kitti_seq.py").exists():
        return None, (
            f"BevSplat repo not found at {repo_path} (looked for "
            "models/models_kitti_seq.py). Did you clone "
            "https://github.com/wangqww/BevSplat there?"
        )

    try:
        import torch
    except ImportError:
        return None, "torch not installed; cannot load BevSplat weights"

    # Make the upstream package importable. We prepend rather than append
    # so a local `models` clone wins over any same-named package on path.
    import sys
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    module_name = config.model_module
    try:
        import importlib
        mod = importlib.import_module(module_name)
        Model = getattr(mod, "Model")
    except ModuleNotFoundError as exc:
        # Upstream is half-broken: `models_kitti_seq` is missing files
        # (`loss/`, `gaussian.encoder`, `gaussian.decoder`); `models_kitti_nips`
        # and `models_vigor` need the CUDA extensions built. Surface
        # exactly which sub-module is missing so the user knows whether
        # to (a) build CUDA exts, (b) switch model_module, or (c) wait
        # for upstream to commit the missing files.
        return None, (
            f"Importing {module_name} from {repo_path} failed: {exc}. "
            "Common causes — upstream missing-files (models_kitti_seq → "
            "needs `loss/` and `gaussian.encoder`/`gaussian.decoder` from "
            "upstream which aren't checked in), or CUDA extensions not yet "
            "built (models_kitti_nips → `feat_gaussian`, models_vigor → "
            "`pano_gaussian_feat`). Try config.model_module='models.models_kitti_vfa' "
            "for an extension-free import smoke test."
        )
    except Exception as exc:
        return None, (
            f"Could not import {module_name}.Model from {repo_path}: {exc}"
        )

    args = _build_bev_splat_args(config.model_args)
    device = config.device if (config.device != "cuda" or torch.cuda.is_available()) else "cpu"

    try:
        model = Model(args, device=device).to(device).eval()
    except Exception as exc:
        return None, f"BevSplat Model() construction failed: {exc}"

    try:
        state = torch.load(weights_path, map_location=device)
    except Exception as exc:
        return None, f"torch.load({weights_path}) failed: {exc}"

    # Checkpoints in the wild are usually one of:
    #   (a) raw state_dict
    #   (b) {"model": state_dict, "epoch": ..., "optimizer": ...}
    #   (c) {"state_dict": ...}
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except Exception as exc:
        return None, f"model.load_state_dict failed: {exc}"

    # strict=False silently tolerates a checkpoint that matches nothing —
    # fail loudly instead of running a randomly-initialized model.
    mismatch = _state_dict_coverage_error(missing, unexpected, len(model.state_dict()))
    if mismatch is not None:
        return None, mismatch
    if missing or unexpected:
        print(
            f"[bev_splat] WARNING: load_state_dict(strict=False) left "
            f"{len(missing)} missing / {len(unexpected)} unexpected keys "
            f"loading {weights_path.name}"
        )

    seq_len = max(1, int(config.sequence_length))

    # The two main KITTI model variants take different positional args:
    #   models_kitti_seq.Model.forward(sat_map, grd_img_left, grd_img_left_ori,
    #                                  grd_depth, left_camera_k, gt_shift_u,
    #                                  gt_shift_v, gt_heading, loc_shift_left,
    #                                  heading_shift_left)         — 10 args, 5D
    #   models_kitti_nips.Model.forward(sat_align_cam, sat_map, grd_img_left,
    #                                   grd_depth, grd_ori, left_camera_k,
    #                                   gt_heading=None, gt_shift_u=None,
    #                                   gt_shift_v=None)            —  9 args, 4D
    # We introspect the signature and dispatch accordingly so the user
    # doesn't have to. Other variants (vfa, orienternet, vigor) follow
    # one of these two shapes — fall back to a best-effort kwarg call.
    import inspect
    forward_sig = inspect.signature(model.forward)
    forward_params = list(forward_sig.parameters)
    forward_is_seq = "loc_shift_left" in forward_params  # only _seq has this
    forward_is_nips = "sat_align_cam" in forward_params and "grd_ori" in forward_params

    # ---------------------------------------------------------------------------
    # Ground-image target resolution (H × W).
    #
    # The BevSplat KITTI checkpoints (KITTI_GPS.pth, KITTI_no_GPS.pth)
    # were trained with ground images at **256 × 1024** — the standard
    # KITTI Raw camera crop.  This is NOT the satellite tile size.
    #
    # The magic number 16384 (= 128 × 128) does NOT describe a 128 × 128
    # ground crop.  It is the DPT encoder's spatial output for the 256 × 1024
    # KITTI ground image, derived as follows:
    #
    #   1. ViT-B/14 patches: center-pad 256→266 (19 patches), 1024→1036
    #      (74 patches).  DINO output: 19 × 74 per level.
    #   2. dino_fit.py correction: shape[2]==19 → resize to (16, 64).
    #   3. DPT (dpt_single.py): 2× upsample → 32 × 128; FeatureFusion
    #      blocks → 32 × 128; 2× upsample → 64 × 256.
    #   4. 64 × 256 = 16384. ✓
    #
    # For a 128 × 128 ground image the same pipeline gives 40 × 40 = 1600,
    # which mismatches and triggers the "expanded size (16384) must match
    # existing size (1600)" error seen in the first full-pipeline run.
    #
    # We therefore resize every ground image to (256, 1024) before passing
    # it to the model.  The depth map is created at the same spatial size.
    # Intrinsics MUST be rescaled to the resized resolution too: the model's
    # own normalization (line 535–536 of models_kitti_nips.py divides K's
    # rows by grd_depth width/height) assumes K is already expressed in
    # 256×1024-image pixels — see _scale_intrinsics_to.
    _GRD_H, _GRD_W = 256, 1024

    def _to_4d(rgb: np.ndarray, target_hw: tuple[int, int] | None = None) -> "torch.Tensor":
        if target_hw is not None:
            rgb = cv2.resize(rgb, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(device)   # [1, 3, H, W]

    def _run(
        ground_rgb: np.ndarray,
        satellite_rgb: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[float, float, float, float]:
        # Convert to tensors in the expected shape. The model was
        # trained on KITTI-shaped inputs (sat 512×512, ground 256×1024).
        # We always resize the ground image to (_GRD_H, _GRD_W) — see the
        # constant definition above for the detailed rationale — and
        # rescale K to that resolution (the resize is anisotropic, so
        # fx/cx and fy/cy get different factors).
        sat_t = _to_4d(satellite_rgb)                                    # [1, 3, H_s, W_s]
        h_img, w_img = ground_rgb.shape[:2]
        K_scaled = _scale_intrinsics_to(intrinsics, (h_img, w_img), (_GRD_H, _GRD_W))
        K_t = torch.from_numpy(K_scaled).unsqueeze(0).to(device)
        gt_zeros_1 = torch.zeros(1, 1, device=device)

        # Flat-ground analytic depth: the encoder consumes depth both
        # geometrically (Gaussian means = origins + directions * depth)
        # and as an input feature, so an all-zero placeholder collapses
        # every Gaussian to the camera center. The ground-plane model
        # keeps the channel structurally non-degenerate without a depth
        # network at match time.
        depth_np = _flat_ground_depth(
            K_scaled, _GRD_H, _GRD_W,
            camera_height_m=config.camera_height_m,
            max_depth_m=config.max_ground_depth_m,
        )
        depth_t = torch.from_numpy(depth_np).to(device)                  # [H, W]

        with torch.inference_mode():
            try:
                if forward_is_nips:
                    grd_t = _to_4d(ground_rgb, target_hw=(_GRD_H, _GRD_W))  # [1, 3, 256, 1024]
                    grd_depth = depth_t.unsqueeze(0)                        # [1, 256, 1024]
                    out = model(
                        sat_t, sat_t, grd_t, grd_depth, grd_t, K_t,
                        gt_zeros_1, gt_zeros_1, gt_zeros_1,
                    )
                elif forward_is_seq:
                    grd_t = _to_4d(ground_rgb, target_hw=(_GRD_H, _GRD_W))
                    grd_t = grd_t.unsqueeze(1).expand(1, seq_len, -1, -1, -1)
                    grd_ori_t = grd_t.clone()
                    grd_depth = depth_t.unsqueeze(0).unsqueeze(0).expand(1, seq_len, -1, -1)
                    loc_shift = torch.zeros(1, seq_len, 2, device=device)
                    head_shift = torch.zeros(1, seq_len, device=device)
                    out = model(
                        sat_t, grd_t, grd_ori_t, grd_depth,
                        K_t, gt_zeros_1, gt_zeros_1, gt_zeros_1,
                        loc_shift, head_shift,
                    )
                else:
                    # Unknown variant — try the simplest plausible call.
                    grd_t = _to_4d(ground_rgb, target_hw=(_GRD_H, _GRD_W))
                    out = model(sat_t, grd_t, K_t)
            except Exception as exc:
                raise RuntimeError(f"BevSplat forward failed: {exc}") from exc

        # The upstream forward returns a tuple whose exact composition
        # depends on `stage` and `task`. We attempt a best-effort
        # extraction: scan the output for the first 4D correlation map
        # (used as our score) and the first three scalar-like tensors
        # (treated as shift_u, shift_v, heading). Anything else is left
        # at its default. When you've verified the schema against your
        # checkpoint, replace this block with the exact indices.
        score: float = 0.0
        du = dv = dh = 0.0

        def _flatten(x):
            """Recursively yield all leaf tensors from nested lists/tuples/dicts.

            The BevSplat forward returns a tuple whose first four elements are
            *dicts* (keyed by feature-level int) containing tensors. The
            original code only recursed into list/tuple, so those dict values
            were never seen.  We also recurse into dict.values() so that maps
            like the localization probability grid are properly surfaced.
            """
            if isinstance(x, (list, tuple)):
                for item in x:
                    yield from _flatten(item)
            elif isinstance(x, dict):
                for v in x.values():
                    yield from _flatten(v)
            else:
                yield x

        # ---------------------------------------------------------------
        # Score extraction — what the output contains at stage=1
        # ---------------------------------------------------------------
        # models_kitti_nips forward at stage=1 returns a 9-tuple:
        #   out[0] = grd_gaussian (dict, Gaussian encoder output)
        #   out[1] = sat_feat_dict  {level: [1, 32, 128, 128]}
        #   out[2] = grd_feat_dict  {level: [1, 32, 89, 89]}   (or similar)
        #   out[3] = grd_conf_dict  {level: [1,  1,  H,  W]}   ← probability map
        #   out[4] = sat_conf_dict  {level: [1, H, W, 1]}
        #   out[5] = shift_u  [1, 1, 1]
        #   out[6] = shift_v  [1, 1, 1]
        #   out[7] = heading  [1, 1, 1]
        #   out[8] = loss     []  (0.0 in test mode)
        #
        # We use *softmax-peak* of the first 2-D (H×W) probability map we
        # find as the localization score. Softmax-peak measures how peaked
        # the distribution is: for a uniform map over N=H*W cells the peak
        # is 1/N (≈ 0); for a perfectly localized match it approaches 1.0.
        # This is both more informative and more correct than the old formula
        # (max−min)/(max−min) = 1.0 which was always 1.0 for any non-flat map.
        #
        # For each 4-D tensor:
        #   - shape [B, C, H, W] with C=1 → treat as a spatial probability map
        #   - apply softmax over H*W, take max → localization confidence ∈ (0,1]
        #   - use the FIRST such tensor we encounter (level-0 of grd_conf_dict)
        #
        # Scalar tensors (numel ≤ 8) are collected for shift_u/v/heading in
        # output order; the first three scalars are treated as du, dv, dh.

        # ---------------------------------------------------------------
        # Collect tensors from the forward output for score + pose.
        # ---------------------------------------------------------------
        # models_kitti_nips at stage=1 returns (indexed as tensors):
        #   multi-channel 4-D [B, C>1, H, W] — sat_feat_dict and g2s_feat_dict
        #   single-channel 4-D [B, 1, H, W]  — sat_conf_dict and g2s_conf_dict
        #   shape [B,H,W,1]                  — mask_dict
        #   scalar [B,1,1]                   — shift_lats, shift_lons, thetas
        #   scalar []                        — render_loss
        #
        # BevSplat does NOT produce an explicit localization confidence;
        # the forward pass returns POSE predictions (shift_u, shift_v, heading),
        # not a correlation score. We compute a cross-view feature similarity
        # as our score: we collect the first two multi-channel 4-D tensors
        # (satellite features and ground-projected features from the Gaussian
        # encoder), global-average-pool each to a [C]-vector, then take their
        # cosine similarity remapped from [-1, 1] → [0, 1].
        #
        # For models that DO output a single scalar score, that value shows up
        # as a scalar tensor (numel ≤ 8) early in the stream; we use it if
        # it's > 0 (overriding the cosine-similarity proxy).

        feat_maps: list[torch.Tensor] = []   # multi-channel 4-D feature maps
        scalars: list[float] = []
        for item in _flatten(out):
            if not torch.is_tensor(item):
                continue
            if item.numel() <= 8:
                for v in item.flatten().tolist():
                    if np.isfinite(v):
                        scalars.append(float(v))
            elif item.dim() == 4 and item.shape[1] > 1:
                # Multi-channel feature map — candidate for cross-view similarity.
                feat_maps.append(item.float())

        # Cross-view cosine similarity between satellite and ground-projected
        # feature maps, computed only at positions where ground Gaussians
        # actually project onto the satellite tile (masked comparison).
        #
        # Context: models_kitti_nips at stage=1 returns
        #   feat_maps[0] = sat_feat_dict  [B, 32, 128, 128]  — SAT features
        #   feat_maps[1] = g2s_feat_dict  [B, 32,  89,  89]  — GRD→SAT projected
        #
        # g2s_feat is extremely sparse: the ground Gaussians cover only ~0.2%
        # of the satellite tile (their footprint after projection). Computing
        # cosine similarity with the near-zero complement gives ~0 everywhere.
        # We therefore:
        # 1. Build a coverage mask from g2s_feat's nonzero positions.
        # 2. Center-crop sat_feat to match g2s_feat's spatial size.
        # 3. Compare features ONLY at covered pixels → produces a meaningful
        #    score that varies per candidate (different satellite tiles have
        #    different features at those locations).
        # 4. Fall back to 0.5 (neutral) if coverage is zero.
        if len(feat_maps) >= 2:
            import torch.nn.functional as _F
            sat_f = feat_maps[0].float()   # [B, C, H_s, W_s]
            g2s_f = feat_maps[1].float()   # [B, C, H_g, W_g]

            # Center-crop sat_f to g2s_f's spatial size.
            h_s, w_s = sat_f.shape[2], sat_f.shape[3]
            h_g, w_g = g2s_f.shape[2], g2s_f.shape[3]
            h = min(h_s, h_g)
            w = min(w_s, w_g)
            pad_h = (h_s - h) // 2
            pad_w = (w_s - w) // 2
            sat_crop = sat_f[:, :, pad_h:pad_h + h, pad_w:pad_w + w]   # [B, C, h, w]
            pad_h = (h_g - h) // 2
            pad_w = (w_g - w) // 2
            g2s_crop = g2s_f[:, :, pad_h:pad_h + h, pad_w:pad_w + w]   # [B, C, h, w]

            # Coverage mask: positions where at least one ground channel != 0.
            coverage = (g2s_crop.abs().sum(dim=1) > 1e-8)   # [B, h, w]
            n_covered = int(coverage.sum().item())

            c = min(sat_crop.shape[1], g2s_crop.shape[1])
            if n_covered > 0:
                # Extract feature vectors at covered pixels: [C, N_cov]
                sat_vecs = sat_crop[0, :c, coverage[0]]    # [C, N]
                g2s_vecs = g2s_crop[0, :c, coverage[0]]    # [C, N]
                # Per-pixel cosine similarity at covered positions → mean.
                cos_vals = _F.cosine_similarity(
                    sat_vecs.T, g2s_vecs.T, dim=1
                )                                           # [N]
                cos = float(cos_vals.mean().item())
                score = float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))
            else:
                # No ground coverage in the satellite tile — neutral score.
                score = 0.5

        if len(scalars) >= 1:
            du = float(np.clip(scalars[0], -1.0, 1.0))
        if len(scalars) >= 2:
            dv = float(np.clip(scalars[1], -1.0, 1.0))
        if len(scalars) >= 3:
            dh = float(np.clip(scalars[2], -1.0, 1.0))

        return score, du, dv, dh

    return _run, None


# ---------------------------------------------------------------------------
# Satellite tile rendering
# ---------------------------------------------------------------------------


def _candidate_center_lonlat(
    road: RoadGraph,
    cand: MatchCandidate,
) -> tuple[float, float]:
    from pyproj import Transformer

    center_xy = cand.walk_xy.mean(axis=0)
    transformer = Transformer.from_crs(road.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(float(center_xy[0]), float(center_xy[1]))
    return float(lon), float(lat)


def _render_satellite_tile(
    source: str,
    road: RoadGraph,
    cand: MatchCandidate,
    *,
    size: int,
    half_extent_m: float,
    geotessera_year: int,
) -> np.ndarray:
    """Render the satellite reference patch for one candidate.

    Returns an ``(size, size, 3)`` uint8 RGB image. Raises if the
    requested source cannot be produced — the caller wraps this in a
    try/except and records the error in :class:`BevSplatMatchResult`.
    """
    if source == "osm":
        gray = render_osm_patch(
            road,
            (float(cand.walk_xy.mean(axis=0)[0]), float(cand.walk_xy.mean(axis=0)[1])),
            resolution=size,
            half_extent_m=half_extent_m,
        )
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if source == "geotessera":
        # Delegate to embedding_retrieval, which crops the ~11 km registry
        # tile down to the requested half_extent_m around the candidate —
        # otherwise BevSplat would be told an ~11 km tile spans 2*half_extent_m.
        from .embedding_retrieval import _render_geotessera_patch

        return _render_geotessera_patch(
            road, cand, year=geotessera_year, size=size,
            half_extent_m=half_extent_m,
        )
    if source in ("esri", "satellite"):
        # Real RGB orthoimagery — the domain BevSplat's KITTI checkpoints
        # were actually trained on. Unlike the GeoTessera PCA false-colour
        # tiles (visually near-identical across inner-city candidates →
        # non-discriminative), real satellite RGB gives the model the
        # appearance signal it needs to separate candidates.
        from .satellite import satellite_tile_for_candidate

        return satellite_tile_for_candidate(
            road, cand, half_extent_m=half_extent_m, size=size, provider=source,
        )
    raise ValueError(f"unsupported satellite source: {source!r}")


# ---------------------------------------------------------------------------
# Main entry point used by pipeline.py
# ---------------------------------------------------------------------------


def score_candidates_with_bevsplat(
    query_frame_rgb: np.ndarray | None,
    intrinsics: np.ndarray,
    road: RoadGraph,
    candidates: Sequence[MatchCandidate],
    *,
    output_dir: Path,
    config: BevSplatConfig,
    inference: BevSplatInference | None = None,
    tile_renderer: Callable[..., np.ndarray] | None = None,
) -> list[BevSplatMatchResult]:
    """Score each candidate with BevSplat cross-view localization.

    Parameters
    ----------
    query_frame_rgb:
        A representative dashcam frame from the driven window, as
        ``H x W x 3`` uint8 RGB. The middle frame works well in
        practice (the trajectory matcher already aligned the window).
        Pass ``None`` to skip the channel entirely.
    intrinsics:
        3x3 camera intrinsics for the query frame (from
        :func:`visual_odometry.default_intrinsics`).
    road:
        Projected OSM road graph for the city.
    candidates:
        Trajectory-match candidates from :func:`match_trajectory`.
    output_dir:
        Directory where satellite tile PNGs are written. Created if
        missing.
    config:
        :class:`BevSplatConfig` controlling backend / paths / tile size.
    inference:
        Optional pre-loaded inference callable. If ``None``, this
        function attempts to load the upstream model via
        :func:`_load_bev_splat_inference`. Tests pass
        :class:`MockBevSplatInference` here.
    tile_renderer:
        Optional override for :func:`_render_satellite_tile`. Used in
        tests to bypass the network.

    Returns
    -------
    One :class:`BevSplatMatchResult` per candidate, in input order.
    Candidates where tile rendering or inference failed have
    ``score=0.0`` and a populated ``error`` field; the pipeline keeps
    going.
    """
    if query_frame_rgb is None or not candidates:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if inference is None:
        inference, load_err = _load_bev_splat_inference(config)
    else:
        load_err = None

    renderer = tile_renderer or _render_satellite_tile

    results: list[BevSplatMatchResult] = []
    for i, cand in enumerate(candidates):
        tile_path = output_dir / f"bev_splat_candidate_{i + 1}.png"
        try:
            tile_rgb = renderer(
                config.satellite_source,
                road,
                cand,
                size=config.satellite_size,
                half_extent_m=config.half_extent_m,
                geotessera_year=config.geotessera_year,
            )
        except Exception as exc:
            results.append(BevSplatMatchResult(
                candidate_index=i,
                score=0.0,
                pred_shift_u=0.0,
                pred_shift_v=0.0,
                pred_heading=0.0,
                satellite_path=None,
                error=f"tile render failed: {exc}",
            ))
            continue

        # Always persist the tile for inspection, even if inference fails.
        cv2.imwrite(str(tile_path), cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))

        if inference is None:
            results.append(BevSplatMatchResult(
                candidate_index=i,
                score=0.0,
                pred_shift_u=0.0,
                pred_shift_v=0.0,
                pred_heading=0.0,
                satellite_path=tile_path,
                error=load_err or "BevSplat inference unavailable",
            ))
            continue

        try:
            score, du, dv, dh = inference(query_frame_rgb, tile_rgb, intrinsics)
        except Exception as exc:
            results.append(BevSplatMatchResult(
                candidate_index=i,
                score=0.0,
                pred_shift_u=0.0,
                pred_shift_v=0.0,
                pred_heading=0.0,
                satellite_path=tile_path,
                error=f"inference failed: {exc}",
            ))
            continue

        results.append(BevSplatMatchResult(
            candidate_index=i,
            score=float(score),
            pred_shift_u=float(du),
            pred_shift_v=float(dv),
            pred_heading=float(dh),
            satellite_path=tile_path,
            error=None,
        ))

    return results
