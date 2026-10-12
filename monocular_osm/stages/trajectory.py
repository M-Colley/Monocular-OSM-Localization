"""Stage 3-4b2: visual odometry, the matcher's trajectory, loop closure."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless: do not try to open a window
import matplotlib.pyplot as plt
import numpy as np

from ..visual_odometry import (
    default_intrinsics,
    estimate_trajectory,
    Trajectory,
    VO_ALGO_VERSION,
)
from .context import RunContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..pipeline import PipelineConfig


def _apply_ground_flow_scale(traj: Trajectory, video_path: Path, frames,
                             cfg: "PipelineConfig") -> Trajectory:
    """Give the trajectory a real speed profile, or leave it alone.

    Only ``xz`` is rescaled — that is what the matcher scores. ``centers``
    and the per-frame (R, t) keep the VO's unit-step convention, because the
    3D consumers (splat, triangulation) carry their own scale handling and
    rescaling them here would be an unmeasured change.

    Never fatal: this is one optional channel, and failing to read the road
    plane must not cost the run its trajectory.
    """
    from ..ground_flow_scale import rescale_steps, step_lengths, track_road_features

    if len(traj.xz) < 3:
        return traj
    try:
        h, w = frames.frames[0].shape[:2]
        print(f"[3b/5] Ground-flow scale: tracking road features "
              f"(stride={cfg.frame_stride})")
        tracks = track_road_features(
            video_path, stride=cfg.frame_stride, start_sec=cfg.vo_start_sec,
            end_sec=cfg.vo_end_sec, size=(w, h))
        unit = np.linalg.norm(np.diff(traj.xz, axis=0), axis=1)
        lengths, confidence = step_lengths(
            tracks, default_intrinsics(w, h),
            step_directions_valid=unit > 1e-9, fallback_lengths=unit)
    except Exception as e:                        # optional channel, never fatal
        print(f"      -> unavailable ({type(e).__name__}: {e}); keeping unit steps")
        return traj

    if confidence <= 0.0:
        print("      -> abstained: the road plane does not explain this footage; "
              "trajectory unchanged")
        return traj
    moving = lengths > 0
    cv = (float(lengths[moving].std() / max(lengths[moving].mean(), 1e-9))
          if moving.any() else 0.0)
    print(f"      -> confidence {confidence:.2f}, step-length CV {cv:.2f} "
          f"(unit steps are ~0.02-0.07 by construction)")
    traj.xz = rescale_steps(traj.xz, lengths)
    return traj


def _plot_trajectory(traj: Trajectory, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(traj.xz[:, 0], traj.xz[:, 1], "-", color="C0", linewidth=1.4)
    ax.scatter(traj.xz[0, 0], traj.xz[0, 1], color="green", s=60, label="start", zorder=5)
    ax.scatter(traj.xz[-1, 0], traj.xz[-1, 1], color="red", s=60, label="end", zorder=5)
    ax.set_aspect("equal")
    ax.set_title("Recovered top-down trajectory (VO, scale-free)")
    ax.set_xlabel("x (arbitrary units)")
    ax.set_ylabel("y (arbitrary units)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _vo_cache_candidates(cfg: "PipelineConfig"):
    """(canonical path, loadable same-version siblings, stale caches) for VO.

    Key: ``trajectory_v2_<s>-<e>_s<stride>_f<max_frames>_vo<N>.npz`` with
    N = :data:`visual_odometry.VO_ALGO_VERSION`. Only files carrying the
    current version are loadable; a sibling with another ``max_frames``
    suffix still qualifies when its frame count matches (checked by
    :func:`_find_vo_cache`). Nothing outside ``cfg.data_dir`` is considered.
    """
    seg_key = (f"trajectory_v2_{cfg.vo_start_sec:.0f}-{cfg.vo_end_sec or 'end'}"
               f"_s{cfg.frame_stride}")
    suffix = f"_vo{VO_ALGO_VERSION}"
    canonical = cfg.data_dir / f"{seg_key}_f{cfg.max_frames or 'auto'}{suffix}.npz"
    siblings = sorted(q for q in cfg.data_dir.glob(f"{seg_key}_f*{suffix}.npz")
                      if q != canonical)
    stale = sorted(q for q in cfg.data_dir.glob(f"{seg_key}*.npz")
                   if not q.stem.endswith(suffix))
    return canonical, siblings, stale


def _find_vo_cache(candidates: list, n_frames: int):
    """First VO cache file whose frame count matches, or ``None``.

    Every candidate — including the canonical key — must PASS the shape
    check; a stale canonical cache from a different stream (fps change on
    re-download) must be recomputed, not loaded. Probes are opened with a
    context manager so no npz handle stays open (an open handle blocks
    deleting/renaming the file on Windows).
    """
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with np.load(candidate) as z_probe:
                shape_ok = z_probe["valid"].shape[0] == n_frames
        except Exception:
            continue
        if shape_ok:
            return candidate
    return None


def run_visual_odometry(ctx: RunContext) -> None:
    """Visual odometry (cached to disk) -> ``ctx.traj`` / ``ctx.n_valid`` / ``ctx.vo_cache``."""
    cfg = ctx.cfg
    video_path = ctx.video_path
    frames = ctx.frames

    # 3. Visual odometry. Cache to disk: VO is the slowest CPU stage of
    # the pipeline (minutes on a 7-minute clip), and the recovered
    # trajectory is fully determined by (video, segment, stride), so a
    # re-run with the same VO parameters can short-circuit straight to
    # step 5.
    #
    # Cache key: trajectory_v2_<s>-<e>_s<stride>_f<max_frames>_vo<N>.npz,
    # where N is visual_odometry.VO_ALGO_VERSION. The version is part of the
    # key because a cache is only valid for the code that wrote it: without
    # it, re-runs silently kept matching trajectories from before the
    # mirroring and burned-in-graphics fixes. Pre-version caches (no _vo<N>)
    # and the old flat-layout fallback at the data-dir PARENT are never
    # loaded — the latter could hand one clip another clip's trajectory
    # whenever their frame counts matched.
    vo_cache_canonical, sibling_caches, stale = _vo_cache_candidates(cfg)
    if stale:
        print(f"[3/5] Ignoring {len(stale)} VO cache(s) written by an older VO "
              f"(current VO_ALGO_VERSION={VO_ALGO_VERSION}): "
              f"{', '.join(q.name for q in stale[:3])}")

    # Every candidate — including the canonical key — must PASS the
    # frame-count shape check before it may be loaded. A stale canonical
    # cache (same key, different stream fps after a re-download) used to
    # slip through and misassociate poses with frames downstream.
    ctx.vo_cache = _find_vo_cache([vo_cache_canonical, *sibling_caches],
                                  len(frames.frames))

    if ctx.vo_cache is not None:
        print(f"[3/5] Loading cached trajectory: {ctx.vo_cache}")
        with np.load(ctx.vo_cache) as z:
            ctx.traj = Trajectory(
                centers=z["centers"],
                xz=z["xz"],
                valid=z["valid"],
                n_inliers=z["n_inliers"].tolist(),
                rotations=z["rotations"],
                translations=z["translations"],
            )
        ctx.n_valid = int(ctx.traj.valid.sum())
        print(f"      -> {ctx.n_valid}/{len(ctx.traj.valid)} valid relative poses, "
              f"trajectory shape {ctx.traj.xz.shape}")
    else:
        ctx.vo_cache = vo_cache_canonical
        print(f"[3/5] Running visual odometry on {len(frames.frames)} frames "
              f"(n_workers={cfg.vo_workers or 'auto'})")
        ctx.traj = estimate_trajectory(frames.frames, n_workers=cfg.vo_workers)
        ctx.n_valid = int(ctx.traj.valid.sum())
        print(f"      -> {ctx.n_valid}/{len(ctx.traj.valid)} valid relative poses, "
              f"trajectory shape {ctx.traj.xz.shape}")
        ctx.vo_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            ctx.vo_cache,
            centers=ctx.traj.centers,
            xz=ctx.traj.xz,
            valid=ctx.traj.valid,
            n_inliers=np.asarray(ctx.traj.n_inliers),
            rotations=ctx.traj.rotations,
            translations=ctx.traj.translations,
            vo_algo_version=np.int64(VO_ALGO_VERSION),
        )
    if cfg.use_ground_flow_scale:
        ctx.traj = _apply_ground_flow_scale(ctx.traj, video_path, frames, cfg)
    _plot_trajectory(ctx.traj, cfg.output_dir / "trajectory.png")
