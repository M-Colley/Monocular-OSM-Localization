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
from .context import _record_channel, RunContext
from .length_prior import _length_sane_here

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


def _plot_xy(xy: np.ndarray, out_path: Path, title: str) -> None:
    """Minimal top-down polyline plot (used for the DA3 trajectory)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(xy[:, 0], xy[:, 1], "-", color="C2", linewidth=1.4)
    ax.scatter(xy[0, 0], xy[0, 1], color="green", s=60, label="start", zorder=5)
    ax.scatter(xy[-1, 0], xy[-1, 1], color="red", s=60, label="end", zorder=5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _pose_txt_xz(path) -> np.ndarray | None:
    """Top-down ``(tx, tz)`` path from a pose text file: KITTI 3x4 rows
    (12 columns) or flattened 4x4 C2W rows (16 columns, VGGT-Long's
    ``camera_poses.txt``). In both row-major layouts the translation sits at
    columns 3/7/11, so the ground projection is columns ``[3, 11]``.
    Returns ``None`` for unreadable/degenerate files."""
    try:
        P = np.loadtxt(str(path))
        if P.ndim != 2 or P.shape[1] not in (12, 16):
            return None
        xz = P[:, [3, 11]].astype(np.float64)
        if len(xz) < 2 or not bool(np.isfinite(xz).all()):
            return None
        return xz
    except Exception:
        return None


def _da3_trajectory_plausible(xy: np.ndarray) -> bool:
    """Reject DA3 camera paths that look like pose-estimation failures.

    DA3 needs visually-overlapping keyframes; on sparse keyframes (long
    clips) its solver can return near-random poses. A real driven path
    almost never reverses heading by >120° between consecutive keyframe
    segments, while a failed solve zigzags constantly — on the Ulm clip
    a failed 48-keyframe solve had ~40% reversals and silently produced
    a 2.3 km localization error. Threshold 0.15 leaves room for genuine
    U-turns and noisy stationary segments.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 3:
        return False
    seg = np.diff(xy, axis=0)
    norms = np.linalg.norm(seg, axis=1)
    moving = norms > 1e-9
    if moving.sum() < 2:
        return False
    d = seg[moving] / norms[moving][:, None]
    cos_between = (d[:-1] * d[1:]).sum(axis=1)
    reversal_ratio = float((cos_between < -0.5).mean())
    return reversal_ratio <= 0.15


def _match_timestamps(frame_timestamps, n_rows: int) -> np.ndarray:
    """Timestamp axis aligned to the matcher trajectory's rows.

    A staged trajectory (OpenVO / DA3 / MapAnything) has FEWER rows than
    extracted frames (Ulm: 1260 poses vs 4196 frames), so indexing it
    with per-frame timestamps is out of bounds. Spread the clip's frame
    time span uniformly over the pose rows instead.
    """
    ts = np.asarray(frame_timestamps, dtype=np.float64)
    if len(ts) == 0:
        return np.zeros(n_rows, dtype=np.float64)
    if n_rows == len(ts):
        return ts
    return np.linspace(ts[0], ts[-1], n_rows)


def _remap_frame_pair_to_poses(
    pair, n_frames: int, n_poses: int
):
    """Map a loop-closure (i, j) frame-index pair into pose-index space.

    ``detect_end_to_start_loop`` indexes the extracted frames, but the
    drift redistribution acts on the matcher trajectory, which has a
    different row count whenever a staged (OpenVO/DA3/MapAnything) path
    is in use. Scale proportionally; return ``None`` when the remapped
    pair is degenerate (so the caller skips instead of warping poses).
    """
    if pair is None:
        return None
    i, j = int(pair[0]), int(pair[1])
    if n_frames == n_poses:
        return (i, j) if 0 <= i < j < n_poses else None
    if n_frames < 2 or n_poses < 2:
        return None
    scale = (n_poses - 1) / (n_frames - 1)
    ip = int(round(i * scale))
    jp = int(round(j * scale))
    if not (0 <= ip < jp < n_poses):
        return None
    return (ip, jp)


def select_matcher_trajectory(ctx: RunContext) -> None:
    """Choose the trajectory the shape matcher consumes -> ``ctx.match_xz`` / ``ctx.match_ts``.

    The VO ``traj`` stays for the splat/IPM renders (tied to the per-frame VO
    poses); only the matcher's input is swapped here, by whichever staged
    source wins: MapAnything, a staged OpenVO path, VGGT-Long, or the
    drift-free Depth Anything 3 camera path.
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    video_path = ctx.video_path
    frames = ctx.frames
    traj = ctx.traj

    # 4b. Optional: use Depth Anything 3's globally-consistent camera path
    # as the trajectory the shape matcher consumes. Monocular VO drifts
    # over long clips (the true Olgastraße route's bearing correlation was
    # only 0.16 vs 0.29–0.32 for *wrong* parallel streets — pure drift),
    # which lets several parallel streets fit equally well. DA3 is metric
    # and multi-frame-consistent, so its trajectory has far less drift.
    # We keep the VO `traj` for the splat/IPM renders (those are tied to
    # the per-frame VO poses) and only swap the matcher's input here.
    ctx.match_xz = traj.xz
    _staged_traj_source = "vo"     # relabelled by whichever staging block wins
    if cfg.use_mapanything_trajectory:
        print("[4b'] MapAnything submap-stitched trajectory (matcher input)")
        from ..mapanything_trajectory import mapanything_trajectory_xy
        end_s = cfg.vo_end_sec if cfg.vo_end_sec else (cfg.vo_start_sec + 420.0)
        ma = mapanything_trajectory_xy(str(video_path), cfg.vo_start_sec, end_s)
        if ma is not None and len(ma[0]) >= 2 and bool(np.isfinite(ma[0]).all()):
            ctx.match_xz = ma[0]
            _staged_traj_source = "mapanything"
            _plot_xy(ctx.match_xz, cfg.output_dir / "trajectory_mapanything.png",
                     "MapAnything submap-stitched trajectory (matcher input)")
            print(f"      -> using MapAnything trajectory: {len(ctx.match_xz)} stitched poses")
        else:
            print("      -> MapAnything unavailable/invalid; keeping VO path")
    # OpenVO is the default matcher trajectory when one is staged for this clip
    # (explicit --openvo-trajectory, else <data_dir>/openvo_trajectory.txt). It is
    # intrinsic-free metric dashcam VO with lower drift than our frame-to-frame VO
    # (196 vs 241 m global-fit RMS on Ulm). Poses are KITTI 3x4 rows; the top-down
    # ground projection is (tx, tz) = cols 3,11. Falls back to VO if none present.
    ov_path = cfg.openvo_trajectory_path
    if ov_path is None and cfg.prefer_openvo_trajectory:
        cand = cfg.data_dir / "openvo_trajectory.txt"
        if cand.exists():
            ov_path = cand
    if ov_path is not None:
        print(f"[4b''] OpenVO metric trajectory (matcher input): {ov_path}")
        ov = _pose_txt_xz(ov_path)
        if ov is not None:
            ctx.match_xz = ov
            _staged_traj_source = "openvo"
            _plot_xy(ov, cfg.output_dir / "trajectory_openvo.png",
                     "OpenVO metric trajectory (matcher input)")
            print(f"      -> using OpenVO trajectory: {len(ov)} poses")
        else:
            print("      -> OpenVO trajectory invalid/unreadable; keeping VO path")
    # VGGT-Long chunked-VGGT trajectory (ICRA'25): explicit opt-in, takes
    # precedence over the staged OpenVO default when given. ~30% lower
    # Procrustes shape RMS than OpenVO on KITTI 0033 (148.6 vs 211.0 m,
    # scripts/test_vggt_long_kitti.py) — but better shape has NOT implied
    # better end-to-end before (MapAnything), hence flag-gated for A/B.
    if cfg.vggt_long_trajectory_path is not None:
        print(f"[4b'''] VGGT-Long trajectory (matcher input): "
              f"{cfg.vggt_long_trajectory_path}")
        vl = _pose_txt_xz(cfg.vggt_long_trajectory_path)
        if vl is not None:
            ctx.match_xz = vl
            _staged_traj_source = "vggt_long"
            _plot_xy(vl, cfg.output_dir / "trajectory_vggt_long.png",
                     "VGGT-Long trajectory (matcher input)")
            print(f"      -> using VGGT-Long trajectory: {len(vl)} poses")
        else:
            print("      -> VGGT-Long trajectory invalid/unreadable; "
                  "keeping prior path")
    ctx.da3_rec = None
    ctx.result_scale_recovery = None   # set by DA3 (4b) or anchors (4c)
    if cfg.use_da3_trajectory:
        print("[4b] Depth Anything 3 trajectory (drift-free matcher input)")
        from ..da3_reconstruction import da3_trajectory_xy, reconstruct_with_da3
        try:
            ctx.da3_rec = reconstruct_with_da3(
                frames.frames,
                n_keyframes=cfg.da3_keyframes,
                valid_mask=traj.valid,
                device="cuda",
            )
            da3_xy = da3_trajectory_xy(ctx.da3_rec)
            da3_valid = len(da3_xy) >= 2 and bool(np.isfinite(da3_xy).all())
            if da3_valid and not _da3_trajectory_plausible(da3_xy):
                # A garbage path scores walks all over the city and is
                # strictly worse than drifty-but-coherent VO.
                print("      -> DA3 trajectory implausible (heading reverses "
                      "constantly — pose solve likely failed on sparse "
                      "keyframes); falling back to VO path")
                _plot_xy(
                    da3_xy, cfg.output_dir / "trajectory_da3_rejected.png",
                    "DA3 trajectory (REJECTED: implausible)",
                )
                da3_valid = False
            if da3_valid:
                ctx.match_xz = da3_xy
                _plot_xy(
                    da3_xy, cfg.output_dir / "trajectory_da3.png",
                    "DA3 globally-consistent trajectory (matcher input)",
                )
                print(f"      -> using DA3 trajectory: {len(da3_xy)} keyframe poses")
                ctx.result_traj_source = "da3"
                # Idea 4: DA3 is metric, so its arc length IS the route
                # length — adopt it as the prior when it sanity-checks
                # against the duration estimate.
                from ..visual_odometry import trajectory_arc_length as _arclen
                da3_len = float(_arclen(da3_xy)[-1])
                if _length_sane_here(ctx, da3_len):
                    print(f"      -> DA3 metric length {da3_len:.0f} m adopted as "
                          f"route prior (was {ctx.estimated_length_m:.0f} m)")
                    ctx.estimated_length_m = float(np.clip(da3_len, 500.0, 12000.0))
                    ctx.result_scale_recovery = {
                        "status": "ok", "source": "da3",
                        "estimated_length_m": round(ctx.estimated_length_m, 1),
                    }
                else:
                    print(f"      -> DA3 metric length {da3_len:.0f} m rejected "
                          f"(sanity vs prior {ctx.estimated_length_m:.0f} m)")
            else:
                if len(da3_xy) < 2 or not bool(np.isfinite(da3_xy).all()):
                    print("      -> DA3 trajectory invalid; falling back to VO path")
                ctx.result_traj_source = _staged_traj_source
        except Exception as e:
            print(f"      -> DA3 trajectory failed ({e}); falling back to VO path")
            _record_channel(channel_status, 'da3_trajectory', "failed", error=e)
            ctx.da3_rec = None
            ctx.result_traj_source = _staged_traj_source
    else:
        ctx.result_traj_source = _staged_traj_source
    # Timestamp axis aligned to the matcher trajectory. The staged
    # OpenVO/DA3/MapAnything paths have FEWER rows than extracted frames
    # (Ulm: 1260 poses vs 4196 frames), so anything that maps a time to
    # a match_xz row must use this axis, not frames.timestamps.
    ctx.match_ts = _match_timestamps(frames.timestamps, len(ctx.match_xz))


def close_loop(ctx: RunContext) -> None:
    """Optional loop-closure drift correction of ``ctx.match_xz``.

    On a route that returns near its start, monocular VO leaves a large
    end-start gap (pure drift) that warps the whole shape. If the last frames
    verifiably revisit the first, redistribute the gap back along the trajectory.
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    frames = ctx.frames

    # 4b2. Optional: loop-closure drift correction. On a route that returns
    # near its start, monocular VO leaves a large end-start gap (pure
    # drift) that warps the whole shape — the dominant geometric error.
    # If the last frames verifiably revisit the first, redistribute the
    # gap back along the trajectory. Pair with a real metric scale
    # (--use-ipm-scale): closing the loop at a wrong scale doesn't help.
    ctx.result_loop_closure = None
    if cfg.enable_loop_closure:
        from ..loop_closure import detect_end_to_start_loop, redistribute_drift
        from ..visual_odometry import trajectory_arc_length as _arclen2
        print("[4b2] Loop-closure drift correction")
        try:
            frame_pair = detect_end_to_start_loop(frames.frames)
            # The detector indexes the extracted FRAMES; the drift fix acts
            # on match_xz POSES, which is a different (shorter) axis when a
            # staged trajectory is in use — remap proportionally.
            pair = _remap_frame_pair_to_poses(
                frame_pair, len(frames.frames), len(ctx.match_xz))
            if pair is not None:
                i, j = pair
                gap = float(np.linalg.norm(ctx.match_xz[j] - ctx.match_xz[i]))
                arclen = float(_arclen2(ctx.match_xz)[-1])
                ctx.match_xz = redistribute_drift(ctx.match_xz, i, j)
                ctx.result_loop_closure = {"closed": True, "i": int(i), "j": int(j),
                                           "frame_i": int(frame_pair[0]),
                                           "frame_j": int(frame_pair[1]),
                                           "gap_units": round(gap, 2)}
                print(f"      -> loop detected (frames {frame_pair[0]}↔"
                      f"{frame_pair[1]} -> poses {i}↔{j}); closed "
                      f"{gap:.1f}-unit gap ({100 * gap / max(arclen, 1e-6):.0f}% of arc)")
            else:
                ctx.result_loop_closure = {"closed": False}
                print("      -> no verifiable loop (end does not revisit start)")
        except Exception as e:
            print(f"      -> loop-closure failed ({e}); using VO path unchanged")
            _record_channel(channel_status, 'loop_closure', "failed", error=e)
