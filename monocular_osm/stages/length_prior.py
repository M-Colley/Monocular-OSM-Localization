"""Stage 2b/2c: the route-length prior for OSM walk enumeration.

The duration-based estimate is the *stable sanity reference* that every
(noisier) scale source - IPM flow (2c), DA3 (4b), OCR anchors (4c) - is gated
against.
"""

from __future__ import annotations

import numpy as np

from ..visual_odometry import default_intrinsics
from .context import _record_channel, RunContext


# Urban average driving speed including traffic-light stops. The Ulm GT
# track averaged 5.0 m/s point-to-point; the driven path is a bit longer
# than the straight segments between fixes, so 5.5 m/s is a reasonable
# city-wide prior.
_URBAN_AVG_SPEED_MPS = 5.5


def _auto_estimated_length_m(duration_sec: float) -> float:
    """Route-length prior from the analyzed duration at urban speed.

    Clamped to [500, 12000]: below 500 m walk enumeration degenerates
    (every micro-walk matches anything), above 12 km the walk depth cap
    truncates walks anyway.
    """
    return float(np.clip(duration_sec * _URBAN_AVG_SPEED_MPS, 500.0, 12000.0))


def _length_sane(
    candidate_m: float,
    duration_prior_m: float,
    user_length_m: float | None = None,
) -> bool:
    """A recovered length is trustworthy if it's within 2x of the
    duration-based estimate (the robust reference) — or, when the user
    supplied an explicit ``--estimated-length-m``, within 2x of THAT.

    Without the user gate, a correct recovered length on a fast
    (highway) clip is rejected against the conservative 5.5 m/s prior
    even though the user already told us the true extent.
    """
    refs = [duration_prior_m]
    if user_length_m is not None and user_length_m > 0:
        refs.append(float(user_length_m))
    return any(0.5 * r <= candidate_m <= 2.0 * r for r in refs)


def _length_sane_here(ctx: RunContext, candidate_m: float) -> bool:
    """A recovered length is trustworthy only if it's within 2x of the
    duration-based estimate - or of an explicit user-provided
    ``--estimated-length-m`` (see module-level ``_length_sane``)."""
    return _length_sane(candidate_m, ctx.duration_prior_m,
                        user_length_m=ctx.cfg.estimated_length_m)


def estimate_length_prior(ctx: RunContext) -> None:
    """Duration-based route-length prior -> ``ctx.duration_prior_m`` / ``ctx.estimated_length_m``."""
    cfg = ctx.cfg
    frames = ctx.frames
    # 2b. Route-length prior for OSM walk enumeration. The
    # duration-based estimate is the *stable sanity reference* that every
    # (noisier) scale source — IPM flow (2c), DA3 (4b), anchors (4c) — is
    # gated against. Gating each source against this fixed baseline
    # rather than against the running prior avoids one unreliable source
    # opening the gate for another (e.g. IPM flow lowering the prior so a
    # wrong anchor scale then passes).
    analyzed_sec = (
        frames.timestamps[-1] - frames.timestamps[0]
        if len(frames.timestamps) >= 2 else 0.0
    )
    ctx.duration_prior_m = _auto_estimated_length_m(analyzed_sec)
    if cfg.estimated_length_m is not None:
        ctx.estimated_length_m = float(cfg.estimated_length_m)
    else:
        ctx.estimated_length_m = ctx.duration_prior_m
        print(f"      -> estimated route length: {ctx.estimated_length_m:.0f} m "
              f"(auto: {analyzed_sec:.0f} s at urban avg speed; "
              f"override with --estimated-length-m)")


def estimate_flow_scale(ctx: RunContext) -> None:
    """Idea 3 (opt-in ``--use-ipm-scale``): ground-plane optical-flow metric scale.

    Sets the length prior from road-feature motion, sanity-gated against the
    duration estimate. Off by default - unreliable without real camera calibration.
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    frames = ctx.frames
    duration_prior_m = ctx.duration_prior_m
    # 2c. Idea 3: ground-plane optical-flow metric scale (opt-in). Sets
    # the length prior from road-feature motion, sanity-gated against the
    # duration estimate. Off by default — unreliable without real camera
    # calibration.
    if cfg.use_ipm_scale:
        print("[2c] Ground-plane optical-flow scale")
        try:
            from ..speed_scale import estimate_route_length_from_flow
            h_img, w_img = frames.frames[0].shape[:2]
            K_flow = default_intrinsics(w_img, h_img)
            # Subsample to ~3 fps so the flow pass is quick.
            step = max(1, int(round(frames.fps / 3.0)) // max(1, cfg.frame_stride))
            sub = frames.frames[::step] if step > 1 else frames.frames
            flow_len, motions = estimate_route_length_from_flow(
                sub, K_flow, camera_height_m=cfg.ipm_scale_camera_height_m,
                pitch_deg=cfg.ipm_scale_pitch_deg, fps=frames.fps,
                frame_stride=cfg.frame_stride * step,
            )
            # IPM flow is a *physical measurement* (optical flow + camera
            # height), so trust it on a looser bound than the generic 2x
            # sanity gate: the duration prior uses a conservative 5.5 m/s,
            # so a faster drive legitimately measures ~2x it (KITTI 0033:
            # true 1705 m vs 876 m prior = 1.95x), which the 2x gate would
            # wrongly reject. Accept within [0.3x, 3x] of the prior.
            if flow_len > 0 and 0.3 * duration_prior_m <= flow_len <= 3.0 * duration_prior_m:
                print(f"      -> flow length {flow_len:.0f} m adopted (was "
                      f"{ctx.estimated_length_m:.0f} m)")
                ctx.estimated_length_m = float(np.clip(flow_len, 500.0, 12000.0))
                ctx.result_ipm_scale = {"status": "ok", "length_m": round(flow_len, 1)}
            else:
                # Report the duration_prior_m the gate ACTUALLY compares
                # against, not estimated_length_m (they differ when the length
                # prior was overridden — printing the wrong one misleads scale
                # debugging).
                print(f"      -> flow length {flow_len:.0f} m rejected (sanity vs "
                      f"prior {duration_prior_m:.0f} m); keeping prior")
                ctx.result_ipm_scale = {"status": "rejected", "length_m": round(flow_len, 1)}
        except Exception as e:
            print(f"      -> IPM-scale failed: {e}")
            _record_channel(channel_status, 'ipm_scale', "failed", error=e)
            ctx.result_ipm_scale = {"status": "error", "error": str(e)}
    else:
        ctx.result_ipm_scale = None
