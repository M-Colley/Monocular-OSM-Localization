"""Shared state of one pipeline run.

``run_pipeline`` used to be a single ~2,500-line function whose stages shared
dozens of function-local variables. Each stage now lives in its own function in
this package and receives a :class:`RunContext` holding exactly the values that
cross a stage boundary (everything else stays local to the stage that computes
it). A stage reads what earlier stages produced, extends the context with its
own outputs, and records a failed or inactive optional channel through
:func:`_record_channel`, so that ``result["channels"]`` can say what ran.

Stage code follows one convention: a value the stage only *reads* is bound to a
local alias at the top (``road = ctx.road``); a value the stage *assigns* is
always written as ``ctx.<name>``, so no stale local copy can diverge from the
context.

This module is deliberately light (no cv2 / torch / matplotlib import) so tests
can build a context without loading any optional stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..fusion import FusionWeights

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..frame_extraction import ExtractedFrames
    from ..osm_data import RoadGraph
    from ..pipeline import PipelineConfig
    from ..trajectory_matching import MatchCandidate
    from ..visual_odometry import Trajectory


@dataclass
class RunContext:
    """State carried between pipeline stages (see the module docstring)."""

    cfg: "PipelineConfig"
    fusion_w: FusionWeights
    # Per-channel outcome record, written to result["channels"]: "ok" |
    # "inactive" | "failed" | "disabled" (see _record_channel).
    channel_status: dict[str, dict] = field(default_factory=dict)
    # The result.json payload, created by the match-summary stage.
    result: dict = field(default_factory=dict)

    # -- acquire --------------------------------------------------------
    video_path: Path | None = None
    frames: "ExtractedFrames | None" = None

    # -- route-length prior ---------------------------------------------
    # The duration-based estimate is the fixed sanity reference every noisier
    # scale source (IPM flow, DA3, OCR anchors) is gated against; the running
    # estimate ``estimated_length_m`` is what the matcher finally uses.
    duration_prior_m: float | None = None
    estimated_length_m: float | None = None
    result_ipm_scale: dict | None = None

    # -- visual odometry --------------------------------------------------
    traj: "Trajectory | None" = None
    n_valid: int | None = None
    vo_cache: Path | None = None

    # -- coarse priors (VPR / plate / VLM) --------------------------------
    osm_around: tuple[float, float, float] | None = None
    frame_seed_source: str | None = None
    vpr_center: tuple | None = None
    vpr_track: Any = None                # (frame_idx, latlon, sims) or None
    c2f_info: dict | None = None         # coarse-to-fine record
    vpr_listing: dict | None = None      # per-pass Mapillary tile split/loss counts
    anchor_origin: str | None = None     # which channel produced vpr_center: vpr|vlm
    vpr_source_used: str | None = None   # effective reference source after fallbacks
    result_plate_anchor: dict | None = None
    plate_center: tuple | None = None
    result_vlm_anchor: dict | None = None

    # -- road graph ---------------------------------------------------------
    road: "RoadGraph | None" = None
    cache_path: Path | None = None       # the graphml the road graph was read from

    # -- sun heading ----------------------------------------------------------
    result_sun_heading: dict | None = None

    # -- matcher trajectory -----------------------------------------------------
    match_xz: np.ndarray | None = None   # the trajectory the shape matcher consumes
    match_ts: np.ndarray | None = None   # its time axis (may be shorter than frames)
    da3_rec: Any = None
    result_scale_recovery: dict | None = None
    result_traj_source: str | None = None
    result_loop_closure: dict | None = None

    # -- OCR anchors ----------------------------------------------------------------
    ocr_anchors: list = field(default_factory=list)
    street_anchors: list = field(default_factory=list)
    anchor_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    anchor_seed: list | None = None
    ocr_anchor_error: str | None = None
    anchor_transform: Any = None         # similarity VO->world (idea 2), if fit
    anchor_world_route: Any = None       # full VO path georeferenced to world xy

    # -- matching -----------------------------------------------------------------------
    locked_scale: float | None = None    # metres per VO unit, when scale_lock is on
    vggt_seed: list = field(default_factory=list)
    result_vggt: dict | None = None
    anchor_gated: bool = False           # enumeration was gated to anchor/VGGT seeds
    vpr_scale_applied: float | None = None
    candidates: "list[MatchCandidate]" = field(default_factory=list)
    geom_pool: "list[MatchCandidate]" = field(default_factory=list)

    # -- appearance / geometry channels --------------------------------------------------
    splat_img_rgb: Any = None
    ipm_canvas: Any = None
    ranking_mode: str = "shape"
    # Aerial and 3D-tile channels hand their rank maps and fusion weights to
    # the consensus; the tile3d mesh/samples are kept for the optional
    # post-selection placement refinement.
    aerial_rank: dict = field(default_factory=dict)
    w_aerial: float = 0.0
    tile3d_rank: dict = field(default_factory=dict)
    w_tile3d: float = 0.0
    t3_mesh_ref: Any = None
    t3_samples_ref: Any = None
    t3_K_ref: Any = None
    t3_skyline_ref: Any = None

    # -- placement / evaluation ------------------------------------------------------------
    anchored_cand: "MatchCandidate | None" = None
    waypoint_evals: Any = None
    anchored_eval: Any = None

    @classmethod
    def create(cls, cfg: "PipelineConfig") -> "RunContext":
        """A fresh context for ``cfg`` with the consensus weights resolved."""
        return cls(cfg=cfg, fusion_w=FusionWeights.from_overrides(cfg.fusion_weights))


def _record_channel(status: dict, name: str, state: str, *,
                    error: BaseException | None = None, **info) -> None:
    """Record a channel's outcome: "ok" | "inactive" | "failed" | "disabled".

    A channel that raised used to be indistinguishable in result.json from
    one that ran and changed nothing (each failure was only printed), which
    makes any ablation over these channels untrustworthy.
    """
    entry: dict = {"status": state, **info}
    if error is not None:
        entry["error"] = f"{type(error).__name__}: {error}"
    status[name] = entry
