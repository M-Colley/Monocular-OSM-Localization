"""Consensus fusion over candidate walks — one explicit, reusable scoring rule.

The pipeline scores each matcher candidate on several channels and combines
them into one consensus score (lower = better). This module holds that rule
so the pipeline and the offline ablation script (``scripts/ablate.py``)
compute EXACTLY the same thing from the same per-candidate feature table:

    score_i = Σ_c  w_c · rank_c(i)  +  Σ_p  penalty_p(i)

Rank channels contribute ``weight × rank`` (rank 1 = best); penalty channels
(plate district, VPR prior, sun heading) are already in score units.

**Effective weights.** :data:`DEFAULT_WEIGHTS` are the values the pipeline
has always used, kept so results do not change silently. Two of them are
easy to misread, so they are spelled out here:

* ``sliding`` falls back to the shape rank whenever sliding-window matching
  did not run (the default), so by default shape effectively carries
  ``shape + sliding = 2.0``, not the 1.0 the README table listed.
* ``aerial`` (when the aerial channel runs) scores the raster overlap of the
  aligned trajectory with the candidate walk — the same pair the shape fit
  already scored — so it is not an independent channel.

Both are kept as-is and reported; changing them is a measurement question
for ``scripts/ablate.py``, not something to do blind.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

# Rank channels in the order they appear in the feature table.
RANK_CHANNELS = ("shape", "sliding", "turn", "anchor", "aerial", "tile3d")
# Penalty channels (already in score units).
PENALTY_CHANNELS = ("plate", "vpr", "sun")


@dataclass
class FusionWeights:
    """Rank-channel weights. ``anchor`` depends on whether OCR anchors gated
    enumeration (see the pipeline comment): 0.25 when they did, 2.0 when
    they did not. ``tile3d`` is the fallback when no adaptive weight is
    computed. A weight only applies when its channel actually ran."""

    shape: float = 1.0
    sliding: float = 1.0
    turn: float = 0.0
    anchor_gated: float = 0.25
    anchor_ungated: float = 2.0
    aerial: float = 0.5
    tile3d: float = 0.4

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, float] | None) -> "FusionWeights":
        w = cls()
        if not overrides:
            return w
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown fusion weight(s): {sorted(unknown)}; "
                             f"known: {sorted(known)}")
        for k, v in overrides.items():
            setattr(w, k, float(v))
        return w


DEFAULT_WEIGHTS = FusionWeights()


@dataclass
class CandidateFeatures:
    """One candidate's per-channel inputs to the fusion rule."""

    shape_rank: int
    sliding_rank: int
    turn_rank: int
    anchor_rank: int
    aerial_rank: int
    tile3d_rank: int
    plate_penalty: float = 0.0
    vpr_penalty: float = 0.0
    sun_penalty: float = 0.0
    # Free-form diagnostics carried into the feature table (raw channel
    # values: rms, corr, anchor distance, tile3d score, ...).
    extra: dict[str, Any] = field(default_factory=dict)


def effective_weights(
    weights: FusionWeights,
    *,
    have_anchors: bool,
    anchor_gated: bool,
    aerial_active: bool,
    tile3d_weight: float | None,
) -> dict[str, float]:
    """The weight each rank channel actually gets on this run."""
    if not have_anchors:
        anchor = 0.0
    elif anchor_gated:
        anchor = weights.anchor_gated
    else:
        anchor = weights.anchor_ungated
    return {
        "shape": weights.shape,
        "sliding": weights.sliding,
        "turn": weights.turn,
        "anchor": anchor,
        "aerial": weights.aerial if aerial_active else 0.0,
        "tile3d": float(tile3d_weight) if tile3d_weight is not None else 0.0,
    }


def fused_score(f: CandidateFeatures, w: Mapping[str, float]) -> float:
    return (
        w["shape"] * f.shape_rank
        + w["sliding"] * f.sliding_rank
        + w["turn"] * f.turn_rank
        + w["anchor"] * f.anchor_rank
        + w["aerial"] * f.aerial_rank
        + w["tile3d"] * f.tile3d_rank
        + f.plate_penalty
        + f.vpr_penalty
        + f.sun_penalty
    )


def consensus_order(features: list[CandidateFeatures],
                    w: Mapping[str, float]) -> list[int]:
    """Candidate indices best-first; ties broken by the matcher's order."""
    return sorted(range(len(features)), key=lambda i: (fused_score(features[i], w), i))


def write_feature_table(
    path: Path,
    features: list[CandidateFeatures],
    effective: Mapping[str, float],
    *,
    routes_latlon: Iterable[list[list[float]]] | None = None,
    channel_status: Mapping[str, Any] | None = None,
    weights: FusionWeights | None = None,
) -> None:
    """Persist everything needed to re-run the fusion offline.

    ``routes_latlon[i]`` is candidate i's aligned route, so an offline script
    can score any re-ranking against ground truth without re-running the
    pipeline. Ground truth itself is never written here.
    """
    routes = list(routes_latlon) if routes_latlon is not None else [None] * len(features)
    rows = []
    for i, (f, r) in enumerate(zip(features, routes)):
        row = asdict(f)
        row["matcher_index"] = i
        row["fused_score"] = fused_score(f, effective)
        if r is not None:
            row["route_latlon"] = r
        rows.append(row)
    payload = {
        "schema": 1,
        "rule": "sum_c w_c*rank_c + plate_penalty + vpr_penalty + sun_penalty",
        "weights_configured": asdict(weights or DEFAULT_WEIGHTS),
        "weights_effective": dict(effective),
        "channels": dict(channel_status or {}),
        "candidates": rows,
    }
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_feature_table(path: Path) -> tuple[list[CandidateFeatures], dict, dict]:
    """(features, effective weights, full payload) from a written table."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {f.name for f in fields(CandidateFeatures)}
    feats = [CandidateFeatures(**{k: v for k, v in row.items() if k in names})
             for row in payload["candidates"]]
    return feats, payload["weights_effective"], payload
