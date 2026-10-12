"""scripts/ablate.py: offline channel ablation and leave-one-clip-out fusion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from monocular_osm.fusion import CandidateFeatures

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ablate_under_test",
                                               ROOT / "scripts" / "ablate.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

EFF = {"shape": 1.0, "sliding": 1.0, "turn": 0.0, "anchor": 0.0,
       "aerial": 0.5, "tile3d": 0.0}


def _rep(clip, rep, rng, n=8, vpr_informative=True):
    """Shape ranks are noise w.r.t. GT error; the VPR penalty tracks it."""
    errors = rng.uniform(50, 3000, n)
    aerial = rng.permutation(n) + 1
    vpr = errors / 100.0 if vpr_informative else rng.uniform(0, 30, n)
    feats = [CandidateFeatures(shape_rank=i + 1, sliding_rank=i + 1, turn_rank=1,
                               anchor_rank=1, aerial_rank=int(aerial[i]), tile3d_rank=1,
                               vpr_penalty=float(vpr[i]))
             for i in range(n)]
    return {"clip": clip, "rep": rep, "feats": feats, "eff": dict(EFF),
            "errors": errors}


def test_variants_cover_active_channels_and_double_shape() -> None:
    rng = np.random.default_rng(0)
    r = _rep("a", "rep1", rng)
    names = set(A.variants_for(r["feats"], r["eff"]))
    assert {"recorded", "shape_only", "no_sliding", "no_aerial", "no_vpr",
            "no_double_shape"} <= names
    assert "no_turn" not in names and "no_tile3d" not in names   # inactive


def test_oracle_bounds_every_variant_and_learning_finds_the_signal() -> None:
    rng = np.random.default_rng(1)
    reps = [_rep(c, f"rep{k}", rng) for c in "abcdef" for k in range(2)]
    table = A.ablate(reps)
    oracle = table["oracle"]["fleet_mean_of_clip_medians_m"]
    for name, v in table.items():
        assert v["fleet_mean_of_clip_medians_m"] >= oracle - 1e-6, name
    # The VPR penalty is the informative channel; removing it must hurt, and
    # the held-out learned fusion must do at least as well as shape alone.
    assert (table["no_vpr"]["fleet_mean_of_clip_medians_m"]
            > table["recorded"]["fleet_mean_of_clip_medians_m"])
    assert (table["learned_loco"]["fleet_mean_of_clip_medians_m"]
            <= table["shape_only"]["fleet_mean_of_clip_medians_m"])


def test_learned_fusion_is_leave_one_clip_out() -> None:
    """A clip must never be scored by weights fitted on itself: with only
    two clips, each held-out fit sees one clip and is skipped (< 2)."""
    rng = np.random.default_rng(2)
    reps = [_rep(c, "rep1", rng) for c in "ab"]
    assert all(p is None for p in A.learned_picks(reps).values())


def test_markdown_lists_variants_and_clips() -> None:
    rng = np.random.default_rng(3)
    md = A.to_markdown(A.ablate([_rep(c, "rep1", rng) for c in "abc"]), "symmetric_mean_m")
    assert "| recorded |" in md and "| oracle |" in md and "symmetric_mean_m" in md


@pytest.mark.parametrize("informative", [False])
def test_uninformative_channels_do_not_crash_the_fit(informative) -> None:
    rng = np.random.default_rng(4)
    reps = [_rep(c, "rep1", rng, vpr_informative=informative) for c in "abcd"]
    A.ablate(reps)   # must not raise
