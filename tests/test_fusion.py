"""The consensus-fusion rule must be explicit, reproducible offline, and
identical to what the pipeline computed before it was factored out."""

from __future__ import annotations

import json

import numpy as np
import pytest

from monocular_osm.fusion import (
    DEFAULT_WEIGHTS,
    CandidateFeatures,
    FusionWeights,
    consensus_order,
    effective_weights,
    fused_score,
    load_feature_table,
    write_feature_table,
)


def _legacy_fused(f: CandidateFeatures, *, have_anchors, anchor_gated,
                  aerial_active, tile3d_weight) -> float:
    """Verbatim arithmetic of the pipeline's former inline _fused()."""
    W_SHAPE, W_SLIDING, W_TURN = 1.0, 1.0, 0.0
    if not have_anchors:
        W_ANCHOR = 0.0
    elif anchor_gated:
        W_ANCHOR = 0.25
    else:
        W_ANCHOR = 2.0
    W_AERIAL = 0.5 if aerial_active else 0.0
    W_TILE3D = tile3d_weight or 0.0
    return (W_SHAPE * f.shape_rank + W_SLIDING * f.sliding_rank
            + W_TURN * f.turn_rank + W_ANCHOR * f.anchor_rank
            + W_AERIAL * f.aerial_rank + W_TILE3D * f.tile3d_rank
            + f.plate_penalty + f.vpr_penalty + f.sun_penalty)


def _random_features(rng, n=10):
    perms = [rng.permutation(n) + 1 for _ in range(5)]
    return [
        CandidateFeatures(
            shape_rank=i + 1, sliding_rank=int(perms[0][i]),
            turn_rank=int(perms[1][i]), anchor_rank=int(perms[2][i]),
            aerial_rank=int(perms[3][i]), tile3d_rank=int(perms[4][i]),
            plate_penalty=float(rng.uniform(0, 20)),
            vpr_penalty=float(rng.uniform(0, 30)),
            sun_penalty=float(rng.uniform(0, 5)),
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("have_anchors,anchor_gated,aerial_active,tile3d", [
    (False, False, True, None), (True, True, True, 0.4),
    (True, False, False, 0.73), (False, False, False, None),
])
def test_default_weights_reproduce_the_pipelines_former_formula(
        have_anchors, anchor_gated, aerial_active, tile3d) -> None:
    rng = np.random.default_rng(0)
    feats = _random_features(rng)
    eff = effective_weights(DEFAULT_WEIGHTS, have_anchors=have_anchors,
                            anchor_gated=anchor_gated, aerial_active=aerial_active,
                            tile3d_weight=tile3d)
    for f in feats:
        assert fused_score(f, eff) == pytest.approx(_legacy_fused(
            f, have_anchors=have_anchors, anchor_gated=anchor_gated,
            aerial_active=aerial_active, tile3d_weight=tile3d))


def test_shape_effectively_counts_twice_when_sliding_is_off() -> None:
    """Documented, not hidden: with sliding windows off the sliding rank IS
    the shape rank, so shape's effective weight is shape + sliding."""
    eff = effective_weights(DEFAULT_WEIGHTS, have_anchors=False, anchor_gated=False,
                            aerial_active=False, tile3d_weight=None)
    f = CandidateFeatures(shape_rank=3, sliding_rank=3, turn_rank=1,
                          anchor_rank=1, aerial_rank=1, tile3d_rank=1)
    assert fused_score(f, eff) == pytest.approx(2.0 * 3)


def test_inactive_channels_get_zero_weight() -> None:
    eff = effective_weights(DEFAULT_WEIGHTS, have_anchors=False, anchor_gated=True,
                            aerial_active=False, tile3d_weight=None)
    assert eff["anchor"] == 0.0 and eff["aerial"] == 0.0 and eff["tile3d"] == 0.0


def test_overrides_apply_and_unknown_names_are_rejected() -> None:
    w = FusionWeights.from_overrides({"sliding": 0.0, "aerial": 0.25})
    assert w.sliding == 0.0 and w.aerial == 0.25 and w.shape == 1.0
    with pytest.raises(ValueError, match="unknown fusion weight"):
        FusionWeights.from_overrides({"shapee": 1.0})
    assert FusionWeights.from_overrides(None) == DEFAULT_WEIGHTS


def test_consensus_ties_keep_matcher_order() -> None:
    f = CandidateFeatures(shape_rank=1, sliding_rank=1, turn_rank=1,
                          anchor_rank=1, aerial_rank=1, tile3d_rank=1)
    eff = {k: 0.0 for k in ("shape", "sliding", "turn", "anchor", "aerial", "tile3d")}
    assert consensus_order([f, f, f], eff) == [0, 1, 2]


def test_feature_table_round_trips_and_rescoring_matches(tmp_path) -> None:
    rng = np.random.default_rng(1)
    feats = _random_features(rng, n=6)
    eff = effective_weights(DEFAULT_WEIGHTS, have_anchors=True, anchor_gated=False,
                            aerial_active=True, tile3d_weight=0.4)
    routes = [[[48.4 + i * 1e-3, 9.99], [48.401 + i * 1e-3, 9.991]] for i in range(6)]
    path = tmp_path / "fusion_features.json"
    write_feature_table(path, feats, eff, routes_latlon=routes,
                        channel_status={"vpr": {"status": "ok"}})
    loaded, eff2, payload = load_feature_table(path)
    assert eff2 == eff
    assert consensus_order(loaded, eff2) == consensus_order(feats, eff)
    assert payload["candidates"][2]["route_latlon"] == routes[2]
    assert payload["channels"]["vpr"]["status"] == "ok"
    # Ground truth is never written into the table.
    assert "gt" not in json.dumps(payload).lower().replace("weights", "")
