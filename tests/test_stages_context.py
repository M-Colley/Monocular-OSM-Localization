"""RunContext and the helpers every stage shares.

``run_pipeline`` was one function whose stages shared function-local variables,
so none of them could be exercised without running all of them. These tests pin
the context's defaults (what a stage may rely on before an earlier stage ran)
and the channel-status bookkeeping the result.json ``channels`` block is built
from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monocular_osm.pipeline import PipelineConfig
from monocular_osm.stages.context import RunContext, _record_channel


def _cfg(tmp_path: Path, **kw) -> PipelineConfig:
    return PipelineConfig(url="https://example.com/v", city="Ulm, Germany",
                          data_dir=tmp_path / "data", output_dir=tmp_path / "out", **kw)


def test_create_resolves_fusion_weights_from_the_config(tmp_path):
    default = RunContext.create(_cfg(tmp_path))
    assert default.fusion_w.sliding == 1.0           # the published weight
    ablated = RunContext.create(_cfg(tmp_path, fusion_weights={"sliding": 0.0}))
    assert ablated.fusion_w.sliding == 0.0
    assert ablated.cfg.fusion_weights == {"sliding": 0.0}


def test_create_rejects_an_unknown_fusion_weight(tmp_path):
    with pytest.raises(ValueError, match="unknown fusion weight"):
        RunContext.create(_cfg(tmp_path, fusion_weights={"nonsense": 1.0}))


def test_a_fresh_context_holds_only_neutral_defaults(tmp_path):
    ctx = RunContext.create(_cfg(tmp_path))
    assert ctx.channel_status == {} and ctx.result == {}
    assert ctx.candidates == [] and ctx.geom_pool == []
    assert ctx.ocr_anchors == [] and ctx.street_anchors == []
    assert ctx.anchor_xy.shape == (0, 2)
    assert ctx.ranking_mode == "shape"
    assert ctx.w_aerial == 0.0 and ctx.w_tile3d == 0.0
    # "not computed yet" is None, never a plausible-looking number
    for name in ("video_path", "frames", "duration_prior_m", "estimated_length_m",
                 "traj", "n_valid", "vpr_center", "vpr_track", "plate_center", "road",
                 "match_xz", "match_ts", "locked_scale", "anchored_cand",
                 "result_scale_recovery", "result_sun_heading"):
        assert getattr(ctx, name) is None, name


def test_contexts_do_not_share_mutable_defaults(tmp_path):
    a, b = RunContext.create(_cfg(tmp_path)), RunContext.create(_cfg(tmp_path))
    a.channel_status["vpr"] = {"status": "ok"}
    a.candidates.append(object())
    a.ocr_anchors.append(object())
    a.result["x"] = 1
    assert b.channel_status == {} and b.candidates == [] and b.ocr_anchors == []
    assert b.result == {}
    assert a.anchor_xy is not b.anchor_xy


def test_record_channel_stores_state_info_and_the_error():
    status: dict = {}
    _record_channel(status, "vpr", "inactive", reason="no references")
    assert status["vpr"] == {"status": "inactive", "reason": "no references"}
    _record_channel(status, "plate", "failed", error=ValueError("bad frame"))
    assert status["plate"] == {"status": "failed", "error": "ValueError: bad frame"}


def test_record_channel_last_write_wins():
    status: dict = {}
    _record_channel(status, "sun", "inactive")
    _record_channel(status, "sun", "failed", error=RuntimeError("boom"))
    assert status["sun"]["status"] == "failed"


def test_the_array_default_is_not_the_module_constant(tmp_path):
    ctx = RunContext.create(_cfg(tmp_path))
    ctx.anchor_xy = np.vstack([ctx.anchor_xy, [[1.0, 2.0]]])
    assert RunContext.create(_cfg(tmp_path)).anchor_xy.shape == (0, 2)
