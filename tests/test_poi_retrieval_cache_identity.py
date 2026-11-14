"""Exact numeric request identities without retrieval/model construction."""
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from monocular_osm import kartaview_vpr as kv
from monocular_osm.stages import poi_only as stage

API_SHA = "9a191d6fb017cd8bdc223e56cd89a497952cda80501874e84f0aa37cdb9d4668"
CENTER = (33.78002955000001, -84.36835845)
INVENTORY = "aa139eef338c7047b1c742c42470fff6f7333311fa82f68e96deaa36d880967c"


@pytest.fixture
def retrieval(tmp_path, monkeypatch):
    calls = []
    def evidence(frames, times, center, **kwargs):
        kwargs.pop("token", None)
        calls.append({"center": center, **kwargs})
        return {"query_indices": np.array([0, 1]), "query_times": np.array([0., 1.]),
            "ref_ids": np.array(["mapillary:one"]), "ref_latlon": np.array([[33.78, -84.36]]),
            "similarities": np.array([[.2], [.3]], np.float32), "provenance": {}}
    monkeypatch.setattr(kv, "kartaview_vpr_evidence", evidence)
    @contextmanager
    def cpu(count):
        assert count == 4
        yield {"device": "cpu", "intra_op_threads": 4, "inter_op_threads": 1}
    monkeypatch.setattr(stage, "torch_cpu_threads", cpu)
    original = stage.file_sha256
    monkeypatch.setattr(stage, "file_sha256", lambda p: API_SHA if Path(p) == Path(kv.__file__) else original(p))
    cfg = SimpleNamespace(vpr_source="mapillary", vpr_query_mode="crops", vpr_query_batch_size=4,
        vpr_region_cache_dir=None, data_dir=tmp_path / "data" / "video", output_dir=tmp_path / "out")
    inputs = {"inventory": {"snapshot_sha256": INVENTORY}, "evidence_artifacts": []}
    def run(*, center=CENTER, radius=3000., cap=1500):
        retrieve = stage._retrieval(SimpleNamespace(cfg=cfg), inputs)
        return retrieve([], [0., 1.], center, radius_m=radius, cap=cap, n_query=2, query_cache={})
    return SimpleNamespace(run=run, calls=calls, cfg=cfg, inputs=inputs)


@pytest.mark.parametrize("radius,cap", [(3000, 1500), (3000., 1500), (3000, 1500.), (3000., 1500.)])
def test_equivalent_numeric_requests_keep_original_floating_radius_seed_tag(retrieval, radius, cap):
    retrieval.run(radius=radius, cap=cap)
    call = retrieval.calls[-1]
    assert Path(call["cache_dir"]).name == "poi_ac558d32cf3436d2"
    assert call["center"] == CENTER and all(type(v) is float for v in call["center"])
    assert type(call["radius_m"]) is float and call["radius_m"] == radius
    assert type(call["cap"]) is int and call["cap"] == cap
    assert call["source"] == "mapillary" and call["device"] == "cpu"
    assert call["query_mode"] == "crops" and call["query_batch_size"] == 4


def test_center_coordinates_are_exact_floats_without_rounding():
    center, radius, cap = stage._canonical_retrieval_request((33, -84), 3000, 1500.)
    assert center == (33., -84.) and all(type(v) is float for v in center)
    assert type(radius) is float and type(cap) is int


def test_one_ulp_center_and_radius_changes_have_distinct_full_request_keys(retrieval):
    moved = (float(np.nextafter(CENTER[0], np.inf)), CENTER[1])
    radius = float(np.nextafter(3000., np.inf))
    retrieval.run()
    retrieval.run(center=moved)
    retrieval.run(radius=radius)
    assert len({c["cache_dir"] for c in retrieval.calls}) == 3
    assert retrieval.calls[1]["center"][0].hex() == moved[0].hex()
    assert retrieval.calls[2]["radius_m"].hex() == radius.hex()
    assert round(moved[0], 4) == round(CENTER[0], 4)


@pytest.mark.parametrize("patch", [
    {"center": None}, {"center": ()}, {"center": (1, 2, 3)}, {"center": (True, 1)},
    {"center": (1, False)}, {"center": (np.bool_(True), 1)}, {"center": (np.nan, 1)}, {"center": (1, np.inf)},
    {"center": ("33.78", -84.)}, {"center": (91., 0.)}, {"center": (0., -181.)},
    {"radius": True}, {"radius": False}, {"radius": np.nan}, {"radius": np.inf},
    {"radius": -1}, {"radius": 0}, {"radius": "3000"}, {"radius": None}, {"radius": 2**53+1},
    {"radius": np.int64(2**53+1)},
    {"cap": True}, {"cap": False}, {"cap": np.nan}, {"cap": np.inf},
    {"cap": 1500.5}, {"cap": "1500"}, {"cap": 0}, {"cap": -1}, {"cap": None},
])
def test_malformed_or_inexact_requests_refuse_before_fetch(retrieval, patch):
    with pytest.raises(ValueError): retrieval.run(**patch)
    assert retrieval.calls == [] and retrieval.inputs["evidence_artifacts"] == []


def test_inventory_and_query_mode_remain_part_of_identity(retrieval):
    retrieval.run()
    retrieval.inputs["inventory"]["snapshot_sha256"] = "1" * 64
    retrieval.run()
    retrieval.cfg.vpr_query_mode = "warp"
    retrieval.run()
    assert len({call["cache_dir"] for call in retrieval.calls}) == 3
    assert retrieval.calls[-1]["query_mode"] == "warp"


def test_source_remains_part_of_identity_and_actual_request(retrieval):
    retrieval.run()
    retrieval.cfg.vpr_source = "panoramax"
    retrieval.run()
    assert retrieval.calls[0]["cache_dir"] != retrieval.calls[1]["cache_dir"]
    assert retrieval.calls[1]["source"] == "panoramax"


def test_external_cache_layout_uses_same_canonical_identity(retrieval):
    retrieval.cfg.vpr_region_cache_dir = retrieval.cfg.data_dir.parent / "external"
    retrieval.run(radius=3000, cap=1500.)
    path = Path(retrieval.calls[0]["cache_dir"])
    assert path.name == "mapillary_region_poi_ac558d32cf3436d2"
    assert path.parent.name == retrieval.cfg.data_dir.name
