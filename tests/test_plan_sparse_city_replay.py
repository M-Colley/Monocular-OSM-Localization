"""Metadata-only all-call replay planning; embeddings/pixels are not loaded."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import plan_sparse_city_replay as planner


def pool(directory, *, n=3, dimensions=4, offset=0., signature=None):
    directory.mkdir(parents=True)
    refs = [{"id": str(i), "lat": 49. + offset + i * .00001, "lon": 8.} for i in range(n)]
    meta = {"signature": signature or {"center": [49., 8.], "radius_m": 3000., "cap": 250}, "refs": refs}
    (directory / "mly_ref_meta_synthetic.json").write_text(json.dumps(meta))
    fingerprint = hashlib.sha1(json.dumps([[r["id"], r["lat"], r["lon"]] for r in refs]).encode()).hexdigest()
    np.savez(directory / "ref_img_store.npz", ids=np.array([r["id"] for r in refs]),
             xy=np.array([[r["lat"], r["lon"]] for r in refs]), raw=np.array([object()]))
    path = directory / f"ref_emb_megaloc_{fingerprint[:10]}.npz"
    np.savez(path, fingerprint=np.array(fingerprint), emb=np.zeros((n, dimensions), np.float32))
    return path


def setup(tmp_path, monkeypatch):
    data, archive = tmp_path / "data/submission", tmp_path / "archive"
    for name in ("mapillary", "mapillary_c2f", "mapillary_city/coarse_a", "mapillary_city/fine_a"):
        pool(archive / name)
    decoding = {"decoder": "viterbi", "n_references": 3, "n_queries": 80}
    record = {"status": "ok", "center": [49., 8.], "radius_m": 3000., "ref_cap": 250, "cell_index": 0,
              "track_decoding": deepcopy(decoding)}
    result = {"provenance": {"config": {"data_dir": str(data), "vpr_source": "mapillary", "city": "Synthetic City",
        "frame_stride": 4, "vo_start_sec": 4., "vo_end_sec": 599., "max_frames": None, "frame_max_width": None}},
        "vpr_coarse_to_fine": {}, "vpr_decoding": {"pass1": deepcopy(decoding), "pass2": deepcopy(decoding)},
        "city_cover_retrieval": {"n_cells": 1,
        "coarse_regions": [{**record, "cache": str(data / "mapillary_city/coarse_a")}],
        "fine_regions": [{**record, "cache": str(data / "mapillary_city/fine_a")}],
        "baseline": {"backbone": {"name": "megaloc", "hub": "pinned-but-not-proven-by-archive"}}},
        "eval": {"forbidden": "must never influence the pool plan"}}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic encoded video")
    monkeypatch.setattr(planner, "video_grid", lambda p, cfg: {"query_mode": "crops", "sample_count": 80})
    return path, archive, video, tmp_path / "plan", result


def test_all_original_calls_and_both_variants_frozen_without_loading_pixels_or_embeddings(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    original = np.load
    class MetadataOnly:
        def __init__(self, *a, **kw): self.value = original(*a, **kw)
        def __enter__(self): return self
        def __exit__(self, *args): self.value.close()
        def __getitem__(self, key):
            assert key not in {"raw", "emb"}, "planning must not materialize image/descriptor arrays"
            return self.value[key]
    monkeypatch.setattr(planner.np, "load", MetadataOnly)
    value, frozen = planner.plan(*args[:4], expected_calls=4)
    assert value["call_count"] == 4 and value["total_served_occurrences"] == 12 and value["union"]["count"] == 3
    assert value["execution_plan"]["status"] == "not_started" and frozen["models_have_not_run"]
    assert value["decoder_profile"]["variants"] == ["raw", "reference_mean"]
    assert all("unverified" in row["embeddings"]["encoding_identity"] for row in value["pools"])
    assert value["execution_plan"]["selected_variant"] is None
    for name, expected in frozen["artifacts"].items(): assert planner.sha(args[3] / name) == expected


def test_resume_never_reblesses_or_overwrites_an_existing_plan(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    planner.plan(*args[:4])
    original = (args[3] / "plan_frozen.json").read_bytes()
    with pytest.raises(ValueError, match="overwrite"): planner.plan(*args[:4])
    assert (args[3] / "plan_frozen.json").read_bytes() == original


@pytest.mark.parametrize("fault", ["extra", "missing", "partial", "count", "signature", "fingerprint", "coordinates", "dimensions", "call_count"])
def test_incomplete_or_stale_archives_cannot_become_a_complete_plan(tmp_path, monkeypatch, fault):
    path, archive, video, output, result = setup(tmp_path, monkeypatch)
    coarse = archive / "mapillary_city/coarse_a"
    if fault == "extra": pool(archive / "unregistered")
    elif fault == "missing": next(coarse.glob("ref_emb*.npz")).unlink()
    elif fault == "partial": result["city_cover_retrieval"]["coarse_regions"][0]["status"] = "failed"
    elif fault == "count": result["city_cover_retrieval"]["n_cells"] = 2
    elif fault == "call_count": result["city_cover_retrieval"]["coarse_regions"][0]["track_decoding"]["n_references"] = 4
    elif fault == "signature":
        m = coarse / "mly_ref_meta_synthetic.json"; v = json.loads(m.read_text()); v["signature"]["cap"] = 1500; m.write_text(json.dumps(v))
    elif fault == "fingerprint":
        emb = next(coarse.glob("ref_emb*.npz")); np.savez(emb, fingerprint=np.array("different"), emb=np.zeros((3, 4), np.float32))
    elif fault == "coordinates":
        store = coarse / "ref_img_store.npz"; np.savez(store, ids=np.array(["0", "1", "2"]), xy=np.zeros((3, 2)))
    else:
        emb = next(coarse.glob("ref_emb*.npz"))
        with np.load(emb) as z: fp = str(z["fingerprint"])
        np.savez(emb, fingerprint=np.array(fp), emb=np.zeros((3, 5), np.float32))
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError): planner.plan(path, archive, video, output)
    assert not output.exists()


def test_duplicate_public_id_position_conflict_refuses_entire_replay(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    directory = args[1] / "mapillary_city/fine_a"
    for path in directory.iterdir(): path.unlink()
    directory.rmdir()
    pool(directory, offset=.01)
    with pytest.raises(ValueError, match="conflicting positions"): planner.plan(*args[:4])


def test_only_registered_count_is_allowed(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="call count"): planner.plan(*args[:4], expected_calls=32)


def test_listing_ids_lost_to_download_are_explicit_and_fingerprint_covers_served_subset(tmp_path):
    path = pool(tmp_path / "pool")
    directory = path.parent
    meta_path = directory / "mly_ref_meta_synthetic.json"
    meta = json.loads(meta_path.read_text())
    meta["refs"].append({"id": "failed-download", "lat": 49., "lon": 8.})
    meta_path.write_text(json.dumps(meta))
    value = planner.inspect_pool(directory, {"kind": "coarse"})
    assert value["listed_count"] == 4 and value["served_count"] == 3
    assert not any("failed-download" in key for key in value["ref_ids"])


def test_original_accuracy_fields_cannot_change_registered_call_plan(tmp_path, monkeypatch):
    _, _, _, _, result = setup(tmp_path, monkeypatch)
    original = planner.call_records(result)
    changed = deepcopy(result)
    changed["eval"] = {"symmetric_error": -1e99, "selected_clip_oracle": "different"}
    changed["city_cover_retrieval"]["selected_city_frames"] = 0
    changed["city_cover_retrieval"]["coarse_regions"][0]["median_similarity"] = 999
    assert planner.call_records(changed) == original


@pytest.mark.parametrize("field,value", [("ground_truth_waypoints", "labels.json"), ("estimated_length_m", 1000.),
    ("osm_around", [49., 8., 3000.]), ("ocr_video_path", "other_video.mp4")])
def test_known_oracle_acquisition_cannot_supply_public_replay_pool(tmp_path, monkeypatch, field, value):
    args = setup(tmp_path, monkeypatch)
    result = args[-1]
    result["provenance"]["config"][field] = value
    with pytest.raises(ValueError, match="priors"): planner.call_records(result)
