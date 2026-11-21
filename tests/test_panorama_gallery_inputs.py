import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import panorama_gallery_inputs as adapter


@pytest.fixture
def pixels(tmp_path):
    family = tmp_path / "gallery"
    family.mkdir()
    rows, sources, expected = [], [], {}
    for pool, indices in enumerate((range(4), range(4, 9))):
        ids = list(indices)
        path = family / f"pool_{pool}.npz"
        raw = np.stack([np.broadcast_to(np.array([10+i, 20+2*i, 30+3*i], np.uint8), (512, 512, 3)) for i in ids])
        xy = np.array([[42+i*.001, -83.] for i in ids])
        np.savez(path, raw=raw, ids=np.array([str(i+1) for i in ids]), xy=xy)
        source = {"source": f"original_{pool}", "copy": path.name,
                  "raw_shape": list(raw.shape), "sha256": adapter.gallery.sha(path)}
        sources.append(source)
        for local, i in enumerate(ids):
            rows.append({"column": i, "id": f"mapillary:{i+1}", "source": source["source"],
                "raw_row": local, "latlon": xy[local].tolist(),
                "pixels_sha256": hashlib.sha256(raw[local].tobytes()).hexdigest()})
            expected[i] = raw[local].copy()
    return family, {"canonical_rows": rows, "raw_sources": sources}, expected


def test_only_fixed_candidate_rows_copied_with_global_indices_and_exact_bgr(pixels, tmp_path):
    family, plan, expected = pixels
    output = tmp_path / "candidate.npy"
    columns = np.array([1, 3, 5, 7])
    record = adapter.stage_candidate_pixels(family, plan, columns, output, check=lambda: {})
    raw = np.load(output, mmap_mode="r", allow_pickle=False)
    try:
        proxy = adapter.CandidatePixels(raw, columns, 9)
        assert len(proxy) == 9 and record["candidate_count"] == 4
        assert not raw.flags.writeable and record["shape"] == [4, 512, 512, 3]
        for index in columns:
            np.testing.assert_array_equal(proxy[index], expected[int(index)])
        with pytest.raises(ValueError, match="outside"):
            proxy[0]
        with pytest.raises(ValueError, match="integer"):
            proxy[True]
    finally:
        adapter.gallery.close_map(raw)
    output.unlink()  # proves the adapter has not retained a Windows map


@pytest.mark.parametrize("fault", ["pixels", "ID", "coordinate", "archive", "missing_source", "header"])
def test_changed_or_missing_pixels_never_form_a_complete_candidate_record(pixels, tmp_path, fault):
    family, plan, _ = pixels
    if fault == "pixels": plan["canonical_rows"][1]["pixels_sha256"] = "0" * 64
    if fault == "ID": plan["canonical_rows"][1]["id"] = "mapillary:999"
    if fault == "coordinate": plan["canonical_rows"][1]["latlon"][0] += .1
    if fault == "archive": plan["raw_sources"][0]["sha256"] = "0" * 64
    if fault == "missing_source": plan["raw_sources"] = plan["raw_sources"][1:]
    if fault == "header": plan["raw_sources"][0]["raw_shape"][1] = 511
    output = tmp_path / "candidate.npy"
    with pytest.raises(ValueError):
        adapter.stage_candidate_pixels(family, plan, np.array([1, 5]), output, check=lambda: {})
    output.unlink(missing_ok=True)


@pytest.mark.parametrize("columns", [[1, 1], [3, 1], [-1], [9], [], [True], [1.5], list(range(641))])
def test_invalid_candidate_union_refused_before_allocating(pixels, tmp_path, columns):
    family, plan, _ = pixels
    output = tmp_path / "candidate.npy"
    with pytest.raises(ValueError):
        adapter.stage_candidate_pixels(family, plan, np.asarray(columns), output, check=lambda: {})
    assert not output.exists()


@pytest.fixture
def prepared(pixels, monkeypatch):
    family, plan, _ = pixels
    ids = np.array([r["id"] for r in plan["canonical_rows"]])
    xy = np.array([r["latlon"] for r in plan["canonical_rows"]])
    times = np.arange(80, dtype=np.float64) / 10
    scores = np.tile(np.arange(9, dtype=np.float32), (80, 1)) / 10
    # Different column slopes ensure the centred ordering is meaningful.
    scores += np.arange(80, dtype=np.float32)[:, None] * np.arange(9, dtype=np.float32)[None, :] * .0001
    (family / "inputs").mkdir()
    np.savez(family / "inputs/full_evidence.npz", raw_cosines=scores, ref_ids=ids,
             ref_latlon=xy, query_times=times, frame_ordinals=np.arange(80), query_indices=np.arange(80))
    adapter.gallery.write(family / "inputs/replay_plan.json", {"synthetic": True})
    plan.update(profile=adapter.gallery.PROFILE, profile_sha256=adapter.gallery.PROFILE_SHA256,
        query={"planned_frame_ordinals": list(range(80))}, per_query_native_pixel_sha256=["a" * 64] * 80,
        query_times_sha256=adapter.gallery._array_sha(times),
        control={"ref_ids_sha256": adapter.gallery._json_sha(ids.tolist()),
            "ref_latlon_sha256": adapter.gallery._array_sha(xy), "raw_cosines_sha256": adapter.gallery._array_sha(scores)})
    adapter.gallery.write(family / "plan.json", plan)
    members = [family / "plan.json", family / "inputs/full_evidence.npz", family / "inputs/replay_plan.json",
               *[family / row["copy"] for row in plan["raw_sources"]]]
    adapter.gallery.write(family / "inputs_frozen.json", {"scoring_has_not_run": True,
        "artifacts": {str(p.relative_to(family)): adapter.gallery.sha(p) for p in members}})
    monkeypatch.setattr(adapter.gallery, "frozen_inventory", lambda path: ({}, {}, {}))
    monkeypatch.setattr(adapter.gallery, "canonical_rows", lambda p, ledger: (plan["canonical_rows"], {}))
    monkeypatch.setattr(adapter.gallery.sparse, "check_headroom", lambda: {})
    # Default check is bound at function definition; use an explicit wrapper.
    actual_stage = adapter.stage_candidate_pixels
    monkeypatch.setattr(adapter, "stage_candidate_pixels", lambda *args, **kw: actual_stage(*args, **kw, check=lambda: {}))
    frames = SimpleNamespace(frames=[np.full((4, 6, 3), i, np.uint8) for i in range(80)], fps=10., closed=False)
    frames.close = lambda: setattr(frames, "closed", True)
    monkeypatch.setattr(adapter.gallery, "native_queries", lambda *args: frames)
    return family, adapter.gallery.sha(family / "inputs_frozen.json"), frames


def test_prepared_context_preserves_full_evidence_and_exact_native_stream_then_cleans_owned_pixels(prepared, tmp_path):
    family, freeze, frames = prepared
    staging = tmp_path / "owned_stage"
    with adapter.prepared_gallery_inputs(family, staging, expected_input_freeze_sha256=freeze) as values:
        assert values["evidence"]["similarities"].shape == (80, 9) and len(values["raw"]) == 9
        assert len(list(values["stream"])) == 80
        h = hashlib.sha256()
        for frame in frames.frames:
            h.update(str((frame.shape, frame.dtype.str)).encode()); h.update(frame.tobytes())
        assert values["query"]["pixels_sha256"] == h.hexdigest()
        assert values["registration"]["model_constructed"] is False
        values["verify_inputs"]()
    assert frames.closed and not (staging / "candidate_raw.npy").exists()


def test_changed_input_freeze_rejected_before_any_staging(prepared, tmp_path):
    family, _, _ = prepared
    staging = tmp_path / "owned_stage"
    with pytest.raises(ValueError, match="freeze changed"):
        with adapter.prepared_gallery_inputs(family, staging, expected_input_freeze_sha256="0" * 64):
            pytest.fail("must not yield")
    assert not staging.exists()


def test_candidate_mutation_or_consumer_failure_closes_frame_and_map_handles(prepared, tmp_path):
    family, freeze, frames = prepared
    staging = tmp_path / "owned_stage"
    with pytest.raises(ValueError, match="candidate pixels changed"):
        with adapter.prepared_gallery_inputs(family, staging, expected_input_freeze_sha256=freeze):
            with (staging / "candidate_raw.npy").open("r+b") as stream:
                stream.seek(-1, 2); stream.write(b"x")
    assert frames.closed and not (staging / "candidate_raw.npy").exists()
