"""Bounded gallery plumbing with fake encoders; no model or frozen data reads."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import pilot_image_gallery as gallery


def digest(frame):
    return hashlib.sha256(memoryview(frame).cast("B")).hexdigest()


@pytest.fixture
def inventory(tmp_path):
    positions = {"a": [42., -83.], "b": [42.1, -83.1], "c": [42.2, -83.2]}
    pools, stores, occurrences = [], [], {}
    family = tmp_path / "family"
    (family / "inputs").mkdir(parents=True)
    for index, names in enumerate((["b", "a"], ["b", "c"])):
        raw = np.empty((2, 512, 512, 3), np.uint8)
        for row, identifier in enumerate(names):
            value = ord(identifier) - ord("a")
            raw[row] = [value+1, value+2, value+3]
        path = family / "inputs" / f"pool_{index}.npz"
        np.savez(path, ids=np.asarray(names), xy=np.asarray([positions[v] for v in names]), raw=raw)
        ids = ["mapillary:" + v for v in names]
        pools.append({"ref_ids": ids, "ref_latlon": [positions[v] for v in names],
            "served_count": 2, "reference_store": {"path": str(path)}})
        stores.append({"path": str(path), "pool_indices": [index], "raw_shape": list(raw.shape),
            "bytes": path.stat().st_size, "status": "available", "sha256": gallery.sha(path)})
        for row, identifier in enumerate(ids):
            occurrences.setdefault(identifier, []).append({"path": str(path), "raw_row": row,
                "pixels_sha256": digest(raw[row]), "latlon": positions[names[row]]})
    ids = sorted(occurrences)
    ledger = {"schema": "reference-raw-pixel-inventory-v1", "storage_color": "BGR_uint8_OpenCV",
        "ids_in_frozen_union_order": ids, "missing_ids": [], "conflicting_pixel_ids": [], "stores": stores,
        "per_id": {identifier: {"canonical_pool_index": 1 if identifier.endswith(":c") else 0,
            "canonical": values[0], "occurrences": values} for identifier, values in occurrences.items()}}
    plan = {"pools": pools, "union": {"ref_ids": ids}}
    return family, plan, ledger


def prepared_plan(plan, ledger):
    rows, stores = gallery.canonical_rows(plan, ledger)
    return {"canonical_rows": rows, "raw_sources": [
        {"source": path, "copy": "inputs/" + Path(path).name, "sha256": store["sha256"],
         "raw_shape": store["raw_shape"], "pool_indices": store["pool_indices"]}
        for path, store in stores.items()]}


def test_canonical_rows_match_first_pool_and_sorted_union(inventory):
    _, plan, ledger = inventory
    rows, stores = gallery.canonical_rows(plan, ledger)
    assert [row["id"] for row in rows] == ["mapillary:a", "mapillary:b", "mapillary:c"]
    assert [row["pool_index"] for row in rows] == [0, 0, 1]
    assert [row["raw_row"] for row in rows] == [1, 0, 1]
    assert len(rows) == 3 and len(stores) == 2


@pytest.mark.parametrize("change", ["missing", "column_order", "canonical_pool", "canonical_origin",
    "canonical_position", "bad_digest", "raw_row", "store_order", "pool_position", "unregistered_store"])
def test_canonical_identity_changes_fail_closed(inventory, change):
    _, plan, ledger = inventory
    item = ledger["per_id"]["mapillary:b"]
    if change == "missing":
        del ledger["per_id"]["mapillary:c"]
    elif change == "column_order":
        ledger["ids_in_frozen_union_order"] = list(reversed(ledger["ids_in_frozen_union_order"]))
    elif change == "canonical_pool":
        item["canonical_pool_index"] = 1
    elif change == "canonical_origin":
        item["canonical"] = item["occurrences"][1]
    elif change == "canonical_position":
        item["canonical"]["latlon"] = [0., 0.]
    elif change == "bad_digest":
        item["canonical"]["pixels_sha256"] = "z" * 64
    elif change == "raw_row":
        item["canonical"]["raw_row"] = 100
    elif change == "store_order":
        ledger["stores"][0]["pool_indices"] = [1]
    elif change == "pool_position":
        plan["pools"][1]["ref_latlon"][0] = [0., 0.]
    else:
        ledger["stores"].append({**ledger["stores"][0], "path": str(Path(ledger["stores"][0]["path"]).with_name("other.npz"))})
    with pytest.raises(ValueError):
        gallery.canonical_rows(plan, ledger)


def test_duplicate_pixel_conflicts_are_retained_without_version_selection(inventory):
    _, plan, ledger = inventory
    ledger["per_id"]["mapillary:b"]["occurrences"][1]["pixels_sha256"] = "1" * 64
    ledger["conflicting_pixel_ids"] = ["mapillary:b"]
    rows, _ = gallery.canonical_rows(plan, ledger)
    assert rows[1]["pool_index"] == 0
    assert rows[1]["pixels_sha256"] == ledger["per_id"]["mapillary:b"]["occurrences"][0]["pixels_sha256"]
    ledger["conflicting_pixel_ids"] = []
    with pytest.raises(ValueError, match="hidden"):
        gallery.canonical_rows(plan, ledger)


def fake_encoder(observed):
    def encode(views, model, torch):
        assert len(views) <= 4
        values = []
        for index in range(len(views)):
            rgb = views[index]
            observed.append(rgb[0, 0].tolist())
            assert rgb.shape == (512, 512, 3) and rgb.dtype == np.uint8
            value = rgb[0, 0].astype(np.float32)
            values.append(value / np.linalg.norm(value))
        return np.asarray(values, np.float32)
    return encode


def test_encode_gallery_streams_pixels_maps_once_and_writes_exact_columns(inventory, monkeypatch):
    family, plan, ledger = inventory
    prepared = prepared_plan(plan, ledger)
    observed, checks = [], []
    monkeypatch.setattr(gallery.image, "DIMENSION", 3)
    monkeypatch.setattr(gallery.sparse, "check_headroom", lambda: checks.append(True) or {"verified": True})
    original = np.lib.npyio.NpzFile.__getitem__
    def guarded(self, name):
        assert name != "raw", "raw pool must be streamed, never materialized through NPZ indexing"
        return original(self, name)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guarded)
    output = family / "descriptors.npy"
    stats, actual_sha = gallery.encode_gallery(family, prepared, None, None, output, encoder=fake_encoder(observed))
    assert observed == [[3, 2, 1], [4, 3, 2], [5, 4, 3]]  # a,b,c; RGB, despite raw b,a order.
    result = np.load(output, mmap_mode="r", allow_pickle=False)
    try:
        assert result.shape == (3, 3) and result.dtype == np.float32
        assert actual_sha == gallery._array_sha(result)
        assert np.allclose(np.linalg.norm(result, axis=1), 1.)
    finally:
        gallery.close_map(result)
    assert len(checks) == 4 and [record["canonical_rows_encoded"] for record in stats] == [2, 1]
    assert not list((family / "raw_staging").glob("*.npy"))
    output.unlink()  # Windows would fail this if the descriptor map were left open.


@pytest.mark.parametrize("change", ["pixel", "id", "position", "archive", "descriptor", "escape"])
def test_encoding_rejects_changed_pixels_or_identity_and_closes_owned_maps(inventory, monkeypatch, change):
    family, plan, ledger = inventory
    prepared = prepared_plan(plan, ledger)
    monkeypatch.setattr(gallery.image, "DIMENSION", 3)
    monkeypatch.setattr(gallery.sparse, "check_headroom", lambda: {})
    encoder = fake_encoder([])
    if change == "pixel":
        prepared["canonical_rows"][0]["pixels_sha256"] = "0" * 64
    elif change == "id":
        prepared["canonical_rows"][0]["id"] = "mapillary:wrong"
    elif change == "position":
        prepared["canonical_rows"][0]["latlon"] = [0., 0.]
    elif change == "archive":
        prepared["raw_sources"][0]["sha256"] = "0" * 64
    elif change == "descriptor":
        encoder = lambda views, model, torch: np.zeros((len(views), 3), np.float32)
    else:
        prepared["raw_sources"][0]["copy"] = "../outside.npz"
    output = family / "descriptors.npy"
    with pytest.raises(ValueError):
        gallery.encode_gallery(family, prepared, None, None, output, encoder=encoder)
    assert not list((family / "raw_staging").glob("*.npy"))
    output.unlink()


def test_headroom_is_checked_before_raw_extraction(inventory, monkeypatch):
    family, plan, ledger = inventory
    monkeypatch.setattr(gallery.image, "DIMENSION", 3)
    def low():
        raise ValueError("headroom")
    monkeypatch.setattr(gallery.sparse, "check_headroom", low)
    monkeypatch.setattr(gallery.image, "extract_raw", lambda *a: pytest.fail("must not extract"))
    output = family / "descriptors.npy"
    with pytest.raises(ValueError, match="headroom"):
        gallery.encode_gallery(family, prepared_plan(plan, ledger), None, None, output)
    output.unlink()


class Tensor:
    def __init__(self, value, sizes):
        self.value, self.sizes = value, sizes
    @property
    def T(self):
        return Tensor(self.value.T, self.sizes)
    def __matmul__(self, other):
        self.sizes.append(other.value.shape[1])
        return Tensor(self.value @ other.value, self.sizes)
    def numpy(self):
        return self.value


def test_score_blocks_preserve_all_columns_and_maximum_of_three_views(monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(gallery.image, "DIMENSION", 3)
    rng = np.random.default_rng(2)
    queries, refs = [rng.normal(size=shape).astype(np.float32) for shape in ((240, 3), (600, 3))]
    for values in (queries, refs):
        values /= np.linalg.norm(values, axis=1, keepdims=True)
    sizes = []
    torch = SimpleNamespace(inference_mode=nullcontext, from_numpy=lambda value: Tensor(value, sizes))
    actual = gallery.similarities(queries, refs, torch)
    expected = (queries @ refs.T).reshape(80, 3, 600).max(axis=1)
    assert np.allclose(actual, expected, rtol=0, atol=1e-6)
    assert sizes == [256, 256, 88] and actual.dtype == np.float32


def test_all_four_full_support_outputs_are_frozen_without_selection(tmp_path):
    ids = np.asarray(["mapillary:a", "mapillary:b", "mapillary:c"])
    coords = np.asarray([[42., -83.], [42.0001, -83.0001], [42.0002, -83.0002]])
    old = np.asarray([[.1, .4, .3], [.2, .5, .2]], np.float32)
    new = np.asarray([[.6, .2, .3], [.7, .2, .3]], np.float32)
    control = {"raw_cosines": old, "ref_ids": ids, "ref_latlon": coords, "query_times": np.array([0., 2.])}
    result = gallery.decode_outputs(tmp_path, new, control)
    assert set(result) == {"frozen_megaloc", "ImAge_GSV"}
    for model in result:
        complete = json.loads((tmp_path / f"predictions_{model}.json").read_text())
        assert complete["selected_variant"] is None
        for variant in ("raw", "reference_mean"):
            record = json.loads((tmp_path / f"prediction_{model}_{variant}.json").read_text())
            assert record["candidate_pruning"] is False and record["all_states_per_query"] == 3
    with pytest.raises(ValueError, match="row and column"):
        gallery.decode_outputs(tmp_path, new[:, :2], control)


@pytest.fixture
def native(monkeypatch, tmp_path):
    pixels = ["a" * 64] * 80
    times = np.arange(80, dtype=np.float64)
    plan = {"query": {"stride": 4, "registered_window": [0., 32.], "dense_count": 100,
        "query_dense_indices": list(range(80)), "fps": 10., "native_shape_hw": [2, 3]},
        "per_query_native_pixel_sha256": pixels}
    frames = SimpleNamespace(frames=[np.zeros((2, 3, 3), np.uint8)] * 80, timestamps=times.copy(), fps=10.,
        sampling={"dense_count": 100, "dense_indices": list(range(80))}, closed=False)
    frames.close = lambda: setattr(frames, "closed", True)
    from monocular_osm import frame_extraction
    calls = []
    monkeypatch.setattr(frame_extraction, "extract_frames", lambda path, **kwargs: calls.append(kwargs) or frames)
    monkeypatch.setattr(gallery.sparse, "verify_query_pixels", lambda *args: pixels)
    return frames, plan, times, calls


def test_queries_keep_exact_native_grid_and_bounded_full_window(native, tmp_path):
    frames, plan, times, calls = native
    assert gallery.native_queries(tmp_path, plan, times) is frames
    assert calls == [{"stride": 4, "max_frames": None, "start_sec": 0., "end_sec": 32.,
        "memory_mb": 128., "max_width": None, "sample_count": 80}]
    assert not frames.closed


@pytest.mark.parametrize("change", ["time", "fps", "count", "indices", "shape", "pixels"])
def test_queries_decline_changed_source_grid_and_cleanup(native, tmp_path, monkeypatch, change):
    frames, plan, times, _ = native
    if change == "time":
        frames.timestamps = times + .01
    elif change == "fps":
        frames.fps = 11.
    elif change == "count":
        frames.sampling["dense_count"] = 101
    elif change == "indices":
        frames.sampling["dense_indices"][4] += 1
    elif change == "shape":
        frames.frames[0] = np.zeros((2, 4, 3), np.uint8)
    else:
        monkeypatch.setattr(gallery.sparse, "verify_query_pixels", lambda *a: ["b" * 64] * 80)
    with pytest.raises(ValueError):
        gallery.native_queries(tmp_path, plan, times)
    assert frames.closed


def test_source_binding_rejects_foreign_driver_before_git(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "ROOT", tmp_path)
    monkeypatch.setattr(gallery.sparse, "source_identity", lambda *a: pytest.fail("must reject before git"))
    with pytest.raises(ValueError, match="another source"):
        gallery.source_binding("pin")


def test_infer_wrong_profile_cannot_import_or_construct_model(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery.image, "load_model", lambda *a: pytest.fail("must not construct"))
    with pytest.raises(ValueError, match="profile differs"):
        gallery.infer(tmp_path, "pin", "wrong")


@pytest.fixture
def ready(inventory, tmp_path, monkeypatch):
    _, plan, ledger = inventory
    parent, audit, official = [tmp_path / name for name in ("parent", "audit", "official")]
    for path in (parent / "inputs", parent / "inference", audit, official):
        path.mkdir(parents=True)
    ids = plan["union"]["ref_ids"]
    positions = np.asarray([ledger["per_id"][identifier]["canonical"]["latlon"] for identifier in ids])
    times = np.arange(80, dtype=np.float64)
    scores = np.tile(np.array([.1, .3, .2], np.float32), (80, 1))
    video = parent / "inputs/video.mp4"
    video.write_bytes(b"fake video; never decoded")
    plan["union"].update(count=3, ref_ids_sha256=gallery._json_sha(ids), ref_latlon_sha256=gallery._array_sha(positions))
    plan["query"] = {"video": {"path": str(video), "sha256": gallery.sha(video)}, "sample_count": 80,
        "query_mode": "crops", "views_per_query": 3, "native_pixels": True, "planned_frame_ordinals": list(range(80)),
        "stride": 1, "registered_window": [0., 80.], "fps": 1., "dense_count": 80,
        "query_dense_indices": list(range(80)), "native_shape_hw": [2, 3]}
    gallery.write(parent / "inputs/replay_plan.json", plan)
    runtime = {"actual_query_times_sha256": gallery._array_sha(times), "per_query_native_pixel_sha256": ["a" * 64] * 80}
    gallery.write(parent / "inference/runtime_and_lineage.json", runtime)
    np.savez(parent / "inference/full_evidence.npz", raw_cosines=scores, ref_ids=np.asarray(ids), ref_latlon=positions,
        query_times=times, query_indices=np.arange(80), frame_ordinals=np.arange(80))
    gallery.write(parent / "inputs_frozen.json", {"scoring_has_not_run": True,
        "artifacts": {f"inputs/{name}": gallery.sha(parent / "inputs" / name) for name in ("video.mp4", "replay_plan.json")}})
    gallery.write(parent / "inference/frozen_predictions.json", {"status": "complete", "scoring_has_not_run": True,
        "selected_variant": None, "artifacts": {name: gallery.sha(parent / "inference" / name)
            for name in ("full_evidence.npz", "runtime_and_lineage.json")}})
    plan_sha = gallery.sha(parent / "inputs/replay_plan.json")
    gallery.write(audit / "pixel_identity.json", ledger)
    gallery.write(audit / "inventory_plan.json", {"source_plan_sha256": plan_sha})
    gallery.write(audit / "feasibility.json", {"ground_truth_opened": False, "model_constructors": 0,
        "forwards": 0, "downloads": 0, "missing_ids": [], "metadata_failures": [], "pixel_stream_failures": [],
        "source_plan_sha256": plan_sha, "available_canonical_ids": 3})
    gallery.write(audit / "inventory_frozen.json", {"ground_truth_opened": False, "source_plan_sha256": plan_sha,
        "artifacts": {name: gallery.sha(audit / name) for name in ("pixel_identity.json", "inventory_plan.json", "feasibility.json")}})
    (official / "network.py").write_text("# fake official source; never executed\n")
    monkeypatch.setattr(gallery.image, "official_files", lambda path: [{"relative": "network.py", "sha256": gallery.sha(path / "network.py")}])
    assets = tmp_path / "assets"
    assets.mkdir()
    checkpoint, receipt = assets / "weights.pth", assets / "receipt.json"
    checkpoint.write_bytes(b"fake weights; never loaded")
    monkeypatch.setattr(gallery.image, "CHECKPOINT_SHA", gallery.sha(checkpoint))
    monkeypatch.setattr(gallery.image, "CHECKPOINT_BYTES", checkpoint.stat().st_size)
    gallery.write(receipt, {"actual_sha256": gallery.sha(checkpoint), "source_pin": gallery.image.OFFICIAL_PIN,
        "official_asset_digest": "sha256:" + gallery.sha(checkpoint)})
    return {"parent": parent, "inventory": audit / "inventory_frozen.json", "official": official,
        "checkpoint": checkpoint, "receipt": receipt, "family": tmp_path / "prepared"}


def test_prepare_freezes_complete_copied_pixel_lineage_and_control_without_models(ready, monkeypatch):
    monkeypatch.setattr(gallery.image, "load_model", lambda *a: pytest.fail("prepare cannot construct"))
    result = gallery.prepare(**ready)
    family = ready["family"]
    assert result["status"] == "prepared_only" and result["reference_count"] == 3 and result["raw_pools"] == 2
    gallery.sparse.verify_freeze(family / "inputs_frozen.json")
    gallery.frozen_inventory(family / "inputs/inventory/inventory_frozen.json")
    plan = gallery.load(family / "plan.json")
    control = gallery.control_inputs(family, plan)
    assert control["raw_cosines"].shape == (80, 3)
    assert [row["pool_index"] for row in plan["canonical_rows"]] == [0, 0, 1]
    assert not (family / "inference").exists()
    with pytest.raises(ValueError, match="isolated"):
        gallery.prepare(**ready)


@pytest.mark.parametrize("changed", ["control", "ledger", "checkpoint"])
def test_prepare_declines_changed_preregistered_inputs_before_copy(ready, changed):
    target = {"control": ready["parent"] / "inference/full_evidence.npz",
              "ledger": ready["inventory"].parent / "pixel_identity.json", "checkpoint": ready["checkpoint"]}[changed]
    with target.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        gallery.prepare(**ready)
    assert not ready["family"].exists()


@pytest.mark.parametrize("stop", [None, "query", "end", "saved_score"])
def test_mocked_actual_prepare_to_infer_freezes_all_four_and_closes_frames(ready, monkeypatch, stop):
    """Exercise real copied paths/freezes/NPYs/decoder with a synthetic encoder."""
    import sys
    from contextlib import contextmanager, nullcontext
    from types import ModuleType
    from monocular_osm.stages import poi_runtime
    gallery.prepare(**ready)
    monkeypatch.setattr(gallery, "source_binding", lambda *args: None)
    frames = SimpleNamespace(frames=[np.ones((2, 3, 3), np.uint8)] * 80, closed=False)
    frames.close = lambda: setattr(frames, "closed", True)
    monkeypatch.setattr(gallery, "native_queries", lambda *args: frames)
    monkeypatch.setattr(gallery.image, "DIMENSION", 3)
    model_calls, encode_counts = [], []
    monkeypatch.setattr(gallery.image, "load_model", lambda *args: model_calls.append(True) or (object(), {"mocked": True}))
    states = []
    def state(*args):
        states.append(True)
        if stop == "saved_score" and len(states) == 2:
            (ready["family"] / "inference/offline_scores.json").write_text("{}")
        return "f" * 64
    monkeypatch.setattr(gallery.image, "state_digest", state)
    def memory():
        if (stop == "end" and (ready["family"] / "inference/prediction_ImAge_GSV_raw.json").exists()
                or stop == "query" and len(encode_counts) == 3):
            raise ValueError("headroom changed")
        return {"available_physical_bytes": 10 * 1024**3, "available_commit_bytes": 11 * 1024**3}
    monkeypatch.setattr(gallery.sparse, "check_headroom", memory)
    def encode(views, model, torch):
        encode_counts.append(len(views))
        assert len(views) <= 4
        values = np.zeros((len(views), 3), np.float32)
        values[:, 0] = 1.
        return values
    monkeypatch.setattr(gallery.image, "encode", encode)
    torch = ModuleType("torch")
    torch.set_num_interop_threads = lambda n: None
    torch.inference_mode = nullcontext
    torch.from_numpy = lambda value: Tensor(value, [])
    torch.cuda = SimpleNamespace(is_initialized=lambda: False)
    tf = ModuleType("torchvision.transforms.functional")
    tf.resize = lambda tensor, size, antialias=True: tensor
    vision, transforms = ModuleType("torchvision"), ModuleType("torchvision.transforms")
    transforms.functional, vision.transforms = tf, transforms
    for name, value in {"torch": torch, "torchvision": vision, "torchvision.transforms": transforms,
                        "torchvision.transforms.functional": tf}.items():
        monkeypatch.setitem(sys.modules, name, value)
    @contextmanager
    def cpu(threads):
        assert threads == 4
        yield {"device": "cpu", "threads": 4}
    monkeypatch.setattr(poi_runtime, "torch_cpu_threads", cpu)
    if stop is not None:
        with pytest.raises(ValueError, match="headroom|score files"):
            gallery.infer(ready["family"], "synthetic_pin", gallery.PROFILE_SHA256)
        assert not (ready["family"] / "inference/frozen_predictions.json").exists()
        assert (ready["family"] / "inference/failure.json").exists() and frames.closed
        assert not list((ready["family"] / "raw_staging").glob("*.npy"))
        return
    result = gallery.infer(ready["family"], "synthetic_pin", gallery.PROFILE_SHA256)
    assert result["status"] == "complete" and result["selected_variant"] is None
    assert model_calls == [True] and encode_counts == [2, 1] + [4] * 60 and frames.closed
    frozen = gallery.sparse.verify_freeze(ready["family"] / "inference/frozen_predictions.json")
    assert len(frozen["variants"]) == 4 and frozen["selected_variant"] is None
    runtime = gallery.load(ready["family"] / "inference/runtime.json")
    assert runtime["legacy_reference_encoding_identity"] == "UNVERIFIED" and runtime["ground_truth_opened"] is False
    assert runtime["memory_end"]["available_commit_bytes"] == 11 * 1024**3
    assert runtime["memory_minimum"]["available_physical_bytes"] == 10 * 1024**3
    assert not list((ready["family"] / "raw_staging").glob("*.npy"))


@pytest.mark.parametrize("method", ["sha", "load", "verify_freeze", "frozen_inventory"])
@pytest.mark.parametrize("filename", ["offline_scores.json", "SCORES.JSON", "input_score.jsonl"])
def test_saved_score_names_decline_before_hash_or_load(monkeypatch, tmp_path, filename, method):
    def blocked(*args):
        pytest.fail("forbidden data must be declined before byte reads")
    if method in ("sha", "load"):
        monkeypatch.setattr(gallery.sparse, method, blocked)
        target = tmp_path / filename
    else:
        target = tmp_path / "freeze.json"
        target.write_text(json.dumps({"ground_truth_opened": False, "scoring_has_not_run": True,
            "artifacts": {filename: "0" * 64}}))
        monkeypatch.setattr(gallery.sparse, "sha", blocked)
    with pytest.raises(ValueError, match="score files"):
        getattr(gallery, method)(target)


def test_resolved_ground_truth_alias_is_declined_before_hash(tmp_path, monkeypatch):
    root = tmp_path / "ground_truth"
    root.mkdir()
    target = root / "route.dat"
    target.write_bytes(b"not allowed")
    alias = tmp_path / "apparently_safe.dat"
    try:
        alias.symlink_to(target)
    except OSError:
        pytest.skip("symlink capability unavailable")
    monkeypatch.setattr(gallery.sparse, "sha", lambda *a: pytest.fail("must reject alias before hash"))
    with pytest.raises(ValueError, match="ground_truth"):
        gallery.sha(alias)


def test_bad_raw_header_declines_before_mmap_and_preserves_original_error(tmp_path, monkeypatch):
    family = tmp_path / "family"
    (family / "inputs").mkdir(parents=True)
    archive = family / "inputs/bad.npz"
    np.savez(archive, ids=np.asarray(["a"]), xy=np.asarray([[42., -83.]]), raw=np.zeros((1, 2, 2, 3), np.uint8))
    plan = {"canonical_rows": [{"column": 0, "id": "mapillary:a", "source": gallery.key(archive),
        "raw_row": 0, "pixels_sha256": "0" * 64, "latlon": [42., -83.]}],
        "raw_sources": [{"source": gallery.key(archive), "copy": "inputs/bad.npz", "sha256": gallery.sha(archive),
                         "raw_shape": [1, 512, 512, 3], "pool_indices": [0]}]}
    monkeypatch.setattr(gallery.image, "DIMENSION", 3)
    monkeypatch.setattr(gallery.sparse, "check_headroom", lambda: {})
    monkeypatch.setattr(gallery.image, "extract_raw", lambda *a: pytest.fail("must not mmap invalid header"))
    output = family / "descriptors.npy"
    with pytest.raises(ValueError, match="raw header before mmap"):
        gallery.encode_gallery(family, plan, None, None, output)
    output.unlink()
    assert not list((family / "raw_staging").glob("*.npy"))


def test_memory_drift_within_pool_declines_before_subsequent_batch(tmp_path, monkeypatch):
    family = tmp_path / "family"
    (family / "inputs").mkdir(parents=True)
    archive = family / "inputs/six.npz"
    raw = np.tile(np.array([1, 2, 3], np.uint8), (6, 512, 512, 1))
    ids, positions = [str(i) for i in range(6)], np.tile([42., -83.], (6, 1))
    np.savez(archive, ids=np.asarray(ids), xy=positions, raw=raw)
    rows = [{"column": i, "id": "mapillary:" + identifier, "source": gallery.key(archive), "raw_row": i,
             "pixels_sha256": digest(raw[i]), "latlon": positions[i].tolist()} for i, identifier in enumerate(ids)]
    plan = {"canonical_rows": rows, "raw_sources": [{"source": gallery.key(archive), "copy": "inputs/six.npz",
        "sha256": gallery.sha(archive), "raw_shape": list(raw.shape), "pool_indices": [0]}]}
    monkeypatch.setattr(gallery.image, "DIMENSION", 3)
    checks, forwards = [], []
    def memory():
        checks.append(True)
        if len(checks) == 3:
            raise ValueError("headroom drift")
        return {}
    def encoder(views, model, torch):
        forwards.append(len(views))
        return fake_encoder([])(views, model, torch)
    monkeypatch.setattr(gallery.sparse, "check_headroom", memory)
    output = family / "descriptors.npy"
    with pytest.raises(ValueError, match="headroom drift"):
        gallery.encode_gallery(family, plan, None, None, output, encoder=encoder)
    assert forwards == [4] and not list((family / "raw_staging").glob("*.npy"))
    output.unlink()
