"""Input-only native80 replay for an explicitly frozen raw-VPR metadata bank.

No retrieval scores, reference coordinates, descriptors, poses, OCR, model
factory, forward, GT or policy selection are loaded. This verifies query input
pixels/clock, not the historical encoder computations or reference encoding.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NATIVE_SHA = "9901e8c6497a30ddf0acff0f804968e1d4f7e38b40a370bdcee007c5eef078bf"
DEPENDENCY_SHA = "f8c30277a3f9ea1a7d031acb0067c18ec95b879adf8edc25cec78238168be068"
SAMPLER = ROOT / "monocular_osm/frame_extraction.py"
SAMPLER_SHA = "9bd5fe5cf63800a2f2606958eb360364bd26c07d1f171120aea2245565695a64"
SOURCE_PATHS = ("monocular_osm/raw_vpr_native.py", "scripts/replay_raw_vpr_native.py",
                "monocular_osm/kartaview_vpr.py", "monocular_osm/stages/acquire.py",
                "monocular_osm/stages/trajectory.py", "monocular_osm/pipeline.py")
APPEARANCE = ("backbone", "actual_model_state_sha256", "preprocessing", "query_mode",
              "query_pixels_sha256", "query_descriptors_sha256")
EXTRACTION = ("vo_start_sec", "vo_end_sec", "frame_stride", "max_frames", "frame_max_width")
PROFILE = {"method": "all-bound-own-request-native80-pixel-clock-replay-v1", "max_calls": 128,
           "max_banks": 2, "max_calls_per_bank": 64, "native_auditor_sha256": NATIVE_SHA, "sampler_sha256": SAMPLER_SHA,
           "query_rows": 80, "max_dense_frames": 20_000, "composition_interval_s": .5,
           "model_forward_recomputed": False, "descriptor_values_recomputed": False,
           "reference_encoding_upgraded": False, "reference_encoding": "UNVERIFIED", "child_completion_certified": False}


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def typed(value):
    array = np.ascontiguousarray(value)
    h = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    if array.size:
        h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def key(path):
    return os.path.normcase(str(Path(path).resolve()))


def ref(value):
    need(isinstance(value, dict) and set(value) == {"path", "sha256"}
         and isinstance(value["path"], str) and is_sha(value["sha256"]), "exact path/SHA role required")
    return key(value["path"]), value["sha256"]


def call_key(bank, role):
    value = (str(bank).strip("/\\") + "/" + str(role).strip("/\\")).replace("\\", "/")
    need(value and all(p not in {"", ".", ".."} for p in value.split("/")), "invalid logical call key")
    return value


def protected_map(value):
    records = value if isinstance(value, list) else [{"path": p, "sha256": h} for p, h in value.items()]
    result = {}
    for record in records:
        path, expected = ref({"path": record["path"], "sha256": record["sha256"]})
        need(path not in result, "duplicate normalized protected path")
        result[path] = expected
    return result


def owner_membership(binding, call, loader):
    """Positive owner freeze membership only; never certify child completion."""
    freeze_ref = binding["owner_freeze_ref"]
    owner = loader(freeze_ref)
    root = Path(freeze_ref["path"]).resolve().parent
    members = {}
    for name, expected in owner["artifacts"].items():
        normalized = name.replace("\\", "/")
        need(not Path(normalized).is_absolute() and ":" not in normalized
             and all(p not in {"", ".", ".."} for p in normalized.split("/")) and is_sha(expected), "unsafe owner member")
        canonical = key(root / normalized)
        need(canonical not in members, "duplicate normalized owner member")
        members[canonical] = expected
    for role in ("request_ref", "metadata_ref", "array_ref"):
        path, expected = ref(call[role])
        need(Path(path).is_relative_to(root) and members.get(path) == expected, "call is not exact member of OWN freeze")


class Ledger:
    """One bounded metadata counter; all expected bytes checked again at end."""
    def __init__(self, check=lambda: None):
        self.protected = {}
        self.total_json = 0
        self.check = check

    def protect(self, record):
        path, expected = ref(record)
        self.check()
        need(sha(path) == expected, "input SHA differs: " + path)
        need(self.protected.setdefault(path, expected) == expected, "conflicting expected path hashes")
        return Path(path)

    def json(self, record):
        path, _ = ref(record)
        size = Path(path).stat().st_size
        need(size <= 16 << 20 and self.total_json + size <= 256 << 20, "metadata budget before body read")
        self.total_json += size
        path = self.protect(record)
        def pairs(items):
            out = {}
            for k, v in items:
                need(k not in out, "duplicate JSON key")
                out[k] = v
            return out
        return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                          parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))

    def members(self, record, names):
        path = self.protect(record)
        # Inspect ONLY requested NPY headers and load ONLY requested members.
        with zipfile.ZipFile(path) as archive:
            entries = archive.namelist()
            need(len(entries) == len(set(entries)) <= 128, "NPZ member/alias cap")
            total = 0
            for name in names:
                member = name + ".npy"
                need(member in entries and "/" not in member and "\\" not in member, "NPZ role absent")
                info = archive.getinfo(member)
                total += info.file_size
                need(total <= 4 << 20, "selected numerical-member budget")
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    need(version in {(1, 0), (2, 0)}, "unsupported NPY version")
                    shape, order, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                                           else np.lib.format.read_array_header_2_0(stream))
                    need(not order and not dtype.hasobject and len(shape) <= 2
                         and math.prod(shape) * dtype.itemsize + stream.tell() == info.file_size,
                         "unsafe NPY storage/header")
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in names}

    def unchanged(self):
        # All user/resource callbacks precede this callback-free final pass.
        self.check()
        for path, expected in self.protected.items():
            need(sha(path) == expected, "protected bytes changed: " + path)


def load_native_grid(ledger, native_ref, dependency_ref):
    need(native_ref["sha256"] == NATIVE_SHA and dependency_ref["sha256"] == DEPENDENCY_SHA,
         "exact original native/dependency source pins required")
    native = ledger.protect(native_ref)
    dependency = ledger.protect(dependency_ref)
    need(dependency == native.with_name("audit_prepared.py"), "true original sibling dependency required")
    ledger.protect({"path": str(SAMPLER), "sha256": SAMPLER_SHA})
    name = "_raw_vpr_genuine_native9901"
    need(name not in sys.modules, "fresh genuine native replay admission required")
    spec = importlib.util.spec_from_file_location(name, native)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    need(Path(module.__file__).resolve() == native
         and Path(module.native_grid.__code__.co_filename).resolve() == native, "native replay origin differs")
    return module.native_grid


def inventory_rows(inventory):
    need(inventory.get("schema") == "closed-raw-vpr-metadata-bank-inventory-v1", "wrong metadata inventory")
    banks = inventory["banks"]
    need(isinstance(banks, list) and 1 <= len(banks) <= PROFILE["max_banks"], "bank bound")
    rows = {}
    for bank in banks:
        need(1 <= len(bank["pair_records"]) <= PROFILE["max_calls_per_bank"], "per-bank physical-call bound")
        for item in bank["pair_records"]:
            name = call_key(bank["key"], item["role"])
            need(name not in rows and item.get("issues") in (None, []), "duplicate/invalid inventory call")
            rows[name] = item
    need(1 <= len(rows) <= PROFILE["max_calls"], "physical-call bound")
    return rows


def make_plan(inventory, calls, bindings, request_loader, metadata_loader, *, request_execution_root=None):
    """Metadata only: preserve EVERY physical alias and OWN request association."""
    rows = inventory_rows(inventory)
    need(isinstance(calls, list) and len(calls) == len(rows), "ALL physical inventory calls required")
    need(bindings.get("schema") == "root-exact-raw-bank-own-request-bindings-v1", "explicit Root request bindings required")
    bound = {r["key"]: r for r in bindings["calls"]}
    need(len(bound) == len(bindings["calls"]) == bindings.get("all_inventory_rows_bound")
         and set(bound) == set(rows), "request-binding membership differs")
    protected = protected_map(inventory["protected_files"])
    binding_files = protected_map(bindings["protected_files"])
    plans, common, seen = [], None, set()
    for call in calls:
        name = call["key"]
        need(name in rows and name not in seen, "call omitted/repeated/extra")
        seen.add(name)
        original = rows[name]
        for role, path_field, sha_field in (("metadata_ref", "metadata", "metadata_sha256"),
                                             ("array_ref", "npz", "npz_sha256")):
            actual_path, expected = ref(call[role])
            need(actual_path == key(original[path_field]) and expected == original[sha_field]
                 and protected.get(actual_path) == expected, "call role outside exact frozen inventory")
        binding = bound[name]
        for role in ("request_ref", "metadata_ref", "array_ref"):
            path, expected = ref(call[role])
            need((path, expected) == ref(binding[role]) and binding_files.get(path) == expected,
                 "OWN physical role differs from Root binding")
        owner_path, owner_sha = ref(binding["owner_freeze_ref"])
        need(binding_files.get(owner_path) == owner_sha, "owner freeze outside Root protected bindings")
        owner_membership(binding, call, metadata_loader)
        request, meta = request_loader(call["request_ref"]), metadata_loader(call["metadata_ref"])
        need(request.get("kind") in {"baseline", "expanded"} and isinstance(request.get("execution_id"), str)
             and request["execution_id"] and is_sha(request.get("video_sha256")), "ordinary OWN request identity missing")
        cfg = request["config"]
        output = Path(cfg["output_dir"])
        if not output.is_absolute():
            need(request_execution_root is not None and Path(request_execution_root).is_absolute(),
                 "relative OWN output requires explicit historical request_execution_root")
            output = Path(request_execution_root) / output
        need(Path(key(original["metadata"])).is_relative_to(Path(key(output))), "metadata not under OWN request output")
        need(meta.get("status") == "complete" and not meta.get("issues")
             and meta.get("score_kind") == "raw_cosine_at_decoded_reference" and meta.get("query_mode") == "warp",
             "complete original warp audit required")
        artifact = meta["array_artifact"]
        need(artifact["sha256"] == call["array_ref"]["sha256"]
             and protected.get(key(artifact["path"])) == artifact["sha256"], "metadata array pointer outside protected source")
        headers = artifact["arrays"]
        appearance = {k: meta[k] for k in APPEARANCE}
        for name_array in ("query_indices", "query_times"):
            header = headers[name_array]
            need(header == original["raw_headers"][name_array] and header["shape"] == [80]
                 and is_sha(header["sha256"]), "exact typed80 query header missing")
            appearance[name_array + "_header"] = header
        need(appearance == original["appearance_identity"], "inventory and actual appearance differ")
        identity = {k: meta[k] for k in (*APPEARANCE, "score_kind", "decoder")}
        identity.update(query_indices_sha256=headers["query_indices"]["sha256"],
                        query_times_sha256=headers["query_times"]["sha256"])
        need(meta.get("comparison_identity") == identity and meta.get("comparison_identity_sha256") == digest(identity),
             "original comparison identity is stale")
        for name_hash in ("query_pixels_sha256", "query_descriptors_sha256", "actual_model_state_sha256"):
            need(is_sha(meta[name_hash]), "actual saved appearance hash missing")
        if common is None:
            common = appearance
        need(common == appearance, "whole-bank actual model/query/typed clock conflict")
        need(cfg.get("max_frames") is None and cfg.get("frame_max_width") is None
             and type(cfg.get("frame_stride")) is int and cfg["frame_stride"] > 0, "native uncapped/unresized extraction required")
        need(cfg.get("answer") == "track" and cfg.get("use_vpr_prior") is True
             and type(cfg.get("retrieval_only")) is bool and type(cfg.get("use_ipm_scale")) is bool, "explicit actual extraction predicate required")
        sparse = cfg["retrieval_only"] and not cfg["use_ipm_scale"]
        extraction = {k: cfg[k] for k in EXTRACTION}
        need(type(extraction["vo_start_sec"]) in (int, float) and math.isfinite(extraction["vo_start_sec"])
             and 0 <= extraction["vo_start_sec"] < extraction["vo_end_sec"]
             and math.isfinite(extraction["vo_end_sec"]), "finite full registered window required")
        plans.append({"key": call["key"], "request_ref": call["request_ref"], "metadata_ref": call["metadata_ref"],
                      "array_ref": call["array_ref"], "video_sha256": request["video_sha256"],
                      "execution_id": request["execution_id"], "source_sha256": request["source_sha256"],
                      "extraction": extraction, "sparse_query_only": sparse,
                      "profile_key": digest(extraction), "appearance_identity": appearance})
    need(len({p["video_sha256"] for p in plans}) == 1 and len({p["execution_id"] for p in plans}) == 1
         and len({digest(p["source_sha256"]) for p in plans}) == 1, "bank source/video/execution conflict")
    membership = {b["key"]: [call_key(b["key"], p["role"]) for p in b["pair_records"]] for b in inventory["banks"]}
    need(len(membership) == len(inventory["banks"]), "duplicate bank key")
    return {"calls": plans, "appearance_identity": common, "bank_membership": membership,
            "request_execution_root": str(Path(request_execution_root).resolve()) if request_execution_root is not None else None,
            "all_physical_aliases_retained": True}


def query_vo_join(query_times, vo_times, max_lag_s):
    need(type(max_lag_s) in (int, float) and math.isfinite(max_lag_s) and 0 <= max_lag_s <= 2., "declared bounded lag required")
    need(query_times.shape == (80,) and query_times.dtype == np.dtype("float64")
         and vo_times.dtype == np.dtype("float64") and vo_times.ndim == 1 and len(vo_times) >= 2
         and np.isfinite(vo_times).all() and np.all(np.diff(vo_times) > 0)
         and np.isfinite(query_times).all() and np.all(np.diff(query_times) > 0)
         and vo_times[0] <= query_times[0] and query_times[-1] <= vo_times[-1], "native query outside signed VO horizon")
    ends, previous = [0], 0
    for i in range(1, len(vo_times)):
        if vo_times[i] - vo_times[previous] >= .5:
            ends.append(i)
            previous = i
    if ends[-1] != len(vo_times) - 1:
        ends.append(len(vo_times) - 1)
    composed = vo_times[ends]
    rows = []
    for t in query_times:
        right = int(np.searchsorted(vo_times, t, side="left"))
        exact = right if vo_times[right] == t else None
        left = right if exact is not None else right - 1
        step = int(np.searchsorted(composed, t, side="left"))
        lag = float(composed[step] - t)
        need(0 <= lag <= max_lag_s, "query-to-composed-clock lag exceeds declaration")
        rows.append({"t_sec": float(t), "vo_exact_index": exact, "vo_left_index": left,
                     "vo_right_index": right, "vo_left_t_sec": float(vo_times[left]),
                     "vo_right_t_sec": float(vo_times[right]), "composed_step": step,
                     "composed_t_sec": float(composed[step]), "composed_lag_s": lag})
    return rows


def verify_vo(grid, vo, signature, video):
    times = vo["timestamps"]
    extraction = signature["extraction"]
    need(signature.get("schema") == 2 and signature.get("vo_algo_version") == 3
         and signature["video"]["size"] == video["size"]
         and key(signature["video"]["path"]) == key(video["path"]), "signed video/VO identity differs")
    need(times.dtype == np.dtype("float64") and times.shape == (extraction["n_frames"],)
         and np.array_equal(times, grid["timestamps"]) and grid["dense_count"] == extraction["n_frames"]
         and hashlib.sha256(np.asarray(times, dtype="<f8").tobytes()).hexdigest() == extraction["timestamps_sha256"]
         and grid["frames_sha256"] == signature["frames_sha256"]
         and grid["fps"] == extraction["fps"] and grid["total_frames"] == extraction["total_frames"],
         "signed full native VO pixels/clock/extraction differ")
    h, w, channels = grid["native_shape"]
    fx = (w / 2.) / np.tan(np.deg2rad(70.) / 2.)
    expected = np.array([[fx, 0., w / 2.], [0., fx, h / 2.], [0., 0., 1.]], dtype=np.float64)
    need(channels == 3 and vo["K"].dtype == np.dtype("float64") and np.array_equal(vo["K"], expected), "native70 camera matrix differs")


def replay(plan, video, vo, signature, grids_loader, native_grid, *, max_lag_s, check=lambda: None):
    """Only two retrieval-grid arrays per physical pair; one decode per profile."""
    query_arrays = {}
    # Admit EVERY original typed query grid before opening even the first video.
    for call in plan["calls"]:
        check()
        need(call["video_sha256"] == video["sha256"], "query and native VO encoded video hashes differ")
        z = grids_loader(call["array_ref"])
        need(set(z) == {"query_indices", "query_times"}, "only two retrieval-grid arrays admitted")
        identity = call["appearance_identity"]
        for name in ("query_indices", "query_times"):
            header = identity[name + "_header"]
            need(z[name].shape == tuple(header["shape"]) and z[name].dtype.str == header["dtype"]
                 and typed(z[name]) == header["sha256"], "query-grid actual bytes differ from frozen header")
        need(z["query_indices"].dtype.kind in "iu" and z["query_times"].dtype == np.dtype("float64")
             and np.isfinite(z["query_times"]).all() and np.all(np.diff(z["query_times"]) > 0), "invalid actual80 clock")
        query_arrays[call["key"]] = z
    cache = {}
    def grid_for(cfg):
        profile_key = digest(cfg)
        if profile_key not in cache:
            check()
            cache[profile_key] = native_grid(video["path"], cfg, check=check)
        return cache[profile_key]
    e = signature["extraction"]
    vo_config = dict(zip(EXTRACTION, (e["start_sec"], e["end_sec"], e["stride"], e["max_frames"], e["frame_max_width"])))
    verify_vo(grid_for(vo_config), vo, signature, video)
    rows = []
    for call in plan["calls"]:
        check()
        grid = grid_for(call["extraction"])
        need(len(grid["query_times"]) == 80, "full native80 query count required")
        z = query_arrays[call["key"]]
        expected_indices = np.arange(80, dtype=np.int64) if call["sparse_query_only"] else np.asarray(grid["dense_indices"], dtype=np.int64)
        pixel_sha = grid["query_pixels_sha256"] if call["sparse_query_only"] else grid["dense_query_pixels_sha256"]
        identity = call["appearance_identity"]
        need(np.array_equal(z["query_indices"], expected_indices) and np.array_equal(z["query_times"], grid["query_times"])
             and identity["query_pixels_sha256"] == pixel_sha, "OWN query indices/native PTS/BGR pixels differ")
        joined = query_vo_join(z["query_times"], vo["timestamps"], max_lag_s)
        for i, row in enumerate(joined):
            row.update(query_row=i, local_query_index=int(z["query_indices"][i]),
                       dense_index=int(grid["dense_indices"][i]),
                       native_ordinal=int(grid["first_ordinal"] + grid["dense_indices"][i] * grid["stride"]))
        rows.append({"key": call["key"], "request_ref": call["request_ref"], "metadata_ref": call["metadata_ref"],
                     "array_ref": call["array_ref"], "profile_key": call["profile_key"],
                     "query_pixels_sha256": pixel_sha, "query_indices_sha256": typed(z["query_indices"]),
                     "query_times_sha256": typed(z["query_times"]), "query_to_vo": joined})
    return {"schema": "raw-vpr-own-native80-replay-v1", "status": "VERIFIED_ALL_BOUND_OWN_NATIVE80_PIXELS_CLOCK_NO_NN",
            "profile": PROFILE, "video_sha256": video["sha256"], "appearance_identity": plan["appearance_identity"],
            "bank_membership": plan["bank_membership"],
            "request_execution_root": plan["request_execution_root"],
            "all_physical_calls_and_aliases": rows, "unique_native_decodes": len(cache),
            "signed_vo_clock_sha256": typed(vo["timestamps"]), "native70_K_sha256": typed(vo["K"]),
            "declared_max_composed_lag_s": max_lag_s, "actual_timestamps_unchanged": True,
            "query_descriptors_and_model_values_recomputed": False, "reference_encoding_upgraded": False,
            "reference_encoding": "UNVERIFIED",
            "original_child_completion_certified": False, "ground_truth_used": False, "selected_answer": None}


def validate_native_facts(facts, records, bank_key):
    """Pure admission seam: complete exact named-bank membership, no subset."""
    need(facts.get("schema") == "raw-vpr-own-native80-replay-v1"
         and facts.get("status") == "VERIFIED_ALL_BOUND_OWN_NATIVE80_PIXELS_CLOCK_NO_NN"
         and facts.get("profile") == PROFILE and facts.get("reference_encoding_upgraded") is False
         and facts.get("reference_encoding") == "UNVERIFIED"
         and facts.get("query_descriptors_and_model_values_recomputed") is False
         and facts.get("original_child_completion_certified") is False, "positive input-only native facts required")
    all_rows = facts["all_physical_calls_and_aliases"]
    by_key = {row["key"]: row for row in all_rows}
    full_members = [k for group in facts["bank_membership"].values() for k in group]
    need(len(by_key) == len(all_rows) == len(full_members) and len(set(full_members)) == len(full_members)
         and set(by_key) == set(full_members), "native proof inventory membership differs")
    expected = facts["bank_membership"].get(bank_key)
    need(expected is not None and len(records) == len(expected)
         and len({record["key"] for record in records}) == len(records)
         and {record["key"] for record in records} == set(expected), "ALL named-bank physical calls required")
    appearance = facts["appearance_identity"]
    for record in records:
        row = by_key[record["key"]]
        for role in ("request_ref", "metadata_ref", "array_ref"):
            need(ref(record[role]) == ref(row[role]), "native facts belong to different physical/OWN request source")
        need(row["query_pixels_sha256"] == appearance["query_pixels_sha256"]
             and row["query_indices_sha256"] == appearance["query_indices_header"]["sha256"]
             and row["query_times_sha256"] == appearance["query_times_header"]["sha256"]
             and len(row["query_to_vo"]) == 80, "native facts differ from all80 actual appearance/clock")
    return appearance
