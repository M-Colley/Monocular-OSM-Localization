"""Within-execution public reference snapshots, without query evidence reuse.

This copies the exact reference bytes a completed baseline consumed. Historical
descriptor encoding remains UNVERIFIED; it is never inferred from a filename.
"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile
import numpy as np
from .provenance import file_sha256, library_versions

SCHEMA = "within-run-public-reference-cache-v1"
PROVIDERS = {"mapillary", "kartaview", "panoramax"}
FAMILIES = {name for source in sorted(PROVIDERS) for name in
            (source, source+"_c2f", source+"_outer", source+"_regions", source+"_city")}
FAMILIES.update({"union", "union_c2f", "union_regions", "union_city"})
MAX_ROWS, MAX_EMBED_BYTES = 20000, 512 * 1024**2
_PAYLOAD = re.compile(r"(?:mly_ref_(?:meta|store)(?:_[0-9a-f]{10})?\.json|ref_meta\.json|pnx_ref_meta\.json|ref_img_store\.npz|ref_emb_[A-Za-z0-9_-]+\.npz)$")


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    if value.size:
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _pool_identity(record):
    names = ("schema", "payload_mode", "request", "location", "backbone", "actual_model_state_sha256", "preprocessing",
             "kv_source_sha256", "runtime_versions", "ref_ids_sha256", "ref_latlon_sha256",
             "ref_embeddings_sha256", "fingerprint", "count", "servable_refs", "files",
             "reference_encoding_identity", "claim", "full_request_origin", "metadata_signature_precision")
    return {name: record[name] for name in names}


def _safe(path, root=None):
    path = Path(path)
    for name in (str(path), str(path.resolve())):
        folded = name.replace("\\", "/").casefold()
        _require("ground_truth" not in folded and not re.search(
            r"(?:^|[/_.-])(?:scores?|scoring)(?:[_.-]|$).*\.jsonl?$", folded),
            "query labels and saved scores are forbidden reference inputs")
    resolved = path.resolve()
    if root is not None:
        _require(resolved.is_relative_to(Path(root).resolve()), "reference path escaped its checked root")
    return resolved


def _read(path):
    path = _safe(path)
    _require(path.stat().st_size <= 64 * 1024**2, "reference JSON exceeds bound")
    return json.loads(path.read_text(encoding="utf8"))


def _sha(path):
    return file_sha256(_safe(path))


def _npy_header(stream):
    version = np.lib.format.read_magic(stream)
    _require(version in ((1, 0), (2, 0)), "unsupported reference NPY header")
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream)
    _require(not fortran and not dtype.hasobject, "reference arrays must be plain C-order values")
    return shape, dtype


def _embedding(path, fingerprint, count, dimensions, expected):
    """Validate and hash float32 rows in bounded chunks, never loading a pool."""
    with np.load(path, allow_pickle=False) as saved:
        _require(str(saved["fingerprint"]) == fingerprint, "reference descriptor fingerprint changed")
    with zipfile.ZipFile(path) as archive:
        _require(archive.namelist().count("emb.npy") == 1, "missing/duplicate descriptor member")
        with archive.open("emb.npy") as stream:
            shape, dtype = _npy_header(stream)
            _require(shape == (count, dimensions) and dtype == np.float32
                and count * dimensions * 4 <= MAX_EMBED_BYTES, "descriptor header exceeds fixed identity/budget")
            # Match kv._audit_tensor_sha exactly, not the NumPy-array hash
            # used for ID/coordinate artifacts (the two have different tags).
            digest = hashlib.sha256(json.dumps([list(shape), "torch.float32"], separators=(",", ":")).encode())
            rows = max(1, 8 * 1024**2 // (dimensions * 4))
            remaining = count
            while remaining:
                take = min(rows, remaining)
                block = stream.read(take * dimensions * 4)
                _require(len(block) == take * dimensions * 4, "truncated reference descriptors")
                values = np.frombuffer(block, dtype=dtype).reshape(take, dimensions)
                _require(np.isfinite(values).all() and np.max(np.abs(np.linalg.norm(values, axis=1)-1)) <= 1e-4,
                         "reference descriptors are not finite normalized rows")
                digest.update(block)
                remaining -= take
            _require(stream.read(1) == b"", "unexpected reference descriptor tail")
    _require(digest.hexdigest() == expected, "descriptor tensor differs from completed baseline audit")


def _location(cfg, directory):
    directory = _safe(directory)
    local = _safe(cfg.data_dir)
    if directory.is_relative_to(local):
        relative = directory.relative_to(local)
        _require(relative.parts and relative.parts[0] in FAMILIES, "not a public reference cache family")
        return {"scope": "local", "relative": relative.as_posix(), "root": str(local)}
    external = getattr(cfg, "vpr_region_cache_dir", None)
    _require(external is not None, "reference directory is outside the source data root")
    root = _safe(Path(external) / Path(cfg.data_dir).name)
    relative = _safe(directory, root).relative_to(root)
    _require(relative.parts and re.match(r"^(?:mapillary|kartaview|panoramax|union)_(?:region|city)_", relative.parts[0]),
             "not a checked external reference family")
    return {"scope": "external", "relative": relative.as_posix(), "root": str(root)}


def public_request(bound):
    """Only acquisition/config scalars; never frames, query arrays, or tokens."""
    source = bound.get("source", "kartaview")
    _require(source in PROVIDERS, "combined/outer evidence needs a separate per-provider proof")
    _require(not bound.get("outer_radius_m"), "outer-ring evidence has no per-pool reference proof")
    center = [float(v) for v in bound["center"]]
    radius, cap = float(bound.get("radius_m", 3000.)), int(bound.get("cap", 1500))
    _require(len(center) == 2 and np.isfinite(center).all() and radius > 0 and 0 < cap <= MAX_ROWS,
             "invalid full reference request")
    return {"source": source, "center": center, "radius_m": radius, "cap": cap,
        "model_name": bound.get("model_name", "megaloc"), "device": bound.get("device"),
        "query_mode": bound.get("query_mode", "warp"), "query_batch_size": bound.get("query_batch_size"),
        "cache_dir": str(_safe(bound["cache_dir"]))}


def describe_pool(audit, cfg):
    """Seal one completed pool using reference fields, never rankings/scores."""
    from . import kartaview_vpr as kv
    request = deepcopy(audit["public_cache_request"])
    directory = _safe(request["cache_dir"])
    location = _location(cfg, directory)
    if audit.get("terminal_search"):
        return _terminal_pool(audit["terminal_search"], request, directory, location)
    pool = audit.get("reference_pool") or {}
    _require(audit.get("status") == "complete" and not audit.get("provider_error"),
             "no completed single-pool descriptor audit")
    _require(all(pool.get(name) == request[name] for name in ("source", "center", "radius_m", "cap")),
             "full actual request differs from completed reference audit")
    identifiers = np.asarray(audit["ref_ids"])
    coordinates = np.asarray(audit["ref_latlon"], float)
    count = len(identifiers)
    _require(0 < count <= request["cap"] and coordinates.shape == (count, 2)
        and len(set(identifiers.tolist())) == count and np.isfinite(coordinates).all(), "invalid reference row inventory")
    prefix = request["source"] + ":"
    _require(all(str(value).startswith(prefix) for value in identifiers), "reference provider IDs changed")
    plain = [str(value)[len(prefix):] for value in identifiers]
    signature = kv._fetch_signature(request["center"], request["radius_m"], request["cap"])
    if request["source"] == "mapillary":
        names = [f"mly_ref_meta_{kv._sig_tag(signature)}.json", "mly_ref_meta.json"]
    else:
        names = ["ref_meta.json" if request["source"] == "kartaview" else "pnx_ref_meta.json"]
    metadata, blob = None, None
    for name in names:
        candidate = _safe(directory / name, directory)
        if candidate.is_file():
            value = _read(candidate)
            if value.get("signature") == signature:
                metadata, blob = candidate, value
                break
    _require(metadata is not None and isinstance(blob.get("refs"), list), "signed provider metadata missing")
    wanted = set(plain)
    refs = [row for row in blob["refs"] if str(row["id"]) in wanted]
    _require([str(row["id"]) for row in refs] == plain
        and np.array_equal(np.asarray([[row["lat"], row["lon"]] for row in refs], float), coordinates),
        "servable reference order/positions differ from provider metadata")
    fingerprint = kv._refs_fingerprint(refs)
    raw_path = _safe(directory / "ref_img_store.npz", directory)
    with zipfile.ZipFile(raw_path) as archive:
        _require(all(archive.namelist().count(name+".npy") == 1 for name in ("ids", "xy", "raw")),
                 "invalid per-ID public pixel store")
        with archive.open("raw.npy") as stream:
            shape, dtype = _npy_header(stream)
            _require(len(shape) == 4 and shape[1:] == (512, 512, 3) and dtype == np.uint8
                and 0 < shape[0] <= MAX_ROWS, "unsupported reference pixel grid")
        for name in ("ids", "xy"):
            with archive.open(name+".npy") as stream:
                small_shape, small_dtype = _npy_header(stream)
                _require(small_shape[0] == shape[0] and np.prod(small_shape) * small_dtype.itemsize <= 32 * 1024**2,
                         "reference ID/position header exceeds bound")
    with np.load(raw_path, allow_pickle=False) as saved:
        raw_ids = [str(value) for value in saved["ids"]]
        raw_xy = np.asarray(saved["xy"], float)
    _require(len(raw_ids) == len(set(raw_ids)) == shape[0] and raw_xy.shape == (len(raw_ids), 2),
             "invalid pixel-store ID order")
    indices = {identifier: i for i, identifier in enumerate(raw_ids)}
    _require(all(identifier in indices for identifier in plain)
        and np.array_equal(np.asarray([raw_xy[indices[value]] for value in plain]), coordinates),
        "pixel IDs/positions do not cover the consumed descriptor rows")
    backbone, preprocessing = deepcopy(audit["backbone"]), deepcopy(audit["preprocessing"])
    model = backbone["name"]
    _require(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", model) and len(audit["actual_model_state_sha256"]) == 64,
             "resolved backbone identity unavailable")
    embedding = None
    for name in (f"ref_emb_{model}_{fingerprint[:10]}.npz", f"ref_emb_{model}.npz"):
        candidate = _safe(directory / name, directory)
        if candidate.is_file():
            try:
                _embedding(candidate, fingerprint, count, int(preprocessing["dimensions"]), audit["ref_embeddings_sha256"])
                embedding = candidate
                break
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                continue
    _require(embedding is not None, "no descriptor cache matches the consumed tensor hash")
    paths = [(metadata, "metadata"), (raw_path, "pixels"), (embedding, "descriptors")]
    if request["source"] == "mapillary":
        image_store = directory / f"mly_ref_store_{kv._sig_tag(signature)}.json"
        if image_store.is_file():
            paths.append((_safe(image_store, directory), "image_store"))
    files = []
    for path, role in paths:
        _require(_PAYLOAD.fullmatch(path.name), "non-public file in reference payload")
        files.append({"name": path.name, "role": role, "sha256": _sha(path), "bytes": path.stat().st_size})
    record = {"status": "complete", "schema": SCHEMA, "payload_mode": "descriptors_and_pixels", "request": request, "location": location,
        "backbone": backbone, "actual_model_state_sha256": audit["actual_model_state_sha256"],
        "preprocessing": preprocessing, "kv_source_sha256": _sha(Path(kv.__file__)),
        "reference_encoding_identity": audit.get("reference_encoding_identity", "UNVERIFIED"),
        "claim": "same consumed reference snapshot; historical encoding remains unverified",
        "metadata_signature": signature, "full_request_origin": "completed baseline call",
        "metadata_signature_precision": "legacy rounded4 centre; full call identity retained separately",
        "ref_ids_sha256": _array_sha(identifiers), "ref_latlon_sha256": _array_sha(coordinates),
        "ref_embeddings_sha256": audit["ref_embeddings_sha256"], "fingerprint": fingerprint,
        "descriptor_hash_schema": "kv_audit_tensor_v1",
        "count": count, "listed_count": len(blob["refs"]), "pixel_count": len(raw_ids),
        "servable_refs": [{key: value for key, value in row.items() if key in {"id", "lat", "lon", "url"}} for row in refs],
        "runtime_versions": library_versions(("numpy", "opencv-python", "torch", "torchvision")),
        "listing": deepcopy(pool.get("fetch")), "files": files}
    record["pool_id"] = _digest(_pool_identity(record))
    return record


def _terminal_pool(terminal, request, directory, location):
    """A proven completed sparse/empty Mapillary search needs metadata only."""
    from . import kartaview_vpr as kv
    count, refs = terminal["count"], terminal["reference_inventory"]
    listing, snapshot = terminal["listing"], terminal["snapshot"]
    signature = kv._fetch_signature(request["center"], request["radius_m"], request["cap"])
    _require(type(count) is int and 0 <= count < 30 and len(refs) == count
        and terminal["status"] in {"complete_empty", "complete_sparse"}
        and request["source"] == terminal["source"] == "mapillary" and terminal["fetch_signature"] == signature
        and listing.get("listing_stats_known") is True and type(listing.get("lost")) is int and listing["lost"] == 0
        and listing.get("listing") == "tiles" and not listing.get("truncated")
        and listing["snapshot"] == snapshot and snapshot["n_refs"] == count
        and kv._reference_snapshot(refs)["refs_sha256"] == snapshot["refs_sha256"],
        "sparse search lacks a completed public listing proof")
    metadata = None
    for name in (f"mly_ref_meta_{kv._sig_tag(signature)}.json", "mly_ref_meta.json"):
        path = _safe(directory / name, directory)
        if path.is_file():
            blob = _read(path)
            if blob.get("signature") == signature and blob.get("refs") == refs:
                metadata = path
                break
    _require(metadata is not None, "completed sparse search has no persistent signed metadata payload")
    identifiers = np.asarray(["mapillary:" + str(row["id"]) for row in refs], dtype=str)
    coordinates = np.asarray([[row["lat"], row["lon"]] for row in refs], float).reshape(-1, 2)
    record = {"status": "complete", "schema": SCHEMA, "payload_mode": "metadata_only", "request": request,
        "location": location, "backbone": None, "actual_model_state_sha256": None, "preprocessing": None,
        "kv_source_sha256": _sha(Path(kv.__file__)),
        "runtime_versions": library_versions(("numpy", "opencv-python", "torch", "torchvision")),
        "reference_encoding_identity": "not applicable: completed metadata-only search",
        "claim": "same completed public listing; no image/descriptor coverage claim",
        "metadata_signature": signature, "full_request_origin": "completed baseline call",
        "metadata_signature_precision": "legacy rounded4 centre; full call identity retained separately",
        "ref_ids_sha256": _array_sha(identifiers), "ref_latlon_sha256": _array_sha(coordinates),
        "ref_embeddings_sha256": None, "fingerprint": kv._refs_fingerprint(refs), "count": count,
        "listed_count": count, "pixel_count": 0, "servable_refs": deepcopy(refs), "listing": deepcopy(listing),
        "files": [{"name": metadata.name, "role": "metadata", "sha256": _sha(metadata), "bytes": metadata.stat().st_size}]}
    record["pool_id"] = _digest(_pool_identity(record))
    return record


def seal_pool_audits(audits, cfg):
    for audit in audits:
        try:
            audit["public_reference_cache"] = describe_pool(audit, cfg)
        except Exception as error:
            audit["public_reference_cache"] = {"schema": SCHEMA, "status": "declined",
                "reason": str(error), "request": deepcopy(audit.get("public_cache_request"))}


def _write(path, value):
    path = _safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _payloads(pool, directory):
    _require(pool.get("schema") == SCHEMA and pool.get("status") == "complete", "pool is not complete")
    _require(_digest(_pool_identity(pool)) == pool.get("pool_id"), "reference pool receipt identity changed")
    result, names = [], set()
    provider = pool["request"]["source"]
    tag = hashlib.sha1(json.dumps(pool["metadata_signature"], sort_keys=True).encode()).hexdigest()[:10]
    metadata_names = ({f"mly_ref_meta_{tag}.json", "mly_ref_meta.json"} if provider == "mapillary" else
                      {"ref_meta.json"} if provider == "kartaview" else {"pnx_ref_meta.json"})
    allowed = {"metadata": metadata_names}
    if pool["payload_mode"] != "metadata_only":
        model = pool["backbone"]["name"]
        allowed.update(pixels={"ref_img_store.npz"}, descriptors={f"ref_emb_{model}_{pool['fingerprint'][:10]}.npz", f"ref_emb_{model}.npz"})
        if provider == "mapillary":
            allowed["image_store"] = {f"mly_ref_store_{tag}.json"}
    for item in pool["files"]:
        name = item["name"]
        _require(name not in names and _PAYLOAD.fullmatch(name) and Path(name).name == name
                 and not re.search(r"query|ocr|trajectory|prediction|ground_truth", name, re.I),
                 "non-public or duplicate payload file")
        _require(name in allowed.get(item["role"], set()), "payload role/name is not from the completed public request")
        names.add(name)
        path = _safe(Path(directory) / name, directory)
        _require(path.is_file() and path.stat().st_size == item["bytes"] and _sha(path) == item["sha256"],
                 "public cache payload changed or is incomplete")
        result.append((path, item))
    roles = {"metadata"} if pool["payload_mode"] == "metadata_only" else {"metadata", "pixels", "descriptors"}
    _require({item["role"] for _, item in result} >= roles,
             "reference family is missing a required payload")
    return result


def copy_pool(pool, source_dir, destination_dir):
    """Copy a complete public family atomically; existing paths are immutable."""
    source_dir, destination_dir = _safe(source_dir), _safe(destination_dir)
    payloads = _payloads(pool, source_dir)
    if source_dir == destination_dir:
        return {"status": "existing_identical", "reason": "already shared checked reference directory"}
    if destination_dir.exists():
        try:
            _payloads(pool, destination_dir)
            return {"status": "existing_identical"}
        except (OSError, ValueError):
            return {"status": "declined", "reason": "destination exists with conflicting payloads"}
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".public-reference-", dir=destination_dir.parent))
    try:
        for path, item in payloads:
            target = staging / item["name"]
            with path.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024**2)
            _require(_sha(target) == item["sha256"], "public payload copy hash mismatch")
        _payloads(pool, source_dir)  # reject concurrent source replacement, too
        _payloads(pool, staging)
        _require(not destination_dir.exists(), "destination appeared during reference copy")
        os.rename(staging, destination_dir)  # same filesystem, before child spawn
        return {"status": "copied"}
    finally:
        if staging.exists():
            _require(staging.resolve().is_relative_to(destination_dir.parent.resolve()), "staging escaped owned cache parent")
            # Only exact files made by this transaction; never recursive deletion.
            for path in staging.iterdir():
                _require(path.is_file() and path.parent == staging, "unexpected staging entry")
                path.unlink()
            staging.rmdir()


def prepare_reuse(source, target_cfg, *, execution_id, video_sha256, source_sha256, child_root):
    """Freeze all available completed baseline pools before expanded launch."""
    from types import SimpleNamespace
    child_root = _safe(child_root)
    receipt = {"schema": SCHEMA, "status": "unavailable", "execution_id": execution_id,
        "video_sha256": video_sha256, "source_sha256": deepcopy(source_sha256),
        "target_data_dir": str(_safe(target_cfg.data_dir)), "pools": [], "declined": [],
        "claim": "same snapshot only; historical reference encoding remains UNVERIFIED",
        "query_artifacts_copied": False, "before_child_launch": True}
    try:
        _require(isinstance(source, dict) and source.get("kind") == "baseline"
            and source.get("status") == "complete" and source.get("worker_returncode") == 0,
            "no completed baseline worker in this execution")
        _require(source.get("execution_id") == execution_id and source.get("video_sha256") == video_sha256
            and source.get("source_sha256") == source_sha256, "baseline belongs to another execution/video/source")
        directory = _safe(source["output_dir"], child_root)
        _require(directory == child_root / "baseline", "baseline output is not this execution's baseline child")
        frozen_path = _safe(source["frozen_artifact"]["path"], directory)
        _require(_sha(frozen_path) == source["frozen_artifact"]["sha256"], "baseline freeze changed")
        frozen = _read(frozen_path)
        _require(frozen.get("kind") == "baseline" and frozen.get("scoring_has_not_run") is True
            and frozen.get("video_sha256") == video_sha256 and frozen.get("source_sha256") == source_sha256,
            "baseline frozen input contract changed")
        request_path = directory / "request.json"
        _require(_sha(request_path) == source["request_sha256"] == frozen["artifacts"].get("request.json"),
                 "baseline launch request changed")
        request = _read(request_path)
        _require(request.get("kind") == "baseline" and request.get("video_sha256") == video_sha256
            and request.get("source_sha256") == source_sha256 and request.get("execution_id") == execution_id
            and _safe(request["parent_output"]) == child_root.parent.parent, "baseline launch identity differs")
        source_cfg = SimpleNamespace(data_dir=Path(request["config"]["data_dir"]),
            vpr_region_cache_dir=request["config"].get("vpr_region_cache_dir"))
        receipt["baseline"] = {"worker_returncode": 0, "frozen_artifact": deepcopy(source["frozen_artifact"]),
            "request_sha256": source["request_sha256"], "output_dir": str(directory)}
        seen = set()
        for name in sorted(frozen["artifacts"]):
            relative = Path(name.replace("\\", "/"))
            if len(relative.parts) != 2 or relative.parts[0] != "retrieval_audit" or not re.fullmatch(r"\d{4}\.json", relative.name):
                continue
            metadata = _safe(directory / relative, directory)
            _require(_sha(metadata) == frozen["artifacts"][name], "completed baseline audit metadata changed")
            audit = _read(metadata)
            pool = audit.get("public_reference_cache") or {}
            try:
                metadata_only = pool.get("payload_mode") == "metadata_only"
                _require(pool.get("status") == "complete" and (audit.get("status") == "complete" or metadata_only),
                         pool.get("reason", "pool has no complete reference snapshot"))
                _require(pool.get("ref_embeddings_sha256") == audit.get("ref_embeddings_sha256")
                    and pool.get("actual_model_state_sha256") == audit.get("actual_model_state_sha256")
                    and pool.get("backbone") == audit.get("backbone") and pool.get("preprocessing") == audit.get("preprocessing"),
                    "snapshot differs from completed baseline encoding audit")
                arrays = audit["array_artifact"]
                array_path = _safe(arrays["path"], directory)
                array_relative = array_path.relative_to(directory).as_posix()
                frozen_arrays = {Path(key.replace("\\", "/")).as_posix(): value for key, value in frozen["artifacts"].items()}
                _require(_sha(array_path) == arrays["sha256"] == frozen_arrays.get(array_relative),
                         "completed baseline reference-array artifact changed")
                if not metadata_only:
                    with np.load(array_path, allow_pickle=False) as saved:
                        _require(_array_sha(saved["ref_ids"]) == pool["ref_ids_sha256"]
                            and _array_sha(saved["ref_latlon"]) == pool["ref_latlon_sha256"],
                            "copied reference inventory differs from frozen baseline arrays")
                source_dir = _safe(pool["request"]["cache_dir"])
                _require(_location(source_cfg, source_dir) == pool["location"], "reference family root changed")
                if pool["pool_id"] in seen:
                    continue
                seen.add(pool["pool_id"])
                location = pool["location"]
                if location["scope"] == "local":
                    target_root = _safe(target_cfg.data_dir)
                else:
                    external = getattr(target_cfg, "vpr_region_cache_dir", None)
                    _require(external is not None, "target has no checked external reference root")
                    target_root = _safe(Path(external) / Path(target_cfg.data_dir).name)
                destination = _safe(target_root / location["relative"], target_root)
                operation = copy_pool(pool, source_dir, destination)
                record = {"pool": pool, "target_dir": str(destination), "operation": operation,
                    "source_audit": {"path": str(metadata), "sha256": _sha(metadata)}}
                if operation["status"] in {"copied", "existing_identical"}:
                    receipt["pools"].append(record)
                else:
                    receipt["declined"].append(record)
            except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
                receipt["declined"].append({"source_audit": str(metadata), "reason": str(error)})
        receipt["status"] = "complete"  # inventory complete, even when no pool qualifies
    except (OSError, ValueError, KeyError, TypeError) as error:
        receipt["reason"] = str(error)
    target = Path(target_cfg.output_dir) / "public_reference_reuse.json"
    _write(target, receipt)
    return {"path": str(target.resolve()), "sha256": _sha(target)}


def _without_directory(request):
    return {key: value for key, value in request.items() if key != "cache_dir"}


@contextmanager
def track_scope(cfg, kind, reuse_artifact=None, *, expected=None):
    """Worker-local capture/replay wrapper; restores the default APIs on exit."""
    from . import kartaview_vpr as kv
    original = kv.kartaview_vpr_track
    signature = inspect.signature(original)
    calls, reused = [], []
    if kind == "expanded" and reuse_artifact is not None:
        path = _safe(reuse_artifact["path"], cfg.output_dir)
        _require(_sha(path) == reuse_artifact["sha256"], "pre-launch public reuse receipt changed")
        receipt = _read(path)
        _require(receipt.get("schema") == SCHEMA and receipt.get("before_child_launch") is True
            and receipt.get("target_data_dir") == str(_safe(cfg.data_dir)), "public reuse receipt targets another child")
        _require(isinstance(expected, dict) and all(receipt.get(name) == expected.get(name)
            for name in ("execution_id", "video_sha256", "source_sha256")),
            "public reuse receipt belongs to another execution/video/source")
        reused = receipt.get("pools") or []

    @wraps(original)
    def tracked(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        audit = kwargs.get("audit")
        request, runtime = None, {"status": "not_requested"}
        old_fetch = None
        try:
            try:
                request = public_request(bound.arguments)
            except (OSError, ValueError, KeyError, TypeError) as error:
                runtime = {"status": "unsupported", "reason": str(error)}
                return original(*args, **kwargs)
            directory = _safe(request["cache_dir"])
            candidates = [record for record in reused if _safe(record["target_dir"]) == directory]
            exact = [record for record in candidates
                     if _without_directory(record["pool"]["request"]) == _without_directory(request)]
            if candidates:
                runtime = {"status": "miss", "requested_cache_dir": str(directory), "reason": "full request mismatch"}
                try:
                    _require(len(exact) == 1, "full actual request does not uniquely match the source snapshot")
                    record, pool = exact[0], exact[0]["pool"]
                    _payloads(pool, directory)
                    _require(_sha(Path(kv.__file__)) == pool["kv_source_sha256"], "reference preprocessing/source changed")
                    _require(library_versions(("numpy", "opencv-python", "torch", "torchvision")) == pool["runtime_versions"],
                             "reference numerical runtime changed")
                    # The assisted worker is explicitly CPU-only. Resolution is
                    # cached, so the ordinary call still constructs one model.
                    if pool["payload_mode"] != "metadata_only":
                        device = request["device"] or "cpu"
                        resolved = kv._resolve_backbone(request["model_name"], device)
                        _require(resolved == pool["backbone"]["name"] and kv.backbone_info() == pool["backbone"]
                            and kv._audit_model_state()["sha256"] == pool["actual_model_state_sha256"],
                            "active model differs from the baseline reference consumer")
                    refs = deepcopy(pool["servable_refs"])
                    _require(kv._refs_fingerprint(refs) == pool["fingerprint"] and len(refs) == pool["count"],
                             "servable public snapshot order changed")
                    old_fetch = kv._fetch_refs_for
                    def snapshot_fetch(source, center, radius_m, cache_dir, cap, token):
                        actual = {"source": source, "center": list(map(float, center)),
                                  "radius_m": float(radius_m), "cap": int(cap)}
                        _require(all(actual[key] == request[key] for key in actual)
                            and _safe(cache_dir) == directory, "nested reference request differs from the copied snapshot")
                        kv.LAST_FETCH_STATS = {**deepcopy(pool.get("listing") or {}), "cache_hit": True,
                                              "within_run_snapshot_reuse": pool["pool_id"]}
                        return deepcopy(refs)
                    kv._fetch_refs_for = snapshot_fetch
                    runtime = {"status": "metadata_hit" if pool["payload_mode"] == "metadata_only" else "hit", "pool_id": pool["pool_id"], "target_dir": str(directory),
                        "source_audit": record["source_audit"], "reference_encoding_identity": pool["reference_encoding_identity"],
                        "listed_count": pool["listed_count"], "same_snapshot_reference_count": pool["count"]}
                except (OSError, ValueError, KeyError, TypeError) as error:
                    runtime["reason"] = str(error)
                    # Do not delete/overwrite a transplant after publication.
                    # A different actual request uses a fresh nested cache.
                    fresh = directory / ("_unshared_" + _digest(_without_directory(request))[:16])
                    kwargs["cache_dir"] = str(fresh)
                    request["cache_dir"] = str(fresh)
                    runtime["effective_cache_dir"] = str(fresh)
            result = original(*args, **kwargs)
            if runtime["status"] == "hit" and isinstance(audit, dict):
                pool = exact[0]["pool"]
                try:
                    _require(audit.get("ref_embeddings_sha256") == pool["ref_embeddings_sha256"]
                        and _array_sha(audit["ref_ids"]) == pool["ref_ids_sha256"]
                        and _array_sha(np.asarray(audit["ref_latlon"], float)) == pool["ref_latlon_sha256"],
                        "receiver did not consume the exact copied reference tensor/order")
                    _payloads(pool, directory)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    runtime.update(status="failed", reason=str(error))
                    audit["status"] = "incomplete"
                    raise
            return result
        finally:
            if old_fetch is not None:
                kv._fetch_refs_for = old_fetch
            record = {"request": request, "reuse": runtime}
            calls.append(record)
            if isinstance(audit, dict):
                audit["public_cache_request"] = request
                audit["public_cache_reuse"] = runtime
    kv.kartaview_vpr_track = tracked
    try:
        yield calls
    finally:
        kv.kartaview_vpr_track = original
