"""Parent-owned catalog of persisted retrieval bytes after child closure.

No array values, RAM-only observations or child success are inferred here.
"""
from __future__ import annotations
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

PAIR = re.compile(r"^([0-9]{4})\.(json|npz)$")
FIVE = {"ref_ids", "ref_latlon", "raw_similarities", "query_indices", "query_times"}
MAX_FILE_BYTES = 64 * 1024**2
MAX_METADATA_BYTES = 2 * 1024**2
MAX_HASH_BYTES = 1024**3
MAX_PAIRS = 512
MAX_ENTRIES = 8192


class Unavailable(RuntimeError):
    """Optional persistence work must yield to mandatory Answer publication."""


class Limit:
    def __init__(self, budget=None):
        self.clock = budget.clock if budget is not None else time.monotonic
        self.maximum_s = min(5., budget.finalization_reserve_s / 4) if budget is not None else 5.
        self.publication_boundary = (budget.deadline - 3 * budget.finalization_reserve_s / 4
                                     if budget is not None else float("inf"))
        self.spent_s = 0.
        self.bytes = 0
        self.stop_at = None

    @contextmanager
    def phase(self):
        start = self.clock()
        self.stop_at = min(start + max(0., self.maximum_s - self.spent_s), self.publication_boundary)
        try:
            self.check()
            yield self.check
            self.check()
        finally:
            self.spent_s += max(0., self.clock() - start)
            self.stop_at = None

    def check(self, byte_count=0):
        self.bytes += byte_count
        if self.bytes > MAX_HASH_BYTES:
            raise Unavailable("optional terminal catalog byte budget exhausted")
        if self.stop_at is None or self.clock() >= self.stop_at:
            raise Unavailable("optional terminal catalog time budget exhausted")


def limit_for(parent):
    if not hasattr(parent, "assisted_catalog_limit"):
        parent.assisted_catalog_limit = Limit(getattr(parent, "runtime_budget", None))
    return parent.assisted_catalog_limit


def _check(byte_count=0):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path, check=_check):
    check()
    if Path(path).stat().st_size > MAX_FILE_BYTES:
        raise Unavailable("optional terminal catalog file exceeds byte bound")
    digest = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as stream:
        while block := stream.read(64 * 1024):
            total += len(block)
            check(len(block))
            if total > MAX_FILE_BYTES:
                raise Unavailable("growing catalog member exceeds byte bound")
            digest.update(block)
    check()
    return digest.hexdigest()


def safe(path, root):
    path, root = Path(path), Path(root).resolve()
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)):
        raise ValueError("retrieval catalog refuses aliases")
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("catalog member escaped child")
    return path


def read(path, check=_check, expected_hash=None):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate metadata key")
            result[key] = value
        return result
    check()
    bound = min(MAX_FILE_BYTES, MAX_METADATA_BYTES)
    if Path(path).stat().st_size > bound:
        raise Unavailable("metadata exceeds catalog bound")
    chunks = []
    total = 0
    with Path(path).open("rb") as stream:
        while block := stream.read(64 * 1024):
            total += len(block)
            check(len(block))
            if total > bound:
                raise Unavailable("growing metadata exceeds catalog bound")
            chunks.append(block)
    blob = b"".join(chunks)
    if expected_hash is not None and hashlib.sha256(blob).hexdigest() != expected_hash:
        raise ValueError("catalog metadata changed between hash and read")
    result = json.loads(blob, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite metadata")))
    check()
    return result


def file_record(path, root, check=_check):
    path = safe(path, root)
    return {"path": str(path), "sha256": sha(path, check), "bytes": path.stat().st_size}


def physical_files(root, check=_check):
    files, pending, count = [], [Path(root)], 0
    while pending:
        check()
        # Unlike rglob's directory-list materialization, scandir lets every
        # discovered entry yield to the deadline and traversal cap immediately.
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                check()
                count += 1
                if count > MAX_ENTRIES:
                    raise Unavailable("optional terminal catalog traversal bound exhausted")
                path = safe(entry.path, root)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False) and path.parent.name == "retrieval_audit" and PAIR.fullmatch(path.name):
                    files.append(path)
                    if len(files) > 2 * MAX_PAIRS:
                        raise Unavailable("optional terminal catalog file-count bound exhausted")
    return sorted(files)


def pair_record(root, stem, files, kind, execution, check=_check):
    row = {"scope": Path(stem).parent.as_posix(), "file_index": int(Path(stem).name),
           "metadata": None, "arrays": None, "logical_id": None, "status": "incomplete", "issues": []}
    for extension, role in (("json", "metadata"), ("npz", "arrays")):
        if extension in files:
            row[role] = file_record(files[extension], root, check)
    if row["metadata"] is None or row["arrays"] is None:
        row["status"] = "orphan"
        return row
    try:
        metadata = read(files["json"], check, row["metadata"]["sha256"])
        writer = metadata["persisted_writer"]
        index = writer["ordinal"]
        if (writer["schema"] != "retrieval-audit-writer-v1" or type(index) is not int or index != row["file_index"]
                or safe(writer["scope"], root) != safe(files["json"].parent, root)
                or safe(writer["array_path"], root) != safe(files["npz"], root)):
            raise ValueError("writer identity differs from persisted pair")
        acquisition = metadata.get("call_index")
        row["writer_ordinal"] = index
        row["acquisition_call_index"] = {"present": "call_index" in metadata, "value": acquisition}
        if acquisition is not None and (type(acquisition) is not int or acquisition != index):
            raise ValueError("acquisition call index contradicts writer ordinal")
        row["logical_id"] = f"{kind}:{execution}:{index:04d}"
        artifact = metadata["array_artifact"]
        if (safe(artifact["path"], root) != safe(files["npz"], root)
                or artifact["sha256"] != row["arrays"]["sha256"]):
            raise ValueError("metadata does not bind its persisted NPZ")
        headers = artifact["arrays"]
        if not isinstance(headers, dict):
            raise ValueError("missing typed array headers")
        for header in headers.values():
            if (not isinstance(header, dict) or set(header) != {"sha256", "shape", "dtype"}
                    or not re.fullmatch(r"[0-9a-f]{64}", str(header["sha256"]))
                    or not isinstance(header["shape"], list)
                    or any(type(n) is not int or n < 0 for n in header["shape"])
                    or not isinstance(header["dtype"], str) or len(header["dtype"]) > 64):
                raise ValueError("invalid declared typed array header")
        row["typed_array_headers"] = headers
        row["producer_status"] = metadata.get("status")
        terminal = (metadata.get("terminal_search") or {}).get("status")
        row["terminal_search_status"] = terminal
        measured = {key: metadata.get(key) for key in (
            "request", "reference_pool", "ref_embeddings_sha256", "query_mode", "query_pixels_sha256",
            "query_descriptors_sha256", "actual_model_state_sha256", "backbone", "preprocessing",
            "comparison_identity", "comparison_identity_sha256", "public_cache_request", "public_cache_reuse")}
        measured["typed_array_headers"] = headers
        row["consumed_identity_sha256"] = digest(measured)
        if metadata.get("status") == "complete" and FIVE <= set(headers):
            row["status"] = "complete_pair"
        elif terminal in {"complete_empty", "complete_sparse"}:
            row["status"] = "terminal_search_pair"
        else:
            row["issues"].append("producer did not declare complete observations")
    except (OSError, ValueError, KeyError, TypeError) as error:
        row["issues"].append(type(error).__name__ + ": " + str(error))
    return row


def collect(child_root, request, closure, check=_check):
    root = safe(child_root, child_root)
    request_path = safe(root / "request.json", root)
    request_ref = file_record(request_path, root, check)
    if digest(read(request_path, check, request_ref["sha256"])) != digest(request):
        raise ValueError("child request changed before catalog")
    kind, execution = request["kind"], request["execution_id"]
    if kind not in {"baseline", "expanded"} or not isinstance(execution, str) or not execution:
        raise ValueError("ordinary execution identity required")
    seal = {"status": "absent", "artifact": None, "request_bound": False}
    seal_path = root / "child_frozen.json"
    if seal_path.exists():
        seal["artifact"] = file_record(seal_path, root, check)
        try:
            frozen = read(seal_path, check, seal["artifact"]["sha256"])
            seal["request_bound"] = (frozen.get("kind") == kind and frozen.get("scoring_has_not_run") is True
                and frozen.get("video_sha256") == request["video_sha256"]
                and frozen.get("source_sha256") == request["source_sha256"]
                and frozen.get("artifacts", {}).get("request.json") == request_ref["sha256"])
            seal["status"] = "present_request_bound" if seal["request_bound"] else "present_unverified"
        except (OSError, ValueError, TypeError, KeyError) as error:
            seal.update(status="invalid", error=type(error).__name__)
    pairs = {}
    for path in physical_files(root, check):
        stem = path.relative_to(root).as_posix().rsplit(".", 1)[0]
        pairs.setdefault(stem, {})[path.suffix[1:]] = path
    if len(pairs) > MAX_PAIRS:
        raise Unavailable("optional terminal catalog pair bound exhausted")
    rows = [pair_record(root, stem, files, kind, execution, check) for stem, files in sorted(pairs.items())]
    groups = {}
    for index, row in enumerate(rows):
        if row["logical_id"] is not None:
            groups.setdefault(row["logical_id"], []).append(index)
    logical = []
    for identity, indices in sorted(groups.items()):
        signatures = {rows[i].get("consumed_identity_sha256") for i in indices}
        conflict = len(signatures) != 1 or None in signatures
        if conflict:
            for i in indices:
                rows[i]["status"] = "conflict"
        logical.append({"logical_id": identity, "physical_pair_indices": indices,
                        "scope_aliases": [rows[i]["scope"] for i in indices], "conflict": conflict})
    verified = closure.get("all_tracked_owned_births_gone") is True
    result = {"schema": "terminal-persisted-retrieval-catalog-v1", "kind": kind, "execution_id": execution,
        "status": "closure_verified" if verified else "closure_unverified", "terminal_authoritative": verified,
        "child_output": str(root), "video_sha256": request["video_sha256"],
        "window": [request["config"]["vo_start_sec"], request["config"]["vo_end_sec"]],
        "source_sha256": deepcopy(request["source_sha256"]), "request": request_ref, "child_seal": seal,
        "actor_closure": deepcopy(closure), "pairs": rows, "logical_calls": logical,
        "scope": "persisted ####.json/npz under this child retrieval_audit tree; observed actor births only",
        "RAM_only_or_full_acquisition_completeness_claim": False, "array_values_checked": False,
        "child_status_or_selection_promoted": False}
    check()
    return result


def verify(catalog, check=_check):
    """Recheck exact physical membership and bytes without array loading."""
    root = Path(catalog["child_output"])
    expected = {safe(ref["path"], root): ref["sha256"] for row in catalog["pairs"]
                for role in ("metadata", "arrays") if (ref := row[role]) is not None}
    actual = {safe(p, root) for p in physical_files(root, check)}
    if actual != set(expected):
        raise ValueError("persisted retrieval inventory changed")
    for ref in (catalog["request"], catalog["child_seal"]["artifact"]):
        if ref is not None:
            expected[safe(ref["path"], root)] = ref["sha256"]
    if catalog["child_seal"]["status"] == "absent" and (root / "child_frozen.json").exists():
        raise ValueError("child seal appeared during catalog")
    for path, expected_hash in expected.items():
        if sha(path, check) != expected_hash:
            raise ValueError("terminal retrieval evidence changed")
    check()
    return expected


def write(path, catalog, check=_check):
    path = Path(path)
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)):
        raise ValueError("catalog output alias")
    if path.exists() or path.resolve().is_relative_to(Path(catalog["child_output"]).resolve()):
        raise ValueError("catalog must be new and outside child")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for block in json.JSONEncoder(indent=2, allow_nan=False).iterencode(catalog):
                check()
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        verify(catalog, check)
        if path.exists():
            raise ValueError("terminal catalog appeared before publication")
        expected = sha(temporary, check)
        check()
        os.link(temporary, path)  # Atomic no-clobber publication on the same filesystem.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(path.resolve()), "sha256": expected}
