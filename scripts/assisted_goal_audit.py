"""Lazy audit of frozen assisted evidence; no models, VO or labels are run.

Legacy goal checks stay dependency-free. Assisted checks additionally bind
JSON/NPZ evidence and replay the fixed policy. Public reference/POI positions
are permitted; they are not query-video GPS or ground truth.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

RUNTIME_CONFIG_FIELDS = ("runtime_budget_s", "runtime_deadline_monotonic_s",
                         "runtime_finalization_reserve_s", "runtime_cleanup_reserve_s")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            value.update(block)
    return value.hexdigest()


def expected_cpu_threads(config):
    """The torch/BLAS threads a run's assisted children were configured with
    (runs before the option existed recorded none and ran with 4)."""
    value = (config or {}).get("assisted_cpu_threads", 4)
    require(type(value) is int and 1 <= value <= 64, "assisted CPU thread setting is invalid")
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_string(value):
    return isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value) is not None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def relative(path):
    path = Path(str(path).replace("\\", "/"))
    require(not path.is_absolute() and ".." not in path.parts, "escaping relative artifact path")
    return path


def load_json(path):
    require(Path(path).stat().st_size <= 64 * 1024**2, "JSON artifact exceeds the audit budget")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    """Check NPY dimensions before allocating any array; pickle is forbidden."""
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist()) <= 64, "NPZ contains too many arrays")
        require(len(archive.namelist()) == len(set(archive.namelist())), "NPZ contains duplicate arrays")
        require(sum(item.file_size for item in archive.infolist()) <= 64 * 1024**2,
                "NPZ exceeds its uncompressed audit budget")
        for item in archive.infolist():
            require(item.filename.endswith(".npy") and "/" not in item.filename, "unexpected NPZ member")
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version in {(2, 0), (3, 0)}:
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported NPY format")
                require(not dtype.hasobject and len(shape) <= 3 and all(x >= 0 for x in shape)
                        and math.prod(shape) * dtype.itemsize <= 64 * 1024**2, "unsafe NPY array")
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


class Files:
    def __init__(self, root, *, cwd=None):
        self.root = Path(root).resolve(strict=True)
        self.cwd = Path(cwd).resolve(strict=True) if cwd is not None else None

    def config_path(self, path, *, strict=True):
        """Resolve execution-CWD paths without consulting the verifier's CWD."""
        value = Path(path)
        if not value.is_absolute():
            require(self.cwd is not None, "relative configuration path has no proven execution working directory")
            require(not value.drive, "drive-relative configuration path is ambiguous")
            value = self.cwd / value
        return value.resolve(strict=strict)

    def path(self, path):
        value = Path(path)
        if not value.is_absolute():
            value = self.root / value
        value = value.resolve(strict=True)
        require(value.is_relative_to(self.root), "artifact escapes the parent output")
        return value

    def child(self, root):
        return Files(self.path(root), cwd=self.cwd)

    def record(self, record, ledger=None):
        require(isinstance(record, dict) and sha_string(record.get("sha256")), "artifact lacks a digest")
        path = self.path(record["path"])
        require(file_sha(path) == record["sha256"], "frozen artifact content changed")
        if ledger is not None:
            require(ledger.get(path) == record["sha256"], "artifact is absent from its child freeze")
        return path

    def freeze(self, path):
        path = self.path(path)
        value = load_json(path)
        require(value.get("scoring_has_not_run") is True, "artifact was not frozen before scoring")
        ledger = {}
        for name, expected in value["artifacts"].items():
            target = self.path(path.parent / relative(name))
            require(target.is_relative_to(path.parent) and sha_string(expected) and file_sha(target) == expected,
                    "child freeze contains an escaping or changed artifact")
            ledger[target] = expected
        return value, ledger


def parent_files(config, inputs, audit):
    """Prove the canonical output and original CWD from frozen absolute roles.

    Absolute configured outputs remain authoritative. A relative output may be
    anchored only by an identical normalized suffix of the common execution
    root; no search, basename fallback, or parent traversal is permitted.
    """
    execution = inputs["execution_id"]
    require(isinstance(execution, str) and re.fullmatch(r"[a-f0-9]{32}", execution) is not None,
            "invalid execution identity for output anchoring")
    roots = set()
    expected_names = {"candidate_artifact": "candidate.json", "result_artifact": "result.json",
                      "frozen_artifact": "child_frozen.json"}
    for record in audit["reached_stages"]:
        artifacts = record.get("artifacts") or []
        if not artifacts:
            continue
        require(len(artifacts) == 3 and {item["role"] for item in artifacts} == set(expected_names),
                "output anchor lacks all three frozen child roles")
        for item in artifacts:
            path = Path(item["path"])
            require(path.is_absolute() and ".." not in path.parts, "output anchor requires absolute frozen artifact paths")
            path = path.resolve(strict=True)
            require(path.name == expected_names[item["role"]] and path.parent.name == record["kind"]
                    and path.parent.parent.name == execution and path.parent.parent.parent.name == "children",
                    "frozen artifact has a different execution layout")
            require(sha_string(item.get("sha256")) and file_sha(path) == item["sha256"], "output anchor artifact changed")
            roots.add(path.parents[3])
    require(len(roots) == 1, "frozen child roles disagree about the canonical parent output")
    root = roots.pop()
    configured = Path(config["output_dir"])
    if configured.is_absolute():
        require(configured.resolve(strict=True) == root, "absolute configured output differs from its frozen execution")
        return Files(root)
    suffix = relative(configured)
    require(not suffix.drive, "drive-relative output is ambiguous")
    count = len(suffix.parts)
    require(count <= len(root.parts) and (not count or Path(*root.parts[-count:]) == suffix),
            "relative configured output is not a suffix of its frozen execution")
    cwd = root.parents[count - 1] if count else root
    files = Files(root, cwd=cwd)
    require(files.config_path(configured) == root, "relative output does not resolve under its proven execution CWD")
    return files


def sources(files, root, declared):
    require(isinstance(declared, dict) and bool(declared), "source identity is missing")
    for name, expected in declared.items():
        path = files.path(Path(root) / "source_snapshot" / relative(name))
        require(sha_string(expected) and file_sha(path) == expected, "source snapshot hash differs")


def comparison(files, value, ledger, window, *, require_all=False):
    """Bind decoded statistics to exact raw matrices and actual audit rows."""
    import numpy as np
    from monocular_osm.assisted_worker import _array_sha, _valid_audit
    artifacts = value.get("audit_artifacts") or []
    if not artifacts and value.get("unavailable_reason"):
        require(not value.get("decoded_reference_cosines"), "unavailable comparison carries promotable scores")
        require(not require_all, "complete visual candidate lacks actual reference audits")
        return []
    require(artifacts, "retrieval comparison lacks frozen full-matrix audits")
    records = {files.record(record, ledger): record for record in artifacts}
    require(len(records) == len(artifacts), "retrieval audit repeats an artifact instead of an outcome")
    metas = sorted((p for p in records if p.suffix == ".json"), key=lambda p: int(p.stem))
    require(len(metas) * 2 == len(records), "retrieval metadata/matrix pairs are incomplete")
    audits = []
    for index, path in enumerate(metas):
        require(int(path.stem) == index, "retrieval audit ordinals are incomplete")
        meta = load_json(path)
        matrix_record = meta.pop("array_artifact")
        matrix_path = files.record(matrix_record, ledger)
        require(matrix_path in records and matrix_path.stem == path.stem, "audit matrix pair is stale")
        arrays = load_npz(matrix_path)
        declared = matrix_record["arrays"]
        require(set(declared) == set(arrays), "audit array inventory is incomplete")
        for name, array in arrays.items():
            require(declared[name] == {"sha256": _array_sha(array), "shape": list(array.shape), "dtype": array.dtype.str},
                    "audit array identity is stale")
        returned = {name.removeprefix("returned_"): array for name, array in arrays.items()
                    if name.startswith("returned_")}
        meta.update({name: array for name, array in arrays.items() if not name.startswith("returned_")})
        if returned:
            meta["returned_track"] = returned
        if _valid_audit(meta):
            require(len(meta["query_indices"]) <= 80 and meta["raw_similarities"].size <= 1_600_000,
                    "comparison exceeds its query/matrix budget")
            refs = meta["ref_latlon"]
            require(np.all(np.abs(refs[:, 0]) <= 90) and np.all(np.abs(refs[:, 1]) <= 180),
                    "reference positions are outside WGS84")
            require(meta["query_times"][0] >= window[0] - 1
                    and (window[1] is None or meta["query_times"][-1] <= window[1] + 1),
                    "retrieval queries are outside the registered video window")
        if require_all and not _valid_audit(meta):
            request = meta.get("request") or {}
            terminal_search({"status": "unavailable", "center": request.get("center"),
                "radius_m": request.get("radius_m"), "ref_cap": request.get("cap"),
                "track_decoding": {"terminal_search": meta.get("terminal_search")}})
        audits.append(meta)
    if require_all:
        successful = [a for a in audits if _valid_audit(a)]
        identities = {digest({key: a[key] for key in ("query_pixels_sha256", "query_descriptors_sha256",
            "actual_model_state_sha256", "backbone", "preprocessing", "query_mode")}
            | {"indices": _array_sha(a["query_indices"]), "times": _array_sha(a["query_times"])}) for a in successful}
        require(len(identities) <= 1, "planned pools used different actual query/model/preprocessing identities")
    if value.get("unavailable_reason"):
        require(not value.get("decoded_reference_cosines"), "unavailable comparison carries promotable scores")
        return audits
    selected = value.get("selected_audit_rows") or []
    require(selected, "comparison has no decoded-reference row selection")
    used, cosines, coords, ids = [], [], [], []
    for position, pair in enumerate(selected):
        require(isinstance(pair, list) and len(pair) == 2 and all(type(i) is int and i >= 0 for i in pair),
                "invalid selected audit row")
        number, row = pair
        require(number < len(audits), "selected audit is missing")
        audit = audits[number]
        require(_valid_audit(audit) and row < len(audit["query_indices"]), "selected audit is inconsistent")
        require(int(audit["query_indices"][row]) == value["query"]["indices"][position],
                "selected row belongs to a different query grid")
        used.append(audit)
        chosen = int(audit["decoded_ref_indices"][row])
        cosines.append(float(audit["raw_similarities"][row, chosen]))
        coords.append(audit["ref_latlon"][chosen].tolist())
        ids.append(str(audit["ref_ids"][chosen]))
    require(len({a["comparison_identity_sha256"] for a in used}) == 1,
            "comparison combines different query/model/decoder identities")
    first = used[0]
    decoder = first["decoder"]
    expected = {"statistic": first["score_kind"],
        "model": {**first["backbone"], "weights_sha256": first["actual_model_state_sha256"]},
        "query": {"indices": first["query_indices"].tolist(), "times": first["query_times"].tolist(),
                  "mode": first["query_mode"], "preprocessing": first["preprocessing"],
                  "pixels_sha256": first["query_pixels_sha256"], "descriptors_sha256": first["query_descriptors_sha256"]},
        "inner_decoder": {"actual": decoder["actual"], "fallback": bool(decoder.get("fallback_reason")),
                          "settings": {k: v for k, v in decoder.items() if k not in {"actual", "fallback_reason"}}},
        "audited_identity": first["comparison_identity"], "audited_identity_sha256": first["comparison_identity_sha256"],
        "decoded_reference_cosines": cosines, "decoded_reference_latlon": coords, "decoded_reference_ids": ids}
    require(len(selected) == len(expected["query"]["indices"])
            and all(value.get(key) == expected_value for key, expected_value in expected.items()),
            "comparison differs from its exact frozen query/matrix evidence")
    sampling = value.get("source_sampling") or {}
    count = sampling.get("dense_count")
    require(sampling.get("mode") == "linspace" and type(count) is int and count >= len(selected)
            and sampling.get("requested_sample_count") == 80 and sampling.get("sample_count") == len(selected)
            and len(selected) == min(80, count)
            and sampling.get("dense_indices") == np.linspace(0, count - 1, min(80, count)).astype(int).tolist()
            and expected["query"]["indices"] == list(range(len(selected))), "comparison omitted part of the declared input grid")
    return audits


def verify_child(files, record, inputs, before):
    """Bind transport, nested files, source/video/window and selected route."""
    from monocular_osm.assisted_worker import _from_snapshot, child_config
    from monocular_osm.provenance import config_snapshot
    from monocular_osm.stages.assisted import child_record
    kind = record["kind"]
    profile = inputs["config"].get("assisted_profile", "v1")
    allowed = {"baseline", "cues", "poi_joint", "metric_pf", "expanded"}
    allowed.add("poi_image_union" if profile == "v2" else "poi_image_track")
    require(kind in allowed, "unknown or incompatible reached stage for assisted profile")
    roles = {item["role"]: item for item in record.get("artifacts") or []}
    if not roles:
        require(record.get("status") == "failed" and bool(record.get("error")), "unfrozen child lacks a terminal failure")
        child = {"status": "failed", "error": record["error"]}
        rebuilt = (child_record(kind, child, files.root, expected=inputs, profile="v3") if profile == "v3"
                   else child_record(kind, child, files.root, expected=inputs))
        require(rebuilt == record, "failed child summary is stale")
        return child, {}
    require(set(roles) == {"result_artifact", "candidate_artifact", "frozen_artifact"}, "incomplete child artifact roles")
    base = files.root / "children" / inputs["execution_id"] / kind
    for role, name in (("candidate_artifact", "candidate.json"), ("result_artifact", "result.json"),
                       ("frozen_artifact", "child_frozen.json")):
        require(files.record(roles[role]) == base / name, "child belongs to a different execution")
    candidate = load_json(base / "candidate.json")
    child = {**candidate, **roles, "worker_returncode": record.get("worker_returncode"), "status": record.get("status")}
    if record.get("reported_status") is not None:
        child.update(reported_status=record["reported_status"], transport_error=record.get("error"))
    rebuilt = (child_record(kind, child, files.root, expected=inputs, profile="v3") if profile == "v3"
               else child_record(kind, child, files.root, expected=inputs))
    require(rebuilt == record, "child record differs from frozen evidence")
    _, ledger = files.child(base).freeze(base / "child_frozen.json")
    request_path = base / "request.json"
    require(request_path in ledger, "child freeze omits its requested inputs")
    request = load_json(request_path)
    require(request["kind"] == kind and request["video_sha256"] == inputs["video"]["sha256"]
            and request["source_sha256"] == inputs["source_sha256"]
            and request["policy_evidence"] == before and files.config_path(request["parent_output"]) == files.root,
            "child request differs from its parent video/source/decision evidence")
    require(candidate.get("kind") == kind, "candidate identifies another stage")
    sources(files, base, candidate["source_sha256"])
    cfg, parent_cfg = candidate["config"], inputs["config"]
    requested = deepcopy(request["config"])
    if requested.get("frame_stride") is None:
        requested["frame_stride"] = cfg.get("frame_stride")
    require(requested == cfg, "child execution changed its requested configuration")
    expected = child_config(_from_snapshot(parent_cfg), kind, parent_cfg.get("video_path") or inputs["video"]["path"])
    expected.output_dir = Path(cfg["output_dir"])
    require(files.config_path(expected.output_dir) == base, "child effective output identifies a different execution")
    if parent_cfg.get("frame_stride") is None or kind == "metric_pf":
        expected.frame_stride = cfg.get("frame_stride")
    expected_snapshot = config_snapshot(expected)
    # Historical v1 freezes precede explicit profile selection. Its absent
    # field means v1 only; an explicitly declared parent must retain the field.
    if "assisted_profile" not in parent_cfg and "assisted_profile" not in cfg:
        expected_snapshot.pop("assisted_profile", None)
    # Historical snapshots predate these operational fields. Explicit fields
    # still participate in the exact parent/child configuration comparison.
    for field in RUNTIME_CONFIG_FIELDS:
        if field not in parent_cfg and field not in cfg:
            expected_snapshot.pop(field, None)
    require(expected_snapshot == cfg, "child changed its component/baseline configuration")
    result = load_json(base / "result.json")
    new_union = is_metadata_union(child.get("comparison"))
    if "candidate" in result:
        require(result["candidate"] == candidate, "component result differs from frozen candidate")
    elif candidate.get("route_latlon") is not None:
        require((result.get("answer") or {}).get("route_latlon") == candidate["route_latlon"], "child selected route is stale")
    if new_union and child.get("status") == "complete":
        answer = result.get("answer") or {}
        require(answer.get("route_latlon") == child["route_latlon"]
                and answer.get("method") == "metadata_union_reference_mean"
                and answer.get("source") == "assisted_v3.full_sequence.reference_mean"
                and answer.get("route_t_sec") == child["route_timestamps"]
                and (result.get("assisted_metadata_union") or {}).get("artifacts") == child["comparison"]["metadata_union_artifacts"],
                "new full child answer has ordinary smoothing or stale route/decoder provenance")
    require(not new_union or profile == "v3" and kind == "expanded", "full metadata comparison outside its explicit v3 expanded stage")
    if kind in {"baseline", "expanded"} and child.get("status") == "complete" and not (child.get("comparison") or {}).get("unavailable_reason") and not new_union:
        retrieval_route(candidate, result)
    return child, ledger


def runtime_evidence(files, cfg, inputs, audit, ledger):
    """Bind an operational deadline receipt without changing policy evidence."""
    path = files.root / "assisted_runtime.json"
    record = audit.get("runtime_budget")
    budgeted = (cfg.get("runtime_budget_s") is not None or cfg.get("runtime_deadline_monotonic_s") is not None
                or record is not None or path in ledger)
    if not budgeted:
        return
    declared = {str(k).replace("\\", "/"): v for k, v in inputs["source_sha256"].items()}
    require("monocular_osm/runtime_budget.py" in declared and sha_string(declared["monocular_osm/runtime_budget.py"]),
            "budgeted parent omits its runtime implementation source")
    require(file_sha(files.root / "source_snapshot/monocular_osm/runtime_budget.py") == declared["monocular_osm/runtime_budget.py"],
            "budgeted runtime implementation snapshot changed")
    require(all(field in cfg for field in RUNTIME_CONFIG_FIELDS), "budgeted parent omits an explicit runtime configuration field")
    require(path in ledger and file_sha(path) == ledger[path] and isinstance(record, dict) and digest(load_json(path)) == digest(record),
            "budgeted parent lacks its frozen/equal runtime receipt")
    fields = {"schema", "origin", "deadline_monotonic_s", "started_monotonic_s", "work_deadline_monotonic_s",
              "finalization_reserve_s", "cleanup_reserve_s", "events"}
    require(set(record) == fields and record["schema"] == "assisted-runtime-budget-v1"
            and record["origin"] in {"benchmark", "config"}, "runtime receipt schema/origin differs")
    for name in fields - {"schema", "origin", "events"}:
        require(finite(record[name]) and record[name] >= 0, "runtime receipt has a nonfinite/negative clock")
    require(record["deadline_monotonic_s"] > 0 and record["finalization_reserve_s"] > 0 and record["cleanup_reserve_s"] > 0
            and record["work_deadline_monotonic_s"] == record["deadline_monotonic_s"] - record["finalization_reserve_s"] - record["cleanup_reserve_s"],
            "runtime receipt changed its reserved work deadline")
    for field, name in (("runtime_finalization_reserve_s", "finalization_reserve_s"), ("runtime_cleanup_reserve_s", "cleanup_reserve_s")):
        require(finite(cfg[field]) and cfg[field] > 0 and cfg[field] == record[name], "runtime receipt/config reserve differs")
    if cfg["runtime_deadline_monotonic_s"] is not None:
        require(finite(cfg["runtime_deadline_monotonic_s"]) and cfg["runtime_deadline_monotonic_s"] > 0
                and record["deadline_monotonic_s"] <= cfg["runtime_deadline_monotonic_s"], "runtime receipt extends its configured absolute deadline")
    if cfg["runtime_budget_s"] is not None:
        require(finite(cfg["runtime_budget_s"]) and cfg["runtime_budget_s"] > 0
                and record["deadline_monotonic_s"] <= record["started_monotonic_s"] + cfg["runtime_budget_s"],
                "runtime receipt extends its configured relative budget")
    records = {r["kind"]: r for r in audit["reached_stages"]}
    require(isinstance(record["events"], list) and len(record["events"]) <= 16, "runtime event inventory exceeds fixed stage bound")
    previous = record["started_monotonic_s"]; seen = set(); completed = set()
    for event in record["events"]:
        kind, status = event["kind"], event["status"]
        require(kind in records and kind not in seen and status in {"not_started", "process_complete", "timed_out"}
                and finite(event["monotonic_s"]) and previous <= event["monotonic_s"] <= record["deadline_monotonic_s"],
                "runtime event kind/clock differs")
        previous = event["monotonic_s"]; seen.add(kind)
        if status in {"not_started", "timed_out"}:
            require(records[kind]["status"] == "failed" and not records[kind].get("artifacts"),
                    "deadline partial/not-started child was promoted")
        if status == "timed_out":
            require(event.get("partial_output_is_selected_or_complete") is False and sha_string(event.get("request_sha256"))
                    and files.config_path(event["partial_output"]) == files.root / "children" / inputs["execution_id"] / kind,
                    "timed-out child identifies another execution or a selected partial")
        if status in {"process_complete", "timed_out"}:
            require(type(event.get("worker_returncode")) is int and event.get("all_tracked_owned_births_gone") is True
                    and event.get("ownership_scope") == "observed_pid_births_only; unobserved_orphans_not_certified",
                    "runtime child lacks its observed-birth cleanup receipt")
            births = event.get("owned_births")
            require(isinstance(births, list) and len(births) <= 4096, "runtime ownership inventory is missing or unbounded")
            pids = set()
            for birth in births:
                require(type(birth.get("pid")) is int and birth["pid"] > 0 and birth["pid"] not in pids
                        and finite(birth.get("create_time")) and birth["create_time"] > 0, "runtime ownership birth is invalid or duplicated")
                pids.add(birth["pid"])
            if status == "process_complete" and records[kind]["status"] == "complete":
                require(event["worker_returncode"] == records[kind].get("worker_returncode") == 0, "runtime completed child exit differs")
                completed.add(kind)
    require(all(r["status"] != "complete" or r["kind"] in completed for r in audit["reached_stages"]),
            "budgeted complete child omits its process completion receipt")


def is_metadata_union(value):
    if not isinstance(value, dict) or "schema" not in value:
        return False
    from monocular_osm.assisted_comparison import SCHEMA
    return isinstance(value, dict) and value.get("schema") == SCHEMA


def metadata_union_math(arrays, output, variant):
    """Independently verify a full path/objective/median5; no optimality claim."""
    import numpy as np
    import pyproj
    from collections import Counter
    from monocular_osm.full_sequence import PROFILE, PROFILE_SHA256
    from monocular_osm.assisted_worker import _array_sha
    require(variant in {"raw", "reference_mean"}, "unsupported full metadata variant")
    ids, refs, score, times = (np.asarray(arrays[k]) for k in ("ref_ids", "ref_latlon", "raw_similarities", "query_times"))
    indices = np.asarray(arrays["query_indices"])
    n = len(ids)
    require(1 <= n <= 10000 and ids.shape == (n,) and ids.dtype.kind in "US" and len(set(ids.tolist())) == n
            and refs.shape == (n, 2) and refs.dtype == np.float64 and np.isfinite(refs).all()
            and np.all(abs(refs[:, 0]) <= 90) and np.all(abs(refs[:, 1]) <= 180)
            and score.shape == (80, n) and score.dtype == np.float32 and np.isfinite(score).all()
            and np.max(abs(score)) <= 1.00001 and indices.dtype.kind in "iu"
            and np.array_equal(indices, np.arange(80)) and times.shape == (80,) and times.dtype == np.float64
            and np.isfinite(times).all() and times[0] >= 0 and np.all(np.diff(times) > 0), "bounded complete full metadata evidence required")
    path = np.asarray(output["chosen_ref_indices"])
    require(path.shape == (80,) and path.dtype.kind in "iu" and path.min() >= 0 and path.max() < n,
            "full metadata path index is outside the complete union")
    order = np.argsort(ids.astype(str), kind="stable")
    lat0 = float(np.mean(refs[order, 0])); angles = np.deg2rad(refs[order, 1])
    sine, cosine = float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles)))
    require(np.hypot(sine, cosine) >= 1e-8, "unstable all-reference metric longitude")
    lon0 = float(np.rad2deg(np.arctan2(sine, cosine)))
    require(not pyproj.network.is_network_enabled(), "audit PROJ networking must be disabled")
    metric = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +ellps=WGS84 +units=m +no_defs")
    forward = pyproj.Transformer.from_crs("EPSG:4326", metric, always_xy=True)
    inverse = pyproj.Transformer.from_crs(metric, "EPSG:4326", always_xy=True)
    x, y = forward.transform(refs[:, 1], refs[:, 0]); xy = np.column_stack([x, y])
    require(np.isfinite(xy).all(), "full metadata metric coordinates are nonfinite")
    raw = score[np.arange(80), path].astype(np.float64)
    means = np.ascontiguousarray(score).mean(axis=0, dtype=np.float64)
    emissions = raw - (means[path] if variant == "reference_mean" else 0.)
    objective = float(emissions[0]); distances = []
    for q in range(1, 80):
        distance = float(np.hypot(*(xy[path[q]] - xy[path[q - 1]])))
        free = PROFILE["transition_base_free_m"] + PROFILE["transition_free_speed_mps"] * max(float(times[q] - times[q - 1]), PROFILE["transition_dt_floor_s"])
        objective = objective - PROFILE["transition_excess_penalty"] * max(distance - free, 0.) + float(emissions[q])
        distances.append(distance)
    chosen = refs[path]; selected_xy = xy[path]
    smooth = np.asarray([np.median(selected_xy[max(0, q - 2):min(80, q + 3)], axis=0) for q in range(80)])
    lon, lat = inverse.transform(smooth[:, 0], smooth[:, 1]); route = np.column_stack([lat, lon])
    dominance = max(Counter(map(tuple, chosen)).values()) / 80
    require(output.get("status") == "complete" and output.get("variant") == variant
            and output.get("candidate_pruning") is False and output.get("full_reference_count") == n
            and output.get("all_states_per_query") == n and output["chosen_ref_ids"] == ids[path].tolist()
            and output["raw_route_latlon"] == chosen.tolist() and output["selected_raw_cosines"] == raw.tolist()
            and output["selected_emissions"] == emissions.tolist() and output["route_timestamps"] == times.tolist()
            and output["dominant_coordinate_fraction"] == dominance
            and output["unique_coordinate_count"] == len(set(map(tuple, chosen)))
            and finite(output["objective"]) and math.isclose(output["objective"], objective, rel_tol=1e-12, abs_tol=1e-8)
            and np.asarray(output["route_latlon"]).shape == (80, 2)
            and np.allclose(output["route_latlon"], route, rtol=0, atol=1e-10),
            "full metadata path/raw scores/objective/actual median-five route differ")
    return {"variant": variant, "raw_cosines": raw.tolist(), "unsmoothed_reference_latlon": chosen.tolist(),
            "similarity_median": float(np.median(raw)), "dominant_coordinate_fraction": dominance,
            "metric_origin_latlon": [lat0, lon0], "objective": objective, "path": path,
            "profile_sha256": PROFILE_SHA256, "transition_distances_m": distances,
            "input_hashes": {"raw_cosines_sha256": _array_sha(score), "ref_ids_sha256": digest(ids.tolist()),
                "ref_latlon_sha256": _array_sha(refs), "query_times_sha256": _array_sha(times),
                "metric_origin_latlon": [lat0, lon0]}}


def metadata_union_rebuild(regions, descriptor_rows=None):
    """Canonical union from every frozen pool; never call the producer builder."""
    import numpy as np
    from monocular_osm.assisted_worker import _array_sha
    descriptor_rows = descriptor_rows or {}
    require(1 <= len(regions) <= 6 and len({r["key"] for r in regions}) == len(regions), "all1..6 metadata pools required")
    entries = {}; occurrences = 0; inputs = {}; grid = None
    for region in sorted(regions, key=lambda r: r["key"]):
        z = region["arrays"]; ids, ll, score = (z[k] for k in ("ref_ids", "ref_latlon", "raw_similarities"))
        n = len(ids)
        require(1 <= n <= 1500 and len(set(ids.tolist())) == n and ids.dtype.kind in "US"
                and ll.shape == (n, 2) and np.isfinite(ll).all() and score.shape == (80, n)
                and score.dtype == np.float32 and np.isfinite(score).all(), "bounded whole metadata pool required")
        current = (_array_sha(z["query_indices"]), _array_sha(z["query_times"]))
        require(grid is None or grid == current, "metadata pools changed actual query grids"); grid = current
        inputs[region["key"]] = {k: _array_sha(v) for k, v in z.items()}; occurrences += n
        for col, identifier in enumerate(ids.tolist()):
            require(isinstance(identifier, str) and identifier.startswith("mapillary:"), "public Mapillary provider IDs required")
            entries.setdefault(identifier, []).append((region["key"], col, ll[col], score[:, col]))
    require(1 <= len(entries) <= 10000, "full metadata union reference bound")
    ids = np.asarray(sorted(entries)); refs = []; columns = []; mapping = []
    for identifier in ids.tolist():
        rows = entries[identifier]; key, col, ref, scores = rows[0]
        require(all(np.array_equal(row[2], ref) for row in rows), "overlap reference coordinates disagree")
        hashes = []
        for name, index, _, _ in rows:
            data = descriptor_rows.get(name)
            hashes.append(_array_sha(data[index:index + 1]) if data is not None else None)
        if len(rows) > 1:
            require(all(v is not None and v == hashes[0] for v in hashes), "overlap lacks identical actual descriptor rows")
            stack = np.stack([row[3].astype(float) for row in rows])
            require(np.max(stack.max(axis=0) - stack.min(axis=0)) <= 1e-6, "overlap raw cosine envelope differs")
        refs.append(ref); columns.append(scores)
        mapping.append({"ref_id": identifier, "canonical_region": key, "canonical_column": col,
            "occurrences": [{"region": row[0], "column": row[1], "descriptor_sha256": h} for row, h in zip(rows, hashes)]})
    first = regions[0]["arrays"]
    return {"ref_ids": ids, "ref_latlon": np.asarray(refs, np.float64),
            "raw_similarities": np.ascontiguousarray(np.column_stack(columns), dtype=np.float32),
            "query_indices": first["query_indices"], "query_times": first["query_times"],
            "mapping": mapping, "reference_occurrences": occurrences, "input_sha256": inputs}


def metadata_descriptor_rows(files, records, regions, ledger):
    """Bind saved overlap rows to the exact consumed tensor and public payload."""
    import numpy as np
    from collections import Counter
    from monocular_osm.assisted_worker import _array_sha
    from monocular_osm.public_reference_cache import _PAYLOAD, _pool_identity
    from monocular_osm.kartaview_vpr import _refs_fingerprint, _sig_tag
    counts = Counter(str(v) for r in regions for v in r["arrays"]["ref_ids"])
    affected = {r["key"] for r in regions if any(counts[str(v)] > 1 for v in r["arrays"]["ref_ids"])}
    require(isinstance(records, list) and len(records) == len(affected)
            and {r["region"] for r in records} == affected, "all and only affected metadata descriptor pools required")
    by_name = {r["key"]: r for r in regions}; result = {}
    for record in records:
        region = by_name[record["region"]]; audit = region["audit"]; z = region["arrays"]
        path = files.record(record["artifact"], ledger); arrays = load_npz(path)
        require(set(arrays) == {"rows"}, "overlap descriptor artifact has unexpected roles")
        rows = arrays["rows"]; n = len(z["ref_ids"])
        require(rows.shape == (n, 8448) and rows.dtype == np.float32 and np.isfinite(rows).all()
                and _array_sha(rows) == record["rows_sha256"], "actual overlap descriptor shape/hash differs")
        tensor = hashlib.sha256(json.dumps([list(rows.shape), "torch.float32"], separators=(",", ":")).encode())
        tensor.update(memoryview(np.ascontiguousarray(rows)).cast("B"))
        receipt = record["receipt"]
        require(receipt.get("status") == "complete" and receipt.get("payload_mode") == "descriptors_and_pixels"
                and digest(_pool_identity(receipt)) == receipt.get("pool_id") and receipt.get("count") == n
                and receipt.get("ref_ids_sha256") == _array_sha(z["ref_ids"])
                and receipt.get("ref_latlon_sha256") == _array_sha(np.asarray(z["ref_latlon"], float))
                and receipt.get("ref_embeddings_sha256") == tensor.hexdigest() == audit["ref_embeddings_sha256"]
                and receipt.get("actual_model_state_sha256") == audit["actual_model_state_sha256"]
                and receipt.get("backbone") == audit["backbone"] and receipt.get("preprocessing") == audit["preprocessing"]
                and receipt.get("reference_encoding_identity") == audit.get("reference_encoding_identity", "UNVERIFIED"),
                "actual overlap rows/public receipt/model identity differs or encoding claim was upgraded")
        request = audit.get("public_cache_request")
        require(isinstance(request, dict) and receipt["request"] == request,
                "descriptor receipt has no exact actual public request")
        refs = receipt["servable_refs"]
        require(["mapillary:" + str(r["id"]) for r in refs] == z["ref_ids"].tolist()
                and np.array_equal(np.asarray([[r["lat"], r["lon"]] for r in refs], float), z["ref_latlon"])
                and _refs_fingerprint(refs) == receipt["fingerprint"], "cached descriptor provider-ID/coordinate order differs")
        tag = _sig_tag(receipt["metadata_signature"]); model = receipt["backbone"]["name"]
        allowed = {"metadata": {f"mly_ref_meta_{tag}.json", "mly_ref_meta.json"}, "pixels": {"ref_img_store.npz"},
            "descriptors": {f"ref_emb_{model}_{receipt['fingerprint'][:10]}.npz", f"ref_emb_{model}.npz"},
            "image_store": {f"mly_ref_store_{tag}.json"}}
        directory = Path(receipt["request"]["cache_dir"]).resolve(strict=True); hashes = {}; descriptor_path = None
        for item in receipt["files"]:
            require(isinstance(item["name"], str) and Path(item["name"]).name == item["name"]
                    and _PAYLOAD.fullmatch(item["name"]) and item["role"] in {"metadata", "pixels", "descriptors", "image_store"},
                    "non-public descriptor receipt payload")
            require(item["name"] in allowed[item["role"]], "descriptor source filename identifies another request/fingerprint")
            target = directory / item["name"]
            for candidate in (target, target.resolve()):
                require(not re.search(r"(?:^|[/_.-])(?:gt|ground[-_]?truth|oxts|labels?)(?=$|[/_.-])",
                    str(candidate).replace("\\", "/").casefold()), "label path in public reference receipt")
            target = target.resolve(strict=True)
            require(target.is_relative_to(directory) and str(target) not in hashes and sha_string(item["sha256"])
                    and target.stat().st_size == item["bytes"] and file_sha(target) == item["sha256"], "protected reference payload changed")
            hashes[str(target)] = item["sha256"]
            if item["role"] == "descriptors":
                require(descriptor_path is None, "multiple actual descriptor payloads"); descriptor_path = target
        require({r["role"] for r in receipt["files"]} >= {"metadata", "pixels", "descriptors"}
                and hashes == record["protected_hashes"] and descriptor_path is not None, "public payload/source inventory differs")
        source_rows = load_npz(descriptor_path)
        require(set(source_rows) == {"emb", "fingerprint"} and np.array_equal(source_rows["emb"], rows)
                and str(source_rows["fingerprint"]) == receipt["fingerprint"], "saved descriptor rows differ from their actual cache source")
        result[record["region"]] = rows
    return result


def metadata_regions(files, value, ledger, window):
    """Load complete frozen regional controls; each remains its own Viterbi proof."""
    import numpy as np
    audits = comparison(files, value, ledger, window, require_all=True)
    require(1 <= len(audits) <= 6, "full metadata candidate requires1..6 complete pools")
    paths = sorted((files.record(r, ledger) for r in value["audit_artifacts"] if Path(r["path"]).suffix == ".npz"), key=lambda p: int(p.stem))
    regions = []
    for i, (audit, path) in enumerate(zip(audits, paths)):
        z = load_npz(path); n = len(z["ref_ids"]); pool = audit.get("reference_pool") or {}; fetch = pool.get("fetch") or {}
        require(audit.get("status") == "complete" and not audit.get("issues") and 1 <= n <= 1500
                and np.array_equal(z["query_indices"], np.arange(80)) and pool.get("source") == "mapillary"
                and pool.get("count") == n and fetch.get("listing") == "tiles" and fetch.get("listing_stats_known") is True
                and not fetch.get("truncated") and type(fetch.get("lost")) is int and fetch["lost"] == 0
                and fetch.get("snapshot", {}).get("n_refs") == n, "full metadata pool has incomplete capped served imagery")
        regions.append({"key": f"audit_{i:04d}", "audit": audit, "arrays": z})
    return regions


def metadata_baseline_binding(files, child, ledger, inputs, baseline_child):
    """Prove the original baseline transport, never an appearance-only substitute."""
    from monocular_osm.stages.assisted import summary
    base = Path(child["candidate_artifact"]["path"]).parent; request = load_json(base / "request.json")
    binding = request.get("metadata_union_baseline") or {}
    expected = {"kind": "baseline", "execution_id": inputs["execution_id"], "video_sha256": inputs["video"]["sha256"],
                "source_sha256": inputs["source_sha256"], "window": inputs["registered_window"]}
    require(binding.get("schema") == "metadata-union-baseline-binding-v3" and binding.get("status") == "complete"
            and binding.get("expected") == expected and baseline_child is not None and baseline_child.get("status") == "complete"
            and baseline_child.get("worker_returncode") == 0, "full metadata baseline is not the actual same-execution complete child")
    baseline_root = files.root / "children" / inputs["execution_id"] / "baseline"
    _, baseline_ledger = files.freeze(baseline_root / "child_frozen.json")
    for role, name in (("candidate_artifact", "candidate.json"), ("frozen_artifact", "child_frozen.json"), ("request_artifact", "request.json")):
        target = files.record(binding[role], None if role == "frozen_artifact" else baseline_ledger)
        require(target == baseline_root / name, "full metadata baseline transport escaped its actual execution")
        if role != "request_artifact":
            require(binding[role]["sha256"] == baseline_child[role]["sha256"], "full metadata binding points to another baseline")
    original = summary("baseline", baseline_child)
    require(binding.get("original_summary") == original, "full metadata baseline replaced original legacy summary/identity")
    return request, binding, original


def metadata_actual_plan(files, child, ledger, inputs, gate, control_ledger, value):
    """Bind all ordered requests/regions/C2F using JSON only, before NPZ loads."""
    record = gate["metadata_union_plan"]; plan_path = files.record(record["artifact"], control_ledger)
    base = Path(child["candidate_artifact"]["path"]).parent
    require(plan_path == base / "metadata_admission/union_plan.json" and ledger.get(plan_path) == record["artifact"]["sha256"],
            "actual metadata request plan is not in both scoped and final freezes")
    plan = load_json(plan_path); live = gate["live_reference_calls"]
    ordered = plan["ordered_requests"]; count = plan["requested_pool_count"]
    regions = child["retrieval_selection"]["multi_region_retrieval"]["regions"]
    require(plan == record["plan"] and plan.get("schema") == "metadata-union-actual-pool-plan-v3"
            and plan.get("status") == "complete" and plan.get("queries") == 80 and plan.get("maximum_pools") == 6
            and plan.get("maximum_references") == 10000 and type(count) is int and 1 <= count <= 6
            and len(ordered) == count and plan.get("regional_plan_count") == len(regions)
            and plan.get("regional_plan") == regions and [r["audit_index"] for r in ordered] == list(range(count)),
            "incomplete or changed all-call metadata plan")
    calls = [{"request": r["request"], "reuse": r["reuse"]} for r in ordered]
    require(live.get("status") == "complete" and live.get("call_count") == live.get("audit_count") == count
            and live.get("calls") == calls and live.get("calls_sha256") == plan.get("calls_sha256") == digest(calls),
            "metadata actual request order changed")
    paths = sorted((files.record(r, control_ledger) for r in value["audit_artifacts"] if Path(r["path"]).suffix == ".json"), key=lambda p: int(p.stem))
    require(len(paths) == count, "an actual metadata pool is absent from frozen control")
    for i, (path, call) in enumerate(zip(paths, calls)):
        meta = load_json(path); request = call["request"]; pool = meta.get("reference_pool") or {}
        require(int(path.stem) == i and meta.get("public_cache_request") == request and meta.get("public_cache_reuse") == call["reuse"]
                and all(pool.get(k) == request[k] for k in ("source", "center", "radius_m", "cap"))
                and request.get("source") == "mapillary" and request.get("model_name") == meta["backbone"]["name"] == "megaloc"
                and request.get("device") == meta["backbone"]["device"] == "cpu" and request.get("query_mode") == meta["query_mode"] == "warp",
                "planned actual metadata call differs from its frozen pool/model/preprocessing")
        require({**meta["backbone"], "weights_sha256": meta["actual_model_state_sha256"]} == value["model"]
                and meta["preprocessing"] == value["query"]["preprocessing"]
                and meta["query_pixels_sha256"] == value["query"]["pixels_sha256"]
                and meta["query_descriptors_sha256"] == value["query"]["descriptors_sha256"],
                "all actual metadata pools must share the declared model and native query identity before array load")
    available = set(range(count)); mapping = []
    for region in regions:
        terminal_search(region)
        matches = [i for i in sorted(available) if all(calls[i]["request"].get(k) == region[field]
            for k, field in (("center", "center"), ("radius_m", "radius_m"), ("cap", "ref_cap")))]
        require(matches, "planned metadata region has no actual request"); chosen = matches[0]; mapping.append(chosen); available.remove(chosen)
    extra = sorted(available)
    require(plan["regional_to_audit_indices"] == mapping and plan["ordinary_refinement_call_indices"] == extra,
            "metadata requests were omitted/reordered or relabeled as refinement")
    if extra:
        trace = plan.get("c2f_trace")
        require(len(extra) == 1 and inputs["config"].get("vpr_coarse_to_fine") is True and isinstance(trace, dict)
                and finite(trace.get("tight_radius_m")) and trace["tight_radius_m"] > 0
                and calls[extra[0]]["request"]["radius_m"] == trace["tight_radius_m"]
                and trace == child["retrieval_selection"].get("vpr_coarse_to_fine"), "unmapped request is not a proven ordinary C2F call")
    return plan, plan_path


def metadata_selected_mappings(regions, value, union):
    """Independently bind original own-row evidence and canonical union rows."""
    rows = value.get("selected_audit_rows"); q_count = len(value["query"]["indices"])
    require(isinstance(rows, list) and len(rows) == q_count and len(value["decoded_reference_ids"]) == q_count,
            "original metadata source-row mapping is incomplete")
    by_id = {v: i for i, v in enumerate(union["ref_ids"].tolist())}
    rich, compact, canonical_exact = [], [], True
    for q, pair in enumerate(rows):
        require(isinstance(pair, list) and len(pair) == 2 and all(type(v) is int and v >= 0 for v in pair),
                "original metadata source-row mapping is invalid")
        number, row = pair
        require(number < len(regions), "original metadata selected pool is outside complete evidence")
        region = regions[number]; z = region["arrays"]
        require(row < len(z["query_indices"]) and int(z["query_indices"][row]) == value["query"]["indices"][q]
                and float(z["query_times"][row]) == value["query"]["times"][q], "original metadata selected query differs")
        selected = int(z["decoded_ref_indices"][row]); identifier = value["decoded_reference_ids"][q]
        require(0 <= selected < len(z["ref_ids"]) and identifier == str(z["ref_ids"][selected])
                and z["ref_latlon"][selected].tolist() == value["decoded_reference_latlon"][q]
                and float(z["raw_similarities"][row, selected]) == value["decoded_reference_cosines"][q],
                "original metadata own selected coordinate/cosine differs")
        require(identifier in by_id, "original metadata selected reference omitted from full union")
        col = by_id[identifier]; entry = union["mapping"][col]
        require(union["ref_latlon"][col].tolist() == value["decoded_reference_latlon"][q]
                and any(r["region"] == region["key"] and r["column"] == selected for r in entry["occurrences"]),
                "original metadata selected occurrence absent from full union")
        canonical = float(union["raw_similarities"][q, col])
        rich.append({"query_row": q, "audit_index": number, "audit_query_row": row, "region": region["key"],
            "region_column": selected, "ref_id": identifier, "raw_cosine": float(z["raw_similarities"][row, selected]),
            "global_column": col, "canonical_union_source": {"region": entry["canonical_region"],
                "region_column": entry["canonical_column"], "raw_cosine": canonical}})
        compact.append({"query_row": q, "global_column": col, "ref_id": identifier})
        canonical_exact = canonical_exact and canonical == value["decoded_reference_cosines"][q]
    return rich, compact if canonical_exact else None


def metadata_admission_matches(actual, expected, legacy_mapping):
    if digest(actual) == digest(expected):
        return True
    if legacy_mapping is None:
        return False
    historical = deepcopy(expected)
    historical["original_selected_sources"]["metadata"] = legacy_mapping
    return digest(actual) == digest(historical)


def metadata_union_component(files, child, ledger, inputs, before, gate, baseline_child):
    """Verify the frozen research bundle using independent union/path arithmetic.

    This binds every full path and objective, rather than proving global DP
    optimality; the independent standalone experiment covers that separate claim.
    """
    import numpy as np
    from monocular_osm import assisted_comparison as adapter, assisted_metadata_union as component
    from monocular_osm import metadata_union as core, assisted_policy as v1, assisted_policy_v3 as v3
    from monocular_osm.assisted_worker import _array_sha
    from monocular_osm.stages.assisted import summary
    request, binding, baseline_summary = metadata_baseline_binding(files, child, ledger, inputs, baseline_child)
    base = Path(child["candidate_artifact"]["path"]).parent
    control_path = files.record(gate["metadata_control"], ledger)
    require(control_path == base / "metadata_admission/control_frozen.json", "metadata control freeze has another scope")
    control, control_ledger = files.freeze(control_path)
    require(control.get("control_scope") == "pre_municipality_metadata_only"
            and control.get("status") == "complete_control_not_final_child" and control.get("kind") == "expanded"
            and control.get("source_sha256") == inputs["source_sha256"] and control.get("video_sha256") == inputs["video"]["sha256"],
            "metadata control source/scope/video differs")
    require(all(ledger.get(p) == h for p, h in control_ledger.items()), "metadata control bytes absent from final child freeze")
    raw_control = control_path.parent / "control_candidate.json"; raw_request = control_path.parent / "control_request.json"
    require(raw_control in control_ledger and raw_request in control_ledger and load_json(raw_request) == request,
            "metadata control request was not the actual expanded request")
    candidate = load_json(raw_control); old_comparison = candidate["comparison"]
    require(candidate.get("control_scope") == "pre_municipality_metadata_only" and candidate.get("status") == "complete"
            and candidate.get("kind") == "expanded" and candidate.get("window") == inputs["registered_window"]
            and candidate.get("source_sha256") == inputs["source_sha256"] and candidate.get("video_sha256") == inputs["video"]["sha256"]
            and old_comparison == gate["comparison"] and candidate["route_latlon"] == old_comparison["decoded_reference_latlon"]
            and candidate["route_timestamps"] == old_comparison["query"]["times"], "metadata control route/comparison is stale")
    actual_plan, plan_path = metadata_actual_plan(files, child, ledger, inputs, gate, control_ledger, old_comparison)
    old_summary = summary("expanded", candidate)
    identity = adapter.baseline_projection(baseline_child["comparison"], baseline_summary)["appearance_identity"]
    require(adapter.baseline_projection(old_comparison, old_summary)["appearance_identity"] == identity,
            "metadata raw model/query identity differs from the actual original baseline")
    regions = metadata_regions(files, old_comparison, control_ledger, inputs["registered_window"])
    require(len(regions) == actual_plan["requested_pool_count"], "actual metadata pools were dropped")
    expected_control_members = {raw_control, raw_request, plan_path} | {files.record(r, control_ledger) for r in old_comparison["audit_artifacts"]}
    require(set(control_ledger) == expected_control_members, "metadata control includes unscoped or omits original artifacts")
    require(gate["evidence"] == {**deepcopy(before), "expanded": old_summary}
            and gate["decision"] == gate["legacy_metadata_decision"] == v1.decide_assisted_policy(gate["evidence"])
            and gate["decision"]["selected_candidate"] != "expanded", "new metadata union ran before exact old-v1 decline")
    bundle = gate["metadata_union"]; artifacts = bundle["artifacts"]
    require(bundle.get("status") == "complete" and artifacts.get("schema") == "metadata-union-artifacts-v3", "new research artifacts unavailable")
    frozen_path = files.record(artifacts["frozen_artifact"], ledger)
    require(frozen_path == base / "metadata_union/component_frozen.json", "new research freeze escaped metadata-only scope")
    frozen, research_ledger = files.freeze(frozen_path)
    require(frozen.get("status") == "complete" and frozen.get("control_scope") == "metadata_union_research_before_selected_route"
            and frozen.get("sole_primary") == "reference_mean" and frozen.get("both_research_outputs") is True
            and frozen.get("profile_sha256") == component.PROFILE_SHA256 and frozen.get("source_sha256") == inputs["source_sha256"]
            and frozen.get("video_sha256") == inputs["video"]["sha256"], "new component profile/source/primary differs")
    require(all(ledger.get(p) == h for p, h in research_ledger.items()), "research artifact was not frozen in final child")
    directory = frozen_path.parent
    names = {str(p.relative_to(directory)).replace("\\", "/") for p in research_ledger}
    required = {"union_evidence.npz", "union_sources.json", "prediction_raw.json", "prediction_reference_mean.json",
                "comparison_raw.json", "comparison_reference_mean.json", "decoder_evidence.json", "controls.json",
                "runtime.json", "candidate.json", "admission.json", "union_plan.json"}
    require(required <= names and all(name in required or re.fullmatch(r"descriptor_rows_[0-9]{4}\.npz", name) for name in names),
            "both full research variants/controls/raw union were not retained")
    require(files.record(artifacts["union_artifact"], research_ledger) == directory / "union_evidence.npz"
            and files.record(artifacts["candidate_artifact"], research_ledger) == directory / "candidate.json", "union artifact role points to another bundle")
    require(file_sha(directory / "union_plan.json") == file_sha(plan_path), "new research replaced the frozen original all-call plan")
    controls = load_json(directory / "controls.json")
    require(controls["reference_encoding_identity"] == "UNVERIFIED legacy reference fingerprints retained", "historical encoding identity upgraded")
    require(all(r["receipt"].get("kv_source_sha256") == inputs["source_sha256"].get("monocular_osm/kartaview_vpr.py")
                for r in controls["descriptor_evidence"]), "descriptor receipt source is not the actual frozen retrieval implementation")
    def control_source(freeze, candidate_path, request_path, comp, expected):
        return {"freeze": str(freeze), "freeze_sha256": file_sha(freeze), "candidate": str(candidate_path),
                "candidate_sha256": file_sha(candidate_path), "identity": expected, "request": str(request_path),
                "audit_metadata": [str(p) for p in sorted((files.path(r["path"]) for r in comp["audit_artifacts"]
                    if Path(r["path"]).suffix == ".json"), key=lambda p: int(p.stem))]}
    baseline_root = Path(baseline_child["candidate_artifact"]["path"]).parent
    expected_controls = {"baseline": control_source(baseline_root / "child_frozen.json", baseline_root / "candidate.json",
        baseline_root / "request.json", baseline_child["comparison"], binding["expected"]),
        "metadata": control_source(control_path, raw_control, raw_request, old_comparison, {**binding["expected"], "kind": "expanded"})}
    legacy_gate = core.legacy_gate(baseline_summary, old_summary)
    require(controls["sources"] == expected_controls and controls["legacy_gate"] == legacy_gate,
            "research controls do not retain the actual original baseline/metadata freezes and legacy identity gate")
    descriptors = metadata_descriptor_rows(files, controls["descriptor_evidence"], regions, research_ledger)
    rebuilt = metadata_union_rebuild(regions, descriptors); saved = load_npz(directory / "union_evidence.npz")
    require(set(saved) == {"ref_ids", "ref_latlon", "raw_similarities", "query_indices", "query_times"}
            and all(_array_sha(saved[k]) == _array_sha(rebuilt[k]) for k in saved), "saved full union differs from all canonical pool rows")
    proof = load_json(directory / "union_sources.json")
    expected_sources = {k: v for k, v in rebuilt.items() if not isinstance(v, np.ndarray)}
    expected_sources.update(appearance_identity=identity, appearance_identity_sha256=digest(identity))
    require(proof == expected_sources, "union canonical mapping/input/actual query identity differs")
    expected_proofs = {}; outputs = {}; maths = {}
    for variant in ("raw", "reference_mean"):
        output = load_json(directory / f"prediction_{variant}.json"); math_record = metadata_union_math(saved, output, variant)
        path = math_record["path"]; maths[variant] = math_record; outputs[variant] = output
        expected_proof = {"schema": core.PROFILE["comparison_schema"], "appearance_identity": identity,
            "appearance_identity_sha256": digest(identity), "actual_decoder_profile_sha256": core.full.PROFILE_SHA256,
            "variant": variant, **{k: math_record[k] for k in ("raw_cosines", "unsmoothed_reference_latlon", "similarity_median", "dominant_coordinate_fraction")},
            "selected_source_rows": [{"query_row": q, "query_index": int(saved["query_indices"][q]), "query_time": float(saved["query_times"][q]),
                "global_column": int(col), "ref_id": rebuilt["mapping"][col]["ref_id"], "region": rebuilt["mapping"][col]["canonical_region"],
                "region_column": rebuilt["mapping"][col]["canonical_column"], "raw_cosine": math_record["raw_cosines"][q]} for q, col in enumerate(path)]}
        require(load_json(directory / f"comparison_{variant}.json") == expected_proof, "full selected global-ID/pool/column/raw proof differs")
        expected_proofs[variant] = expected_proof
    decoder = load_json(directory / "decoder_evidence.json")
    require(decoder.get("status") == "complete" and decoder.get("profile_sha256") == core.full.PROFILE_SHA256
            and digest(decoder["profile"]) == core.full.PROFILE_SHA256 and decoder["inputs"] == maths["raw"]["input_hashes"]
            and decoder.get("selected_variant") is None, "actual full decoder profile/grid/source evidence differs")
    expected_new = adapter.union_comparison({"status": "complete", "profile_sha256": core.PROFILE_SHA256,
        "outputs": outputs, "comparisons": expected_proofs, "union": saved}, baseline_child["comparison"], baseline_summary,
        old_comparison, old_summary, requested_pool_count=len(regions))
    baseline_audits = comparison(files, baseline_child["comparison"],
        files.freeze(baseline_root / "child_frozen.json")[1], inputs["registered_window"])
    baseline_mapping = []
    for q, (number, row) in enumerate(baseline_child["comparison"]["selected_audit_rows"]):
        audit = baseline_audits[number]; col = int(audit["decoded_ref_indices"][row])
        baseline_mapping.append({"query_row": q, "audit_index": number, "audit_query_row": row,
            "region": f"audit_{number:04d}", "region_column": col, "ref_id": str(audit["ref_ids"][col]),
            "raw_cosine": float(audit["raw_similarities"][row, col])})
    metadata_mapping, legacy_mapping = metadata_selected_mappings(regions, old_comparison, rebuilt)
    original_base, _, _, _ = core.legacy_summary(baseline_child["comparison"], baseline_summary)
    new_base = {**original_base, "score_identity": digest(identity)}
    sole = {"status": "complete", "route_valid": True, "point_count": 80, "score_kind": "raw_cosine",
        "score_identity": digest(identity), "similarity_median": maths["reference_mean"]["similarity_median"],
        "dominant_coordinate_fraction": maths["reference_mean"]["dominant_coordinate_fraction"]}
    expected_standalone = {"baseline_original": baseline_summary, "baseline_new_comparison": new_base,
        "metadata_original": old_summary, "legacy_gate": legacy_gate, "sole_primary_summary": sole,
        "sole_primary": "reference_mean", "decision": core.legacy_gate(new_base, sole), "decoder_identity_not_forged": True,
        "original_selected_sources": {"baseline": baseline_mapping, "metadata": metadata_mapping}}
    require(metadata_admission_matches(controls["standalone_admission"], expected_standalone, legacy_mapping),
            "standalone raw admissions/control mappings are stale")
    new = load_json(directory / "candidate.json")
    require(new == {"status": "complete", "route_latlon": outputs["reference_mean"]["route_latlon"],
                    "route_timestamps": saved["query_times"].tolist(), "comparison": expected_new}, "new sole-primary actual median5 answer differs")
    evidence = {**deepcopy(before), "baseline": adapter.augment_baseline_summary(baseline_child["comparison"], baseline_summary),
                "expanded": adapter.union_summary(expected_new, new["route_latlon"], new["route_timestamps"])}
    admission = load_json(directory / "admission.json"); decision = v3.decide_assisted_policy(evidence)
    require(admission == {"evidence": evidence, "decision": decision, "sole_primary": "reference_mean", "raw_is_diagnostic_only": True, "default_adopted": False}
            and bundle["evidence"] == evidence and bundle["decision"] == decision, "new raw-comparison admission/pure policy differs")
    runtime = load_json(directory / "runtime.json")
    require(runtime.get("profile_sha256") == component.PROFILE_SHA256 and runtime.get("profile") == component.PROFILE
            and runtime.get("source_sha256") == inputs["source_sha256"] and runtime.get("appearance_identity") == identity
            and runtime.get("new_neural_forwards") is False and runtime.get("network") is False
            and runtime.get("external_watchdog_required") is True and runtime.get("DP_started") is True
            and finite(runtime.get("elapsed_s")) and 0 <= runtime["elapsed_s"] <= 3600
            and runtime.get("estimated_extra_bytes", math.inf) <= 400 * 1024**2
            and len(runtime.get("threadpools") or []) >= 2 and all(p.get("num_threads") == 2 for p in runtime["threadpools"]),
            "new metadata runtime/source/resource evidence differs")
    require(all(runtime[name][k] >= 8 * 1024**3 for name in ("memory_before", "memory_end")
                for k in ("available_physical_bytes", "available_commit_bytes")), "new metadata lacks physical/true-commit floors")
    admitted = decision["selected_candidate"] == "expanded"
    require(is_metadata_union(child.get("comparison")) is admitted,
            "admitted metadata union and final full comparison kind disagree")
    if is_metadata_union(child.get("comparison")):
        final = deepcopy(child["comparison"]); final.pop("metadata_union_artifacts", None); final.pop("regional_audit_artifacts", None)
        require(admitted and final == expected_new and child["comparison"]["metadata_union_artifacts"] == artifacts
                and child["route_latlon"] == new["route_latlon"] and child["route_timestamps"] == new["route_timestamps"], "final selected full metadata path differs from admitted research")
    return {"admitted": admitted, "new_comparison": expected_new, "regions": regions,
            "audit_scope": "all-union source/raw/path/objective/median5/gates; global optimality not replayed"}


def retrieval_route(child, physical):
    """Retain the ordinary saved six-decimal track and median-five semantics."""
    from monocular_osm.baselines import vpr_track_only
    value = child["comparison"]
    expected = [[round(float(t), 2), round(float(ll[0]), 6), round(float(ll[1]), 6), round(float(c), 4)]
                for t, ll, c in zip(value["query"]["times"], value["decoded_reference_latlon"],
                                    value["decoded_reference_cosines"])]
    require((physical.get("vpr_prior") or {}).get("track") == expected,
            "saved ordinary retrieval track differs from its raw decoded evidence")
    prediction = vpr_track_only(physical)
    require(prediction is not None and prediction.route_latlon == child["route_latlon"]
            and child.get("route_timestamps") == value["query"]["times"],
            "retrieval route differs from the ordinary median-five selected track")


def metric_package(files, prepared, ledger, inputs, auto_quality=None):
    """Recompute native validity/plane quality from frozen measured arrays."""
    import numpy as np
    from monocular_osm.assisted_motion import PROFILE, PROFILE_SHA256
    from monocular_osm.assisted_policy import CONFIG_SHA256, policy_config
    from monocular_osm.assisted_worker import _array_sha
    require(prepared.get("profile_sha256") == PROFILE_SHA256
            and prepared.get("policy_config_sha256") == CONFIG_SHA256, "native metric profile changed")
    freeze = files.record(prepared["artifacts"]["freeze"], ledger)
    frozen = load_json(freeze)
    require(frozen.get("scoring_has_not_run") is True and frozen.get("profile_sha256") == PROFILE_SHA256
            and frozen.get("policy_config_sha256") == CONFIG_SHA256, "native metric package lacks its fixed inference freeze")
    for record in frozen["artifacts"].values():
        files.record(record, ledger)
    registration = load_json(freeze.parent / "metric_registration.json")
    summary = load_json(freeze.parent / "metric_summary.json")
    require(summary == prepared["summary"] and registration["window"] == inputs["registered_window"]
            and registration["video"]["sha256"] == inputs["video"]["sha256"]
            and registration["profile"] == dict(PROFILE), "native metric identity or summary is stale")
    require(all(value is False for value in registration["input_policy"].values())
            and set(registration["input_policy"]) == {"ground_truth", "external_trajectory", "calibration",
                "gps_or_oxts", "position_prior", "filter_execution"}, "native preflight used forbidden inputs")
    require(frozen["source_sha256"] == registration["source_sha256"] == prepared["source_sha256"],
            "native metric source declaration is stale")
    sources(files, freeze.parent, registration["source_sha256"])
    parent_sources = {str(k).replace("\\", "/"): v for k, v in inputs["source_sha256"].items()}
    require(all(parent_sources.get("monocular_osm/" + name) == sha for name, sha in registration["source_sha256"].items()),
            "native metric source differs from the parent source")
    if auto_quality is not None:
        require(registration["auto_quality"] == summary["auto_quality"] == auto_quality,
                "native metric preflight belongs to a different auto-VO measurement")
    if summary.get("status") != "complete":
        require(summary.get("eligible") is False, "incomplete native metric package was admitted")
        return summary
    data = load_npz(freeze.parent / "native_metric_inputs.npz")
    valid, times = data["valid"], data["timestamps"]
    native = summary["native_quality"]
    require(str(data["profile_sha256"]) == PROFILE_SHA256 and str(data["input_signature"]) == summary["input_signature"]
            and summary["input_signature"] == digest({"registration_sha256": file_sha(freeze.parent / "metric_registration.json"),
                                                     "native_quality": native, "plane": summary["plane"]}), "native metric signature is stale")
    require(valid.dtype.kind == "b" and valid.ndim == 1 and times.shape == valid.shape
            and np.isfinite(times).all() and np.all(np.diff(times) > 0)
            and native["n_frames"] == len(valid) and native["n_valid"] == int(valid.sum())
            and native["valid_fraction"] == float(valid.mean())
            and native["duration_s"] == float(times[-1] - times[0])
            and native["timestamps_sha256"] == _array_sha(times)
            and native["K_sha256"] == _array_sha(data["K"]), "native VO measurements differ from policy scalars")
    grid = registration["native_grid"]
    shape = native["native_shape"]
    from monocular_osm.visual_odometry import default_intrinsics
    require(len(shape) == 3 and shape[2] == 3 and all(type(n) is int and n > 0 for n in shape)
            and np.array_equal(data["K"], default_intrinsics(shape[1], shape[0], hfov_deg=70.))
            and grid["stride"] == native["stride"] == max(1, round(grid["fps"] / 10.))
            and grid["fps"] == native["fps"] and sha_string(native["pixels_sha256"]), "native camera or 10Hz input grid changed")
    ordinals = grid["first_ordinal"] + np.arange(len(times), dtype="<i8") * grid["stride"]
    require(grid["expected_n_frames"] == len(times) and native["full_declared_grid"] is True
            and np.array_equal(data["frame_ordinals"], ordinals)
            and native["frame_ordinals_sha256"] == _array_sha(data["frame_ordinals"]), "native metric grid is incomplete")
    stop = min(grid["container_frame_count"], int(inputs["registered_window"][1] * grid["fps"])
               if inputs["registered_window"][1] is not None else grid["container_frame_count"])
    require(grid["first_ordinal"] == int(inputs["registered_window"][0] * grid["fps"])
            and grid["stop_ordinal_exclusive"] == stop
            and len(times) == max(0, (stop - grid["first_ordinal"] + grid["stride"] - 1) // grid["stride"])
            and native["first_s"] == float(times[0]) and native["last_s"] == float(times[-1])
            and times[0] >= inputs["registered_window"][0] - 1
            and (inputs["registered_window"][1] is None or times[-1] <= inputs["registered_window"][1] + 1),
            "native metric package truncated or changed the registered full input window")
    for name, shape in {"centers": (len(times), 3), "xz": (len(times), 2), "n_inliers": (len(times),),
                        "rotations": (len(times), 3, 3), "translations": (len(times), 3), "K": (3, 3)}.items():
        require(data[name].shape == shape and np.isfinite(data[name]).all(), "native VO arrays are incomplete")
    raw, lengths = data["plane_raw_m"], data["plane_lengths_m"]
    moving = valid[1:] & (np.linalg.norm(np.diff(data["xz"], axis=0), axis=1) > 1e-6)
    require(raw.shape == lengths.shape == moving.shape and np.isfinite(lengths).all()
            and np.all(lengths >= 0) and not np.isinf(raw).any(), "native plane arrays are invalid")
    measured = np.isfinite(raw) & moving & (lengths > 0)
    coverage = float(measured.sum() / max(1, moving.sum()))
    spread = float(np.median(np.abs(raw[measured] / lengths[measured] - 1))) if measured.sum() >= 10 else None
    require(summary["plane"]["coverage"] == coverage and summary["plane"]["rel_spread"] == spread
            and summary["plane"]["camera_height_m"] == 1.5, "plane admission scalars are not measured values")
    policy = policy_config()
    eligible = (auto_quality is not None and auto_quality["duration_s"] >= policy["pf_duration_min_s"]
                and auto_quality["valid_fraction"] >= policy["pf_vo_valid_min"]
                and native["valid_fraction"] >= policy["pf_vo_valid_min"] and coverage >= policy["pf_plane_coverage_min"]
                and spread is not None and spread <= policy["pf_plane_rel_spread_max"])
    require(summary["eligible"] is eligible, "native metric admission differs from fixed thresholds")
    return summary


def cues(files, child, ledger, inputs, reader_issues):
    import numpy as np
    from monocular_osm.assisted_worker import _array_sha
    from monocular_osm.poi_clues import build_poi_hypotheses
    result = child["cues"]
    quality = child["auto_quality"]
    data = load_npz(files.record(quality["vo_artifact"], ledger))
    valid, times = data["valid"], data["timestamps"]
    require(valid.dtype.kind == "b" and valid.ndim == 1 and times.shape == valid.shape
            and len(times) >= 2 and np.isfinite(times).all() and np.all(np.diff(times) > 0)
            and quality["n_frames"] == len(valid) and quality["n_valid"] == int(valid.sum())
            and quality["valid_fraction"] == float(valid.mean())
            and quality["duration_s"] == float(times[-1] - times[0])
            and quality["timestamps_sha256"] == _array_sha(times)
            and quality["video_sha256"] == inputs["video"]["sha256"]
            and quality["window"] == inputs["registered_window"], "cue VO policy scalars lack native video measurements")
    require(result["vo"] == quality and result["duration_s"] == quality["duration_s"], "cue summary is stale")
    signed = load_npz(files.record(quality["signed_vo_cache"], ledger))
    signature = json.loads(str(signed["cache_signature"].item()))
    extraction, runtime = signature["extraction"], signature["odometry_runtime"]
    require(signature["schema"] == 2 and signature["vo_algo_version"] == int(signed["vo_algo_version"])
            and type(signature["vo_algo_version"]) is int and signature["vo_algo_version"] > 0
            and all(np.array_equal(signed[key], data[key]) for key in ("xz", "valid"))
            and runtime["workers"] == 2 and all(isinstance(runtime[k], str) and runtime[k] for k in ("numpy", "opencv"))
            and sha_string(signature["frames_sha256"])
            and files.config_path(signature["video"]["path"]) == files.config_path(inputs["video"]["path"])
            and extraction["start_sec"] == inputs["registered_window"][0]
            and extraction["end_sec"] == inputs["registered_window"][1]
            and extraction["stride"] == quality["stride"] and extraction["fps"] == quality["fps"]
            and extraction["n_frames"] == len(times) and extraction["max_frames"] is None
            and extraction["frame_max_width"] is None and extraction["frame_memory_mb"] == 128.
            and extraction["timestamps_sha256"] == hashlib.sha256(np.asarray(times, dtype="<f8").tobytes()).hexdigest(),
            "cue signed VO cache lacks the full native pixel/extraction/runtime identity")
    require(data["xz"].shape == (len(times), 2) and np.isfinite(data["xz"]).all()
            and data["K"].shape == (3, 3) and np.isfinite(data["K"]).all()
            and times[0] >= inputs["registered_window"][0] - 1
            and (inputs["registered_window"][1] is None or times[-1] <= inputs["registered_window"][1] + 1),
            "cue VO arrays are incomplete or outside the registered video")
    if result["ocr"]["status"] == "complete":
        base = Path(child["candidate_artifact"]["path"]).parent
        ocr = load_json(base / "poi_ocr.json")
        inventory = load_json(base / "poi_inventory.json")
        require(base / "poi_ocr.json" in ledger and base / "poi_inventory.json" in ledger,
                "native clues lack frozen OCR/inventory snapshots")
        inventory_meta = child["inventory"]
        bbox, query = inventory["bbox_wsen"], inventory["query"]
        require(inventory["city"] == inputs["city"] and inventory["scope"] == "named_city_bbox"
                and inventory["snapshot_sha256"] == digest(inventory["pois"])
                and sha_string(inventory["response_sha256"])
                and len(bbox) == 4 and all(finite(v) for v in bbox)
                and f"({bbox[1]:.7f},{bbox[0]:.7f},{bbox[3]:.7f},{bbox[2]:.7f})" in query
                and 'nwr["shop"]' in query and "out center tags" in query
                and files.record({"path": inventory_meta["path"], "sha256": inventory_meta["file_sha256"]}, ledger) == base / "poi_inventory.json"
                and all(inventory_meta[k] == inventory[k] for k in ("city", "scope", "snapshot_sha256", "response_sha256", "query", "bbox_wsen")),
                "cue inventory is not its fingerprinted full named-city public snapshot")
        meta = child["ocr_provenance"]
        from monocular_osm.scene_text import is_position_stamp
        from monocular_osm.visual_odometry import default_intrinsics
        shape = meta["frame_shape_hw"]
        require(len(shape) == 2 and all(type(n) is int and n > 0 for n in shape)
                and np.array_equal(data["K"], default_intrinsics(shape[1], shape[0]))
                and all(not is_position_stamp(item["text"]) and times[0] <= item["t_sec"] <= times[-1]
                        for item in ocr["detections"]), "cue OCR contains position stamps or a different pixel/camera grid")
        require(ocr["provenance"] == meta and meta["video_sha256"] == inputs["video"]["sha256"]
                and meta.get("source") == "same_video_native_frames" and meta.get("native_bbox_grid") is True
                and meta.get("sample_count") == 24 and meta.get("min_len") == 2
                and meta.get("min_confidence") == .3 and meta.get("super_res") is False
                and meta.get("use_gpu") is False and meta.get("position_stamps_rejected") is True
                and meta.get("position_stamp_filter") == "scene_text.is_position_stamp"
                and meta["window"] == [float(times[0]), float(times[-1])]
                and meta["K_sha256"] == _array_sha(data["K"])
                and digest(ocr["detections"]) == meta["detections_sha256"]
                and not reader_issues(meta), "cue OCR lacks its fixed native/CPU/runtime identity")
        hypotheses = build_poi_hypotheses(ocr["detections"], inventory)
        require(hypotheses == child["clues"] and result["ocr"]["clue_branch_count"] == len(hypotheses["candidate_clusters"])
                and result["ocr"]["detections_sha256"] == meta["detections_sha256"]
                and result["ocr"]["inventory_sha256"] == inventory["snapshot_sha256"], "cue branches do not match literal frozen evidence")
    if child.get("metric_preflight"):
        measured = metric_package(files, child["metric_preflight"], ledger, inputs, quality)
        plane = {**measured["plane"], "native_quality": measured["native_quality"], "native_eligible": measured["eligible"]}
        require(result["plane"] == plane, "cue plane summary differs from its metric package")
    else:
        require(result["plane"].get("measured") is False, "cue claims an unmeasured road-plane signal")
        from monocular_osm.assisted_policy import policy_config
        policy = policy_config()
        require(quality["duration_s"] < policy["pf_duration_min_s"]
                or quality["valid_fraction"] < policy["pf_vo_valid_min"], "eligible auto-VO omitted native plane preflight")


def image_child(files, child, ledger, inputs, cue):
    import numpy as np
    from monocular_osm.poi_image_track import PROFILE, PROFILE_SHA256, _validated_evidence, plan_single_branch
    plan = child["image_plan"]
    require(plan == plan_single_branch(cue["clues"], cue["ocr_provenance"]), "image plan differs from the sole native clue branch")
    require(plan["profile"] == dict(PROFILE) and plan["profile_sha256"] == PROFILE_SHA256
            and child["uses_rigid_vo"] is False and child["retained_branch_count"] == child["successful_branch_count"] == 1
            and child["radius_m"] == 3000 and child["cap"] == 1500
            and child["query_mode"] == "crops" and child["score_normalization"] == "reference_mean",
            "image-only component changed its fixed bbox-v2/no-VO profile")
    records = child["evidence_artifacts"]
    require(len(records) == 1, "single-branch image component has an incomplete reference plan")
    base = Path(child["candidate_artifact"]["path"]).parent
    evidence = load_npz(files.record({**records[0], "path": base / relative(records[0]["path"])}, ledger))
    evidence["provenance"] = child["provenance"]["source"]
    indices, times, refs, scores, ids, frame, xy, _ = _validated_evidence(evidence, plan, inputs["video"]["sha256"])
    from monocular_osm.assisted_worker import _array_sha
    native = load_npz(files.record(cue["auto_quality"]["vo_artifact"]))
    grid = child["acquisition"]
    require(grid["n_frames"] == len(native["timestamps"]) and grid["stride"] == cue["auto_quality"]["stride"]
            and grid["timestamps_sha256"] == _array_sha(native["timestamps"])
            and np.array_equal(indices, np.linspace(0, grid["n_frames"] - 1, min(80, grid["n_frames"])).astype(int))
            and np.array_equal(times, native["timestamps"][indices]), "image queries differ from the native video grid")
    chosen = np.asarray(child["chosen_ref_indices"])
    require(chosen.dtype.kind in "iu" and chosen.shape == indices.shape
            and np.all(chosen >= 0) and np.all(chosen < len(refs)), "image chosen reference indices are invalid")
    centered = scores - scores.mean(axis=0, keepdims=True)
    require(child["chosen_ref_ids"] == ids[chosen].tolist()
            and child["raw_route_latlon"] == refs[chosen].tolist()
            and child["selected_raw_cosines"] == scores[np.arange(len(times)), chosen].tolist()
            and child["selected_centered_scores"] == centered[np.arange(len(times)), chosen].tolist()
            and child["route_timestamps"] == times.tolist() and child["query_indices"] == indices.tolist(),
            "image output differs from its full reference/query matrix")
    selected_xy = xy[chosen]
    smoothed = np.array([np.median(selected_xy[max(0, i - 2):i + 3], axis=0) for i in range(len(times))])
    route = np.column_stack(frame.to_latlon(smoothed[:, 0], smoothed[:, 1]))
    require(np.allclose(child["route_latlon"], route, rtol=0, atol=1e-10), "image route is not its fixed metric median track")
    from monocular_osm.poi_image_track import _provenance
    require(child["provenance"] == _provenance(plan, inputs["video"]["sha256"], indices, times, refs, scores, ids,
                                             evidence["provenance"]), "image full-matrix/input provenance is stale")
    cpu = evidence["provenance"].get("cpu_runtime") or {}
    require(cpu.get("device") == "cpu" and cpu.get("intra_op_threads") == expected_cpu_threads(inputs.get("config"))
            and type(cpu.get("inter_op_threads")) is int and cpu["inter_op_threads"] > 0,
            "image acquisition lacks bounded actual CPU runtime proof")


def image_union_child(files, child, ledger, inputs, cue):
    """Replay every literal pool and the full union DP, without image/model I/O."""
    import numpy as np
    from monocular_osm.assisted_worker import _array_sha
    from monocular_osm.poi_image_union import plan_image_union, localize_image_union
    from monocular_osm.stages.poi_only import _safe
    plan = plan_image_union(cue["clues"], cue["ocr_provenance"])
    base = Path(child["candidate_artifact"]["path"]).parent
    require(child["image_union_plan"] == plan and plan["status"] == "ready",
            "image union changed or pruned its complete literal branch plan")
    persisted = files.path(base / "image_union_plan.json")
    require(persisted in ledger and load_json(persisted) == plan,
            "image union plan was not persisted in the child freeze")
    native = load_npz(files.record(cue["auto_quality"]["vo_artifact"]))
    times = native["timestamps"]
    grid, auto = child["acquisition"], cue["auto_quality"]
    require(grid["n_frames"] == auto["n_frames"] == len(times)
            and grid["stride"] == auto["stride"]
            and grid["timestamps_sha256"] == auto["timestamps_sha256"] == _array_sha(times)
            and grid["native_frame_shape_hw"] == cue["ocr_provenance"]["frame_shape_hw"]
            and cue["window"] == auto["window"] == inputs["registered_window"],
            "image union queries differ from the frozen native cue grid/window")
    regions = child["regions"]
    require(len(regions) == len(plan["region_plans"]) and not child["failed_regions"],
            "image union omitted or failed a planned literal region")
    array_keys = {"query_indices", "query_times", "ref_ids", "ref_latlon", "similarities"}
    matrices, records = {}, []
    for region, expected in zip(regions, plan["region_plans"]):
        require(region.get("status") == "complete" and region["hypothesis_id"] == expected["hypothesis_id"]
                and region["request_sha256"] == expected["request_sha256"],
                "image union region order/completion differs from its frozen plan")
        provenance = region["provenance"]
        record = provenance["matrix_artifact"]
        path = files.record({**record, "path": base / relative(record["path"])}, ledger)
        require(path not in matrices, "image union reused a matrix artifact for multiple branches")
        arrays = load_npz(path)
        require(set(arrays) == array_keys, "image region matrix omitted or added evidence arrays")
        runtime = provenance["cpu_runtime"]
        require(all(runtime.get(key) == child["cpu_runtime"].get(key) for key in
                    ("device", "intra_op_threads", "inter_op_threads", "torch_version"))
                and provenance["backbone"] == child["actual_backbone"]
                and provenance["query_identity"]["actual_model_state_sha256"] == child["actual_model_state"]["sha256"],
                "image region model/runtime differs from the actual visual child")
        matrices[path] = {"status": "complete", **arrays, "provenance": provenance}
        records.append(record)

    def retrieve(region, *, n_query, query_mode, query_cache):
        require(n_query == 80 and query_mode == "crops", "image replay changed its fixed query profile")
        position = plan["region_plans"].index(region)
        path = files.path(base / relative(records[position]["path"]))
        return matrices[path]

    replay = localize_image_union(plan, timestamps=times, query_video_sha256=inputs["video"]["sha256"],
                                  retrieve_evidence=retrieve, query_cache={})
    require(replay.get("status") == "complete", "image union full evidence replay declined: " + str(replay.get("reason")))
    expected_union = replay.pop("reference_union")
    declared_union = child["reference_union"]
    union_record = declared_union["matrix_artifact"]
    records.append(union_record)
    require(child["evidence_artifacts"] == records,
            "image union artifact list differs from all planned regions plus the complete union")
    union_path = files.record({**union_record, "path": base / relative(union_record["path"])}, ledger)
    require(union_path not in matrices, "image union artifact aliases a branch matrix")
    saved = load_npz(union_path)
    require(set(saved) == array_keys, "image union matrix omitted or added evidence arrays")
    for name in array_keys:
        expected = np.asarray(expected_union[name])
        require(saved[name].dtype == expected.dtype and np.array_equal(saved[name], expected),
                "image union matrix differs from the deterministic all-region union")
    array_names = {"query_indices", "query_times", "ref_latlon", "similarities"}
    compact = {key: value for key, value in expected_union.items() if key not in array_names}
    require({key: value for key, value in declared_union.items() if key != "matrix_artifact"} == _safe(compact),
            "image union membership/canonical occurrence/identity claims changed")
    require(all(child.get(key) == value for key, value in _safe(replay).items()),
            "image union route/decoder/profile/provenance differs from its full-matrix replay")


def poi_artifacts(files, child, ledger, cue):
    """Check the selected joint component's real matrices and reused native VO."""
    import numpy as np
    from monocular_osm.assisted_worker import _array_sha
    inputs, audit = child["poi_inputs"], child["poi_audit"]
    require(file_sha(inputs["map"]["cache_path"]) == inputs["map"]["sha256"], "joint whole-city map database changed")
    base = Path(child["candidate_artifact"]["path"]).parent
    native = load_npz(files.record(cue["auto_quality"]["vo_artifact"]))
    vo = load_npz(files.path(base / "poi_vo.npz"))
    require(base / "poi_vo.npz" in ledger and all(np.array_equal(vo[k], native[k]) for k in ("xz", "valid", "timestamps", "K")),
            "joint component replaced its already acquired native VO/camera grid")
    require(child["route_timestamps"] == native["timestamps"].tolist(), "joint route does not cover its full native video timeline")
    for region, record in zip(audit["regions"], inputs["evidence_artifacts"]):
        evidence = load_npz(files.record({**record, "path": base / relative(record["path"])}, ledger))
        declared = region["provenance"]["evidence_identity"]
        expected = {"query_indices": evidence["query_indices"].tolist(), "query_times": evidence["query_times"].tolist(),
            "ref_ids": evidence["ref_ids"].astype(str).tolist(), "similarities_sha256": _array_sha(evidence["similarities"]),
            "ref_latlon_sha256": _array_sha(evidence["ref_latlon"])}
        require(expected == declared and np.array_equal(evidence["query_times"], native["timestamps"][evidence["query_indices"]]),
                "joint region matrix differs from its complete same-video query/reference identity")


def terminal_search(record):
    """A public completed empty/sparse listing is a result; errors are not."""
    if record.get("status") == "ok" and not record.get("error"):
        return True
    proof = (record.get("track_decoding") or {}).get("terminal_search") or {}
    count = proof.get("count")
    require(record.get("status") == "unavailable" and not record.get("error")
            and type(count) is int and 0 <= count < 30
            and proof.get("status") == ("complete_empty" if count == 0 else "complete_sparse")
            and proof.get("source") == "mapillary", "unavailable reference region has no completed listing proof")
    stats, snapshot = proof["listing"], proof["snapshot"]
    refs = proof["reference_inventory"]
    require(stats.get("listing_stats_known") is True and stats.get("listing") == "tiles"
            and not stats.get("truncated") and type(stats.get("lost")) is int and stats["lost"] == 0
            and stats["snapshot"] == snapshot and snapshot["n_refs"] == count and len(refs) == count,
            "sparse listing is partial or has stale counts")
    from monocular_osm.kartaview_vpr import _reference_snapshot, _fetch_signature
    require(_reference_snapshot(refs)["refs_sha256"] == snapshot["refs_sha256"], "sparse inventory digest is stale")
    require(proof["fetch_signature"] == _fetch_signature(record["center"], record["radius_m"], record["ref_cap"]),
            "sparse proof identifies another planned reference region")
    return True


def expansion_v3(files, child, ledger, inputs, before, baseline_child):
    from monocular_osm import assisted_policy as v1
    from monocular_osm.stages.assisted import summary
    selection = child["retrieval_selection"]; multi = selection["multi_region_retrieval"]
    regions = multi["regions"]
    require(regions and len(regions) <= inputs["config"]["vpr_region_cap"], "metadata reference plans are incomplete")
    complete = True
    for region in regions:
        try:
            terminal_search(region)
        except (ValueError, KeyError, TypeError):
            complete = False
    gate = selection["assisted_expansion_gate"]
    require(gate.get("ordered_subplans") == ["metadata_and_corridors", "full_support_metadata_union_only_after_legacy_decline",
                                           "municipality_only_if_not_admitted"], "v3 metadata-first ordering changed")
    old = gate.get("comparison")
    if old is None:
        require(bool(gate.get("error")) and gate.get("municipality_skipped") is False and child["status"] != "complete",
                "unknown metadata failure was promoted to complete expansion")
        return
    union = gate.get("metadata_union")
    verified = metadata_union_component(files, child, ledger, inputs, before, gate, baseline_child) if union is not None else None
    old_audits = ([r["audit"] for r in verified["regions"]] if verified else
                  comparison(files, old, ledger, inputs["registered_window"], require_all=complete))
    coords = old.get("decoded_reference_latlon") or []
    evidence = {**deepcopy(before), "expanded": summary("expanded", {
        "status": "complete" if complete and len(coords) >= 2 else "partial", "route_latlon": coords, "comparison": old})}
    decision = v1.decide_assisted_policy(evidence)
    require(gate["evidence"] == evidence and gate["decision"] == gate["legacy_metadata_decision"] == decision,
            "v3 did not first replay exact original-v1 metadata admission/decoder identity")
    admitted = decision["selected_candidate"] == "expanded"
    if admitted:
        require(union is None and gate.get("metadata_control") is None and not is_metadata_union(child.get("comparison")),
                "new full decode ran after legacy metadata was already admitted")
    elif union is not None:
        admitted = verified["admitted"]
    else:
        require(not is_metadata_union(child.get("comparison")), "new full comparison has no frozen both-variant research proof")
    require(gate.get("municipality_skipped") is admitted, "v3 municipality skip disagrees with actual original/new admissions")
    city = selection.get("city_cover_retrieval"); all_regions = list(regions)
    if admitted:
        require(city is None and child["status"] == "complete" and not gate.get("error"), "future municipality/partial output after metadata admission")
    else:
        require(isinstance(city, dict), "declined metadata did not retain its municipality fallback")
        coarse, fine = city.get("coarse_regions") or [], city.get("fine_regions") or []
        if city.get("status") not in {"ok", "unavailable"} or not coarse:
            complete = False
        else:
            n = city["n_cells"]
            require(type(n) is int and 0 < n <= 64 and len(coarse) == n
                    and [r["cell_index"] for r in coarse] == list(range(n))
                    and len(fine) == len(city.get("densified_cell_indices") or []) <= inputs["config"]["vpr_city_fine_regions"]
                    and [r["cell_index"] for r in fine] == (city.get("densified_cell_indices") or []), "municipality cells omitted or reordered")
            require(city["plan"]["boundary_source"]["query"] == inputs["city"]
                    and city["plan"]["boundary_validation"]["administrative_boundary_verified"] is True, "municipality is not a verified named-city plan")
            for region in coarse + fine:
                try:
                    terminal_search(region)
                except (ValueError, KeyError, TypeError):
                    complete = False
            require(all(r["ref_cap"] == inputs["config"]["vpr_city_coarse_cap"] for r in coarse)
                    and all(r["ref_cap"] == inputs["config"]["vpr_ref_cap"] for r in fine), "municipality budgets changed")
            all_regions.extend(coarse + fine)
    require(child["status"] != "complete" or complete, "incomplete v3 search was promoted")
    if child["status"] == "complete":
        if is_metadata_union(child.get("comparison")):
            old_final = deepcopy(old); old_final["audit_artifacts"] = child["comparison"]["regional_audit_artifacts"]
            audits = comparison(files, old_final, ledger, inputs["registered_window"], require_all=True)
            require(len(audits) == len(old_audits), "full path changed or pruned original metadata pools")
            from monocular_osm.assisted_worker import _array_sha
            for actual, original in zip(audits, old_audits):
                require(all(_array_sha(actual[k]) == _array_sha(original[k]) for k in
                    ("query_indices", "query_times", "ref_ids", "ref_latlon", "raw_similarities", "decoded_ref_indices", "decoded_cosines")),
                    "full selected candidate changed original regional evidence")
        else:
            audits = comparison(files, child["comparison"], ledger, inputs["registered_window"], require_all=True)
        for region in all_regions:
            if region["status"] != "ok":
                continue
            require(any((a.get("reference_pool") or {}).get("center") == region["center"]
                        and (a.get("reference_pool") or {}).get("radius_m") == region["radius_m"]
                        and (a.get("reference_pool") or {}).get("cap") == region["ref_cap"] for a in audits),
                    "successful planned v3 region lacks its actual complete matrix")


def expansion(files, child, ledger, inputs, before, *, baseline_child=None):
    if inputs["config"].get("assisted_profile", "v1") == "v3":
        return expansion_v3(files, child, ledger, inputs, before, baseline_child)
    from monocular_osm.assisted_profiles import resolve_policy
    from monocular_osm.stages.assisted import summary
    decide_assisted_policy = resolve_policy(inputs["config"].get("assisted_profile", "v1")).decide_assisted_policy
    selection = child["retrieval_selection"]
    multi = selection["multi_region_retrieval"]
    regions = multi["regions"]
    require(regions and len(regions) <= inputs["config"]["vpr_region_cap"], "metadata reference plans are incomplete")
    complete = True
    for region in regions:
        try:
            terminal_search(region)
        except (ValueError, KeyError, TypeError):
            complete = False
    gate = selection["assisted_expansion_gate"]
    if gate.get("error"):
        require(gate.get("municipality_skipped") is False and child["status"] != "complete",
                "failed metadata admission was promoted to a complete expanded candidate")
        return
    require(gate["ordered_subplans"] == ["metadata_and_corridors", "municipality_only_if_not_admitted"],
            "expanded search changed its metadata-first plan")
    comparison(files, gate["comparison"], ledger, inputs["registered_window"], require_all=complete)
    metadata_route = gate["comparison"].get("decoded_reference_latlon") or []
    status = "complete" if complete and len(metadata_route) >= 2 else "partial"
    evidence = {**deepcopy(before), "expanded": summary("expanded", {
        "status": status, "route_latlon": metadata_route, "comparison": gate["comparison"]})}
    decision = decide_assisted_policy(evidence)
    require(gate["evidence"] == evidence and gate["decision"] == decision
            and gate["municipality_skipped"] is (decision["selected_candidate"] == "expanded"),
            "metadata municipality-admission decision is stale or is not the fixed policy")
    city = selection.get("city_cover_retrieval")
    all_regions = list(regions)
    if gate["municipality_skipped"]:
        require(city is None, "municipality search ran after metadata was already admitted")
    else:
        require(isinstance(city, dict), "required municipality search is missing")
        coarse, fine = city.get("coarse_regions") or [], city.get("fine_regions") or []
        if city.get("status") not in {"ok", "unavailable"} or not coarse:
            complete = False
        else:
            n = city["n_cells"]
            require(type(n) is int and 0 < n <= 64 and len(coarse) == n
                    and [r["cell_index"] for r in coarse] == list(range(n))
                    and len(fine) == len(city.get("densified_cell_indices") or []) <= inputs["config"]["vpr_city_fine_regions"]
                    and [r["cell_index"] for r in fine] == (city.get("densified_cell_indices") or []), "municipality cell plan is incomplete")
            require(city["plan"]["boundary_source"]["query"] == inputs["city"]
                    and city["plan"]["boundary_validation"]["administrative_boundary_verified"] is True,
                    "municipality plan is not a verified named-city boundary")
            for region in coarse + fine:
                try:
                    terminal_search(region)
                except (ValueError, KeyError, TypeError):
                    complete = False
            require(all(r["ref_cap"] == inputs["config"]["vpr_city_coarse_cap"] for r in coarse)
                    and all(r["ref_cap"] == inputs["config"]["vpr_ref_cap"] for r in fine),
                    "municipality reference budgets differ from their fixed plan")
            all_regions.extend(coarse + fine)
    require(child["status"] != "complete" or complete, "partial/failed expanded reference plans were promoted")
    if child["status"] == "complete":
        audits = comparison(files, child["comparison"], ledger, inputs["registered_window"], require_all=True)
        for region in all_regions:
            if region["status"] != "ok":
                continue
            require(any((a.get("reference_pool") or {}).get("center") == region["center"]
                        and (a.get("reference_pool") or {}).get("radius_m") == region["radius_m"]
                        and (a.get("reference_pool") or {}).get("cap") == region["ref_cap"] for a in audits),
                    "successful planned region lacks its bound full-matrix audit")


def audit_assisted(rep, score, city, segment, *, poi_issues, filter_issues, reader_issues, ocr_identity):
    """Return JSON-safe issues and semantic identities for fleet consistency."""
    result = {"issues": [], "models": [], "weights": [], "runtimes": [], "ocr_models": [], "policy": None,
              "source": None, "configuration": None, "repeat_inputs": None}
    try:
        from monocular_osm.assisted_profiles import resolve_policy
        from monocular_osm.assisted_worker import _array_sha
        from monocular_osm.stages.assisted import REQUESTS, summary, valid_route
        provenance = rep["provenance"]
        cfg, inputs, audit = provenance["config"], provenance["assisted_inputs"], provenance["assisted_audit"]
        profile = cfg.get("assisted_profile", "v1")
        policy_module = resolve_policy(profile)
        CONFIG_SHA256, POLICY_VERSION = policy_module.CONFIG_SHA256, policy_module.POLICY_VERSION
        decide_assisted_policy, policy_config = policy_module.decide_assisted_policy, policy_module.policy_config
        window = [float(value) if value else None for value in segment.split(":")]
        require(cfg.get("answer") == "assisted" and cfg.get("city") == city
                and (score.get("meta") or {}).get("mode") == "assisted"
                and (score.get("meta") or {}).get("method") == "assisted_policy", "selected assisted mode/method/config disagree")
        require(inputs.get("experimental") is True and inputs.get("validation_status") == "posthoc_dev_unvalidated"
                and inputs["city"] == city and inputs["registered_window"] == window and inputs["config"] == cfg,
                "assisted profile is not the registered experimental video/city input")
        require(Path(inputs["video"]["path"]).is_absolute() and sha_string(inputs["video"]["sha256"])
                and file_sha(inputs["video"]["path"]) == inputs["video"]["sha256"], "assisted encoded video content changed or is unavailable")
        require(re.fullmatch(r"[a-f0-9]{32}", inputs["execution_id"]) is not None, "missing unique assisted execution identity")
        forbidden = ("ground_truth", "calibration", "sensor_timestamps", "raw_images", "gps_or_oxts",
                     "manual_region_coordinates", "external_trajectory", "external_ocr_video")
        require(all(inputs["input_policy"].get(key) is False for key in forbidden), "assisted external/GT input policy is unproven")
        require(cfg.get("max_frames") is None and cfg.get("frame_max_width") is None
                and cfg.get("vpr_query_mode") == "warp" and cfg.get("ocr_engine") == "rapidocr"
                and cfg.get("vpr_source") in {"mapillary", "kartaview", "panoramax"}
                and cfg.get("poi_radius_m") == 1000 and cfg.get("poi_ref_cap") == 500 and cfg.get("camera_height_m") == 1.5,
                "assisted fixed component/input profile changed")
        require(not any(cfg.get(key) for key in ("ground_truth_waypoints", "ground_truth_streets", "openvo_trajectory_path",
                "vggt_long_trajectory_path", "use_da3_trajectory", "ocr_video_path", "use_ocr_super_res", "use_ground_flow_scale",
                "use_ipm_scale", "estimated_length_m", "graph_filter", "graph_filter_vpr", "graph_filter_map_path"))
                and (cfg.get("osm_around") is None or cfg.get("osm_around_source") == "video_title"), "assisted contains external/manual inputs")
        policy = inputs["policy"]
        require(policy["version"] == audit["policy_version"] == POLICY_VERSION
                and policy["config_sha256"] == audit["config_sha256"] == CONFIG_SHA256
                and policy["config"] == policy_config(), "assisted policy constants/version changed")
        declared = {str(key).replace("\\", "/"): value for key, value in inputs["source_sha256"].items()}
        required_sources = {"main.py", "monocular_osm/assisted_policy.py", "monocular_osm/assisted_worker.py",
            "monocular_osm/stages/assisted.py", "monocular_osm/stages/prior.py", "monocular_osm/pipeline.py",
            "monocular_osm/kartaview_vpr.py", "monocular_osm/poi_clues.py", "monocular_osm/poi_alignment.py",
            "monocular_osm/poi_localization.py", "monocular_osm/poi_image_track.py", "monocular_osm/assisted_motion.py"}
        required_sources |= {"monocular_osm/" + name for name in ("visual_odometry.py", "frame_extraction.py",
            "plane_speed.py", "graph_filter.py", "provenance.py", "baselines.py", "position.py",
            "stages/poi_runtime.py", "stages/poi_only.py", "stages/trajectory.py", "stages/acquire.py",
            "stages/graph_filter.py", "stages/report.py")}
        policy_source = "monocular_osm/assisted_policy.py"
        if profile == "v2":
            policy_source = "monocular_osm/assisted_policy_v2.py"
            required_sources |= {"monocular_osm/" + name for name in (
                "assisted_profiles.py", "assisted_policy_v2.py", "assisted_image_union.py",
                "poi_image_union.py", "full_sequence.py", "sparse_sequence.py", "geo.py")}
        elif profile == "v3":
            policy_source = "monocular_osm/assisted_policy_v3.py"
            from monocular_osm.stages.assisted import V3_REQUIRED_SOURCES
            required_sources |= V3_REQUIRED_SOURCES | {"monocular_osm/public_reference_cache.py"}
        require(required_sources <= set(declared), "assisted source snapshot omits a required component")
        require(policy["source_sha256"] == declared[policy_source], "policy source is not bound to the parent snapshot")
        files = parent_files(cfg, inputs, audit)
        sources(files, files.root, inputs["source_sha256"])
        physical = load_json(files.root / "result.json")
        require((physical.get("provenance") or {}).get("assisted_inputs") == inputs
                and (physical.get("provenance") or {}).get("assisted_audit") == audit
                and (physical.get("provenance") or {}).get("config") == cfg, "summary is stale relative to the actual parent result")
        frozen, parent_ledger = files.freeze(files.root / "frozen_prediction.json")
        require(frozen["inputs"] == inputs and frozen.get("prediction_available") is True, "parent freeze lacks its selected prediction/inputs")
        for name in ("result.json", "assisted_inputs.json", "assisted_audit.json"):
            require(files.root / name in parent_ledger, "parent freeze omits its result/input/audit")
        runtime_evidence(files, cfg, inputs, audit, parent_ledger)
        records, trace, trace_inputs = audit["reached_stages"], audit["decision_trace"], audit["decision_inputs"]
        require(0 < len(records) == len(trace) == len(trace_inputs) <= 6, "requested stage/decision transcript is incomplete")
        evidence, children, ledgers = {}, {}, {}
        next_kind, terminal = "baseline", None
        for index, record in enumerate(records):
            kind = record["kind"]
            require(kind == next_kind and kind not in children, "policy stages were skipped, repeated or reordered")
            before = deepcopy(evidence)
            for artifact in record.get("artifacts") or []:
                files.record(artifact, parent_ledger)
            child, ledger = verify_child(files, record, inputs, before)
            children[kind], ledgers[kind] = child, ledger
            new_comparison = is_metadata_union(child.get("comparison"))
            require(not new_comparison or profile == "v3" and kind == "expanded", "new full comparison reached an incompatible stage/profile")
            expanded_v3 = profile == "v3" and kind == "expanded" and child.get("status") in {"complete", "partial"}
            if expanded_v3:
                expansion(files, child, ledger, inputs, before, baseline_child=children.get("baseline"))
            if child.get("comparison") is not None and not new_comparison and not expanded_v3:
                comparison(files, child["comparison"], ledger, window,
                           require_all=kind == "expanded" and child.get("status") == "complete")
            if child.get("status") == "complete":
                if kind in {"baseline", "expanded", "poi_joint", "poi_image_track", "poi_image_union"}:
                    runtime = child["cpu_runtime"]
                    require(runtime.get("device") == "cpu" and runtime.get("intra_op_threads") == expected_cpu_threads(cfg)
                            and type(runtime.get("inter_op_threads")) is int and runtime["inter_op_threads"] > 0
                            and isinstance(runtime.get("torch_version"), str) and bool(runtime["torch_version"])
                            and child.get("model_downloads_disabled") is True, "visual child lacks actual bounded CPU/model runtime proof")
                    result["runtimes"].append(digest({key: runtime[key] for key in
                        ("device", "intra_op_threads", "inter_op_threads", "torch_version")}))
                    from monocular_osm.kartaview_vpr import HUB_MEGALOC
                    backbone, weights = child["actual_backbone"], child["actual_model_state"]
                    require(backbone == {"name": "megaloc", "hub": HUB_MEGALOC, "device": "cpu"}
                            and sha_string(weights.get("sha256")), "visual child lacks its actual registered model-state identity")
                    if kind in {"baseline", "expanded"} and not child.get("comparison", {}).get("unavailable_reason"):
                        model = (child["comparison"]["appearance_identity"]["model"] if new_comparison else child["comparison"]["model"])
                        require(model == {**backbone, "weights_sha256": weights["sha256"]},
                                "decoded comparison belongs to different actual model weights")
                    result["weights"].append(digest({**backbone, "weights_sha256": weights["sha256"]}))
                if kind == "cues":
                    cues(files, child, ledger, inputs, reader_issues)
                    if child["cues"]["ocr"]["status"] == "complete":
                        result["ocr_models"].append(ocr_identity(child["ocr_provenance"]))
                elif kind == "expanded":
                    if profile != "v3":
                        expansion(files, child, ledger, inputs, before)
                elif kind == "poi_joint":
                    issues = poi_issues({"provenance": {"config": child["config"], "poi_inputs": child["poi_inputs"],
                        "poi_localization_audit": child["poi_audit"]}}, {"meta": {"method": "poi_only"}}, city, segment)
                    require(not issues, "; ".join(issues))
                    require(child["poi_audit"]["clues"] == children["cues"]["clues"], "joint POI child changed native clues/branches")
                    poi_artifacts(files, child, ledger, children["cues"])
                elif kind == "poi_image_track":
                    image_child(files, child, ledger, inputs, children["cues"])
                elif kind == "poi_image_union":
                    image_union_child(files, child, ledger, inputs, children["cues"])
                elif kind == "metric_pf":
                    require(child["metric_preflight"] == children["cues"]["metric_preflight"], "PF child replaced its native preflight")
                    measured = metric_package(files, child["metric_preflight"], ledgers["cues"], inputs, children["cues"]["auto_quality"])
                    require(measured["eligible"] is True and child.get("metric_only") is True
                            and child.get("uses_user_spatial_prior") is False and child.get("position_observation_count") == 0,
                            "PF child lacks independently eligible native metric odometry")
                    require(child["config"]["frame_stride"] == measured["native_quality"]["stride"], "PF grid differs from native metric preflight")
                    native_times = load_npz(files.path(Path(child["metric_preflight"]["artifacts"]["freeze"]["path"]).parent
                                                      / "native_metric_inputs.npz"))["timestamps"].tolist()
                    route_times = child["route_timestamps"]
                    require(route_times[0] == native_times[0] and route_times[-1] == native_times[-1]
                            and set(route_times) <= set(native_times), "PF route times differ from the measured native video grid")
                    issues = filter_issues({"provenance": {"config": child["config"], "graph_filter_inputs": child["filter_inputs"]}},
                                           {"meta": {"method": "graph_filter_only"}}, city)
                    require(not issues, "; ".join(issues))
                    graph = child["filter_inputs"]["map"]
                    require(file_sha(graph["cache_path"]) == graph["sha256"], "PF whole-city map database changed")
            derived = summary(kind, child, profile="v3") if profile == "v3" else summary(kind, child)
            require(record["summary"] == derived, "reached stage scalars are not derived from its frozen candidate")
            evidence[kind] = derived
            require(trace_inputs[index] == evidence and trace[index] == decide_assisted_policy(evidence), "assisted decision replay differs")
            decision = trace[index]
            if decision["next_action"] in REQUESTS:
                next_kind = REQUESTS[decision["next_action"]]
            else:
                require(index == len(records) - 1, "unrequested future stages were executed")
                terminal = decision
            if kind == "baseline" and not child.get("comparison", {}).get("unavailable_reason"):
                comp = child["comparison"]
                result["models"].append(digest({"model": comp["model"], "preprocessing": comp["query"]["preprocessing"]}))
        require(terminal is not None and audit["evidence"] == evidence, "assisted transcript did not reach a terminal decision")
        selected = terminal["selected_candidate"]
        require(selected == audit["selected_candidate"] and selected in children, "selected candidate disagrees with policy replay")
        child = children[selected]
        require(valid_route(child) is not None and child.get("worker_returncode") == 0, "selected assisted child is partial/failed")
        if selected == "baseline":
            require(not (child.get("comparison") or {}).get("unavailable_reason")
                    and bool((child.get("comparison") or {}).get("audit_artifacts")),
                    "selected baseline lacks actual decoded-reference evidence")
        route_sha = digest(child["route_latlon"])
        require(route_sha == audit["selected_route_sha256"]
                and audit["selected_child_artifact_sha256"] == child["candidate_artifact"]["sha256"], "selected child/route identity is stale")
        answer = physical["answer"]
        require(answer.get("mode") == "assisted" and answer.get("method") == "assisted_policy"
                and answer.get("source") == "result.assisted_position" and answer.get("experimental") is True
                and answer.get("confidence_calibrated") is False and digest(answer["route_latlon"]) == route_sha
                and provenance["assisted_selected_answer"]["route_sha256"] == route_sha, "primary Answer differs from its selected child")
        result["policy"] = digest(policy)
        from scripts.benchmark import _effective_config
        result["source"] = digest(inputs["source_sha256"])
        result["configuration"] = digest(_effective_config({"config": cfg}))
        repeat = {"video": inputs["video"]["sha256"], "source": digest(inputs["source_sha256"])}
        repeat["configuration"] = digest(_effective_config({"config": cfg}, within_clip=True))
        if "cues" in children and children["cues"].get("status") == "complete":
            cue = children["cues"]
            native = load_npz(files.record(cue["auto_quality"]["vo_artifact"]))
            repeat["vo"] = digest({key: _array_sha(native[key]) for key in ("xz", "valid", "timestamps", "K")})
            if cue["cues"]["ocr"]["status"] == "complete":
                repeat.update(ocr=cue["ocr_provenance"]["detections_sha256"], inventory=cue["inventory"]["snapshot_sha256"])
        for kind, name in (("poi_joint", "poi_inputs"), ("metric_pf", "filter_inputs")):
            if children.get(kind, {}).get("status") == "complete":
                repeat[kind + "_map"] = children[kind][name]["map"]["sha256"]
        result["repeat_inputs"] = repeat
    except Exception as error:
        result["issues"].append(f"assisted frozen-input audit: {type(error).__name__}: {error}")
    return result
