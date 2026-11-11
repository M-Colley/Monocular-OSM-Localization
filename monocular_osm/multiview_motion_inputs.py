"""Bounded two-phase native-window packet protocol; no geometry or NN."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STUDY_REF = {"path": "E:/monocular-osm-multiview-motion-proof/run1/study_contract.json",
             "sha256": "5365fff80cbe5bc2eddcba2903832969ba1abf1b3df40b20c144cae281fd9e0d"}
READ_LEDGER_REF = {"path": "E:/monocular-osm-multiview-motion-proof/run1/native_read_ledger.json",
                   "sha256": "d70a15d5c1a2270768635799eda2a8106d2493cff9d48abbc004c30cc0712706"}
SOURCE_PATHS = ("monocular_osm/multiview_motion_inputs.py", "monocular_osm/multiview_motion.py",
                "scripts/prepare_multiview_motion_inputs.py", "scripts/multiview_motion_probe.py")
MAX_JSON = 16 << 20
TOTAL_JSON = 256 << 20
MAX_NPZ = 256 << 20
MAX_PREPARED = 8 << 30
WINDOW_MEMBERS = ("frames_bgr", "video_ordinals", "timestamps", "K", "native_read_rows")
PREPARED_SCHEMA = "frozen-native-multiview-inputs-v1"
PREPARED_STATUS = "PREPARED_NATIVE_PIXEL_CLOCK_WINDOWS_NO_GEOMETRY"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest_ok(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def safe(path):
    path = Path(path)
    require(path.is_absolute(), "absolute paths required")
    for part in (path, *path.parents):
        require(not part.is_symlink() and not getattr(part, "is_junction", lambda: False)(),
                "symlink/junction paths forbidden")
    return path.resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    path = safe(path)
    return {"path": str(path), "sha256": sha(path)}


def same_ref(left, right):
    return (isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right) == {"path", "sha256"}
            and digest_ok(left["sha256"]) and left["sha256"] == right["sha256"]
            and os.path.normcase(str(safe(left["path"]))) == os.path.normcase(str(safe(right["path"]))))


def opaque_path(path):
    """Canonicalize a historical ref without any stat/open/resolve of its asset."""
    require(isinstance(path, str) and Path(path).is_absolute(), "absolute historical ref required")
    return os.path.normcase(os.path.abspath(path))


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_json(path, value):
    with Path(path).open("xb") as f:
        f.write(json_bytes(value))
        f.flush()
        os.fsync(f.fileno())


def strict_json(body):
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, "duplicate JSON key")
            d[k] = v
        return d
    def bad(_):
        raise ValueError("nonfinite JSON constant")
    return json.loads(body.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=bad)


class Ledger:
    def __init__(self):
        self.protected = {}
        self.json_bytes_read = 0

    def add(self, role):
        require(isinstance(role, dict) and set(role) == {"path", "sha256"}
                and digest_ok(role["sha256"]), "exact path/SHA role required")
        path = safe(role["path"])
        require(path.is_file() and sha(path) == role["sha256"], "protected bytes differ")
        key = str(path)
        require(self.protected.get(key, role["sha256"]) == role["sha256"], "conflicting role SHA")
        self.protected[key] = role["sha256"]
        return path

    def json(self, role):
        path = safe(role["path"])
        n = path.stat().st_size
        require(n <= MAX_JSON and self.json_bytes_read + n <= TOTAL_JSON, "metadata byte budget exceeded")
        self.json_bytes_read += n
        path = self.add(role)
        body = path.read_bytes()
        require(len(body) == n and hashlib.sha256(body).hexdigest() == role["sha256"], "JSON changed during read")
        return strict_json(body)

    def verify(self):
        for path, expected in self.protected.items():
            require(safe(path).is_file() and sha(path) == expected, "final protected bytes differ")


def typed(array):
    import numpy as np
    require(isinstance(array, np.ndarray) and array.flags.c_contiguous and not array.dtype.hasobject,
            "plain contiguous array required")
    h = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    if array.size:
        h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def typed_header(array):
    return {"shape": list(array.shape), "dtype": array.dtype.str, "sha256": typed(array)}


def windows(n, first, stride):
    require(type(n) is int and 38 <= n <= 20000 and type(first) is int and first >= 0
            and type(stride) is int and stride > 0 and (n-1)*stride+1 <= 80000, "native read bounds")
    result = []
    for j in range(32):
        c = 3 + ((n-7)*j+15)//31
        rows = list(range(c-3, c+4))
        result.append({"window_id": j, "center_row": c, "native_read_rows": rows,
                       "video_ordinals": [first+r*stride for r in rows]})
    return result


def load_config(path, expected, phase):
    require(phase in ("prepare", "run"), "unknown phase")
    a = Ledger()
    config = a.json({"path": str(safe(path)), "sha256": expected})
    keys = {"schema", "source_hashes", "study_registration_ref", "native_read_ledger_ref", "output_dir"}
    if phase == "run":
        keys.add("prepared_input_acceptance_ref")
    require(set(config) == keys and config["schema"] == ("native-multiview-input-prepare-v1" if phase == "prepare"
            else "native-multiview-motion-probe-run-v1"), "exact phase configuration required")
    sources = config["source_hashes"]
    require(isinstance(sources, dict) and set(sources) == set(SOURCE_PATHS), "exact same-source closure required")
    for name in SOURCE_PATHS:
        a.add({"path": str(ROOT/name), "sha256": sources[name]})
    require(same_ref(config["study_registration_ref"], STUDY_REF) and same_ref(config["native_read_ledger_ref"], READ_LEDGER_REF),
            "exact prospective study and read ledger required")
    study = a.json(config["study_registration_ref"])
    read = a.json(config["native_read_ledger_ref"])
    validate_metadata(study, read)
    require(sources["monocular_osm/multiview_motion.py"] == study["frontend_source_to_promote_ref"]["sha256"],
            "frontend differs from exact prospectively declared source")
    safe(config["output_dir"])
    a.study, a.read = study, read
    a.verify()
    return config, a


def validate_metadata(study, read):
    require(study.get("schema") == "root-prospective-native-multiview-window-study-v1"
            and same_ref(study.get("native_read_ledger_ref"), READ_LEDGER_REF), "study/ledger binding differs")
    require(study.get("ground_truth_or_map_or_OCR_used") is False
            and study.get("new_neural_forwards_or_downloads") is False
            and study.get("policy_admission") is False and study.get("selected_answer") is None,
            "input-only study required")
    require(read.get("schema") == "root-fixed32-native-read-window-lineage-v1"
            and read.get("new_pixels_decoded") is False and read.get("ground_truth_used") is False,
            "metadata-only original read ledger required")
    shape = read.get("expected_native_shape")
    require(isinstance(shape, list) and len(shape) == 3 and all(type(x) is int and x > 0 for x in shape)
            and shape[2] == 3 and shape[0]*shape[1] <= 8500000, "full native shape exceeds bound")
    n, first, stride = (read[k] for k in ("native_dense_count", "first_native_video_ordinal", "original_video_ordinal_stride"))
    expected = windows(n, first, stride)
    require(read.get("last_native_video_ordinal") == first+(n-1)*stride, "native ordinal interval differs")
    require(len(read.get("windows", [])) == 32, "all32 windows required")
    for actual, wanted in zip(read["windows"], expected):
        require(set(actual) == set(wanted) | {"signed_decoder_times_s"}
                and type(actual["window_id"]) is int and type(actual["center_row"]) is int
                and all(type(x) is int for k in ("native_read_rows", "video_ordinals") for x in actual[k])
                and all(actual[k] == v for k, v in wanted.items()), "fixed original read cohort differs")
        times = actual["signed_decoder_times_s"]
        require(len(times) == 7 and all(type(t) is float and math.isfinite(t) and t > 0 for t in times)
                and all(x < y for x, y in zip(times, times[1:])), "positive increasing signed clock required")
    for name in ("full_signed_native_clock_typed_sha256", "full_signed_native_clock_raw_F64_sha256",
                 "full_signed_native_frames_sha256", "native70_K_typed_sha256"):
        require(digest_ok(read.get(name)), "missing full native digest")


def zip_preflight(path, names=None):
    require(Path(path).stat().st_size <= MAX_NPZ, "NPZ storage cap exceeded")
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        require(0 < len(entries) <= 64 and len({e.filename for e in entries}) == len(entries), "NPZ duplicate/member cap")
        require(all(not e.is_dir() and e.filename.endswith(".npy") and "/" not in e.filename
                    and "\\" not in e.filename and not e.flag_bits & 1 for e in entries), "unsafe NPZ member")
        require(all(0 <= e.file_size <= MAX_NPZ for e in entries)
                and sum(e.file_size for e in entries) <= MAX_NPZ, "NPZ uncompressed cap exceeded")
        if names is not None:
            require({e.filename for e in entries} == {k+".npy" for k in names}, "exact NPZ members required")
        import numpy as np
        headers = {}
        # All ZipInfo bounds above precede the first NPY header read.
        for e in entries:
            with z.open(e) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "unsupported NPY version")
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream)
                require(not fortran and not dtype.hasobject and dtype.kind in "biufU"
                        and all(type(x) is int and x >= 0 for x in shape), "unsafe NPY header")
                require(math.prod(shape)*dtype.itemsize == e.file_size-stream.tell(), "NPY header/body size differs")
                headers[e.filename[:-4]] = (shape, dtype)
        return headers


def load_named_npz(path, names):
    import numpy as np
    headers = zip_preflight(path)
    require(set(names) <= set(headers), "required original NPZ members missing")
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], copy=True, order="C") for k in names}


def read_prepared_window(record, ledger, check=lambda: None):
    import numpy as np
    check()
    path = ledger.add(record["array_ref"])
    headers = zip_preflight(path, WINDOW_MEMBERS)
    shape = tuple(record["typed_headers"]["frames_bgr"]["shape"])
    require(len(shape) == 4 and shape[0] == 7 and shape[3] == 3 and shape[1]*shape[2] <= 8500000,
            "native window shape differs")
    specs = {"frames_bgr": (shape, np.dtype("uint8")), "video_ordinals": ((7,), np.dtype("int64")),
             "timestamps": ((7,), np.dtype("float64")), "K": ((3, 3), np.dtype("float64")),
             "native_read_rows": ((7,), np.dtype("int64"))}
    require(headers == specs, "exact typed window headers required")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.ascontiguousarray(z[k]) for k in WINDOW_MEMBERS}
    require({k: typed_header(v) for k, v in arrays.items()} == record["typed_headers"], "window typed bytes differ")
    require([typed(frame) for frame in arrays["frames_bgr"]] == record["frame_typed_sha256"], "per-frame pixels differ")
    require(np.array_equal(arrays["video_ordinals"], record["video_ordinals"])
            and np.array_equal(arrays["native_read_rows"], record["native_read_rows"])
            and np.array_equal(arrays["timestamps"], record["signed_decoder_times_s"]), "window row/clock lineage differs")
    require(np.isfinite(arrays["K"]).all() and np.isfinite(arrays["timestamps"]).all()
            and np.all(arrays["timestamps"] > 0) and np.all(np.diff(arrays["timestamps"]) > 0),
            "finite positive original window timing/K required")
    if hasattr(ledger, "read"):
        require(list(shape[1:]) == ledger.read["expected_native_shape"]
                and typed(arrays["K"]) == ledger.read["native70_K_typed_sha256"], "original native shape/K differs")
    check()
    require(sha(path) == record["array_ref"]["sha256"], "window changed after loading")
    return arrays


def output_membership(folder, expected):
    actual = list(Path(folder).iterdir())
    require(all(p.is_file() and not p.is_symlink() for p in actual)
            and {p.name for p in actual} == set(expected), "exclusive output membership differs")


def admit_prepared(config, ledger):
    """Authenticate the closed whole cohort before any window NPZ member read."""
    qa = ledger.json(config["prepared_input_acceptance_ref"])
    require(qa.get("schema") == "root-closed-native-multiview-prepared-inputs-v1"
            and qa.get("status") == "ACCEPTED_NATIVE_PIXEL_CLOCK_WINDOWS_NO_GEOMETRY"
            and same_ref(qa.get("study_ref"), STUDY_REF) and same_ref(qa.get("native_read_ledger_ref"), READ_LEDGER_REF)
            and type(qa.get("window_count")) is int and qa["window_count"] == 32,
            "closed Root full32 input acceptance required")
    for key in ("all_observed_worker_births_closed", "all_protected_input_hashes_unchanged",
                "full_dense_native_pixels_clock_verified", "no_geometry_computed"):
        require(qa.get(key) is True, "Root input proof/closure incomplete")
    ledger.json(qa["producer_registration_ref"])
    result = ledger.json(qa["result_ref"])
    require(result.get("status") == "complete" and type(result.get("returncode")) is int
            and result["returncode"] == 0 and result.get("reason") is None, "prepare actor not closed rc0")
    closure = ledger.json(qa["external_closure_ref"])
    require(closure.get("schema") == "root-native-multiview-input-external-closure-v1"
            and closure.get("prepared_manifest_ref") == qa["prepared_manifest_ref"]
            and closure.get("registration_ref") == qa["producer_registration_ref"]
            and closure.get("result_ref") == qa["result_ref"]
            and closure.get("worker_births_closed") is True
            and closure.get("protected_inputs_unchanged") is True, "external prepare closure differs")
    manifest = ledger.json(qa["prepared_manifest_ref"])
    require(manifest.get("schema") == PREPARED_SCHEMA and manifest.get("status") == PREPARED_STATUS
            and manifest.get("source_hashes") == config["source_hashes"]
            and same_ref(manifest.get("study_registration_ref"), STUDY_REF)
            and same_ref(manifest.get("native_read_ledger_ref"), READ_LEDGER_REF)
            and manifest.get("full_dense_native_pixels_clock_verified") is True
            and manifest.get("no_geometry_computed") is True, "same-source full prepared master required")
    require(manifest.get("geometry_computed") is False and manifest.get("neural_forward_run") is False
            and manifest.get("ground_truth_used") is False and manifest.get("policy_admission") is False,
            "input-only master claims differ")
    read = ledger.read
    proof = manifest["full_native_proof"]
    expected_proof = {"dense_count": read["native_dense_count"],
        "raw_ordinal_read_count": read["last_native_video_ordinal"]-read["first_native_video_ordinal"]+1,
        "frames_sha256": read["full_signed_native_frames_sha256"],
        "clock_typed_sha256": read["full_signed_native_clock_typed_sha256"],
        "clock_raw_F64_sha256": read["full_signed_native_clock_raw_F64_sha256"],
        "native70_K_typed_sha256": read["native70_K_typed_sha256"], "native_shape": read["expected_native_shape"]}
    require(all(proof.get(k) == v for k, v in expected_proof.items())
            and proof.get("positive_CAP_PROP_POS_MSEC_exact_cached_F64") is True
            and proof.get("independent_container_PTS_verified") is False
            and proof.get("K_is_measured_calibration") is False, "whole original pixel/clock/K proof differs")
    records = manifest.get("windows")
    require(isinstance(records, list) and len(records) == 32, "prepared all32 required")
    folder = safe(qa["prepared_manifest_ref"]["path"]).parent
    names = ["manifest.json"]
    total = 0
    for record, original in zip(records, ledger.read["windows"]):
        require(all(record.get(k) == v for k, v in original.items()), "prepared original window lineage differs")
        path = ledger.add(record["array_ref"])
        name = f"window_{record['window_id']:02d}.npz"
        require(path == folder/name and path.stat().st_size == record["bytes"], "window physical path/size differs")
        require(set(record["typed_headers"]) == set(WINDOW_MEMBERS)
                and len(record["frame_typed_sha256"]) == 7, "prepared typed/frame inventory differs")
        total += record["bytes"]
        names.append(name)
    require(total == manifest.get("total_npz_bytes") and total <= MAX_PREPARED, "total prepared bound differs")
    protected = manifest["protected_hashes"]
    required_roles = [config["study_registration_ref"], config["native_read_ledger_ref"],
        manifest["prepare_config_ref"], {"path": read["video_binding"]["path"], "sha256": read["video_binding"]["sha256"]},
        *read["basis_refs"].values(), *[r["array_ref"] for r in records],
        *[{"path": str(ROOT/name), "sha256": expected} for name, expected in config["source_hashes"].items()]]
    historical = {opaque_path(path): expected for path, expected in protected.items()}
    require(len(historical) == len(protected) and all(digest_ok(v) for v in historical.values()),
            "ambiguous historical protected map")
    require(set(historical) == {opaque_path(r["path"]) for r in required_roles},
            "exact prepare-time protected role inventory required")
    for role in required_roles:
        require(historical.get(opaque_path(role["path"])) == role["sha256"], "prepared historical binding differs")
    # Original raw assets and the prepare config are opaque Root-closed history.
    # Only the numerical phase's genuinely consumed sources/controls/packets are
    # freshly hashed. Do not reopen video, caches or original basis assets here.
    for role in [config["study_registration_ref"], config["native_read_ledger_ref"],
                 *[{"path": str(ROOT/name), "sha256": expected} for name, expected in config["source_hashes"].items()]]:
        ledger.add(role)
    ledger.historical_protected_refs = dict(protected)
    output_membership(folder, names)
    ledger.verify()
    return manifest
