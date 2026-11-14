"""Immutable graph-artifact speed context, not Root admission or a speed model.

The finite worker authenticates Root/closed packet bytes before this pure
constructor. All catalog rows are consumed once. Runtime lookup has no I/O
or map-wide hashing; numeric context is never legal/observed vehicle speed.
"""
from dataclasses import replace
import hashlib
import json
from types import MappingProxyType
import numpy as np

from .graph_filter import RoadNetwork
from . import road_segment_semantics as sem

MAX_RAW = 500_000
MAX_SEGMENTS = 500_000
MAX_EDGES = 100_000
MAX_MEMBERS = 1_000_000
MAX_BASE_BYTES = 110_400_032
MAX_PARTICLES = 50_000
PARSER_PROFILE_SHA256 = "c47606d49caa4ef8bc594a82b97fbc7d3fbbc559d3e134331e814750c97b0da2"
STATUS = MappingProxyType({"numeric_base_context": 1, "missing_base": 2,
    "unsupported_type": 3, "unsupported_base": 4, "unsupported_override": 5,
    "nonpositive_or_nonfinite": 6, "parallel_mapping": 7, "tiny_mapping": 8,
    "synthetic_mapping": 9, "raw_geometry_mapping": 10})
PROFILE = MappingProxyType({"method": "immutable-base-road-speed-context-v1",
    "parser_profile_sha256": PARSER_PROFILE_SHA256, "raw_edges": MAX_RAW,
    "segments": MAX_SEGMENTS, "edges": MAX_EDGES, "memberships": MAX_MEMBERS,
    "catalog_bytes": 192 << 20, "record_bytes": 8192,
    "base_typed_bytes": MAX_BASE_BYTES, "max_particles": MAX_PARTICLES,
    "storage": "OWN immutable bytes-backed arrays; no writable owners",
    "lookup": "original start segment; searchsorted right then edge clip; constant identity guards",
    "binding": "all18 BASE before explicit genuine normalized turn clone; distinct runtime proof",
    "scalar_types": "Python str/int/float and fixed NumPy integer/floating tags; never eval/import",
    "original_OSM_or_way_direction_or_legal_applicability_verified": False,
    "Root_authority_admitted_by_this_pure_module": False,
    "motion_model_or_policy_implemented": False})


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


PROFILE_SHA256 = hashlib.sha256(_json(dict(PROFILE))).hexdigest()


def _need(value, message):
    if not value:
        raise ValueError(message)


def _freeze(array):
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _base_arrays(net):
    _need(isinstance(net, RoadNetwork), "genuine RoadNetwork required")
    _need(all(isinstance(getattr(net, name), np.ndarray) for name in
              ("edge_len", "seg_gstart", "succ_table")), "base count arrays required")
    e, s = len(net.edge_len), len(net.seg_gstart)
    d = net.succ_table.shape[1] if net.succ_table.ndim == 2 else 0
    _need(0 < e <= MAX_EDGES and 0 < s <= MAX_SEGMENTS and 0 < d <= 32,
          "complete base network capacity exceeded")
    _need(isinstance(net.edge_keys, (list, tuple)) and len(net.edge_keys) == e,
          "base edge-key count must precede conversion")
    keys = np.asarray([sem._key(k) for k in net.edge_keys], np.int64).reshape(-1, 3)
    arrays = {name: keys if name == "edge_keys" else getattr(net, name) for name in sem.NET_ARRAYS}
    shapes = {"origin": (2,), "xy_max": (2,), "edge_keys": (e, 3),
              "seg_xy0": (s, 2), "seg_dir": (s, 2), "edge_mid": (e, 2),
              "seg_gstart": (s,), "seg_heading": (s,)}
    shapes.update({k: (e, d) for k in ("succ_table", "succ_logprior", "succ_head")})
    for name, array in arrays.items():
        integer = name in {"edge_keys", "seg_lo", "seg_hi", "reverse_of", "succ_table"}
        _need(isinstance(array, np.ndarray) and array.shape == shapes.get(name, (e,))
              and array.dtype == np.dtype("int64" if integer else "float64")
              and array.flags.c_contiguous, "base shape/dtype/layout: " + name)
        if name != "succ_logprior":
            _need(np.isfinite(array).all(), "nonfinite base array")
    _need(sum(a.nbytes for a in arrays.values()) <= MAX_BASE_BYTES, "base typed byte cap exceeded")
    valid = net.succ_table >= 0
    _need(np.all((net.succ_table >= -1) & (net.succ_table < e))
          and np.count_nonzero(valid) <= 1_000_000, "base successor bounds")
    _need(net.allow_uturn is False and net.normalise_turn_prior is False and net.turn_prior_kappa == 0.
          and np.all(net.succ_logprior[valid] == 0.)
          and np.all(np.isneginf(net.succ_logprior[~valid])), "original unnormalised base required")
    _need(np.all(net.edge_len > 0) and net.seg_lo[0] == 0 and net.seg_hi[-1] == s
          and np.all(net.seg_hi > net.seg_lo)
          and np.array_equal(net.seg_hi[:-1], net.seg_lo[1:]), "base segment partition")
    return arrays


_INT_TYPES = {"builtins.int", *("numpy." + n for n in
    ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "int_", "uint"))}
_FLOAT_TYPES = {"builtins.float", *("numpy." + n for n in
    ("float16", "float32", "float64", "longdouble", "float128"))}


def _parse_record(row):
    attrs = row["attributes"]
    _need(isinstance(attrs, dict) and all(type(k) is str for k in attrs), "typed attribute dictionary required")
    for value in attrs.values():
        _need(isinstance(value, dict) and set(value) == {"type", "value"}
              and type(value["type"]) is str, "typed attribute envelope required")
    if any(k.startswith("maxspeed:") for k in attrs):
        parsed = sem.parse_maxspeed({k: None for k in attrs if k.startswith("maxspeed:")})
    elif "maxspeed" not in attrs:
        parsed = sem.parse_maxspeed({})
    else:
        envelope = attrs["maxspeed"]; name, value = envelope["type"], envelope["value"]
        supported = ((name == "builtins.str" and type(value) is str)
                     or (name in _INT_TYPES and type(value) is int)
                     or (name in _FLOAT_TYPES and type(value) is float))
        if not supported:
            return 0., STATUS["unsupported_type"]
        parsed = sem.parse_maxspeed({"maxspeed": value})
    if parsed["status"] == "base_numeric_context":
        return parsed["mps"], STATUS["numeric_base_context"]
    return 0., STATUS[parsed["reason"]]


def _record_bytes(row):
    # Lower bound on serialized size, before canonical encoding of one row.
    stack = [(row, 0)]; work = 0
    while stack:
        value, depth = stack.pop(); work += 1
        _need(depth <= 32 and work <= 8192, "bounded catalog record traversal required")
        if isinstance(value, dict):
            _need(all(type(k) is str for k in value), "catalog keys must be strings")
            _need(len(value) <= 8192, "catalog container cap exceeded")
            stack.extend((v, depth + 1) for pair in value.items() for v in pair)
        elif isinstance(value, list):
            _need(len(value) <= 8192, "catalog container cap exceeded")
            stack.extend((v, depth + 1) for v in value)
        elif type(value) is str:
            work += len(value)
        elif type(value) is int:
            _need(value.bit_length() <= 8192, "catalog integer cap exceeded")
        else:
            _need(value is None or type(value) in (bool, float), "plain catalog JSON required")
        _need(work <= 8192, "catalog record content cap exceeded")
    return sem.record_bytes(row)


def _packet_arrays(packet, net):
    _need(isinstance(packet, dict) and set(packet) == {"arrays", "metadata"}, "exact semantic packet required")
    arrays, meta = packet["arrays"], packet["metadata"]
    _need(isinstance(arrays, dict) and isinstance(meta, dict)
          and set(arrays) == set(sem.ARRAY_NAMES)
          and all(isinstance(v, np.ndarray) and v.ndim > 0 for v in arrays.values()), "semantic typed arrays required")
    _need(meta.get("profile_sha256") == sem.PROFILE_SHA256
          and hashlib.sha256(_json(meta["profile"])).hexdigest() == sem.PROFILE_SHA256,
          "exact frozen semantic profile required")
    _need(meta.get("net_binding") == sem.net_binding(net), "all18 BASE sidecar binding differs")
    r, e, s = len(arrays["raw_edge_keys"]), net.n_edges, len(net.seg_gstart)
    m = len(arrays["compact_path_raw_edge_index"])
    _need(0 < r <= MAX_RAW and 0 < m <= MAX_MEMBERS, "semantic count cap exceeded")
    shapes = {"raw_edge_keys": (r, 3), "compact_edge_keys": (e, 3), "compact_path_indptr": (e + 1,),
              "compact_path_raw_edge_index": (m,), "segment_raw_edge_index": (s,),
              "segment_semantic_status": (s,), "raw_edge_disposition": (r,)}
    _need(sum(a.nbytes for a in arrays.values()) <= 64 << 20, "semantic byte cap exceeded")
    _need(meta.get("compact_edge_count") == e and meta.get("path_memberships") == m
          and meta.get("semantic_typed_array_bytes") == sum(a.nbytes for a in arrays.values()),
          "semantic declared counts/bytes differ")
    for name, array in arrays.items():
        dtype = np.dtype("uint8" if name in {"segment_semantic_status", "raw_edge_disposition"} else "int64")
        _need(isinstance(array, np.ndarray) and array.shape == shapes[name] and array.dtype == dtype
              and array.flags.c_contiguous, "semantic shape/dtype/layout differs")
        _need(meta["array_headers"][name] == {"shape": list(array.shape), "dtype": array.dtype.str,
              "sha256": sem.typed_sha(array)}, "semantic typed bytes differ")
    _need(np.array_equal(arrays["compact_edge_keys"], np.asarray(net.edge_keys, np.int64)), "compact keys differ")
    ptr, members = arrays["compact_path_indptr"], arrays["compact_path_raw_edge_index"]
    _need(ptr[0] == 0 and ptr[-1] == m and np.all(np.diff(ptr) > 0)
          and np.all((members >= 0) & (members < r)), "semantic path partition differs")
    hits = np.bincount(members, minlength=r)
    _need(np.all(hits <= 1), "raw member reused")
    raw_index, status = arrays["segment_raw_edge_index"], arrays["segment_semantic_status"]
    exact = status == sem.SEG_STATUS["exact_source_context"]
    _need(np.isin(status, list(sem.SEG_STATUS.values())).all()
          and np.all(raw_index[~exact] == -1)
          and np.all((raw_index[exact] >= 0) & (raw_index[exact] < r)), "invalid semantic sentinel/status")
    _need(np.all(hits[raw_index[exact]] == 1), "unselected raw semantic reference")
    flags = arrays["raw_edge_disposition"]
    _need(np.all((flags > 0) & (flags <= 127))
          and np.array_equal((flags & sem.RAW_STATUS["selected_path"]) != 0, hits != 0),
          "raw disposition differs")
    return arrays, _json(meta)


class RoadSpeedContext:
    """Owned immutable lookup data, explicitly bound to one base/runtime pair."""
    __slots__ = ("_base", "_base_arrays", "_binding", "_receipt", "_mps", "_status",
                 "_runtime", "_runtime_arrays", "_keys", "_crs")

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError("immutable road context field")
        object.__setattr__(self, name, value)

    def __init__(self, base, mps, status, binding, receipt):
        self._base = base; self._keys = base.edge_keys; self._crs = base.crs
        self._base_arrays = tuple(getattr(base, k) for k in sem.NET_ARRAYS if k != "edge_keys")
        self._binding = _json(binding)
        self._mps = _freeze(mps); self._status = _freeze(status)
        self._receipt = _json({**receipt, "segment_mps_sha256": sem.typed_sha(self._mps),
                               "segment_status_sha256": sem.typed_sha(self._status)})
        self._runtime = None; self._runtime_arrays = None

    @property
    def segment_mps(self):
        return self._mps

    @property
    def segment_status(self):
        return self._status

    @property
    def receipt(self):
        return json.loads(self._receipt)

    def require_base(self, net):
        _need(net is self._base and net.edge_keys is self._keys and net.crs == self._crs
              and net.allow_uturn is False and net.normalise_turn_prior is False and net.turn_prior_kappa == 0.,
              "context requires its OWN original base")
        self._identities(net, self._base_arrays)

    @staticmethod
    def _identities(net, expected):
        for name, array in zip((n for n in sem.NET_ARRAYS if n != "edge_keys"), expected):
            _need(getattr(net, name) is array and not array.flags.writeable, "immutable runtime geometry was replaced")

    def bind_runtime(self, net):
        self.require_base(self._base)
        _need(self._runtime is None and isinstance(net, RoadNetwork) and net is not self._base,
              "context runtime bind is single-use")
        _need(net.edge_keys is self._keys and net.crs == self._crs and net.allow_uturn is False
              and net.normalise_turn_prior is True and net.turn_prior_kappa == 0., "known v3 transform required")
        for name in sem.NET_ARRAYS:
            if name not in {"edge_keys", "succ_logprior"}:
                _need(getattr(net, name) is getattr(self._base, name), "turn transform changed geometry/topology")
        expected = self._base.with_turn_prior(0., normalise=True)
        _need(np.array_equal(net.succ_logprior, expected.succ_logprior), "turn transform weight differs")
        runtime = replace(net, succ_logprior=_freeze(net.succ_logprior), _transformers={})
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(self, "_runtime_arrays", tuple(getattr(runtime, k) for k in sem.NET_ARRAYS if k != "edge_keys"))
        receipt = self.receipt
        receipt["runtime_binding"] = sem.net_binding(runtime)
        receipt["known_turn_transform_verified"] = True
        object.__setattr__(self, "_receipt", _json(receipt))
        return runtime

    def require_runtime(self, net):
        _need(net is self._runtime and net.edge_keys is self._keys and net.crs == self._crs
              and net.normalise_turn_prior is True and net.turn_prior_kappa == 0. and net.allow_uturn is False,
              "context runtime identity differs")
        self._identities(net, self._runtime_arrays)

    def lookup_previous(self, net, edge, offset):
        self.require_runtime(net)
        edge, offset = np.asarray(edge), np.asarray(offset)
        _need(edge.ndim == 1 and edge.shape == offset.shape and edge.size <= MAX_PARTICLES
              and edge.dtype.kind in "iu" and offset.dtype == np.float64
              and np.all((edge >= 0) & (edge < len(net.edge_len)))
              and np.isfinite(offset).all() and np.all(offset >= 0)
              and np.all(offset <= net.edge_len[edge]), "bounded previous edge/offset required")
        g = net.edge_base[edge] + offset
        segment = np.searchsorted(net.seg_gstart, g, side="right") - 1
        segment = np.clip(segment, net.seg_lo[edge], net.seg_hi[edge] - 1)
        status = self._status[segment]
        return {"segment": segment, "mps": self._mps[segment], "status": status,
                "eligible": status == STATUS["numeric_base_context"]}


def prepare(base_net, packet, catalog_iter, *, catalog_sha256, catalog_bytes):
    """Pure construction AFTER finite Root/role admission by the worker.

    Catalog callbacks may perform bounded reads only during this construction.
    No iterator, record list, source buffers or file handle survives in context.
    """
    _need(sem.PROFILE_SHA256 == PARSER_PROFILE_SHA256, "frozen c476 parser required")
    _need(type(catalog_sha256) is str and len(catalog_sha256) == 64
          and all(c in "0123456789abcdef" for c in catalog_sha256)
          and type(catalog_bytes) is int and 0 < catalog_bytes <= 192 << 20,
          "bounded expected catalog identity required")
    base_arrays = _base_arrays(base_net)
    binding = sem.net_binding(base_net)
    arrays, meta_bytes = _packet_arrays(packet, base_net)
    meta = json.loads(meta_bytes)
    r, s = len(arrays["raw_edge_keys"]), len(base_net.seg_gstart)
    _need(meta.get("catalog_sha256") == catalog_sha256 and meta.get("catalog_bytes") == catalog_bytes
          and meta.get("raw_edge_count") == r and meta.get("segment_count") == s,
          "catalog/packet identity differs")
    frozen = {name: _freeze(array) for name, array in base_arrays.items() if name != "edge_keys"}
    frozen["edge_keys"] = tuple(tuple(int(v) for v in row) for row in base_arrays["edge_keys"])
    own_base = replace(base_net, **frozen, _transformers={})
    _need(sem.net_binding(own_base) == binding, "immutable base copy differs")
    raw_keys = _freeze(arrays["raw_edge_keys"])
    segment_raw = _freeze(arrays["segment_raw_edge_index"])
    segment_mapping_status = _freeze(arrays["segment_semantic_status"])
    raw_mps = np.zeros(r, np.float64); raw_status = np.zeros(r, np.uint8)
    digest = hashlib.sha256(); total = count = 0
    for i, row in enumerate(catalog_iter):
        _need(i < r and isinstance(row, dict)
              and set(row) == {"schema", "index", "key", "endpoint_xy", "attributes"}
              and row["schema"] == "raw-directed-edge-catalog-record-v1"
              and type(row["index"]) is int and row["index"] == i,
              "complete ordered catalog schema required")
        _need(sem._key(row["key"]) == tuple(int(v) for v in raw_keys[i]), "catalog raw key differs")
        xy = row["endpoint_xy"]
        _need(isinstance(xy, list) and len(xy) == 2
              and all(isinstance(p, list) and len(p) == 2 for p in xy)
              and all(type(v) in (int, float) and np.isfinite(v) for p in xy for v in p),
              "finite catalog endpoints required")
        encoded = _record_bytes(row); total += len(encoded)
        _need(total <= catalog_bytes and total <= 192 << 20, "catalog aggregate bytes exceeded")
        digest.update(encoded)
        raw_mps[i], raw_status[i] = _parse_record(row)
        count += 1
    _need(count == r and total == catalog_bytes and digest.hexdigest() == catalog_sha256,
          "ENTIRE expected catalog required")
    # No callback or file access follows the exhausted iterator.
    _need(sem.net_binding(base_net) == binding and _json(packet["metadata"]) == meta_bytes,
          "context inputs changed during catalog construction")
    _packet_arrays(packet, base_net)
    mps = np.zeros(s, np.float64); status = np.zeros(s, np.uint8)
    exact = segment_mapping_status == sem.SEG_STATUS["exact_source_context"]
    selected = segment_raw[exact]
    mps[exact] = raw_mps[selected]; status[exact] = raw_status[selected]
    for source, target in (("ambiguous_parallel_neutral", "parallel_mapping"),
                           ("tiny_replacement_neutral", "tiny_mapping"),
                           ("synthetic_neutral", "synthetic_mapping"),
                           ("raw_geometry_neutral", "raw_geometry_mapping")):
        status[segment_mapping_status == sem.SEG_STATUS[source]] = STATUS[target]
    _need(np.all(status > 0) and np.isfinite(mps).all()
          and np.all(mps[status != STATUS["numeric_base_context"]] == 0), "incomplete segment context")
    receipt = {"schema": "road-speed-context-receipt-v1", "profile_sha256": PROFILE_SHA256,
        "parser_profile_sha256": PARSER_PROFILE_SHA256, "base_binding": binding,
        "catalog_sha256": catalog_sha256, "catalog_bytes": catalog_bytes,
        "raw_count": r, "segment_count": s, "raw_numeric_context_count": int(np.count_nonzero(raw_mps)),
        "segment_status_enum": dict(STATUS),
        "segment_reason_counts": {name: int(np.count_nonzero(status == code)) for name, code in STATUS.items()},
        "OWN_immutable_bytes_backed": True, "ENTIRE_catalog_consumed": True,
        "known_turn_transform_verified": False, "Root_authority_admitted": False,
        "original_OSM_or_way_direction_or_legal_applicability_verified": False,
        "runtime_file_IO_or_map_rehash": False}
    return own_base, RoadSpeedContext(own_base, mps, status, binding, receipt)


prepare_road_context = prepare


def moving_schedule(motion):
    """Freeze genuine coarsener support once; never infer a stop/metric scale."""
    _need(isinstance(motion.step, np.ndarray) and motion.step.ndim == 1,
          "complete genuine Motion required")
    t = len(motion.step)
    _need(0 < t <= 1200 and motion.times.dtype == np.float64 and motion.times.shape == (t + 1,)
          and np.isfinite(motion.times).all(), "bounded full composed clock required")
    dt = np.diff(motion.times)
    _need(np.all(dt > 0) and np.isfinite(dt).all(), "positive composed dt required")
    for name in ("step", "unknown_seconds", "stationary_seconds"):
        value = getattr(motion, name)
        _need(value.dtype == np.float64 and value.shape == (t,) and np.isfinite(value).all()
              and np.all(value >= 0), "finite motion support required")
    counts, ends = motion.counts, motion.native_end_indices
    _need(counts.dtype == np.int64 and counts.shape == (t, 3)
          and np.all((counts >= 0) & (counts <= 100_000))
          and ends.dtype == np.int64 and ends.shape == (t,)
          and np.all((ends >= 1) & (ends <= 100_000)) and np.all(np.diff(np.r_[0, ends]) > 0),
          "bounded complete native support counts required")
    native_count = np.diff(np.r_[0, ends])
    _need(np.array_equal(counts.sum(axis=1), native_count)
          and np.array_equal(counts[:, 0] == 0, motion.step == 0)
          and np.array_equal(counts[:, 1] == 0, motion.unknown_seconds == 0)
          and np.array_equal(counts[:, 2] == 0, motion.stationary_seconds == 0),
          "native count/duration support is inconsistent")
    _need(type(motion.unit_seconds) in (int, float) and not isinstance(motion.unit_seconds, bool)
          and np.isfinite(motion.unit_seconds) and motion.unit_seconds > 0, "nonmetric unit_seconds required")
    eligible = ((counts[:, 0] == native_count) & (counts[:, 0] > 0)
                & (counts[:, 1] == 0) & (counts[:, 2] == 0)
                & (motion.unknown_seconds == 0) & (motion.stationary_seconds == 0)
                & (motion.step > 0))
    parent_step = motion.step + motion.unknown_seconds / motion.unit_seconds
    _need(np.isfinite(parent_step).all(), "finite inherited predict step required")
    return {"eligible": _freeze(eligible), "dt": _freeze(dt), "measured_step": _freeze(motion.step),
            "parent_step": _freeze(parent_step), "counts": _freeze(counts), "native_end_indices": _freeze(ends)}
