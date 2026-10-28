"""Bounded offline OSM way geometry, deliberately not a routing graph.

The public GDAL dataset feature iterator consumes every layer in one pass,
before any spatial filter. Full selected way WKB and every emitted OSM tag
are retained in SQLite, with an RTree of geographic bounds. Intersections,
connectivity, legal access and direction of travel are not inferred.

Driver/API references: https://gdal.org/en/stable/drivers/vector/osm.html
and https://gdal.org/en/stable/api/raster_c_api.html#_CPPv425GDALDatasetGetNextFeature12GDALDatasetHP9OGRLayerHPd16GDALProgressFuncPv
Source metadata framing follows the primary OSMPBF schemas at
https://github.com/openstreetmap/OSM-binary/tree/master/osmpbf (field definitions,
not copied generated parsers). Coordinates are not decoded by that metadata pass.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import threading
import time
import zlib

GIB = 1024**3
PHYSICAL_HIGHWAYS = ("motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link",
    "unclassified", "residential", "living_street", "service", "road", "track")
PROFILE = {"version": "whole-extract-physical-road-geometry-v1", "bbox": None,
    "mask": None, "sql_filter": None, "highways": list(PHYSICAL_HIGHWAYS),
    "access_policy": "physical highway class only; private/no/destination access retained, not legal routing",
    "excluded": "area=yes; highway outside fixed list (including construction/proposed/footway/pedestrian)",
    "geometry": "full source OSM way LineString WKB, WGS84 longitude/latitude; no simplify/split/snap",
    "tags": "all_tags JSON; report_all_tags=yes, no ignored keys; OSM way ID preserved",
    "topology": False, "routed_graph": False, "oneway_enforced": False,
    "reading": "GDALDatasetGetNextFeature all layers; node/way indexes enabled; no spatial filters",
    "assembly_guard": "each selected source-way ID/exact tags/node-reference count agrees; no independent coordinate/order verification",
    "coordinate_authority": "GDAL OSM node-coordinate assembly; ordered source node-reference hash retained separately",
    "max_source_bytes": GIB, "max_features_seen": 50_000_000, "max_ways": 10_000_000,
    "max_vertices": 100_000_000, "max_way_vertices": 20_000, "max_wkb_bytes": 1024**2,
    "max_tags_bytes": 1024**2, "batch_rows": 512, "batch_bytes": 16*1024**2,
    "max_output_bytes": 8*GIB, "max_temp_bytes": 16*GIB, "sqlite_cache_KiB": 32768,
    "gdal_memory_spill_MiB": 32, "max_rss_bytes": 3*GIB,
    "min_physical_bytes": 8*GIB, "min_true_commit_bytes": 8*GIB,
    "min_disk_free_bytes": 16*GIB, "wall_budget_s": 7200,
    "resource_interval_s": 2., "warning_policy": "any GDAL warning/error refuses complete output",
    "query_routes_or_labels_used": False}
_LOCK = threading.RLock()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


PROFILE_SHA256 = digest(PROFILE)


def safe_path(value):
    path = Path(value)
    for candidate in (path, path.resolve()):
        for part in candidate.parts:
            lower = part.lower()
            require(not re.search(r"(?:^|[_.-])(?:gt|ground[_-]?truth|oxts|labels?)(?=[_.-]|$)", lower)
                and not re.search(r"(?:^|[_.-])(?:scores?|scoring)(?=[_.-]|$).*\.jsonl?$", lower),
                "label/score paths are forbidden")
    require(not str(path).startswith(("/vsi", "http:", "https:")), "local paths only")
    return path


def file_sha(path):
    path = safe_path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8*1024**2):
            h.update(block)
    return h.hexdigest()


def physical_road(tags):
    require(isinstance(tags, dict) and all(isinstance(k, str) and isinstance(v, str)
        for k, v in tags.items()), "OSM tags must be a string mapping")
    return tags.get("highway") in PHYSICAL_HIGHWAYS and tags.get("area", "").lower() != "yes"


def _varint(body, offset):
    value = 0
    for shift in range(0, 70, 7):
        require(offset < len(body), "truncated PBF varint")
        byte = body[offset]; offset += 1; value |= (byte & 127) << shift
        if not byte & 128:
            require(value < 2**64, "PBF varint exceeds64bits")
            return value, offset
    raise ValueError("PBF varint exceeds10bytes")


def _fields(body):
    body = memoryview(body); offset = 0
    while offset < len(body):
        key, offset = _varint(body, offset); number, wire = key >> 3, key & 7
        require(number > 0, "invalid PBF field number")
        if wire == 0: value, offset = _varint(body, offset)
        elif wire == 2:
            size, offset = _varint(body, offset)
            require(size <= len(body)-offset, "truncated PBF bytes")
            value = body[offset:offset+size]; offset += size
        elif wire in {1, 5}:
            size = 8 if wire == 1 else 4
            require(size <= len(body)-offset, "truncated PBF fixed field")
            value = body[offset:offset+size]; offset += size
        else: raise ValueError("unsupported PBF wire type")
        yield number, wire, value


def _packed(value, wire):
    if wire == 0: yield value
    else:
        require(wire == 2, "invalid PBF repeated integer encoding")
        offset = 0
        while offset < len(value):
            result, offset = _varint(value, offset)
            yield result


def _pbf_payloads(path, guard):
    """Bounded standard raw/zlib PBF framing; no coordinate interpretation."""
    total = 0; headers = 0
    with Path(path).open("rb") as handle:
        while prefix := handle.read(4):
            guard.check()
            require(len(prefix) == 4, "truncated PBF blob prefix")
            length = struct.unpack(">I", prefix)[0]
            require(0 < length <= 64*1024, "PBF header bound exceeded")
            header = handle.read(length); require(len(header) == length, "truncated PBF header")
            fields = {n: (w, v) for n, w, v in _fields(header)}
            require(1 in fields and 3 in fields and fields[1][0] == 2 and fields[3][0] == 0, "PBF blob type/size missing")
            kind, size = bytes(fields[1][1]), fields[3][1]
            require(0 < size <= 32*1024**2, "PBF compressed blob bound exceeded")
            blob = handle.read(size); require(len(blob) == size, "truncated PBF blob")
            values = {n: (w, v) for n, w, v in _fields(blob)}
            require(not (set(values) & {4, 5, 6, 7}), "unsupported PBF compression")
            if 1 in values:
                require(values[1][0] == 2 and 3 not in values, "ambiguous PBF payload")
                raw = bytes(values[1][1])
            else:
                require(2 in values and 3 in values and values[2][0] == 0 and values[3][0] == 2, "missing PBF zlib payload/size")
                size = values[2][1]; require(0 < size <= 32*1024**2, "PBF raw blob bound exceeded")
                decoder = zlib.decompressobj(); raw = decoder.decompress(values[3][1], size+1)
                require(len(raw) == size and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail, "PBF zlib size/end differs")
            require(len(raw) <= 32*1024**2, "PBF raw blob bound exceeded")
            total += len(raw); headers += 1
            require(total <= 32*GIB and headers <= 1_000_000, "PBF whole-source parse bound exceeded")
            yield kind, raw


def _source_ways_pbf(path, guard):
    """OSMPBF Way metadata only; geometry coordinates remain GDAL-authority."""
    seen_header = False
    for kind, raw in _pbf_payloads(path, guard):
        if kind == b"OSMHeader":
            require(not seen_header, "duplicated/out-of-order PBF header")
            features = [bytes(v).decode("utf-8") for n, w, v in _fields(raw) if n == 4 and w == 2]
            require("OsmSchema-V0.6" in features and set(features) <= {"OsmSchema-V0.6", "DenseNodes"}, "unsupported/history PBF required features")
            seen_header = True
            continue
        require(kind == b"OSMData" and seen_header, "unknown PBF blob type or missing header")
        block = list(_fields(raw)); tables = [v for n, w, v in block if n == 1 and w == 2]
        require(len(tables) == 1, "PBF string table missing/duplicated")
        strings = [bytes(v).decode("utf-8", "strict") for n, w, v in _fields(tables[0]) if n == 1 and w == 2]
        require(strings and strings[0] == "" and len(strings) <= 1_000_000, "PBF string table lacks index0 or exceeds bound")
        for n, wire, group in block:
            if n != 2: continue
            require(wire == 2, "invalid PBF primitive group")
            for n2, wire2, way in _fields(group):
                if n2 != 3: continue
                require(wire2 == 2, "invalid PBF way message")
                identity, keys, vals, refs = None, [], [], []
                for number, wire3, value in _fields(way):
                    if number == 1:
                        require(identity is None and wire3 == 0, "PBF way ID invalid/duplicated")
                        identity = value
                    elif number in {2, 3}:
                        target = keys if number == 2 else vals; target.extend(_packed(value, wire3))
                        require(len(target) <= 50_000, "PBF way tag bound exceeded")
                    elif number == 8: refs.append((wire3, value))
                require(identity is not None and 0 < identity < 2**63 and len(keys) == len(vals), "PBF way ID/tags invalid")
                tags = {}
                for key, value in zip(keys, vals):
                    require(key < len(strings) and value < len(strings), "PBF tag index outside table")
                    require(strings[key] not in tags, "duplicate source OSM tag key")
                    tags[strings[key]] = strings[value]
                if not physical_road(tags):
                    yield None; continue
                count, previous, node_hash = 0, 0, hashlib.sha256()
                for wire3, value in refs:
                    for encoded in _packed(value, wire3):
                        previous += (encoded >> 1) ^ -(encoded & 1)
                        require(0 < previous < 2**63, "PBF source node reference invalid")
                        count += 1; node_hash.update(previous.to_bytes(8, "little"))
                        require(count <= PROFILE["max_way_vertices"], "source way vertex bound exceeded")
                require(count >= 2, "physical source road has fewer than2nodes")
                yield identity, tags, count, node_hash.hexdigest()


def _source_ways_xml(path, guard):
    import xml.etree.ElementTree as ET
    root = None
    for event, element in ET.iterparse(path, events=("start", "end")):
        if root is None: root = element
        if event != "end" or element.tag not in {"node", "way", "relation"}: continue
        guard.check()
        if element.tag == "way":
            tags, nodes = {}, []
            for child in element:
                if child.tag == "tag":
                    require(child.attrib["k"] not in tags, "duplicate source OSM tag key")
                    tags[child.attrib["k"]] = child.attrib["v"]
                elif child.tag == "nd": nodes.append(int(child.attrib["ref"]))
            if physical_road(tags):
                identity = int(element.attrib["id"])
                require(0 < identity < 2**63 and 2 <= len(nodes) <= PROFILE["max_way_vertices"]
                    and all(0 < node < 2**63 for node in nodes), "invalid XML way ID/references")
                h = hashlib.sha256()
                for node in nodes: h.update(node.to_bytes(8, "little"))
                yield identity, tags, len(nodes), h.hexdigest()
            else: yield None
        root.clear()


def source_way_ledger(source, target, guard):
    """Bind source metadata/counts, without claiming independent coordinates."""
    target = safe_path(target); require(not target.exists(), "source ledger exists")
    iterator = _source_ways_pbf(source, guard) if Path(source).suffix.lower() == ".pbf" else _source_ways_xml(source, guard)
    connection = sqlite3.connect(target)
    counts = Counter(); batch = []; batch_bytes = 0
    def flush():
        nonlocal batch_bytes
        if batch:
            guard.check(force=True)
            connection.executemany("INSERT INTO expected VALUES(?,?,?,?)", batch)
            connection.commit(); batch.clear(); batch_bytes = 0
        require(target.stat().st_size <= PROFILE["max_output_bytes"], "source ledger size bound exceeded")
    try:
        connection.executescript("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA cache_size=-32768; CREATE TABLE expected(osm_id INTEGER PRIMARY KEY,tags_json TEXT NOT NULL,n_refs INTEGER NOT NULL,node_refs_sha256 TEXT NOT NULL);")
        for value in iterator:
            counts["source_ways_seen"] += 1
            require(counts["source_ways_seen"] <= PROFILE["max_features_seen"], "source-way scan bound exceeded")
            if value is None: continue
            identity, tags, count, node_hash = value
            counts["selected_ways"] += 1; counts["selected_vertices"] += count
            require(counts["selected_ways"] <= PROFILE["max_ways"] and counts["selected_vertices"] <= PROFILE["max_vertices"], "source-way ledger bound exceeded")
            text = json.dumps(tags, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            require(len(text.encode()) <= PROFILE["max_tags_bytes"], "source way tags budget exceeded")
            size = len(text.encode())+80
            require(size <= PROFILE["batch_bytes"], "source ledger row exceeds batch byte bound")
            if batch and batch_bytes+size > PROFILE["batch_bytes"]: flush()
            batch.append((identity, text, count, node_hash))
            batch_bytes += size
            if len(batch) >= PROFILE["batch_rows"] or batch_bytes == PROFILE["batch_bytes"]: flush()
        flush()
        require(counts["selected_ways"] > 0 and connection.execute("PRAGMA quick_check").fetchone() == ("ok",), "empty/invalid source way ledger")
        guard.check(force=True)
        return dict(counts)
    finally:
        connection.close(); iterator.close()


def checked_way(osm_id, tags, wkb):
    """Validate full way bytes without changing vertex order or geometry."""
    from shapely import from_wkb, get_coordinates
    require(isinstance(osm_id, str) and re.fullmatch(r"[1-9][0-9]*", osm_id)
        and int(osm_id) < 2**63, "positive signed-64 OSM way ID required")
    require(physical_road(tags), "way is outside fixed physical-road predicate")
    require(isinstance(wkb, bytes) and 0 < len(wkb) <= PROFILE["max_wkb_bytes"], "way WKB budget exceeded")
    tag_json = json.dumps(tags, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    require(len(tag_json.encode()) <= PROFILE["max_tags_bytes"], "way tags budget exceeded")
    geom = from_wkb(wkb)
    require(geom.geom_type == "LineString" and not geom.is_empty and not geom.has_z and not geom.has_m,
        "full nonempty2D way LineString required")
    xy = get_coordinates(geom)
    require(2 <= len(xy) <= PROFILE["max_way_vertices"] and all(math.isfinite(float(v))
        for v in xy.flat) and (abs(xy[:, 0]) <= 180).all() and (abs(xy[:, 1]) <= 90).all(),
        "invalid/unbounded WGS84 way vertices")
    require(geom.length > 0 and geom.is_valid, "invalid or zero-length way")
    return (int(osm_id), tags["highway"], tag_json, wkb, len(xy), *map(float, geom.bounds))


def memory_snapshot():
    import psutil
    physical = int(psutil.virtual_memory().available)
    if os.name == "nt":
        class Info(C.Structure):
            _fields_ = [("cb", C.c_ulong)] + [(name, C.c_size_t) for name in (
                "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
                "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
                (name, C.c_ulong) for name in ("HandleCount", "ProcessCount", "ThreadCount")]
        info = Info(); info.cb = C.sizeof(info)
        fn = C.WinDLL("psapi").GetPerformanceInfo
        fn.argtypes, fn.restype = [C.POINTER(Info), C.c_ulong], C.c_int
        require(bool(fn(C.byref(info), C.sizeof(info))), "true system commit query failed")
        free_commit = int(info.CommitLimit-info.CommitTotal)*int(info.PageSize)
    else:
        body = Path("/proc/meminfo").read_text()
        values = {m.group(1): int(m.group(2))*1024 for m in re.finditer(r"^(CommitLimit|Committed_AS):\s+(\d+) kB$", body, re.M)}
        require(len(values) == 2, "true system commit query unavailable")
        free_commit = values["CommitLimit"]-values["Committed_AS"]
    return {"physical_available": physical, "true_commit_available": free_commit,
        "rss": int(psutil.Process().memory_info().rss)}


class ResourceGuard:
    def __init__(self, output, temporary, *, snapshot=memory_snapshot):
        self.output, self.temporary = Path(output).resolve(strict=True), Path(temporary).resolve(strict=True)
        require(self.temporary.is_relative_to(self.output) and self.temporary != self.output,
            "temporary directory must stay inside owned output family")
        self.snapshot, self.started, self.last = snapshot, time.monotonic(), -math.inf
        self.minimum = {}; self.peak_rss = 0; self.samples = 0

    def check(self, *, force=False):
        now = time.monotonic()
        if not force and now-self.last < PROFILE["resource_interval_s"]:
            return
        self.last = now
        require(now-self.started < PROFILE["wall_budget_s"], "road extraction wall budget exceeded")
        sample = self.snapshot(); self.samples += 1
        for name, minimum in (("physical_available", PROFILE["min_physical_bytes"]),
                ("true_commit_available", PROFILE["min_true_commit_bytes"])):
            self.minimum[name] = min(self.minimum.get(name, sample[name]), sample[name])
            require(sample[name] >= minimum, "road extraction memory headroom refused: "+name)
        self.peak_rss = max(self.peak_rss, sample["rss"])
        require(sample["rss"] <= PROFILE["max_rss_bytes"], "road extraction RSS bound exceeded")
        require(shutil.disk_usage(self.output).free >= PROFILE["min_disk_free_bytes"], "road extraction disk headroom refused")
        output_bytes, temporary_bytes = 0, 0
        for path in self.output.rglob("*"):
            resolved = path.resolve()
            require(resolved.is_relative_to(self.output), "output/temp path escaped owned family")
            if path.is_file():
                if resolved.is_relative_to(self.temporary): temporary_bytes += path.stat().st_size
                else: output_bytes += path.stat().st_size
        require(output_bytes <= PROFILE["max_output_bytes"], "road extraction output disk bound exceeded")
        require(temporary_bytes <= PROFILE["max_temp_bytes"], "road extraction temporary disk bound exceeded")

    def record(self):
        return {"minimum": self.minimum, "peak_rss": self.peak_rss, "samples": self.samples,
            "elapsed_s": time.monotonic()-self.started}


class GdalWays:
    """Public C API from the GDAL binary already installed with pyogrio."""
    def __init__(self):
        import pyogrio
        root = Path(pyogrio.__file__).resolve().parent.parent
        candidates = sorted((root/"pyogrio.libs").glob("*gdal*.dll")) if os.name == "nt" else sorted((root/"pyogrio.libs").glob("libgdal*.so*"))
        require(len(candidates) == 1, "one pyogrio-owned GDAL library required; no alternate loader")
        self.library_path = candidates[0].resolve()
        require(self.library_path.is_relative_to(root), "foreign GDAL library")
        self.lib = C.CDLL(str(self.library_path))
        self.identity = {"pyogrio_version": pyogrio.__version__, "gdal_version": list(pyogrio.__gdal_version__),
            "library": str(self.library_path), "library_sha256": file_sha(self.library_path),
            "interface": "public GDAL C dataset iterator; no PyArrow required"}
        ptr, char = C.c_void_p, C.c_char_p
        signatures = {"GDALAllRegister": (None, []), "GDALOpenEx": (ptr, [char, C.c_uint, C.POINTER(char), C.POINTER(char), C.POINTER(char)]),
            "GDALClose": (None, [ptr]), "GDALDatasetGetNextFeature": (ptr, [ptr, C.POINTER(ptr), C.POINTER(C.c_double), ptr, ptr]),
            "GDALDatasetTestCapability": (C.c_int, [ptr, char]), "OGR_L_GetName": (char, [ptr]),
            "OGR_L_GetSpatialRef": (ptr, [ptr]), "OSRGetAuthorityCode": (char, [ptr, char]),
            "OGR_F_GetFieldIndex": (C.c_int, [ptr, char]), "OGR_F_IsFieldSetAndNotNull": (C.c_int, [ptr, C.c_int]),
            "OGR_F_GetFieldAsString": (char, [ptr, C.c_int]), "OGR_F_GetGeometryRef": (ptr, [ptr]),
            "OGR_G_WkbSize": (C.c_int, [ptr]), "OGR_G_ExportToWkb": (C.c_int, [ptr, C.c_int, ptr]),
            "OGR_F_Destroy": (None, [ptr]), "CPLGetConfigOption": (char, [char, char]),
            "CPLSetConfigOption": (None, [char, char]), "CPLPushErrorHandler": (None, [ptr]), "CPLPopErrorHandler": (None, [])}
        for name, (result, args) in signatures.items():
            fn = getattr(self.lib, name); fn.restype, fn.argtypes = result, args
        self.lib.GDALAllRegister()

    def rows(self, path, config, temporary, guard):
        """Yield every eligible whole way. Warnings refuse, never truncate."""
        require(Path(path).suffix.lower() in {".pbf", ".osm"}, "OSM/PBF source required")
        options = {"CPL_TMPDIR": str(Path(temporary).resolve()), "OSM_MAX_TMPFILE_SIZE": "32",
            "OSM_INDEX_POINTS": "YES", "OSM_USE_POINTS_INDEX": "YES", "OSM_INDEX_WAYS": "YES",
            "OSM_USE_WAYS_INDEX": "YES", "OSM_USE_CUSTOM_INDEXING": "YES", "OSM_COMPRESS_NODES": "YES",
            "GDAL_NUM_THREADS": "2", "OGR_INTERLEAVED_READING": "YES"}
        errors, callback_errors = [], []
        ERROR = C.CFUNCTYPE(None, C.c_int, C.c_int, C.c_char_p)
        PROGRESS = C.CFUNCTYPE(C.c_int, C.c_double, C.c_char_p, C.c_void_p)
        @ERROR
        def on_error(level, code, message):
            if level >= 2 and len(errors) < 50:
                errors.append((level, code, (message or b"").decode("utf-8", "replace")[:4096]))
        @PROGRESS
        def progress(percent, message, data):
            try:
                guard.check()
                return not errors
            except Exception as error:
                callback_errors.append(error)
                return 0
        def field(feature, name, *, optional=False):
            index = self.lib.OGR_F_GetFieldIndex(feature, name.encode())
            require(index >= 0, "missing OSM field "+name)
            if not self.lib.OGR_F_IsFieldSetAndNotNull(feature, index):
                require(optional, "missing OSM "+name)
                return None
            value = self.lib.OGR_F_GetFieldAsString(feature, index)
            return value.decode("utf-8", "strict") if value else ""
        counts = Counter(); self.stats = {"layers_seen": counts, "bbox": None, "warnings": errors}
        with _LOCK:
            old = {k: self.lib.CPLGetConfigOption(k.encode(), None) for k in options}
            dataset = None
            self.lib.CPLPushErrorHandler(C.cast(on_error, C.c_void_p))
            try:
                for key, value in options.items(): self.lib.CPLSetConfigOption(key.encode(), value.encode())
                drivers = (C.c_char_p*2)(b"OSM", None)
                opts = (C.c_char_p*4)(("CONFIG_FILE="+str(Path(config).resolve())).encode(), b"TAGS_FORMAT=JSON", b"INTERLEAVED_READING=YES", None)
                dataset = self.lib.GDALOpenEx(str(Path(path).resolve()).encode("utf-8"), 4, drivers, opts, None)
                require(dataset and not errors, "GDAL OSM open refused: "+str(errors[:3]))
                require(self.lib.GDALDatasetTestCapability(dataset, b"RandomLayerRead"), "dataset interleaved feature reading unavailable")
                while True:
                    guard.check()
                    layer, fraction = C.c_void_p(), C.c_double()
                    feature = self.lib.GDALDatasetGetNextFeature(dataset, C.byref(layer), C.byref(fraction), C.cast(progress, C.c_void_p), None)
                    if callback_errors or errors:
                        if feature: self.lib.OGR_F_Destroy(feature)
                        if callback_errors: raise callback_errors[0]
                        require(False, "GDAL geometry warning/error refused: "+str(errors[:3]))
                    if not feature: break
                    try:
                        require(layer.value, "GDAL returned feature without owning layer")
                        name = self.lib.OGR_L_GetName(layer).decode("utf-8", "strict")
                        counts[name] += 1
                        require(sum(counts.values()) <= PROFILE["max_features_seen"], "source feature bound exceeded")
                        if name != "lines": continue
                        srs = self.lib.OGR_L_GetSpatialRef(layer)
                        require(srs and self.lib.OSRGetAuthorityCode(srs, None) == b"4326", "OSM layer must declare EPSG4326")
                        text = field(feature, "all_tags", optional=True)
                        tags = json.loads(text) if text else {}
                        highway = field(feature, "highway", optional=True)
                        require(not highway or tags.get("highway") == highway,
                            "dedicated highway tag missing from complete all_tags")
                        if not physical_road(tags): continue
                        osm_id = field(feature, "osm_id")
                        geometry = self.lib.OGR_F_GetGeometryRef(feature)
                        require(geometry, "selected road has missing way geometry")
                        size = self.lib.OGR_G_WkbSize(geometry)
                        require(0 < size <= PROFILE["max_wkb_bytes"], "GDAL way WKB bound exceeded")
                        buffer = C.create_string_buffer(size)
                        require(self.lib.OGR_G_ExportToWkb(geometry, 1, buffer) == 0, "GDAL way WKB export failed")
                        yield checked_way(osm_id, tags, bytes(buffer.raw))
                    finally:
                        self.lib.OGR_F_Destroy(feature)
                guard.check(force=True)
                require(not errors, "GDAL warnings prevent complete extraction")
            finally:
                if dataset: self.lib.GDALClose(dataset)
                for key, value in old.items(): self.lib.CPLSetConfigOption(key.encode(), value)
                self.lib.CPLPopErrorHandler()
        require(not errors, "GDAL close warnings prevent complete extraction")
        require(file_sha(self.library_path) == self.identity["library_sha256"], "GDAL binary changed")


OSM_CONFIG = """[general]
closed_ways_are_polygons=area=yes
report_all_ways=yes
report_all_tags=yes
tags_format=json
[points]
osm_id=yes
attributes=
other_tags=no
[lines]
osm_id=yes
attributes=highway
ignore=
all_tags=yes
other_tags=no
[multipolygons]
osm_id=yes
attributes=
other_tags=no
[multilinestrings]
osm_id=yes
attributes=
other_tags=no
[other_relations]
osm_id=yes
attributes=
other_tags=no
"""


def build_index(rows, path, guard, *, source_ledger=None):
    """Consume one pass into a new on-disk index, preserving complete WKB."""
    path = safe_path(path)
    require(not path.exists(), "road index already exists")
    temporary = path.parent/"temporary"/"sqlite"
    temporary.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    expected = sqlite3.connect("file:"+str(Path(source_ledger).resolve()).replace("\\", "/")+"?mode=ro", uri=True) if source_ledger else None
    locked, old_temp = False, None
    totals = Counter(); batch, batch_bytes = [], 0
    def flush():
        nonlocal batch_bytes
        if not batch: return
        guard.check(force=True)
        connection.executemany("INSERT INTO ways VALUES(?,?,?,?,?,?,?,?,?)", [r[:5]+(r[5], r[7], r[6], r[8]) for r in batch])
        connection.executemany("INSERT INTO way_bounds VALUES(?,?,?,?,?)", [(r[0], r[5], r[7], r[6], r[8]) for r in batch])
        connection.commit(); batch.clear(); batch_bytes = 0
        require(path.stat().st_size <= PROFILE["max_output_bytes"], "road index output bound exceeded")
    try:
        # SQLite's setting is process-global. This standalone extractor has a
        # single connection/thread; serialize and restore it on every exit.
        _LOCK.acquire(); locked = True
        old_temp = connection.execute("PRAGMA temp_store_directory").fetchone()
        connection.execute("PRAGMA temp_store_directory='"+str(temporary.resolve()).replace("'", "''")+"'")
        actual_temp = connection.execute("PRAGMA temp_store_directory").fetchone()
        require(actual_temp and Path(actual_temp[0]).resolve() == temporary.resolve(), "SQLite temporary placement unavailable")
        connection.executescript("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA temp_store=FILE; PRAGMA cache_size=-32768; CREATE TABLE ways(osm_id INTEGER PRIMARY KEY,highway TEXT NOT NULL,tags_json TEXT NOT NULL,wkb BLOB NOT NULL,n_vertices INTEGER NOT NULL,minx REAL NOT NULL,maxx REAL NOT NULL,miny REAL NOT NULL,maxy REAL NOT NULL); CREATE VIRTUAL TABLE way_bounds USING rtree(osm_id,minx,maxx,miny,maxy);")
        for row in rows:
            if expected:
                original = expected.execute("SELECT tags_json,n_refs FROM expected WHERE osm_id=?", (row[0],)).fetchone()
                require(original == (row[2], row[4]), "GDAL way geometry/tags differ from full source-way reference ledger")
            totals["ways"] += 1; totals["vertices"] += row[4]; totals["wkb_bytes"] += len(row[3]); totals["highway:"+row[1]] += 1
            require(totals["ways"] <= PROFILE["max_ways"] and totals["vertices"] <= PROFILE["max_vertices"], "road inventory bound exceeded")
            size = len(row[3])+len(row[2].encode())
            require(size <= PROFILE["batch_bytes"], "road row exceeds batch byte bound")
            if batch and batch_bytes+size > PROFILE["batch_bytes"]: flush()
            batch.append(row); batch_bytes += size
            if len(batch) >= PROFILE["batch_rows"] or batch_bytes >= PROFILE["batch_bytes"]: flush()
        flush()
        require(totals["ways"] > 0, "empty road inventory cannot be complete")
        if expected:
            require(expected.execute("SELECT count(*) FROM expected").fetchone()[0] == totals["ways"],
                "selected source ways omitted by GDAL geometry assembly")
        require(connection.execute("PRAGMA quick_check").fetchone() == ("ok",), "road SQLite integrity refused")
        require(connection.execute("SELECT count(*) FROM ways").fetchone()[0] == totals["ways"]
            and connection.execute("SELECT count(*) FROM way_bounds").fetchone()[0] == totals["ways"], "road index count differs")
        content = hashlib.sha256()
        for row in connection.execute("SELECT osm_id,highway,tags_json,wkb,n_vertices FROM ways ORDER BY osm_id"):
            header = json.dumps([row[0], row[1], row[2], row[4]], separators=(",", ":"), ensure_ascii=False).encode()
            content.update(len(header).to_bytes(8, "little")); content.update(header)
            content.update(len(row[3]).to_bytes(8, "little")); content.update(row[3])
        guard.check(force=True)
        return {"counts": dict(totals), "content_sha256": content.hexdigest(), "crs": "EPSG:4326",
            "axis_order": "WKBx=longitude,y=latitude", "spatial_index": "SQLite RTree whole-way bounds",
            "full_way_geometry": True, "source_way_counts_and_tags_verified": expected is not None,
            "coordinate_authority": "GDAL; independent metadata guard does not re-derive coordinates/order",
            "bbox": None, "topology": False, "legal_routing": False}
    finally:
        try:
            connection.close()
            if expected: expected.close()
            close = getattr(rows, "close", None)
            if close: close()
        finally:
            try:
                if locked:
                    restore = sqlite3.connect(":memory:")
                    try:
                        value = old_temp[0] if old_temp else ""
                        restore.execute("PRAGMA temp_store_directory='"+value.replace("'", "''")+"'")
                    finally:
                        restore.close()
            finally:
                if locked: _LOCK.release()
