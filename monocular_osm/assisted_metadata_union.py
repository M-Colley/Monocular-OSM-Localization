"""Source-bound v3 metadata component; no acquisition, neural calls or defaults.

Later worker wiring seals a metadata control before invoking this module. The
ordinary baseline child is an independent frozen control, possibly many pools.
The caller retains original legacy admissions and handles municipality fallback.
"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time
import zipfile
import numpy as np
from . import metadata_union as core, assisted_comparison as comparison

PROFILE = {"version": "source-bound-metadata-component-v3", "posthoc_DEV": True,
    "core_profile_sha256": core.PROFILE_SHA256, "comparison_profile_sha256": comparison.PROFILE_SHA256,
    "queries": 80, "max_requested_pools": 6, "max_references": 10000,
    "sole_primary": "reference_mean", "secondary": "raw diagnostic only",
    "legacy_priority": "exact original v1 decoder/identity gate first", "extra_memory_limit_MiB": 400,
    "cpu_threads": 2, "minimum_free_physical_and_true_commit_GiB": 8, "DP_wall_budget_s": 3600,
    "external_watchdog_required": True, "no_acquisition_or_model_calls": True, "default_adopted": False}
PROFILE_SHA256 = core.json_sha(PROFILE)
ARRAYS = {"query_indices": (80,"iu",8), "query_times": (80,"f",8), "ref_ids": (1500,"US",512),
    "ref_latlon": (3000,"f",8), "raw_similarities": (120000,"f",4), "decoded_ref_indices": (80,"iu",8),
    "decoded_cosines": (80,"f",8), "returned_indices": (80,"iu",8), "returned_latlon": (160,"f",8), "returned_cosines": (80,"f",8)}


def safe(path):
    p = Path(path)
    for candidate in (p,p.resolve()):
        text = str(candidate).replace("\\","/").casefold()
        core.require(re.search(r"(?:^|[/_.-])(?:gt|ground[-_]?truth|oxts|labels?)(?=$|[/_.-])",text) is None
            and not (candidate.name.casefold().endswith((".json",".jsonl"))
            and re.search(r"(?:^|[_.-])(?:scores?|scoring)(?:[_.-]|$)", candidate.name.casefold())), "labels/saved scores are not component inputs")
    return p


def sha(path):
    h = hashlib.sha256()
    with safe(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2),b""): h.update(block)
    return h.hexdigest()


def relative(name):
    core.require(isinstance(name,str), "artifact name must be a string")
    value = PurePosixPath(name.replace("\\","/"))
    core.require(value.parts and not value.is_absolute() and ".." not in value.parts
        and not any(":" in part for part in value.parts), "unsafe frozen artifact")
    return Path(*value.parts)


def _json_bound(path, hashes):
    path = safe(path).resolve(strict=True)
    core.require(path in hashes and sha(path) == hashes[path] and path.stat().st_size <= 8*1024**2,
        "control must be hash-bound before JSON load")
    return json.loads(path.read_text(encoding="utf8"))


def _arrays_bound(path, hashes, specs):
    path = safe(path).resolve(strict=True)
    core.require(path in hashes and sha(path) == hashes[path] and path.stat().st_size <= 64*1024**2,
        "control must be hash-bound before array load")
    with zipfile.ZipFile(path) as archive:
        core.require(len(archive.namelist()) == len(specs) and set(archive.namelist()) == {k+".npy" for k in specs}, "unexpected array roles")
        for key,(maximum,kinds,size) in specs.items():
            with archive.open(key+".npy") as stream:
                version = np.lib.format.read_magic(stream)
                core.require(version in {(1,0),(2,0)}, "unsupported NPY header")
                reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
                shape,fortran,dtype = reader(stream)
            core.require(not fortran and not dtype.hasobject and dtype.kind in kinds and dtype.itemsize <= size
                and math.prod(shape) <= maximum and archive.getinfo(key+".npy").file_size <= maximum*size+65536,
                "typed array header exceeds component bound")
            core.require(not (kinds == "f" and size == 4) or dtype == np.float32, "actual float32 evidence required")
    with np.load(path,allow_pickle=False) as saved: return {key:saved[key] for key in specs}


@dataclass(frozen=True)
class FrozenControl:
    comparison: dict
    summary: dict
    regions: list
    source: dict
    protected_hashes: dict

    def verify(self):
        core.require(self.protected_hashes and self.source.get("freeze") and self.source.get("candidate"), "actual frozen control source missing")
        core.require(self.protected_hashes.get(safe(self.source["freeze"]).resolve(strict=True)) == self.source["freeze_sha256"]
            and self.protected_hashes.get(safe(self.source["candidate"]).resolve(strict=True)) == self.source["candidate_sha256"], "control source receipt changed")
        for path,digest in self.protected_hashes.items():
            core.require(sha(path) == digest, "frozen control/source changed")
        candidate = _json_bound(self.source["candidate"],self.protected_hashes)
        core.require(candidate["comparison"] == self.comparison, "in-memory comparison differs from frozen control")
        request = _json_bound(self.source["request"],self.protected_hashes);expected = self.source["identity"]
        core.require(all(candidate.get(k) == expected[k] for k in ("kind","video_sha256","source_sha256","window"))
            and all(request.get(k) == expected[k] for k in ("kind","execution_id","video_sha256","source_sha256"))
            and [request["config"]["vo_start_sec"],request["config"]["vo_end_sec"]] == expected["window"], "control source identity no longer matches frozen request/candidate")
        core.legacy_summary(self.comparison,self.summary)
        core.require(len(self.source["audit_metadata"]) == len(self.regions), "control audit source list changed")
        for region,path in zip(self.regions,self.source["audit_metadata"]):
            metadata = _json_bound(path,self.protected_hashes)
            core.require(region["audit"] == metadata, "in-memory audit differs from frozen source")
            declared = metadata["array_artifact"]["arrays"]
            core.require(set(declared) == set(region["arrays"]), "control array inventory changed")
            for key,array in region["arrays"].items():
                core.require(declared[key] == {"sha256":core.full._array_sha(array),"shape":list(array.shape),"dtype":array.dtype.str},
                    "control in-memory evidence changed")


def load_legacy_control(frozen_artifact, candidate_relative, original_summary, expected, *, request_relative="request.json"):
    """Load a sealed baseline/metadata control, after binding every file SHA.

expected has kind (baseline/expanded), execution_id, video_sha256, window and
source_sha256. Metadata snapshots use the same protocol in their own directory.
Neither a candidate's scalar claims nor an unfrozen path authorizes a load.
"""
    core.require(expected.get("kind") in {"baseline","expanded"}, "only baseline/metadata controls supported")
    freeze = safe(frozen_artifact["path"]).resolve(strict=True)
    core.require(core.digest(frozen_artifact["sha256"]) and sha(freeze) == frozen_artifact["sha256"], "control freeze changed before load")
    hashes = {freeze:frozen_artifact["sha256"]};frozen = _json_bound(freeze,hashes);root = freeze.parent
    core.require(isinstance(frozen.get("artifacts"),dict) and 1 <= len(frozen["artifacts"]) <= 20000, "bounded frozen control inventory required")
    targets = [];names = set()
    for name,digest in frozen["artifacts"].items():
        path = safe(root/relative(name)).resolve(strict=True);key = relative(name).as_posix().casefold()
        core.require(path.is_relative_to(root) and key not in names and core.digest(digest), "aliased/escaped frozen control")
        names.add(key);targets.append((path,digest))
    for path,digest in targets:
        core.require(sha(path) == digest, "original control artifact changed before load");hashes[path] = digest
    core.require(frozen.get("scoring_has_not_run") is True and frozen.get("kind") == expected["kind"]
        and frozen.get("video_sha256") == expected["video_sha256"] and frozen.get("source_sha256") == expected["source_sha256"], "control frozen input identity differs")
    request = _json_bound(root/relative(request_relative),hashes)
    candidate_path = root/relative(candidate_relative);candidate = _json_bound(candidate_path,hashes)
    request_window = [request["config"]["vo_start_sec"],request["config"]["vo_end_sec"]]
    core.require(request.get("kind") == candidate.get("kind") == expected["kind"]
        and request.get("execution_id") == expected["execution_id"] and request_window == expected["window"]
        and candidate.get("window") == expected["window"] and candidate.get("status") == "complete"
        and all(request.get(k) == candidate.get(k) == expected[k] for k in ("video_sha256","source_sha256")), "candidate/request differs from same-execution source/video/window")
    c = candidate["comparison"];core.legacy_summary(c,original_summary)
    identity,indices,times = core.appearance(c)
    core.require(len(indices) == 80 and np.array_equal(indices,np.arange(80)), "all80 baseline/metadata queries required")
    comparison.legacy_summary(c,candidate["route_latlon"],expanded=expected["kind"] == "expanded")
    records = c["audit_artifacts"];paths = {}
    core.require(isinstance(records,list) and 2 <= len(records) <= 16, "all original audit artifacts required")
    for record in records:
        path = safe(record["path"]).resolve(strict=True)
        core.require(path.is_relative_to(root) and re.fullmatch(r"retrieval_audit/[0-9]{4}\.(json|npz)",path.relative_to(root).as_posix())
            and path not in paths and path in hashes and hashes[path] == record["sha256"], "audit is unbound, repeated or escaped")
        paths[path] = record["sha256"]
    frozen_paths = {p for p in hashes if p.parent == root/"retrieval_audit" and p.suffix in {".json",".npz"}}
    core.require(set(paths) == frozen_paths and len(paths)%2 == 0, "an original control pool was omitted")
    n = len(paths)//2
    core.require(set(paths) == {root/"retrieval_audit"/f"{i:04d}.{suffix}" for i in range(n) for suffix in ("json","npz")}, "control audit ordinals/pairs incomplete")
    regions = []
    for i in range(n):
        metadata_path = root/"retrieval_audit"/f"{i:04d}.json";array_path = metadata_path.with_suffix(".npz")
        metadata = _json_bound(metadata_path,hashes);arrays = _arrays_bound(array_path,hashes,ARRAYS)
        core.require(safe(metadata["array_artifact"]["path"]).resolve(strict=True) == array_path
            and metadata["array_artifact"]["sha256"] == hashes[array_path], "metadata points to another matrix")
        core.require(core.digest(metadata.get("ref_embeddings_sha256")), "actual consumed reference tensor hash required")
        region = {"key":f"audit_{i:04d}","audit":metadata,"arrays":arrays};core.validate_region(region,identity,indices,times);regions.append(region)
    core.prove_baseline_selection(regions,c)
    used = [regions[pair[0]]["audit"] for pair in c["selected_audit_rows"]]
    first = used[0];decoder = first["decoder"]
    expected_decoder = {"actual":decoder["actual"],"fallback":bool(decoder.get("fallback_reason")),
        "settings":{k:v for k,v in decoder.items() if k not in {"actual","fallback_reason"}}}
    core.require(len({row["comparison_identity_sha256"] for row in used}) == 1
        and c["inner_decoder"] == expected_decoder and c["audited_identity"] == first["comparison_identity"]
        and c["audited_identity_sha256"] == first["comparison_identity_sha256"], "legacy decoder/source comparison is stale")
    sampling = c["source_sampling"];dense = sampling["dense_count"]
    core.require(sampling.get("mode") == "linspace" and type(dense) is int and dense >= 80
        and sampling.get("sample_count") == sampling.get("requested_sample_count") == 80
        and sampling.get("dense_indices") == np.linspace(0,dense-1,80).astype(int).tolist(), "original native sampling proof incomplete")
    start,end = expected["window"]
    core.require(times[0] >= (start or 0)-1 and (end is None or times[-1] <= end+1), "queries outside registered window")
    value = FrozenControl(deepcopy(c),deepcopy(original_summary),regions,
        {"freeze":str(freeze),"freeze_sha256":frozen_artifact["sha256"],"candidate":str(candidate_path),
         "candidate_sha256":hashes[candidate_path],"identity":deepcopy(expected),
         "request":str(root/relative(request_relative)),
         "audit_metadata":[str(root/"retrieval_audit"/f"{i:04d}.json") for i in range(n)]},hashes)
    value.verify();return value


def headroom():
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
                ("total_phys","available_phys","total_page","available_page","total_virtual","available_virtual","extended")]
        class Performance(ctypes.Structure):
            _fields_ = [("cb",ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in
                ("CommitTotal","CommitLimit","CommitPeak","PhysicalTotal","PhysicalAvailable","SystemCache","KernelTotal","KernelPaged","KernelNonpaged","PageSize")]+[(k,ctypes.c_ulong) for k in ("HandleCount","ProcessCount","ThreadCount")]
        m = Memory();m.length = ctypes.sizeof(m);p = Performance();p.cb = ctypes.sizeof(p)
        core.require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            and ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p),ctypes.sizeof(p)), "physical/actual commit inspection unavailable")
        physical,commit = int(m.available_phys),int((p.CommitLimit-p.CommitTotal)*p.PageSize)
    else:
        values = {k:int(v.split()[0])*1024 for k,v in (line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())}
        physical,commit = values["MemAvailable"],values["CommitLimit"]-values["Committed_AS"]
    core.require(min(physical,commit) >= 8*1024**3, "8GiB physical and true commit required")
    return {"available_physical_bytes":physical,"available_commit_bytes":commit}


@contextmanager
def cpu_two_threads():
    """Actual installed NumPy/SciPy BLAS controls; no extra dependency/model."""
    import scipy
    controlled = [];records = []
    try:
        for module in (np,scipy):
            directory = Path(module.__file__).parent.parent/(module.__name__+".libs")
            for path in directory.glob("*openblas*"):
                library = ctypes.CDLL(str(path));get = setter = None
                for name in ("scipy_openblas_get_num_threads64_","scipy_openblas_get_num_threads","openblas_get_num_threads64_","openblas_get_num_threads"):
                    try:get = getattr(library,name);setter = getattr(library,name.replace("get_num_threads","set_num_threads"));break
                    except AttributeError:continue
                core.require(get is not None and setter is not None, "actual BLAS control unavailable")
                get.argtypes = [];get.restype = ctypes.c_int;setter.argtypes = [ctypes.c_int];setter.restype = None
                old = int(get());setter(2);controlled.append((library,setter,old))
                core.require(int(get()) == 2, "actual CPU2 limit failed")
                records.append({"filepath":str(path),"num_threads":int(get()),"previous_num_threads":old})
        core.require(len(records) >= 2, "NumPy/SciPy CPU2 proof missing")
        yield records
    finally:
        for library,setter,old in reversed(controlled):setter(old)


@dataclass(frozen=True)
class DescriptorEvidence:
    rows: np.ndarray
    receipt: dict
    protected_hashes: dict

    def verify(self):
        core.require(self.receipt.get("status") == "complete" and self.receipt.get("payload_mode") == "descriptors_and_pixels"
            and {v.get("role") for v in self.receipt.get("files",[])} >= {"metadata","pixels","descriptors"}, "complete actual public receipt required")
        for path,digest in self.protected_hashes.items():core.require(sha(path) == digest, "protected reference payload changed")
        core.require(self.protected_hashes and self.rows.dtype == np.float32 and self.rows.ndim == 2
            and self.rows.shape == (self.receipt["count"],8448) and np.isfinite(self.rows).all(), "actual protected descriptor rows missing")
        value = hashlib.sha256(json.dumps([list(self.rows.shape),"torch.float32"],separators=(",",":")).encode())
        value.update(memoryview(np.ascontiguousarray(self.rows)).cast("B"))
        core.require(value.hexdigest() == self.receipt["ref_embeddings_sha256"], "descriptor memory no longer matches receipt")


def load_descriptor_rows(region,cfg,*,describe=None):
    """Resolve actual rows through a full existing public-cache receipt.

No query/image inference or acquisition. The public API validates the signed
inventory, pixel identities, consumed descriptor tensor, source and location.
"""
    if describe is None:
        from .public_reference_cache import describe_pool
        describe = describe_pool
    audit = {**deepcopy(region["audit"]),**region["arrays"]}
    pool = describe(audit,cfg)
    ids,refs = region["arrays"]["ref_ids"],region["arrays"]["ref_latlon"]
    core.require(pool.get("status") == "complete" and pool.get("payload_mode") == "descriptors_and_pixels"
        and pool.get("count") == len(ids) and pool.get("ref_ids_sha256") == core.full._array_sha(ids)
        and pool.get("ref_latlon_sha256") == core.full._array_sha(np.asarray(refs,float))
        and pool.get("ref_embeddings_sha256") == audit["ref_embeddings_sha256"]
        and pool.get("actual_model_state_sha256") == audit["actual_model_state_sha256"]
        and pool.get("backbone") == audit["backbone"] and pool.get("preprocessing") == audit["preprocessing"], "public receipt differs from actual consumed reference rows")
    directory = safe(pool["request"]["cache_dir"]).resolve(strict=True);hashes = {};descriptors = []
    for item in pool["files"]:
        core.require(Path(item["name"]).name == item["name"] and core.digest(item["sha256"]), "unsafe public receipt payload")
        path = safe(directory/item["name"]).resolve(strict=True)
        core.require(path.is_relative_to(directory) and path not in hashes and path.stat().st_size == item["bytes"]
            and sha(path) == item["sha256"], "public payload changed before descriptor load")
        hashes[path] = item["sha256"]
        if item["role"] == "descriptors":descriptors.append(path)
    core.require({v["role"] for v in pool["files"]} >= {"metadata","pixels","descriptors"}, "complete public payload receipt missing")
    core.require(len(descriptors) == 1 and 1 <= len(ids) <= 1500, "one bounded actual descriptor payload required")
    arrays = _arrays_bound(descriptors[0],hashes,{"emb":(1500*8448,"f",4),"fingerprint":(1,"US",256)})
    rows = arrays["emb"]
    core.require(rows.shape == (len(ids),8448) and str(arrays["fingerprint"]) == pool["fingerprint"]
        and np.isfinite(rows).all() and np.max(abs(np.linalg.norm(rows,axis=1)-1)) <= 1e-4, "actual descriptor rows/order/fingerprint invalid")
    digest = hashlib.sha256(json.dumps([list(rows.shape),"torch.float32"],separators=(",",":")).encode())
    digest.update(memoryview(np.ascontiguousarray(rows)).cast("B"))
    core.require(digest.hexdigest() == audit["ref_embeddings_sha256"], "actual descriptor tensor differs from consumed audit")
    evidence = DescriptorEvidence(rows,deepcopy(pool),hashes);evidence.verify();return evidence


def evaluate_metadata_union(baseline,metadata,*,requested_pool_count,reference_cfg=None,
                            descriptor_loader=load_descriptor_rows,resource_gate=headroom,
                            numerical_context=cpu_two_threads,monotonic=time.monotonic):
    """Return a legacy-preserved, unavailable, or complete fixed union result.

Both controls come from load_legacy_control. Output candidate is always the
reference-mean median5 route, while its comparison retains all raw selections.
The caller freezes ALL research outputs and source records before adoption.
An external owned-worker watchdog enforces the3600s/8GiB limits throughout.
"""
    began = monotonic();dp_started = False;descriptor_evidence = [];research = None
    sources = {}
    try:
        core.require(isinstance(baseline,FrozenControl) and isinstance(metadata,FrozenControl), "source-bound controls required")
        sources = {"baseline":deepcopy(baseline.source),"metadata":deepcopy(metadata.source)}
        baseline.verify();metadata.verify()
        core.require(baseline.source["identity"]["kind"] == "baseline" and metadata.source["identity"]["kind"] == "expanded", "baseline and metadata control roles differ")
        for name in ("execution_id","video_sha256","window","source_sha256"):
            core.require(baseline.source["identity"][name] == metadata.source["identity"][name], "control execution/source/video/window differs")
        base,base_identity,_,_ = core.legacy_summary(baseline.comparison,baseline.summary)
        old,old_identity,_,_ = core.legacy_summary(metadata.comparison,metadata.summary)
        legacy_gate = core.legacy_gate(base,old)
        if legacy_gate["selected_candidate"] == "expanded":
            return {"status":"preserved","selection":"legacy_metadata","legacy_gate":legacy_gate,
                "union_decoded":False,"sources":sources,"profile_sha256":PROFILE_SHA256}
        core.require(base["similarity_median"] < core.policy.policy_config()["baseline_median_min"], "already-admitted baseline must precede union")
        core.require(base_identity == old_identity and type(requested_pool_count) is int
            and 1 <= requested_pool_count <= 6 and len(metadata.regions) == requested_pool_count,
            "all requested1..6 complete comparable metadata pools required")
        core.require(base["point_count"] == old["point_count"] == 80, "fixed full80 query support required")
        core.prove_baseline_selection(baseline.regions,baseline.comparison)
        core.prove_baseline_selection(metadata.regions,metadata.comparison)
        regions = [{"key":r["key"],"audit":deepcopy(r["audit"]),"arrays":r["arrays"]} for r in metadata.regions]
        all_ids = [str(i) for r in regions for i in r["arrays"]["ref_ids"]]
        unique,counts = np.unique(np.asarray(all_ids),return_counts=True)
        overlaps = set(unique[counts > 1].tolist())
        affected = [r for r in regions if overlaps.intersection(r["arrays"]["ref_ids"].tolist())]
        # Full DP/Python mapping/raw copies reserve96MiB; only affected pools
        # need additional actual descriptor rows. At6x1500 this stays<400MiB.
        extra = 96*1024**2+sum(len(r["arrays"]["ref_ids"])*8448*4 for r in affected)
        core.require(len(unique) <= 10000 and extra <= 400*1024**2, "full-support/descriptor allocation exceeds fixed400MiB budget")
        before = resource_gate()
        for region in affected:
            core.require(reference_cfg is not None, "overlap needs actual receipt-bound descriptor source")
            original_audit = deepcopy(region["audit"])
            evidence = descriptor_loader(region,reference_cfg)
            core.require(isinstance(evidence,DescriptorEvidence), "actual descriptor evidence required, not hash claims")
            evidence.verify();receipt = evidence.receipt
            core.require(region["audit"] == original_audit and receipt.get("count") == len(region["arrays"]["ref_ids"])
                and receipt.get("ref_ids_sha256") == core.full._array_sha(region["arrays"]["ref_ids"])
                and receipt.get("ref_latlon_sha256") == core.full._array_sha(np.asarray(region["arrays"]["ref_latlon"],float))
                and receipt.get("ref_embeddings_sha256") == original_audit["ref_embeddings_sha256"]
                and receipt.get("actual_model_state_sha256") == original_audit["actual_model_state_sha256"]
                and receipt.get("backbone") == original_audit["backbone"] and receipt.get("preprocessing") == original_audit["preprocessing"], "descriptor receipt differs from frozen affected pool")
            region["descriptor_rows"] = evidence.rows;descriptor_evidence.append(evidence)
            resource_gate()
        # Validate every duplicate/coordinate/score envelope before invoking
        # any DP, including when actual descriptor receipts are available.
        validated_union = core.build_union(regions,baseline.comparison)
        core.require(len(validated_union["ref_ids"]) == len(unique), "full union membership changed")
        del validated_union
        baseline.verify();metadata.verify()
        core.require(monotonic()-began <= 3600, "registered3600s component budget exceeded before DP")
        with numerical_context() as threads:
            dp_started = True
            research = core.replay_metadata_union(regions,baseline.comparison,baseline.summary,
                metadata.comparison,metadata.summary,baseline.regions)
        core.require(research["status"] == "complete", "fixed raw and reference-mean outputs incomplete")
        core.require(monotonic()-began <= 3600, "registered3600s component budget exceeded")
        after = resource_gate();baseline.verify();metadata.verify()
        for evidence in descriptor_evidence:evidence.verify()
        c = comparison.union_comparison(research,baseline.comparison,baseline.summary,metadata.comparison,
            metadata.summary,requested_pool_count=requested_pool_count)
        summary = comparison.union_summary(c,c["route_latlon"],c["route_timestamps"])
        candidate = {"status":"complete","route_latlon":deepcopy(c["route_latlon"]),
            "route_timestamps":deepcopy(c["route_timestamps"]),"comparison":c}
        return {"status":"complete","selection":research["selection"],"candidate":candidate,"summary":summary,
            "baseline_summary":comparison.augment_baseline_summary(baseline.comparison,baseline.summary),
            "legacy_gate":legacy_gate,"sources":sources,"research":research,"descriptor_receipts":[e.receipt for e in descriptor_evidence],
            "descriptor_evidence":[{"region":region["key"],"rows":e.rows,"receipt":e.receipt,
                "protected_hashes":{str(path):digest for path,digest in e.protected_hashes.items()}}
                for region,e in zip(affected,descriptor_evidence)],
            "runtime":{"elapsed_s":monotonic()-began,"memory_before":before,"memory_end":after,"threadpools":threads,
                "estimated_extra_bytes":extra,"DP_started":True,"new_neural_forwards":False,"network":False,
                "external_watchdog_required":True},"union_decoded":True,"profile":deepcopy(PROFILE),"profile_sha256":PROFILE_SHA256,
            "confidence":None,"default_adopted":False}
    except (OSError,ValueError,KeyError,TypeError,IndexError,OverflowError,zipfile.BadZipFile) as error:
        return {"status":"unavailable","selection":None,"reason":f"{type(error).__name__}: {error}","sources":sources,
            "union_decoded":dp_started,"partial_research":research,"profile_sha256":PROFILE_SHA256,
            "runtime":{"elapsed_s":monotonic()-began,"DP_started":dp_started,"new_neural_forwards":False,"network":False},
            "confidence":None,"default_adopted":False}
