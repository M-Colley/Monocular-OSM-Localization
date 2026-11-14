"""Numerical-only fixed32 native-window diagnostic; never opens video.

ALL32 metadata/source bindings and Root's closed prepared-input acceptance
precede NPZ members. Tracking failures and independent gauges are preserved.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from monocular_osm import multiview_motion as frontend
from monocular_osm import multiview_motion_inputs as p
from scripts import prepare_multiview_motion_inputs as prepare

FRONTEND_SHA256="f6abbf1de3af69c8733291d7ebd81ec9e56e28395466fe70c203dcd3469190b6"
FRONTEND_PROFILE_SHA256="7c1e7828481f3b11073e1e84c724da7ece8e3a8ca609062f207de43a201352bd"
INPUT_ARRAYS={"frames_bgr","video_ordinals","timestamps","K","native_read_rows"}
TRACK_ARRAYS={"seed_ids","fit_seed","check_seed","points_px","valid",
    "forward_backward_error_px","forward_patch_error","rejection_reason","ordinals","timestamps","K"}
POSE_ARRAYS={"centres":(7,3),"rotations":(7,3,3),"adjacent_steps_gauge":(6,),
             "rotation_only_rotations":(7,3,3)}
MAX_DERIVED_RAW_BYTES=2<<20
MAX_DERIVED_NPZ_BYTES=4<<20
MAX_WINDOW_JSON_BYTES=1<<20
MAX_AGGREGATE_JSON_BYTES=4<<20


@contextmanager
def cpu_two_threads():
    """Actual installed NumPy/SciPy BLAS control; no optional dependency."""
    import ctypes
    import scipy
    controlled=[];records=[]
    try:
        for module in (np,scipy):
            directory=Path(module.__file__).parent.parent/(module.__name__+".libs")
            for path in directory.glob("*openblas*"):
                library=ctypes.CDLL(str(path));get=setter=None
                for name in ("scipy_openblas_get_num_threads64_","scipy_openblas_get_num_threads",
                             "openblas_get_num_threads64_","openblas_get_num_threads"):
                    try:
                        get=getattr(library,name);setter=getattr(library,name.replace("get_num_threads","set_num_threads"));break
                    except AttributeError:continue
                p.require(get is not None and setter is not None,"actual BLAS control unavailable")
                get.argtypes=[];get.restype=ctypes.c_int;setter.argtypes=[ctypes.c_int];setter.restype=None
                old=int(get());setter(2);controlled.append((library,setter,old))
                p.require(int(get())==2,"actual CPU2 limit failed")
                records.append({"filepath":str(path),"num_threads":int(get()),"previous_num_threads":old})
        p.require(len(records)>=2,"NumPy/SciPy CPU2 proof missing")
        yield records
    finally:
        for library,setter,old in reversed(controlled):setter(old)


def headroom():
    physical=int(psutil.virtual_memory().available);commit=prepare.available_commit()
    p.require(min(physical,commit)>=8<<30,"physical/truecommit8GiB headroom required")
    return {"available_physical_bytes":physical,"available_commit_bytes":commit}


def origins():
    p.require(Path(__file__).resolve()==ROOT/"scripts/multiview_motion_probe.py"
        and Path(frontend.__file__).resolve()==ROOT/"monocular_osm/multiview_motion.py"
        and Path(p.__file__).resolve()==ROOT/"monocular_osm/multiview_motion_inputs.py"
        and Path(prepare.__file__).resolve()==ROOT/"scripts/prepare_multiview_motion_inputs.py",
        "genuine executing worker/frontend/protocol origins required")
    p.require(p.sha(frontend.__file__)==FRONTEND_SHA256,"registered exact frontend source changed")
    profile_sha=hashlib.sha256(json.dumps(frontend.PROFILE,sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()
    p.require(profile_sha==FRONTEND_PROFILE_SHA256,"frontend numerical profile changed")


def finite_json(value):
    """Explicit JSON nulls with JSON-pointer/kind inventory; NPZ stays raw."""
    nonfinite=[];nodes=0
    def clean(item,path,depth):
        nonlocal nodes
        nodes+=1
        if nodes>100000 or depth>32:raise ValueError("bounded diagnostic JSON structure required")
        if isinstance(item,np.generic):return clean(item.item(),path,depth)
        if item is None or isinstance(item,(str,bool,int)):return item
        if isinstance(item,float):
            if np.isfinite(item):return item
            nonfinite.append({"path":path,"kind":"nan" if np.isnan(item) else
                "positive_infinity" if item>0 else "negative_infinity"})
            return None
        if isinstance(item,(list,tuple)):
            return [clean(x,path+"/"+str(i),depth+1) for i,x in enumerate(item)]
        if isinstance(item,dict):
            if not all(type(k) is str for k in item):raise ValueError("diagnostic JSON keys must be strings")
            return {k:clean(v,path+"/"+k.replace("~","~0").replace("/","~1"),depth+1)
                    for k,v in item.items()}
        raise ValueError("diagnostic arrays/objects must not enter JSON metadata")
    return clean(value,"",0),nonfinite


def track_arrays(tracks,body):
    arrays={key:value for key,value in tracks.items() if isinstance(value,np.ndarray)}
    p.require(set(arrays)==TRACK_ARRAYS,"all original tracker ndarray roles required")
    ids=arrays["seed_ids"];n=len(ids)
    expected={"seed_ids":(np.int32,(n,)),"fit_seed":(np.bool_,(n,)),
        "check_seed":(np.bool_,(n,)),"points_px":(np.float64,(7,n,2)),
        "valid":(np.bool_,(7,n)),"forward_backward_error_px":(np.float64,(6,n)),
        "forward_patch_error":(np.float64,(6,n)),"ordinals":(np.int64,(7,)),
        "timestamps":(np.float64,(7,)),"K":(np.float64,(3,3))}
    p.require(0<=n<=800 and all(arrays[k].dtype==dtype and arrays[k].shape==shape
        and arrays[k].flags.c_contiguous for k,(dtype,shape) in expected.items()),
        "bounded complete typed raw tracker arrays required")
    reasons=arrays["rejection_reason"]
    p.require(reasons.dtype==np.dtype("<U32") and reasons.shape==(6,n) and reasons.flags.c_contiguous,
        "all original fixed-width rejection reasons required")
    p.require(np.array_equal(ids,np.arange(n,dtype=np.int32))
        and np.array_equal(arrays["fit_seed"],ids%2==0)
        and np.array_equal(arrays["check_seed"],ids%2==1),"original pre-rejection seed split changed")
    for track_name,input_name in (("ordinals","video_ordinals"),("timestamps","timestamps"),("K","K")):
        p.require(p.typed(arrays[track_name])==p.typed(body[input_name]),"tracker changed native input identity")
    p.require(tuple(tracks["native_shape"])==tuple(body["frames_bgr"].shape[1:])
        and tracks.get("resized") is False and tracks.get("failed_seed_ids_retained") is True
        and tracks.get("LK_error_is_covariance") is False,"full native tracking/unknown uncertainty semantics required")
    return arrays


def process_window(body,*,check=lambda:None):
    """One resident input; derived arrays and honest plain metadata only."""
    p.require(set(body)==INPUT_ARRAYS,"exact five native prepared input roles required")
    before={name:p.typed(value) for name,value in body.items()}
    tracks=None;geometry=None;failure=None
    check()
    try:
        tracks=frontend.track_window(list(body["frames_bgr"]),body["K"],
            body["video_ordinals"],body["timestamps"],check=check)
    except (cv2.error,np.linalg.LinAlgError) as error:
        # Never fabricate the unreturned local state as empty/zero tracks.
        failure={"stage":"track_window","exception_type":type(error).__name__,"message":str(error)}
    arrays={"native_read_rows":body["native_read_rows"].copy()}
    if tracks is not None:
        arrays.update(track_arrays(tracks,body))
        geometry=frontend.solve_window(tracks,body["K"],check=check)
        p.require(geometry.get("status") in {"complete_relative_gauge_candidate","unavailable"}
            and all(geometry.get(name) is False for name in
                ("metric_scale_recovered","measured_stop","calibrated_covariance","static_world_verified")),
            "candidate or explicit unknown required; no physical certificates")
        pose={key:value for key,value in geometry.items() if isinstance(value,np.ndarray)}
        p.require(set(pose)<=set(POSE_ARRAYS) and all(v.dtype==np.float64 and v.shape==POSE_ARRAYS[k]
            and not v.dtype.hasobject and np.isfinite(v).all() for k,v in pose.items()),
            "bounded genuine relative-pose ndarray roles required")
        arrays.update(pose)
    else:
        arrays.update(video_ordinals=body["video_ordinals"].copy(),timestamps=body["timestamps"].copy(),K=body["K"].copy())
    check()
    p.require(all(p.typed(body[name])==digest for name,digest in before.items()),
        "numerical frontend changed prepared input values")
    p.require(sum(a.nbytes for a in arrays.values())<=MAX_DERIVED_RAW_BYTES
        and all(not a.dtype.hasobject for a in arrays.values()),"derived window payload exceeds fixed bound")
    metadata={"status":"failed" if failure else geometry["status"],
        "raw_tracks_available":tracks is not None,"failure":failure,
        "tracking":None if tracks is None else {k:v for k,v in tracks.items() if not isinstance(v,np.ndarray)},
        "geometry":None if geometry is None else {k:v for k,v in geometry.items() if not isinstance(v,np.ndarray)},
        "track_array_roles":[] if tracks is None else sorted(TRACK_ARRAYS),
        "pose_array_roles":[] if geometry is None else sorted(k for k,v in geometry.items() if isinstance(v,np.ndarray)),
        "NPZ_nonfinite_values_preserved":True,"independent_window_gauge_only":True,
        "no_global_trajectory":True,"selected_answer":None,"policy_admission":False}
    cleaned,inventory=finite_json(metadata)
    cleaned["json_nonfinite_path_inventory"]=inventory
    return arrays,cleaned


def write_json_bound(path,value,maximum):
    p.require(len(p.json_bytes(value))<=maximum,"derived diagnostic JSON exceeds fixed bound")
    p.write_json(path,value)


def write_arrays(path,arrays):
    p.require(sum(a.nbytes for a in arrays.values())<=MAX_DERIVED_RAW_BYTES
        and all(not a.dtype.hasobject for a in arrays.values()),"bounded non-pickle derived arrays required")
    with path.open("xb") as stream:np.savez(stream,**arrays)
    p.require(path.stat().st_size<=MAX_DERIVED_NPZ_BYTES,"derived NPZ exceeds4MiB bound")
    return {key:p.typed_header(value) for key,value in arrays.items()}


def execute(config_path,expected,*,check=lambda:None):
    began=time.monotonic();process=psutil.Process()
    original_affinity=process.cpu_affinity();p.require(len(original_affinity)>=2,"CPU2 required")
    original_cv_threads=cv2.getNumThreads();process.cpu_affinity(original_affinity[:2]);cv2.setNumThreads(2)
    resource={"samples":0,"peak_sampled_RSS_bytes":0,"minimum_physical_bytes":None,"minimum_commit_bytes":None}
    def poll():
        check()
        p.require(time.monotonic()-began<=1800.,"numerical diagnostic exceeded1800s inner deadline")
        rss=process.memory_info().rss;p.require(rss<=3<<30,"numerical diagnostic exceeded3GiB RSS")
        free=headroom();resource["samples"]+=1
        resource["peak_sampled_RSS_bytes"]=max(resource["peak_sampled_RSS_bytes"],rss)
        for target,source in (("minimum_physical_bytes","available_physical_bytes"),("minimum_commit_bytes","available_commit_bytes")):
            old=resource[target];resource[target]=free[source] if old is None else min(old,free[source])
    try:
        poll();origins()
        config_path=Path(config_path).resolve(strict=True)
        config,ledger=p.load_config(config_path,expected,"run")
        manifest=p.admit_prepared(config,ledger)  # ALL32/closed Root gate before any NPZ member.
        p.require(isinstance(getattr(ledger,"historical_protected_refs",None),dict)
            and bool(ledger.historical_protected_refs),"consumed-only admission with opaque historical proof required")
        rows=manifest["windows"]
        p.require([row["window_id"] for row in rows]==list(range(32)),"exact fixed ordered32 windows required")
        root_qa=ledger.json(config["prepared_input_acceptance_ref"])
        manifest_path=ledger.add(root_qa["prepared_manifest_ref"])
        output=Path(config["output_dir"])
        output=(config_path.parent/output if not output.is_absolute() else output).resolve()
        p.require(not output.is_relative_to(manifest_path.parent)
            and not manifest_path.parent.is_relative_to(output)
            and not any(Path(path).is_relative_to(output) for path in ledger.protected),
            "new numerical family must be separate from protected inputs")
        output.mkdir(parents=True,exist_ok=False)
        artifacts={};windows=[]
        with cpu_two_threads() as threads:
            for row in rows:
                poll()
                body=p.read_prepared_window(row,ledger,check=poll)
                arrays,metadata=process_window(body,check=poll)
                metadata.update(window_id=row["window_id"],center_row=row["center_row"],
                    prepared_input_window=deepcopy(row),frontend_profile_sha256=FRONTEND_PROFILE_SHA256,
                    raw_track_records_not_replaced_or_interpolated=True)
                stem=f"window_{row['window_id']:02d}"
                array_path=output/(stem+".npz");json_path=output/(stem+".json")
                headers=write_arrays(array_path,arrays)
                metadata.update(array_ref={"path":str(array_path),"sha256":p.sha(array_path)},typed_headers=headers)
                write_json_bound(json_path,metadata,MAX_WINDOW_JSON_BYTES)
                artifacts[array_path.name]=metadata["array_ref"]["sha256"]
                artifacts[json_path.name]=p.sha(json_path)
                windows.append({"window_id":row["window_id"],"status":metadata["status"],
                    "raw_tracks_available":metadata["raw_tracks_available"],
                    "array_ref":metadata["array_ref"],"result_ref":{"path":str(json_path),"sha256":artifacts[json_path.name]}})
                del body,arrays,metadata,headers  # No native frame/track storage crosses next loader.
            thread_proof=deepcopy(threads)
        poll()
        summary={"schema":"native-multiview-motion-window-results-v1","window_count":32,"windows":windows,
            "status_counts":dict(Counter(row["status"] for row in windows)),
            "all_fixed_windows_retained":True,"raw_track_windows_available":sum(row["raw_tracks_available"] for row in windows),
            "independent_window_gauges_only":True,"no_global_trajectory":True,
            "metric_scale_recovered":False,"calibrated_covariance":False,"static_world_verified":False,
            "geometry_candidate_is_physical_motion_certificate":False,"selected_answer":None,"policy_admission":False}
        write_json_bound(output/"results.json",summary,MAX_AGGREGATE_JSON_BYTES)
        artifacts["results.json"]=p.sha(output/"results.json")
        master={"schema":"native-multiview-motion-probe-freeze-v1",
            "status":"CLOSED_FIXED32_RELATIVE_GEOMETRY_DIAGNOSTIC_NO_ACCURACY",
            "config_ref":{"path":str(config_path),"sha256":expected},
            "source_hashes":config["source_hashes"],"study_registration_ref":config["study_registration_ref"],
            "native_read_ledger_ref":config["native_read_ledger_ref"],
            "prepared_input_acceptance_ref":config["prepared_input_acceptance_ref"],
            "prepared_manifest_ref":root_qa["prepared_manifest_ref"],
            "window_count":32,"windows":windows,"status_counts":summary["status_counts"],
            "all_fixed_windows_retained":True,"raw_track_windows_available":summary["raw_track_windows_available"],
            "frontend_source_sha256":FRONTEND_SHA256,"frontend_profile_sha256":FRONTEND_PROFILE_SHA256,
            "artifacts":artifacts,"protected_hashes":dict(ledger.protected),"threadpools":thread_proof,
            "historical_prepare_time_protected_refs":deepcopy(ledger.historical_protected_refs),
            "historical_originals_reopened_or_rehashed_by_numerical_worker":False,
            "fresh_integrity_scope":"consumed current source/config/study/ledger/RootQA/closure/manifest/32packets only",
            "resources_sampled":resource,"external_CPU2_RSS3GiB_floor8GiB_outer1830s_guard_required":True,
            "external_actor_closure_verified":False,"new_neural_forwards_or_video_reopen":False,
            "original_R_T_or_ground_truth_or_maps_or_OCR_used":False,
            "independent_window_gauges_only":True,"no_global_trajectory":True,
            "metric_scale_recovered":False,"calibrated_covariance":False,"static_world_verified":False,
            "selected_answer":None,"policy_admission":False,"counts_toward_goal":False}
        poll()  # Last callback; checks/publication below are callback-free.
        for name,digest in artifacts.items():p.require(p.sha(output/name)==digest,"derived output changed before final publication")
        origins();ledger.verify()
        p.require({q.name for q in output.iterdir()}==set(artifacts),"unexpected numerical output membership")
        p.require(time.monotonic()-began<=1800.,"final integrity checks exceeded numerical inner deadline")
        write_json_bound(output/"frozen_run.json",master,MAX_AGGREGATE_JSON_BYTES)
        return master
    finally:
        cv2.setNumThreads(original_cv_threads);process.cpu_affinity(original_affinity)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--config-sha256",required=True)
    args=parser.parse_args(argv)
    execute(args.config,args.config_sha256)


if __name__=="__main__":main()
