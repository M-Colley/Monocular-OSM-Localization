"""Prepare/infer twelve frozen heading diagnostics; no models/API/labels."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import zipfile
if __name__ == "__main__":
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="2"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1",PROJ_NETWORK="OFF",PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from monocular_osm import heading_sequence as core, heading_motion as motion, reference_metadata as metadata
from scripts import replay_full_sequence as base
require,relative=base.require,base.relative
check_headroom,offline=base.check_headroom,base.offline
ROLES=("megaloc","image","metadata","motion")
DIMENSIONS={"megaloc":8448,"image":6144}
MEMBERS={"megaloc":("full_evidence.npz","query_descriptors.npz","reference_union.npz","runtime_and_lineage.json"),
    "image":("full_evidence.npz","query_descriptors.npy","reference_descriptors.npy","runtime.json"),
    "metadata":("all_reference_metadata.json","metadata_plan.json","request_ledger.json"),
    "motion":("native_motion.npz","motion_registration.json","motion_summary.json")}
PROFILE={"version":"frozen-heading-replay-v1","posthoc_DEV":True,"roles":list(ROLES),
    "dimensions":dict(DIMENSIONS),"queries":80,"views":3,"reference_block":256,"cpu_threads":2,
    "cosine_replay_atol":2e-6,"descriptor_norm_atol":1e-4,
    "core_profile_sha256":core.PROFILE_SHA256,"motion_profile_sha256":motion.PROFILE_SHA256,
    "outputs":[model+"/"+v+"/"+m for model in ("megaloc","image") for v in core.PROFILE["variants"] for m in core.PROFILE["modes"]],
    "selected_variant":None,"models_or_forwards":False,"labels_or_scores":False,
    "legacy_megaloc_reference_encoding_identity":"UNVERIFIED; no encoder-only causal claim"}
PROFILE_SHA256=core._json_sha(PROFILE)
SOURCES=("scripts/replay_heading_sequence.py","scripts/replay_full_sequence.py","scripts/pilot_sparse_city_replay.py",
    "scripts/plan_sparse_city_replay.py",*sorted(str(p.relative_to(ROOT)).replace("\\","/") for p in (ROOT/"monocular_osm").rglob("*.py")))

def allowed(path):
    p=Path(path)
    for v in (p,p.resolve()):
        s,n=str(v).replace("\\","/").casefold(),v.name.casefold()
        require("ground_truth" not in s and not (n.endswith((".json",".jsonl")) and re.search(r"(?:^|[_.-])(?:scores?|scoring)(?:[_.-]|$)",n)),"labels/saved scores are outside replay inputs")
    return p
def sha(path):return base.sha(allowed(path))
def load(path):return base.load(allowed(path))
def digest(v):return isinstance(v,str) and re.fullmatch("[0-9a-f]{64}",v) is not None
def write(path,value):
    with allowed(path).open("x",encoding="utf8") as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def source_identity(expected,previous=None):
    require(Path(__file__).resolve()==ROOT/"scripts/replay_heading_sequence.py" and base.ROOT==ROOT,"executed runner/utility escaped source pin")
    for n,m in list(sys.modules.items()):
        if n.startswith(("scripts.","monocular_osm.")) or n=="monocular_osm":
            require(getattr(m,"__file__",None) and Path(m.__file__).resolve().is_relative_to(ROOT),"loaded module escaped source pin")
    base.source_identity(expected)
    result={n:sha(ROOT/n) for n in SOURCES}
    require(previous is None or result==previous,"execution source bytes changed")
    return result

def freeze(path,expected=None,role=None):
    path=allowed(path).resolve(strict=True);require(expected is None or sha(path)==expected,"registered freeze changed")
    value=load(path);targets=[]
    for n,d in value["artifacts"].items():
        allowed(n);p=allowed(path.parent/relative(n)).resolve(strict=True)
        require(p.is_relative_to(path.parent),"frozen member escaped root");targets.append((p,d))
    for p,d in targets:require(sha(p)==d,"frozen input member changed")
    if role in ("megaloc","image"):
        require(value.get("status")=="complete" and value.get("scoring_has_not_run") is True and value.get("selected_variant") is None,"complete pre-label model evidence required")
    elif role=="metadata":
        require(value.get("status")=="complete" and value.get("all_ids_retained") is True and value.get("models_or_labels") is False and value.get("selected_answer") is None,"complete input-only metadata required")
    elif role=="motion":require(value.get("ground_truth_used") is False and value.get("no_predictions_yet") is True,"fresh input-only motion required")
    return value

def specification(path):
    s=load(path);require(isinstance(s,dict) and set(s)==set(ROLES),"exact four fixed roles required")
    for v in s.values():
        require(isinstance(v,dict) and set(v)=={"freeze","freeze_sha256"} and isinstance(v["freeze"],str) and digest(v["freeze_sha256"]),"fixed freeze path/SHA schema required");allowed(v["freeze"])
    return s

def bounded_header(path,key,maximum,kinds,size,shape=None):
    path=allowed(path);require(path.stat().st_size<=512*1024**2,"array disk budget exceeded")
    archive=zipfile.ZipFile(path) if key is not None else None
    if archive:require(archive.namelist().count(key+".npy")==1,"missing/duplicate NPY member")
    with archive.open(key+".npy") if archive else path.open("rb") as stream:
        version=np.lib.format.read_magic(stream);require(version in {(1,0),(2,0)},"unsupported NPY header")
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        dims,fortran,dtype=reader(stream)
    if archive:
        require(archive.getinfo(key+".npy").file_size<=maximum*size+65536,"inflated member exceeds budget");archive.close()
    require(not dtype.hasobject and dtype.kind in kinds and dtype.itemsize<=size and math.prod(dims)<=maximum and not fortran,"unsafe typed array header")
    require(not (kinds=="f" and size==4) or dtype==np.float32,"float32 descriptor/cosine header required")
    require(shape is None or tuple(dims)==tuple(shape),"fixed role dimensions differ")

def npz_arrays(path,specs):
    with zipfile.ZipFile(allowed(path)) as z:require(len(z.namelist())==len(specs) and set(z.namelist())=={k+".npy" for k in specs},"unexpected NPZ roles")
    for k,(n,t,b) in specs.items():bounded_header(path,k,n,t,b)
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in specs}

def close_arrays(*arrays):
    for a in arrays:
        if getattr(a,"_mmap",None) is not None:a._mmap.close()

def tensor_sha(a):
    h=hashlib.sha256(json.dumps([list(a.shape),"torch.float32"],separators=(",",":")).encode());h.update(memoryview(np.ascontiguousarray(a)).cast("B"));return h.hexdigest()

def descriptors(paths,model,n,runtime):
    d=DIMENSIONS[model]
    if model=="megaloc":
        q=npz_arrays(paths["query_descriptors.npz"],{"descriptors":(240*d,"f",4)})["descriptors"]
        saved=npz_arrays(paths["reference_union.npz"],{"embeddings":(10000*d,"f",4),"ref_ids":(10000,"US",512),"ref_latlon":(20000,"f",8)})
        r,membership=saved["embeddings"],(saved["ref_ids"],saved["ref_latlon"]);actual=runtime["runtime"]
        require(actual["backbone"]["name"]=="megaloc" and actual["backbone"]["hub"]=="gmberton/MegaLoc:5fe0dd697c4a70ba3e23607f6716ab3c606b16db"
            and digest(actual["actual_model_state"]["sha256"]) and actual["query_preprocessing"]["mode"]=="crops" and actual["query_preprocessing"]["views"]==3,"MegaLoc model/crop identity differs")
        require(tensor_sha(q)==actual["query_audit"]["query_descriptors_sha256"] and core._array_sha(r)==runtime["reference_embeddings_sha256"],"MegaLoc descriptor identity differs")
    else:
        bounded_header(paths["query_descriptors.npy"],None,240*d,"f",4,(240,d));bounded_header(paths["reference_descriptors.npy"],None,10000*d,"f",4,(n,d))
        q=np.load(paths["query_descriptors.npy"],mmap_mode="r",allow_pickle=False);r=np.load(paths["reference_descriptors.npy"],mmap_mode="r",allow_pickle=False);membership=None
        p=runtime["canonical_preprocessing"]
        require(p["backbone"]=="dinov2" and p["dimensions"]==d and p["official_pin"]=="5d5e2d8cad11fe7ba699fdbd1f6b0a3c69dade18"
            and p["network_input_color"]=="RGB" and digest(runtime["actual_model_state_sha256"]),"ImAge model identity differs")
        require(core._array_sha(q)==runtime["query_descriptors_sha256"] and core._array_sha(r)==runtime["reference_descriptors_sha256"],"ImAge descriptor identity differs")
    require(q.shape==(240,d) and r.shape==(n,d) and q.dtype==r.dtype==np.float32,"descriptor count/dimension/dtype differs")
    for a in (q,r):
        for start in range(0,len(a),256):
            b=a[start:start+256];require(np.isfinite(b).all() and np.max(np.abs(np.linalg.norm(b,axis=1)-1))<=1e-4,"descriptors must be finite normalized rows")
    return q,r,membership

def metadata_rows(paths,ids,latlon):
    saved,plan,ledger=[load(paths[n]) for n in MEMBERS["metadata"]]
    rebuilt=metadata.plan_requests(ids,latlon,provenance=plan["provenance"])
    require(all(plan[k]==v for k,v in rebuilt.items()),"metadata all-ID request plan differs")
    points={str(i):p.tolist() for i,p in zip(ids,latlon)}
    require(set(saved)==set(points) and len(saved)==len(ids),"metadata must cover every gallery ID")
    require({r["reference_id"]:[r["lat"],r["lon"]] for r in plan["references"]}==points,"metadata public positions differ")
    require(ledger["all_batches_successful"] is True and len(ledger["batches"])==len(plan["batches"]),"metadata incomplete")
    parsed={};root=Path(paths["metadata_plan.json"]).parent
    for batch,outcome in zip(plan["batches"],ledger["batches"]):
        require(outcome["success"] is True and outcome["index"]==batch["index"],"metadata batch incomplete")
        p=root/"attempts"/outcome["attempts"][-1];request,response=load(p/"request.json"),load(p/"response.json")
        require((p/"response.body").stat().st_size<=metadata.PROFILE["max_decoded_body_bytes"],"metadata body budget exceeded")
        body=allowed(p/"response.body").read_bytes()
        require(request["params"]==batch["params"] and response["status"]==200 and response["error_type"] is None
            and response["parsed_successfully"] is True and response["body_redacted_or_reencoded"] is False
            and hashlib.sha256(body).hexdigest()==response["stored_body_sha256"]==response["decoded_body_sha256"],"metadata response identity differs")
        parsed.update(metadata.parse_response(body,batch["reference_ids"]))
    require(parsed==saved,"metadata differs from frozen raw responses")
    cameras,angles=[],[]
    for key in ids:
        row=saved[str(key)];cameras.append(row["fields"]["camera_type"] if row["field_status"]["camera_type"]=="present" else None)
        angles.append(row["fields"]["computed_compass_angle"] if row["field_status"]["computed_compass_angle"]=="present" else np.nan)
    return cameras,np.asarray(angles,np.float64)

def inspect_bundle(paths,plans):
    shared,runtimes,matrices=None,{},{}
    specs={"ref_ids":(10000,"US",512),"ref_latlon":(20000,"f",8),"query_times":(80,"f",8),"query_indices":(80,"iu",8),"frame_ordinals":(80,"iu",8)}
    for model in ("megaloc","image"):
        keys={"raw_cosines":(800000,"f",4)} if model=="megaloc" else {"image_raw_cosines":(800000,"f",4),"megaloc_raw_cosines":(800000,"f",4)}
        a=npz_arrays(paths[model]["full_evidence.npz"],{**specs,**keys})
        require(a["query_times"].shape==(80,) and a["ref_ids"].ndim==1 and 1<=len(a["ref_ids"])<=10000,"complete80-query gallery required")
        if shared is None:shared={k:a[k] for k in specs}
        else:require(all(np.array_equal(shared[k],a[k]) for k in specs),"encoder ID/position/query grids differ")
        runtimes[model]=load(paths[model]["runtime_and_lineage.json" if model=="megaloc" else "runtime.json"])
        matrices[model]=a["raw_cosines" if model=="megaloc" else "image_raw_cosines"]
        if model=="image":require(np.array_equal(a["megaloc_raw_cosines"],matrices["megaloc"]),"frozen MegaLoc control changed")
        q,r,membership=descriptors(paths[model],model,len(shared["ref_ids"]),runtimes[model])
        try:require(membership is None or np.array_equal(membership[0],shared["ref_ids"]) and np.array_equal(membership[1],shared["ref_latlon"]),"reference descriptor columns differ")
        finally:close_arrays(q,r)
    require(np.array_equal(shared["query_indices"],np.arange(80)),"full query indices changed")
    n=len(shared["ref_ids"])
    require(shared["ref_latlon"].shape==(n,2) and np.isfinite(shared["ref_latlon"]).all()
        and np.all(np.abs(shared["ref_latlon"][:,0])<=90) and np.all(np.abs(shared["ref_latlon"][:,1])<=180)
        and len(set(shared["ref_ids"].tolist()))==n and all(str(v).startswith("mapillary:") for v in shared["ref_ids"]),"invalid full public reference membership")
    require(all(v.shape==(80,n) and v.dtype==np.float32 and np.isfinite(v).all() and np.abs(v).max()<=1.00001 for v in matrices.values()),"invalid full float32 cosine matrix")
    query,other=plans["megaloc"]["query"],plans["image"]["query"]
    fields=("registered_window","stride","fps","planned_frame_ordinals","native_shape_hw","dense_count","query_dense_indices")
    require(all(query[k]==other[k] for k in fields) and query["video"]["sha256"]==other["video"]["sha256"],"video/window/native query plans differ")
    require(np.array_equal(shared["frame_ordinals"],query["planned_frame_ordinals"]),"query ordinal plan changed")
    pixels=runtimes["megaloc"]["per_query_native_pixel_sha256"]
    require(len(pixels)==80 and all(digest(v) for v in pixels) and pixels==runtimes["image"]["per_query_native_pixel_sha256"]==plans["image"]["per_query_native_pixel_sha256"],"encoder native pixels differ")
    require(all(core._array_sha(shared["query_times"])==r["actual_query_times_sha256"] for r in runtimes.values()),"query timestamp identity changed")
    native=npz_arrays(paths["motion"]["native_motion.npz"],{"centers":(300000,"f",8),"xz":(200000,"f",8),"valid":(100000,"b",1),"n_inliers":(100000,"iu",8),
        "rotations":(900000,"f",8),"translations":(300000,"f",8),"timestamps":(100000,"f",8),"frame_ordinals":(100000,"iu",8),"native_frame_pixels_sha256":(100000,"US",256),"K":(9,"f",8)})
    reg,summary=load(paths["motion"]["motion_registration.json"]),load(paths["motion"]["motion_summary.json"])
    require(summary["status"]=="complete" and summary["ground_truth_used"] is False and reg["ground_truth_used"] is False and reg["legacy_cache_used"] is False
        and reg["inputs"]["video_sha256"]==query["video"]["sha256"] and reg["inputs"]["window"]==query["registered_window"]
        and reg["stride"]==query["stride"] and reg["fps"]==query["fps"] and reg["native_shape_hw"]==query["native_shape_hw"],"motion video/grid differs")
    require(summary["motion_npz_sha256"]==sha(paths["motion"]["native_motion.npz"])
        and core._array_sha(native["timestamps"])==summary["timestamps_sha256"] and core._array_sha(native["frame_ordinals"])==summary["frame_ordinals_sha256"]
        and core._array_sha(native["native_frame_pixels_sha256"])==summary["native_frame_pixels_sha256"],"motion typed identity changed")
    yaw=motion.local_yaw_evidence(native["rotations"],native["valid"],native["n_inliers"],native["timestamps"],native["frame_ordinals"],shared["query_times"],shared["frame_ordinals"])
    require(native["native_frame_pixels_sha256"][yaw["query_native_indices"]].tolist()==pixels,"native motion/query pixels differ")
    height,width=query["native_shape_hw"];fx=(width/2)/np.tan(np.radians(35.));k=np.array([[fx,0,width/2],[0,fx,height/2],[0,0,1.]])
    require(np.array_equal(native["K"],k),"intrinsics differ from generic70-degree control")
    offsets=motion.crop_bearings_deg(k,(height,width));cameras,angles=metadata_rows(paths["metadata"],shared["ref_ids"],shared["ref_latlon"])
    identity={k:core._array_sha(v) for k,v in shared.items()};identity.update(video_sha256=query["video"]["sha256"],native_pixels_sha256=core._json_sha(pixels),
        matrices={k:core._array_sha(v) for k,v in matrices.items()},cameras_sha256=core._json_sha(cameras),headings_sha256=core._array_sha(angles),offsets_sha256=core._array_sha(offsets))
    return shared,runtimes,matrices,cameras,angles,offsets,yaw,identity

def original_inputs(spec):
    records,paths,plans=[],{},{}
    for role in ROLES:
        item=spec[role];path=allowed(item["freeze"]).resolve(strict=True);f=freeze(path,item["freeze_sha256"],role)
        require(set(MEMBERS[role])<=set(f["artifacts"]),"frozen role members missing")
        paths[role]={n:path.parent/relative(n) for n in MEMBERS[role]}
        if role=="metadata":paths[role].update({n:path.parent/relative(n) for n in f["artifacts"]})
        records.append((role,"original_freeze.json",path,item["freeze_sha256"]))
        records.extend((role,n,p,f["artifacts"][n]) for n,p in paths[role].items())
        if role in ("megaloc","image"):
            runtime=load(paths[role]["runtime_and_lineage.json" if role=="megaloc" else "runtime.json"])
            upstream=path.parent.parent/"inputs_frozen.json";uf=freeze(upstream,runtime["input_freeze_sha256"])
            require(uf.get("scoring_has_not_run") is True,"upstream input freeze was not pre-label")
            plan=path.parent.parent/("inputs/replay_plan.json" if role=="megaloc" else "plan.json");plans[role]=load(plan)
            members={str(relative(k)).replace("\\","/"):v for k,v in uf["artifacts"].items()}
            key=str(plan.relative_to(upstream.parent)).replace("\\","/")
            require(key in members and sha(plan)==members[key],"upstream query plan was not in input freeze")
            records.extend([(role,"upstream_inputs_frozen.json",upstream,sha(upstream)),(role,"upstream_plan.json",plan,sha(plan))])
    return records,paths,plans

@offline()
def prepare(spec,family,expected_commit):
    sources=source_identity(expected_commit);check_headroom();spec=allowed(spec).resolve(strict=True);declared=specification(spec);family=allowed(family).resolve()
    roots=[Path(v["freeze"]).resolve().parent.parent if role in ("megaloc","image") else Path(v["freeze"]).resolve().parent for role,v in declared.items()]
    require(not family.exists() and not family.is_relative_to(ROOT) and not any(family.is_relative_to(p) for p in roots),"new isolated replay family required")
    records,paths,plans=original_inputs(declared);*_,identity=inspect_bundle(paths,plans)
    family.mkdir(parents=True);copied=[]
    try:
        base.copy_verified(spec,family/"inputs/spec.json",sha(spec),copied)
        for role,n,p,d in records:base.copy_verified(p,family/"inputs"/role/relative(n),d,copied)
        source_identity(expected_commit,sources);original_inputs(declared);base.verify_originals(copied)
        write(family/"input_lineage.json",{"copied":copied,"labels_opened":False})
        write(family/"plan.json",{"source_commit":expected_commit,"source_sha256":sources,"profile":PROFILE,"profile_sha256":PROFILE_SHA256,
            "input_identity":identity,"all_requested_outputs":PROFILE["outputs"],"selected_variant":None})
        artifacts={str(Path(r["path"]).relative_to(family)):r["sha256"] for r in copied};artifacts.update({n:sha(family/n) for n in ("plan.json","input_lineage.json")})
        write(family/"inputs_frozen.json",{"scoring_has_not_run":True,"models_have_not_run":True,"artifacts":artifacts})
        return {"status":"planned_only","outputs":12,"profile_sha256":PROFILE_SHA256,"input_freeze_sha256":sha(family/"inputs_frozen.json")}
    except Exception as error:
        write(family/"preparation_failed.json",{"status":"failed","error":type(error).__name__+": "+str(error),"scoring_has_not_run":True});raise

def recompute_views(queries,references,saved):
    require(queries.shape[0]==240 and references.shape[0]==saved.shape[1],"descriptor replay grid differs")
    result=np.empty((80,3,len(references)),np.float32)
    for start in range(0,len(references),256):
        stop=min(len(references),start+256);result[:,:,start:stop]=(queries@references[start:stop].T).reshape(80,3,stop-start)
    require(np.isfinite(result).all() and np.max(np.abs(result.max(axis=1)-saved))<=PROFILE["cosine_replay_atol"],"max-view cosine replay differs from frozen matrix")
    return result

@offline()
def infer(family,expected_commit,input_freeze_sha256):
    require(digest(input_freeze_sha256),"explicit prepared input-freeze SHA required")
    sources=source_identity(expected_commit);family=allowed(family).resolve(strict=True);freeze(family/"inputs_frozen.json",input_freeze_sha256)
    plan,lineage=load(family/"plan.json"),load(family/"input_lineage.json")
    require(plan["profile"]==PROFILE and plan["profile_sha256"]==PROFILE_SHA256 and plan["all_requested_outputs"]==PROFILE["outputs"] and plan["source_commit"]==expected_commit
        and plan["source_sha256"]==sources and plan["selected_variant"] is None,"registered twelve-output source/profile changed")
    declared=specification(family/"inputs/spec.json");original_inputs(declared);base.verify_originals(lineage["copied"])
    paths={role:{n:family/"inputs"/role/relative(n) for n in MEMBERS[role]} for role in ROLES}
    f=load(family/"inputs/metadata/original_freeze.json");paths["metadata"].update({n:family/"inputs/metadata"/relative(n) for n in f["artifacts"]})
    plans={role:load(family/"inputs"/role/"upstream_plan.json") for role in ("megaloc","image")}
    before=check_headroom();shared,runtimes,matrices,cameras,angles,offsets,yaw,identity=inspect_bundle(paths,plans)
    require(identity==plan["input_identity"],"prepared full input/grid identity changed")
    output=family/"predictions";require(not output.exists(),"refuse rerunning or overwriting completed/partial inference");output.mkdir();completed=[];started=time.monotonic()
    try:
        from monocular_osm.provenance import snapshot_sources
        actual_sources=snapshot_sources([ROOT/n for n in SOURCES],output)
        np.savez(output/"motion_evidence.npz",vo_yaw_delta_deg=yaw["vo_yaw_delta_deg"],vo_quality=yaw["vo_quality"],crop_offsets_deg=offsets)
        write(output/"motion_evidence.json",{k:v for k,v in yaw.items() if k not in ("vo_yaw_delta_deg","vo_quality","query_native_indices")})
        stats={}
        for model in ("megaloc","image"):
            memory=check_headroom();begin=time.monotonic();q,r,_=descriptors(paths[model],model,len(shared["ref_ids"]),runtimes[model])
            try:views=recompute_views(q,r,matrices[model])
            finally:close_arrays(q,r)
            np.save(output/("view_cosines_"+model+".npy"),views,allow_pickle=False)
            print("heading replay: "+model+" cosine replay verified; exact all-state six-way MAP starting",flush=True)
            result=core.decode_heading_tracks(views,shared["ref_ids"],shared["ref_latlon"],shared["query_times"],cameras,angles,offsets,vo_yaw_delta_deg=yaw["vo_yaw_delta_deg"],vo_quality=yaw["vo_quality"])
            require(result["profile_sha256"]==core.PROFILE_SHA256 and set(result["outputs"])=={v+"/"+m for v in core.PROFILE["variants"] for m in core.PROFILE["modes"]},"decoder omitted a declared output")
            write(output/("predictions_"+model+".json"),result)
            for name,value in result["outputs"].items():write(output/("prediction_"+model+"_"+name.replace("/","_")+".json"),value);completed.append(model+"/"+name)
            stats[model]={"elapsed_s":time.monotonic()-begin,"view_cosines_sha256":core._array_sha(views),"allocation":result["allocation"],"memory_before":memory}
            del views,result
            print("heading replay: "+model+" all six paths complete",flush=True)
        require(completed==PROFILE["outputs"],"not all twelve paths finished")
        source_identity(expected_commit,sources);freeze(family/"inputs_frozen.json");original_inputs(declared);base.verify_originals(lineage["copied"])
        require(all(sha(ROOT/n)==d for n,d in actual_sources.items()),"source bytes changed")
        require("torch" not in sys.modules,"model runtime imported during model-free replay")
        write(output/"runtime.json",{"status":"complete","source_commit":expected_commit,"source_sha256":sources,"profile_sha256":PROFILE_SHA256,"input_freeze_sha256":sha(family/"inputs_frozen.json"),
            "input_identity":identity,"models":stats,"elapsed_s":time.monotonic()-started,"memory_before":before,"memory_end":check_headroom(),"new_forwards":False,"labels_opened":False,"scoring_called":False,"selected_variant":None})
        write(output/"frozen_predictions.json",{"status":"complete","scoring_has_not_run":True,"all_requested_outputs":completed,"source_commit":expected_commit,"profile_sha256":PROFILE_SHA256,
            "selected_variant":None,"input_freeze_sha256":sha(family/"inputs_frozen.json"),"artifacts":{str(p.relative_to(output)):sha(p) for p in output.rglob("*") if p.is_file()}})
        return {"status":"complete","outputs":12,"freeze_sha256":sha(output/"frozen_predictions.json")}
    except Exception as error:
        write(output/"failure.json",{"status":"failed","completed_outputs":completed,"error":type(error).__name__+": "+str(error),"scoring_has_not_run":True});raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    setup=sub.add_parser("prepare");setup.add_argument("--spec",type=Path,required=True);run=sub.add_parser("infer")
    run.add_argument("--input-freeze-sha256",required=True)
    for p in (setup,run):p.add_argument("--family",type=Path,required=True);p.add_argument("--expected-commit",required=True)
    args=vars(parser.parse_args(argv));command=args.pop("command");print(json.dumps(prepare(**args) if command=="prepare" else infer(**args),indent=2))
if __name__=="__main__":main()
