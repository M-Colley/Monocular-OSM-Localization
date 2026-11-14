"""Separate source-bound original-segment sidecar prepare worker; no model/GT."""
from pathlib import Path
import argparse
import importlib
import inspect
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import junction_sequence_probe as p
from monocular_osm import road_segment_semantics as core
from monocular_osm import road_segment_semantics_io as io
SOURCE_PATHS=frozenset(p.SOURCE_PATHS)|{"scripts/prepare_road_segment_semantics.py","monocular_osm/road_segment_semantics.py","monocular_osm/road_segment_semantics_io.py"}
DEPENDENCY_MODULES=("osmnx","osmnx.simplification","osmnx.projection","osmnx.io","osmnx.graph","osmnx.settings","osmnx.convert","osmnx.utils","networkx","pyproj","numpy","shapely","geopandas")
SEGMENT_SEMANTICS_STUDY_REF={"path":"E:/monocular-osm-joint-vpr-proof/run18/segment_semantics_study_contract.json","sha256":"f6fbdc57e1e2de60932a59d7023b7f09030c16ab8ed9aae7e68ba1f62c3e4d11"}
GENERATED=frozenset(("network.npz","network.json","matched.json","topology_receipt.json","plan.json","raw_edge_catalog.jsonl","segment_semantics.npz","segment_semantics.json"))
INPUTS=frozenset(("raw_graphml","junction_inventory","temporal_ocr"))


def role(ref,base):
    io.require(isinstance(ref,dict)and set(ref)=={"path","sha256"},"exact path/SHA role")
    h=ref["sha256"];io.require(isinstance(h,str)and len(h)==64 and all(c in "0123456789abcdef"for c in h),"expected SHA256 required")
    path=Path(ref["path"]);path=(base/path if not path.is_absolute()else path).resolve(strict=True)
    io.require(path.is_file(),"regular input/source file required")
    return path,h


class Ledger:
    def __init__(self,base,check):self.base=Path(base);self.check=check;self.protected={}
    def add(self,ref,cap):
        self.check();path,h=role(ref,self.base)
        io.require(path.stat().st_size<=cap,"role byte cap before body")
        io.require(io.sha(path)==h,"expected input/source bytes differ")
        io.require(self.protected.setdefault(str(path),h)==h,"conflicting source/input SHA")
        return path
    def json(self,ref):return io.read_json(self.add(ref,io.MAX_JSON))


def dependency_refs():
    return {name:{"path":str(Path(importlib.import_module(name).__file__).resolve()),"sha256":io.sha(importlib.import_module(name).__file__)}for name in DEPENDENCY_MODULES}


def admit_sources(config,ledger):
    hashes=config["source_hashes"]
    io.require(isinstance(hashes,dict)and SOURCE_PATHS<=set(hashes),"full necessary source closure required")
    for name,h in hashes.items():
        path=(ROOT/name).resolve(strict=True)
        io.require(not Path(name).is_absolute()and path.is_relative_to(ROOT)and path.suffix==".py","source outside executing checkout")
        ledger.add({"path":str(path),"sha256":h},io.MAX_JSON)
    for module in (p,core,io):
        io.require(Path(module.__file__).resolve()==ROOT/(module.__name__.replace(".","/")+".py"),"foreign worker/core/IO origin")
    supplied=config["dependency_source_refs"]
    io.require(isinstance(supplied,dict)and set(supplied)==set(DEPENDENCY_MODULES),"exact installed dependency source roles required")
    prefix=Path(sys.prefix).resolve()
    modules={}
    for name in DEPENDENCY_MODULES:
        module=importlib.import_module(name);actual=Path(module.__file__).resolve()
        expected=ledger.add(supplied[name],io.MAX_JSON)
        io.require(actual==expected and actual.is_relative_to(prefix),"installed dependency origin differs; no checkout retarget")
        modules[name]=module
    ox=modules["osmnx"]
    for function,name in ((ox.load_graphml,"osmnx.io"),(ox.simplify_graph,"osmnx.simplification"),(ox.project_graph,"osmnx.projection")):
        io.require(Path(inspect.getsourcefile(function)).resolve()==Path(modules[name].__file__).resolve(),"foreign critical installed helper")
    return {name:getattr(module,"__version__",None)for name,module in modules.items()}


def equal_ref(a,b,base):
    pa,ha=role(a,base);pb,hb=role(b,base)
    io.require(pa==pb and ha==hb,"registered input/study role differs")


def execute(config_path,expected,*,check=lambda:None):
    started=time.monotonic()
    def poll():
        check();io.require(time.monotonic()-started<=1800.,"complete semantic prepare exceeded inner deadline")
    config_path=Path(config_path).resolve(strict=True);ledger=Ledger(config_path.parent,poll)
    cfgref={"path":str(config_path),"sha256":expected};config=ledger.json(cfgref)
    required={"schema","source_hashes","inputs","study_registration_ref","dependency_source_refs","output_dir"}
    io.require(isinstance(config,dict)and required<=set(config)<=required|{"clues_member","origin_video_sha256"}
        and config["schema"]=="road-segment-semantics-prepare-v1"and set(config["inputs"])==INPUTS,"fixed semantic prepare configuration")
    versions=admit_sources(config,ledger)
    equal_ref(config["study_registration_ref"],SEGMENT_SEMANTICS_STUDY_REF,ledger.base)
    study=ledger.json(config["study_registration_ref"])
    io.require(study.get("schema")=="root-prospective-original-segment-semantics-preparation-study-v1"
        and study.get("speed_model_implemented")is False and study.get("ground_truth_used")is False
        and study.get("policy_admission")is False,"prospective sidecar-only study required")
    io.require(set(study["inputs"])==INPUTS,"registered complete input roles required")
    for key in INPUTS:equal_ref(config["inputs"][key],study["inputs"][key],ledger.base)
    _,inputs,_,_,_,clues_core=p.dependencies()
    io.require(Path(inputs.prepare_graph_with_semantics.__code__.co_filename).resolve()==ROOT/"monocular_osm/junction_inputs.py","foreign semantic graph wrapper")
    output=Path(config["output_dir"]);output=(ledger.base/output if not output.is_absolute()else output).resolve()
    input_paths=[role(ref,ledger.base)[0]for ref in config["inputs"].values()]
    io.require(not any(path.is_relative_to(output)for path in [*input_paths,*map(Path,ledger.protected)]),"output would contain protected input/source")
    output.mkdir(parents=True,exist_ok=False)  # Source/dependency/study gates above; payload bodies below.
    paths={key:ledger.add(ref,1<<30 if key=="raw_graphml"else io.MAX_JSON)for key,ref in config["inputs"].items()}
    inventory=io.read_json(paths["junction_inventory"]);temporal=io.read_json(paths["temporal_ocr"])
    member=config.get("clues_member","clues");io.require(member is None or member=="clues","literal clues member required")
    clues=temporal if member is None else temporal[member]
    origin=config.get("origin_video_sha256")if member is None else temporal.get("origin_video_sha256")
    io.require(isinstance(clues,dict)and isinstance(origin,str)and len(origin)==64 and all(c in "0123456789abcdef"for c in origin),"literal clue/video-origin provenance required")
    if "origin_video_sha256"in config:io.require(config["origin_video_sha256"]==origin,"temporal/config video origin differs")
    plan=inputs.plan_inputs(inventory["plan"]["city"],inventory["plan"]["bbox_wsen"],bbox_source=inventory["plan"]["bbox_source"])
    matched=inputs.annotate_correlations(clues_core.match_motorway_junctions(clues,inventory))
    import osmnx as ox
    poll();owned={"raw":ox.load_graphml(paths["raw_graphml"])}
    writer=io.CatalogWriter(output/"raw_edge_catalog.jsonl",check=poll)
    try:
        net,topology,packet=inputs.prepare_graph_with_semantics(owned.pop("raw"),inventory,matched,plan,emit_record=writer.emit,copy_graph=False)
        catalog_info=writer.finish()
    finally:writer.close()

    poll()
    io.require(set(packet)=={"arrays","metadata"}and packet["metadata"].get("profile_sha256")==core.PROFILE_SHA256,"genuine pure semantic packet required")
    import numpy as np
    io.require(np.array_equal(packet["arrays"]["compact_edge_keys"],np.asarray(net.edge_keys,np.int64))
        and len(packet["arrays"]["segment_raw_edge_index"])==len(net.seg_gstart),"sidecar/old network binding differs")
    p.save_network(output,net);del net
    network_metadata=io.read_json(output/"network.json")
    io.write_packet(output,packet,catalog_info,network_metadata);del packet
    p.write_json(output/"matched.json",matched);p.write_json(output/"topology_receipt.json",topology);p.write_json(output/"plan.json",plan)
    generated={name:io.sha(output/name)for name in GENERATED}
    master={"schema":"road-segment-semantics-prepare-freeze-v1","status":"PREPARED_ROAD_SEGMENT_SEMANTICS_INPUTS_NO_ACCURACY",
        "config_ref":cfgref,"source_hashes":config["source_hashes"],"dependency_source_refs":config["dependency_source_refs"],
        "dependency_versions":versions,"study_registration_ref":config["study_registration_ref"],"artifacts":generated,
        "protected_hashes":dict(ledger.protected),"semantic_profile_sha256":core.PROFILE_SHA256,"catalog":catalog_info,
        "origin_video_sha256":origin,"plan_sha256":plan["plan_sha256"],"preparation_worker_must_exit_before_run":True,
        "loaded_graph_artifact_only":True,"original_OSM_or_acquisition_history_verified":False,
        "all18_old_arrays_compared_to_prior_by_worker":False,"semantic_root_acceptance_required_before_future_consumer":True,
        "speed_model_implemented":False,"motion_model_changed":False,"ground_truth_used":False,"policy_admission":False,
        "selected_answer":None,"accuracy_admitted":False,"external_actor_closure_verified":False,
        "resource_enforcement":"Root CPU2/RSS3GiB/headroom8GiB/outer1830 guard required; inner1800"}
    poll()  # LAST callback. No callbacks after exact membership and expected byte checks.
    io.require({path.name for path in output.iterdir()}==GENERATED,"unexpected semantic preparation membership")
    p.verify_files({**ledger.protected,**{str(output/name):h for name,h in generated.items()}})
    p.write_json(output/"frozen_prepare.json",master)
    return master


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True);parser.add_argument("--config-sha256",required=True)
    args=parser.parse_args(argv);execute(args.config,args.config_sha256)


if __name__=="__main__":main()
