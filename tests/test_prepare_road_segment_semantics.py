"""Tiny genuine graph fixtures only; no actual maps, video, GT or NN."""
from pathlib import Path
from copy import deepcopy
import io as bytesio
import json
import zipfile
import numpy as np
import osmnx as ox
import pytest
from pyproj import Transformer
from scripts import prepare_road_segment_semantics as w
from tests.test_junction_inputs import fixture as graph_fixture


def owned(root,path):
    path=Path(path).resolve();assert path.is_relative_to(root.resolve()),"fixture mutation outside tmp"
    return path


def write(root,path,value):
    path=owned(root,path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(w.p.json_bytes(value));return ref(path)


def ref(path):return {"path":str(Path(path).resolve()),"sha256":w.io.sha(path)}


def execute(tmp,config,**kw):
    r=write(tmp,tmp/"config.json",config);return w.execute(r["path"],r["sha256"],**kw)


def setup(tmp,monkeypatch,variant="basic"):
    graph,inventory,_,_=graph_fixture()
    for _,_,d in graph.edges(data=True):d.update(highway="primary",maxspeed="50 mph",oneway=True,reversed=False)
    if variant=="reverse":
        for u,v,k,d in list(graph.edges(keys=True,data=True)):graph.add_edge(v,u,key=k+9,**{**d,"reversed":True})
    if variant=="parallel":graph.add_edge(1,2,key=7,osmid=11,length=100.,highway="service",maxspeed="30",oneway=True,reversed=False)
    if variant=="longtag":graph.edges[1,2,0]["name"]="x"*9000
    tf=Transformer.from_crs(graph.graph["crs"],"EPSG:4326",always_xy=True)
    for _,d in graph.nodes(data=True):d["x"],d["y"]=tf.transform(d["x"],d["y"])
    graph.graph["crs"]="EPSG:4326"
    path=owned(tmp,tmp/"raw.graphml");ox.save_graphml(graph,path)
    clues=w.p.dependencies()[-1].extract_road_sign_clues([{"text":"EXIT 12A","confidence":.99,"t_sec":1.,"bbox":[100.,100.,180.,130.]}],[480,640])
    roles={"raw_graphml":ref(path),"junction_inventory":write(tmp,tmp/"inventory.json",inventory),
        "temporal_ocr":write(tmp,tmp/"ocr.json",{"clues":clues,"origin_video_sha256":"a"*64})}
    study=write(tmp,tmp/"study.json",{"schema":"root-prospective-original-segment-semantics-preparation-study-v1","inputs":roles,
        "speed_model_implemented":False,"ground_truth_used":False,"policy_admission":False})
    monkeypatch.setattr(w,"SEGMENT_SEMANTICS_STUDY_REF",study)
    return {"schema":"road-segment-semantics-prepare-v1","source_hashes":{n:w.io.sha(w.ROOT/n)for n in w.SOURCE_PATHS},
        "inputs":roles,"study_registration_ref":study,"dependency_source_refs":w.dependency_refs(),"output_dir":str(tmp/"new")}


@pytest.mark.parametrize("variant",["basic","reverse","parallel"])
def test_genuine_end_to_end_all18_and_full_old_outputs_equal(tmp_path,monkeypatch,variant):
    config=setup(tmp_path,monkeypatch,variant)
    old={k:deepcopy(v)for k,v in config.items()if k not in {"study_registration_ref","dependency_source_refs"}}
    old.update(schema="junction-sequence-probe-prepare-v1",output_dir=str(tmp_path/"old"))
    r=write(tmp_path,tmp_path/"old_config.json",old);w.p.prepare(r["path"],r["sha256"])
    master=execute(tmp_path,config);new=Path(config["output_dir"])
    assert set(master["artifacts"])==w.GENERATED and {p.name for p in new.iterdir()}==w.GENERATED|{"frozen_prepare.json"}
    with np.load(new/"network.npz",allow_pickle=False)as a,np.load(tmp_path/"old/network.npz",allow_pickle=False)as b:
        assert set(a.files)==set(b.files)==set(w.p.NET_ARRAYS)
        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    for name in ("network.json","matched.json","topology_receipt.json","plan.json"):
        assert w.io.read_json(new/name)==w.io.read_json(tmp_path/"old"/name)
    packet,metadata=w.io.read_packet(new)
    rows=list(w.io.iter_catalog(new/"raw_edge_catalog.jsonl"))
    assert len(rows)==len(packet["arrays"]["raw_edge_keys"])
    np.testing.assert_array_equal([r["key"]for r in rows],packet["arrays"]["raw_edge_keys"])
    assert metadata["loaded_graph_artifact_only"] is True and metadata["original_OSM_or_acquisition_history_verified"] is False
    assert master["speed_model_implemented"] is master["ground_truth_used"] is master["policy_admission"] is False
    assert master["selected_answer"] is None and master["all18_old_arrays_compared_to_prior_by_worker"] is False
    for n,h in master["artifacts"].items():assert w.io.sha(new/n)==h


def test_output_reserved_before_first_payload_body(tmp_path,monkeypatch):
    config=setup(tmp_path,monkeypatch)
    # Observe the lower IO payload call without replacing the genuine helper.
    original_read=w.io.read_json
    def read(path):
        if Path(path)==Path(config["inputs"]["junction_inventory"]["path"]):assert Path(config["output_dir"]).is_dir()
        return original_read(path)
    monkeypatch.setattr(w.io,"read_json",read)
    execute(tmp_path,config)


@pytest.mark.parametrize("fault",["source","dependency_hash","dependency_origin","dependency_missing","study_ref","study_schema","study_inputs","knob","origin"])
def test_sources_dependencies_and_study_precede_payload(tmp_path,monkeypatch,fault):
    config=setup(tmp_path,monkeypatch)
    if fault=="source":config["source_hashes"]["monocular_osm/road_segment_semantics_io.py"]="f"*64
    if fault=="dependency_hash":config["dependency_source_refs"]["osmnx.io"]["sha256"]="f"*64
    if fault=="dependency_origin":config["dependency_source_refs"]["osmnx.io"]=config["dependency_source_refs"]["osmnx.graph"]
    if fault=="dependency_missing":config["dependency_source_refs"].pop("osmnx.io")
    if fault=="study_ref":config["study_registration_ref"]={**config["study_registration_ref"],"sha256":"f"*64}
    if fault in ("study_schema","study_inputs"):
        study=w.io.read_json(config["study_registration_ref"]["path"])
        if fault=="study_schema":study["schema"]="old-study"
        else:study["inputs"]["raw_graphml"]["sha256"]="f"*64
        config["study_registration_ref"]=write(tmp_path,config["study_registration_ref"]["path"],study)
        monkeypatch.setattr(w,"SEGMENT_SEMANTICS_STUDY_REF",config["study_registration_ref"])
    if fault=="knob":config["speed_model"]="urban"
    if fault=="origin":config["origin_video_sha256"]="f"*64
    original=w.io.read_json
    def no_payload(path):
        if Path(path)in [Path(config["inputs"][k]["path"])for k in ("junction_inventory","temporal_ocr")]:
            if fault!="origin":pytest.fail("payload before source/study/dependency gate")
        return original(path)
    monkeypatch.setattr(w.io,"read_json",no_payload)
    with pytest.raises((ValueError,KeyError)):execute(tmp_path,config)
    if fault!="origin":assert not Path(config["output_dir"]).exists()
    assert not(Path(config["output_dir"])/"frozen_prepare.json").exists()


@pytest.mark.parametrize("kind",["source_input","generated","extra"])
def test_last_callback_mutations_refuse_master(tmp_path,monkeypatch,kind):
    config=setup(tmp_path,monkeypatch);out=Path(config["output_dir"]);fired=[]
    def check():
        if fired or not(out/"plan.json").exists():return
        target=config["inputs"]["junction_inventory"]["path"]if kind=="source_input"else out/("segment_semantics.npz"if kind=="generated"else"extra.txt")
        owned(tmp_path,target).write_bytes(b"private final-boundary mutation");fired.append(True)
    with pytest.raises(ValueError):execute(tmp_path,config,check=check)
    assert fired and not(out/"frozen_prepare.json").exists()


@pytest.mark.parametrize("fault",["record","catalog"])
def test_unavailable_capacity_retains_partial_without_authority(tmp_path,monkeypatch,fault):
    config=setup(tmp_path,monkeypatch,"longtag"if fault=="record"else"basic")
    if fault=="catalog":monkeypatch.setattr(w.io,"MAX_CATALOG",1)
    with pytest.raises(ValueError):execute(tmp_path,config)
    assert Path(config["output_dir"]).exists()and not(Path(config["output_dir"])/"frozen_prepare.json").exists()


@pytest.mark.parametrize("data",[b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}',b'{"a":{"b":1,"b":2}}'])
def test_duplicate_and_nonfinite_json_refused(data):
    with pytest.raises(ValueError):w.io.strict_json(data)


def tiny_arrays():
    return {"raw_edge_keys":np.array([[1,2,0],[2,3,0]],np.int64),"compact_edge_keys":np.array([[1,3,0]],np.int64),
        "compact_path_indptr":np.array([0,2],np.int64),"compact_path_raw_edge_index":np.array([0,1],np.int64),
        "segment_raw_edge_index":np.array([0,1],np.int64),"segment_semantic_status":np.array([1,1],np.uint8),"raw_edge_disposition":np.array([1,1],np.uint8)}


def archive(root,path,arrays,*,replacement=None,extra=None):
    path=owned(root,path)
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED)as z:
        for name,a in arrays.items():
            b=bytesio.BytesIO();np.save(b,a,allow_pickle=False)
            z.writestr(name+".npy",replacement if replacement is not None and name==list(arrays)[-1]else b.getvalue())
        if extra is not None:z.writestr(*extra)
    return path


def test_semantic_archive_exact_typed_roundtrip(tmp_path):
    arrays=tiny_arrays();headers=w.io.validate_arrays(arrays)
    path=archive(tmp_path,tmp_path/"sidecar.npz",arrays)
    actual=w.io.load_arrays(path,headers)
    for k in arrays:np.testing.assert_array_equal(actual[k],arrays[k])


def test_all_zip_caps_before_first_numpy_header(tmp_path,monkeypatch):
    arrays=tiny_arrays();headers=w.io.validate_arrays(arrays)
    path=archive(tmp_path,tmp_path/"bomb.npz",arrays,replacement=b"x"*70_000)
    monkeypatch.setattr(w.io,"_npy_header",lambda *a:pytest.fail("header read before ALL ZipInfo caps"))
    with pytest.raises(ValueError,match="ALL semantic ZIP"):w.io.load_arrays(path,headers)


@pytest.mark.parametrize("fault",["object","dtype","shape","extra","typed_drift","payload_tail"])
def test_semantic_archive_no_object_shape_extras_or_trailing_payload(tmp_path,fault):
    arrays=tiny_arrays();headers=w.io.validate_arrays(arrays)
    if fault=="object":arrays['raw_edge_disposition']=np.array([object(),object()],object)
    if fault=="dtype":arrays['raw_edge_disposition']=arrays['raw_edge_disposition'].astype(np.int32)
    if fault=="shape":arrays['raw_edge_disposition']=np.ones(3,np.uint8)
    if fault=="typed_drift":arrays['raw_edge_disposition'][0]=2
    if fault=="object":
        with pytest.raises(ValueError):w.io.validate_arrays(arrays)
        return
    replacement=None
    if fault=="payload_tail":
        b=bytesio.BytesIO();np.save(b,arrays[list(arrays)[-1]]);replacement=b.getvalue()+b'junk'
    path=archive(tmp_path,tmp_path/"bad.npz",arrays,replacement=replacement,extra=('extra.npy',b'')if fault=="extra"else None)
    with pytest.raises(ValueError):w.io.load_arrays(path,headers)


@pytest.mark.parametrize("fault",["record","count","file"])
def test_catalog_all_budgets_before_first_json_decode(tmp_path,monkeypatch,fault):
    config=setup(tmp_path,monkeypatch);execute(tmp_path,config)
    path=Path(config["output_dir"])/"raw_edge_catalog.jsonl"
    if fault=="record":owned(tmp_path,path).write_bytes(path.read_bytes()+b'x'*9000+b'\n')
    if fault=="count":monkeypatch.setattr(w.io,"MAX_RAW",1)
    if fault=="file":monkeypatch.setattr(w.io,"MAX_CATALOG",1)
    monkeypatch.setattr(w.io,"strict_json",lambda *a:pytest.fail("decode before full preflight"))
    with pytest.raises(ValueError):list(w.io.iter_catalog(path))


def test_npy_header_length_before_header_body_read():
    from types import SimpleNamespace
    calls=[]
    class Stream:
        def read(self,n):
            calls.append(n)
            return b'\x93NUMPY\x02\x00'if n==8 else (1_000_000).to_bytes(4,'little')
    with pytest.raises(ValueError,match="header cap"):w.io._npy_header(Stream(),SimpleNamespace(file_size=2_000_000))
    assert calls==[8,4]


@pytest.mark.parametrize("fault",["dtype","oversize","object","shape"])
def test_declared_array_header_caps_before_archive_open(tmp_path,monkeypatch,fault):
    headers=w.io.validate_arrays(tiny_arrays())
    if fault=="dtype":headers['raw_edge_keys']['dtype']='>i8'
    if fault=="object":headers['raw_edge_keys']['dtype']='|O'
    if fault=="oversize":headers['raw_edge_keys']['shape']=[500001,3]
    if fault=="shape":headers['raw_edge_keys']['shape']=[2,3,1]
    monkeypatch.setattr(w.io.zipfile,'ZipFile',lambda *a:pytest.fail('archive before header cap'))
    with pytest.raises(ValueError):w.io.load_arrays(tmp_path/'never-open.npz',headers)


@pytest.mark.parametrize("fault",["duplicate","nonfinite","index","key"])
def test_catalog_schema_and_keys_refuse_bad_record(tmp_path,monkeypatch,fault):
    config=setup(tmp_path,monkeypatch);execute(tmp_path,config);out=Path(config['output_dir'])
    path=out/'raw_edge_catalog.jsonl';lines=path.read_bytes().splitlines(keepends=True)
    if fault=="duplicate":lines[0]=lines[0].replace(b'{',b'{"index":999,',1)
    elif fault=="nonfinite":lines[0]=lines[0].replace(b'"index":0',b'"index":NaN')
    else:
        row=json.loads(lines[0]);row['index'if fault=='index'else'key']=1 if fault=='index'else[9,9,9]
        lines[0]=w.io.line_bytes(row)
    owned(tmp_path,path).write_bytes(b''.join(lines))
    if fault=='key':
        meta=w.io.read_json(out/'segment_semantics.json');meta['catalog'].update(bytes=path.stat().st_size,sha256=w.io.sha(path))
        meta['core_metadata'].update(catalog_bytes=path.stat().st_size,catalog_sha256=w.io.sha(path))
        owned(tmp_path,out/'segment_semantics.json').write_bytes(w.p.json_bytes(meta))
        with pytest.raises(ValueError,match='keys'):w.io.read_packet(out)
    else:
        with pytest.raises(ValueError):list(w.io.iter_catalog(path))


@pytest.mark.parametrize("target",["segment_semantics.json","raw_edge_catalog.jsonl"])
def test_reader_last_callback_cannot_mutate_previous_role(tmp_path,monkeypatch,target):
    config=setup(tmp_path,monkeypatch);execute(tmp_path,config);out=Path(config["output_dir"]);fired=[]
    original=w.io.load_arrays
    def loaded(*a,**k):
        value=original(*a,**k);fired.append(True);return value
    monkeypatch.setattr(w.io,"load_arrays",loaded);changed=[]
    def check():
        if fired and not changed:owned(tmp_path,out/target).write_bytes(b"private late reader mutation");changed.append(True)
    with pytest.raises(ValueError,match="changed during read"):w.io.read_packet(out,check=check)
    assert changed
