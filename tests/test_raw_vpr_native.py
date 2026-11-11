"""Private synthetic capture/files only: no cases, models, real video or GT."""
import copy
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from monocular_osm import raw_vpr_native as r
from scripts import replay_raw_vpr_native as cli


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return {"path": str(path), "sha256": r.sha(path)}


class Capture:
    def __init__(self, shifted=False): self.index, self.shifted = 0, shifted
    def isOpened(self): return True
    def release(self): pass
    def set(self, prop, value): self.index = int(value); return True
    def read(self):
        if self.index >= 161: return False, None
        image = np.full((8, 12, 3), self.index % 255, np.uint8)
        self.index += 1
        return True, image
    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 10., cv2.CAP_PROP_FRAME_COUNT: 161,
                cv2.CAP_PROP_POS_FRAMES: self.index + int(self.shifted),
                cv2.CAP_PROP_POS_MSEC: (self.index - 1) * 100.}[prop]


@pytest.fixture(scope="module")
def genuine_grid():
    native = r.ROOT / "output/assisted_phoenix_v3_input_audit_20261028/audit_inference.py"
    return r.load_native_grid(r.Ledger(), {"path": str(native), "sha256": r.NATIVE_SHA},
                             {"path": str(native.with_name("audit_prepared.py")), "sha256": r.DEPENDENCY_SHA})


def decoder(genuine, events=None):
    def grid(path, cfg, *, check=lambda: None):
        if events is not None: events.append("decode")
        return genuine(path, cfg, check=check, capture_factory=lambda _: Capture())
    return grid


def header(a): return {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": r.typed(a)}


def fixture(tmp, genuine, sparse=True):
    path = tmp / "private.fake"; path.write_bytes(b"private fake encoded source")
    video = {"path": str(path), "sha256": r.sha(path), "size": path.stat().st_size}
    cfg = dict(vo_start_sec=0., vo_end_sec=16.1, frame_stride=2, max_frames=None, frame_max_width=None,
               retrieval_only=sparse, answer="track", use_vpr_prior=True, use_ipm_scale=False, output_dir=str(tmp / "child"))
    qg = decoder(genuine)(video["path"], cfg)
    q = {"query_indices": np.arange(80, dtype=np.int64) if sparse else np.asarray(qg["dense_indices"], np.int64),
         "query_times": qg["query_times"]}
    vg = decoder(genuine)(video["path"], {**cfg, "frame_stride": 1})
    fx = 6. / np.tan(np.deg2rad(70.) / 2.)
    vo = {"timestamps": vg["timestamps"], "K": np.array([[fx, 0., 6.], [0., fx, 4.], [0., 0., 1.]])}
    sig = dict(schema=2, vo_algo_version=3, video={"path": video["path"], "size": video["size"]},
               extraction=dict(start_sec=0., end_sec=16.1, stride=1, max_frames=None, frame_max_width=None,
                               n_frames=161, fps=10., total_frames=161,
                               timestamps_sha256=hashlib.sha256(vo["timestamps"].astype("<f8").tobytes()).hexdigest()),
               frames_sha256=vg["frames_sha256"])
    request = dict(kind="expanded", execution_id="private", video_sha256=video["sha256"], config=cfg,
                   source_sha256={"monocular_osm/assisted_worker.py": "a" * 64})
    rr = write(tmp / "child/request.json", request)
    rows, calls, protected, objects = [], [], [], {}
    return fixture_rows(tmp, cfg, qg, q, rr, rows, calls, protected, objects, video, vo, sig, request)


def fixture_rows(tmp, cfg, qg, q, rr, rows, calls, protected, objects, video, vo, sig, request):
    for role in ("retrieval_audit/0000", "metadata_admission/retrieval_audit/0000"):
        path = tmp / "child" / (role + ".npz"); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **q, raw_similarities=np.array([object()], dtype=object))
        ar = {"path": str(path), "sha256": r.sha(path)}
        app = dict(backbone={"name": "megaloc"}, actual_model_state_sha256="b" * 64, preprocessing={"warp_size": [512,512]},
                   query_mode="warp", query_pixels_sha256=qg["query_pixels_sha256" if cfg["retrieval_only"] else "dense_query_pixels_sha256"],
                   query_descriptors_sha256="c" * 64)
        identity = dict(**app, score_kind="raw_cosine_at_decoded_reference", decoder={"actual": "viterbi"},
                        query_indices_sha256=r.typed(q["query_indices"]), query_times_sha256=r.typed(q["query_times"]))
        headers = {k: header(v) for k,v in q.items()}
        meta = dict(**app, score_kind=identity["score_kind"], decoder=identity["decoder"], status="complete", issues=[],
                    array_artifact={**ar, "arrays": headers}, comparison_identity=identity, comparison_identity_sha256=r.digest(identity))
        mr = write(path.with_suffix(".json"), meta)
        rows.append(dict(role=role, metadata=mr["path"], metadata_sha256=mr["sha256"], npz=ar["path"], npz_sha256=ar["sha256"],
                         issues=[], raw_headers=headers, appearance_identity={**app, "query_indices_header": header(q["query_indices"]),
                                                                            "query_times_header": header(q["query_times"])}))
        calls.append(dict(key=r.call_key("expanded",role), request_ref=rr, metadata_ref=mr, array_ref=ar))
        protected.extend([mr, ar]); objects[r.key(mr["path"])] = meta
    owner = write(tmp / "child/child_frozen.json", {"artifacts": {
        str(Path(v["path"]).relative_to(tmp/"child")): v["sha256"] for v in [rr,*protected]}})
    objects[r.key(owner["path"])] = json.loads(Path(owner["path"]).read_text(encoding="utf-8"))
    inv = dict(schema="closed-raw-vpr-metadata-bank-inventory-v1", banks=[dict(key="expanded",child="expanded",pair_records=rows)], protected_files=protected)
    bindings = dict(schema="root-exact-raw-bank-own-request-bindings-v1", all_inventory_rows_bound=2,
                    calls=[{**c,"owner_freeze_ref":owner} for c in calls], protected_files=[rr,owner,*protected])
    plan = r.make_plan(inv,calls,bindings,lambda _:request,lambda v:objects[r.key(v["path"])])
    return inv,calls,bindings,plan,video,vo,sig,q,request,objects


def test_sparse_prefix_is_local_row_not_vo_or_dense_index(tmp_path, genuine_grid):
    _, calls, _, plan, video, vo, sig, q, _, _ = fixture(tmp_path,genuine_grid)
    events=[]
    facts=r.replay(plan,video,vo,sig,lambda _:(events.append("grid") or q),decoder(genuine_grid,events),max_lag_s=1.)
    assert events==["grid","grid","decode","decode"]
    row=facts["all_physical_calls_and_aliases"][0]["query_to_vo"][-1]
    assert (row["local_query_index"],row["dense_index"],row["native_ordinal"],row["vo_exact_index"])==(79,80,160,160)
    assert r.validate_native_facts(facts,calls,"expanded")==plan["appearance_identity"]
    assert facts["unique_native_decodes"]==2 and facts["reference_encoding_upgraded"] is False


def test_dense_prefix_uses_selected_dense_index(tmp_path,genuine_grid):
    _,_,_,plan,video,vo,sig,q,_,_=fixture(tmp_path,genuine_grid,False)
    facts=r.replay(plan,video,vo,sig,lambda _:q,decoder(genuine_grid),max_lag_s=1.)
    assert facts["all_physical_calls_and_aliases"][0]["query_to_vo"][-1]["local_query_index"]==80


@pytest.mark.parametrize("fault",["subset","extra","request","appearance","issues","resize","cap","comparison","owner"])
def test_metadata_own_request_and_alias_gates(tmp_path,genuine_grid,fault):
    inv,calls,bindings,_,_,_,_,_,request,objects=fixture(tmp_path,genuine_grid)
    if fault=="subset":calls.pop()
    if fault=="extra":calls.append(copy.deepcopy(calls[0]))
    if fault=="request":calls[-1]["request_ref"]={"path":str(tmp_path/"foreign"),"sha256":"f"*64}
    if fault=="appearance":inv["banks"][0]["pair_records"][-1]["appearance_identity"]["actual_model_state_sha256"]="f"*64
    if fault=="issues":inv["banks"][0]["pair_records"][-1]["issues"]=["incomplete"]
    if fault=="resize":request["config"]["frame_max_width"]=960
    if fault=="cap":request["config"]["max_frames"]=400
    if fault=="comparison":objects[r.key(calls[-1]["metadata_ref"]["path"])]["comparison_identity_sha256"]="0"*64
    if fault=="owner":objects[r.key(bindings["calls"][0]["owner_freeze_ref"]["path"])]["artifacts"].pop("request.json")
    with pytest.raises(ValueError):r.make_plan(inv,calls,bindings,lambda _:request,lambda v:objects[r.key(v["path"])])


@pytest.mark.parametrize("fault",["typed","pixel","vo_clock","vo_pixels","K","video"])
def test_pixels_vo_camera_mismatches_refuse(tmp_path,genuine_grid,fault):
    _,_,_,plan,video,vo,sig,q,_,_=fixture(tmp_path,genuine_grid)
    if fault=="typed":q["query_indices"]=q["query_indices"].astype(np.int32)
    if fault=="pixel":plan["calls"][-1]["appearance_identity"]["query_pixels_sha256"]="0"*64
    if fault=="vo_clock":sig["extraction"]["timestamps_sha256"]="0"*64
    if fault=="vo_pixels":sig["frames_sha256"]="0"*64
    if fault=="K":vo["K"][0,0]+=.1
    if fault=="video":video["sha256"]="0"*64
    with pytest.raises(ValueError):r.replay(plan,video,vo,sig,lambda _:q,decoder(genuine_grid),max_lag_s=1.)


def test_last_alias_grid_refuses_before_first_decode(tmp_path,genuine_grid):
    _,_,_,plan,video,vo,sig,q,_,_=fixture(tmp_path,genuine_grid)
    loaded=[]
    def grids(_):
        loaded.append(1)
        if len(loaded)==2:raise ValueError("last alias missing")
        return q
    with pytest.raises(ValueError,match="last alias"):
        r.replay(plan,video,vo,sig,grids,lambda *a,**k:pytest.fail("video before all aliases"),max_lag_s=1.)


def test_join_brackets_never_round_actual_time():
    times=np.arange(161,dtype=np.float64)*.1;queries=np.linspace(.025,15.925,80).astype(np.float64)
    rows=r.query_vo_join(queries,times,1.)
    assert rows[0]["t_sec"]==.025 and rows[0]["vo_exact_index"] is None
    assert (rows[0]["vo_left_index"],rows[0]["vo_right_index"])==(0,1)
    assert rows[0]["composed_lag_s"]==.475
    with pytest.raises(ValueError,match="lag exceeds"):r.query_vo_join(queries,times,.01)
    with pytest.raises(ValueError,match="outside signed"):r.query_vo_join(queries+1.,times,1.)


def test_genuine_replay_refuses_shifted_decoder_ordinal(tmp_path,genuine_grid):
    cfg=dict(vo_start_sec=0.,vo_end_sec=16.1,frame_stride=2,max_frames=None,frame_max_width=None)
    with pytest.raises(ValueError,match="ordinal shifted"):
        genuine_grid(str(tmp_path/"private.fake"),cfg,capture_factory=lambda _:Capture(True))


def test_named_bank_no_subset_or_borrowed_source(tmp_path,genuine_grid):
    _,calls,_,plan,video,vo,sig,q,_,_=fixture(tmp_path,genuine_grid)
    facts=r.replay(plan,video,vo,sig,lambda _:q,decoder(genuine_grid),max_lag_s=1.)
    with pytest.raises(ValueError,match="ALL named-bank"):r.validate_native_facts(facts,calls[:1],"expanded")
    changed=copy.deepcopy(calls);changed[-1]["request_ref"]["sha256"]="f"*64
    with pytest.raises(ValueError,match="physical/OWN"):r.validate_native_facts(facts,changed,"expanded")


def test_owner_backslash_normalization_refuses_duplicate_keys(tmp_path,genuine_grid):
    inv,calls,bindings,_,_,_,_,_,request,objects=fixture(tmp_path,genuine_grid)
    owner=objects[r.key(bindings["calls"][0]["owner_freeze_ref"]["path"])]["artifacts"]
    name=next(n for n in owner if "retrieval_audit" in n)
    alternate=name.replace("\\","/") if "\\" in name else name.replace("/","\\")
    owner[alternate]=owner[name]
    with pytest.raises(ValueError,match="duplicate normalized owner"):
        r.make_plan(inv,calls,bindings,lambda _:request,lambda v:objects[r.key(v["path"])])


def test_budget_refuses_before_hash_body_open(tmp_path,monkeypatch):
    path=tmp_path/"private.json";path.write_text("{}",encoding="utf-8")
    ledger=r.Ledger();ledger.total_json=256<<20
    monkeypatch.setattr(r,"sha",lambda *a:pytest.fail("body/hash after exhausted budget"))
    with pytest.raises(ValueError,match="budget before body"):
        ledger.json(dict(path=str(path),sha256="0"*64))


def test_final_callback_mutation_is_caught_by_callback_free_pass(tmp_path):
    a,b=tmp_path/"a",tmp_path/"b";a.write_bytes(b"first");b.write_bytes(b"second")
    ledger=r.Ledger()
    for path in (a,b):ledger.protect(dict(path=str(path),sha256=r.sha(path)))
    def mutate():
        assert a.resolve().is_relative_to(tmp_path.resolve())
        a.write_bytes(b"changed")
    ledger.check=mutate
    with pytest.raises(ValueError,match="protected bytes changed"):ledger.unchanged()


def test_standalone_reads_only_six_allowed_members_and_publishes(tmp_path,genuine_grid,monkeypatch):
    inv,calls,bindings,_,video,vo,sig,_,_,_=fixture(tmp_path,genuine_grid)
    ir=write(tmp_path/"inventory.json",inv);bindings["inventory_ref"]=ir
    br=write(tmp_path/"bindings.json",bindings)
    vp,sp=tmp_path/"vo.npz",tmp_path/"signed.npz"
    np.savez(vp,**vo,centers=np.array([object()],dtype=object))
    np.savez(sp,cache_signature=np.array(json.dumps(sig)),vo_algo_version=np.int64(3),
             rotations=np.array([object()],dtype=object))
    config=dict(schema="raw-vpr-native-replay-config-v1",source_hashes={n:r.sha(r.ROOT/n) for n in r.SOURCE_PATHS},
                inventory=ir,own_request_bindings=br,calls=calls,video=video,
                cues_vo=dict(path=str(vp),sha256=r.sha(vp)),signed_vo_cache=dict(path=str(sp),sha256=r.sha(sp)),
                max_composed_lag_s=1.,output_dir=str(tmp_path/"out"))
    cr=write(tmp_path/"config.json",config)
    opened=[];original=np.lib.npyio.NpzFile.__getitem__
    def getitem(archive,name):
        opened.append(name)
        assert name in {"query_indices","query_times","timestamps","K","cache_signature","vo_algo_version"}
        return original(archive,name)
    monkeypatch.setattr(np.lib.npyio.NpzFile,"__getitem__",getitem)
    facts=cli.execute(cr["path"],cr["sha256"],native_grid=decoder(genuine_grid))
    assert set(opened)=={"query_indices","query_times","timestamps","K","cache_signature","vo_algo_version"}
    assert facts["external_actor_closure_verified"] is False
    assert (tmp_path/"out/native_replay.json").is_file()


def test_native_loader_wrong_pin_refuses_before_execution(tmp_path):
    with pytest.raises(ValueError,match="original native/dependency"):
        r.load_native_grid(r.Ledger(),dict(path=str(tmp_path/"none.py"),sha256="0"*64),
                           dict(path=str(tmp_path/"audit_prepared.py"),sha256=r.DEPENDENCY_SHA))
