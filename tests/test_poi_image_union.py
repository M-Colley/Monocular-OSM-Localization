"""Model-free contracts and full-union sequence behavior."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from monocular_osm import poi_image_union as union
from monocular_osm.full_sequence import decode_full_tracks


VIDEO = "a" * 64


def inputs(count=2):
    branches = []
    for index in range(count):
        identifiers = [f"node/{2*index}", f"node/{2*index+1}"]
        branches.append({"id": f"branch_{index}", "poi_ids": identifiers,
                         "pois": [{"id": key, "lat": 50 + index*.00001 + j*.000002,
                                   "lon": 10.} for j, key in enumerate(identifiers)]})
    clues = {"selected_answer": None, "confidence_calibrated": False,
             "detections_sha256": "b"*64, "candidate_clusters": branches,
             "inventory": {"city": "Synthetic municipality", "scope": "named_city_bbox", "snapshot_sha256": "c"*64}}
    native = {"source": "same_video_native_frames", "video_sha256": VIDEO,
              "native_bbox_grid": True, "sample_count": 24, "frame_shape_hw": [720, 1280],
              "min_len": 2, "min_confidence": .3, "super_res": False, "use_gpu": False,
              "position_stamp_filter": "scene_text.is_position_stamp", "position_stamps_rejected": True,
              "window": [0., 160.]}
    return clues, native


def evidence(region, ids=None, scores=None, *, timestamps=None):
    timestamps = np.arange(80, dtype=float) if timestamps is None else np.asarray(timestamps, float)
    indices = np.linspace(0, len(timestamps)-1, 80).astype(np.int64)
    times = timestamps[indices]
    ids = ["mapillary:a"] if ids is None else list(ids)
    if scores is None:
        scores = np.full((80, len(ids)), .2, np.float32)
    refs = np.array([[50. + (int(hashlib.sha1(key.encode()).hexdigest()[:4], 16) % 1000)*1e-7, 10.] for key in ids])
    raw = {"status": "complete", "query_indices": indices, "query_times": times,
            "similarities": np.ascontiguousarray(scores, dtype=np.float32), "ref_ids": ids, "ref_latlon": refs,
            "provenance": {"center": region["center"], "cap": 1500, "radius_m": 3000, "source": "mapillary",
                "query_mode": "crops", "backbone": {"name": "megaloc", "hub": "official/pinned", "device": "cpu"},
                "reference_encoding_identity": "unverified legacy revision/preprocessing",
                "preprocessing": {"reference_color": "RGB", "reference_size": [512, 512],
                    "reference_resize": "opencv INTER_LINEAR", "scale": "divide by255", "mean": [.485,.456,.406],
                    "std": [.229,.224,.225], "descriptor_normalization": "L2", "dimensions": 8448},
                "query_identity": {"query_video_sha256": VIDEO, "query_pixels_sha256": "d"*64,
                    "query_descriptors_sha256": "e"*64, "actual_model_state_sha256": "f"*64,
                    "query_indices_sha256": union._array_sha(indices), "query_times_sha256": union._array_sha(times),
                    "query_mode": "crops", "views": 3, "aggregation": union.PROFILE["crop_aggregation"]}}}
    set_listing(raw)
    return raw


def set_listing(raw, extras=()):
    refs=[{"id":key.split(':',1)[1],"lat":float(ll[0]),"lon":float(ll[1])} for key,ll in zip(raw["ref_ids"],raw["ref_latlon"])]
    refs+=list(extras)
    triples=[[r["id"],r["lat"],r["lon"]] for r in refs]
    digest=hashlib.sha256(json.dumps(triples,separators=(',',':')).encode()).hexdigest()
    p=raw["provenance"]
    p["fetch"]={"listing_stats_known":True,"listing":"tiles","lost":0,"snapshot":{"n_refs":len(refs),"refs_sha256":digest}}
    p["acquisition"]={"status":"complete","listing_complete":True,"cap":1500,"listed_references":refs,
        "listed_ref_ids":["mapillary:"+r["id"] for r in refs],"listed_ref_latlon":[[r["lat"],r["lon"]] for r in refs],
        "listing_snapshot_sha256":digest,"unserved_references":[{"ref_id":"mapillary:"+r["id"],"reason":"not_in_embedded_reference_snapshot"} for r in extras]}


def run(callback, count=2, *, plan=None, timestamps=None, query_cache=None):
    if plan is None:
        plan = union.plan_image_union(*inputs(count))
    return union.localize_image_union(plan, timestamps=np.arange(80, dtype=float) if timestamps is None else timestamps,
        query_video_sha256=VIDEO, retrieve_evidence=callback, query_cache=query_cache)


def test_union_uses_every_branch_and_only_fixed_mean_full_dp():
    first = np.full((80, 1), .7, np.float32)
    second = np.r_[np.full(40, .1), np.full(40, .6)].astype(np.float32)[:, None]
    calls = []
    def callback(region, **kwargs):
        calls.append(region["hypothesis_id"])
        return evidence(region, ["mapillary:a"] if len(calls)==1 else ["mapillary:b"], first if len(calls)==1 else second)
    result = run(callback)
    assert result["status"] == "complete"
    assert calls == ["branch_0", "branch_1"]
    assert result["chosen_ref_ids"] == ["mapillary:a"]*40 + ["mapillary:b"]*40
    assert result["reference_union"]["reference_count"] == 2
    assert result["successful_branch_count"] == result["retained_branch_count"] == 2
    pool = result["reference_union"]
    control = decode_full_tracks(pool["similarities"], pool["ref_ids"], pool["ref_latlon"], pool["query_times"])
    assert control["outputs"]["raw"]["chosen_ref_ids"] == ["mapillary:a"]*80
    assert result["route_latlon"] == control["outputs"]["reference_mean"]["route_latlon"]
    assert result["objective"] == control["outputs"]["reference_mean"]["objective"]
    assert "outputs" not in result and result["score_normalization"] == "reference_mean"
    assert not result["uses_rigid_vo"] and not result["confidence_calibrated"]
    assert result["candidate_pruning"] is False


def test_all_six_branches_share_one_explicit_query_cache():
    calls, encoded = [], []
    cache = {}
    def callback(region, **kwargs):
        calls.append((region, kwargs))
        if "queries" not in kwargs["query_cache"]:
            kwargs["query_cache"]["queries"] = "descriptor proof"
            encoded.append(1)
        return evidence(region, ["mapillary:"+region["hypothesis_id"]])
    result = run(callback, count=6, query_cache=cache)
    assert result["status"] == "complete" and result["reference_union"]["reference_count"] == 6
    assert len(encoded) == 1 and len(calls) == 6
    assert all(kw["query_cache"] is cache and kw["n_query"] == 80 and kw["query_mode"] == "crops" for _, kw in calls)
    assert all(r["radius_m"] == 3000 and r["cap"] == 1500 for r, _ in calls)


@pytest.mark.parametrize("count", [0, 7, 20])
def test_branch_budget_declines_without_any_callback_or_truncation(count):
    plan = union.plan_image_union(*inputs(count))
    result = run(lambda *a, **k: pytest.fail("no acquisition allowed"), plan=plan)
    assert result["status"] == "inactive" and result["retained_branch_count"] == count
    assert len(plan["clues"]["candidate_clusters"]) == count
    assert "route_latlon" not in result


def test_plan_roundtrip_and_rehashed_center_tamper_rejected_before_acquisition():
    plan = json.loads(json.dumps(union.plan_image_union(*inputs())))
    assert run(lambda r, **k: evidence(r), plan=plan)["status"] == "complete"
    plan["region_plans"][0]["center"][0] += .001
    plan["plan_sha256"] = union._json_sha({k:v for k,v in plan.items() if k!="plan_sha256"})
    result = run(lambda *a, **k: pytest.fail("tampered plan"), plan=plan)
    assert result["status"] == "inactive" and "plan or profile changed" in result["reason"]


@pytest.mark.parametrize("fault", ["poi_subset", "duplicate_poi", "duplicate_hypothesis", "wide_branch", "nonfinite", "native_resize", "gps_stamps"])
def test_invalid_planning_inputs_refuse_whole_branch(fault):
    clues, native = inputs()
    branch = clues["candidate_clusters"][0]
    if fault == "poi_subset": branch["poi_ids"] = branch["poi_ids"][:1]
    elif fault == "duplicate_poi": branch["pois"][1] = deepcopy(branch["pois"][0])
    elif fault == "duplicate_hypothesis": clues["candidate_clusters"][1]["id"] = branch["id"]
    elif fault == "wide_branch": branch["pois"][1]["lat"] += 1
    elif fault == "nonfinite": branch["pois"][0]["lat"] = float("nan")
    elif fault == "native_resize": native["native_bbox_grid"] = False
    else: native["position_stamps_rejected"] = False
    with pytest.raises(ValueError): union.plan_image_union(clues, native)


def test_global_dedupe_memberships_remap_and_permutation_invariant():
    plans = union.plan_image_union(*inputs())
    def callback(region, **kw):
        ids = ["mapillary:c", "mapillary:a"] if region["hypothesis_id"]=="branch_0" else ["mapillary:b", "mapillary:c"]
        cols = {"mapillary:a":np.linspace(.1,.8,80,dtype=np.float32), "mapillary:b":np.linspace(.7,.2,80,dtype=np.float32), "mapillary:c":np.zeros(80,np.float32)}
        return evidence(region, ids, np.column_stack([cols[k] for k in ids]))
    result = run(callback, plan=plans)
    assert result["status"] == "complete"
    pool = result["reference_union"]
    assert pool["ref_ids"] == ["mapillary:a", "mapillary:b", "mapillary:c"]
    assert pool["occurrence_count"] == 4 and pool["memberships"]["mapillary:c"] == ["branch_0", "branch_1"]
    assert result["chosen_ref_ids"] == [pool["ref_ids"][i] for i in result["chosen_ref_indices"]]
    clues, native = inputs(); clues["candidate_clusters"].reverse()
    def permuted(region, **kw):
        item = callback(region, **kw)
        item["ref_ids"].reverse(); item["ref_latlon"] = item["ref_latlon"][::-1]; item["similarities"] = item["similarities"][:, ::-1]
        return item
    other = run(permuted, plan=union.plan_image_union(clues,native))
    assert result["route_latlon"] == other["route_latlon"]
    assert result["chosen_ref_ids"] == other["chosen_ref_ids"] and result["objective"] == other["objective"]
    assert pool["hashes"] == other["reference_union"]["hashes"]


@pytest.mark.parametrize("fault", ["coordinates", "score_column", "pixel_sha256", "embedding_sha256", "encoding_identity"])
def test_duplicate_conflict_declines_without_selecting_a_subset(fault):
    calls = []
    def callback(region, **kw):
        item = evidence(region)
        item["provenance"]["reference_identities"] = {"mapillary:a": {"pixel_sha256":"1"*64,"embedding_sha256":"2"*64,"encoding_identity":"known encoding"}}
        if calls:
            if fault == "coordinates":
                item["ref_latlon"][0,0] += .00001
                set_listing(item)
            elif fault == "score_column": item["similarities"][1,0] += .001
            else: item["provenance"]["reference_identities"]["mapillary:a"][fault] = "3"*64
        calls.append(1)
        return item
    result = run(callback)
    assert len(calls)==2 and result["status"] == "inactive"
    assert "duplicate reference ID" in result["reason"] and "route_latlon" not in result
    assert len(result["regions"]) == 2


def test_tolerated_duplicate_uses_frozen_request_order_not_score_max():
    plan = union.plan_image_union(*inputs())
    chosen = min(plan["region_plans"], key=lambda r:(r["request_sha256"],r["hypothesis_id"]))["hypothesis_id"]
    def callback(region, **kw):
        base = .2 if region["hypothesis_id"] == chosen else .2000005
        return evidence(region,scores=np.full((80,1),base,np.float32))
    result = run(callback,plan=plan)
    assert result["status"] == "complete"
    assert np.all(result["reference_union"]["similarities"]==np.float32(.2))
    clues,native=inputs();clues["candidate_clusters"].reverse()
    other=run(callback,plan=union.plan_image_union(clues,native))
    assert result["reference_union"]["hashes"] == other["reference_union"]["hashes"]


def test_duplicate_tolerance_cannot_bridge_incompatible_extremes():
    def callback(region, **kw):
        i=int(region["hypothesis_id"].split('_')[-1])
        return evidence(region,scores=np.full((80,1),.2+i*.0000009,np.float32))
    result=run(callback,count=3)
    assert result["status"]=="inactive" and "conflicting" in result["reason"]


@pytest.mark.parametrize("fault", ["partial", "bad_status", "none", "exception", "query_pixels", "query_descriptors", "model_state", "mode", "size", "backbone", "grid", "times", "cosine", "dtype", "too_many", "source_ids", "outside_bbox", "stale_hash", "stale_ids", "inventory"])
def test_bad_region_never_promotes_other_branch(fault):
    calls=[]
    def callback(region, **kw):
        calls.append(region["hypothesis_id"])
        item=evidence(region)
        if len(calls)==1:return item
        p=item["provenance"]
        if fault=="partial":item["status"]="partial"
        elif fault=="bad_status":item["status"]=[]
        elif fault=="none":return None
        elif fault=="exception":raise ValueError("secret-token-must-not-be-output")
        elif fault in {"query_pixels","query_descriptors","model_state"}:
            key={"query_pixels":"query_pixels_sha256","query_descriptors":"query_descriptors_sha256","model_state":"actual_model_state_sha256"}[fault];p["query_identity"][key]="0"*64
        elif fault=="mode":p["query_mode"]="warp"
        elif fault=="size":p["preprocessing"]["reference_size"]=[322,322]
        elif fault=="backbone":p["backbone"]["hub"]="other/pin"
        elif fault=="grid":item["query_indices"][1]+=1
        elif fault=="times":item["query_times"][1]+=.01
        elif fault=="cosine":item["similarities"][0,0]=np.nan
        elif fault=="dtype":item["similarities"]=item["similarities"].astype(np.float64)
        elif fault=="too_many":return evidence(region,[f"mapillary:{i}" for i in range(1501)])
        elif fault=="source_ids":item["ref_ids"]=["other:a"]
        elif fault=="outside_bbox":item["ref_latlon"][0,0]+=1
        elif fault=="stale_hash":p["evidence_identity"]={"similarities_sha256":"0"*64}
        elif fault=="stale_ids":p["evidence_identity"]={"ref_ids":["mapillary:other"]}
        elif fault=="inventory":p["reference_inventory"]={"n_refs":1,"refs_sha256":"0"*64}
        return item
    result=run(callback)
    assert len(calls)==2 and result["status"]=="inactive"
    assert result["successful_branch_count"]==1 and len(result["failed_regions"])==1
    assert "route_latlon" not in result and "secret-token" not in json.dumps(result)


def test_all_later_branches_attempted_after_earlier_failure():
    calls=[]
    def callback(region, **kw):
        calls.append(region["hypothesis_id"])
        if len(calls)==1:raise RuntimeError("private URL")
        return evidence(region)
    result=run(callback,count=4)
    assert len(calls)==4 and len(result["regions"])==4 and result["successful_branch_count"]==3
    assert result["status"]=="inactive" and "route_latlon" not in result


def test_exact_nonuniform_native_grid_and_immutable_union():
    times=10+np.arange(161)*.4+np.sin(np.arange(161))*.001
    clues,native=inputs();native["window"]=[float(times[0]),float(times[-1])]
    plan=union.plan_image_union(clues,native)
    result=run(lambda r,**k:evidence(r,timestamps=times),plan=plan,timestamps=times)
    assert result["status"]=="complete"
    expected=np.linspace(0,160,80).astype(np.int64)
    np.testing.assert_array_equal(result["reference_union"]["query_indices"],expected)
    np.testing.assert_array_equal(result["route_timestamps"],times[expected])
    for key in ["query_indices","query_times","ref_latlon","similarities"]:
        a=result["reference_union"][key]
        with pytest.raises(ValueError):a.setflags(write=True)
    claims=result["reference_union"]["reference_identities"]["mapillary:a"]
    assert all(v is None for v in claims.values())
    assert all(r["provenance"]["reference_encoding_identity"].startswith("unverified") for r in result["regions"])


@pytest.mark.parametrize("fault", ["short", "unordered", "outside", "video"])
def test_bad_parent_grid_refuses_before_callback(fault):
    plan=union.plan_image_union(*inputs());times=np.arange(80,dtype=float);video=VIDEO
    if fault=="short":times=times[:79]
    elif fault=="unordered":times[1]=times[0]
    elif fault=="outside":times+=200
    else:video="0"*64
    result=union.localize_image_union(plan,timestamps=times,query_video_sha256=video,retrieve_evidence=lambda *a,**k:pytest.fail("parentgrid"))
    assert result["status"]=="inactive" and not result["regions"]


def test_known_unserved_photos_are_accounted_without_inventing_causes():
    def callback(region, **kw):
        raw=evidence(region)
        set_listing(raw,[{"id":"unserved1","lat":50.,"lon":10.}, {"id":"unserved2","lat":50.00001,"lon":10.}])
        return raw
    result=run(callback)
    assert result["status"]=="complete" and result["reference_union"]["reference_count"]==1
    for row in result["regions"]:
        acquisition=row["provenance"]["acquisition"]
        assert len(acquisition["listed_references"])==3 and len(acquisition["unserved_references"])==2
        assert all(x["reason"]=="not_in_embedded_reference_snapshot" for x in acquisition["unserved_references"])


@pytest.mark.parametrize("fault",["missing","status","unknown_listing","lost","bool_lost","fetch_snapshot","record_snapshot","listed_coord","listed_id","listed_duplicate","missing_outcome","overlap","unknown_outcome","redundant_ids","redundant_coords","provider"])
def test_acquisition_complete_is_not_inferred_from_valid_matrix_or_query_hash(fault):
    calls=[]
    def callback(region, **kw):
        raw=evidence(region);calls.append(1)
        if len(calls)==1:return raw
        set_listing(raw,[{"id":"unserved","lat":50.,"lon":10.}])
        p=raw["provenance"];a=p["acquisition"];f=p["fetch"]
        if fault=="missing":p.pop("acquisition")
        elif fault=="status":a["status"]="unknown"
        elif fault=="unknown_listing":f["listing_stats_known"]=False
        elif fault=="lost":f["lost"]=1
        elif fault=="bool_lost":f["lost"]=False
        elif fault=="fetch_snapshot":f["snapshot"]["refs_sha256"]="0"*64
        elif fault=="record_snapshot":a["listing_snapshot_sha256"]="0"*64
        elif fault=="listed_coord":a["listed_references"][0]["lat"]+=.001
        elif fault=="listed_id":a["listed_references"][0]["id"]="unrelated"
        elif fault=="listed_duplicate":a["listed_references"][1]=deepcopy(a["listed_references"][0])
        elif fault=="missing_outcome":a["unserved_references"]=[]
        elif fault=="overlap":a["unserved_references"].append({"ref_id":"mapillary:a","reason":"download_failed"})
        elif fault=="unknown_outcome":a["unserved_references"][0]["reason"]="unknown"
        elif fault=="redundant_ids":a["listed_ref_ids"].reverse()
        elif fault=="redundant_coords":a["listed_ref_latlon"][0][0]+=.001
        else:
            p["source"]="panoramax";raw["ref_ids"]=["panoramax:a"]
        return raw
    result=run(callback)
    assert len(calls)==2 and result["status"]=="inactive" and result["successful_branch_count"]==1
    assert "route_latlon" not in result
    assert result["regions"][1]["reported_status"]=="complete"


def test_supplied_matrices_and_caller_plan_remain_unchanged():
    plan=union.plan_image_union(*inputs());before=deepcopy(plan);returned=[]
    def callback(region,**kw):
        raw=evidence(region);returned.append(raw)
        region["poi_ids"].append("not-a-real-POI")
        return raw
    result=run(callback,plan=plan)
    assert result["status"]=="complete" and plan==before
    for raw in returned:
        np.testing.assert_array_equal(raw["similarities"],np.full((80,1),.2,np.float32))
    returned[0]["similarities"][:]=0
    assert np.all(result["reference_union"]["similarities"]==np.float32(.2))
