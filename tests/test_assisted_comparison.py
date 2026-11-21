"""Actual raw identity/provenance distinctions on bounded synthetic80 grids."""
from copy import deepcopy
from collections import Counter
import numpy as np
import pytest
from monocular_osm import assisted_comparison as c, metadata_union as m
from tests.test_metadata_union import make_region,comparison,summary,coords


def fixed_case():
    q=80;category=np.arange(q)%4
    useful=np.where(category[:,None]==np.arange(4)[None,:],.9,.1).astype(np.float32)
    regions=[make_region("metadata_0000",np.column_stack([useful[:,:2],np.full(q,.95)]),coords([0,50,5000]),["a0","a1","hub"]),
        make_region("metadata_0001",np.column_stack([useful[:,2:],np.full(q,.2)]),coords([100,150,2000]),["a2","a3","junk"])]
    base=make_region("baseline_0000",np.full((q,1),.1),coords([-5000]),["independent_base"])
    bc=comparison(np.full(q,.1),np.repeat(base["arrays"]["ref_latlon"],q,axis=0),["mapillary:independent_base"]*q)
    mc=comparison(np.full(q,.95),np.repeat(regions[0]["arrays"]["ref_latlon"][-1:],q,axis=0),["mapillary:hub"]*q)
    return regions,bc,summary(bc),mc,summary(mc,True),[base]


def full_comparison():
    args=fixed_case();result=m.replay_metadata_union(*args)
    proof=c.union_comparison(result,args[1],args[2],args[3],args[4],requested_pool_count=2)
    return args,result,proof


def test_projection_is_separate_and_original_legacy_fields_remain_exact():
    args=fixed_case();before=deepcopy(args[1:3]);augmented=c.augment_baseline_summary(args[1],args[2])
    projection=c.validated_projection(augmented["metadata_union_baseline"],augmented)
    assert args[1:3]==before and augmented["score_identity"]==args[2]["score_identity"]
    assert projection["appearance_identity_sha256"]!=augmented["score_identity"]
    assert c.legacy_summary(args[1],args[1]["decoded_reference_latlon"])==args[2]


def test_truthful_full_decoder_and_selected_raw_evidence_no_viterbi_forgery():
    args,result,proof=full_comparison();summary=c.union_summary(proof,proof["route_latlon"],proof["route_timestamps"])
    assert "inner_decoder" not in proof and proof["actual_decoder"]["family"]=="full_sequence"
    assert summary["score_identity"]!=args[2]["score_identity"]
    assert summary["similarity_median"]==float(np.float32(.9))
    assert summary["dominant_coordinate_fraction"]==.25
    assert result["outputs"]["raw"]["dominant_coordinate_fraction"]==1.
    assert summary["actual_decoder"]["variant"]=="reference_mean"
    assert summary["dominant_coordinate_fraction"]!=max(Counter(map(tuple,proof["route_latlon"])).values())/80


@pytest.mark.parametrize("bad",["decoder","raw_primary","legacy_id","pool_count","grid","NaN","column","cosine","smooth_scalar"])
def test_bad_or_forged_full_proofs_decline(bad):
    _,_,p=full_comparison()
    if bad=="decoder":p["inner_decoder"]={"actual":"viterbi"}
    if bad=="raw_primary":p["actual_decoder"]["variant"]="raw"
    if bad=="legacy_id":p["appearance_identity_sha256"]=p["baseline_projection"]["original_score_identity"]
    if bad=="pool_count":p["complete_pool_count"]=1
    if bad=="grid":p["query_times"][3]+=.1
    if bad=="NaN":p["raw_column_proof"]["raw_cosines"][0]=float("nan")
    if bad=="column":p["raw_column_proof"]["selected_source_rows"][0]["global_column"]=10001
    if bad=="cosine":p["raw_column_proof"]["selected_source_rows"][0]["raw_cosine"]=-1
    if bad=="smooth_scalar":p["raw_column_proof"]["dominant_coordinate_fraction"]=0
    with pytest.raises(ValueError):c.union_summary(p,p["route_latlon"],p["route_timestamps"])


def test_original_decoder_mismatch_changes_legacy_id_but_not_raw_projection():
    args=fixed_case();other=deepcopy(args[1]);other["inner_decoder"]["settings"]["query_dt_s"]=7.
    other_summary=summary(other)
    assert other_summary["score_identity"]!=args[2]["score_identity"]
    assert c.baseline_projection(other,other_summary)["appearance_identity_sha256"]==c.baseline_projection(args[1],args[2])["appearance_identity_sha256"]
