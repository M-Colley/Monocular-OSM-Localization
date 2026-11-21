"""Fixed v1 legacy precedence; only the new union compares new raw identities."""
from copy import deepcopy
import pytest
from monocular_osm import assisted_policy as base, assisted_policy_v3 as v3, assisted_comparison as c
from tests.test_assisted_comparison import full_comparison,fixed_case
from tests.test_metadata_union import summary,coords


def evidence():
    args,_,proof=full_comparison()
    return {"baseline":c.augment_baseline_summary(args[1],args[2]),"cues":{"status":"failed"},
        "expanded":c.union_summary(proof,proof["route_latlon"],proof["route_timestamps"])}


def semantic(value):
    return {k:v for k,v in value.items() if k not in {"policy_version","config_sha256"}}


def test_new_fixed_primary_admits_without_rewriting_original_baseline():
    data=evidence();before=deepcopy(data);decision=v3.decide_assisted_policy(data)
    assert decision["selected_candidate"]=="expanded" and data==before
    details=decision["evidence_summary"]["metadata_union"]
    assert details["original_baseline_score_identity"]==data["baseline"]["score_identity"]
    assert details["new_raw_appearance_identity"]==data["expanded"]["score_identity"]
    assert details["new_raw_appearance_identity"]!=details["original_baseline_score_identity"]


def test_appearance_identity_cannot_admit_previously_incomparable_legacy_metadata():
    args=fixed_case();legacy=deepcopy(args[3]);legacy["inner_decoder"]["settings"]["query_dt_s"]=7.
    legacy["decoded_reference_latlon"]=coords(list(range(80))).tolist()
    data={"baseline":args[2],"cues":{"status":"failed"},"expanded":summary(legacy,True)}
    assert c.baseline_projection(legacy,data["expanded"])["appearance_identity_sha256"]==c.baseline_projection(args[1],args[2])["appearance_identity_sha256"]
    assert data["baseline"]["score_identity"]!=data["expanded"]["score_identity"]
    assert v3.decide_assisted_policy(data)["selected_candidate"]=="baseline"
    assert semantic(v3.decide_assisted_policy(data))==semantic(base.decide_assisted_policy(data))


def test_already_admitted_original_legacy_track_still_wins_unchanged():
    args=fixed_case();legacy=deepcopy(args[3]);legacy["decoded_reference_latlon"]=coords(list(range(80))).tolist()
    data={"baseline":args[2],"cues":{"status":"failed"},"expanded":summary(legacy,True)}
    assert semantic(v3.decide_assisted_policy(data))==semantic(base.decide_assisted_policy(data))
    assert v3.decide_assisted_policy(data)["selected_candidate"]=="expanded"
    # A union cannot masquerade as that earlier already-admitted route.
    union=evidence();union["expanded"]["legacy_metadata"]={"comparison":legacy,"summary":summary(legacy,True)}
    assert v3.decide_assisted_policy(union)["selected_candidate"]=="baseline"


@pytest.mark.parametrize("earlier",["strong","cues","joint","image","pf"])
def test_earlier_v1_stages_ignore_all_future_union_fields(earlier):
    data=evidence();data["expanded"]={"candidate_kind":"metadata_union_reference_mean","corrupt":float("nan")}
    route={"status":"complete","route_valid":True,"point_count":80}
    if earlier=="strong":data["baseline"]["similarity_median"] = .2
    if earlier=="cues":data["cues"]={"status":"not_run"}
    if earlier in {"joint","image","pf"}:
        data["cues"]={"status":"complete","duration_s":200.,"vo":{"status":"complete","valid_fraction":.99 if earlier!="image" else .5},
            "ocr":{"status":"complete","clue_branch_count":0 if earlier=="pf" else 1},
            "plane":{"status":"complete","measured":True,"coverage":.8,"rel_spread":.1}}
    if earlier=="joint":data["poi_joint"]={**route,"retained_branch_count":1,"successful_branch_count":1}
    if earlier=="image":data["poi_image_track"]={**route,"retained_branch_count":1,"successful_branch_count":1,
        "uses_rigid_vo":False,"radius_m":3000.,"cap":1500,"query_mode":"crops","score_normalization":"reference_mean"}
    if earlier=="pf":data["metric_pf"]={**route,"metric_only":True,"uses_user_spatial_prior":False,"position_observation_count":0}
    expected=base.decide_assisted_policy({**data,"expanded":{"status":"not_run"}})
    assert semantic(v3.decide_assisted_policy(data))==semantic(expected)


@pytest.mark.parametrize("bad",["raw","missing_baseline","old_id","partial","pools","scalar","decoder","projection","missing_kind_old_id"])
def test_bad_union_proofs_cannot_promote_new_candidate(bad):
    data=evidence();candidate=data["expanded"]
    if bad=="raw":candidate["candidate_kind"]="metadata_union_raw"
    if bad=="missing_baseline":data["baseline"].pop("metadata_union_baseline")
    if bad=="old_id":candidate["score_identity"]=data["baseline"]["score_identity"]
    if bad=="partial":candidate["status"]="partial"
    if bad=="pools":candidate["complete_pool_count"]=1
    if bad=="scalar":candidate["dominant_coordinate_fraction"]=0.
    if bad=="decoder":candidate["actual_decoder"]["variant"]="raw"
    if bad=="projection":data["baseline"]["metadata_union_baseline"]["original_similarity_median"]=.09
    if bad=="missing_kind_old_id":candidate.pop("candidate_kind");candidate["score_identity"]=data["baseline"]["score_identity"]
    decision=v3.decide_assisted_policy(data)
    assert decision["selected_candidate"]=="baseline"
    assert decision["evidence_summary"]["metadata_union"]["admitted"] is False


def test_configuration_detached_and_old_constants_unmodified():
    config=v3.policy_config();config["baseline_median_min"]=0
    assert v3.policy_config()["baseline_median_min"]==base.policy_config()["baseline_median_min"]==.162
    assert v3.policy_config()["image_only_branch_count"]==1
    assert v3.CONFIG_SHA256!=base.CONFIG_SHA256
