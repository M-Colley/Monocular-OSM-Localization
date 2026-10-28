"""Controller/summary contracts for the declared full-support v3 comparison."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from monocular_osm import assisted_comparison as comparison, assisted_policy_v3 as policy
from monocular_osm.stages import assisted as stage
from monocular_osm.stages.context import RunContext
from monocular_osm.provenance import file_sha256
from tests.test_assisted_comparison import full_comparison
from tests.test_stages_assisted import candidate as legacy_candidate


@pytest.fixture
def full():
    args, research, proof = full_comparison()
    baseline = {"status": "complete", "route_latlon": deepcopy(args[1]["decoded_reference_latlon"]),
                "route_timestamps": deepcopy(args[1]["query"]["times"]), "comparison": deepcopy(args[1])}
    expanded = {"status": "complete", "route_latlon": deepcopy(proof["route_latlon"]),
                "route_timestamps": deepcopy(proof["route_timestamps"]), "comparison": deepcopy(proof)}
    return baseline, expanded, research


def test_baseline_adds_separate_projection_without_rewriting_any_original_field(full):
    baseline, _, _ = full; before = deepcopy(baseline)
    original = stage.summary("baseline", baseline)
    augmented = stage.summary("baseline", baseline, profile="v3")
    assert {k: augmented[k] for k in original} == original
    assert augmented == comparison.augment_baseline_summary(baseline["comparison"], original)
    assert augmented["score_identity"] != augmented["metadata_union_baseline"]["appearance_identity_sha256"]
    assert baseline == before


def test_expanded_summary_uses_actual_new_route_times_and_raw_selected_columns(full):
    _, child, _ = full
    actual = stage.summary("expanded", child, profile="v3")
    assert actual == comparison.union_summary(child["comparison"], child["route_latlon"], child["route_timestamps"])
    assert actual["actual_decoder"]["family"] == "full_sequence"
    assert actual["actual_decoder"]["variant"] == "reference_mean"
    assert actual["similarity_median"] == pytest.approx(.9)
    assert actual["dominant_coordinate_fraction"] == .25
    assert "inner_decoder" not in actual["comparison_evidence"]
    assert stage.summary("expanded", child)["score_identity"] is None


@pytest.mark.parametrize("fault", ["unknown_schema", "raw", "forged_decoder", "raw_column", "pool_count",
                                    "route", "times", "missing_times", "partial"])
def test_invalid_new_kind_or_proof_never_masquerades_as_legacy(full, fault):
    baseline, child, _ = full
    c = child["comparison"]
    if fault == "unknown_schema": c["schema"] = "future_unknown"
    if fault == "raw": c["candidate_kind"] = "metadata_union_raw"
    if fault == "forged_decoder": c["inner_decoder"] = baseline["comparison"]["inner_decoder"]
    if fault == "raw_column": c["raw_column_proof"]["selected_source_rows"][0]["raw_cosine"] = -1.
    if fault == "pool_count": c["complete_pool_count"] = 1
    if fault == "route": child["route_latlon"][0][0] += .01
    if fault == "times": child["route_timestamps"][-1] += .01
    if fault == "missing_times": child.pop("route_timestamps")
    if fault == "partial": child["status"] = "partial"
    summary = stage.summary("expanded", child, profile="v3")
    assert summary["score_identity"] is None and summary["similarity_median"] is None
    evidence = {"baseline": stage.summary("baseline", baseline, profile="v3"),
                "cues": {"status": "failed"}, "expanded": summary}
    assert policy.decide_assisted_policy(evidence)["selected_candidate"] == "baseline"


def test_new_marker_cannot_smuggle_a_copied_legacy_identity(full):
    baseline, _, _ = full
    child = legacy_candidate(.9)
    child["comparison"]["schema"] = "future_unknown"
    child["comparison"]["candidate_kind"] = "metadata_union_reference_mean"
    summary = stage.summary("expanded", child, profile="v3")
    assert summary["score_identity"] is None


@pytest.mark.parametrize("key", ["candidate_kind", "comparison_schema", "metadata_union_profile_sha256"])
def test_conflicting_child_and_comparison_labels_decline(full, key):
    _,child,_=full;child[key]="incompatible"
    assert stage.summary("expanded",child,profile="v3")["score_identity"] is None


@pytest.mark.parametrize("kind", ["baseline", "expanded", "cues", "poi_joint", "metric_pf"])
def test_default_two_argument_summary_and_v2_remain_identical(kind):
    child = legacy_candidate(.21) if kind != "cues" else {"status": "complete", "cues": {"duration_s": 3.}}
    assert stage.summary(kind, child) == stage.summary(kind, child, profile="v1") == stage.summary(kind, child, profile="v2")
    if kind != "baseline":
        assert stage.summary(kind, child, profile="v3") == stage.summary(kind, child)


@pytest.mark.parametrize("profile", ["v1", "v2"])
def test_old_child_record_keeps_two_argument_summary_callbacks(tmp_path,monkeypatch,profile):
    monkeypatch.setattr(stage,"summary",lambda kind,child:{"callback_kind":kind,"status":child["status"]})
    result=stage.child_record("baseline",{"status":"failed"},tmp_path,profile=profile)
    assert result["summary"]=={"callback_kind":"baseline","status":"failed"}


def run(tmp_path, responses, *, profile="v3", hook=None):
    video = tmp_path / "video.mp4"; video.write_bytes(b"synthetic encoded video")
    cfg = SimpleNamespace(output_dir=tmp_path / "output", city="Example", vo_start_sec=0., vo_end_sec=395., assisted_profile=profile)
    cfg.output_dir.mkdir();ctx = RunContext(cfg=cfg, fusion_w=None, video_path=video);calls=[];saved={}
    def child(kind, context):
        calls.append(kind)
        if hook:hook(kind, context, saved)
        payload=deepcopy(responses[kind]);payload.update(video_sha256=context.result["assisted_inputs"]["video"]["sha256"],
            window=context.result["assisted_inputs"]["registered_window"],source_sha256=context.result["assisted_inputs"]["source_sha256"])
        out=cfg.output_dir/"fake_children"/kind;out.mkdir(parents=True)
        for name,value in (("candidate.json",payload),("result.json",{"synthetic_acquisition":True})):
            (out/name).write_text(json.dumps(value))
        if kind=="expanded" and "comparison" in payload and "schema" in payload["comparison"]:
            for name in ("metadata_union_raw.json","metadata_union_reference_mean.json","baseline_control.json","metadata_control.json"):
                (out/name).write_text(json.dumps({"synthetic_research_or_control":name}))
        frozen=out/"child_frozen.json"
        frozen.write_text(json.dumps({"scoring_has_not_run":True,"artifacts":{str(p.relative_to(out)):file_sha256(p) for p in out.rglob('*') if p.is_file()}}))
        response={**payload,"worker_returncode":0,**{role:{"path":str(out/name),"sha256":file_sha256(out/name)}
            for role,name in (("candidate_artifact","candidate.json"),("result_artifact","result.json"),("frozen_artifact","child_frozen.json"))}}
        saved[kind]=deepcopy(response);return response
    def finalize(context):
        if "assisted_position" in context.result:context.result["answer"]=deepcopy(context.result["assisted_position"])
        (cfg.output_dir/"result.json").write_text(json.dumps(context.result))
    result=stage.run_assisted(ctx,finalize=finalize,run_child=child)
    return result,calls,ctx,saved


def test_controller_passes_actual_baseline_copy_and_freezes_all_union_research_controls(tmp_path,full):
    baseline,expanded,_=full
    def hook(kind,ctx,saved):
        if kind=="expanded":
            assert ctx.assisted_baseline_child==saved["baseline"]
            assert ctx.assisted_baseline_child is not saved["baseline"]
            assert {"frozen_artifact","candidate_artifact","result_artifact"}<=ctx.assisted_baseline_child.keys()
            ctx.assisted_baseline_child["comparison"]["decoded_reference_cosines"][0]=-1.
    result,calls,ctx,saved=run(tmp_path,{"baseline":baseline,"cues":{"status":"failed"},"expanded":expanded},hook=hook)
    assert calls==["baseline","cues","expanded"]
    assert result["answer"]["route_latlon"]==expanded["route_latlon"]
    assert result["answer"]["route_t_sec"]==expanded["route_timestamps"]
    audit=result["assisted_audit"]
    assert audit["selected_candidate"]=="expanded"
    assert audit["reached_stages"][0]["comparison"]==baseline["comparison"]
    assert [policy.decide_assisted_policy(e) for e in audit["decision_inputs"]]==audit["decision_trace"]
    source={n.replace('\\','/') for n in result["assisted_inputs"]["source_sha256"]}
    assert stage.V3_REQUIRED_SOURCES<=source
    parent=json.loads((ctx.cfg.output_dir/"frozen_prediction.json").read_text())
    child_path=Path(saved["expanded"]["frozen_artifact"]["path"])
    assert parent["artifacts"][str(child_path.relative_to(ctx.cfg.output_dir))]==file_sha256(child_path)
    child=json.loads(child_path.read_text())
    assert {"metadata_union_raw.json","metadata_union_reference_mean.json","baseline_control.json","metadata_control.json"}<=child["artifacts"].keys()


@pytest.mark.parametrize("profile", ["v1","v2","v3"])
def test_strong_baseline_stops_without_future_child_or_transport_state(tmp_path,profile):
    baseline=legacy_candidate(.21)
    result,calls,ctx,_=run(tmp_path,{"baseline":baseline},profile=profile)
    assert calls==["baseline"] and not hasattr(ctx,"assisted_baseline_child")
    assert result["answer"]["route_latlon"]==baseline["route_latlon"]
    assert result["answer"]["route_t_sec"]==baseline["route_timestamps"]


def test_v3_refuses_incomplete_source_snapshot_before_any_child(tmp_path,monkeypatch):
    original=stage.snapshot_sources
    def incomplete(*args,**kwargs):
        values=original(*args,**kwargs)
        return {k:v for k,v in values.items() if k.replace('\\','/')!="monocular_osm/full_sequence.py"}
    monkeypatch.setattr(stage,"snapshot_sources",incomplete)
    with pytest.raises(ValueError,match="source snapshot omits"):
        run(tmp_path,{"baseline":legacy_candidate(.21)})


def test_normal_pipeline_v3_strong_baseline_has_real_selected_answer_map_and_freeze(tmp_path,monkeypatch):
    from monocular_osm import pipeline, assisted_worker as worker
    video=tmp_path/'video.mp4';video.write_bytes(b'synthetic encoded input')
    cfg=pipeline.PipelineConfig(url='',city='Example',data_dir=tmp_path/'data/clip',output_dir=tmp_path/'output',
        video_path=video,answer='assisted',assisted_profile='v3',vo_end_sec=3.)
    original=legacy_candidate(.21);calls=[]
    def child(kind,ctx):
        calls.append(kind);assert kind=='baseline'
        child_cfg=worker.child_config(cfg,kind,ctx.video_path);child_cfg.output_dir=ctx.assisted_child_root/kind
        child_cfg.output_dir.mkdir(parents=True)
        response=worker._finish(child_cfg,{'kind':kind,'video_sha256':ctx.result['assisted_inputs']['video']['sha256'],
            'source_sha256':ctx.result['assisted_inputs']['source_sha256']},original)
        return {**response,'worker_returncode':0}
    monkeypatch.setattr(worker,'run_child',child)
    monkeypatch.setattr(pipeline,'extract_video_frames',lambda *_a,**_k:pytest.fail('parent decoded unrequested frames'))
    monkeypatch.setattr(pipeline,'vpr_coarse_prior',lambda *_a,**_k:pytest.fail('parent ran unrequested retrieval'))
    result=pipeline.run_pipeline(cfg)
    assert calls==['baseline'] and result['answer']['method']=='assisted_policy'
    assert result['answer']['route_latlon']==original['route_latlon']
    assert result['answer']['route_t_sec']==original['route_timestamps']
    assert (cfg.output_dir/'answer.png').is_file()
    frozen=json.loads((cfg.output_dir/'frozen_prediction.json').read_text())
    assert frozen['prediction_available'] and frozen['scoring_has_not_run']
    assert result['assisted_audit']['policy_version']==policy.POLICY_VERSION
    assert 'metadata_union_baseline' in result['assisted_audit']['evidence']['baseline']
