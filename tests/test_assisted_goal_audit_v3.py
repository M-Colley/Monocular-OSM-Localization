"""Model-free transport and independent full-union arithmetic for v3."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import assisted_goal_audit as audit
from monocular_osm import assisted_worker as worker, full_sequence
from tests.test_assisted_comparison import full_comparison
from tests.test_assisted_worker_v3 import fixture as worker_fixture


def write(path,value):
    Path(path).write_text(json.dumps(value,allow_nan=False),encoding='utf8')


def test_independent_path_math_matches_both_fixed_full_variants():
    _,research,_=full_comparison();union=research['union']
    for variant,output in research['outputs'].items():
        checked=audit.metadata_union_math(union,output,variant)
        assert checked['raw_cosines']==output['selected_raw_cosines']
        assert checked['objective']==pytest.approx(output['objective'],abs=1e-10)
        assert checked['dominant_coordinate_fraction']==output['dominant_coordinate_fraction']


@pytest.mark.parametrize('fault',['objective','emission','raw','chosen_id','chosen_column','raw_route','route','time','pruning','states','dominance'])
def test_changed_full_path_cost_raw_column_or_smoothed_route_refuses(fault):
    _,research,_=full_comparison();v=deepcopy(research['outputs']['reference_mean'])
    if fault=='objective':v['objective']+=.01
    if fault=='emission':v['selected_emissions'][0]+=.01
    if fault=='raw':v['selected_raw_cosines'][0]+=.01
    if fault=='chosen_id':v['chosen_ref_ids'][0]='mapillary:another'
    if fault=='chosen_column':v['chosen_ref_indices'][0]=99999
    if fault=='raw_route':v['raw_route_latlon'][0][0]+=.001
    if fault=='route':v['route_latlon'][0][0]+=.001
    if fault=='time':v['route_timestamps'][-1]+=.01
    if fault=='pruning':v['candidate_pruning']=True
    if fault=='states':v['all_states_per_query']-=1
    if fault=='dominance':v['dominant_coordinate_fraction']=0.
    with pytest.raises(ValueError):audit.metadata_union_math(research['union'],v,'reference_mean')


def test_independent_union_is_all_pool_lexical_canonical_without_decoder_call(monkeypatch):
    args,research,_=full_comparison()
    monkeypatch.setattr(full_sequence,'decode_full_tracks',lambda *_a,**_k:pytest.fail('auditor may not run global DP'))
    rebuilt=audit.metadata_union_rebuild(args[0])
    for key in ('ref_ids','ref_latlon','raw_similarities','query_indices','query_times'):
        np.testing.assert_array_equal(rebuilt[key],research['union'][key])
    assert rebuilt['mapping']==research['union']['mapping']


@pytest.mark.parametrize('fault',['overlap_without_rows','coordinates','scores','grid'])
def test_conflicting_or_unbound_overlap_cannot_enter_union(fault):
    args,_,_=full_comparison();regions=deepcopy(args[0]);regions.append(deepcopy(regions[0]));regions[-1]['key']='overlap'
    rows={r['key']:np.ones((len(r['arrays']['ref_ids']),8448),np.float32) for r in regions}
    if fault=='overlap_without_rows':rows={}
    if fault=='coordinates':regions[-1]['arrays']['ref_latlon'][0,0]+=.01
    if fault=='scores':regions[-1]['arrays']['raw_similarities'][0,0]-=.01
    if fault=='grid':regions[-1]['arrays']['query_times'][0]+=.01
    with pytest.raises(ValueError):audit.metadata_union_rebuild(regions,rows)


def component_case(tmp_path,monkeypatch):
    ctx,request,parent=worker_fixture(tmp_path,monkeypatch)
    assert worker._metadata_admission(ctx,request),ctx.result.get('assisted_expansion_gate')
    write(ctx.cfg.output_dir/'request.json',request)
    selected=ctx.assisted_metadata_union_selected
    child={'status':'complete','route_latlon':deepcopy(selected['route_latlon']),
        'route_timestamps':deepcopy(selected['route_timestamps']),
        'comparison':worker._comparison(ctx,ctx.cfg.output_dir),
        'candidate_artifact':{'path':str(ctx.cfg.output_dir/'candidate.json')},
        'retrieval_selection':deepcopy(ctx.result)}
    # Real worker transport passes through _safe/JSON before the parent audit.
    child=json.loads(json.dumps(child))
    inputs={'execution_id':request['execution_id'],'video':{'sha256':request['video_sha256']},
        'source_sha256':request['source_sha256'],'registered_window':[0.,395.],
        'config':{'assisted_profile':'v3','vpr_region_cap':6,'vpr_city_fine_regions':3,'vpr_city_coarse_cap':250,'vpr_ref_cap':1500}}
    root=ctx.cfg.output_dir.parents[2]
    ledger={p:audit.file_sha(p) for p in root.rglob('*') if p.is_file()}
    return SimpleNamespace(ctx=ctx,request=request,parent=parent,child=child,inputs=inputs,
        before=deepcopy(request['policy_evidence']),gate=deepcopy(child['retrieval_selection']['assisted_expansion_gate']),
        files=audit.Files(root),ledger=ledger)


def check_component(case):
    return audit.metadata_union_component(case.files,case.child,case.ledger,case.inputs,case.before,
                                         case.gate,case.parent.assisted_baseline_child)


def test_actual_worker_frozen_bundle_independent_arithmetic_passes_without_optimality_replay(tmp_path,monkeypatch):
    case=component_case(tmp_path,monkeypatch)
    monkeypatch.setattr(full_sequence,'decode_full_tracks',lambda *_a,**_k:pytest.fail('no globalDP in strict transport verifier'))
    checked=check_component(case)
    assert checked['admitted'] and 'global optimality not replayed' in checked['audit_scope']
    assert audit.expansion_v3(case.files,case.child,case.ledger,case.inputs,case.before,case.parent.assisted_baseline_child) is None


@pytest.mark.parametrize('fault',['baseline_video','baseline_window','baseline_execution','baseline_summary','missing_component',
    'missing_control','new_kind','actual_route','actual_times','legacy_priority','municipality_future'])
def test_stale_source_bindings_or_future_stage_cannot_promote_new_full_path(tmp_path,monkeypatch,fault):
    case=component_case(tmp_path,monkeypatch)
    if fault.startswith('baseline_'):
        binding=case.request['metadata_union_baseline']
        if fault=='baseline_video':binding['expected']['video_sha256']='f'*64
        if fault=='baseline_window':binding['expected']['window']=[0.,400.]
        if fault=='baseline_execution':binding['expected']['execution_id']='another'
        if fault=='baseline_summary':binding['original_summary']['score_identity']='f'*64
        p=case.ctx.cfg.output_dir/'request.json';write(p,case.request);case.ledger[p]=audit.file_sha(p)
    if fault=='missing_component':case.gate['metadata_union']['artifacts']['frozen_artifact']['sha256']='f'*64
    if fault=='missing_control':case.gate['metadata_control']['sha256']='f'*64
    if fault=='new_kind':case.child['comparison']['candidate_kind']='metadata_union_raw'
    if fault=='actual_route':case.child['route_latlon'][0][0]+=.001
    if fault=='actual_times':case.child['route_timestamps'][-1]+=.01
    if fault=='legacy_priority':case.gate['decision']['selected_candidate']='expanded'
    if fault=='municipality_future':case.child['retrieval_selection']['city_cover_retrieval']={'status':'ok'}
    with pytest.raises((ValueError,KeyError)):
        if fault=='municipality_future':audit.expansion_v3(case.files,case.child,case.ledger,case.inputs,case.before,case.parent.assisted_baseline_child)
        else:check_component(case)


@pytest.mark.parametrize('fault',['count','order','regional_count','missing_request','mapping','extra','query_mode'])
def test_actual_plan_refuses_before_any_numerical_array_load(tmp_path,monkeypatch,fault):
    case=component_case(tmp_path,monkeypatch)
    record=case.gate['metadata_union_plan'];p=Path(record['artifact']['path']);plan=json.loads(p.read_text())
    if fault=='count':plan['requested_pool_count']=7
    if fault=='order':plan['ordered_requests'].reverse()
    if fault=='regional_count':plan['regional_plan_count']+=1
    if fault=='missing_request':plan['ordered_requests'].pop()
    if fault=='mapping':plan['regional_to_audit_indices'].reverse()
    if fault=='extra':plan['ordinary_refinement_call_indices']=[0]
    if fault=='query_mode':plan['ordered_requests'][0]['request']['query_mode']='crops'
    write(p,plan);record.update(artifact={'path':str(p),'sha256':audit.file_sha(p)},plan=plan)
    control=Path(case.gate['metadata_control']['path']);body=json.loads(control.read_text());body['artifacts']['union_plan.json']=audit.file_sha(p)
    write(control,body);case.gate['metadata_control']['sha256']=audit.file_sha(control)
    case.ledger[p]=audit.file_sha(p);case.ledger[control]=audit.file_sha(control)
    monkeypatch.setattr(audit,'load_npz',lambda *_:pytest.fail('request/source plan must refuse before NPZ allocation'))
    with pytest.raises(ValueError):check_component(case)


def descriptor_case(tmp_path,monkeypatch):
    from tests.test_public_reference_cache import case as public_case,tensor_sha
    from monocular_osm import public_reference_cache as cache
    c=public_case.__wrapped__(tmp_path,monkeypatch)
    rows=np.zeros((2,8448),np.float32);rows[0,0]=1.;rows[1,1]=1.
    np.savez(c.embedding,emb=rows,fingerprint=c.fp)
    c.audit['preprocessing']['dimensions']=8448;c.audit['ref_embeddings_sha256']=tensor_sha(rows)
    receipt=cache.describe_pool(c.audit,c.cfg)
    regions=[{'key':name,'audit':deepcopy(c.audit),'arrays':{'ref_ids':c.ids.copy(),'ref_latlon':c.xy.copy()}} for name in ('a','b')]
    parent=tmp_path/'output';parent.mkdir();records=[];ledger={}
    for name in ('a','b'):
        p=parent/f'rows_{name}.npz';np.savez(p,rows=rows)
        records.append({'region':name,'artifact':{'path':str(p),'sha256':audit.file_sha(p)},'rows_sha256':worker._array_sha(rows),
            'receipt':deepcopy(receipt),'protected_hashes':{str((c.directory/v['name']).resolve()):v['sha256'] for v in receipt['files']}})
        ledger[p]=audit.file_sha(p)
    return SimpleNamespace(files=audit.Files(parent),regions=regions,records=records,ledger=ledger,receipt=receipt,rows=rows,c=c)


def test_overlapping_actual_saved_rows_bind_public_metadata_pixels_descriptor_receipt(tmp_path,monkeypatch):
    c=descriptor_case(tmp_path,monkeypatch)
    result=audit.metadata_descriptor_rows(c.files,c.records,c.regions,c.ledger)
    assert set(result)=={'a','b'}
    np.testing.assert_array_equal(result['a'],c.rows)


@pytest.mark.parametrize('fault',['row','receipt_model','encoding_claim','request','missing','source_payload'])
def test_cached_descriptor_identity_or_source_conflict_cannot_pass(tmp_path,monkeypatch,fault):
    c=descriptor_case(tmp_path,monkeypatch)
    if fault=='row':
        p=Path(c.records[0]['artifact']['path']);rows=c.rows.copy();rows[0,0]=.5;np.savez(p,rows=rows)
        c.records[0]['artifact']['sha256']=c.ledger[p]=audit.file_sha(p)
    if fault=='receipt_model':c.records[0]['receipt']['actual_model_state_sha256']='b'*64
    if fault=='encoding_claim':c.records[0]['receipt']['reference_encoding_identity']='VERIFIED'
    if fault=='request':c.records[0]['receipt']['request']['center'][0]+=.01
    if fault=='missing':c.records.pop()
    if fault=='source_payload':c.c.embedding.write_bytes(c.c.embedding.read_bytes()+b'changed')
    with pytest.raises(ValueError):audit.metadata_descriptor_rows(c.files,c.records,c.regions,c.ledger)


def test_v3_strong_baseline_full_goal_roundtrip_uses_profile_child_record(tmp_path,monkeypatch):
    from tests import test_check_assisted_goal as legacy
    original=legacy.PipelineConfig
    monkeypatch.setattr(legacy,'PipelineConfig',lambda **kwargs:original(**{**kwargs,'assisted_profile':'v3'}))
    case=legacy.fixture(tmp_path,monkeypatch)
    report=legacy.check(case)
    assert report['goal_met'],legacy.reasons(report)
    assert 'metadata_union_baseline' in case.ctx.result['assisted_audit']['evidence']['baseline']


def test_v3_source_snapshot_requires_actual_full_decoder_and_adapter_modules(tmp_path,monkeypatch):
    from tests import test_check_assisted_goal as legacy
    original=legacy.PipelineConfig
    monkeypatch.setattr(legacy,'PipelineConfig',lambda **kwargs:original(**{**kwargs,'assisted_profile':'v3'}))
    case=legacy.fixture(tmp_path,monkeypatch)
    for key in list(case.ctx.result['assisted_inputs']['source_sha256']):
        if key.replace('\\','/')=='monocular_osm/full_sequence.py':case.ctx.result['assisted_inputs']['source_sha256'].pop(key)
    legacy.refresh_parent(case)
    assert not legacy.check(case)['goal_met']


def full_parent_case(tmp_path,monkeypatch):
    """Real controller/worker-freeze/metadata-DP code, synthetic acquisition only."""
    from monocular_osm import pipeline,assisted_metadata_union as component,baselines,provenance
    from monocular_osm.stages import assisted as stage
    from monocular_osm.stages.context import RunContext
    from tests.test_assisted_comparison import fixed_case
    from tests.test_assisted_metadata_union import memory,cpu
    video=tmp_path/'video.mp4';video.write_bytes(b'synthetic video source for no-model audit')
    cfg=pipeline.PipelineConfig(url='',city='Example',data_dir=tmp_path/'data/clip',output_dir=tmp_path/'out',video_path=video,
        answer='assisted',assisted_profile='v3',frame_stride=6,vo_start_sec=0.,vo_end_sec=395.)
    cfg.output_dir.mkdir();parent=RunContext(cfg=cfg,fusion_w=None,video_path=video)
    regions,bc,_,mc,_,base=fixed_case();returned={};saved={}
    evaluate=component.evaluate_metadata_union
    monkeypatch.setattr(component,'evaluate_metadata_union',lambda b,m,**kwargs:evaluate(b,m,resource_gate=memory,numerical_context=cpu,**kwargs))
    def raw_audit(region,center):
        z=region['arrays'];a={**deepcopy(region['audit']),**{k:v.copy() for k,v in z.items() if not k.startswith('returned_')}}
        a['returned_track']={k:z['returned_'+k].copy() for k in ('indices','latlon','cosines')}
        a['ref_embeddings_sha256']='e'*64;a['reference_pool'].update(center=center,radius_m=3000.,cap=1500)
        return a
    def child(kind,context):
        if kind=='cues':raise RuntimeError('synthetic unavailable OCR/VO, no acquisition')
        child_cfg=worker.child_config(cfg,kind,video);child_cfg.output_dir=context.assisted_child_root/kind;child_cfg.output_dir.mkdir(parents=True)
        request={'kind':kind,'config':provenance.config_snapshot(child_cfg),'execution_id':context.result['assisted_inputs']['execution_id'],
            'video_sha256':context.result['assisted_inputs']['video']['sha256'],'source_sha256':context.result['assisted_inputs']['source_sha256'],
            'parent_output':str(cfg.output_dir.resolve()),'policy_evidence':deepcopy(context.assisted_evidence)}
        sampling={'mode':'linspace','dense_count':80,'dense_indices':list(range(80)),'sample_count':80,'requested_sample_count':80}
        frames=SimpleNamespace(timestamps=np.arange(80)*5.,sampling=sampling)
        if kind=='baseline':
            audits=[raw_audit(base[0],[40.,-80.05])]
            r=audits[0]['returned_track'];track=tuple(r[k] for k in ('indices','latlon','cosines'))
            ctx=SimpleNamespace(cfg=child_cfg,frames=frames,retrieval_audits=audits,vpr_track=track)
            comp=worker._comparison(ctx,child_cfg.output_dir)
            physical={'vpr_prior':{'track':[[round(float(t),2),round(float(ll[0]),6),round(float(ll[1]),6),round(float(s),4)]
                for t,ll,s in zip(comp['query']['times'],comp['decoded_reference_latlon'],comp['decoded_reference_cosines'])]}}
            route=baselines.vpr_track_only(physical).route_latlon
            physical['answer']={'route_latlon':route,'route_t_sec':frames.timestamps.tolist(),'mode':'track','method':'vpr_track_only'}
        else:
            request['metadata_union_baseline']=worker._metadata_baseline_binding(context,request)
            audits=[];calls=[];planned=[]
            for i,region in enumerate(regions):
                center=[40.+i*.01,-80.];a=raw_audit(region,center);audits.append(a)
                calls.append({'request':{'source':'mapillary','center':center,'radius_m':3000.,'cap':1500,'cache_dir':str(child_cfg.data_dir/f'mapillary_region_{i}'),
                    'model_name':'megaloc','device':'cpu','query_mode':'warp','query_batch_size':4},'reuse':{'status':'not_requested'}})
                planned.append({'status':'ok','center':center,'radius_m':3000.,'ref_cap':1500})
            ctx=SimpleNamespace(cfg=child_cfg,frames=frames,retrieval_audits=audits,assisted_public_reference_calls=calls,c2f_info=None,
                vpr_track=(np.arange(80),np.asarray(mc['decoded_reference_latlon']),np.asarray(mc['decoded_reference_cosines'])),
                vpr_center=(40.,-80.),result={'multi_region_retrieval':{'regions':planned}})
            assert worker._metadata_admission(ctx,request),ctx.result['assisted_expansion_gate']
            comp=worker._comparison(ctx,child_cfg.output_dir);selected=ctx.assisted_metadata_union_selected;route=selected['route_latlon']
            physical={**ctx.result,'answer':{'route_latlon':route,'route_t_sec':selected['route_timestamps'],'mode':'track',
                'method':'metadata_union_reference_mean','source':'assisted_v3.full_sequence.reference_mean'},
                'assisted_metadata_union':{'artifacts':selected['artifacts']}}
        write(child_cfg.output_dir/'request.json',request);write(child_cfg.output_dir/'result.json',physical)
        candidate={'status':'complete','route_latlon':route,'route_timestamps':frames.timestamps.tolist(),'comparison':comp,
            'cpu_runtime':{'device':'cpu','intra_op_threads':4,'inter_op_threads':1,'torch_version':'synthetic'},
            'actual_backbone':audits[0]['backbone'],'actual_model_state':{'sha256':audits[0]['actual_model_state_sha256']},'model_downloads_disabled':True}
        if kind=='expanded':candidate['retrieval_selection']={k:deepcopy(physical.get(k)) for k in ('multi_region_retrieval','city_cover_retrieval','vpr_coarse_to_fine','assisted_expansion_gate')}
        response={**worker._finish(child_cfg,request,candidate),'worker_returncode':0};returned[kind]=response;saved[kind]=ctx
        return response
    def finalize(context):
        context.result['answer']={**context.result['assisted_position'],'mode':'assisted','method':'assisted_policy','source':'result.assisted_position'}
        context.result['provenance']={'config':provenance.config_snapshot(cfg),'assisted_inputs':context.result['assisted_inputs'],'assisted_audit':context.result['assisted_audit']}
        write(cfg.output_dir/'result.json',context.result)
    stage.run_assisted(parent,finalize=finalize,run_child=child)
    return SimpleNamespace(parent=parent,children=returned,contexts=saved,finalize=finalize)


def full_audit(case):
    from scripts.benchmark import _provenance_of
    return audit.audit_assisted({'provenance':_provenance_of(case.parent.result)},{'meta':{'mode':'assisted','method':'assisted_policy'}},
        'Example','0:395',poi_issues=None,filter_issues=None,reader_issues=None,ocr_identity=None)


def test_actual_controller_new_selected_path_full_goal_transport_passes(tmp_path,monkeypatch):
    case=full_parent_case(tmp_path,monkeypatch)
    # Any accidentally applied legacy regional-winner adapter would reject the
    # new full path. The complete source/matrix/objective audit must accept it.
    report=full_audit(case)
    assert not report['issues'],report['issues']
    assert case.parent.result['assisted_audit']['selected_candidate']=='expanded'
    assert case.children['expanded']['comparison']['schema']=='assisted-raw-reference-comparison-v3'


def test_admitted_union_cannot_be_downgraded_to_legacy_final_child_with_replayed_parent_fallback(tmp_path,monkeypatch):
    from monocular_osm import baselines,assisted_policy_v3 as policy
    from monocular_osm.stages import assisted as stage
    case=full_parent_case(tmp_path,monkeypatch);child=case.children['expanded'];out=Path(child['candidate_artifact']['path']).parent
    old=deepcopy(child['retrieval_selection']['assisted_expansion_gate']['comparison'])
    track=[[round(float(t),2),round(float(ll[0]),6),round(float(ll[1]),6),round(float(s),4)]
        for t,ll,s in zip(old['query']['times'],old['decoded_reference_latlon'],old['decoded_reference_cosines'])]
    physical=json.loads((out/'result.json').read_text());physical['vpr_prior']={'track':track}
    old_route=baselines.vpr_track_only(physical).route_latlon
    child.update(comparison=old,route_latlon=old_route,route_timestamps=old['query']['times'])
    physical['answer']={'route_latlon':old_route,'route_t_sec':old['query']['times'],'mode':'track','method':'vpr_track_only'}
    write(out/'result.json',physical)
    payload={k:v for k,v in child.items() if k not in {'candidate_artifact','result_artifact','frozen_artifact','worker_returncode'}}
    write(out/'candidate.json',payload)
    frozen=json.loads((out/'child_frozen.json').read_text())
    for name in ('candidate.json','result.json'):frozen['artifacts'][name]=audit.file_sha(out/name)
    write(out/'child_frozen.json',frozen)
    for role,name in (('candidate_artifact','candidate.json'),('result_artifact','result.json'),('frozen_artifact','child_frozen.json')):
        child[role]['sha256']=audit.file_sha(out/name)
    ctx=case.parent;data=ctx.result['assisted_audit'];data['reached_stages'][-1]=stage.child_record('expanded',child,ctx.cfg.output_dir,
        ctx.result['assisted_inputs'],profile='v3')
    evidence={};traces=[];trace_inputs=[]
    for record in data['reached_stages']:
        evidence[record['kind']]=deepcopy(record['summary']);trace_inputs.append(deepcopy(evidence));traces.append(policy.decide_assisted_policy(evidence))
    assert traces[-1]['selected_candidate']=='baseline'
    data.update(evidence=evidence,decision_inputs=trace_inputs,decision_trace=traces,selected_candidate='baseline',
        selected_route_sha256=stage.digest(case.children['baseline']['route_latlon']),
        selected_child_artifact_sha256=case.children['baseline']['candidate_artifact']['sha256'])
    original=case.children['baseline'];ctx.result['assisted_position'].update(route_latlon=original['route_latlon'],route_t_sec=original['route_timestamps'],source='assisted_child:baseline')
    case.finalize(ctx);stage.freeze(ctx)
    report=full_audit(case)
    assert report['issues'],'internally admitted full union was silently lost into a reblessed legacy/fallback transcript'


def test_embedded_candidate_cannot_hide_a_stale_displayed_full_child_answer(tmp_path,monkeypatch):
    from monocular_osm.stages import assisted as stage
    case=full_parent_case(tmp_path,monkeypatch);child=case.children['expanded'];out=Path(child['candidate_artifact']['path']).parent
    physical=json.loads((out/'result.json').read_text());physical['candidate']=json.loads((out/'candidate.json').read_text())
    physical['answer']['route_latlon'][0][0]+=.001;write(out/'result.json',physical)
    frozen=json.loads((out/'child_frozen.json').read_text());frozen['artifacts']['result.json']=audit.file_sha(out/'result.json');write(out/'child_frozen.json',frozen)
    for role,name in (('result_artifact','result.json'),('frozen_artifact','child_frozen.json')):child[role]['sha256']=audit.file_sha(out/name)
    case.parent.result['assisted_audit']['reached_stages'][-1]=stage.child_record('expanded',child,case.parent.cfg.output_dir,
        case.parent.result['assisted_inputs'],profile='v3')
    case.finalize(case.parent);stage.freeze(case.parent)
    report=full_audit(case)
    assert report['issues'],'embedded candidate bypassed unconditional displayed-answer route binding'
