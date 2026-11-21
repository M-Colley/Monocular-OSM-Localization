"""V3 worker source/route/freeze integration on synthetic raw matrices only."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from monocular_osm import assisted_worker as w, assisted_metadata_union as component
from monocular_osm import assisted_comparison as comparison
from tests.test_assisted_comparison import fixed_case
from tests.test_assisted_metadata_union import seal_control,memory,cpu


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8')


def fixture(tmp_path,monkeypatch,*,profile='v3',weak=False):
    regions,bc,bs,mc,ms,base=fixed_case()
    if weak:
        for region in regions:
            arrays=region['arrays'];raw=arrays['raw_similarities'].copy();raw[:,:2]=np.where(raw[:,:2]>.8,.15,raw[:,:2])
            decoded=component.core.full.decode_full_tracks(raw,arrays['ref_ids'],arrays['ref_latlon'],arrays['query_times'])['outputs']['raw']
            chosen=np.asarray(decoded['chosen_ref_indices'],int);arrays['raw_similarities']=raw;arrays['decoded_ref_indices']=chosen
            arrays['decoded_cosines']=raw[np.arange(80),chosen];arrays['returned_cosines']=arrays['decoded_cosines'].astype(float)
            arrays['returned_latlon']=arrays['ref_latlon'][chosen]
    execution=tmp_path/'parent/children/synthetic_execution';base_root=execution/'baseline'
    artifact,_,expected=seal_control(base_root,base,bc,bs,'baseline')
    old=Path(artifact['path']);freeze_path=old.with_name('child_frozen.json');old.rename(freeze_path)
    write(base_root/'result.json',{'answer':{'route_latlon':bc['decoded_reference_latlon']}})
    frozen=json.loads(freeze_path.read_text());frozen['artifacts']['result.json']=w.file_sha256(base_root/'result.json');write(freeze_path,frozen)
    child=json.loads((base_root/'candidate.json').read_text())
    child.update(candidate_artifact={'path':str((base_root/'candidate.json').resolve()),'sha256':w.file_sha256(base_root/'candidate.json')},
        frozen_artifact={'path':str(freeze_path.resolve()),'sha256':w.file_sha256(freeze_path)},
        result_artifact={'path':str((base_root/'result.json').resolve()),'sha256':w.file_sha256(base_root/'result.json')},worker_returncode=0)
    output=execution/'expanded';output.mkdir()
    cfg=SimpleNamespace(output_dir=output,data_dir=tmp_path/'data/clip',assisted_profile=profile,vo_start_sec=0.,vo_end_sec=395.)
    request={'kind':'expanded','config':{'vo_start_sec':0.,'vo_end_sec':395.},'execution_id':'synthetic_execution',
        'video_sha256':expected['video_sha256'],'source_sha256':expected['source_sha256'],
        'parent_output':str(execution.parent.parent),'policy_evidence':{'baseline':comparison.augment_baseline_summary(child['comparison'],bs),'cues':{'status':'failed'}}}
    parent=SimpleNamespace(cfg=cfg,assisted_child_root=execution,assisted_baseline_child=child)
    request['metadata_union_baseline']=w._metadata_baseline_binding(parent,request)
    assert request['metadata_union_baseline']['status']=='complete'
    audits=[];calls=[];planned=[]
    for i,region in enumerate(regions):
        audit={**deepcopy(region['audit']),**deepcopy(region['arrays'])};arrays=region['arrays']
        audit['ref_embeddings_sha256']='e'*64
        audit['returned_track']={k:arrays['returned_'+k].copy() for k in ('indices','latlon','cosines')}
        for key in list(audit):
            if key.startswith('returned_') and key!='returned_track':audit.pop(key)
        center=[40.+i*.01,-80.]
        audit['reference_pool'].update(center=list(center),radius_m=3000.,cap=1500)
        call={'request':{'source':'mapillary','center':list(center),'radius_m':3000.,'cap':1500,
            'cache_dir':str(cfg.data_dir/f'mapillary_region_{i}'),'model_name':'megaloc','device':'cpu','query_mode':'warp','query_batch_size':4},
            'reuse':{'status':'not_requested'}}
        audits.append(audit);calls.append(call);planned.append({'status':'ok','center':center,'radius_m':3000.,'ref_cap':1500})
    sampling={'mode':'linspace','dense_count':80,'dense_indices':list(range(80)),'sample_count':80,'requested_sample_count':80}
    frames=SimpleNamespace(timestamps=np.arange(80)*5.,sampling=sampling,close=lambda:None)
    track=(np.arange(80),np.asarray(mc['decoded_reference_latlon']),np.asarray(mc['decoded_reference_cosines']))
    ctx=SimpleNamespace(cfg=cfg,frames=frames,retrieval_audits=audits,assisted_public_reference_calls=calls,
        vpr_track=track,vpr_center=(40.,-80.),result={'multi_region_retrieval':{'regions':planned}})
    evaluate=component.evaluate_metadata_union
    def bounded(base,metadata,**kwargs):return evaluate(base,metadata,resource_gate=memory,numerical_context=cpu,**kwargs)
    monkeypatch.setattr(component,'evaluate_metadata_union',bounded)
    return ctx,request,parent


def test_v3_binding_uses_actual_saved_frozen_baseline_and_original_v1_summary(tmp_path,monkeypatch):
    _,request,parent=fixture(tmp_path,monkeypatch)
    binding=request['metadata_union_baseline']
    assert binding['original_summary']['score_identity']==request['policy_evidence']['baseline']['score_identity']
    assert binding['expected']['execution_id']=='synthetic_execution'
    assert binding['candidate_artifact']==parent.assisted_baseline_child['candidate_artifact']
    assert 'metadata_union_baseline' not in binding['original_summary']


@pytest.mark.parametrize('bad',['rc','payload','source','window','execution'])
def test_wrong_saved_baseline_is_honestly_unavailable(tmp_path,monkeypatch,bad):
    _,request,parent=fixture(tmp_path,monkeypatch)
    if bad=='rc':parent.assisted_baseline_child['worker_returncode']=9
    if bad=='payload':parent.assisted_baseline_child['route_latlon'][0][0]+=.01
    if bad=='source':request['source_sha256']={'other':'f'*64}
    if bad=='window':request['config']['vo_end_sec']=400.
    if bad=='execution':request['execution_id']='different'
    assert w._metadata_baseline_binding(parent,request)['status']=='unavailable'


def test_live_calls_attach_exact_actual_request_and_reuse_before_save(tmp_path,monkeypatch):
    ctx,_,_=fixture(tmp_path,monkeypatch)
    assert all('public_cache_request' not in audit for audit in ctx.retrieval_audits)
    proof=w._attach_metadata_calls(ctx)
    assert proof['status']=='complete' and proof['call_count']==proof['audit_count']==2
    for audit,call in zip(ctx.retrieval_audits,ctx.assisted_public_reference_calls):
        assert audit['public_cache_request']==call['request'] and audit['public_cache_reuse']==call['reuse']


def test_stale_control_inputs_are_refused_before_any_overwrite(tmp_path,monkeypatch):
    ctx,request,_=fixture(tmp_path,monkeypatch)
    directory=ctx.cfg.output_dir/'metadata_admission';directory.mkdir();path=directory/'control_request.json'
    path.write_bytes(b'preserve original failed input')
    with pytest.raises(ValueError,match='stale'):w._seal_metadata_control(ctx,request,{}, {})
    assert path.read_bytes()==b'preserve original failed input' and not (directory/'control_candidate.json').exists()


@pytest.mark.parametrize('bad',['missing','reordered','geometry','preprocessing','stale'])
def test_live_mapping_never_guesses_or_partially_attaches(tmp_path,monkeypatch,bad):
    ctx,_,_=fixture(tmp_path,monkeypatch)
    if bad=='missing':ctx.assisted_public_reference_calls.pop()
    if bad=='reordered':ctx.assisted_public_reference_calls.reverse()
    if bad=='geometry':ctx.assisted_public_reference_calls[1]['request']['center'][0]+=.1
    if bad=='preprocessing':ctx.assisted_public_reference_calls[1]['request']['query_mode']='crops'
    if bad=='stale':ctx.retrieval_audits[1]['public_cache_request']={'stale':True}
    result=w._attach_metadata_calls(ctx)
    assert result['status']=='unavailable' and 'public_cache_request' not in ctx.retrieval_audits[0]


def test_all_research_outputs_and_control_freeze_exist_before_new_route_applies(tmp_path,monkeypatch):
    ctx,request,_=fixture(tmp_path,monkeypatch);original=tuple(v.copy() for v in ctx.vpr_track)
    writer=w._write_result_json;observed=[]
    def checked(path,value):
        if Path(path).name=='component_frozen.json':
            assert all(np.array_equal(old,new) for old,new in zip(original,ctx.vpr_track))
            assert Path(path).parent.joinpath('prediction_raw.json').is_file()
            assert Path(path).parent.joinpath('prediction_reference_mean.json').is_file()
            observed.append(True)
        return writer(path,value)
    monkeypatch.setattr(w,'_write_result_json',checked)
    assert w._metadata_admission(ctx,request) is True,ctx.result.get('assisted_expansion_gate')
    assert observed==[True] and (ctx.cfg.output_dir/'metadata_admission/control_frozen.json').is_file()
    frozen=json.loads((ctx.cfg.output_dir/'metadata_union/component_frozen.json').read_text())
    assert frozen['both_research_outputs'] is True and frozen['sole_primary']=='reference_mean'
    assert len(ctx.vpr_track[0])==80 and not np.array_equal(ctx.vpr_track[1],original[1])
    c=w._comparison(ctx,ctx.cfg.output_dir)
    assert c['schema']==comparison.SCHEMA and 'inner_decoder' not in c
    assert ctx.vpr_track[1].tolist()==c['route_latlon']
    assert comparison.union_summary(c,c['route_latlon'],c['route_timestamps'])['dominant_coordinate_fraction']==.25


def test_declined_reference_mean_never_uses_raw_diagnostic_and_allows_municipality(tmp_path,monkeypatch):
    ctx,request,_=fixture(tmp_path,monkeypatch,weak=True);original=tuple(v.copy() for v in ctx.vpr_track)
    assert w._metadata_admission(ctx,request) is False
    assert all(np.array_equal(old,new) for old,new in zip(original,ctx.vpr_track))
    assert not hasattr(ctx,'assisted_metadata_union_selected')
    record=ctx.result['assisted_expansion_gate'];assert record['municipality_skipped'] is False
    assert record['metadata_union']['decision']['selected_candidate']=='baseline'
    path=ctx.cfg.output_dir/'metadata_union'
    assert (path/'prediction_raw.json').is_file() and (path/'prediction_reference_mean.json').is_file()
    assert w._comparison(ctx,ctx.cfg.output_dir)['inner_decoder']['actual']=='viterbi'


def test_single_regional_plan_keeps_all_complete_coarse_and_fine_pools(tmp_path,monkeypatch):
    ctx,request,_=fixture(tmp_path,monkeypatch)
    ctx.result['multi_region_retrieval']['regions']=ctx.result['multi_region_retrieval']['regions'][:1]
    ctx.cfg.vpr_coarse_to_fine=True
    ctx.c2f_info={'pass1_center':[40.,-80.],'pass2_center':[40.01,-80.],'tight_radius_m':3000.,'kept':'pass-2'}
    load=component.load_legacy_control;seen=[]
    def checked(frozen_artifact,candidate_relative,*args,**kwargs):
        if candidate_relative=='control_candidate.json':
            frozen=json.loads(Path(frozen_artifact['path']).read_text());assert 'union_plan.json' in frozen['artifacts']
            plan=json.loads(Path(frozen_artifact['path']).with_name('union_plan.json').read_text())
            assert plan['regional_plan_count']==1 and plan['requested_pool_count']==2
            assert plan['regional_to_audit_indices']==[0] and plan['ordinary_refinement_call_indices']==[1]
            seen.append(True)
        return load(frozen_artifact,candidate_relative,*args,**kwargs)
    monkeypatch.setattr(component,'load_legacy_control',checked)
    assert w._metadata_admission(ctx,request) is True
    assert seen==[True]
    plan=ctx.result['assisted_expansion_gate']['metadata_union_plan']['plan']
    assert len(plan['ordered_requests'])==2 and plan['c2f_trace']==ctx.c2f_info
    assert w.file_sha256(ctx.cfg.output_dir/'metadata_admission/union_plan.json')==w.file_sha256(ctx.cfg.output_dir/'metadata_union/union_plan.json')


@pytest.mark.parametrize('bad',['missing_call','overbudget','unexplained_extra'])
def test_all_actual_plan_is_never_filtered_to_fit_or_cover_missing_requests(tmp_path,monkeypatch,bad):
    ctx,request,_=fixture(tmp_path,monkeypatch);original=tuple(v.copy() for v in ctx.vpr_track)
    if bad=='missing_call':
        region=deepcopy(ctx.result['multi_region_retrieval']['regions'][0]);region['center']=[41.,-80.]
        ctx.result['multi_region_retrieval']['regions'].append(region)
    if bad=='overbudget':
        for i in range(2,7):
            audit=deepcopy(ctx.retrieval_audits[0]);call=deepcopy(ctx.assisted_public_reference_calls[0]);region=deepcopy(ctx.result['multi_region_retrieval']['regions'][0])
            center=[40.+i*.01,-80.];audit['reference_pool']['center']=list(center);call['request']['center']=list(center);region['center']=list(center)
            ctx.retrieval_audits.append(audit);ctx.assisted_public_reference_calls.append(call);ctx.result['multi_region_retrieval']['regions'].append(region)
    if bad=='unexplained_extra':ctx.result['multi_region_retrieval']['regions']=ctx.result['multi_region_retrieval']['regions'][:1]
    monkeypatch.setattr(component,'evaluate_metadata_union',lambda *_args,**_kwargs:pytest.fail('invalid full plan cannot run DP'))
    assert w._metadata_admission(ctx,request) is False
    assert all(np.array_equal(old,new) for old,new in zip(original,ctx.vpr_track))
    plan=ctx.result['assisted_expansion_gate']['metadata_union_plan']['plan']
    assert plan['status']=='unavailable' and len(plan['ordered_requests'])==len(ctx.assisted_public_reference_calls)
    if bad=='overbudget':assert plan['requested_pool_count']==7
    assert not (ctx.cfg.output_dir/'metadata_admission/control_frozen.json').exists()


@pytest.mark.parametrize('failure',['binding','partial','write','final_write','postfreeze_mutation'])
def test_failure_keeps_legacy_route_and_fallback_before_adoption(tmp_path,monkeypatch,failure):
    ctx,request,_=fixture(tmp_path,monkeypatch);original=tuple(v.copy() for v in ctx.vpr_track)
    if failure=='binding':request['metadata_union_baseline']['status']='unavailable'
    if failure=='partial':ctx.result['multi_region_retrieval']['regions'][1]['status']='failed'
    writer=w._write_result_json
    def faulty(path,value):
        path=Path(path)
        if failure=='write' and path.name=='prediction_reference_mean.json':raise OSError('synthetic write failure')
        if failure=='final_write' and path==ctx.cfg.output_dir/'metadata_admission/admission.json':raise OSError('synthetic final failure')
        result=writer(path,value)
        if failure=='postfreeze_mutation' and path==ctx.cfg.output_dir/'metadata_admission/admission.json':
            target=ctx.cfg.output_dir/'metadata_union/prediction_raw.json'
            if target.exists():target.write_text('{}')
        return result
    monkeypatch.setattr(w,'_write_result_json',faulty)
    assert w._metadata_admission(ctx,request) is False
    assert all(np.array_equal(old,new) for old,new in zip(original,ctx.vpr_track))
    assert not hasattr(ctx,'assisted_metadata_union_selected')
    assert ctx.result['assisted_expansion_gate']['municipality_skipped'] is False


@pytest.mark.parametrize('bad',['route','times','artifact'])
def test_final_full_comparison_refuses_stale_adopted_route_or_proof(tmp_path,monkeypatch,bad):
    ctx,request,_=fixture(tmp_path,monkeypatch);assert w._metadata_admission(ctx,request)
    if bad=='route':ctx.vpr_track[1][0,0]+=.001
    if bad=='times':ctx.frames.timestamps[0]+=.01
    if bad=='artifact':(ctx.cfg.output_dir/'metadata_union/comparison_reference_mean.json').write_text('{}')
    with pytest.raises(ValueError):w._comparison(ctx,ctx.cfg.output_dir)


def test_pipeline_child_answer_is_exact_metric_median5_without_ordinary_resmoothing(tmp_path,monkeypatch):
    from monocular_osm import pipeline,position,public_reference_cache as cache
    ctx,request,_=fixture(tmp_path,monkeypatch);calls=ctx.assisted_public_reference_calls
    monkeypatch.setattr(pipeline.RunContext,'create',lambda cfg:ctx)
    @contextmanager
    def scope(*args,**kwargs):yield calls
    monkeypatch.setattr(cache,'track_scope',scope)
    def stages(context):
        assert context.assisted_public_reference_calls is calls
        assert context.assisted_city_admission() is True
        # Ordinary track reporting rounds and smooths. A deliberately different
        # ordinary answer proves the worker preserves the selected DP artifact.
        distorted=(context.vpr_track[1]+.001).tolist()
        context.result['answer']={'route_latlon':distorted,'route_t_sec':context.frames.timestamps.tolist()}
        return context.result
    monkeypatch.setattr(pipeline,'_run_pipeline_stages',stages)
    monkeypatch.setattr(position,'plot_answer',lambda answer,path,**kwargs:Path(path).write_bytes(b'selected map'))
    candidate=w._pipeline_child(ctx.cfg,'expanded',request)
    saved=ctx.assisted_metadata_union_selected
    assert candidate['status']=='complete' and candidate['route_latlon']==saved['route_latlon']
    assert candidate['route_timestamps']==saved['route_timestamps']
    result=json.loads((ctx.cfg.output_dir/'result.json').read_text())
    assert result['answer']['route_latlon']==candidate['route_latlon']
    assert result['answer']['method']=='metadata_union_reference_mean'
    assert comparison.union_summary(candidate['comparison'],candidate['route_latlon'],candidate['route_timestamps'])['score_kind']=='raw_cosine'


def test_exact_original_legacy_metadata_admission_skips_union_and_baseline_loading(tmp_path,monkeypatch):
    from tests.test_metadata_union import make_region,coords
    ctx,request,_=fixture(tmp_path,monkeypatch)
    region=make_region('legacy',np.eye(80,dtype=np.float32)*.8+.1,coords(np.arange(80)*10),[f'legacy_{i}' for i in range(80)],chosen=np.arange(80))
    arrays=region['arrays'];audit={**region['audit'],**arrays}
    audit['returned_track']={name:arrays['returned_'+name] for name in ('indices','latlon','cosines')}
    audit['reference_pool'].update(center=[40.,-80.],radius_m=3000.,cap=1500)
    ctx.retrieval_audits=[audit];ctx.assisted_public_reference_calls=ctx.assisted_public_reference_calls[:1]
    ctx.result['multi_region_retrieval']['regions']=ctx.result['multi_region_retrieval']['regions'][:1]
    ctx.vpr_track=(arrays['query_indices'],arrays['returned_latlon'],arrays['returned_cosines'])
    original=tuple(v.copy() for v in ctx.vpr_track)
    monkeypatch.setattr(w,'_load_metadata_baseline',lambda *_:pytest.fail('legacy priority cannot load new controls'))
    monkeypatch.setattr(component,'evaluate_metadata_union',lambda *_args,**_kwargs:pytest.fail('legacy priority cannot run union'))
    request['metadata_union_baseline']={'status':'unavailable'}
    assert w._metadata_admission(ctx,request) is True
    assert all(np.array_equal(old,new) for old,new in zip(original,ctx.vpr_track))
    assert not (ctx.cfg.output_dir/'metadata_union').exists() and not hasattr(ctx,'assisted_metadata_union_selected')
    assert ctx.result['assisted_expansion_gate']['legacy_metadata_decision']['selected_candidate']=='expanded'


@pytest.mark.parametrize('profile',['v1','v2','v3'])
def test_only_v3_expanded_request_adds_frozen_baseline_transport(tmp_path,monkeypatch,profile):
    from monocular_osm.pipeline import PipelineConfig
    from monocular_osm import public_reference_cache as cache
    video=tmp_path/'video.mp4';video.write_bytes(b'unit encoded video')
    cfg=PipelineConfig(url='',city='Synthetic City',answer='assisted',assisted_profile=profile,
        data_dir=tmp_path/'data/clip',output_dir=tmp_path/'out',video_path=video)
    parent=SimpleNamespace(cfg=cfg,video_path=video,assisted_child_root=cfg.output_dir/'children/execution',assisted_evidence={},
        result={'assisted_inputs':{'execution_id':'execution','video':{'sha256':w.file_sha256(video)},'source_sha256':{}}})
    monkeypatch.setattr(w,'_commit_headroom',lambda:9*1024**3)
    def reuse(source,cfg,**kwargs):
        path=cfg.output_dir/'public_reference_reuse.json';write(path,{'status':'unavailable'})
        return {'path':str(path),'sha256':w.file_sha256(path)}
    monkeypatch.setattr(cache,'prepare_reuse',reuse)
    bindings=[]
    def binding(*args):bindings.append(True);return {'status':'unavailable','reason':'synthetic no baseline'}
    monkeypatch.setattr(w,'_metadata_baseline_binding',binding)
    def popen(command,**kwargs):
        path=Path(command[-1]);request=json.loads(path.read_text())
        assert ('metadata_union_baseline' in request) is (profile=='v3')
        path.with_name('response.json').write_text(json.dumps({'status':'unavailable'}))
        return SimpleNamespace(wait=lambda:0)
    monkeypatch.setattr(w.subprocess,'Popen',popen)
    assert w.run_child('expanded',parent)['worker_returncode']==0
    assert bool(bindings) is (profile=='v3')


def test_actual_overlap_rows_are_saved_inside_component_freeze_before_adoption(tmp_path,monkeypatch):
    from tests.test_assisted_metadata_union import controls,descriptor_fixture
    from monocular_osm import assisted_policy_v3 as policy
    base,metadata=controls(tmp_path/'sources',overlap=True)
    def loader(region,cfg):
        _,pool,_=descriptor_fixture(tmp_path/'descriptors'/region['key'],region)
        return component.load_descriptor_rows(region,cfg,describe=lambda *_:pool)
    result=component.evaluate_metadata_union(base,metadata,requested_pool_count=3,reference_cfg=object(),
        descriptor_loader=loader,resource_gate=memory,numerical_context=cpu)
    assert result['status']=='complete'
    cfg=SimpleNamespace(output_dir=tmp_path/'worker');cfg.output_dir.mkdir()
    ctx=SimpleNamespace(cfg=cfg)
    evidence={'baseline':result['baseline_summary'],'cues':{'status':'failed'},'expanded':result['summary']}
    artifacts=w._save_metadata_union(ctx,result,evidence,policy.decide_assisted_policy(evidence))
    w._verify_metadata_union_artifacts(artifacts)
    root=cfg.output_dir/'metadata_union';saved=json.loads((root/'controls.json').read_text())['descriptor_evidence']
    assert len(saved)==2
    for record,original in zip(saved,result['descriptor_evidence']):
        path=Path(record['artifact']['path']);assert w.file_sha256(path)==record['artifact']['sha256']
        with np.load(path,allow_pickle=False) as data:np.testing.assert_array_equal(data['rows'],original['rows'])
        assert record['rows_sha256']==w._array_sha(original['rows'])
