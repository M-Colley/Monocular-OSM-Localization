"""Synthetic four-freeze contracts; no real inputs, models or labels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from scripts import replay_heading_sequence as r


def saved(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);r.write(path,value)


def frozen(root,flags):
    path=root/'frozen.json'
    saved(path,{**flags,'artifacts':{str(p.relative_to(root)):r.sha(p) for p in root.rglob('*') if p.is_file()}})
    return {'freeze':str(path),'freeze_sha256':r.sha(path)}


def infer(family):return r.infer(family,'pin',r.sha(family/'inputs_frozen.json'))


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'source_identity',lambda *a,**k:{})
    monkeypatch.setattr(r,'check_headroom',lambda:{'available_physical_bytes':9*1024**3,'available_commit_bytes':9*1024**3})
    monkeypatch.delitem(sys.modules,'torch',raising=False)
    ids=np.array(['mapillary:1','mapillary:2']);ll=np.array([[52.,0.],[52.,.0001]])
    ordinals=120+4*np.arange(80,dtype=np.int64);times=ordinals.astype(float)/30
    pixels=[hashlib.sha256(str(i).encode()).hexdigest() for i in ordinals]
    video_sha='a'*64
    query={'video':{'path':'synthetic.mp4','sha256':video_sha},'registered_window':[4.,15.],
        'fps':30.,'stride':4,'planned_frame_ordinals':ordinals.tolist(),'native_shape_hw':[720,1280],
        'dense_count':80,'query_dense_indices':list(range(80))}
    shared={'ref_ids':ids,'ref_latlon':ll,'query_times':times,'query_indices':np.arange(80,dtype=np.int64),'frame_ordinals':ordinals}
    spec={};matrices={};desc={}
    for model in ['megaloc','image']:
        root=tmp_path/model;out=root/'inference';out.mkdir(parents=True)
        plan=root/('inputs/replay_plan.json' if model=='megaloc' else 'plan.json')
        saved(plan,{'query':query,**({'per_query_native_pixel_sha256':pixels} if model=='image' else {})})
        saved(root/'inputs_frozen.json',{'scoring_has_not_run':True,'artifacts':{str(plan.relative_to(root)):r.sha(plan)}})
        d=r.DIMENSIONS[model];q=np.zeros((240,d),np.float32);q[np.arange(240),np.arange(240)%2]=1
        refs=np.zeros((2,d),np.float32);refs[0,0]=refs[1,1]=1
        matrix=(q@refs.T).reshape(80,3,2).max(axis=1);matrices[model]=matrix;desc[model]=(q,refs)
        runtime={'input_freeze_sha256':r.sha(root/'inputs_frozen.json'),'per_query_native_pixel_sha256':pixels,'actual_query_times_sha256':r.core._array_sha(times)}
        if model=='megaloc':
            np.savez(out/'query_descriptors.npz',descriptors=q);np.savez(out/'reference_union.npz',embeddings=refs,ref_ids=ids,ref_latlon=ll)
            np.savez(out/'full_evidence.npz',**shared,raw_cosines=matrix)
            runtime.update(reference_embeddings_sha256=r.core._array_sha(refs),runtime={
                'backbone':{'name':'megaloc','hub':'gmberton/MegaLoc:5fe0dd697c4a70ba3e23607f6716ab3c606b16db'},
                'actual_model_state':{'sha256':'b'*64},'query_preprocessing':{'mode':'crops','views':3},
                'query_audit':{'query_descriptors_sha256':r.tensor_sha(q)}})
            saved(out/'runtime_and_lineage.json',runtime)
        else:
            np.save(out/'query_descriptors.npy',q,allow_pickle=False);np.save(out/'reference_descriptors.npy',refs,allow_pickle=False)
            np.savez(out/'full_evidence.npz',**shared,image_raw_cosines=matrix,megaloc_raw_cosines=matrices['megaloc'])
            runtime.update(query_descriptors_sha256=r.core._array_sha(q),reference_descriptors_sha256=r.core._array_sha(refs),actual_model_state_sha256='c'*64,
                canonical_preprocessing={'backbone':'dinov2','dimensions':d,'official_pin':'5d5e2d8cad11fe7ba699fdbd1f6b0a3c69dade18','network_input_color':'RGB'})
            saved(out/'runtime.json',runtime)
        spec[model]=frozen(out,{'status':'complete','scoring_has_not_run':True,'selected_variant':None})
    root=tmp_path/'motion';root.mkdir();fx=640/np.tan(np.radians(35.));K=np.array([[fx,0,640],[0,fx,360],[0,0,1.]])
    native={'centers':np.zeros((80,3)),'xz':np.zeros((80,2)),'valid':np.ones(80,bool),'n_inliers':np.full(80,20),
        'rotations':np.repeat(np.eye(3)[None],80,axis=0),'translations':np.zeros((80,3)),'timestamps':times,
        'frame_ordinals':ordinals,'native_frame_pixels_sha256':np.array(pixels),'K':K}
    np.savez(root/'native_motion.npz',**native)
    saved(root/'motion_registration.json',{'ground_truth_used':False,'legacy_cache_used':False,'inputs':{'video_sha256':video_sha,'window':[4.,15.]},'stride':4,'fps':30.,'native_shape_hw':[720,1280]})
    saved(root/'motion_summary.json',{'status':'complete','ground_truth_used':False,'motion_npz_sha256':r.sha(root/'native_motion.npz'),
        **{key:r.core._array_sha(native[field]) for key,field in [('timestamps_sha256','timestamps'),('frame_ordinals_sha256','frame_ordinals'),('native_frame_pixels_sha256','native_frame_pixels_sha256')]}})
    spec['motion']=frozen(root,{'ground_truth_used':False,'no_predictions_yet':True})
    root=tmp_path/'metadata';root.mkdir();plan=r.metadata.plan_requests(ids,ll,provenance={'source':'synthetic'})
    allrows={};outcomes=[]
    for batch in plan['batches']:
        name=f"batch_{batch['index']:04d}_attempt_1";p=root/'attempts'/name;p.mkdir(parents=True)
        data=[{'id':v.split(':')[1],'camera_type':'perspective','computed_compass_angle':float(int(v.split(':')[1])*45)} for v in batch['reference_ids']]
        body=json.dumps({'data':data}).encode();(p/'response.body').write_bytes(body);h=hashlib.sha256(body).hexdigest()
        saved(p/'request.json',{'params':batch['params']});saved(p/'response.json',{'status':200,'error_type':None,'parsed_successfully':True,'body_redacted_or_reencoded':False,'stored_body_sha256':h,'decoded_body_sha256':h})
        allrows.update(r.metadata.parse_response(body,batch['reference_ids']));outcomes.append({'index':batch['index'],'success':True,'attempts':[name]})
    saved(root/'metadata_plan.json',plan);saved(root/'all_reference_metadata.json',allrows);saved(root/'request_ledger.json',{'all_batches_successful':True,'batches':outcomes})
    spec['metadata']=frozen(root,{'status':'complete','all_ids_retained':True,'models_or_labels':False,'selected_answer':None})
    path=tmp_path/'spec.json';saved(path,spec)
    return path,tmp_path/'replay',spec,desc


def test_prepare_copies_four_bound_roles_and_preregisters_twelve_without_decode(fixture,monkeypatch):
    path,family,_,_=fixture
    monkeypatch.setattr(r.core,'decode_heading_tracks',lambda *a,**k:pytest.fail('prepare must not decode'))
    result=r.prepare(path,family,'pin');assert result['outputs']==12
    plan=r.load(family/'plan.json');assert plan['all_requested_outputs']==r.PROFILE['outputs']
    r.freeze(family/'inputs_frozen.json');r.base.verify_originals(r.load(family/'input_lineage.json')['copied'])
    assert not (family/'predictions').exists()


def test_real_synthetic_replay_freezes_all_twelve_and_refuses_resume(fixture):
    path,family,_,_=fixture;r.prepare(path,family,'pin');result=infer(family)
    assert result['outputs']==12;f=r.freeze(family/'predictions/frozen_predictions.json',result['freeze_sha256'])
    assert f['all_requested_outputs']==r.PROFILE['outputs'] and f['selected_variant'] is None
    assert len(list((family/'predictions').glob('prediction_*.json')))==12
    runtime=r.load(family/'predictions/runtime.json');assert not runtime['labels_opened'] and not runtime['new_forwards']
    with pytest.raises(ValueError,match='overwrit'):infer(family)


def test_second_model_failure_preserves_first_six_but_never_master(fixture,monkeypatch):
    path,family,_,_=fixture;r.prepare(path,family,'pin');original=r.core.decode_heading_tracks;calls=[]
    def decode(*a,**k):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('synthetic second model failure')
        return original(*a,**k)
    monkeypatch.setattr(r.core,'decode_heading_tracks',decode)
    with pytest.raises(RuntimeError):infer(family)
    assert len(list((family/'predictions').glob('prediction_megaloc_*.json')))==6
    assert not (family/'predictions/frozen_predictions.json').exists()
    assert len(r.load(family/'predictions/failure.json')['completed_outputs'])==6


def rewrite(path,value):
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf8')


def rebind(spec_path,spec,role):
    f=Path(spec[role]['freeze']);value=r.load(f)
    value['artifacts']={n:r.sha(f.parent/r.relative(n)) for n in value['artifacts']}
    rewrite(f,value);spec[role]['freeze_sha256']=r.sha(f);rewrite(spec_path,spec)


@pytest.mark.parametrize('fault',['extra_role','missing_role','tuning','bad_sha'])
def test_spec_disallows_per_clip_knobs_and_incomplete_roles(fixture,fault):
    path,family,spec,_=fixture
    if fault=='extra_role':spec['selected']={}
    elif fault=='missing_role':spec.pop('metadata')
    elif fault=='tuning':spec['megaloc']['temperature']=.1
    else:spec['image']['freeze_sha256']='not-a-digest'
    rewrite(path,spec)
    with pytest.raises(ValueError):r.prepare(path,family,'pin')
    assert not family.exists()


@pytest.mark.parametrize('role',['megaloc','image','metadata','motion'])
def test_partial_original_freeze_refuses_before_copy(fixture,role):
    path,family,spec,_=fixture;f=Path(spec[role]['freeze']);v=r.load(f)
    if role in ['megaloc','image','metadata']:v['status']='partial'
    else:v['ground_truth_used']=True
    rewrite(f,v);spec[role]['freeze_sha256']=r.sha(f);rewrite(path,spec)
    with pytest.raises(ValueError):r.prepare(path,family,'pin')
    assert not family.exists()


@pytest.mark.parametrize('fault',['pixel','intrinsics','ordinal'])
def test_rehashed_but_semantically_wrong_motion_cannot_bind_queries(fixture,fault):
    path,family,spec,_=fixture;root=Path(spec['motion']['freeze']).parent
    with np.load(root/'native_motion.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    if fault=='pixel':a['native_frame_pixels_sha256'][0]='f'*64
    elif fault=='intrinsics':a['K'][0,0]*=.99
    else:a['frame_ordinals']+=1
    np.savez(root/'native_motion.npz',**a);s=r.load(root/'motion_summary.json');s['motion_npz_sha256']=r.sha(root/'native_motion.npz')
    s['frame_ordinals_sha256']=r.core._array_sha(a['frame_ordinals']);s['native_frame_pixels_sha256']=r.core._array_sha(a['native_frame_pixels_sha256'])
    rewrite(root/'motion_summary.json',s);rebind(path,spec,'motion')
    with pytest.raises(ValueError):r.prepare(path,family,'pin')
    assert not family.exists()


def test_metadata_subset_is_not_accepted_as_all_gallery_metadata(fixture):
    path,family,spec,_=fixture;p=Path(spec['metadata']['freeze']).parent/'all_reference_metadata.json'
    v=r.load(p);v.pop(next(iter(v)));rewrite(p,v);rebind(path,spec,'metadata')
    with pytest.raises(ValueError,match='every gallery ID'):r.prepare(path,family,'pin')


def test_model_identity_is_checked_independently_of_descriptor_shape(fixture):
    path,family,spec,_=fixture;p=Path(spec['megaloc']['freeze']).parent/'runtime_and_lineage.json'
    v=r.load(p);v['runtime']['backbone']['name']='different_model';rewrite(p,v);rebind(path,spec,'megaloc')
    with pytest.raises(ValueError,match='model/crop identity'):r.prepare(path,family,'pin')


def test_replay_uses_every_reference_block_and_preserves_view_order():
    rng=np.random.default_rng(3);q=rng.normal(size=(240,8)).astype(np.float32);refs=rng.normal(size=(601,8)).astype(np.float32)
    q/=np.linalg.norm(q,axis=1,keepdims=True);refs/=np.linalg.norm(refs,axis=1,keepdims=True)
    expected=(q@refs.T).reshape(80,3,601)
    actual=r.recompute_views(q,refs,expected.max(axis=1))
    np.testing.assert_allclose(actual,expected,rtol=0,atol=r.PROFILE['cosine_replay_atol'])
    assert actual.dtype==np.float32 and actual.shape==(80,3,601)
    bad=expected.max(axis=1).copy();bad[20,600]+=.001
    with pytest.raises(ValueError,match='cosine replay'):r.recompute_views(q,refs,bad)


@pytest.mark.parametrize('fault',['float16','object','oversize_id','fortran','extra'])
def test_unsafe_headers_decline_before_numpy_array_materialization(tmp_path,monkeypatch,fault):
    path=tmp_path/'array.npz';data={'x':np.zeros((2,3),np.float32)};spec={'x':(6,'f',4)}
    if fault=='float16':data['x']=data['x'].astype(np.float16)
    elif fault=='object':data['x']=data['x'].astype(object)
    elif fault=='oversize_id':data={'x':np.array(['x'*129])};spec={'x':(1,'US',512)}
    elif fault=='fortran':data['x']=np.asfortranarray(data['x'])
    else:data['unexpected']=np.array([1])
    np.savez(path,**data)
    monkeypatch.setattr(r.np,'load',lambda *a,**k:pytest.fail('unsafe header materialized'))
    with pytest.raises(ValueError):r.npz_arrays(path,spec)


@pytest.mark.parametrize('member',['ground_truth/hidden.json','OFFLINE_scores.json','run2_official_registered_score.json'])
def test_label_or_saved_score_member_rejected_before_hash(tmp_path,monkeypatch,member):
    path=tmp_path/'frozen.json';saved(path,{'artifacts':{member:'a'*64}})
    monkeypatch.setattr(r,'sha',lambda *a:pytest.fail('forbidden member hashed'))
    with pytest.raises(ValueError,match='labels/saved scores'):r.freeze(path)


def test_foreign_loaded_script_refused_before_source_git_or_hash(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules,'scripts.foreign_heading_helper',SimpleNamespace(__file__=str(tmp_path/'foreign.py')))
    monkeypatch.setattr(r.base,'source_identity',lambda *a:pytest.fail('foreign source reached git'))
    with pytest.raises(ValueError,match='escaped source'):r.source_identity('pin')


def test_headroom_refuses_before_decoder(fixture,monkeypatch):
    path,family,_,_=fixture;r.prepare(path,family,'pin')
    monkeypatch.setattr(r,'check_headroom',lambda:r.require(False,'headroom low'))
    monkeypatch.setattr(r.core,'decode_heading_tracks',lambda *a,**k:pytest.fail('must not decode'))
    with pytest.raises(ValueError,match='headroom'):infer(family)
    assert not (family/'predictions').exists()


def test_original_mutation_after_first_decode_prevents_master(fixture,monkeypatch):
    path,family,spec,_=fixture;r.prepare(path,family,'pin');decode=r.core.decode_heading_tracks;changed=[]
    def mutate(*a,**k):
        result=decode(*a,**k)
        if not changed:
            (Path(spec['metadata']['freeze']).parent/'all_reference_metadata.json').write_text('{}');changed.append(True)
        return result
    monkeypatch.setattr(r.core,'decode_heading_tracks',mutate)
    with pytest.raises(ValueError,match='changed'):infer(family)
    assert not (family/'predictions/frozen_predictions.json').exists() and (family/'predictions/failure.json').exists()


def test_prepared_freeze_cannot_be_reblessed_between_commands(fixture):
    path,family,_,_=fixture;prepared=r.prepare(path,family,'pin')
    p=family/'plan.json';v=r.load(p);v['all_requested_outputs']=v['all_requested_outputs'][:6];rewrite(p,v)
    p=family/'inputs_frozen.json';v=r.load(p);v['artifacts']['plan.json']=r.sha(family/'plan.json');rewrite(p,v)
    with pytest.raises(ValueError,match='registered freeze changed'):
        r.infer(family,'pin',prepared['input_freeze_sha256'])
    assert not (family/'predictions').exists()


def test_in_run_generated_plan_and_manifest_mutation_cannot_be_reblessed(fixture,monkeypatch):
    path,family,_,_=fixture;prepared=r.prepare(path,family,'pin');decode=r.core.decode_heading_tracks;changed=[]
    def mutate(*a,**k):
        result=decode(*a,**k)
        if not changed:
            p=family/'plan.json';v=r.load(p);v['input_identity']['unregistered']='changed during decoding';rewrite(p,v)
            p=family/'inputs_frozen.json';v=r.load(p);v['artifacts']['plan.json']=r.sha(family/'plan.json');rewrite(p,v)
            changed.append(True)
        return result
    monkeypatch.setattr(r.core,'decode_heading_tracks',mutate)
    with pytest.raises(ValueError,match='registered freeze changed'):
        r.infer(family,'pin',prepared['input_freeze_sha256'])
    assert not (family/'predictions/frozen_predictions.json').exists()
    assert (family/'predictions/failure.json').exists()
