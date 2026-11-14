"""Six-output planning/freeze completeness and bounded failure regressions."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from scripts import replay_full_sequence as r


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(r,'source_identity',lambda expected:expected)
    monkeypatch.setattr(r,'check_headroom',lambda:{'available_physical_bytes':8*1024**3,'available_commit_bytes':8*1024**3})
    spec={'groups':[]}
    for name,models in [('night',{'megaloc':'megaloc_raw_cosines','image':'image_raw_cosines'}),('city',{'megaloc':'raw_cosines'})]:
        original=tmp_path/('original_'+name);original.mkdir()
        data={'ref_ids':np.array(['b','a','c']),'ref_latlon':np.array([[42.,-83.],[42.001,-83.],[42.,-83.001]]),'query_times':np.arange(4,dtype=float)}
        for i,key in enumerate(models.values()):data[key]=np.array([[.8,.3,.2],[.2,.6,.4],[.1,.3,.5],[.5,.3,.2]],np.float32)+i*.01
        np.savez(original/'full_evidence.npz',**data)
        r.write(original/'frozen_predictions.json',{'status':'complete','scoring_has_not_run':True,'artifacts':{'full_evidence.npz':r.sha(original/'full_evidence.npz')}})
        spec['groups'].append({'name':name,'freeze':str(original/'frozen_predictions.json'),'freeze_sha256':r.sha(original/'frozen_predictions.json'),'evidence':'full_evidence.npz','models':models})
    path=tmp_path/'spec.json';r.write(path,spec)
    return path,tmp_path/'isolated'


def prepared(tmp_path,monkeypatch):
    spec,family=setup(tmp_path,monkeypatch);r.prepare(spec,family,'pin');return spec,family


def test_prepare_freezes_both_groups_and_all_six_outputs_without_decoding(tmp_path,monkeypatch):
    spec,family=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(r.full,'decode_full_tracks',lambda *args:pytest.fail('planning must not decode'))
    result=r.prepare(spec,family,'pin');assert result['outputs']==6 and result['groups']==2
    plan=r.load(family/'plan.json');assert len(plan['all_requested_outputs'])==6
    assert [g['name'] for g in plan['groups']]==['night','city'] and plan['selected_variant'] is None
    r.verify_freeze(family/'inputs_frozen.json');r.verify_originals(r.load(family/'input_lineage.json')['copied'])
    assert not (family/'predictions').exists()


def test_actual_pure_replay_freezes_night_four_then_city_two_and_master(tmp_path,monkeypatch):
    spec,family=prepared(tmp_path,monkeypatch)
    result=r.infer(family,'pin');assert result['outputs']==6
    master=r.verify_freeze(family/'predictions/frozen_predictions.json')
    assert len(master['all_requested_outputs'])==6 and master['selected_variant'] is None
    assert len(list((family/'predictions/night').glob('prediction_*.json')))==4
    assert len(list((family/'predictions/city').glob('prediction_*.json')))==2
    for group in ['night','city']:
        frozen=r.verify_freeze(family/'predictions'/group/'frozen_predictions.json')
        assert frozen['profile_sha256']==r.full.PROFILE_SHA256
    assert r.load(family/'predictions/runtime.json')['new_forwards'] is False
    with pytest.raises(ValueError,match='overwrite'):r.infer(family,'pin')


@pytest.mark.parametrize('fault',['unknown_knob','duplicate_group','duplicate_matrix','unsafe_name','non_raw_role','missing_model'])
def test_spec_refuses_per_group_tuning_or_incomplete_ambiguous_roles(tmp_path,monkeypatch,fault):
    path,family=setup(tmp_path,monkeypatch);spec=r.load(path)
    if fault=='unknown_knob':spec['groups'][0]['top_k']=64
    if fault=='duplicate_group':spec['groups'][1]['name']='night'
    if fault=='duplicate_matrix':spec['groups'][0]['models']['image']='megaloc_raw_cosines'
    if fault=='unsafe_name':spec['groups'][0]['name']='../escape'
    if fault=='non_raw_role':spec['groups'][0]['models']['image']='chosen_scores'
    if fault=='missing_model':spec['groups'][0]['models']={}
    r.write(path,spec)
    with pytest.raises(ValueError):r.prepare(path,family,'pin')
    assert not family.exists()


def test_source_or_evidence_tamper_refused_before_family_creation(tmp_path,monkeypatch):
    path,family=setup(tmp_path,monkeypatch);group=r.load(path)['groups'][0]
    (Path(group['freeze']).parent/'full_evidence.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):r.prepare(path,family,'pin')
    assert not family.exists()


def test_reblessed_plan_cannot_omit_the_second_declared_group(tmp_path,monkeypatch):
    _,family=prepared(tmp_path,monkeypatch);plan=r.load(family/'plan.json')
    plan['groups']=plan['groups'][:1];plan['all_requested_outputs']=plan['all_requested_outputs'][:4];r.write(family/'plan.json',plan)
    freeze=r.load(family/'inputs_frozen.json');freeze['artifacts']['plan.json']=r.sha(family/'plan.json');r.write(family/'inputs_frozen.json',freeze)
    with pytest.raises(ValueError,match='omitted'):r.infer(family,'pin')
    assert not (family/'predictions').exists()


def test_second_group_failure_preserves_night_freeze_but_never_master_success(tmp_path,monkeypatch):
    _,family=prepared(tmp_path,monkeypatch);decode=r.full.decode_full_tracks;calls=[]
    def fail_later(*args):
        calls.append(True)
        if len(calls)==3:raise RuntimeError('later group interrupted')
        return decode(*args)
    monkeypatch.setattr(r.full,'decode_full_tracks',fail_later)
    with pytest.raises(RuntimeError):r.infer(family,'pin')
    assert (family/'predictions/night/frozen_predictions.json').exists()
    assert not (family/'predictions/frozen_predictions.json').exists()
    failure=r.load(family/'predictions/failure.json');assert failure['status']=='failed' and len(failure['completed_groups'])==1


def test_headroom_declines_before_first_decode_and_preserves_failure(tmp_path,monkeypatch):
    _,family=prepared(tmp_path,monkeypatch)
    monkeypatch.setattr(r,'check_headroom',lambda:r.require(False,'headroom low'))
    monkeypatch.setattr(r.full,'decode_full_tracks',lambda *args:pytest.fail('must not decode'))
    with pytest.raises(ValueError,match='headroom'):r.infer(family,'pin')
    assert (family/'predictions/failure.json').exists()
    assert not (family/'predictions/frozen_predictions.json').exists()


@pytest.mark.parametrize('fault',['float16','too_many_refs','oversized_id','object'])
def test_unsafe_headers_refused_before_array_materialization(tmp_path,monkeypatch,fault):
    path=tmp_path/'matrix.npz'
    data={'ref_ids':np.array(['a','b','c']),'ref_latlon':np.zeros((3,2)),'query_times':np.arange(4.),'raw_cosines':np.zeros((4,3),np.float32)}
    if fault=='float16':data['raw_cosines']=data['raw_cosines'].astype(np.float16)
    if fault=='too_many_refs':data['raw_cosines']=np.zeros((4,10001),np.float32)
    if fault=='oversized_id':data['ref_ids']=np.array(['x'*129]*3)
    if fault=='object':data['raw_cosines']=data['raw_cosines'].astype(object)
    np.savez(path,**data)
    monkeypatch.setattr(r.np,'load',lambda *a,**k:pytest.fail('must refuse from header first'))
    with pytest.raises(ValueError):r.arrays(path,{'model':'raw_cosines'})


def test_original_mutation_after_decode_refuses_group_and_master_freeze(tmp_path,monkeypatch):
    _,family=prepared(tmp_path,monkeypatch);decode=r.full.decode_full_tracks
    original=r.load(family/'plan.json')['groups'][0]['original_evidence'];changed=[]
    def mutate(*args):
        result=decode(*args)
        if not changed:Path(original).write_bytes(b'changed source');changed.append(True)
        return result
    monkeypatch.setattr(r.full,'decode_full_tracks',mutate)
    with pytest.raises(ValueError,match='changed'):r.infer(family,'pin')
    assert not (family/'predictions/night/frozen_predictions.json').exists()
    assert not (family/'predictions/frozen_predictions.json').exists()
    assert (family/'predictions/failure.json').exists()


def test_prepare_never_overwrites_or_reblesses_existing_family(tmp_path,monkeypatch):
    spec,family=prepared(tmp_path,monkeypatch);before=(family/'inputs_frozen.json').read_bytes()
    with pytest.raises(ValueError,match='isolated'):r.prepare(spec,family,'pin')
    assert (family/'inputs_frozen.json').read_bytes()==before


def test_model_roles_remain_distinct_and_never_merge_encoder_metadata(tmp_path,monkeypatch):
    _,family=prepared(tmp_path,monkeypatch);r.infer(family,'pin')
    runtime=r.load(family/'predictions/night/runtime.json')['models']
    assert runtime['image']['source_matrix_key']=='image_raw_cosines'
    assert runtime['megaloc']['source_matrix_key']=='megaloc_raw_cosines'
    assert runtime['image']['raw_matrix_sha256']!=runtime['megaloc']['raw_matrix_sha256']
    assert not any('backbone' in value for value in runtime.values())
