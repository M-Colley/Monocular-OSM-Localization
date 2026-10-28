"""Bounded real pure DP plus mocked-control/cache/resource tests; no models."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
import pytest
from monocular_osm import assisted_metadata_union as a, assisted_policy_v3 as policy
from tests.test_assisted_comparison import fixed_case
from tests.test_metadata_union import make_region,comparison,summary,coords


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False),encoding='utf8')


def seal_control(root,regions,c,s,kind,*,candidate_edit=None):
    c=deepcopy(c);regions=deepcopy(regions);records=[]
    for i,region in enumerate(regions):
        directory=root/'retrieval_audit';directory.mkdir(parents=True,exist_ok=True)
        arrays=directory/f'{i:04d}.npz';np.savez(arrays,**region['arrays'])
        meta=deepcopy(region['audit'])
        n=len(region['arrays']['ref_ids']);reference_rows=np.zeros((n,8448),np.float32);reference_rows[np.arange(n),np.arange(n)]=1.
        meta.setdefault('ref_embeddings_sha256',tensor_sha(reference_rows))
        meta['array_artifact']={'path':str(arrays.resolve()),'sha256':a.sha(arrays),
            'arrays':{k:{'sha256':a.core.full._array_sha(v),'shape':list(v.shape),'dtype':v.dtype.str} for k,v in region['arrays'].items()}}
        path=arrays.with_suffix('.json');write(path,meta)
        records.extend({'path':str(p.resolve()),'sha256':a.sha(p)} for p in (arrays,path))
    c['audit_artifacts']=records
    first=regions[c['selected_audit_rows'][0][0]]['audit']
    c['audited_identity']=deepcopy(first['comparison_identity']);c['audited_identity_sha256']=first['comparison_identity_sha256']
    c['source_sampling']={'mode':'linspace','dense_count':80,'dense_indices':list(range(80)),'sample_count':80,'requested_sample_count':80}
    expected={'kind':kind,'execution_id':'synthetic_execution','video_sha256':'a'*64,'window':[0.,395.],
        'source_sha256':{'main.py':'d'*64}}
    request={**expected,'config':{'vo_start_sec':0.,'vo_end_sec':395.}}
    candidate={**expected,'status':'complete','route_latlon':deepcopy(c['decoded_reference_latlon']),
        'route_timestamps':c['query']['times'],'comparison':c}
    if candidate_edit:candidate_edit(candidate)
    write(root/'request.json',request);write(root/'candidate.json',candidate)
    frozen={'kind':kind,'scoring_has_not_run':True,'video_sha256':expected['video_sha256'],'source_sha256':expected['source_sha256'],
        'artifacts':{p.relative_to(root).as_posix():a.sha(p) for p in root.rglob('*') if p.is_file()}}
    write(root/'control_frozen.json',frozen)
    return {'path':str((root/'control_frozen.json').resolve()),'sha256':a.sha(root/'control_frozen.json')},s,expected


def controls(tmp_path,*,overlap=False,admitted=False,mixed=False,descriptor_conflict=False):
    regions,bc,bs,mc,ms,base=fixed_case()
    if mixed:
        extra=make_region('fine',np.full((80,1),.11),coords([-8000]),['fine_base'])
        base.append(extra)
        for q in range(80):
            bc['selected_audit_rows'][q]=[q%2,q]
            if q%2:
                for key,value in [('decoded_reference_ids','mapillary:fine_base'),('decoded_reference_cosines',float(np.float32(.11))),
                    ('decoded_reference_latlon',extra['arrays']['ref_latlon'][0].tolist())]:bc[key][q]=value
        bs=summary(bc)
    if overlap:regions.append(deepcopy(regions[0]));regions[-1]['key']='overlap'
    if descriptor_conflict:
        n=len(regions[-1]['arrays']['ref_ids']);rows=np.zeros((n,8448),np.float32);rows[np.arange(n),np.arange(n)+1]=1.
        regions[-1]['audit']['ref_embeddings_sha256']=tensor_sha(rows)
    if admitted:
        # Eighty distinct coordinates are needed for the unchanged collapse gate.
        ids=np.array([f'mapillary:admitted_{i}' for i in range(80)])
        region=make_region('legacy_admitted',np.eye(80,dtype=np.float32)*.8+.1,coords(np.arange(80)),[str(i).split(':')[-1] for i in ids],chosen=np.arange(80))
        regions=[region]
        mc=comparison(region['arrays']['raw_similarities'][np.arange(80),np.arange(80)],region['arrays']['ref_latlon'],region['arrays']['ref_ids']);ms=summary(mc,True)
    def load(root,rs,c,s,kind):
        artifact,s,expected=seal_control(root,rs,c,s,kind)
        return a.load_legacy_control(artifact,'candidate.json',s,expected)
    return load(tmp_path/'baseline',base,bc,bs,'baseline'),load(tmp_path/'metadata',regions,mc,ms,'expanded')


@contextmanager
def cpu():yield [{'num_threads':2},{'num_threads':2}]
def memory():return {'available_physical_bytes':20*1024**3,'available_commit_bytes':20*1024**3}


def run(base,metadata,**kwargs):
    return a.evaluate_metadata_union(base,metadata,requested_pool_count=len(metadata.regions),resource_gate=memory,numerical_context=cpu,**kwargs)


def test_import_is_model_free():
    subprocess.run([sys.executable,'-c',"import sys; from monocular_osm import assisted_metadata_union; assert 'torch' not in sys.modules"],check=True)


def test_source_bound_full80_component_and_v3_policy_select_same_fixed_primary(tmp_path):
    base,metadata=controls(tmp_path);result=run(base,metadata)
    assert result['status']=='complete' and result['summary']['candidate_kind']=='metadata_union_reference_mean'
    assert result['summary']['dominant_coordinate_fraction']==.25
    assert result['research']['outputs']['raw']['dominant_coordinate_fraction']==1.
    assert result['runtime']['estimated_extra_bytes']<400*1024**2
    decision=policy.decide_assisted_policy({'baseline':result['baseline_summary'],'cues':{'status':'failed'},'expanded':result['summary']})
    assert decision['selected_candidate']=='expanded'
    assert base.comparison['inner_decoder']['actual']=='viterbi'
    assert 'inner_decoder' not in result['candidate']['comparison']


def test_second_and_mixed_baseline_pools_prove_own_rows_outside_union(tmp_path):
    base,metadata=controls(tmp_path,mixed=True);result=run(base,metadata)
    assert result['status']=='complete'
    proof=result['research']['admission']['original_selected_sources']['baseline']
    assert [row['audit_index'] for row in proof]==[q%2 for q in range(80)]
    assert 'mapillary:fine_base' not in result['research']['union']['ref_ids']


def test_already_admitted_original_legacy_metadata_bypasses_all_new_work(tmp_path,monkeypatch):
    base,metadata=controls(tmp_path,admitted=True)
    monkeypatch.setattr(a.core,'replay_metadata_union',lambda *_:pytest.fail('legacy admission must bypass DP'))
    def forbidden(*_):pytest.fail('legacy admission must bypass new resources/descriptors')
    result=a.evaluate_metadata_union(base,metadata,requested_pool_count=1,resource_gate=forbidden,descriptor_loader=forbidden)
    assert result['status']=='preserved' and result['selection']=='legacy_metadata' and result['union_decoded'] is False


def test_source_mutation_declines_before_any_array_load(tmp_path,monkeypatch):
    _,bc,bs,_,_,rs=fixed_case();artifact,s,expected=seal_control(tmp_path,rs,bc,bs,'baseline')
    path=tmp_path/'retrieval_audit/0000.npz';path.write_bytes(path.read_bytes()+b'changed')
    monkeypatch.setattr(a.np,'load',lambda *_args,**_kwargs:pytest.fail('source must bind before array load'))
    with pytest.raises(ValueError,match='artifact changed'):a.load_legacy_control(artifact,'candidate.json',s,expected)


@pytest.mark.parametrize('bad',['video_sha256','source_sha256','execution_id','window'])
def test_different_execution_native_video_window_or_source_declines_before_arrays(tmp_path,monkeypatch,bad):
    _,bc,bs,_,_,rs=fixed_case();artifact,s,expected=seal_control(tmp_path,rs,bc,bs,'baseline')
    expected[bad]=['different'] if bad=='window' else {'other':'e'*64} if bad=='source_sha256' else 'e'*64
    monkeypatch.setattr(a.np,'load',lambda *_args,**_kwargs:pytest.fail('identity must bind before arrays'))
    with pytest.raises(ValueError):a.load_legacy_control(artifact,'candidate.json',s,expected)


def test_forbidden_member_rejected_before_its_hash(tmp_path,monkeypatch):
    _,bc,bs,_,_,rs=fixed_case();artifact,s,expected=seal_control(tmp_path,rs,bc,bs,'baseline')
    freeze=Path(artifact['path']);value=json.loads(freeze.read_text());value['artifacts']['offline_scores.json']='f'*64
    (tmp_path/'offline_scores.json').write_text('{}');write(freeze,value);artifact['sha256']=a.sha(freeze)
    original=a.sha
    def checked(path):
        assert Path(path).name!='offline_scores.json','label must be refused before hash'
        return original(path)
    monkeypatch.setattr(a,'sha',checked)
    with pytest.raises(ValueError,match='labels'):a.load_legacy_control(artifact,'candidate.json',s,expected)


@pytest.mark.parametrize('name',['GT.json','labels.json','ground-truth/data.json','OXTS/data.json','input_Gt.json','Labels/cache.dat'])
def test_forbidden_alias_tokens_refuse_before_file_open(tmp_path,monkeypatch,name):
    monkeypatch.setattr(Path,'open',lambda *_args,**_kwargs:pytest.fail('label must not be read or hashed'))
    with pytest.raises(ValueError,match='labels'):a.sha(tmp_path/name)


def test_resolved_forbidden_alias_refuses_before_file_open(tmp_path,monkeypatch):
    alias=tmp_path/'innocent.json';target=tmp_path/'ground-truth'/'data.json';resolve=Path.resolve
    def resolved(self,*args,**kwargs):return target if self==alias else resolve(self,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',resolved)
    monkeypatch.setattr(Path,'open',lambda *_args,**_kwargs:pytest.fail('resolved label alias must not be opened'))
    with pytest.raises(ValueError,match='labels'):a.sha(alias)


@pytest.mark.parametrize('bad',['memory','count','in_memory_matrix','in_memory_comparison'])
def test_missing_invalid_or_mutated_controls_decline_before_DP(tmp_path,monkeypatch,bad):
    base,metadata=controls(tmp_path)
    monkeypatch.setattr(a.core,'replay_metadata_union',lambda *_:pytest.fail('invalid input must decline before DP'))
    kwargs={'requested_pool_count':2,'resource_gate':memory,'numerical_context':cpu}
    if bad=='memory':
        def fail():raise ValueError('physical/truecommit below8GiB')
        kwargs['resource_gate']=fail
    if bad=='count':kwargs['requested_pool_count']=1
    if bad=='in_memory_matrix':metadata.regions[0]['arrays']['raw_similarities'][0,0]=0
    if bad=='in_memory_comparison':metadata.comparison['decoded_reference_cosines'][0]=0
    result=a.evaluate_metadata_union(base,metadata,**kwargs)
    assert result['status']=='unavailable' and result['runtime']['DP_started'] is False


def test_hash_claims_without_actual_overlap_rows_decline_before_DP(tmp_path,monkeypatch):
    base,metadata=controls(tmp_path,overlap=True)
    monkeypatch.setattr(a.core,'replay_metadata_union',lambda *_:pytest.fail('hash strings cannot authorize DP'))
    result=run(base,metadata,reference_cfg=object(),descriptor_loader=lambda *_:{'row_hashes':['a'*64]})
    assert result['status']=='unavailable' and result['runtime']['DP_started'] is False


def tensor_sha(rows):
    value=hashlib.sha256(json.dumps([list(rows.shape),'torch.float32'],separators=(',',':')).encode())
    value.update(memoryview(np.ascontiguousarray(rows)).cast('B'));return value.hexdigest()


def descriptor_fixture(tmp_path,region,*,different=False):
    n=len(region['arrays']['ref_ids']);rows=np.zeros((n,8448),np.float32);rows[np.arange(n),np.arange(n)+(1 if different else 0)]=1.
    tmp_path.mkdir(parents=True,exist_ok=True);path=tmp_path/'descriptors.npz';np.savez(path,emb=rows,fingerprint=np.array('fingerprint'))
    meta=tmp_path/'metadata.json';meta.write_text('{}');pixels=tmp_path/'pixels.bin';pixels.write_bytes(b'API-verified native pixels')
    region['audit'].setdefault('ref_embeddings_sha256',tensor_sha(rows))
    pool={'status':'complete','payload_mode':'descriptors_and_pixels','count':n,'fingerprint':'fingerprint',
        'ref_ids_sha256':a.core.full._array_sha(region['arrays']['ref_ids']),
        'ref_latlon_sha256':a.core.full._array_sha(region['arrays']['ref_latlon']),
        'ref_embeddings_sha256':tensor_sha(rows),'actual_model_state_sha256':region['audit']['actual_model_state_sha256'],
        'backbone':region['audit']['backbone'],'preprocessing':region['audit']['preprocessing'],'request':{'cache_dir':str(tmp_path)},
        'files':[{'name':p.name,'role':role,'bytes':p.stat().st_size,'sha256':a.sha(p)} for p,role in [(meta,'metadata'),(pixels,'pixels'),(path,'descriptors')]]}
    return rows,pool,path


def test_actual_descriptor_rows_and_complete_receipt_bind_before_load(tmp_path):
    region=fixed_case()[0][0];rows,pool,_=descriptor_fixture(tmp_path,region)
    evidence=a.load_descriptor_rows(region,object(),describe=lambda *_:pool)
    np.testing.assert_array_equal(evidence.rows,rows)
    evidence.verify();evidence.rows[0,0]=.5
    with pytest.raises(ValueError,match='receipt'):evidence.verify()


def test_changed_reference_payload_declines_before_array_deserialization(tmp_path,monkeypatch):
    region=fixed_case()[0][0];_,pool,path=descriptor_fixture(tmp_path,region);path.write_bytes(path.read_bytes()+b'changed')
    monkeypatch.setattr(a.np,'load',lambda *_args,**_kwargs:pytest.fail('changed descriptor payload cannot load'))
    with pytest.raises(ValueError,match='payload changed'):a.load_descriptor_rows(region,object(),describe=lambda *_:pool)


def test_real_rows_enable_overlap_with_all_occurrences_retained(tmp_path):
    base,metadata=controls(tmp_path/'good',overlap=True)
    def descriptor(region,cfg):
        rows,pool,_=descriptor_fixture(tmp_path/'descriptors'/region['key'],region)
        # describe_pool would validate these same source-bound actual rows.
        return a.load_descriptor_rows(region,cfg,describe=lambda *_:pool)
    result=run(base,metadata,reference_cfg=object(),descriptor_loader=descriptor)
    assert result['status']=='complete' and len(result['descriptor_receipts'])==2
    assert result['research']['union']['reference_occurrences']==9 and len(result['research']['union']['ref_ids'])==6


def test_conflicting_actual_receipt_bound_rows_decline_whole_union_before_DP(tmp_path,monkeypatch):
    base,metadata=controls(tmp_path/'source',overlap=True,descriptor_conflict=True)
    def descriptor(region,cfg):
        _,pool,_=descriptor_fixture(tmp_path/'descriptors'/region['key'],region,different=region['key']=='audit_0002')
        return a.load_descriptor_rows(region,cfg,describe=lambda *_:pool)
    monkeypatch.setattr(a.core,'replay_metadata_union',lambda *_:pytest.fail('conflicting rows must fail before DP'))
    result=run(base,metadata,reference_cfg=object(),descriptor_loader=descriptor)
    assert result['status']=='unavailable' and result['runtime']['DP_started'] is False
    assert 'descriptor' in result['reason']


@pytest.mark.parametrize('phase',['before','after'])
def test_registered_wall_budget_refuses_primary_before_or_after_bounded_DP(tmp_path,monkeypatch,phase):
    base,metadata=controls(tmp_path)
    calls=iter([0.,4001.,4002.] if phase=='before' else [0.,1.,4001.,4002.])
    if phase=='before':monkeypatch.setattr(a.core,'replay_metadata_union',lambda *_:pytest.fail('expired work cannot start DP'))
    result=run(base,metadata,monotonic=lambda:next(calls))
    assert result['status']=='unavailable' and '3600' in result['reason']
    assert result['runtime']['DP_started'] is (phase=='after')


def test_unused_partial_pool_is_not_silently_dropped(tmp_path):
    rs,bc,bs,mc,ms,_=fixed_case();rs[1]['audit']['status']='partial'
    artifact,s,expected=seal_control(tmp_path,rs,mc,ms,'expanded')
    with pytest.raises(ValueError,match='complete'):a.load_legacy_control(artifact,'candidate.json',s,expected)


def test_actual_thread_context_sets_both_libraries_to_two():
    with a.cpu_two_threads() as rows:
        assert len(rows)>=2 and all(row['num_threads']==2 for row in rows)
