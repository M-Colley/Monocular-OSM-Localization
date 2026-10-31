"""Synthetic snapshots/fake encoders only: no models, network or labels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from monocular_osm import kartaview_vpr as kv
from monocular_osm import public_reference_cache as cache


def typed(array):
    array=np.ascontiguousarray(array)
    h=hashlib.sha256(str((array.shape,array.dtype.str)).encode())
    if array.size:h.update(array.tobytes())
    return h.hexdigest()


def tensor_sha(array):
    h=hashlib.sha256(json.dumps([list(array.shape),'torch.float32'],separators=(',',':')).encode())
    h.update(np.ascontiguousarray(array,dtype=np.float32).tobytes())
    return h.hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


@pytest.fixture
def case(tmp_path,monkeypatch):
    asset=tmp_path/'assets'/'kv_source.py';asset.parent.mkdir();asset.write_text('# fixed source\n',encoding='utf8')
    monkeypatch.setattr(kv,'__file__',str(asset))
    monkeypatch.setattr(cache,'library_versions',lambda *a,**k:{'numpy':'fixture','torch':'fixture'})
    cfg=SimpleNamespace(data_dir=tmp_path/'data'/'baseline'/'slug',vpr_region_cache_dir=None)
    directory=cfg.data_dir/'mapillary';directory.mkdir(parents=True)
    request={'source':'mapillary','center':[50.00004,10.00004],'radius_m':3000.,'cap':1500,
        'model_name':'megaloc','device':'cpu','query_mode':'warp','query_batch_size':4,'cache_dir':str(directory.resolve())}
    refs=[{'id':str(i),'lat':50.,'lon':10.+i*.0001} for i in (1,2,3)]
    served=refs[:2];ids=np.array(['mapillary:1','mapillary:2']);xy=np.array([[r['lat'],r['lon']] for r in served])
    emb=np.array([[1.,0.,0.],[0.,1.,0.]],np.float32)
    fp=kv._refs_fingerprint(served);sig=kv._fetch_signature(request['center'],3000,1500)
    tag=kv._sig_tag(sig);meta=directory/f'mly_ref_meta_{tag}.json';snapshot=kv._reference_snapshot(refs)
    listing={'listing':'tiles','tiles':1,'lost':0,'listing_stats_known':True,'snapshot':snapshot}
    write(meta,{'signature':sig,'refs':refs,'listing_stats':listing,'snapshot':snapshot})
    write(directory/f'mly_ref_store_{tag}.json',{r['id']:[r['lat'],r['lon']] for r in refs})
    raw=np.empty((2,512,512,3),np.uint8);raw[0]=[17,37,97];raw[1]=[9,23,179]
    np.savez(directory/'ref_img_store.npz',ids=np.array(['1','2']),xy=xy,raw=raw)
    embedding=directory/f'ref_emb_megaloc_{fp[:10]}.npz';np.savez(embedding,emb=emb,fingerprint=fp)
    audit={'status':'complete','public_cache_request':request,'reference_pool':{
        **{k:request[k] for k in ('source','center','radius_m','cap')},'fetch':listing},
        'ref_ids':ids,'ref_latlon':xy,'ref_embeddings_sha256':tensor_sha(emb),
        'backbone':{'name':'megaloc','hub':'official/pin','device':'cpu'},'actual_model_state_sha256':'f'*64,
        'preprocessing':{'version':1,'dimensions':3,'color':'BGR to RGB','warp_size':[512,512]},
        'reference_encoding_identity':'UNVERIFIED: legacy revision/preprocessing','decoded_reference_indices':[1]}
    pool=cache.describe_pool(audit,cfg)
    return SimpleNamespace(root=tmp_path,cfg=cfg,directory=directory,request=request,refs=refs,served=served,
        ids=ids,xy=xy,emb=emb,fp=fp,sig=sig,meta=meta,embedding=embedding,audit=audit,pool=pool,asset=asset)


def prepared(c,*,source_change=None,audit_change=None):
    execution='e'*32;video='a'*64;source_sha={'fixture.py':'b'*64}
    child_root=c.root/'parent'/'children'/execution;baseline=child_root/'baseline';baseline.mkdir(parents=True)
    request={'kind':'baseline','video_sha256':video,'source_sha256':source_sha,'execution_id':execution,
        'parent_output':str(child_root.parent.parent),'config':{'data_dir':str(c.cfg.data_dir),'vpr_region_cache_dir':None}}
    write(baseline/'request.json',request)
    arrays=baseline/'retrieval_audit'/'0000.npz';arrays.parent.mkdir();np.savez(arrays,ref_ids=c.ids,ref_latlon=c.xy)
    metadata={k:deepcopy(v) for k,v in c.audit.items() if k not in ('ref_ids','ref_latlon')}
    metadata.update(public_reference_cache=deepcopy(c.pool),array_artifact={'path':str(arrays),'sha256':digest(arrays)})
    if audit_change:audit_change(metadata)
    meta_path=baseline/'retrieval_audit'/'0000.json';write(meta_path,metadata)
    members={str(p.relative_to(baseline)):digest(p) for p in [baseline/'request.json',arrays,meta_path]}
    frozen=baseline/'child_frozen.json'
    write(frozen,{'kind':'baseline','scoring_has_not_run':True,'video_sha256':video,'source_sha256':source_sha,'artifacts':members})
    source={'kind':'baseline','status':'complete','worker_returncode':0,'execution_id':execution,
        'video_sha256':video,'source_sha256':source_sha,'output_dir':str(baseline),
        'frozen_artifact':{'path':str(frozen),'sha256':digest(frozen)},'request_sha256':digest(baseline/'request.json')}
    if source_change:source_change(source)
    target=SimpleNamespace(data_dir=c.root/'data'/'expanded'/'slug',output_dir=child_root/'expanded',vpr_region_cache_dir=None)
    artifact=cache.prepare_reuse(source,target,execution_id=execution,video_sha256=video,source_sha256=source_sha,child_root=child_root)
    receipt=json.loads(Path(artifact['path']).read_text(encoding='utf8'))
    return SimpleNamespace(artifact=artifact,receipt=receipt,target=target,source=source,child_root=child_root,
        expected={'execution_id':execution,'video_sha256':video,'source_sha256':source_sha})


def test_all_consumed_public_rows_and_honest_encoding(case):
    p=case.pool
    assert p['status']=='complete' and p['payload_mode']=='descriptors_and_pixels'
    assert [r['id'] for r in p['servable_refs']]==['1','2']  # not just decoded winner2
    assert p['count']==2 and p['listed_count']==3 and p['pixel_count']==2
    assert p['reference_encoding_identity'].startswith('UNVERIFIED')
    assert p['request']['center']==[50.00004,10.00004] and p['metadata_signature']['center']==[50.,10.]
    assert p['ref_ids_sha256']==typed(case.ids) and p['ref_latlon_sha256']==typed(case.xy)
    assert p['ref_embeddings_sha256']==tensor_sha(case.emb)
    assert {x['role'] for x in p['files']}=={'metadata','image_store','pixels','descriptors'}
    assert all('query' not in x['name'] and 'ocr' not in x['name'] for x in p['files'])


def test_prepare_copies_only_snapshot_and_binds_receipt_before_child(case):
    p=prepared(case)
    assert p.receipt['status']=='complete' and len(p.receipt['pools'])==1,p.receipt
    assert p.receipt['query_artifacts_copied'] is False and p.receipt['before_child_launch'] is True
    assert p.artifact['sha256']==digest(p.artifact['path'])
    assert sorted(x.name for x in (p.target.data_dir/'mapillary').iterdir())==sorted(x['name'] for x in case.pool['files'])


def install_fake_encoder(case,monkeypatch,*,bad_order=False):
    calls={'queries':0,'fetch':0,'model':0,'directories':[]}
    def fresh_fetch(source,center,radius_m,cache_dir,cap,token):
        calls['fetch']+=1;return deepcopy(case.refs)
    monkeypatch.setattr(kv,'_fetch_refs_for',fresh_fetch)
    vectors={'1':np.array([1.,0.,0.],np.float32),'2':np.array([0.,1.,0.],np.float32),'3':np.array([2**-.5,2**-.5,0.],np.float32)}
    def track(frames_bgr,center,radius_m=3000.,*,cache_dir=None,n_query=80,device=None,model_name='megaloc',source='mapillary',token=None,cap=1500,query_mode='warp',query_batch_size=4,audit=None):
        calls['queries']+=1;calls['directories'].append(cache_dir)
        refs=kv._fetch_refs_for(source,center,radius_m,cache_dir,cap,token)
        emb=np.array([vectors[str(row['id'])] for row in refs],np.float32)
        query=np.array([[2**-.5,2**-.5,0.]],np.float32)  # freshly computed each call
        chosen=int((query@emb.T)[0].argmax())
        ids=np.array(['mapillary:'+str(row['id']) for row in refs]);xy=np.array([[r['lat'],r['lon']] for r in refs])
        if audit is not None:
            audit.clear();audit.update(status='complete',ref_embeddings_sha256=tensor_sha(emb),
                ref_ids=ids[::-1] if bad_order else ids,ref_latlon=xy[::-1] if bad_order else xy)
        return (refs[chosen]['id'],query.copy())
    monkeypatch.setattr(kv,'kartaview_vpr_track',track)
    def resolve(*a,**k):calls['model']+=1;return 'megaloc'
    monkeypatch.setattr(kv,'_resolve_backbone',resolve)
    monkeypatch.setattr(kv,'backbone_info',lambda:deepcopy(case.audit['backbone']))
    monkeypatch.setattr(kv,'_audit_model_state',lambda:{'sha256':case.audit['actual_model_state_sha256']})
    return calls,track,fresh_fetch


def invoke(c,p,**changes):
    arguments={k:v for k,v in c.request.items() if k not in ('cache_dir','center')}
    arguments['cache_dir']=str(p.target.data_dir/'mapillary');arguments.update(changes)
    audit={};answer=kv.kartaview_vpr_track(None,c.request['center'],audit=audit,**arguments)
    return answer,audit


def test_replay_all_served_snapshot_with_fresh_queries_identical_route(case,monkeypatch):
    p=prepared(case);calls,original,fetch=install_fake_encoder(case,monkeypatch)
    # The newly available third photo would win; exact reuse must not add it.
    monkeypatch.setattr(kv,'_fetch_refs_for',lambda *a,**k:deepcopy(case.served))
    baseline,_=invoke(case,p)
    monkeypatch.setattr(kv,'_fetch_refs_for',fetch)
    with cache.track_scope(p.target,'expanded',p.artifact,expected=p.expected):
        answer,audit=invoke(case,p)
    assert answer[0]==baseline[0]=='1' and np.array_equal(answer[1],baseline[1])
    assert calls['queries']==2 and calls['fetch']==0 and calls['model']==1
    assert audit['public_cache_reuse']['status']=='hit'
    assert audit['public_cache_request']['center']==case.request['center']  # sink.clear() cannot erase capture
    assert kv.kartaview_vpr_track is original and kv._fetch_refs_for is fetch


def test_assisted_per_call_checkpoint_keeps_genuine_reuse_hit_before_final_export(case,monkeypatch):
    from functools import wraps
    import sys
    from monocular_osm import assisted_worker as worker, pipeline
    from monocular_osm.stages import prior
    signature = kv.kartaview_vpr_track
    p = prepared(case)
    calls, encoder, _ = install_fake_encoder(case, monkeypatch)

    @wraps(signature)
    def track(frames, center, **kwargs):
        audit = kwargs['audit']
        encoder(frames, center, **{k: v for k, v in kwargs.items()
            if k in {'radius_m', 'cache_dir', 'device', 'source', 'token', 'cap',
                     'query_mode', 'query_batch_size', 'audit'}})
        audit.update(query_indices=np.array([0, 1]), query_times=np.array([0., 1.]),
            raw_similarities=np.array([[.1, .2], [.3, .4]], np.float32))
        return audit['query_indices'], audit['ref_latlon'].copy(), np.array([.1, .4])

    monkeypatch.setattr(kv, 'kartaview_vpr_track', track)
    monkeypatch.setitem(sys.modules, 'osmnx', SimpleNamespace())
    monkeypatch.setattr(prior, 'select_retrieval_device', lambda: 'cpu')
    cfg = pipeline.PipelineConfig(url='', city='Synthetic', data_dir=p.target.data_dir,
        output_dir=p.target.output_dir, vpr_source='mapillary', vpr_multi_region=True,
        vpr_coarse_to_fine=False, vpr_city_cover=False, assisted_profile='v1')
    observed = []

    def regions(cfg, frames, center, radius, cap, source, token, device, dt, call, **kwargs):
        for ordinal in range(2):
            result = call(frames.frames, center, radius_m=radius, cap=cap, source=source,
                device=device, cache_dir=str(cfg.data_dir/'mapillary'), query_batch_size=4)
            # These checks run before the pipeline or its final comparison returns.
            meta = json.loads((cfg.output_dir/f'retrieval_audit/{ordinal:04d}.json').read_text())
            assert meta['public_cache_reuse']['status'] == 'hit'
            assert meta['public_cache_request']['cache_dir'] == str((cfg.data_dir/'mapillary').resolve())
            assert meta['persisted_writer']['ordinal'] == ordinal and 'call_index' not in meta
            observed.append(meta['array_artifact']['sha256'])
        return result, {'n_usable_regions': 2}, {}

    monkeypatch.setattr(prior, '_multi_region_vpr', regions)

    def stages(ctx):
        ctx.frames = SimpleNamespace(frames=[None, None], timestamps=np.array([0., 1.]), close=lambda: None)
        ctx.osm_around = (*case.request['center'], 3000.)
        ctx.estimated_length_m = 100.
        prior.vpr_coarse_prior(ctx)
        return ctx.result

    monkeypatch.setattr(pipeline, '_run_pipeline_stages', stages)
    worker._pipeline_child(cfg, 'expanded', {**p.expected, 'public_reference_reuse': p.artifact})
    assert len(observed) == 2 and calls['queries'] == 2 and calls['fetch'] == 0
    for ordinal, before in enumerate(observed):
        meta = json.loads((cfg.output_dir/f'retrieval_audit/{ordinal:04d}.json').read_text())
        assert meta['array_artifact']['sha256'] == before
        assert meta['public_cache_reuse']['status'] == 'hit'


def test_legacy_encoding_claim_is_bound_to_pool_identity(case):
    pool=deepcopy(case.pool);pool['reference_encoding_identity']='VERIFIED current encoder'
    with pytest.raises(ValueError,match='identity changed'):
        cache.copy_pool(pool,case.directory,case.root/'target')


@pytest.mark.parametrize('listing_kind',['tiles','graph'])
def test_metadata_only_requires_complete_tiles_without_encoding_claim(case,listing_kind):
    write(case.meta,{'signature':case.sig,'refs':[]})
    snapshot=kv._reference_snapshot([])
    terminal={'count':0,'reference_inventory':[],'status':'complete_empty','source':'mapillary','fetch_signature':case.sig,
        'snapshot':snapshot,'listing':{'listing':listing_kind,'listing_stats_known':True,'lost':0,'snapshot':snapshot}}
    audit={'public_cache_request':case.request,'terminal_search':terminal}
    if listing_kind=='graph':
        with pytest.raises(ValueError,match='listing'):cache.describe_pool(audit,case.cfg)
    else:
        pool=cache.describe_pool(audit,case.cfg)
        assert pool['payload_mode']=='metadata_only' and pool['count']==0
        assert pool['ref_embeddings_sha256'] is None and pool['actual_model_state_sha256'] is None
        assert pool['reference_encoding_identity'].startswith('not applicable')
        assert len(pool['files'])==1 and pool['files'][0]['role']=='metadata'


def test_copy_is_atomic_identical_and_never_overwrites(case):
    destination=case.root/'new'/'mapillary'
    before={x['name']:digest(case.directory/x['name']) for x in case.pool['files']}
    assert cache.copy_pool(case.pool,case.directory,destination)['status']=='copied'
    assert {name:digest(destination/name) for name in before}==before
    assert {name:digest(case.directory/name) for name in before}==before
    assert cache.copy_pool(case.pool,case.directory,destination)['status']=='existing_identical'
    (destination/'ref_img_store.npz').write_bytes(b'owned different bytes')
    assert cache.copy_pool(case.pool,case.directory,destination)['status']=='declined'
    assert (destination/'ref_img_store.npz').read_bytes()==b'owned different bytes'
    assert not list(destination.parent.glob('.public-reference-*'))


def test_concurrent_source_change_never_publishes_partial_family(case,monkeypatch):
    original=cache.shutil.copyfileobj;calls=[]
    def mutate(src,dst,**kw):
        original(src,dst,**kw)
        if not calls:case.meta.write_text(case.meta.read_text(encoding='utf8')+'\n',encoding='utf8')
        calls.append(1)
    monkeypatch.setattr(cache.shutil,'copyfileobj',mutate)
    destination=case.root/'new'/'mapillary'
    with pytest.raises(ValueError,match='changed|hash mismatch'):cache.copy_pool(case.pool,case.directory,destination)
    assert not destination.exists() and not list(destination.parent.glob('.public-reference-*'))


@pytest.mark.parametrize('fault',['query_name','wrong_role','escape'])
def test_forbidden_payload_names_roles_and_traversal(case,fault):
    pool=deepcopy(case.pool)
    if fault=='query_name':pool['files'][0]['name']='ref_emb_query.npz'
    elif fault=='wrong_role':pool['files'][0]['role']='pixels'
    else:pool['files'][0]['name']='../ref_meta.json'
    pool['pool_id']=cache._digest(cache._pool_identity(pool))
    with pytest.raises(ValueError):cache.copy_pool(pool,case.directory,case.root/'target')


@pytest.mark.parametrize('fault',['incomplete','order','positions','tensor','model_file','preprocessing'])
def test_incomplete_or_mismatched_pool_cannot_be_sealed(case,fault):
    a=deepcopy(case.audit)
    if fault=='incomplete':a['status']='incomplete'
    elif fault=='order':a['ref_ids']=a['ref_ids'][::-1];a['ref_latlon']=a['ref_latlon'][::-1]
    elif fault=='positions':a['ref_latlon'][0,0]+=.001
    elif fault=='tensor':a['ref_embeddings_sha256']='0'*64
    elif fault=='model_file':a['backbone']['name']='other'
    else:a['preprocessing']['dimensions']=4
    with pytest.raises(ValueError):cache.describe_pool(a,case.cfg)


@pytest.mark.parametrize('change',[{'status':'failed'},{'worker_returncode':1},{'execution_id':'other'},{'video_sha256':'0'*64},{'source_sha256':{}}])
def test_prepare_requires_success_same_execution_video_source(case,change):
    p=prepared(case,source_change=lambda source:source.update(change))
    assert p.receipt['status']=='unavailable' and not p.receipt['pools']
    assert not (p.target.data_dir/'mapillary').exists()


def test_prepare_retains_incomplete_pool_decline(case):
    p=prepared(case,audit_change=lambda a:a.update(status='incomplete'))
    assert p.receipt['status']=='complete' and not p.receipt['pools'] and len(p.receipt['declined'])==1
    assert not (p.target.data_dir/'mapillary').exists()


@pytest.mark.parametrize('fault',['rounded_center_collision','model','source','query_mode'])
def test_mismatch_uses_fresh_nested_directory_without_overwrite(case,monkeypatch,fault):
    p=prepared(case);calls,original,_=install_fake_encoder(case,monkeypatch)
    changes={}
    if fault=='model':monkeypatch.setattr(kv,'_audit_model_state',lambda:{'sha256':'0'*64})
    elif fault=='source':case.asset.write_text('# changed source\n',encoding='utf8')
    elif fault=='query_mode':changes['query_mode']='crops'
    before={x['name']:digest(p.target.data_dir/'mapillary'/x['name']) for x in case.pool['files']}
    with cache.track_scope(p.target,'expanded',p.artifact,expected=p.expected):
        if fault=='rounded_center_collision':
            center=[50.00003,10.00004]
            assert kv._fetch_signature(center,3000,1500)==case.sig
            audit={};kv.kartaview_vpr_track(None,center,cache_dir=str(p.target.data_dir/'mapillary'),device='cpu',audit=audit)
        else:_,audit=invoke(case,p,**changes)
    assert audit['public_cache_reuse']['status']=='miss' and '_unshared_' in calls['directories'][-1]
    assert before=={name:digest(p.target.data_dir/'mapillary'/name) for name in before}
    assert kv.kartaview_vpr_track is original


def test_receiver_order_mismatch_fails_and_restores_hooks(case,monkeypatch):
    p=prepared(case);_,original,fetch=install_fake_encoder(case,monkeypatch,bad_order=True)
    with pytest.raises(ValueError,match='exact copied reference tensor/order'):
        with cache.track_scope(p.target,'expanded',p.artifact,expected=p.expected):invoke(case,p)
    assert kv.kartaview_vpr_track is original and kv._fetch_refs_for is fetch


def test_cross_execution_receipt_refuses_before_encoder(case,monkeypatch):
    p=prepared(case);calls,_,_=install_fake_encoder(case,monkeypatch)
    with pytest.raises(ValueError,match='another execution'):
        with cache.track_scope(p.target,'expanded',p.artifact,expected={**p.expected,'execution_id':'other'}):pass
    assert calls['model']==calls['queries']==0


@pytest.mark.parametrize('name',['offline_scores.json','trial_registered_score.json','score_trial.json','Scoring_frozen.JSON','scores.jsonl'])
def test_label_path_guard_runs_before_digest(case,monkeypatch,name):
    forbidden=case.root/name;write(forbidden,{'synthetic':True})
    monkeypatch.setattr(cache,'file_sha256',lambda *a:pytest.fail('forbidden bytes hashed'))
    with pytest.raises(ValueError,match='forbidden reference inputs'):cache._sha(forbidden)


def test_resolved_alias_guard_runs_before_digest(case,monkeypatch):
    forbidden=case.root/'labels'/'ground_truth'/'metadata.json';write(forbidden,{'synthetic':True})
    alias=case.directory/'ref_meta.json'
    try:alias.symlink_to(forbidden)
    except (OSError,NotImplementedError):pytest.skip('symlink capability unavailable')
    monkeypatch.setattr(cache,'file_sha256',lambda *a:pytest.fail('aliased forbidden bytes hashed'))
    with pytest.raises(ValueError,match='forbidden reference inputs'):cache._sha(alias)


def test_descriptor_header_matches_actual_tensor_audit_schema(case):
    torch=pytest.importorskip('torch')
    assert tensor_sha(case.emb)==kv._audit_tensor_sha(torch.from_numpy(case.emb))
