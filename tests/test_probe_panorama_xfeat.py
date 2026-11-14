"""No models, API, real galleries or evaluation data in these tests."""
import copy
import hashlib
from pathlib import Path
import socket
import numpy as np
import pytest
from scripts import probe_panorama_xfeat as p


def pixel_digest(images):
    h=hashlib.sha256()
    for a in images:
        a=np.ascontiguousarray(a);h.update(str((a.shape,a.dtype.str)).encode());h.update(memoryview(a).cast('B'))
    return h.hexdigest()


POINTS=np.array([[10,10],[500,10],[500,500],[10,500],[200,200],[250,250],[300,300],[350,350]],np.float32)
F=np.diag([1.,1.,0.])


class Backend:
    def __init__(self):self.feature_calls=0;self.matches=[]
    def features(self,image):
        self.feature_calls+=1
        assert image.shape==(512,512,3) and image.dtype==np.uint8
        return POINTS.copy(),np.full((8,64),1/8,np.float32)
    def correspondences(self,first,second):
        self.matches.append((id(first[1]),id(second[1])));return first[0],second[0]


@pytest.fixture
def synthetic(tmp_path,monkeypatch):
    monkeypatch.setattr(p.cv2,'findFundamentalMat',lambda a,b,**kw:(F.copy(),np.ones((len(a),1),np.uint8)))
    ids=np.array([f'mapillary:{i+1}' for i in range(8)])
    evidence={'similarities':np.array([np.arange(8),np.arange(8)[::-1]],np.float32),'ref_ids':ids,
        'ref_latlon':np.column_stack([np.full(8,42.),-83.+np.arange(8)*.001]),'query_times':np.array([0.,2.])}
    pixels=[np.full((6,8,3),[10,20,30],np.uint8),np.full((6,8,3),[40,50,60],np.uint8)]
    raw=np.broadcast_to(np.array([5,15,25],np.uint8),(8,8,16,3)).copy()
    cameras=dict.fromkeys(ids,None);cameras[ids[0]]='spherical';cameras[ids[1]]='fisheye'
    return evidence,{'pixels_sha256':pixel_digest(pixels)},cameras,raw,pixels


def search(tmp_path,synthetic,**kwargs):
    evidence,query,cameras,raw,pixels=synthetic;backend=Backend()
    result=p.paired_search(tmp_path,evidence,query,cameras,raw,backend,stream=enumerate(pixels),check=lambda path:{},**kwargs)
    return result,backend


def test_geometry_and_extractor_constants_remain_exact():
    for k in ('top_k','query_views','geometry','coverage','score','tie_breaks','zero_evidence','xfeat'):
        assert p.PROFILE[k]==p.original.PROFILE[k]
    assert p.PROFILE['variants']==['original_single','panorama_aware']
    assert not p.PROFILE['panorama']['heading_claim'] and p.PROFILE['selected_variant'] is None


def test_exhaustive_views_shared_query_features_and_unknown_passthrough(tmp_path,synthetic):
    result,backend=search(tmp_path,synthetic)
    assert len(result['records']['original_single'])==48 and len(result['records']['panorama_aware'])==66
    assert result['unique_pairs']==len(backend.matches)==72 and backend.feature_calls==6+8+4
    for q in range(2):
        rows=[r for r in result['records']['panorama_aware'] if r['query_index']==q and r['ref_index']==0]
        assert {(r['view'],r['reference_yaw_deg']) for r in rows}=={(v,yaw) for v in range(3) for yaw in [0.,90.,180.,270.]}
    baseline={(r['query_index'],r['ref_index'],r['view']):r for r in result['records']['original_single']}
    for row in result['records']['panorama_aware']:
        if row['ref_index']!=0:
            assert row['reference_yaw_deg'] is None
            assert row['evidence']==baseline[row['query_index'],row['ref_index'],row['view']]['evidence']
    assert len({pair[0] for pair in backend.matches})==6
    assert all(s['selected_view']==0 and s['selected_reference_view_index']==0 for s in result['selections']['panorama_aware'])


def test_saved_points_F_and_masks_replay_every_pair(tmp_path,synthetic):
    result,_=search(tmp_path,synthetic)
    for chunk in result['chunks']:
        with np.load(tmp_path/'correspondences'/chunk['path'],allow_pickle=False) as z:
            for rows in result['records'].values():
                for row in rows:
                    if Path(row['evidence']['chunk']).name==chunk['path']:
                        assert p.replay_geometry(row,z)=={k:row[k] for k in ['status','match_count','inlier_count','query_coverage','reference_coverage','score']}
    assert result['correspondence_bytes']==sum(c['bytes'] for c in result['chunks'])


def test_original_control_must_match_all_pairs_and_route(tmp_path,synthetic):
    result,_=search(tmp_path,synthetic);e=synthetic[0];s=result['selections']['original_single'];idx=[r['selected_ref_index'] for r in s]
    extra={'reference_view_index','reference_yaw_deg','evidence'}
    prior={'records':[{k:v for k,v in r.items() if k not in extra} for r in result['records']['original_single']],
        'prediction':{'selections':copy.deepcopy(s),'chosen_ref_ids':e['ref_ids'][idx].tolist(),
            'route_latlon':e['ref_latlon'][idx].tolist(),'route_timestamps':e['query_times'].tolist()}}
    p.compare_control(result,prior,e);prior['records'][0]['score']+=.001
    with pytest.raises(ValueError,match='pair evidence'):p.compare_control(result,prior,e)


def test_zero_geometry_uses_unchanged_centered_argmax(tmp_path,synthetic,monkeypatch):
    monkeypatch.setattr(p.cv2,'findFundamentalMat',lambda *a,**k:(None,None))
    result,_=search(tmp_path,synthetic)
    for variant in p.PROFILE['variants']:
        assert [s['selected_ref_index'] for s in result['selections'][variant]]==[7,0]
        assert all(s['selected_view'] is None and s['reason']=='no_positive_geometry' for s in result['selections'][variant])


def test_empty_sky_or_query_features_keep_every_pair_and_fallback(tmp_path,synthetic):
    class Empty(Backend):
        def features(self,image):
            self.feature_calls+=1
            return np.empty((0,2),np.float32),np.empty((0,64),np.float32)
    e,q,c,raw,pixels=synthetic;backend=Empty()
    result=p.paired_search(tmp_path,e,q,c,raw,backend,stream=enumerate(pixels),check=lambda path:{})
    assert result['unique_pairs']==72 and backend.feature_calls==18
    assert all(r['match_count']==0 and r['score']==0 for rows in result['records'].values() for r in rows)
    for variant in p.PROFILE['variants']:
        assert [s['selected_ref_index'] for s in result['selections'][variant]]==[7,0]
    assert all(v['count']==0 and len(v['sha256'])==64 for event in result['feature_events'] for v in event['views'])


def test_none_F_never_reads_uninitialised_mask(monkeypatch):
    class Poison:
        def __array__(self,*a,**k):pytest.fail('F=None mask must not be read')
    monkeypatch.setattr(p.cv2,'findFundamentalMat',lambda *a,**k:(None,Poison()))
    metrics,fit=p.captured_geometry(POINTS,POINTS)
    assert metrics['score']==0 and fit['F'].size==fit['mask'].size==0 and fit['status']=='none'


def test_saved_F_preserves_dtype_sensitive_rank_decision(tmp_path,monkeypatch):
    matrix=np.diag(np.array([1.,1.,1e-8],np.float32))
    assert np.linalg.matrix_rank(matrix)==2 and np.linalg.matrix_rank(matrix.astype(np.float64))==3
    monkeypatch.setattr(p.cv2,'findFundamentalMat',lambda *a,**k:(matrix.copy(),np.ones((8,1),np.uint8)))
    metrics,fit=p.captured_geometry(POINTS,POINTS)
    writer=p.CorrespondenceWriter(tmp_path/'correspondences');writer.start(0)
    saved=writer.add(POINTS,POINTS,fit);writer.finish()
    with np.load(tmp_path/saved['chunk'],allow_pickle=False) as z:
        assert z[saved['prefix']+'_F'].dtype==np.float32
        assert p.replay_geometry({'evidence':saved,'correspondences_sha256':saved['correspondences_sha256']},z)==metrics


def test_actual_opencv_epipolar_fit_is_retained_and_replayed_without_a_model(tmp_path):
    rng=np.random.default_rng(14);xyz=np.column_stack([rng.uniform(-3,3,40),rng.uniform(-2,2,40),rng.uniform(8,20,40)])
    angle=.07;R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    other=xyz@R.T+np.array([1.,.05,0.])
    def project(points):return np.ascontiguousarray(points[:,:2]/points[:,2:]*256+255.5,dtype=np.float32)
    a,b=project(xyz),project(other);metrics,fit=p.captured_geometry(a,b)
    assert metrics['status']=='ok' and metrics['inlier_count']>=8 and fit['mask'].size==40
    writer=p.CorrespondenceWriter(tmp_path/'correspondences');writer.start(0);saved=writer.add(a,b,fit);writer.finish()
    with np.load(tmp_path/saved['chunk'],allow_pickle=False) as z:
        assert p.replay_geometry({'evidence':saved,'correspondences_sha256':saved['correspondences_sha256']},z)==metrics


@pytest.mark.parametrize('fault',['pixels','missing_camera','missing_query','row_order','memory'])
def test_partial_or_changed_input_refuses_complete_component(tmp_path,synthetic,fault):
    e,q,c,raw,pixels=synthetic;stream=list(enumerate(pixels));check=lambda path:{}
    if fault=='pixels':q['pixels_sha256']='0'*64
    if fault=='missing_camera':del c[e['ref_ids'][0]]
    if fault=='missing_query':stream=stream[:1]
    if fault=='row_order':stream=stream[::-1]
    if fault=='memory':
        def check(path):raise ValueError('headroom')
    out=tmp_path/'component'
    with pytest.raises(ValueError):p.run_component(out,e,q,c,raw,Backend(),stream=stream,check=check)
    assert (out/'failure.json').is_file() and not (out/'frozen_prediction.json').exists()
    assert not (out/'prediction_original_single.json').exists()


def test_correspondence_disk_budget_refuses_rather_than_prunes(tmp_path,synthetic,monkeypatch):
    monkeypatch.setitem(p.PROFILE['correspondence_evidence'],'max_total_bytes',1)
    with pytest.raises(ValueError,match='no trimming'):search(tmp_path,synthetic)
    assert not list((tmp_path/'correspondences').glob('*.npz'))


def test_network_blocked_and_socket_restored():
    before=socket.socket.connect
    with p.offline(),socket.socket() as s:
        with pytest.raises(RuntimeError,match='network'):s.connect(('127.0.0.1',1))
    assert socket.socket.connect is before


@pytest.mark.parametrize('name',['ground_truth/route.json','offline_scores.json','run2_official_registered_score.json'])
def test_forbidden_manifest_members_refused_before_hash(tmp_path,name,monkeypatch):
    frozen=tmp_path/'frozen.json';p.write(frozen,{'artifacts':{name:'no'}});original=p.sha
    def spy(path):
        assert Path(path)==frozen,'forbidden member hashed'
        return original(path)
    monkeypatch.setattr(p,'sha',spy)
    with pytest.raises(ValueError,match='outside'):p.verify_freeze(frozen)


@pytest.fixture
def metadata_family(tmp_path,synthetic):
    e=synthetic[0];root=tmp_path/'metadata';root.mkdir()
    plan=p.metadata.plan_requests(e['ref_ids'],e['ref_latlon'],provenance={})
    body=__import__('json').dumps({'data':[{'id':str(v).split(':')[1],'camera_type':'spherical' if i==0 else 'perspective'} for i,v in enumerate(e['ref_ids'])]}).encode()
    batch=plan['batches'][0];prefix=root/'attempts/batch_0000_attempt_1';prefix.mkdir(parents=True)
    p.write(prefix/'request.json',{'endpoint':p.metadata.PROFILE['endpoint'],'params':batch['params'],'batch_index':0})
    p.write(prefix/'response.json',{'status':200,'error_type':None,'parsed_successfully':True,
        'body_redacted_or_reencoded':False,'stored_body_sha256':hashlib.sha256(body).hexdigest(),'decoded_body_sha256':hashlib.sha256(body).hexdigest()})
    (prefix/'response.body').write_bytes(body)
    p.write(root/'metadata_plan.json',plan)
    p.write(root/'all_reference_metadata.json',p.metadata.parse_response(body,batch['reference_ids']))
    p.write(root/'request_ledger.json',{'all_batches_successful':True,'requests':1,'decoded_body_bytes':len(body),
        'batches':[{'index':0,'success':True,'attempts':['batch_0000_attempt_1']}]})
    def freeze():
        path=root/'acquisition_frozen.json'
        if path.exists():path.unlink()
        p.write(path,{'status':'complete','all_ids_retained':True,'models_or_labels':False,
            'selected_answer':None,'artifacts':p.artifact_hashes(root)})
        return path,p.sha(path)
    return root,freeze


def test_camera_metadata_rebuilt_from_executed_frozen_responses(metadata_family,synthetic):
    _,freeze=metadata_family;path,h=freeze();record,cameras=p.inspect_metadata(path,h,synthetic[0])
    assert record['reference_count']==8 and record['panorama_count']==1 and len(cameras)==8


@pytest.mark.parametrize('fault',['records','request','partial','IDs','response'])
def test_reblessed_metadata_cannot_change_public_camera_or_membership(metadata_family,synthetic,fault):
    root,freeze=metadata_family;e=synthetic[0]
    target={'records':root/'all_reference_metadata.json','request':root/'attempts/batch_0000_attempt_1/request.json',
        'response':root/'attempts/batch_0000_attempt_1/response.json'}.get(fault)
    if target:
        value=p.read(target);target.unlink()
        if fault=='records':value[e['ref_ids'][0]]['fields']['camera_type']='perspective'
        if fault=='request':value['params']['fields']='id'
        if fault=='response':value['body_redacted_or_reencoded']=True
        p.write(target,value)
    path,h=freeze()
    if fault=='partial':
        value=p.read(path);value['status']='partial_unknown';path.unlink();p.write(path,value);h=p.sha(path)
    if fault=='IDs':e['ref_ids'][0]='mapillary:999'
    with pytest.raises(ValueError):p.inspect_metadata(path,h,e)


def test_group_uses_one_backend_and_shared_correspondence_budget(tmp_path,synthetic):
    e,q,c,raw,pixels=synthetic;backend=Backend()
    jobs=[dict(name=n,evidence=e,query=q,cameras=c,raw=raw,stream=enumerate(pixels)) for n in ['night','municipal']]
    result=p.run_group(tmp_path,jobs,backend,check=lambda path:{})
    assert set(result['galleries'])=={'night','municipal'} and result['correspondence_budget_used']['pairs']==144
    assert result['correspondence_budget_used']['bytes']==sum(p.read(tmp_path/n/'component_runtime.json')['correspondence_bytes'] for n in ['night','municipal'])
    assert backend.feature_calls==36
    assert not (tmp_path/'frozen_predictions.json').exists()  # Only verified top-level execution may freeze.


def test_second_gallery_failure_never_blesses_first(tmp_path,synthetic):
    e,q,c,raw,pixels=synthetic
    jobs=[dict(name=n,evidence=e,query=q,cameras=c,raw=raw,stream=enumerate(pixels if n=='night' else pixels[:1])) for n in ['night','municipal']]
    with pytest.raises(ValueError):p.run_group(tmp_path,jobs,Backend(),check=lambda path:{})
    assert (tmp_path/'night/prediction_original_single.json').exists()
    assert (tmp_path/'municipal/failure.json').exists() and not (tmp_path/'frozen_predictions.json').exists()


@pytest.mark.parametrize('fault',['state','weight','threads'])
def test_actual_backend_identity_mismatch_refused_and_paths_are_not_identity(fault):
    from types import SimpleNamespace
    assets={'root':'old','commit':'official','weight':{'path':'old.pt','sha256':'a'*64}}
    runtime={n:v for n,v in zip(['torch_version','cpu_threads','interop_threads','device','weight_load','lighterglue_constructed'],['version',2,1,'cpu','strict',False])}
    prior={'backend':{'actual_numerical_state':{'sha256':'b'*64},'assets':assets,'runtime':runtime}}
    backend=SimpleNamespace(state=copy.deepcopy(prior['backend']['actual_numerical_state']),assets=copy.deepcopy(assets),runtime=copy.deepcopy(runtime))
    backend.assets['root']='relocated';backend.assets['weight']['path']='relocated.pt'
    p.compatible_backend(backend,prior)
    if fault=='state':backend.state['sha256']='c'*64
    if fault=='weight':backend.assets['weight']['sha256']='c'*64
    if fault=='threads':backend.runtime['cpu_threads']=4
    with pytest.raises(ValueError):p.compatible_backend(backend,prior)


@pytest.fixture
def mocked_execute(tmp_path,synthetic,monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace,ModuleType
    import sys
    e,q,c,raw,pixels=synthetic;out=tmp_path/'run';flags={'constructor':0,'closed':0,'checks':0}
    args=SimpleNamespace(gallery_input_freeze_sha256='input-sha',gallery_metadata_freeze=tmp_path/'metadata.json',
        gallery_metadata_freeze_sha256='metadata-sha',assets=tmp_path/'assets',source_commit='pin')
    job={'output':out,'reg':{'video':{'path':'unused'}},'evidence':e,'query':q,'prior':{},'cameras':c,
        'gallery':tmp_path/'gallery','gallery_meta':{'metadata':'same'},'sources':{}}
    flags['job']=job;flags['real_source_identity']=p.source_identity
    monkeypatch.setattr(p,'source_identity',lambda *a:{})
    monkeypatch.setattr(p,'inspect_group',lambda a:job)
    def prepare(a,j):
        out.mkdir();j['_owned_output']=True;p.write(out/'preregistration.json',{'profile':p.PROFILE_SHA256})
    monkeypatch.setattr(p,'prepare_output',prepare)
    def resource(path):
        flags['checks']+=1
        if flags.get('fault')=='memory' and flags['checks']==4:raise ValueError('final headroom')
        return {'physical_bytes':9*1024**3,'commit_bytes':9*1024**3}
    monkeypatch.setattr(p,'resources',resource)
    @contextmanager
    def night(*a):
        try:yield raw,{'owned':'staged'}
        finally:flags['closed']+=1
    monkeypatch.setattr(p,'night_pixels',night)
    monkeypatch.setattr(p.inputs,'stream_queries',lambda *a:enumerate(pixels))
    @contextmanager
    def municipal(*a,**k):
        ev=dict(e,similarities=np.tile(e['similarities'],(40,1)))
        try:yield {'registration':{'owned':'staged'},'evidence':ev,'query':q,'raw':raw,'stream':enumerate(pixels),'verify_inputs':lambda:None}
        finally:flags['closed']+=1
    adapter=ModuleType('scripts.panorama_gallery_inputs');adapter.prepared_gallery_inputs=municipal
    adapter.__file__=str(p.ROOT/'scripts/panorama_gallery_inputs.py')
    monkeypatch.setitem(sys.modules,'scripts.panorama_gallery_inputs',adapter)
    monkeypatch.setattr(p,'inspect_metadata',lambda *a:(job['gallery_meta'],c))
    monkeypatch.setattr(p,'compatible_backend',lambda *a:None)
    class Fake:
        assets={};runtime={};state={}
        def __init__(self,*a):
            flags['constructor']+=1;assert (out/'inputs_frozen.json').exists()
        def verify_unchanged(self):
            if flags.get('fault')=='backend':raise ValueError('changed actual state')
    monkeypatch.setattr(p,'ReleasedXFeat',Fake)
    def group(output,jobs,backend):
        assert [j['name'] for j in jobs]==['night','municipal']
        for j in jobs:
            folder=out/j['name'];folder.mkdir()
            if flags.get('fault')=='second' and j['name']=='municipal':raise ValueError('second failed')
            for v in p.PROFILE['variants']:p.write(folder/f'prediction_{v}.json',{'selected_answer':None})
        if flags.get('fault')=='score':(out/'offline_scores.json').write_text('forbidden')
        return {'galleries':{'night':{},'municipal':{}}}
    monkeypatch.setattr(p,'run_group',group)
    def final(*a):
        assert flags['closed']==2  # Source validation must happen after context cleanup.
        if flags.get('fault')=='inputs':raise ValueError('original input changed')
    monkeypatch.setattr(p,'final_inputs',final)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')
    return args,out,flags


def test_top_level_freezes_only_all_four_after_inputs_staged_before_one_constructor(mocked_execute):
    args,out,flags=mocked_execute
    result=p.execute(args);frozen=p.verify_freeze(out/'frozen_predictions.json',result['freeze_sha256'])
    assert frozen['all_four_outputs_complete'] and frozen['selected_variant'] is None
    assert flags['constructor']==1 and flags['closed']==2
    assert len([n for n in frozen['artifacts'] if 'prediction_' in n])==4


@pytest.mark.parametrize('fault',['second','backend','inputs','memory','score'])
def test_top_level_partial_stale_or_low_memory_never_gets_master_success(mocked_execute,fault):
    args,out,flags=mocked_execute;flags['fault']=fault
    with pytest.raises(ValueError):p.execute(args)
    assert flags['closed']==2 and (out/'failure.json').exists() and not (out/'frozen_predictions.json').exists()


@pytest.mark.parametrize('module',[p.inputs,p.original,p.orb,p.xbackend])
def test_foreign_helper_rejected_before_source_git_or_hash(module,tmp_path,monkeypatch):
    monkeypatch.setattr(module,'__file__',str(tmp_path/'foreign.py'))
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**k:pytest.fail('git before imported-helper origin check'))
    with pytest.raises(ValueError,match='foreign'):p.source_identity('pin')


def test_loaded_package_escape_refused_before_source_git(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules,'monocular_osm.foreign',SimpleNamespace(__file__=str(tmp_path/'foreign.py')))
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**k:pytest.fail('git before loaded origin check'))
    with pytest.raises(ValueError,match='escaped'):p.source_identity('pin')


def test_foreign_deferred_gallery_helper_declines_before_backend_constructor(mocked_execute,tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    args,out,flags=mocked_execute
    flags['job']['sources']={name:p.sha(p.ROOT/name) for name in p.SOURCES}
    monkeypatch.setattr(p,'source_identity',flags['real_source_identity'])
    monkeypatch.setitem(sys.modules,'scripts.pilot_image_gallery',SimpleNamespace(__file__=str(tmp_path/'foreign.py')))
    with pytest.raises(ValueError,match='escaped'):p.execute(args)
    assert flags['constructor']==0 and (out/'failure.json').exists() and not (out/'frozen_predictions.json').exists()


def test_wrong_profile_cannot_inspect_data_or_construct_backend(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(p,'source_identity',lambda *a:pytest.fail('source or data access before profile check'))
    with pytest.raises(ValueError,match='profile'):p.inspect_group(SimpleNamespace(profile_sha256='wrong'))


def test_resources_require_both_actual_commit_and_physical(monkeypatch,tmp_path):
    monkeypatch.setattr(p.inputs,'headroom',lambda:{'physical_bytes':9*1024**3,'commit_bytes':7*1024**3})
    if p.os.name=='nt':
        with pytest.raises(ValueError,match='AND'):p.resources(tmp_path)


@pytest.mark.parametrize('changed',['gallery','night'])
def test_final_inputs_rechecks_sources_mutated_by_context_cleanup(tmp_path,monkeypatch,changed):
    from contextlib import contextmanager
    from types import SimpleNamespace
    night,gallery,prior,out=[tmp_path/n for n in ['night','gallery','prior','output']]
    for root in (night,gallery,prior,out):root.mkdir()
    extra=night/'reference_snapshot.json';extra.write_text('{}')
    original=gallery/'protected.bin';original.write_bytes(b'original')
    p.write(night/'frozen_prediction.json',{'artifacts':{'reference_snapshot.json':p.sha(extra)}})
    p.write(gallery/'inputs_frozen.json',{'artifacts':{'protected.bin':p.sha(original)}})
    p.write(prior/'frozen_prediction.json',{'artifacts':{}})
    p.write(out/'inputs_frozen.json',{'artifacts':{}})
    metadata=[]
    for name in ['metadata_night.json','metadata_gallery.json']:
        path=tmp_path/name;p.write(path,{'artifacts':{}});metadata.append(path)
    reg={'frozen_prediction':{'path':str(night/'frozen_prediction.json'),'sha256':p.sha(night/'frozen_prediction.json')}}
    for name in ['matrix','recovery_lineage','raw_store','video']:
        path=night/(name+'.bin');path.write_bytes(name.encode());reg[name]={'path':str(path),'sha256':p.sha(path)}
    (night/'query_input.json').write_text('{}');reg['query_input_sha256']=p.sha(night/'query_input.json')
    args=SimpleNamespace(input_run=night,prior_run=prior,gallery_input_freeze_sha256=p.sha(gallery/'inputs_frozen.json'),
        metadata_freeze=metadata[0],metadata_freeze_sha256=p.sha(metadata[0]),gallery_metadata_freeze=metadata[1],
        gallery_metadata_freeze_sha256=p.sha(metadata[1]),source_commit='pin',assets=tmp_path/'assets')
    job={'reg':reg,'gallery':gallery,'prior':{'freeze_sha256':p.sha(prior/'frozen_prediction.json')},'output':out,'sources':{},'assets':{}}
    monkeypatch.setattr(p,'source_identity',lambda *a:{})
    monkeypatch.setattr(p,'inspect_assets',lambda *a:{})
    p.final_inputs(args,job)  # The setup is valid before cleanup.
    @contextmanager
    def acquired_sources():
        try:yield
        finally:(original if changed=='gallery' else extra).write_bytes(b'changed only during cleanup')
    with acquired_sources():pass
    with pytest.raises(ValueError,match='frozen member changed'):p.final_inputs(args,job)
