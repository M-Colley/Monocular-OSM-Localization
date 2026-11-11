"""Private numerical/authority/finality fixtures; no actual study payloads."""
from pathlib import Path
import weakref
import cv2
import numpy as np
import pytest
from scripts import multiview_motion_probe as w
from tests import test_prepare_multiview_motion_inputs as prepare
from tests.multiview_motion_fixtures import render
from tests.multiview_motion_fixtures import scene


def owned(root,path):
    path=Path(path).resolve()
    assert path.is_relative_to(Path(root).resolve())
    return path


def rewrite(case,path,value):
    path=owned(case['root'],path)
    path.write_bytes(w.p.json_bytes(value))
    return w.p.ref(path)


def packet(tmp_path,monkeypatch):
    case=prepare.fixture(tmp_path,monkeypatch)
    manifest,_=prepare.run(case)
    config,qa=prepare.close_packet(case,manifest)
    config['output_dir']=str(case['root']/'numerical')
    path=case['root']/'numerical_config.json'
    w.p.write_json(path,config)
    case.update(run_config=config,run_config_ref=w.p.ref(path),qa=qa,manifest=manifest)
    return case


def run(case,**kwargs):
    return w.execute(case['run_config_ref']['path'],case['run_config_ref']['sha256'],**kwargs)


def reseal(case):
    case['run_config_ref']=rewrite(case,case['run_config_ref']['path'],case['run_config'])


def synthetic_lost_tracks(frames,K,ordinals,timestamps,*,check=lambda:None):
    check();n=4
    points=np.full((7,n,2),np.nan);points[0]=np.column_stack([np.arange(n),np.zeros(n)])
    valid=np.zeros((7,n),bool);valid[0]=True
    reasons=np.full((6,n),'not_active',dtype='<U32');reasons[0]='forward_failure'
    ids=np.arange(n,dtype=np.int32)
    return dict(status='tracked',seed_ids=ids,fit_seed=ids%2==0,check_seed=ids%2==1,
        points_px=points,valid=valid,forward_backward_error_px=np.full((6,n),np.nan),
        forward_patch_error=np.full((6,n),np.nan),rejection_reason=reasons,
        ordinals=ordinals.copy(),timestamps=timestamps.copy(),K=K.copy(),native_shape=frames[0].shape,
        resized=False,persistent_count=0,failed_seed_ids_retained=True,LK_error_is_covariance=False)


def unknown_geometry(tracks,K,*,check=lambda:None):
    check()
    return dict(status='unavailable',reason='private no persistent support',translation_status='unknown',
        centres=None,rotations=None,metric_scale_recovered=False,measured_stop=False,
        calibrated_covariance=False,static_world_verified=False,
        all_fit_seed_ids=[0,2],all_check_seed_ids=[1,3],discarded_seed_ids=[0,1,2,3])


def stub_frontend(monkeypatch):
    monkeypatch.setattr(w.frontend,'track_window',synthetic_lost_tracks)
    monkeypatch.setattr(w.frontend,'solve_window',unknown_geometry)


def test_finite_JSON_null_inventory_never_zero_or_array_expansion():
    value={'a/b~c':[1.,np.float32(np.nan),float('inf'),-float('inf')],'ok':None}
    clean,paths=w.finite_json(value)
    assert clean=={'a/b~c':[1.,None,None,None],'ok':None}
    assert paths==[{'path':'/a~1b~0c/1','kind':'nan'},
        {'path':'/a~1b~0c/2','kind':'positive_infinity'},
        {'path':'/a~1b~0c/3','kind':'negative_infinity'}]
    with pytest.raises(ValueError):w.finite_json({'raw':np.zeros(100)})
    with pytest.raises(ValueError):w.finite_json({1:'not a string key'})


def test_genuine_physical_image_window_keeps_all_raw_failure_arrays_and_relative_pose(tmp_path):
    frames,K,_,trueC=render()
    body=dict(frames_bgr=np.stack([np.repeat(f[:,:,None],3,axis=2) for f in frames]),K=K,
        video_ordinals=np.arange(100,128,4,dtype=np.int64),timestamps=12.+np.arange(7)*4./30.,
        native_read_rows=np.arange(7,dtype=np.int64))
    before={k:w.p.typed(v) for k,v in body.items()}
    old=cv2.getNumThreads();cv2.setNumThreads(2)
    try:
        with w.cpu_two_threads():arrays,meta=w.process_window(body)
    finally:cv2.setNumThreads(old)
    assert meta['status']=='complete_relative_gauge_candidate' and meta['raw_tracks_available']
    assert set(w.TRACK_ARRAYS)<=set(arrays) and 'frames_bgr' not in arrays
    assert np.isnan(arrays['points_px']).any() and np.isnan(arrays['forward_backward_error_px']).any()
    assert np.max(np.linalg.norm(arrays['centres']-trueC/np.linalg.norm(trueC[3]),axis=1))<.05
    assert {k:w.p.typed(v) for k,v in body.items()}==before
    assert meta['json_nonfinite_path_inventory']==[]
    assert [row['frame'] for row in meta['geometry']['later_view_check_reprojection']]==[1,2,4,5,6]
    assert len({row['check_denominator'] for row in meta['geometry']['later_view_check_reprojection']})==1
    path=tmp_path/'derived.npz';headers=w.write_arrays(path,arrays)
    with np.load(path,allow_pickle=False) as saved:
        assert set(saved.files)==set(arrays)
        for k,value in arrays.items():assert w.p.typed(saved[k])==w.p.typed(value)
    assert headers['points_px']==w.p.typed_header(arrays['points_px'])


def test_all32_unavailable_or_failed_retained_one_input_window_resident(tmp_path,monkeypatch):
    case=packet(tmp_path,monkeypatch);stub_frontend(monkeypatch)
    failed_ordinal=case['read']['windows'][7]['video_ordinals'][0]
    def tracking(frames,K,ordinals,times,**kwargs):
        if ordinals[0]==failed_ordinal:raise cv2.error('private tracking failure')
        return synthetic_lost_tracks(frames,K,ordinals,times,**kwargs)
    monkeypatch.setattr(w.frontend,'track_window',tracking)
    original=w.p.read_prepared_window;previous=[]
    def load(*args,**kwargs):
        if previous:assert previous[-1]() is None,'prior native window retained into next callback'
        value=original(*args,**kwargs);previous.append(weakref.ref(value['frames_bgr']));return value
    monkeypatch.setattr(w.p,'read_prepared_window',load)
    monkeypatch.setattr(cv2,'VideoCapture',lambda *a,**k:pytest.fail('numerical video reopen'))
    result=run(case)
    assert previous[-1]() is None and result['window_count']==32
    assert [row['window_id'] for row in result['windows']]==list(range(32))
    assert result['status_counts']=={'unavailable':31,'failed':1}
    assert result['raw_track_windows_available']==31 and not result['external_actor_closure_verified']
    output=Path(case['run_config']['output_dir'])
    fail=w.p.strict_json((output/'window_07.json').read_bytes())
    assert fail['failure']['stage']=='track_window' and fail['raw_tracks_available'] is False
    assert fail['geometry'] is None and fail['track_array_roles']==[]
    with np.load(output/'window_07.npz',allow_pickle=False) as z:
        assert set(z.files)=={'native_read_rows','video_ordinals','timestamps','K'}
    with np.load(output/'window_00.npz',allow_pickle=False) as z:
        assert np.isnan(z['points_px'][1:]).all() and z['rejection_reason'][0,1]=='forward_failure'
    assert len(list(output.iterdir()))==66 and len(result['artifacts'])==65


@pytest.mark.parametrize('fault',['closure','count','source','last_array','manifest_source'])
def test_whole32_root_source_integrity_gate_before_ANY_NPZ_member(tmp_path,monkeypatch,fault):
    case=packet(tmp_path,monkeypatch)
    if fault in ('closure','count'):
        qa=dict(case['qa']);qa['all_observed_worker_births_closed']=False if fault=='closure' else True
        if fault=='count':qa['window_count']=31
        case['run_config']['prepared_input_acceptance_ref']=rewrite(case,
            case['run_config']['prepared_input_acceptance_ref']['path'],qa)
    elif fault=='source':
        source=case['root']/'source'/'monocular_osm/multiview_motion.py'
        owned(case['root'],source).write_bytes(b'private source drift')
    elif fault=='last_array':
        path=owned(case['root'],case['manifest']['windows'][-1]['array_ref']['path'])
        path.write_bytes(path.read_bytes()+b'private last input drift')
    else:
        manifest=dict(case['manifest']);manifest['source_hashes']={}
        changed=rewrite(case,case['qa']['prepared_manifest_ref']['path'],manifest)
        qa=dict(case['qa'],prepared_manifest_ref=changed)
        case['run_config']['prepared_input_acceptance_ref']=rewrite(case,
            case['run_config']['prepared_input_acceptance_ref']['path'],qa)
    reseal(case)
    monkeypatch.setattr(w.p,'read_prepared_window',lambda *a,**k:pytest.fail('NPZ before wholecohort admission'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('array before wholecohort admission'))
    with pytest.raises(ValueError):run(case)
    assert not Path(case['run_config']['output_dir']).exists()


@pytest.mark.parametrize('fault',['array','json','root_qa','source','extra','origin'])
def test_last_callback_drift_refuses_final_master(tmp_path,monkeypatch,fault):
    case=packet(tmp_path,monkeypatch);stub_frontend(monkeypatch)
    output=Path(case['run_config']['output_dir']);fired=[]
    def check():
        if fired or not (output/'results.json').exists():return
        if fault=='origin':
            fake=owned(case['root'],case['root']/'fake_frontend.py');fake.write_text('# private',encoding='utf-8')
            monkeypatch.setattr(w.frontend,'__file__',str(fake))
        else:
            paths={'array':output/'window_00.npz','json':output/'window_00.json',
                'root_qa':Path(case['run_config']['prepared_input_acceptance_ref']['path']),
                'source':case['root']/'source'/'scripts/multiview_motion_probe.py','extra':output/'unexpected.txt'}
            path=owned(case['root'],paths[fault])
            path.write_bytes((path.read_bytes() if path.exists() else b'')+b'private late drift')
        fired.append(True)
    with pytest.raises(ValueError):run(case,check=check)
    assert fired and not (output/'frozen_run.json').exists()


@pytest.mark.parametrize('fault',['root_qa','extra'])
def test_final_derived_hash_work_cannot_escape_last_input_and_membership_guards(tmp_path,monkeypatch,fault):
    case=packet(tmp_path,monkeypatch);stub_frontend(monkeypatch)
    output=Path(case['run_config']['output_dir']);fired=[];real_sha=w.p.sha
    def sha(path):
        if not fired and Path(path)==output/'window_00.npz' and (output/'results.json').exists():
            target=(Path(case['run_config']['prepared_input_acceptance_ref']['path'])
                    if fault=='root_qa' else output/'unexpected_after_hash.txt')
            target=owned(case['root'],target)
            target.write_bytes((target.read_bytes() if target.exists() else b'')+b'private hash-boundary mutation')
            fired.append(True)
        return real_sha(path)
    monkeypatch.setattr(w.p,'sha',sha)
    with pytest.raises(ValueError):run(case)
    assert fired and not (output/'frozen_run.json').exists()


def small_body():
    return dict(frames_bgr=np.zeros((7,32,32,3),np.uint8),K=np.eye(3),
        video_ordinals=np.arange(7,dtype=np.int64),timestamps=np.arange(1.,8.),
        native_read_rows=np.arange(7,dtype=np.int64))


def test_own_input_mutation_refused_without_zero_repair(monkeypatch):
    body=small_body()
    def mutate(frames,*args,**kwargs):
        result=synthetic_lost_tracks(frames,*args,**kwargs);frames[0][0,0,0]=1;return result
    monkeypatch.setattr(w.frontend,'track_window',mutate)
    monkeypatch.setattr(w.frontend,'solve_window',unknown_geometry)
    with pytest.raises(ValueError,match='changed prepared'):w.process_window(body)
    assert body['frames_bgr'][0,0,0,0]==1


def test_nonfinite_solver_metadata_explicit_null_paths_raw_NPZ_nan(monkeypatch):
    stub_frontend(monkeypatch)
    def geometry(*args,**kwargs):
        result=unknown_geometry(*args,**kwargs)
        result['metrics']={'a':float('nan'),'b':float('inf')};return result
    monkeypatch.setattr(w.frontend,'solve_window',geometry)
    arrays,meta=w.process_window(small_body())
    assert meta['geometry']['metrics']=={'a':None,'b':None}
    assert meta['json_nonfinite_path_inventory']==[
        {'path':'/geometry/metrics/a','kind':'nan'},
        {'path':'/geometry/metrics/b','kind':'positive_infinity'}]
    assert np.isnan(arrays['points_px'][1:]).all()


def test_resource_callback_not_downgraded_to_window_unavailable():
    def timeout():raise TimeoutError('private resource failure')
    with pytest.raises(TimeoutError):w.process_window(small_body(),check=timeout)


def test_failed_CHECK_geometry_keeps_partial_metrics_full_denominator_and_raw_ids(monkeypatch):
    tracks,K,_,_=scene('observable')
    tracks['points_px'][5,1,0]+=80.  # private misassociation in an originally CHECK track
    n=len(tracks['seed_ids'])
    tracks.update(status='tracked',forward_backward_error_px=np.zeros((6,n)),
        forward_patch_error=np.zeros((6,n)),rejection_reason=np.full((6,n),'accepted',dtype='<U32'),
        native_shape=(480,640,3),resized=False,persistent_count=n,
        failed_seed_ids_retained=True,LK_error_is_covariance=False)
    body=dict(frames_bgr=np.zeros((7,480,640,3),np.uint8),K=K,
        video_ordinals=tracks['ordinals'].copy(),timestamps=tracks['timestamps'].copy(),
        native_read_rows=np.arange(7,dtype=np.int64))
    monkeypatch.setattr(w.frontend,'track_window',lambda *a,**k:tracks)
    old=cv2.getNumThreads();cv2.setNumThreads(2)
    try:arrays,meta=w.process_window(body)
    finally:cv2.setNumThreads(old)
    assert meta['status']=='unavailable' and meta['raw_tracks_available'] is True
    rows=meta['geometry']['later_view_check_reprojection']
    assert [row['frame'] for row in rows]==[1,2,4,5]
    assert all(row['check_denominator']==80 for row in rows)
    assert rows[-1]['rmse_px']>4 and meta['geometry']['centres'] is None
    assert arrays['points_px'][5,1,0]==tracks['points_px'][5,1,0]
    assert meta['geometry']['all_check_seed_ids']==list(range(1,160,2))


def test_numerical_phase_never_opens_original_historical_assets(tmp_path,monkeypatch):
    case=packet(tmp_path,monkeypatch);stub_frontend(monkeypatch)
    blocked={Path(r['path']).resolve() for r in case['read']['basis_refs'].values()}
    blocked.add(Path(case['read']['video_binding']['path']).resolve())
    blocked.add(Path(case['manifest']['prepare_config_ref']['path']).resolve())
    original=Path.open
    def open_checked(path,*args,**kwargs):
        assert path.resolve() not in blocked,'numerical phase reopened historical original'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',open_checked)
    result=run(case)
    assert result['window_count']==32 and result['new_neural_forwards_or_video_reopen'] is False
