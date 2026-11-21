"""Portable native-array geometry regressions; no actual videos or ignored IO."""
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from monocular_osm import multiview_motion as f
from tests import multiview_motion_fixtures as p


@pytest.fixture(autouse=True)
def threads_and_no_true_pose_input(monkeypatch):
    old=cv2.getNumThreads();cv2.setNumThreads(2)
    original=f.solve_window
    @wraps(original)
    def guard(tracks,K,**kwargs):
        assert not {'rotations','translations','centres','trueR','trueC','true_R','true_T'} & set(tracks)
        return original(tracks,K,**kwargs)
    monkeypatch.setattr(f,'solve_window',guard)
    try:yield
    finally:cv2.setNumThreads(old)


def test_exact_reviewed_source_and_profile_with_no_ignored_output_dependency():
    # This version's promotion requires byte equivalence, not a rewritten kernel.
    assert hashlib.sha256(Path(f.__file__).read_bytes()).hexdigest()==(
        'f6abbf1de3af69c8733291d7ebd81ec9e56e28395466fe70c203dcd3469190b6')
    profile=json.dumps(f.PROFILE,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert hashlib.sha256(profile).hexdigest()==(
        '7c1e7828481f3b11073e1e84c724da7ece8e3a8ca609062f207de43a201352bd')
    assert list(inspect.signature(f.solve_window).parameters)==['tracks','K','check']
    assert Path(p.__file__).resolve().parent==Path(__file__).resolve().parent


@pytest.mark.parametrize('kind,noise,complete',[
    ('observable',0.,True),('observable',.1,True),('rotation_only',.1,False),('low_parallax',.1,False)])
def test_pose_is_estimated_from_tracks_and_weak_translation_stays_unknown(kind,noise,complete):
    tracks,K,trueR,trueC=p.scene(kind,noise)
    before={key:value.tobytes() for key,value in tracks.items() if isinstance(value,np.ndarray)}
    result=f.solve_window(tracks,K)
    assert all(value.tobytes()==before[key] for key,value in tracks.items() if isinstance(value,np.ndarray))
    assert result['metric_scale_recovered'] is result['calibrated_covariance'] is result['measured_stop'] is False
    if not complete:
        assert result['status']=='unavailable' and result['translation_status']=='unknown' and result['centres'] is None
        return
    assert result['status']=='complete_relative_gauge_candidate'
    scale=np.linalg.norm(trueC[3])
    assert np.max(np.linalg.norm(result['centres']-trueC/scale,axis=1))<.1
    assert max(f.rotation_degrees(a@b.T) for a,b in zip(result['rotations'],trueR))<1.
    assert [r['frame'] for r in result['later_view_check_reprojection']]==[1,2,4,5,6]
    assert all(r['check_denominator']==80 for r in result['later_view_check_reprojection'])


def test_physical_3D_images_track_then_estimate_R_T_without_true_pose_arguments():
    frames,K,trueR,trueC=p.render()
    before=[frame.tobytes() for frame in frames]
    tracks=f.track_window(frames,K,np.arange(100,128,4,dtype=np.int64),12.+np.arange(7,dtype=np.float64)*4./30.)
    result=f.solve_window(tracks,K)
    assert result['status']=='complete_relative_gauge_candidate'
    assert [frame.tobytes() for frame in frames]==before
    assert tracks['persistent_count']>=40 and len(tracks['seed_ids'])<=800
    assert np.max(np.linalg.norm(result['centres']-trueC/np.linalg.norm(trueC[3]),axis=1))<.05
    assert max(f.rotation_degrees(a@b.T) for a,b in zip(result['rotations'],trueR))<1.
    actual=result['adjacent_steps_gauge']
    expected=np.linalg.norm(np.diff(trueC,axis=0),axis=1)/np.linalg.norm(trueC[3])
    assert np.max(np.abs(actual-expected))<.02 and actual.max()/actual.min()>3.
    rows=result['later_view_check_reprojection']
    assert len({r['check_denominator'] for r in rows})==1
    assert all(not r['invalid_check_ids'] and r['rmse_px']<=4. for r in rows)
    assert all(all(seed%2==0 for seed in r['fit_inlier_ids']) for r in rows)
    assert result['static_world_verified'] is result['metric_scale_recovered'] is False


def test_original_nonphysical_point_patch_image_negative_is_retained():
    result=p.rendered_case()
    assert result['status']=='unavailable' and result['persistent_count']>=40
    assert 'held-out later-view geometry failed' in result['reason']


def test_blank_disconnected_and_corrupt_check_track_never_force_motion():
    tracks,K,_,_=p.scene()
    blank=f.track_window([np.zeros((480,640),np.uint8) for _ in range(7)],K,tracks['ordinals'],tracks['timestamps'])
    assert blank['status']=='unavailable_no_seeds'
    result=f.solve_window(blank,K)
    assert result['status']=='unavailable' and not result['measured_stop']
    tracks['valid'][3,:]=False
    assert f.solve_window(tracks,K)['status']=='unavailable'
    tracks,K,_,_=p.scene()
    tracks['points_px'][5,1,0]+=80.
    result=f.solve_window(tracks,K)
    assert result['status']=='unavailable'
    assert result['later_view_check_reprojection'][-1]['check_denominator']==80


def test_ALL_E_refinement_Wahba_and_PnP_inputs_are_FIT_only(monkeypatch):
    tracks,K,_,_=p.scene('observable',.1)
    pixels=tracks['points_px'];fit=np.flatnonzero(tracks['fit_seed'])
    pair_sequence=[(0,3),(1,6)];e_calls=[];f_calls=[];pnp_calls=[];refine_calls=[];wahba_calls=[]
    real_E,real_F,real_pnp,real_refine,real_wahba=(f.cv2.findEssentialMat,f.cv2.findFundamentalMat,
        f.cv2.solvePnPRansac,f.cv2.solvePnP,f.rotate_only)
    normals={i:cv2.undistortPoints(pixels[i,fit].reshape(-1,1,2),K,None).reshape(-1,2) for i in range(7)}
    bearings={i:f.bearing(pixels[i,fit],K) for i in range(7)}
    rows=lambda a:{tuple(row) for row in a}
    def E(first,later,*args,**kwargs):
        pair=pair_sequence[len(e_calls)];e_calls.append(pair)
        np.testing.assert_array_equal(first,pixels[pair[0],fit])
        np.testing.assert_array_equal(later,pixels[pair[1],fit])
        return real_E(first,later,*args,**kwargs)
    def F(first,later,*args,**kwargs):
        pair=pair_sequence[len(f_calls)];f_calls.append(pair)
        expected={(tuple(a),tuple(b)) for a,b in zip(normals[pair[0]],normals[pair[1]])}
        assert all((tuple(a),tuple(b)) in expected for a,b in zip(first,later))
        return real_F(first,later,*args,**kwargs)
    def PnP(world,target,*args,**kwargs):
        view=(1,2,4,5,6)[len(pnp_calls)];pnp_calls.append(view)
        assert rows(target)<=rows(pixels[view,fit])
        return real_pnp(world,target,*args,**kwargs)
    def refine(world,target,*args,**kwargs):
        view=(1,2,4,5,6)[len(refine_calls)];refine_calls.append(view)
        assert rows(target)<=rows(pixels[view,fit])
        return real_refine(world,target,*args,**kwargs)
    def Wahba(first,later):
        pair=([(0,i) for i in range(1,7)]+[(1,3)])[len(wahba_calls)];wahba_calls.append(pair)
        np.testing.assert_array_equal(first,bearings[pair[0]])
        np.testing.assert_array_equal(later,bearings[pair[1]])
        return real_wahba(first,later)
    monkeypatch.setattr(f.cv2,'findEssentialMat',E)
    monkeypatch.setattr(f.cv2,'findFundamentalMat',F)
    monkeypatch.setattr(f.cv2,'solvePnPRansac',PnP)
    monkeypatch.setattr(f.cv2,'solvePnP',refine)
    monkeypatch.setattr(f,'rotate_only',Wahba)
    result=f.solve_window(tracks,K)
    assert result['status']=='complete_relative_gauge_candidate'
    assert e_calls==f_calls==pair_sequence and pnp_calls==refine_calls==[1,2,4,5,6]
    assert len(wahba_calls)==7
    assert [row['frame'] for row in result['later_view_check_reprojection']]==[1,2,4,5,6]
    assert all(row['check_denominator']==80 for row in result['later_view_check_reprojection'])

def test_tracking_loss_never_reassigns_original_FIT_CHECK_ID_parity(monkeypatch):
    # One odd seed fails at first forward LK; the later odd seed stays CHECK.
    seeds=np.array([[30.,10.],[10.,10.],[50.,10.],[20.,10.],[40.,10.]],np.float32).reshape(-1,1,2)
    monkeypatch.setattr(f.cv2,'goodFeaturesToTrack',lambda *a,**k:seeds.copy())
    calls=[]
    def flow(first,later,points,*args,**kwargs):
        forward=len(calls)%2==0;calls.append(len(points))
        moved=points.copy();moved[:,:,0]+=1. if forward else -1.
        status=np.ones((len(points),1),np.uint8)
        if len(calls)==1:status[1]=0
        return moved,status,np.zeros((len(points),1),np.float32)
    monkeypatch.setattr(f.cv2,'calcOpticalFlowPyrLK',flow)
    K=np.array([[60.,0.,32.],[0.,60.,32.],[0.,0.,1.]])
    tracks=f.track_window([np.zeros((64,64),np.uint8) for _ in range(7)],K,
        np.arange(0,28,4,dtype=np.int64),np.arange(7,dtype=np.float64)*.1)
    np.testing.assert_array_equal(tracks['seed_ids'],np.arange(5,dtype=np.int32))
    np.testing.assert_array_equal(tracks['fit_seed'],[True,False,True,False,True])
    np.testing.assert_array_equal(tracks['check_seed'],[False,True,False,True,False])
    assert not tracks['valid'][1:,1].any() and tracks['valid'][:,3].all()
    assert tracks['points_px'].shape==(7,5,2) and np.isnan(tracks['points_px'][1:,1]).all()
    assert tracks['rejection_reason'][0,1]=='forward_failure'
    assert np.all(tracks['rejection_reason'][1:,1]=='not_active')
    assert tracks['failed_seed_ids_retained'] is True and tracks['persistent_count']==4


@pytest.mark.parametrize('fault',['non_native_float','pixel_cap','clock','duplicate_ordinal','wrong_K'])
def test_input_bounds_refuse_before_any_image_or_geometry_callback(monkeypatch,fault):
    tracks,K,_,_=p.scene()
    frames=[np.zeros((480,640),np.uint8) for _ in range(7)]
    ordinals=tracks['ordinals'].copy();times=tracks['timestamps'].copy()
    if fault=='non_native_float':frames=[v.astype(np.float32) for v in frames]
    if fault=='pixel_cap':
        huge=np.lib.stride_tricks.as_strided(np.zeros(1,np.uint8),shape=(9000,9000),strides=(0,0))
        frames=[huge]*7
    if fault=='clock':times[2]=times[1]
    if fault=='duplicate_ordinal':ordinals[2]=ordinals[1]
    if fault=='wrong_K':K=K.astype(np.float32)
    monkeypatch.setattr(f.cv2,'cvtColor',lambda *a,**k:pytest.fail('conversion before input bound'))
    monkeypatch.setattr(f.cv2,'goodFeaturesToTrack',lambda *a,**k:pytest.fail('GFTT before input bound'))
    with pytest.raises(ValueError):f.track_window(frames,K,ordinals,times)


def test_supplied_K_and_original_split_cannot_be_silently_replaced():
    tracks,K,_,_=p.scene()
    tracks['K']=K.copy();tracks['K'][0,0]+=1.
    with pytest.raises(ValueError):f.solve_window(tracks,K)
    tracks,K,_,_=p.scene();tracks['fit_seed']=~tracks['fit_seed']
    with pytest.raises(ValueError):f.solve_window(tracks,K)
