"""Portable synthetic fixtures; no ignored outputs, actual files or model IO.

True poses are used only by the renderer and test oracle. The frontend receives
only images/tracks, native K and clocks. This fixture does not certify real K.
"""
import hashlib
import cv2
import numpy as np
from monocular_osm import multiview_motion as f


def yaw(degrees):
    a = np.deg2rad(degrees)
    return np.array([[np.cos(a), 0., np.sin(a)], [0., 1., 0.], [-np.sin(a), 0., np.cos(a)]])

def project(points, rotations, centres, K):
    pixels = []
    for R, C in zip(rotations, centres):
        h = ((points-C) @ R.T) @ K.T
        assert np.all(h[:, 2] > 0)
        pixels.append(h[:, :2]/h[:, 2:])
    return np.asarray(pixels)

def scene(kind='observable', noise=0.):
    rng = np.random.default_rng(902)
    K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
    depths = rng.uniform(8., 20., 160) if kind != 'low_parallax' else rng.uniform(50., 100., 160)
    points = np.column_stack([rng.uniform(-.2, .2, len(depths))*depths,
                              rng.uniform(-.15, .15, len(depths))*depths, depths])
    rotations = np.stack([yaw(i*.35) for i in range(7)])
    centres = np.array([[0.,0.,0.],[.12,0.,.06],[.3,0.,.12],[.6,0.,.2],
                        [.9,0.,.28],[1.3,0.,.36],[1.8,0.,.45]])
    if kind == 'rotation_only':
        centres[:] = 0.
    if kind == 'low_parallax':
        centres = np.column_stack([np.zeros(7), np.zeros(7), np.arange(7)*1e-4])
    pixels = project(points, rotations, centres, K)
    pixels += np.random.default_rng(882).normal(0., noise, pixels.shape)
    assert np.all(pixels >= 0) and np.all(pixels[:, :, 0] < 640) and np.all(pixels[:, :, 1] < 480)
    ids = np.arange(pixels.shape[1], dtype=np.int32)
    tracks = {'points_px': pixels, 'valid': np.ones(pixels.shape[:2], bool),
        'seed_ids': ids, 'fit_seed': ids%2 == 0, 'check_seed': ids%2 == 1,
        'K': K.copy(), 'native_shape': (480,640), 'ordinals': np.arange(100,128,4,dtype=np.int64),
        'timestamps': 12. + np.arange(7,dtype=np.float64)*4./30.}
    return tracks, K, rotations, centres

def rendered_case():
    tracks,K,_,_=scene('observable')
    rng=np.random.default_rng(411)
    patches=rng.integers(40,256,(len(tracks['seed_ids']),11,11),dtype=np.uint8)
    frames=[]
    for row in tracks['points_px']:
        frame=np.full((480,640),10,np.uint8)
        for (x,y),patch in zip(row,patches):
            left,top=int(np.floor(x))-7,int(np.floor(y))-7
            M=np.array([[1.,0.,x-left-5.],[0.,1.,y-top-5.]])
            stamp=cv2.warpAffine(patch,M,(16,16),flags=cv2.INTER_LINEAR)
            frame[top:top+16,left:left+16]=np.maximum(frame[top:top+16,left:left+16],stamp)
        frames.append(frame)
    before=[hashlib.sha256(v.tobytes()).hexdigest() for v in frames]
    measured=f.track_window(frames,K,tracks['ordinals'],tracks['timestamps'])
    assert before==[hashlib.sha256(v.tobytes()).hexdigest() for v in frames]
    assert len(measured['seed_ids'])<=800
    np.testing.assert_array_equal(measured['fit_seed'],measured['seed_ids']%2==0)
    assert not np.any(measured['valid'][1:] & ~measured['valid'][:-1])
    accepted=measured['rejection_reason']=='accepted'
    assert np.all(measured['forward_backward_error_px'][accepted]<=.75)
    result=f.solve_window(measured,K)
    result.update(fixture='rendered_native_point_patches',seed_count=len(measured['seed_ids']),
        persistent_count=measured['persistent_count'],synthetic_frame_hashes=before,
        rejected_by_step={str(i+1):int((measured['valid'][i]&~measured['valid'][i+1]).sum()) for i in range(6)},
        all_input_frame_bytes_unchanged=True)
    return result


def render():
    K=np.array([[600.,0.,320.],[0.,600.,240.],[0.,0.,1.]])
    rotations=np.stack([yaw(i*.35) for i in range(7)])
    centres=np.array([[0.,0.,0.],[.12,0.,.06],[.3,0.,.12],[.6,0.,.2],
        [.9,0.,.28],[1.3,0.,.36],[1.8,0.,.45]])
    rng=np.random.default_rng(20261111)
    panels=[]
    for y in np.linspace(-.25,.25,6):
        for x in np.linspace(-.36,.36,9):
            z=float(rng.uniform(8.,20.)); sx,sy=z*.055,z*.05
            vertices=np.array([[x*z-sx/2,y*z-sy/2,z],[x*z+sx/2,y*z-sy/2,z],
                [x*z+sx/2,y*z+sy/2,z],[x*z-sx/2,y*z+sy/2,z]])
            texture=rng.integers(25,256,(64,64),dtype=np.uint8)
            texture=cv2.GaussianBlur(texture,(0,0),1.5)
            texture=cv2.normalize(texture,None,20,250,cv2.NORM_MINMAX).astype(np.uint8)
            panels.append((vertices,texture,z))
    frames=[]
    src=np.array([[0.,0.],[63.,0.],[63.,63.],[0.,63.]],np.float32)
    for R,C in zip(rotations,centres):
        image=np.full((480,640),10,np.uint8)
        zbuffer=np.full((480,640),np.inf)
        for vertices,texture,z in panels:
            camera=(vertices-C)@R.T; h=camera@K.T
            quad=h[:,:2]/h[:,2:]
            left=max(0,int(np.floor(quad[:,0].min())));right=min(640,int(np.ceil(quad[:,0].max()))+1)
            top=max(0,int(np.floor(quad[:,1].min())));bottom=min(480,int(np.ceil(quad[:,1].max()))+1)
            if right<=left or bottom<=top:continue
            local=(quad-np.array([left,top])).astype(np.float32)
            H=cv2.getPerspectiveTransform(src,local)
            colour=cv2.warpPerspective(texture,H,(right-left,bottom-top),flags=cv2.INTER_LINEAR)
            mask=cv2.warpPerspective(np.full((64,64),255,np.uint8),H,
                (right-left,bottom-top),flags=cv2.INTER_NEAREST)>0
            yy,xx=np.mgrid[top:bottom,left:right]
            rays=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones(xx.shape)],axis=-1)
            world_rays=rays@R
            depth=(z-C[2])/world_rays[:,:,2]
            region=zbuffer[top:bottom,left:right]
            visible=mask&(depth>0)&(depth<region)
            image[top:bottom,left:right][visible]=colour[visible]
            region[visible]=depth[visible]
        frames.append(image)
    return frames,K,rotations,centres
