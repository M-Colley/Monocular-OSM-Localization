"""Genuine public RapidOCR/TextDetector methods; fake sessions, no weights/NN."""
from copy import deepcopy
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from omegaconf import OmegaConf
import pytest
from rapidocr import RapidOCR
from rapidocr.ch_ppocr_det.main import TextDetector
from monocular_osm import native_text_detection as m, native_pixel_mask as mask

def pixels(h=64,w=96):
    y,x=np.indices((h,w));return np.stack(((x*7+y*3)%255,(x+y*5)%256,(x^y)%256),axis=-1).astype(np.uint8)
def quad(left=10,top=20,right=40,bottom=32):
    return np.array([[[left,top],[right,top],[right,bottom],[left,bottom]]],np.float32)
class NativeSession:
    def __init__(self,path):self._model_path=path
    def get_providers(self):return ['CPUExecutionProvider']
    def get_inputs(self):return [SimpleNamespace(name='image',shape=[1,3,'H','W'],type='tensor(float)')]
    def get_outputs(self):return [SimpleNamespace(name='maps',shape=[1,1,'H','W'],type='tensor(float)')]
class FakeSession:
    def __init__(self,path,error=None):self.session=NativeSession(path);self.calls=0;self.error=error
    def __call__(self,value):
        self.calls+=1;self.input_object=value;self.input_copy=value.copy();self.draw=(random.random(),np.random.random())
        if self.error:raise self.error
        self.output=np.full((1,1)+value.shape[2:],np.float32(self.draw[1]),np.float32);return self.output
class FakePostprocess:
    def __init__(self,boxes,error=None):
        self.boxes=boxes;self.error=error;self.calls=0;self.thresh=.3;self.box_thresh=.5
        self.max_candidates=1000;self.unclip_ratio=1.6;self.min_size=3;self.score_mode='fast';self.dilation_kernel=None
    def __call__(self,value,canvas):
        self.calls+=1;self.input_object=value;self.input_copy=value.copy();self.canvas=canvas
        if self.error:raise self.error
        self.returned=(self.boxes.copy(),[.01]*len(self.boxes));return self.returned

def engine(tmp,boxes=None,*,session_error=None,post_error=None,preprocess=False,padding=False):
    path=str((tmp/'never_opened_fake_model.onnx').resolve())
    cfg=OmegaConf.create({'Global':{'text_score':.99,'min_height':30,'width_height_ratio':8,'use_det':True,
        'use_cls':False,'use_rec':False,'min_side_len':30,'max_side_len':2000,'return_word_box':False,
        'return_single_char_box':False,'use_preprocess_img':preprocess,'use_vertical_padding':padding,
        'font_path':None,'model_root_dir':None},'Det':{'model_path':path}})
    e=RapidOCR.__new__(RapidOCR);e._initialize(cfg);d=TextDetector.__new__(TextDetector)
    d.limit_side_len=32;d.limit_type='max';d.mean=[.5,.5,.5];d.std=[.5,.5,.5];d.preprocess_op=None
    d.session=FakeSession(path,session_error);d.postprocess_op=FakePostprocess(quad()if boxes is None else boxes,post_error);e.text_det=d
    return e,{'schema':'externally-frozen-detector-model-binding-v1','model':{'path':path,'sha256':'a'*64,'size':123}}
def equal_rng(a,b):assert a[0]==b[0]and np.array_equal(a[1],b[1])and a[2:]==b[2:]

def test_genuine_public_values_RNG_all_quads_and_observer_restoration(tmp_path):
    boxes=np.concatenate((quad(),quad(50,28,70,36),quad()))
    baseline,_=engine(tmp_path,boxes);e,b=engine(tmp_path,boxes)
    py=random.getstate();nr=np.random.get_state();adapter=m.NativeDetectionAdapter(e,model_binding=b)
    assert random.getstate()==py;equal_rng(np.random.get_state(),nr)
    random.seed(13);np.random.seed(17);direct=baseline(pixels(),use_det=True,use_cls=False,use_rec=False)
    py=random.getstate();nr=np.random.get_state();random.seed(13);np.random.seed(17);out=adapter.detect(pixels())
    assert out['status']=='complete_native_detection_only',out
    np.testing.assert_array_equal(out['quads'],direct.boxes);assert len(out['quads'])==3
    assert e.text_cls is e.text_rec is None and random.getstate()==py;equal_rng(np.random.get_state(),nr)
    a,d=baseline.text_det,e.text_det
    np.testing.assert_array_equal(a.session.input_copy,d.session.input_copy);np.testing.assert_array_equal(a.session.output,d.session.output)
    assert d.postprocess_op.input_object is d.session.output and d.session.calls==d.postprocess_op.calls==1
    assert type(d.preprocess_op).__name__=='DetPreProcess'
    assert all(k not in vars(e)for k in('build_final_output','preprocess_img'))and 'get_preprocess'not in vars(d)
    r=out['receipt'];assert r['session']['completed']==r['postprocess']['completed']==1
    assert r['observers_restored']is r['all_returned_quads_retained']is True
    _,_,mr=mask.mask_native_bgr(pixels(),out['mask_quads']);assert mr['detector_box_count']==3

def test_completed_empty_vs_no_session(tmp_path):
    e,b=engine(tmp_path,np.array([],np.float64));a=m.NativeDetectionAdapter(e,model_binding=b)
    r=a.detect(pixels());assert r['status']=='complete_native_detection_only',r
    assert r['quads'].shape==r['mask_quads'].shape==(0,4,2)and r['quads'].dtype==np.float32
    assert r['receipt']['completed_empty']is True
    e,b=engine(tmp_path);a=m.NativeDetectionAdapter(e,model_binding=b)
    r=a.detect(pixels(1,1));assert r['status']=='unavailable'and r['quads']is None
    assert r['receipt']['session']['attempted']==0 and r['receipt']['postprocess']['completed']==0

@pytest.mark.parametrize('where',['session','post'])
def test_errors_never_become_empty_and_observers_restore(tmp_path,where):
    e,b=engine(tmp_path,**{where+'_error':RuntimeError('private '+where+' failure')})
    s,p=e.text_det.session,e.text_det.postprocess_op;a=m.NativeDetectionAdapter(e,model_binding=b)
    r=a.detect(pixels());assert r['status']=='unavailable'and r['quads']is None
    assert r['receipt']['session']['completed']==int(where=='post')and r['receipt']['postprocess']['completed']==0
    assert e.text_det.session is s and e.text_det.postprocess_op is p
    assert all(k not in vars(e)for k in ('preprocess_img','build_final_output'))

def test_BaseException_restores_then_propagates(tmp_path):
    e,b=engine(tmp_path,session_error=KeyboardInterrupt());s,p=e.text_det.session,e.text_det.postprocess_op
    a=m.NativeDetectionAdapter(e,model_binding=b)
    with pytest.raises(KeyboardInterrupt):a.detect(pixels())
    assert e.text_det.session is s and e.text_det.postprocess_op is p
    assert all(k not in vars(e)for k in ('preprocess_img','build_final_output'))
    assert a._lock.acquire(blocking=False);a._lock.release()

@pytest.mark.parametrize('kind',['global_resize','padding','right_boundary'])
def test_genuine_public_native_restoration_and_raw_boundary_preservation(tmp_path,kind):
    if kind=='global_resize':
        e,b=engine(tmp_path,quad(5,5,15,10),preprocess=True);e.max_side_len=48;e.cfg.Global.max_side_len=48
        e.text_det.limit_side_len=64
        frame=pixels();expected=quad(5,5,15,10)*np.array([1.5,2.],np.float32)
    elif kind=='padding':
        e,b=engine(tmp_path,quad(10,28,30,32),padding=True);frame=pixels(8,96);expected=quad(10,2,30,6)
    else:
        # 50x50 -> public upscale64x64. A canvas63 corner maps49.21875,
        # inside raw [0,W] but beyond last index49. Preserve it raw.
        e,b=engine(tmp_path,quad(50,40,63,50),preprocess=True);e.min_side_len=64;e.cfg.Global.min_side_len=64
        frame=pixels(50,50);expected=quad(50,40,63,50)*np.float32(50/64)
    a=m.NativeDetectionAdapter(e,model_binding=b);out=a.detect(frame)
    assert out['status']=='complete_native_detection_only',out
    np.testing.assert_array_equal(out['quads'],expected)
    assert out['receipt']['native_operation_order']==['preprocess','padding_1']
    if kind=='right_boundary':
        assert out['quads'][0,1,0]>49 and out['mask_quads'][0,1,0]==49
        assert out['receipt']['mask_operand_conversion']['changed_coordinate_count']>0

@pytest.mark.parametrize('raw',[
    np.array([[[96,20],[96,22],[95.5,22],[95.5,20]]],np.float32),
    np.array([[[10,64],[20,64],[20,63.5],[10,63.5]]],np.float64),
    np.array([[[96,64],[95.9,64],[95.9,63.9],[96,63.9]]],np.float64),quad()])
def test_mask_operands_raw_all_unchanged_complete_edge_witness(raw):
    before=raw.copy();operands,r=m.mask_operands((64,96,3),raw)
    np.testing.assert_array_equal(raw,before);assert raw.dtype==operands.dtype and raw.shape==operands.shape
    expected=raw.copy();expected[:,:,0]=np.minimum(expected[:,:,0],95);expected[:,:,1]=np.minimum(expected[:,:,1],63)
    np.testing.assert_array_equal(operands,expected)
    assert r['changed_coordinate_count']==np.count_nonzero(expected!=raw)
    assert len(r['changed_coordinates'])==r['changed_coordinate_count']and r['raw_returned_quads_preserved']is True
    mask.mask_native_bgr(pixels(),operands)

@pytest.mark.parametrize('axis,value',[(0,-.1),(1,-.1),(0,96.01),(1,64.01),(0,np.nan),(1,np.inf)])
def test_mask_operands_reject_beyond_documented_boundary(axis,value):
    q=quad();q[0,0,axis]=value
    with pytest.raises(ValueError):m.mask_operands((64,96,3),q)

def test_identity_supports_genuine_OmegaConf_Path_nodes_without_model_IO(tmp_path,monkeypatch):
    e,b=engine(tmp_path);e.cfg.Global.model_root_dir=tmp_path/'never_opened_models'
    e.cfg.Det.model_root_dir=tmp_path/'never_opened_models'
    old_stat,old_open=Path.stat,Path.open
    def stat(path,*a,**kw):
        assert path.suffix!='.onnx','model stat forbidden in metadata adapter';return old_stat(path,*a,**kw)
    def op(path,*a,**kw):
        assert path.suffix!='.onnx','model body forbidden in metadata adapter';return old_open(path,*a,**kw)
    monkeypatch.setattr(Path,'stat',stat);monkeypatch.setattr(Path,'open',op)
    a=m.NativeDetectionAdapter(e,model_binding=b);identity=a.identity
    assert identity['settings']['config']['Global']['model_root_dir']==str(tmp_path/'never_opened_models')
    identity['external_model_binding']['model']['sha256']='b'*64
    assert a.identity['external_model_binding']['model']['sha256']=='a'*64

@pytest.mark.parametrize('bad',['lazy','foreign','recognition','model_path','binding_hash','settings_changed','session_changed'])
def test_preloaded_identity_and_no_implicit_model_construction(tmp_path,bad):
    e,b=engine(tmp_path)
    if bad=='lazy':e.text_det=None
    if bad=='foreign':e.preprocess_img=lambda *x:pytest.fail('foreign pipeline called')
    if bad=='recognition':e.use_rec=True
    if bad=='model_path':e.cfg.Det.model_path=str(tmp_path/'other_unopened.onnx')
    if bad=='binding_hash':b['model']['sha256']='x'*64
    if bad not in('settings_changed','session_changed'):
        with pytest.raises(ValueError):m.NativeDetectionAdapter(e,model_binding=b)
        return
    a=m.NativeDetectionAdapter(e,model_binding=b);session=e.text_det.session
    if bad=='settings_changed':e.text_det.postprocess_op.box_thresh=.8
    else:e.text_det.session=FakeSession(b['model']['path'])
    r=a.detect(pixels());assert r['status']=='unavailable'and session.calls==0

@pytest.mark.parametrize('fault',['session_map','post_quads','native_pixels','session_input'])
def test_mutated_or_over_budget_observations_refuse_whole_frame(tmp_path,monkeypatch,fault):
    e,b=engine(tmp_path)
    if fault=='post_quads':e.text_det.postprocess_op.boxes=np.repeat(quad(),1001,axis=0)
    if fault=='native_pixels':monkeypatch.setattr(m,'MAX_PIXELS',64*96-1)
    if fault=='session_map':monkeypatch.setattr(m,'MAX_OUTPUT_VALUES',10)
    if fault=='session_input':monkeypatch.setattr(m,'MAX_INPUT_VALUES',10)
    a=m.NativeDetectionAdapter(e,model_binding=b);r=a.detect(pixels())
    assert r['status']=='unavailable'and r['quads']is None
    if fault in('session_input','native_pixels'):assert e.text_det.session.calls==0

def test_preallocation_guard_for_huge_network_resize_and_cropwork(tmp_path):
    e,b=engine(tmp_path);e.text_det.limit_type='min';e.text_det.limit_side_len=65536
    a=m.NativeDetectionAdapter(e,model_binding=b);r=a.detect(pixels())
    assert r['status']=='unavailable'and e.text_det.session.calls==0
    boxes=np.repeat(quad(0,0,191,127),1000,axis=0);e,b=engine(tmp_path,boxes)
    a=m.NativeDetectionAdapter(e,model_binding=b);r=a.detect(pixels(128,192))
    # all1000 valid boxes exceed the global crop budget, never a chosen subset.
    assert r['status']=='unavailable'and r['receipt']['postprocess']['completed']==0

def test_shared_engine_lock_refuses_second_observer_without_forward(tmp_path):
    e,b=engine(tmp_path);a=m.NativeDetectionAdapter(e,model_binding=b);other=m.NativeDetectionAdapter(e,model_binding=b)
    assert a._lock.acquire(blocking=False)
    try:r=other.detect(pixels());assert r['status']=='unavailable'and e.text_det.session.calls==0
    finally:a._lock.release()

@pytest.mark.parametrize('kind',['global','padding'])
def test_public_budget_preflight_precedes_opencv(tmp_path,monkeypatch,kind):
    import cv2
    e,b=engine(tmp_path,preprocess=kind=='global',padding=kind=='padding')
    if kind=='global':e.min_side_len=65536;e.cfg.Global.min_side_len=65536
    else:e.width_height_ratio=1e-9;e.cfg.Global.width_height_ratio=1e-9
    a=m.NativeDetectionAdapter(e,model_binding=b)
    monkeypatch.setattr(cv2,'resize',lambda *a,**kw:pytest.fail('resize before budget refusal'))
    monkeypatch.setattr(cv2,'copyMakeBorder',lambda *a,**kw:pytest.fail('padding before budget refusal'))
    r=a.detect(pixels());assert r['status']=='unavailable'and e.text_det.session.calls==0

def test_restored_crop_budget_precedes_native_resize(tmp_path,monkeypatch):
    import cv2
    e,b=engine(tmp_path,np.repeat(quad(0,0,63,31),1000,axis=0),preprocess=True)
    e.max_side_len=64;e.cfg.Global.max_side_len=64;e.text_det.limit_side_len=64
    a=m.NativeDetectionAdapter(e,model_binding=b);original=cv2.resize;calls=[]
    def resize(*args,**kw):
        calls.append(True);assert len(calls)<=2,'restored crop resize before cap'
        return original(*args,**kw)
    monkeypatch.setattr(cv2,'resize',resize)
    r=a.detect(pixels(128,192));assert r['status']=='unavailable'and 'restored crop work cap'in r['reason']
    assert r['receipt']['postprocess']['completed']==1 and not r['receipt']['public_native_output_completed']

def test_readonly_frame_and_external_abort_preserved(tmp_path):
    frame=pixels();frame.setflags(write=False);before=frame.tobytes()
    e,b=engine(tmp_path);a=m.NativeDetectionAdapter(e,model_binding=b);r=a.detect(frame)
    assert r['status']=='complete_native_detection_only'and frame.tobytes()==before and not frame.flags.writeable
    def check():raise RuntimeError('private resource refusal')
    r=a.detect(frame,check=check);assert r['status']=='unavailable'
    assert e.text_det.session is a.session and e.text_det.postprocess_op is a.postprocess
