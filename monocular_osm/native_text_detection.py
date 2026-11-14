"""Observed genuine RapidOCR detection-only calls; no detector/model construction.

Root must authenticate frozen source/config/model bindings and external resource
closure. These receipts witness program execution, never text recall or model
identity by themselves. No recognized text or new detector math is produced.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import threading
from types import MappingProxyType
import weakref
import numpy as np

MAX_PIXELS=16_000_000
MAX_BOXES=1000
MAX_INPUT_VALUES=48_000_000
MAX_OUTPUT_VALUES=16_000_000
MAX_CROP_PIXELS=16_000_000
MASK_OPERAND_PROFILE=MappingProxyType({'schema':'native-outer-boundary-mask-operands-v1',
    'raw_bounds':'0<=x<=W;0<=y<=H','mask_only':'x=min(rawx,W-1);y=min(rawy,H-1)',
    'raw_quads_changed':False,'shift_rescale_or_drop':False,'held_mask_source_sha256':'30ad5721fb0ced78f625807bfdb316c0056ec0234f5bff7021ef69d961fdae10'})
PROFILE=MappingProxyType({'method':'observed-native-RapidOCR-detection-only-v1',
    'use_det':True,'use_cls':False,'use_rec':False,'max_native_pixels':MAX_PIXELS,
    'max_quads':MAX_BOXES,'max_session_input_F32_values':MAX_INPUT_VALUES,
    'max_session_output_F32_values':MAX_OUTPUT_VALUES,'max_canvas_crop_pixels':MAX_CROP_PIXELS,
    'max_restored_crop_pixels':MAX_CROP_PIXELS,'empty':'only one completed session AND postprocess returning zero quads',
    'preallocation':'frozen public resize/padding/DetPreProcess shape preflight; reject overflow, never replace math',
    'coordinates':'raw public quads unchanged; separate declared mask-only outer-boundary endpoint operands',
    'mask_operand_policy':'native-outer-boundary-mask-operands-v1',
    'models':'preloaded detector only; explicit external model binding, no download or construction',
    'recall_or_actual_model_identity_certified':False})

def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
PROFILE_SHA256=hashlib.sha256(_json(dict(PROFILE))).hexdigest()
MASK_OPERAND_PROFILE_SHA256=hashlib.sha256(_json(dict(MASK_OPERAND_PROFILE))).hexdigest()
_locks=weakref.WeakKeyDictionary()
_locks_guard=threading.Lock()

def _need(value,message):
    if not value:raise ValueError(message)

def _array(value,limit):
    _need(isinstance(value,np.ndarray)and value.dtype.kind in 'biuf'and value.dtype.itemsize<=8
        and value.size<=limit,'bounded numeric observation required')
    _need(np.isfinite(value).all(),'nonfinite observation')
    h=hashlib.sha256(_json([list(value.shape),value.dtype.str]))
    if value.size:h.update(memoryview(np.ascontiguousarray(value)).cast('B'))
    return {'shape':list(value.shape),'dtype':value.dtype.str,'sha256':h.hexdigest()}

def _plain(value):
    from omegaconf import OmegaConf
    def plain(v,depth=0):
        _need(depth<=32,'metadata nesting cap')
        if OmegaConf.is_config(v):v=OmegaConf.to_container(v,resolve=False,enum_to_str=True)
        if isinstance(v,dict):
            _need(len(v)<=1024 and all(isinstance(k,str)for k in v),'bounded metadata dictionary')
            return {k:plain(x,depth+1)for k,x in v.items()}
        if isinstance(v,(tuple,list)):
            _need(len(v)<=4096,'metadata list cap');return [plain(x,depth+1)for x in v]
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        return v
    value=plain(value)
    data=_json(value);_need(len(data)<=1<<20 and b'${'not in data,'bounded static JSON metadata required')
    return json.loads(data)

def _source(obj):
    function=getattr(obj,'__func__',obj);cls=function if inspect.isfunction(function)else type(obj)
    path=inspect.getsourcefile(cls)
    result={'module':cls.__module__,'qualname':cls.__qualname__,'source':None}
    if path:
        p=Path(path);_need(p.stat().st_size<=1<<20,'class source cap')
        result['source']={'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    return result

def _shape32(h,w,ratio):
    _need(math.isfinite(ratio)and 0<ratio<=65536,'finite resize ratio budget')
    result=tuple(int(round(int(v*ratio)/32)*32)for v in (h,w))
    _need(min(result)>0 and result[0]*result[1]<=MAX_PIXELS,'planned resized canvas cap before allocation')
    return result

def _public_shape(engine,shape):
    h,w=shape;cfg=engine.cfg.Global
    if cfg.use_preprocess_img:
        lo,hi=engine.min_side_len,engine.max_side_len
        _need(all(type(v)in(int,float)and math.isfinite(v)and 0<v<=65536 for v in (lo,hi)),'bounded global resize settings')
        if max(h,w)>hi:h,w=_shape32(h,w,float(hi)/max(h,w))
        if min(h,w)<lo:h,w=_shape32(h,w,float(lo)/min(h,w))
    canvas=(h,w)
    if cfg.use_vertical_padding:
        ratio=engine.width_height_ratio;minimum=engine.min_height
        _need(type(ratio)in(int,float)and math.isfinite(ratio)and(ratio==-1 or ratio>0)
            and type(minimum)in(int,float)and math.isfinite(minimum)and 0<=minimum<=65536,'bounded padding settings')
        if h<=minimum or(ratio!=-1 and w/h>ratio):
            division=w/ratio;_need(math.isfinite(division)and abs(division)<=MAX_PIXELS,'padding work cap')
            amount=int(abs(max(int(division),minimum)*2-h)/2);canvas=(h+2*amount,w)
    _need(canvas[0]*canvas[1]<=MAX_PIXELS,'planned padded canvas cap before allocation')
    return (h,w),canvas

def _settings(engine):
    d=engine.text_det;p=d.postprocess_op
    return {'config':_plain(engine.cfg),'public':_plain({k:getattr(engine,k)for k in ('max_side_len','min_side_len','width_height_ratio','min_height','use_det','use_cls','use_rec')}),
        'detector':_plain({k:getattr(d,k)for k in ('limit_side_len','limit_type','mean','std')}),
        'postprocess':_plain({k:getattr(p,k)for k in ('thresh','box_thresh','max_candidates','unclip_ratio','min_size','score_mode')}),
        'dilation':None if p.dilation_kernel is None else _array(p.dilation_kernel,16)}

def runtime_metadata(engine,model_binding):
    """Source/config/runtime metadata only; NEVER opens/hashes model weights.

    Root authenticates the supplied model ref before constructing the session.
    The observed native session path and declared Det.model_path must match it.
    """
    binding=_plain(model_binding)
    _need(set(binding)=={'schema','model'}and binding['schema']=='externally-frozen-detector-model-binding-v1','external model binding grammar')
    model=binding['model'];_need(set(model)=={'path','sha256','size'}and isinstance(model['path'],str)
        and Path(model['path']).is_absolute()and type(model['size'])is int and 0<model['size']<=1<<30
        and isinstance(model['sha256'],str)and len(model['sha256'])==64
        and all(c in '0123456789abcdef'for c in model['sha256']),'positive bounded model role required')
    d=engine.text_det;session=d.session;native=getattr(session,'session',None)
    actual=getattr(native,'_model_path',None);declared=engine.cfg.Det.get('model_path')
    key=lambda p:os.path.normcase(os.path.normpath(str(p)))
    _need(actual is not None and declared is not None and key(actual)==key(declared)==key(model['path']),'explicit configured/loaded model path differs')
    versions={}
    for package in ('rapidocr','onnxruntime','numpy','opencv-python','Pillow','omegaconf','pyclipper','shapely'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    return {'schema':'observed-native-detector-runtime-metadata-v1','profile_sha256':PROFILE_SHA256,
        'mask_operand_profile_sha256':MASK_OPERAND_PROFILE_SHA256,
        'classes':{k:_source(v)for k,v in {'public_engine':engine,'detector':d,'session':session,'postprocess':d.postprocess_op,'image_loader':engine.load_img}.items()},
        'public_functions':{k:_source(v)for k,v in {**{n:getattr(engine,n)for n in ('preprocess_img','build_final_output','detect_and_crop','crop_text_regions')},
            **{n:engine.build_final_output.__func__.__globals__[n]for n in ('map_boxes_to_original','map_img_to_original')},
            **{n:engine.detect_and_crop.__func__.__globals__[n]for n in ('apply_vertical_padding',)},
            **{n:engine.preprocess_img.__func__.__globals__[n]for n in ('resize_image_within_bounds',)},
            'get_rotate_crop_image':engine.crop_text_regions.__func__.__globals__['get_rotate_crop_image'],'detector_preprocess_factory':d.get_preprocess}.items()},
        'settings':_settings(engine),'versions':versions,'external_model_binding':binding,
        'observed_session_model_path':str(actual),'providers':list(native.get_providers()),
        'session_inputs':[{'name':v.name,'shape':list(v.shape),'type':v.type}for v in native.get_inputs()],
        'session_outputs':[{'name':v.name,'shape':list(v.shape),'type':v.type}for v in native.get_outputs()],
        'model_bytes_hashed_or_loaded_by_adapter':False,'Root_source_dependency_model_authority_verified_by_adapter':False}

class _Proxy:
    def __init__(self,original,call):self.original=original;self.call=call
    def __getattr__(self,name):return getattr(self.original,name)
    def __call__(self,*args,**kwargs):return self.call(self.original,*args,**kwargs)

class NativeDetectionAdapter:
    def __init__(self,engine,*,model_binding):
        from rapidocr import RapidOCR
        from rapidocr.ch_ppocr_det.main import TextDetector
        from rapidocr.utils.load_image import LoadImage
        _need(type(engine)is RapidOCR and type(engine.text_det)is TextDetector,'preloaded genuine RapidOCR/TextDetector required; no lazy model load')
        _need(type(engine.load_img)is LoadImage,'genuine ndarray image loader required')
        for name in ('preprocess_img','build_final_output','detect_and_crop','run_ocr_steps','crop_text_regions','update_params','_load_det_model'):
            _need(getattr(getattr(engine,name),'__func__',None)is getattr(RapidOCR,name),'foreign public pipeline override')
        _need(engine.use_det is True and engine.use_cls is False and engine.use_rec is False,'preconfigured detection-only engine required')
        _need(getattr(engine.text_det.get_preprocess,'__func__',None)is TextDetector.get_preprocess
            and engine.text_det.sorted_boxes is TextDetector.sorted_boxes,'foreign detector preprocessing/sort override')
        self.engine=engine;self.detector=engine.text_det;self.session=self.detector.session;self.postprocess=self.detector.postprocess_op
        self._identity=_json(runtime_metadata(engine,model_binding));self._settings=_json(_settings(engine))
        with _locks_guard:self._lock=_locks.setdefault(engine,threading.Lock())
    @property
    def identity(self):return json.loads(self._identity)

    def detect(self,native_bgr,*,check=lambda:None):
        """One full-frame public call, all native quads or explicit unavailable.

        Caller owns a blocking-session deadline and authenticates identity.
        Original frame/callable objects are preserved. BaseException propagates
        after observer restoration; ordinary failures retain partial witnesses.
        """
        trace={'schema':'observed-native-detection-only-receipt-v1','profile_sha256':PROFILE_SHA256,
            'runtime_identity':self.identity,'call':{'use_det':True,'use_cls':False,'use_rec':False},
            'session':{'attempted':0,'completed':0},'postprocess':{'attempted':0,'completed':0},
            'public_native_output_completed':False,'detector_recall_certified':False,
            'Root_source_dependency_model_authority_verified_by_adapter':False,'blocking_deadline_requires_external_guard':True}
        if not self._lock.acquire(blocking=False):return {'status':'unavailable','quads':None,'receipt':trace,'reason':'detector already in use'}
        saved=[]
        def install(obj,name,value):
            exists=name in vars(obj);old=vars(obj).get(name);saved.append((obj,name,exists,old));setattr(obj,name,value)
        def restore():
            for obj,name,exists,old in reversed(saved):
                if exists:setattr(obj,name,old)
                else:delattr(obj,name)
            saved.clear()
            if isinstance(self.detector.preprocess_op,_Proxy):self.detector.preprocess_op=self.detector.preprocess_op.original
        try:
            _need(isinstance(native_bgr,np.ndarray)and native_bgr.dtype==np.uint8 and native_bgr.ndim==3
                and native_bgr.shape[2]==3 and native_bgr.flags.c_contiguous and 0<native_bgr.shape[0]*native_bgr.shape[1]<=MAX_PIXELS,'native uint8 contiguous HxWx3 BGR cap')
            native_header=_array(native_bgr,MAX_PIXELS*3);trace['native_input']=native_header
            _need(self.engine.text_det is self.detector and self.detector.session is self.session
                and self.detector.postprocess_op is self.postprocess and _json(_settings(self.engine))==self._settings,'frozen detector settings or objects changed')
            frame=native_bgr.copy(order='C');observed={};check()
            original_preprocess=self.engine.preprocess_img;original_get=self.detector.get_preprocess
            def public_preprocess(ori):
                _need(_array(ori,MAX_PIXELS*3)==native_header,'public loader changed native pixels')
                shape,canvas=_public_shape(self.engine,ori.shape[:2]);observed['canvas']=canvas
                trace['planned_public_shape']=list(shape);trace['planned_detector_canvas']=list(canvas)
                check();result=original_preprocess(ori)
                _need(isinstance(result,tuple)and len(result)==2 and result[0].shape==shape+(3,),'genuine public preprocessing shape differs')
                return result
            def get_preprocess():
                original=original_get()
                def call(preprocess,value):
                    _need(value.dtype==np.uint8 and value.shape==observed['canvas']+(3,),'actual padded canvas differs')
                    trace['canvas_BGR']=_array(value,MAX_PIXELS*3)
                    side=self.detector.limit_side_len;kind=self.detector.limit_type
                    _need(type(side)in(int,float)and math.isfinite(side)and 0<side<=65536 and kind in {'min','max'},'bounded detector resize settings')
                    h,w=value.shape[:2];ratio=1.
                    if(kind=='max'and max(h,w)>side):ratio=float(side)/max(h,w)
                    if(kind=='min'and min(h,w)<side):ratio=float(side)/min(h,w)
                    # A genuine zero-dimension preprocessor skip stays no-forward.
                    expected=tuple(int(round(int(v*ratio)/32)*32)for v in(h,w))
                    _need(min(expected)>=0 and expected[0]*expected[1]<=MAX_PIXELS,'network input cap before preprocess allocation')
                    check();result=preprocess(value)
                    if result is not None:
                        _need(result.shape==(1,3)+expected,'genuine detector preprocessing shape differs')
                        trace['preprocessed_input']=_array(result,MAX_INPUT_VALUES);observed['session_input']=result
                    else:trace['preprocess_returned_None']=True
                    return result
                return _Proxy(original,call)
            def session_call(original,value):
                row=trace['session'];row['attempted']+=1;_need(row['attempted']==1,'exactly one detector session required')
                _need(isinstance(value,np.ndarray)and value.dtype==np.float32 and value.ndim==4
                    and value.shape[:2]==(1,3)and min(value.shape[2:])>0,'genuine detector NCHW Float32 input required')
                _need(value is observed.get('session_input'),'session must consume genuine preprocessor output')
                row['input']=_array(value,MAX_INPUT_VALUES);check();result=original(value)
                _need(_array(value,MAX_INPUT_VALUES)==row['input'],'session changed its input values')
                _need(isinstance(result,np.ndarray)and result.dtype==np.float32 and result.ndim==4
                    and result.shape[:2]==(1,1)and min(result.shape[2:])>0,'genuine detector Float32 map required')
                row['output']=_array(result,MAX_OUTPUT_VALUES);row['completed']=1;observed['predictions']=result
                return result
            def post_call(original,value,canvas):
                row=trace['postprocess'];row['attempted']+=1
                _need(row['attempted']==1 and trace['session']['completed']==1 and value is observed['predictions']
                    and _array(value,MAX_OUTPUT_VALUES)==trace['session']['output'],'postprocess must consume actual session output')
                _need(isinstance(canvas,(tuple,list))and len(canvas)==2 and all(type(v)is int and v>0 for v in canvas)
                    and canvas[0]*canvas[1]<=MAX_PIXELS,'detector canvas cap')
                row['canvas_shape']=list(canvas);check();result=original(value,canvas)
                _need(isinstance(result,tuple)and len(result)==2,'genuine postprocess pair required')
                boxes,scores=result;count=_quad_count(boxes);_need(isinstance(scores,(list,tuple,np.ndarray))and len(scores)==count,'all postprocess scores required')
                row['raw_quads']=_array(boxes,MAX_BOXES*8);row['quad_count']=count
                _need(all(isinstance(v,(int,float,np.integer,np.floating))and not isinstance(v,(bool,np.bool_))and math.isfinite(v)for v in scores),'finite postprocess scores')
                row['scores']=_array(np.asarray(scores,np.float64),MAX_BOXES)
                total=0
                for box in boxes if count else ():
                    _need(np.all(box[:,0]>=0)and np.all(box[:,0]<=canvas[1]-1)and np.all(box[:,1]>=0)and np.all(box[:,1]<=canvas[0]-1),'postprocess quads outside detector canvas')
                    width=int(max(np.linalg.norm(box[0]-box[1]),np.linalg.norm(box[2]-box[3])))
                    height=int(max(np.linalg.norm(box[0]-box[3]),np.linalg.norm(box[1]-box[2])))
                    _need(width>0 and height>0,'degenerate detector crop');total+=width*height
                    _need(total<=MAX_CROP_PIXELS,'all detector crop work cap; no subset')
                row['canvas_crop_pixels']=total;row['completed']=1
                _need(_array(value,MAX_OUTPUT_VALUES)==trace['session']['output'],'postprocess changed session map values')
                return result
            original_final=self.engine.build_final_output
            def final_call(ori,det,cls,rec,crops,ops):
                _need(_array(ori,MAX_PIXELS*3)==native_header and trace['postprocess']['completed']==1,'native restoration input or postprocess witness missing')
                _need(set(ops)=={'preprocess','padding_1'}and set(ops['preprocess'])=={'ratio_h','ratio_w'}
                    and set(ops['padding_1'])=={'top','left'},'genuine native restoration operations required')
                rh,rw=(ops['preprocess'][k]for k in ('ratio_h','ratio_w'))
                _need(all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in (rh,rw)),'finite native restoration ratios')
                _need(isinstance(crops,list)and len(crops)==trace['postprocess']['quad_count'],'all detector crops retained')
                total=0
                for crop in crops:
                    _need(isinstance(crop,np.ndarray)and crop.ndim==3 and crop.shape[2]==3,'genuine RGB/BGR crop shape')
                    h,w=round(crop.shape[0]*rh),round(crop.shape[1]*rw);_need(h>0 and w>0,'positive restored crop')
                    total+=h*w;_need(total<=MAX_CROP_PIXELS,'all restored crop work cap; no subset')
                trace['native_coordinate_operations']=_plain(ops);trace['native_operation_order']=list(ops)
                trace['restored_crop_pixels']=total;check();result=original_final(ori,det,cls,rec,crops,ops)
                trace['public_native_output_completed']=True;return result
            install(self.detector,'session',_Proxy(self.session,session_call))
            install(self.detector,'postprocess_op',_Proxy(self.postprocess,post_call))
            install(self.detector,'get_preprocess',get_preprocess)
            install(self.engine,'preprocess_img',public_preprocess)
            install(self.engine,'build_final_output',final_call)
            try:result=self.engine(frame,use_det=True,use_cls=False,use_rec=False)
            finally:restore()
            check();_need(_array(native_bgr,MAX_PIXELS*3)==native_header and _array(frame,MAX_PIXELS*3)==native_header,'native pixels changed during detector call')
            _need(self.engine.text_det is self.detector and self.detector.session is self.session and self.detector.postprocess_op is self.postprocess
                and _json(_settings(self.engine))==self._settings,'detector configuration changed during call')
            _need(self.engine.use_det is True and self.engine.use_cls is False and self.engine.use_rec is False,'recognition/classification unexpectedly enabled')
            _need(trace['session']['completed']==trace['postprocess']['completed']==1 and trace['public_native_output_completed'],'no completed session/postprocess/native return')
            count=trace['postprocess']['quad_count'];quads=getattr(result,'boxes',None)
            if count==0:
                _need(quads is None or(isinstance(quads,np.ndarray)and quads.size==0),'empty postprocess/native count differs')
                quads=np.empty((0,4,2),np.float32)
            else:
                _need(_quad_count(quads)==count,'native output silently dropped detections')
                quads=quads.copy(order='C')
            trace['native_quads']=_array(quads,MAX_BOXES*8);trace['all_returned_quads_retained']=True
            mask_quads,conversion=mask_operands(native_bgr.shape,quads)
            trace['mask_quads']=_array(mask_quads,MAX_BOXES*8);trace['mask_operand_conversion']=conversion
            trace['completed_empty']=count==0;trace['observers_restored']=True
            return {'status':'complete_native_detection_only','quads':quads,'mask_quads':mask_quads,'receipt':trace}
        except Exception as error:
            restore();trace['observers_restored']=True
            return {'status':'unavailable','quads':None,'receipt':trace,'reason':type(error).__name__+': '+str(error)[:512]}
        finally:
            restore();self._lock.release()

def _quad_count(value):
    _need(isinstance(value,np.ndarray)and value.dtype.kind in 'iuf'and value.dtype.itemsize<=8
        and value.ndim>=1 and len(value)<=MAX_BOXES,'bounded complete quadrilateral array required')
    _need((len(value)==0 and value.size==0)or(value.ndim==3 and value.shape[1:]==(4,2)),'all quadrilateral coordinates required')
    _array(value,MAX_BOXES*8);return len(value)

def mask_operands(native_shape,raw_quads):
    """Root-approved mask-only endpoint policy; raw detector output stays exact."""
    _need(isinstance(native_shape,(tuple,list))and len(native_shape)==3 and native_shape[2]==3
        and all(type(v)is int and v>0 for v in native_shape)and native_shape[0]*native_shape[1]<=MAX_PIXELS,'bounded native shape')
    count=_quad_count(raw_quads);_need(raw_quads.ndim==3 and raw_quads.shape[1:]==(4,2),'explicit complete native quad shape')
    h,w=native_shape[:2]
    _need(np.all(raw_quads[:,:,0]>=0)and np.all(raw_quads[:,:,0]<=w)
        and np.all(raw_quads[:,:,1]>=0)and np.all(raw_quads[:,:,1]<=h),'raw quads beyond documented native outer boundary')
    result=raw_quads.copy(order='C')
    result[:,:,0]=np.minimum(result[:,:,0],w-1);result[:,:,1]=np.minimum(result[:,:,1],h-1)
    changed=np.argwhere(result!=raw_quads)
    receipt={'schema':'native-mask-operand-conversion-receipt-v1','profile_sha256':MASK_OPERAND_PROFILE_SHA256,
        'profile':dict(MASK_OPERAND_PROFILE),'native_shape':list(native_shape),'quad_count':count,
        'raw_quads':_array(raw_quads,MAX_BOXES*8),'mask_quads':_array(result,MAX_BOXES*8),
        'changed_coordinates':[[int(i),int(v),int(axis),float(raw_quads[i,v,axis]),float(result[i,v,axis])]for i,v,axis in changed],
        'changed_coordinate_count':len(changed),'raw_returned_quads_preserved':True,'order_count_dtype_preserved':True}
    return result,receipt
