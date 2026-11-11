"""Bounded exact graph-artifact semantic sidecar; no motion/selection model.

Raw row order is the genuine loaded graph directed-edge enumeration. Values
describe that artifact, not unconverted OSM, legal applicability or way order.
Unknown/ambiguous semantics is neutral; geometry and road states never change.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
import numpy as np

TOKEN = "_source_keyed_path_v1"
MAX_RAW = 500_000
MAX_COMPACT = 100_000
MAX_SEGMENTS = 500_000
MAX_MEMBERS = 1_000_000
MAX_CATALOG_BYTES = 192 << 20
MAX_RECORD_BYTES = 8192
MAX_DEPTH = 4
MAX_ARRAY_BYTES = 64 << 20
SEG_STATUS = MappingProxyType({"exact_source_context":1,"ambiguous_parallel_neutral":2,
    "tiny_replacement_neutral":3,"synthetic_neutral":4,"raw_geometry_neutral":5})
RAW_STATUS = MappingProxyType({"selected_path":1,"unchanged_singleton":2,
    "omitted_parallel":4,"zero_chord":8,"tiny_chord":16,"raw_geometry":32,
    "selected_ambiguous_parallel":64})
ARRAY_NAMES = ("raw_edge_keys","compact_edge_keys","compact_path_indptr",
    "compact_path_raw_edge_index","segment_raw_edge_index","segment_semantic_status","raw_edge_disposition")
PROFILE = MappingProxyType({"method":"original-keyed-segment-semantics-v1",
    "raw_order":"genuine loaded directed graph edge enumeration; explicit keys, not sorted/XML order",
    "raw_edges":MAX_RAW,"raw_nodes":MAX_RAW,"compact_edges":MAX_COMPACT,
    "segments":MAX_SEGMENTS,"path_memberships":MAX_MEMBERS,
    "catalog_bytes":MAX_CATALOG_BYTES,"record_bytes":MAX_RECORD_BYTES,"raw_value_depth":MAX_DEPTH,
    "semantic_array_bytes":MAX_ARRAY_BYTES,"token":TOKEN,
    "catalog_record":"raw-directed-edge-catalog-record-v1; typed envelopes; streaming canonical JSONL including newline",
    "raw_geometry_vertex_cap":128,
    "geometry_value_encoding":"exact loaded LineString coordinates and WKB hex; unsupported geometry types refuse",
    "raw_disposition":"uint8 bitflags selected1,unchanged2,omittedparallel4,zero8,tiny16,rawgeometry32,ambiguous64",
    "segment_status":"1 exact source context; 2 parallel ambiguity; 3 tiny replacement; 4 synthetic; 5 rawgeometry unsupported",
    "parser":"positive scalar numeric base km/h or mph FULLMATCH; unsupported overrides neutral",
    "source_way_orientation":"unknown, including oneway=-1/reversed=False",
    "original_OSM_coverage":"unknown; graph-artifact fields only",
    "mapping":"exact keyed path + genuine construction trace; no nearest, averaging or new nodes",
    "lookup":"genuine global searchsorted(side=right), then clip within original edge; indices only, numeric context needs catalog parse",
    "motion_model_or_policy_changed":False,"speed_imputation":False,"legal_speed_observation":False})
PROFILE_SHA256=hashlib.sha256(json.dumps(dict(PROFILE),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def need(value,message):
    if not value:raise ValueError(message)

def _key(value):
    need(isinstance(value,(tuple,list)) and len(value)==3,"exact directed key triple required")
    need(all(isinstance(x,(int,np.integer)) and not isinstance(x,(bool,np.bool_))
             and -(1<<63)<=int(x)<(1<<63) for x in value),"signed int64 key required")
    return tuple(int(x) for x in value)

def concat_paths(values):
    out=[]
    for path in values:
        need(isinstance(path,(tuple,list)) and len(path)>0,"ordered nonempty keyed path required")
        need(len(out)+len(path)<=MAX_MEMBERS,"path membership cap exceeded")
        out.extend(_key(k) for k in path)
    return tuple(out)

def record_bytes(record):
    b=json.dumps(record,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode('utf-8')+b'\n'
    need(len(b)<=MAX_RECORD_BYTES,"catalog record byte cap exceeded")
    return b

_SPEED=re.compile(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:\s*(km/h|mph))?",re.I)
def parse_maxspeed(attributes):
    result={"status":"neutral","mps":None,"legal_applicability_verified":False,
            "source_way_orientation":"unknown","original_OSM_coverage":"unknown"}
    if any(k.startswith('maxspeed:') for k in attributes):return {**result,"reason":"unsupported_override"}
    value=attributes.get('maxspeed')
    if value is None:return {**result,"reason":"missing_base"}
    unit='km/h'
    if isinstance(value,str):
        text=value.strip();match=_SPEED.fullmatch(text)
        if match is None:return {**result,"reason":"unsupported_base"}
        unit=(match.group(1) or 'km/h').lower()
        number=float(text[:match.start(1)].strip()) if match.group(1) else float(text)
    elif isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_)):
        try:number=float(value)
        except (OverflowError,ValueError):return {**result,"reason":"nonpositive_or_nonfinite"}
    else:return {**result,"reason":"unsupported_base"}
    if not math.isfinite(number) or number<=0:return {**result,"reason":"nonpositive_or_nonfinite"}
    mps=number*.44704 if unit=='mph' else number/3.6
    if not math.isfinite(mps) or mps<=0:return {**result,"reason":"nonpositive_or_nonfinite"}
    return {**result,"status":"base_numeric_context","mps":mps,"reason":None,"unit":unit}

def _value(value,budget,depth=0):
    need(depth<=MAX_DEPTH,"tag value depth cap exceeded")
    kind=type(value).__module__+'.'+type(value).__qualname__
    budget[0]-=32+len(kind)
    need(budget[0]>=0,"catalog value budget exceeded")
    if value is None or isinstance(value,(bool,np.bool_)):
        body=None if value is None else bool(value)
    elif isinstance(value,(int,np.integer)):
        need(int(value).bit_length()<=8192,"catalog integer cap exceeded")
        body=int(value);budget[0]-=len(str(body))
    elif isinstance(value,(float,np.floating)):
        if isinstance(value,np.floating):need(value.dtype.itemsize<=8,"unsupported extended precision graph float")
        body=float(value);need(math.isfinite(body),"nonfinite graph attribute")
    elif isinstance(value,str):
        need(len(value)<=MAX_RECORD_BYTES,"catalog string cap exceeded")
        budget[0]-=len(json.dumps(value,ensure_ascii=False).encode('utf-8'));body=value
    elif isinstance(value,(list,tuple)):
        need(len(value)<=MAX_RECORD_BYTES//32,"catalog container cap exceeded")
        body=[_value(v,budget,depth+1) for v in value]
    elif isinstance(value,dict):
        need(len(value)<=MAX_RECORD_BYTES//32 and all(type(k)is str for k in value),"bounded string attribute keys required")
        body={}
        for k,v in value.items():
            need(len(k)<=MAX_RECORD_BYTES,"attribute key cap exceeded")
            budget[0]-=len(k.encode('utf-8'));body[k]=_value(v,budget,depth+1)
    elif getattr(value,'geom_type',None)=='LineString':
        sequence=value.coords
        need(len(sequence)<=128,"raw geometry vertex cap exceeded before copy")
        coords=np.asarray(sequence,np.float64)
        need(coords.ndim==2 and len(coords)<=128 and np.isfinite(coords).all(),"bounded finite raw geometry required")
        binary=value.wkb_hex
        need(len(binary)<=MAX_RECORD_BYTES,"raw geometry binary cap exceeded")
        budget[0]-=coords.size*24+len(binary)
        body={'coordinates':coords.tolist(),'wkb_hex':binary}
    else:raise ValueError("unsupported loaded graph attribute type: "+kind)
    need(budget[0]>=0,"catalog value budget exceeded")
    return {'type':kind,'value':body}

@dataclass
class Catalog:
    keys: np.ndarray
    endpoint_xy: np.ndarray
    geometry_present: np.ndarray
    index: dict
    crs: str
    encoded_sha256: str = ''
    encoded_bytes: int = 0

def _record(graph,catalog,i):
    key=tuple(int(x) for x in catalog.keys[i]);data=graph.edges[key]
    need(TOKEN not in data,"private keyed token namespace collision")
    budget=[MAX_RECORD_BYTES-512]
    need(len(data)<=MAX_RECORD_BYTES//32 and all(type(k)is str for k in data),"bounded string attribute keys required")
    attrs={}
    for name,value in data.items():
        need(len(name)<=MAX_RECORD_BYTES,"attribute name cap exceeded")
        budget[0]-=len(json.dumps(name,ensure_ascii=False).encode('utf-8'))
        attrs[name]=_value(value,budget)
    return {'schema':'raw-directed-edge-catalog-record-v1','index':i,'key':list(key),
        'endpoint_xy':catalog.endpoint_xy[i].tolist(),'attributes':attrs}

def attach(graph,*,emit_record):
    """Stream exact catalog before collision-refused tokens modify owned graph."""
    need(callable(emit_record) and graph.is_directed() and graph.is_multigraph(),"directed multigraph/emitter required")
    n=graph.number_of_edges()
    need(0<n<=MAX_RAW and 0<graph.number_of_nodes()<=MAX_RAW,"raw graph cap exceeded")
    keys=np.empty((n,3),np.int64);xy=np.empty((n,2,2),np.float64);geometry=np.empty(n,np.bool_);index={}
    for i,(u,v,k,data) in enumerate(graph.edges(keys=True,data=True)):
        key=_key((u,v,k));need(key not in index and TOKEN not in data,"raw key/token collision")
        keys[i]=key;index[key]=i
        xy[i]=[[float(graph.nodes[j]['x']),float(graph.nodes[j]['y'])] for j in (u,v)]
        need(np.isfinite(xy[i]).all(),"finite raw endpoints required")
        geometry[i]=data.get('geometry') is not None
    catalog=Catalog(keys,xy,geometry,index,str(graph.graph.get('crs','')))
    need(bool(catalog.crs),"raw graph CRS required")
    h=hashlib.sha256();total=0
    for i in range(n):
        row=_record(graph,catalog,i);b=record_bytes(row);total+=len(b)
        need(total<=MAX_CATALOG_BYTES,"catalog aggregate byte cap exceeded")
        emit_record(row);need(record_bytes(row)==b,"catalog callback mutated record")
        h.update(b)
    # No callbacks after this complete graph snapshot recheck and token attach.
    need(graph.number_of_edges()==n,"raw membership drift")
    verify=hashlib.sha256()
    for i,(u,v,k) in enumerate(graph.edges(keys=True)):
        need(_key((u,v,k))==tuple(keys[i]),"raw enumeration drift")
        current=np.array([[float(graph.nodes[j]['x']),float(graph.nodes[j]['y'])] for j in (u,v)])
        need(np.array_equal(current,xy[i]),"raw endpoint drift")
        verify.update(record_bytes(_record(graph,catalog,i)))
    need(verify.digest()==h.digest(),"raw attributes drift during catalog publication")
    catalog.encoded_sha256=h.hexdigest();catalog.encoded_bytes=total
    for key in index:graph.edges[key][TOKEN]=(key,)
    return catalog

NET_ARRAYS=('origin','edge_len','edge_base','seg_lo','seg_hi','seg_gstart','seg_xy0','seg_dir',
    'seg_heading','edge_head_start','edge_head_end','edge_mid','reverse_of','succ_table',
    'succ_logprior','succ_head','xy_max','edge_keys')
def typed_sha(value):
    need(isinstance(value,np.ndarray) and value.flags.c_contiguous and not value.dtype.hasobject,"plain contiguous typed array required")
    h=hashlib.sha256(json.dumps([list(value.shape),value.dtype.str],separators=(',',':')).encode())
    h.update(memoryview(value).cast('B'));return h.hexdigest()

def net_binding(net):
    arrays={n:typed_sha(np.asarray(net.edge_keys,dtype=np.int64).reshape(-1,3) if n=='edge_keys' else getattr(net,n)) for n in NET_ARRAYS}
    return {'schema':'compact-network-semantic-binding-v1','crs':str(net.crs),'allow_uturn':bool(net.allow_uturn),
        'turn_prior_kappa':float(net.turn_prior_kappa),'normalise_turn_prior':bool(net.normalise_turn_prior),'arrays':arrays}

def _trace_geometry(row,net,i):
    need(row.get('schema')=='road-network-construction-trace-v1' and type(row.get('edge_index'))is int
         and row['edge_index']==i and _key(row['edge_key'])==_key(net.edge_keys[i]),"construction trace identity mismatch")
    for name in ('raw_vertices','oriented_vertices','compact_vertices'):
        value=row[name]
        need(isinstance(value,np.ndarray) and value.dtype==np.float64 and value.ndim==2 and value.shape[1]==2
             and len(value)<=MAX_MEMBERS+1 and np.isfinite(value).all(),"bounded finite construction vertices required")
    raw,oriented,compact=(row[n] for n in ('raw_vertices','oriented_vertices','compact_vertices'))
    for name in ('geometry_present','geometry_reversed','geometry_fallback','synthetic'):
        need(type(row[name])is bool,"typed construction flag required")
    need(len(oriented)>=2,"oriented endpoint fallback required")
    if not row['geometry_fallback']:
        need(np.array_equal(oriented,raw[::-1] if row['geometry_reversed'] else raw),"geometry orientation trace mismatch")
    else:need(len(raw)<2 and len(oriented)==2 and not row['geometry_reversed'],"invalid geometry fallback trace")
    expected=np.r_[0,np.flatnonzero(np.hypot(*np.diff(oriented,axis=0).T)>1e-6)+1].astype(np.int64)
    keep=row['keep_indices']
    need(isinstance(keep,np.ndarray) and keep.dtype==np.int64 and np.array_equal(keep,expected),"construction keep mask mismatch")
    clean=oriented[keep]
    synthetic=len(clean)<2
    if synthetic:clean=np.vstack([clean[:1],clean[:1]+[.05,0.]])
    need(row['synthetic']is synthetic and np.array_equal(compact,clean),"clean/fallback geometry trace mismatch")
    lo,hi=int(net.seg_lo[i]),int(net.seg_hi[i])
    need(hi-lo==len(compact)-1 and 0<=lo<hi<=len(net.seg_gstart),"compact segment membership mismatch")
    xy0=compact[:-1]-net.origin;xy1=compact[1:]-net.origin;d=xy1-xy0;length=np.hypot(d[:,0],d[:,1])
    need(np.all(length>0) and np.array_equal(net.seg_xy0[lo:hi],xy0)
         and np.array_equal(net.seg_dir[lo:hi],d/length[:,None])
         and np.array_equal(net.seg_heading[lo:hi],np.arctan2(d[:,1],d[:,0])),"compact geometry differs from genuine trace")
    return lo,hi,keep

def bind_mapping(catalog,trace,net):
    """Bind seven arrays to exact raw keys, keyed paths and construction trace."""
    r=len(catalog.keys);e=int(net.n_edges);s=len(net.seg_gstart)
    need(0<r<=MAX_RAW and 0<e<=MAX_COMPACT and 0<s<=MAX_SEGMENTS
         and isinstance(trace,list) and len(trace)==e,"semantic mapping count cap/membership mismatch")
    need(catalog.keys.shape==(r,3) and catalog.keys.dtype==np.int64 and catalog.endpoint_xy.shape==(r,2,2)
         and len(catalog.index)==r and len(catalog.encoded_sha256)==64,"complete catalog index required")
    memberships=0
    for row in trace:
        path=row['source_keyed_path'];need(isinstance(path,(tuple,list)) and len(path)>0,"explicit keyed source path required")
        memberships+=len(path);need(memberships<=MAX_MEMBERS,"path membership cap exceeded")
    payload=r*24+e*24+(e+1)*8+memberships*8+s*8+s+r
    need(payload<=MAX_ARRAY_BYTES,"semantic typed array budget exceeded before allocation")
    keys=np.asarray([_key(k) for k in net.edge_keys],dtype=np.int64)
    indptr=np.empty(e+1,np.int64);indptr[0]=0;members=np.empty(memberships,np.int64)
    disposition=np.zeros(r,np.uint8);selected=np.zeros(r,np.bool_)
    for i,row in enumerate(trace):
        path=tuple(_key(k) for k in row['source_keyed_path'])
        need(path[0][0]==int(keys[i,0]) and path[-1][1]==int(keys[i,1])
             and all(a[1]==b[0] for a,b in zip(path,path[1:])),"nonjoining source keyed path")
        for j,key in enumerate(path):
            need(key in catalog.index,"invented source keyed edge")
            idx=catalog.index[key];need(not selected[idx],"source keyed edge selected more than once")
            members[indptr[i]+j]=idx;selected[idx]=True;disposition[idx]=RAW_STATUS['selected_path']
            if len(path)==1 and key==tuple(keys[i]):disposition[idx]|=RAW_STATUS['unchanged_singleton']
        indptr[i+1]=indptr[i]+len(path)
    selected_uv={tuple(int(x) for x in catalog.keys[i,:2]) for i in np.flatnonzero(selected)}
    ambiguous=set()
    for i in np.flatnonzero(~selected):
        uv=tuple(int(x) for x in catalog.keys[i,:2])
        need(uv in selected_uv,"raw edge omitted without selected parallel witness")
        disposition[i]=RAW_STATUS['omitted_parallel'];ambiguous.add(uv)
    for i in np.flatnonzero(selected):
        if tuple(int(x) for x in catalog.keys[i,:2]) in ambiguous:
            disposition[i]|=RAW_STATUS['selected_ambiguous_parallel']
    from pyproj import CRS,Transformer
    same_crs=CRS.from_user_input(catalog.crs)==CRS.from_user_input(net.crs)
    transform=None if same_crs else Transformer.from_crs(catalog.crs,net.crs,always_xy=True)
    projected=np.empty_like(catalog.endpoint_xy);raw_length=np.empty(r,np.float64)
    for start in range(0,r,4096):
        stop=min(r,start+4096);source=catalog.endpoint_xy[start:stop]
        if transform is None:projected[start:stop]=source
        else:
            x,y=transform.transform(source[:,:,0],source[:,:,1])
            projected[start:stop,:,0]=x;projected[start:stop,:,1]=y
        need(np.isfinite(projected[start:stop]).all(),"raw endpoint projection unavailable")
        d=projected[start:stop,1]-projected[start:stop,0]
        raw_length[start:stop]=np.hypot(d[:,0],d[:,1])
    nong=~catalog.geometry_present
    disposition[nong & (raw_length==0)]|=RAW_STATUS['zero_chord']
    disposition[nong & (raw_length>0) & (raw_length<=1e-6)]|=RAW_STATUS['tiny_chord']
    disposition[catalog.geometry_present]|=RAW_STATUS['raw_geometry']
    segment_raw=np.full(s,-1,np.int64);status=np.zeros(s,np.uint8)
    for i,row in enumerate(trace):
        lo,hi,keep=_trace_geometry(row,net,i)
        ids=members[indptr[i]:indptr[i+1]]
        pairs=projected[ids]
        need(len(ids)==1 or np.array_equal(pairs[:-1,1],pairs[1:,0]),"raw endpoint chain coordinates disagree")
        raw_geometry=bool(np.any(catalog.geometry_present[ids]))
        if not raw_geometry:
            chain=np.vstack([pairs[0,0],pairs[:,1]])
            need(np.array_equal(chain,row['oriented_vertices']),"source path and projected construction coordinates differ")
        if row['synthetic']:
            status[lo:hi]=SEG_STATUS['synthetic_neutral'];continue
        if raw_geometry:
            status[lo:hi]=SEG_STATUS['raw_geometry_neutral'];continue
        if any(tuple(int(x) for x in catalog.keys[j,:2]) in ambiguous for j in ids):
            status[lo:hi]=SEG_STATUS['ambiguous_parallel_neutral'];continue
        for j,(a,b) in enumerate(zip(keep,keep[1:])):
            if np.any(raw_length[ids[a:b-1]]>0):
                status[lo+j]=SEG_STATUS['tiny_replacement_neutral']
            else:
                segment_raw[lo+j]=ids[b-1];status[lo+j]=SEG_STATUS['exact_source_context']
    need(np.all(status>0) and np.all(disposition>0),"incomplete raw/segment disposition")
    arrays=dict(zip(ARRAY_NAMES,(catalog.keys,keys,indptr,members,segment_raw,status,disposition)))
    need(sum(v.nbytes for v in arrays.values())==payload,"semantic payload accounting mismatch")
    metadata={'schema':'original-keyed-segment-semantics-v1','status':'complete_semantic_sidecar_no_motion_model',
        'profile_sha256':PROFILE_SHA256,'profile':dict(PROFILE),'net_binding':net_binding(net),
        'raw_edge_count':r,'compact_edge_count':e,'segment_count':s,'path_memberships':memberships,
        'catalog_sha256':catalog.encoded_sha256,'catalog_bytes':catalog.encoded_bytes,
        'raw_order':PROFILE['raw_order'],'semantic_typed_array_bytes':payload,
        'array_headers':{n:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':typed_sha(v)} for n,v in arrays.items()},
        'segment_status_enum':dict(SEG_STATUS),'raw_disposition_flags':dict(RAW_STATUS),
        'raw_disposition_is_bitmask':True,'coverage':'loaded_graph_artifact_only; original_OSM_unknown',
        'source_way_orientation':'unknown','motion_model_implemented':False,'policy_admission':False}
    return {'arrays':arrays,'metadata':metadata}

def lookup_previous(net,packet,edge,offset):
    """Exact start-segment indices; binding rechecked, no semantic/model choice.

    O(network+sidecar+P log S), never P*map. A future accepted immutable
    consumer may separately validate once, outside this sidecar prerequisite.
    """
    meta=packet['metadata'];arrays=packet['arrays']
    need(meta['profile_sha256']==PROFILE_SHA256 and meta['net_binding']==net_binding(net),"sidecar network binding mismatch")
    need(set(arrays)==set(ARRAY_NAMES),"exact semantic array membership required")
    for n,v in arrays.items():
        h=meta['array_headers'][n]
        need(h=={'shape':list(v.shape),'dtype':v.dtype.str,'sha256':typed_sha(v)},"semantic array drift")
    edge=np.asarray(edge);offset=np.asarray(offset)
    need(edge.dtype.kind in 'iu' and offset.dtype.kind in 'fiu' and edge.shape==offset.shape
         and edge.size<=50_000 and np.all(edge>=0) and np.all(edge<net.n_edges)
         and np.isfinite(offset).all() and np.all(offset>=0) and np.all(offset<=net.edge_len[edge]),"bounded actual previous edge/offset required")
    g=net.edge_base[edge]+offset
    seg=np.searchsorted(net.seg_gstart,g,side='right')-1
    seg=np.clip(seg,net.seg_lo[edge],net.seg_hi[edge]-1)
    raw=arrays['segment_raw_edge_index'][seg];state=arrays['segment_semantic_status'][seg]
    return {'segment_index':seg,'raw_edge_index':raw,'status':state,
            'neutral':state!=SEG_STATUS['exact_source_context']}
