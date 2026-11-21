"""Bounded separate semantic sidecar storage; no model or authority upgrade."""
from pathlib import Path
import ast
import hashlib
import json
import math
import os
import zipfile
import numpy as np
MAX_JSON=16<<20
MAX_CATALOG=192<<20
MAX_RECORD=8192
MAX_RAW=500_000
MAX_EDGES=100_000
MAX_SEGMENTS=500_000
MAX_MEMBERS=1_000_000
MAX_ARRAY_BYTES=64<<20
MAX_HEADER=4096
ARRAY_NAMES=frozenset(("raw_edge_keys","compact_edge_keys","compact_path_indptr","compact_path_raw_edge_index","segment_raw_edge_index","segment_semantic_status","raw_edge_disposition"))
DTYPES={k:np.dtype("uint8"if k in {"segment_semantic_status","raw_edge_disposition"}else"int64")for k in ARRAY_NAMES}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb")as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def strict_json(data):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,"duplicate JSON key");result[key]=value
        return result
    def bad(value):raise ValueError("nonfinite JSON constant")
    try:value=json.loads(data,object_pairs_hook=pairs,parse_constant=bad)
    except (RecursionError,UnicodeError)as e:raise ValueError("invalid bounded JSON")from e
    stack=[iter((value,))]
    while stack:
        try:item=next(stack[-1])
        except StopIteration:stack.pop();continue
        if isinstance(item,float):require(math.isfinite(item),"nonfinite JSON number")
        if isinstance(item,dict):stack.append(iter(item.values()))
        elif isinstance(item,list):stack.append(iter(item))
        require(len(stack)<=128,"JSON nesting exceeds bounded schema")
    return value


def line_bytes(record):
    return json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")+b"\n"



def validate_record(row,index=None):
    keys={"schema","index","key","endpoint_xy","attributes"}
    require(isinstance(row,dict)and set(row)==keys and row["schema"]=="raw-directed-edge-catalog-record-v1","catalog record schema")
    require(type(row["index"])is int and 0<=row["index"]<MAX_RAW and (index is None or row["index"]==index),"catalog index/order")
    require(isinstance(row["key"],list)and len(row["key"])==3 and all(type(v)is int and -(1<<63)<=v<(1<<63)for v in row["key"]),"catalog int64 key")
    xy=row["endpoint_xy"]
    require(isinstance(xy,list)and len(xy)==2 and all(isinstance(p,list)and len(p)==2 for p in xy),"catalog endpoint dimensions")
    require(all(type(v)in (int,float)and math.isfinite(v)for p in xy for v in p),"finite catalog endpoints")
    require(isinstance(row["attributes"],dict)and all(isinstance(v,dict)and set(v)=={"type","value"}and isinstance(v["type"],str)for v in row["attributes"].values()),"typed graph attributes")
    return row

class CatalogWriter:
    def __init__(self,path,*,check=lambda:None):
        self.path=Path(path);self.check=check;self.stream=self.path.open("xb")
        self.count=0;self.bytes=0;self.hash=hashlib.sha256();self.closed=False
    def emit(self,record):
        from .road_segment_semantics import record_bytes
        self.check();require(not self.closed,"catalog writer closed")
        validate_record(record,self.count)
        data=record_bytes(record)
        require(data==line_bytes(record),"catalog canonical encoding differs")
        require(len(data)<=MAX_RECORD and self.count<MAX_RAW and self.bytes+len(data)<=MAX_CATALOG,"catalog record/count/byte cap; no truncation")
        self.stream.write(data);self.hash.update(data);self.count+=1;self.bytes+=len(data)
    def close(self):
        if not self.closed:self.stream.flush();os.fsync(self.stream.fileno());self.stream.close();self.closed=True
    def finish(self):
        self.close();require(0<self.count<=MAX_RAW,"nonempty complete catalog required")
        return {"record_count":self.count,"bytes":self.bytes,"sha256":self.hash.hexdigest()}


def catalog_preflight(path,*,check=lambda:None):
    path=Path(path);size=path.stat().st_size
    require(0<size<=MAX_CATALOG,"catalog file byte cap before body")
    count=total=0
    with path.open("rb")as f:
        while True:
            check();line=f.readline(MAX_RECORD+1)
            if not line:break
            require(len(line)<=MAX_RECORD and line.endswith(b"\n"),"catalog record byte cap/framing")
            count+=1;total+=len(line)
            require(count<=MAX_RAW and total<=MAX_CATALOG,"catalog cumulative cap before decode")
    require(total==size and count>0,"catalog size/count changed")
    return count,size


def iter_catalog(path,*,expected_count=None,check=lambda:None):
    count,size=catalog_preflight(path,check=check)
    require(expected_count is None or expected_count==count,"catalog count differs")
    expected_hash=sha(path)
    observed=0
    with Path(path).open("rb")as f:
        while True:
            check();line=f.readline(MAX_RECORD+1)
            if not line:break
            require(len(line)<=MAX_RECORD and line.endswith(b"\n"),"catalog changed after preflight")
            observed+=1;require(observed<=count,"catalog grew after preflight")
            value=strict_json(line.decode("utf-8"));require(isinstance(value,dict),"catalog record object required")
            yield validate_record(value,observed-1)
    require(observed==count and Path(path).stat().st_size==size and sha(path)==expected_hash,"catalog changed during decode")


def typed_sha(a):
    require(isinstance(a,np.ndarray)and a.flags.c_contiguous and not a.dtype.hasobject,"plain contiguous typed array required")
    h=hashlib.sha256(json.dumps([list(a.shape),a.dtype.str],separators=(",",":")).encode())
    h.update(memoryview(a).cast("B"));return h.hexdigest()


def validate_arrays(arrays):
    require(set(arrays)==ARRAY_NAMES,"exact separate semantic membership required")
    for k,a in arrays.items():
        require(isinstance(a,np.ndarray)and a.ndim==(2 if k in {"raw_edge_keys","compact_edge_keys"}else 1)and a.dtype==DTYPES[k]and a.dtype.isnative and a.flags.c_contiguous,"semantic dtype/layout: "+k)
    r=len(arrays["raw_edge_keys"]);e=len(arrays["compact_edge_keys"]);s=len(arrays["segment_raw_edge_index"]);m=len(arrays["compact_path_raw_edge_index"])
    require(0<r<=MAX_RAW and 0<e<=MAX_EDGES and 0<s<=MAX_SEGMENTS and 0<=m<=MAX_MEMBERS,"semantic capacities exceeded")
    shapes={"raw_edge_keys":(r,3),"compact_edge_keys":(e,3),"compact_path_indptr":(e+1,),"compact_path_raw_edge_index":(m,),"segment_raw_edge_index":(s,),"segment_semantic_status":(s,),"raw_edge_disposition":(r,)}
    require(all(a.shape==shapes[k]for k,a in arrays.items())and sum(a.nbytes for a in arrays.values())<=MAX_ARRAY_BYTES,"semantic shape/byte bound")
    return {k:{"shape":list(a.shape),"dtype":a.dtype.str,"sha256":typed_sha(a)}for k,a in arrays.items()}


def validate_headers(headers):
    require(isinstance(headers,dict)and set(headers)==ARRAY_NAMES,"semantic header membership")
    for k,h in headers.items():
        require(isinstance(h,dict)and set(h)=={"shape","dtype","sha256"}and isinstance(h["shape"],list)
            and all(type(v)is int and v>=0 for v in h["shape"]),"typed semantic headers required")
        require(h["dtype"]==DTYPES[k].str and isinstance(h["sha256"],str)and len(h["sha256"])==64
            and all(c in "0123456789abcdef"for c in h["sha256"]),"semantic header dtype/hash")
    def first(k):
        require(len(headers[k]["shape"])>0,"non-scalar semantic header required")
        return headers[k]["shape"][0]
    r=first("raw_edge_keys");e=first("compact_edge_keys");s=first("segment_raw_edge_index");m=first("compact_path_raw_edge_index")
    require(0<r<=MAX_RAW and 0<e<=MAX_EDGES and 0<s<=MAX_SEGMENTS and m<=MAX_MEMBERS,"semantic header capacities")
    shapes={"raw_edge_keys":[r,3],"compact_edge_keys":[e,3],"compact_path_indptr":[e+1],"compact_path_raw_edge_index":[m],"segment_raw_edge_index":[s],"segment_semantic_status":[s],"raw_edge_disposition":[r]}
    require(all(headers[k]["shape"]==shape for k,shape in shapes.items()),"semantic header shape")
    sizes={k:math.prod(h["shape"])*DTYPES[k].itemsize for k,h in headers.items()}
    require(sum(sizes.values())<=MAX_ARRAY_BYTES,"semantic header cumulative payload cap")
    return sizes


def _npy_header(stream,info):
    magic=stream.read(8)
    require(len(magic)==8 and magic[:6]==b"\x93NUMPY"and magic[6:]in (b"\x01\x00",b"\x02\x00",b"\x03\x00"),"NPY magic/version")
    n=2 if magic[6]==1 else 4;b=stream.read(n);require(len(b)==n,"short NPY prefix")
    length=int.from_bytes(b,"little")
    require(0<length<=MAX_HEADER and 8+n+length<=info.file_size,"NPY header cap before read")
    data=stream.read(length);require(len(data)==length,"short NPY header")
    try:h=ast.literal_eval(data.decode("utf-8"if magic[6]==3 else"latin1"))
    except (ValueError,SyntaxError,RecursionError)as e:raise ValueError("invalid bounded NPY header")from e
    require(isinstance(h,dict)and set(h)=={"descr","shape","fortran_order"}and h["fortran_order"]is False,"plain C NPY header")
    require(isinstance(h["shape"],tuple)and all(type(v)is int and v>=0 for v in h["shape"]),"NPY dimensions")
    dtype=np.dtype(h["descr"]);require(not dtype.hasobject,"object NPY forbidden")
    return h["shape"],dtype,8+n+length


def load_arrays(path,headers,*,check=lambda:None):
    sizes=validate_headers(headers);path=Path(path)
    require(path.stat().st_size<=MAX_ARRAY_BYTES+7*(MAX_HEADER+128)+65536,"semantic NPZ archive cap before body")
    with zipfile.ZipFile(path)as archive:
        entries=archive.infolist();names={k+".npy"for k in ARRAY_NAMES}
        require(len(entries)==7 and {i.filename for i in entries}==names,"semantic ZIP exact membership")
        require(all(0<i.file_size<=sizes[i.filename[:-4]]+MAX_HEADER+12 and not(i.flag_bits&1)for i in entries)
            and sum(i.file_size for i in entries)<=MAX_ARRAY_BYTES+7*(MAX_HEADER+12),"ALL semantic ZIP caps before first header")
        offsets={}
        for info in entries:
            check();key=info.filename[:-4]
            with archive.open(info)as f:shape,dtype,offset=_npy_header(f,info)
            require(list(shape)==headers[key]["shape"]and dtype==DTYPES[key]and dtype.isnative
                and info.file_size==offset+sizes[key],"semantic NPY header/payload EOF differs")
            offsets[key]=offset
        arrays={}
        for info in entries:
            check();key=info.filename[:-4]
            with archive.open(info)as f:
                f.read(offsets[key]);body=f.read(sizes[key]);require(len(body)==sizes[key]and f.read(1)==b"","semantic payload EOF")
            value=np.frombuffer(body,dtype=DTYPES[key]).reshape(headers[key]["shape"]).copy()
            require(typed_sha(value)==headers[key]["sha256"],"semantic typed body hash differs")
            arrays[key]=value
    validate_arrays(arrays);return arrays


def write_packet(output,packet,catalog_info,network_metadata):
    output=Path(output);require(set(packet)=={"arrays","metadata"},"pure semantic packet grammar")
    headers=validate_arrays(packet["arrays"]);validate_headers(headers)
    require(catalog_info["record_count"]==headers["raw_edge_keys"]["shape"][0]
        and (output/"raw_edge_catalog.jsonl").stat().st_size==catalog_info["bytes"]
        and sha(output/"raw_edge_catalog.jsonl")==catalog_info["sha256"],"complete emitted catalog differs")
    meta=packet["metadata"]
    require(meta.get("array_headers")==headers and meta.get("catalog_sha256")==catalog_info["sha256"]
        and meta.get("catalog_bytes")==catalog_info["bytes"] and meta.get("raw_edge_count")==catalog_info["record_count"],"pure semantic metadata/catalog/header binding")
    require(meta["net_binding"]["arrays"]==network_metadata["array_sha256"]and meta["net_binding"]["crs"]==str(network_metadata["crs"]),"pure sidecar network binding")
    metadata={"schema":"road-segment-semantics-storage-v1","core_metadata":packet["metadata"],"arrays":headers,
        "catalog":dict(catalog_info),"network_array_sha256":network_metadata["array_sha256"],
        "loaded_graph_artifact_only":True,"original_OSM_or_acquisition_history_verified":False,
        "motion_model_implemented":False,"ground_truth_used":False,"policy_admission":False}
    with (output/"segment_semantics.npz").open("xb")as f:np.savez(f,**packet["arrays"])
    data=(json.dumps(metadata,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
    require(len(data)<=MAX_JSON,"semantic storage metadata cap")
    with (output/"segment_semantics.json").open("xb")as f:f.write(data)
    return metadata


def read_packet(output,*,check=lambda:None):
    output=Path(output);meta=output/"segment_semantics.json"
    require(meta.stat().st_size<=MAX_JSON,"semantic metadata cap before body")
    metadata_hash=sha(meta)
    metadata=read_json(meta)
    keys={"schema","core_metadata","arrays","catalog","network_array_sha256","loaded_graph_artifact_only","original_OSM_or_acquisition_history_verified","motion_model_implemented","ground_truth_used","policy_admission"}
    require(set(metadata)==keys and metadata["schema"]=="road-segment-semantics-storage-v1"
        and metadata["loaded_graph_artifact_only"]is True and all(metadata[k]is False for k in
        ("original_OSM_or_acquisition_history_verified","motion_model_implemented","ground_truth_used","policy_admission")),"semantic storage schema/scope")
    validate_headers(metadata["arrays"])
    info=metadata["catalog"];catalog=output/"raw_edge_catalog.jsonl"
    count,size=catalog_preflight(catalog,check=check)
    require(info=={"record_count":count,"bytes":size,"sha256":sha(catalog)},"catalog storage identity differs")
    header=metadata["arrays"]["raw_edge_keys"]
    h=hashlib.sha256(json.dumps([header["shape"],header["dtype"]],separators=(",",":")).encode())
    for row in iter_catalog(catalog,expected_count=count,check=check):
        h.update(np.asarray(row["key"],dtype=np.int64).tobytes())
    require(h.hexdigest()==header["sha256"],"streamed catalog keys differ from raw header")
    core=metadata["core_metadata"]
    require(core.get("array_headers")==metadata["arrays"]and core.get("catalog_sha256")==info["sha256"]
        and core.get("catalog_bytes")==size and core.get("raw_edge_count")==count
        and core.get("net_binding",{}).get("arrays")==metadata["network_array_sha256"],"semantic storage/core bindings")
    npz=output/"segment_semantics.npz"
    require(npz.stat().st_size<=MAX_ARRAY_BYTES+7*(MAX_HEADER+128)+65536,"semantic NPZ cap before hash")
    npz_hash=sha(npz)
    arrays=load_arrays(npz,metadata["arrays"],check=check)
    require(count==len(arrays["raw_edge_keys"]),"catalog/typed raw count differs")
    check()  # LAST callback; final expected hashes are callback-free.
    for path,digest in ((meta,metadata_hash),(catalog,info["sha256"]),(npz,npz_hash)):
        require(sha(path)==digest,"semantic storage changed during read")
    return {"arrays":arrays,"metadata":metadata["core_metadata"]},metadata


def read_json(path):
    path=Path(path);require(path.stat().st_size<=MAX_JSON,"JSON byte cap before body")
    with path.open("rb")as f:data=f.read(MAX_JSON+1)
    require(len(data)<=MAX_JSON,"JSON grew beyond byte cap")
    return strict_json(data)
