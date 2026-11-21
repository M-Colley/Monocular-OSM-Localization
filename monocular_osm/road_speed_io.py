"""Finite R19 admission; historical raw-map/source refs stay opaque."""
from pathlib import Path
import hashlib
import json
import math
import os
import zipfile
import numpy as np
from . import road_segment_semantics_io as storage
from . import road_segment_semantics as semantics
from scripts import junction_sequence_probe as p
NET_NAMES=frozenset(p.NET_ARRAYS)
OLD_FILES=frozenset(("network.npz","network.json","matched.json","topology_receipt.json","plan.json"))
SEM_FILES=OLD_FILES|{"raw_edge_catalog.jsonl","segment_semantics.npz","segment_semantics.json"}
INTEGER=frozenset(("edge_keys","seg_lo","seg_hi","reverse_of","succ_table"))
MAX_BASE_BYTES=512<<20


def opaque_map(value):
    """Validate historical spelling/digests without resolving or touching paths."""
    storage.require(isinstance(value,dict)and len(value)<=4096,"bounded historical map")
    result={}
    for path,digest in value.items():
        storage.require(isinstance(path,str)and 0<len(path)<=32768 and isinstance(digest,str)
            and len(digest)==64 and all(c in '0123456789abcdef'for c in digest),"historical map grammar")
        key=path.replace('\\','/').casefold()
        storage.require(key not in result,"duplicate historical path spelling")
        result[key]=digest
    return result


def prepared_inputs(config,ledger,expected_sources,facts):
    """R19-only basic admission; original prepare input refs are opaque history."""
    path=ledger.protect(config['inputs']['prepared']);master=ledger.json(config['inputs']['prepared'])
    storage.require(master.get('schema')=='junction-sequence-probe-prepare-freeze-v1'
        and master.get('status')=='PREPARED_COMPACT_JUNCTION_SEQUENCE_INPUTS_NO_ACCURACY'
        and master.get('preparation_worker_must_exit_before_run')is True
        and master.get('ground_truth_used')is False and master.get('selected_answer')is None
        and master.get('accuracy_admitted')is False,"fresh basic preparation required")
    storage.require(master.get('origin_video_sha256')==config['video_binding']['sha256']==facts['video_sha256'],"fresh/native video origins differ")
    historic=opaque_map(master['protected_hashes'])
    source={k:v for k,v in historic.items()if k.endswith('.py')}
    storage.require(source==opaque_map(expected_sources),"fresh prepare/run complete current sources differ")
    storage.require(set(master['artifacts'])==OLD_FILES and {q.name for q in path.parent.iterdir()}==OLD_FILES|{path.name},"exact fresh basic membership")
    members={}
    for name,digest in master['artifacts'].items():
        member=path.parent/name
        storage.require(member.stat().st_size<=(MAX_BASE_BYTES if name.endswith('.npz')else storage.MAX_JSON),"fresh basic role cap before hash")
        members[name]=ledger.protect({'path':str(member),'sha256':digest})
    return path,members


def _metadata(ledger,ref):
    path=Path(ref['path']);path=ledger.base/path if not path.is_absolute()else path
    storage.require(path.stat().st_size<=storage.MAX_JSON,"semantic authority JSON cap before hash")
    return ledger.json(ref)


def _births(value):
    storage.require(isinstance(value,list)and 0<len(value)<=4096,"closed observed births required")
    # Root-authenticated closure owns the platform-specific birth tuple grammar.
    storage.require(all(isinstance(row,(list,tuple))and len(row)==2 for row in value),"closed birth pairs required")


def admit_semantics(config,ledger,study,equal_ref):
    """Only current sealed outputs/closed metadata are consumed, never old files maps."""
    refs=config['semantic_inputs']
    storage.require(isinstance(refs,dict)and set(refs)=={'root_acceptance','prepared_master'},"exact semantic authority roles")
    equal_ref(refs['root_acceptance'],study['semantic_root_acceptance_ref'],ledger.base)
    equal_ref(refs['prepared_master'],study['semantic_prepared_master_ref'],ledger.base)
    seal=_metadata(ledger,refs['root_acceptance'])
    storage.require(seal.get('schema')=='root-sealed-original-segment-semantics-acceptance-v1'
        and seal.get('status')=='ACCEPTED_EXACT_GRAPH_ARTIFACT_SEMANTICS_NO_MODEL_OR_ACCURACY',"exact Root semantic acceptance")
    storage.require(all(seal.get(k)is False for k in ('original_OSM_tag_coverage_verified','original_way_direction_verified',
        'acquisition_history_upgraded','speed_model_implemented','ground_truth_used','counts_toward_goal'))
        and seal.get('selected_answer')is None,"semantic evidence must not upgrade historical authority")
    equal_ref(seal['prepared_master_ref'],refs['prepared_master'],ledger.base)
    producer=_metadata(ledger,seal['producer_closure_ref']);validator=_metadata(ledger,seal['root_validation_closure_ref'])
    integrity=_metadata(ledger,seal['integrity_ref'])
    storage.require(producer.get('schema')=='root-original-segment-semantic-producer-external-closure-v1'
        and producer.get('status')=='CLOSED_PREPARED_PACKET_NO_SEMANTIC_AUTHORITY_YET'
        and producer.get('all_source_inputs_outputs_unchanged')is True,"closed semantic producer required")
    storage.require(validator.get('schema')=='root-independent-segment-validation-external-closure-v1'
        and validator.get('all_frozen_source_inputs_and_outputs_unchanged')is True,"closed independent semantic validator required")
    _births(producer['all_observed_worker_births_closed']);_births(validator['all_observed_worker_births_closed'])
    equal_ref(producer['master_ref'],refs['prepared_master'],ledger.base)
    equal_ref(validator['integrity_ref'],seal['integrity_ref'],ledger.base)
    storage.require(integrity.get('schema')=='root-independent-segment-semantics-integrity-v1'
        and integrity.get('status')=='PASS_ALL_SOURCE_RECORDS_AND_SEGMENTS'
        and integrity.get('checks')==seal['checks'] and integrity.get('ground_truth_used')is False,"independent semantic integrity required")
    checks=seal['checks']
    storage.require(all(checks.get(k)is True for k in ('old18_typed_arrays_exact','full_old_network_matched_topology_plan_metadata_exact',
        'all_loaded_attribute_values_and_types_verified','compact_key_to_original_path_independently_replayed',
        'every_original_edge_disposition_rebuilt','every_segment_mapping_or_neutral_reason_rebuilt'))
        and all(checks.get(k)is False for k in ('new_motion_model','original_OSM_coverage_or_way_direction_upgraded','ground_truth_used')),"complete original semantic checks required")
    master=_metadata(ledger,refs['prepared_master'])
    storage.require(master.get('schema')=='road-segment-semantics-prepare-freeze-v1'
        and master.get('status')=='PREPARED_ROAD_SEGMENT_SEMANTICS_INPUTS_NO_ACCURACY'
        and master.get('semantic_profile_sha256')==semantics.PROFILE_SHA256
        and master.get('loaded_graph_artifact_only')is True
        and all(master.get(k)is False for k in ('original_OSM_or_acquisition_history_verified','speed_model_implemented',
            'motion_model_changed','ground_truth_used','policy_admission','accuracy_admitted'))
        and master.get('selected_answer')is None and set(master['artifacts'])==SEM_FILES,"complete sealed semantic packet required")
    opaque_map(master['protected_hashes'])  # Validation only: no stat/open/resolve/hash of original refs.
    root=Path(ledger.protect(refs['prepared_master'])).parent
    storage.require({q.name for q in root.iterdir()}==SEM_FILES|{'frozen_prepare.json'},"sealed semantic exact membership")
    members={}
    for name,digest in master['artifacts'].items():
        path=root/name
        limit=(storage.MAX_CATALOG if name.endswith('.jsonl')else MAX_BASE_BYTES if name=='network.npz'
            else storage.MAX_ARRAY_BYTES+7*(storage.MAX_HEADER+128)+65536 if name.endswith('.npz')else storage.MAX_JSON)
        storage.require(path.stat().st_size<=limit,"sealed role cap before hash")
        members[name]=ledger.protect({'path':str(path),'sha256':digest})
    meta=_metadata(ledger,{'path':str(members['segment_semantics.json']),'sha256':master['artifacts']['segment_semantics.json']})
    network=_metadata(ledger,{'path':str(members['network.json']),'sha256':master['artifacts']['network.json']})
    validate_storage_metadata(meta,network,master)
    return {'master':master,'members':members,'storage':meta,
        'historical_binding':'opaque historical metadata binding; original raw bytes not freshly verified'}


def validate_storage_metadata(meta,network,master):
    keys={'schema','core_metadata','arrays','catalog','network_array_sha256','loaded_graph_artifact_only',
        'original_OSM_or_acquisition_history_verified','motion_model_implemented','ground_truth_used','policy_admission'}
    storage.require(set(meta)==keys and meta['schema']=='road-segment-semantics-storage-v1'
        and meta['loaded_graph_artifact_only']is True and all(meta[k]is False for k in
        ('original_OSM_or_acquisition_history_verified','motion_model_implemented','ground_truth_used','policy_admission')),"semantic storage scope before values")
    storage.validate_headers(meta['arrays']);core=meta['core_metadata'];info=meta['catalog']
    storage.require(core.get('status')=='complete_semantic_sidecar_no_motion_model'
        and core.get('profile_sha256')==semantics.PROFILE_SHA256
        and hashlib.sha256(json.dumps(core['profile'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==semantics.PROFILE_SHA256
        and core.get('array_headers')==meta['arrays'] and core.get('catalog_sha256')==info.get('sha256')==master['artifacts']['raw_edge_catalog.jsonl']
        and core.get('catalog_bytes')==info.get('bytes') and core.get('raw_edge_count')==info.get('record_count')
        and meta['arrays']['raw_edge_keys']['shape'][0]==info.get('record_count')
        and network.get('schema')=='compact-junction-road-network-v1'and set(network['array_sha256'])==NET_NAMES
        and core.get('net_binding',{}).get('arrays')==meta['network_array_sha256']==network['array_sha256']
        and core['net_binding']['crs']==str(network['crs']),"semantic pure profile/catalog/base binding before values")


def load_base_network(path,metadata,graph_filter,*,check=lambda:None):
    """All18 caps/headers/shapes before any numeric array materialization."""
    path=Path(path);storage.require(path.stat().st_size<=MAX_BASE_BYTES,"base archive cap before body")
    hashes=metadata.get("array_sha256")
    storage.require(metadata.get("schema")=="compact-junction-road-network-v1"and isinstance(hashes,dict)and set(hashes)==NET_NAMES,"exact old18 base metadata")
    maxima={k:100000 for k in NET_NAMES}
    maxima.update(origin=2,xy_max=2,edge_keys=300000,edge_mid=200000,seg_gstart=500000,seg_heading=500000,
        seg_xy0=1000000,seg_dir=1000000,succ_table=3200000,succ_head=3200000,succ_logprior=3200000)
    with zipfile.ZipFile(path)as archive:
        entries=archive.infolist()
        storage.require(len(entries)==18 and {i.filename for i in entries}=={k+".npy"for k in NET_NAMES},"exact old18 ZIP membership")
        storage.require(all(0<i.file_size<=8*maxima[i.filename[:-4]]+storage.MAX_HEADER+12 and not(i.flag_bits&1)for i in entries)
            and sum(i.file_size for i in entries)<=MAX_BASE_BYTES,"ALL base ZIP caps before first header")
        headers={}
        for info in entries:
            check();key=info.filename[:-4]
            with archive.open(info)as f:shape,dtype,offset=storage._npy_header(f,info)
            expected=np.dtype('int64'if key in INTEGER else'float64')
            storage.require(dtype==expected and dtype.isnative and 1<=len(shape)<=2 and math.prod(shape)<=maxima[key],"base dtype/dimension cap")
            storage.require(info.file_size==offset+math.prod(shape)*8,"base NPY exact payload EOF")
            headers[key]=(shape,dtype,offset)
        e=headers['edge_len'][0][0];s=headers['seg_gstart'][0][0];succ=headers['succ_table'][0]
        storage.require(0<e<=100000 and 0<s<=500000 and len(succ)==2 and succ[0]==e and 0<succ[1]<=32,"base inferred capacities")
        d=succ[1];shapes={"origin":(2,),"xy_max":(2,),"edge_keys":(e,3),"edge_mid":(e,2),"seg_xy0":(s,2),"seg_dir":(s,2)}
        shapes.update({k:(s,)for k in ('seg_gstart','seg_heading')})
        shapes.update({k:(e,d)for k in ('succ_table','succ_head','succ_logprior')})
        storage.require(all(shape==shapes.get(k,(e,))for k,(shape,_,_)in headers.items()),"ALL old18 exact shapes before values")
        arrays={}
        for info in entries:
            check();key=info.filename[:-4];shape,dtype,offset=headers[key];size=math.prod(shape)*8
            with archive.open(info)as f:
                f.read(offset);body=f.read(size);storage.require(len(body)==size and f.read(1)==b'',"base exact body EOF/CRC")
            value=np.frombuffer(body,dtype=dtype).reshape(shape).copy()
            storage.require(storage.typed_sha(value)==hashes[key],"base typed bytes differ")
            arrays[key]=value
    return p.load_network(arrays,metadata,graph_filter)


def load_context(admitted,fresh_members,ledger,graph_filter,context_core,*,check=lambda:None):
    """Finite equal-base proof, then one immutable context from the entire catalog."""
    old=admitted['members'];master=admitted['master'];meta=admitted['storage']
    def current_json(path):
        return ledger.json({'path':str(path),'sha256':ledger.protected[os.path.normcase(str(path.resolve()))]})
    fresh_meta=current_json(fresh_members['network.json']);old_meta=current_json(old['network.json'])
    storage.require(fresh_meta==old_meta,"fresh/R18 full old network metadata differs")
    for name in ('matched.json','topology_receipt.json','plan.json'):
        storage.require(current_json(fresh_members[name])==current_json(old[name]),"fresh/R18 full old metadata differs: "+name)
    net=load_base_network(fresh_members['network.npz'],fresh_meta,graph_filter,check=check)
    old_net=load_base_network(old['network.npz'],old_meta,graph_filter,check=check)
    for name in NET_NAMES:
        storage.require(np.asarray(getattr(net,name)).dtype==np.asarray(getattr(old_net,name)).dtype
            and np.asarray(getattr(net,name)).shape==np.asarray(getattr(old_net,name)).shape
            and np.asarray(getattr(net,name)).tobytes()==np.asarray(getattr(old_net,name)).tobytes(),"fresh/R18 all18 exact array mismatch: "+name)
    del old_net
    keys={'schema','core_metadata','arrays','catalog','network_array_sha256','loaded_graph_artifact_only',
        'original_OSM_or_acquisition_history_verified','motion_model_implemented','ground_truth_used','policy_admission'}
    storage.require(set(meta)==keys and meta['schema']=='road-segment-semantics-storage-v1'
        and meta['loaded_graph_artifact_only']is True and all(meta[k]is False for k in
        ('original_OSM_or_acquisition_history_verified','motion_model_implemented','ground_truth_used','policy_admission')),"semantic storage scope")
    storage.validate_headers(meta['arrays'])
    info=meta['catalog'];core=meta['core_metadata'];catalog=old['raw_edge_catalog.jsonl']
    count,size=storage.catalog_preflight(catalog,check=check)
    storage.require(info=={'record_count':count,'bytes':size,'sha256':master['artifacts']['raw_edge_catalog.jsonl']}
        and core.get('array_headers')==meta['arrays'] and core.get('catalog_sha256')==info['sha256']
        and core.get('catalog_bytes')==size and core.get('raw_edge_count')==count
        and core.get('net_binding',{}).get('arrays')==meta['network_array_sha256']==fresh_meta['array_sha256'],"complete catalog/base/storage identity")
    arrays=storage.load_arrays(old['segment_semantics.npz'],meta['arrays'],check=check)
    storage.require(count==len(arrays['raw_edge_keys']),"catalog/typed key count differs")
    own,context=context_core.prepare(net,{'arrays':arrays,'metadata':core},
        storage.iter_catalog(catalog,expected_count=count,check=check),catalog_sha256=info['sha256'],catalog_bytes=size)
    return own,context,{'all18_base_arrays_exact':True,'full_old_five_metadata_exact':True,
        'entire_sealed_catalog_consumed':True,'historical_refs_opaque':True,'original_raw_bytes_freshly_verified':False}


def verify_memberships(admitted,fresh_path):
    """No callback/body read: final sealed and fresh packet exact member sets."""
    root=admitted['members']['network.npz'].parent
    storage.require({q.name for q in root.iterdir()}==SEM_FILES|{'frozen_prepare.json'},"late sealed semantic membership changed")
    storage.require({q.name for q in fresh_path.parent.iterdir()}==OLD_FILES|{fresh_path.name},"late fresh basic membership changed")
