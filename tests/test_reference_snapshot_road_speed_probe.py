"""Private semantic authorities and genuine synthetic road/50k fixtures only."""
from copy import deepcopy
from pathlib import Path
import io as bytesio
import zipfile
import numpy as np
import pytest
from scripts import reference_snapshot_road_speed_probe as w
from tests import test_reference_snapshot_future_guide_probe as parent
from tests.test_junction_sequence_probe import roads
from tests.test_prepare_road_segment_semantics import owned
from tests.test_raw_vpr_junction_probe import write,ref


def reseal(tmp,config):
    return write(tmp,tmp/'run_config.json',config)


def execute(tmp,config,**kwargs):
    r=reseal(tmp,config);return w.execute(r['path'],r['sha256'],**kwargs)


def fixture(tmp,monkeypatch,bank='baseline',target='legacy',numeric=True):
    monkeypatch.setattr(parent,'w',w)
    config,report,qa=parent.fixture(tmp,monkeypatch,bank,target)
    config['schema']='reference-snapshot-road-speed-probe-run-v1'
    root=owned(tmp,tmp/'semantic');root.mkdir()
    fresh=Path(config['inputs']['prepared']['path']).parent
    for name in w.semantic_io.OLD_FILES:owned(tmp,root/name).write_bytes((fresh/name).read_bytes())
    graph,_=roads()
    if numeric:
        for _,_,data in graph.edges(data=True):data['maxspeed']='50'
    core=w.road_segment_semantics;storage=w.road_segment_semantics_io
    writer=storage.CatalogWriter(owned(tmp,root/'raw_edge_catalog.jsonl'))
    catalog=core.attach(graph,emit_record=writer.emit);info=writer.finish()
    gf=w.p.dependencies()[3];trace=[];net=gf.RoadNetwork.from_graph(graph,construction_trace=trace)
    for row in trace:row['source_keyed_path']=graph.edges[row['edge_key']][core.TOKEN]
    packet=core.bind_mapping(catalog,trace,net)
    storage.write_packet(root,packet,info,w.p.read_json(root/'network.json'))
    master={'schema':'road-segment-semantics-prepare-freeze-v1','status':'PREPARED_ROAD_SEGMENT_SEMANTICS_INPUTS_NO_ACCURACY',
        'semantic_profile_sha256':core.PROFILE_SHA256,'artifacts':{n:w.p.sha(root/n)for n in w.semantic_io.SEM_FILES},
        'protected_hashes':{'Z:/never_open_historical_map.graphml':'a'*64,'Z:/never_open_original_source.py':'b'*64},
        'loaded_graph_artifact_only':True,'original_OSM_or_acquisition_history_verified':False,'speed_model_implemented':False,
        'motion_model_changed':False,'ground_truth_used':False,'policy_admission':False,'accuracy_admitted':False,'selected_answer':None}
    mr=write(tmp,root/'frozen_prepare.json',master)
    checks={k:True for k in ('old18_typed_arrays_exact','full_old_network_matched_topology_plan_metadata_exact',
        'all_loaded_attribute_values_and_types_verified','compact_key_to_original_path_independently_replayed',
        'every_original_edge_disposition_rebuilt','every_segment_mapping_or_neutral_reason_rebuilt')}
    checks.update(new_motion_model=False,original_OSM_coverage_or_way_direction_upgraded=False,ground_truth_used=False)
    integrity=write(tmp,tmp/'semantic_integrity.json',{'schema':'root-independent-segment-semantics-integrity-v1',
        'status':'PASS_ALL_SOURCE_RECORDS_AND_SEGMENTS','checks':checks,'ground_truth_used':False})
    producer=write(tmp,tmp/'semantic_producer_closed.json',{'schema':'root-original-segment-semantic-producer-external-closure-v1',
        'status':'CLOSED_PREPARED_PACKET_NO_SEMANTIC_AUTHORITY_YET','master_ref':mr,
        'all_source_inputs_outputs_unchanged':True,'all_observed_worker_births_closed':[[123,1.2]],'files':[{'path':'Z:/forbidden_map','sha256':'c'*64}]})
    validation=write(tmp,tmp/'semantic_validation_closed.json',{'schema':'root-independent-segment-validation-external-closure-v1',
        'all_frozen_source_inputs_and_outputs_unchanged':True,'all_observed_worker_births_closed':[[456,3.4]],'integrity_ref':integrity,
        'files':[{'path':'Z:/forbidden_original_source','sha256':'d'*64}]})
    seal={'schema':'root-sealed-original-segment-semantics-acceptance-v1',
        'status':'ACCEPTED_EXACT_GRAPH_ARTIFACT_SEMANTICS_NO_MODEL_OR_ACCURACY','prepared_master_ref':mr,
        'producer_closure_ref':producer,'root_validation_closure_ref':validation,'integrity_ref':integrity,'checks':checks,
        **{k:False for k in ('original_OSM_tag_coverage_verified','original_way_direction_verified','acquisition_history_upgraded',
            'speed_model_implemented','ground_truth_used','counts_toward_goal')},'selected_answer':None}
    sr=write(tmp,tmp/'semantic_root.json',seal)
    config['semantic_inputs']={'root_acceptance':sr,'prepared_master':mr}
    study={'schema':'root-prospective-road-speed-ou-snapshot-study-v1','kernel_mode':w.kernel_core.MODE,
        'controls':[{'name':b+'_'+h,'bank_key':b,'heading_target':h}for b in ('baseline','expanded')for h in w.TARGETS],
        'semantic_root_acceptance_ref':sr,'semantic_prepared_master_ref':mr,'snapshot_DATA_study_ref':config['study_registration_ref'],
        'normalized_heading_study_ref':config['heading_study_registration_ref'],'future_proposal_study_ref':config['future_guide_study_registration_ref']}
    studyref=write(tmp,tmp/'road_study.json',study)
    config['road_speed_study_registration_ref']=studyref;monkeypatch.setattr(w,'ROAD_SPEED_STUDY_REF',studyref)
    return config,report,qa


@pytest.mark.parametrize('bank',['baseline','expanded'])
@pytest.mark.parametrize('target',['legacy','normalized-junction-v1'])
def test_all_four_fixed_controls_with_genuine_context_and_witness(tmp_path,monkeypatch,bank,target):
    config,_,_=fixture(tmp_path,monkeypatch,bank,target)
    result=execute(tmp_path,config)
    assert result['numerical_status']=='complete_diagnostic_only'
    assert result['proposal_only']is False and result['future_guide_is_proposal_only']is True
    assert result['road_speed_mode']==w.kernel_core.MODE and result['semantic_equality']['all18_base_arrays_exact']is True
    saved=w.admission.read_snapshot_report(w.Ledger(tmp_path,lambda:None),ref(Path(config['output_dir'])/'result.json'))
    assert saved['road_speed_context_receipt']['ENTIRE_catalog_consumed']is True
    assert [v['query_row']for v in saved['appearance_events']]==list(range(80))
    assert saved['native_heading_variance_unchanged']is True if 'native_heading_variance_unchanged'in saved else saved['profile']['native_heading_variance_unchanged']is True
    with np.load(Path(config['output_dir'])/'result.npz',allow_pickle=False)as z:
        assert z['road_witness_previous_edge'].shape[1]==50000 and z['road_witness_uniform'].dtype==np.float64
        assert np.any(z['road_context_count']>0) and np.any(z['road_component_count']>0)
    assert not any('never_open_historical'in k or 'forbidden_'in k for k in result['protected_hashes'])


@pytest.mark.parametrize('fault',['road_ref','road_schema','DATA_ref','heading_ref','future_ref','helper_source','io_source',
    'context_source','knob','wrong_mode','semantic_ref','semantic_status','producer_closed','validation_closed','checks','master_status','storage_schema','storage_profile'])
def test_new_and_parent_authority_gates_before_numeric(tmp_path,monkeypatch,fault):
    config,_,_=fixture(tmp_path,monkeypatch)
    if fault in ('road_ref','DATA_ref','heading_ref','future_ref'):
        key={'road_ref':'road_speed_study_registration_ref','DATA_ref':'study_registration_ref','heading_ref':'heading_study_registration_ref','future_ref':'future_guide_study_registration_ref'}[fault]
        config[key]={**config[key],'sha256':'f'*64}
    if fault.endswith('_source'):
        name={'helper_source':'road_speed_kernel','io_source':'road_speed_io','context_source':'road_speed_context'}[fault]
        config['source_hashes']['monocular_osm/'+name+'.py']='f'*64
    if fault=='knob':config['road_extra_sd']=.8
    if fault in ('road_schema','wrong_mode'):
        study=w.p.read_json(config['road_speed_study_registration_ref']['path']);study['schema'if fault=='road_schema'else'kernel_mode']='wrong'
        r=write(tmp_path,config['road_speed_study_registration_ref']['path'],study);config['road_speed_study_registration_ref']=r;monkeypatch.setattr(w,'ROAD_SPEED_STUDY_REF',r)
    if fault=='semantic_ref':config['semantic_inputs']['root_acceptance']={**config['semantic_inputs']['root_acceptance'],'sha256':'f'*64}
    if fault in ('storage_schema','storage_profile'):
        root=Path(config['semantic_inputs']['prepared_master']['path']).parent
        meta=w.p.read_json(root/'segment_semantics.json')
        if fault=='storage_schema':meta['schema']='old-storage'
        else:meta['core_metadata']['profile_sha256']='f'*64
        write(tmp_path,root/'segment_semantics.json',meta);rebind_semantics(tmp_path,config,monkeypatch)
    if fault in ('semantic_status','producer_closed','validation_closed','checks','master_status'):
        seal=w.p.read_json(config['semantic_inputs']['root_acceptance']['path'])
        if fault=='semantic_status':seal['status']='accepted'
        if fault=='checks':seal['checks']['old18_typed_arrays_exact']=False
        if fault in ('producer_closed','validation_closed'):
            key='producer_closure_ref'if fault=='producer_closed'else'root_validation_closure_ref'
            obj=w.p.read_json(seal[key]['path']);obj['all_observed_worker_births_closed']=[];seal[key]=write(tmp_path,seal[key]['path'],obj)
        if fault=='master_status':
            obj=w.p.read_json(config['semantic_inputs']['prepared_master']['path']);obj['status']='wrong'
            config['semantic_inputs']['prepared_master']=write(tmp_path,config['semantic_inputs']['prepared_master']['path'],obj)
        config['semantic_inputs']['root_acceptance']=write(tmp_path,config['semantic_inputs']['root_acceptance']['path'],seal)
        study=w.p.read_json(config['road_speed_study_registration_ref']['path']);study.update(semantic_root_acceptance_ref=config['semantic_inputs']['root_acceptance'],semantic_prepared_master_ref=config['semantic_inputs']['prepared_master'])
        r=write(tmp_path,config['road_speed_study_registration_ref']['path'],study);config['road_speed_study_registration_ref']=r;monkeypatch.setattr(w,'ROAD_SPEED_STUDY_REF',r)
    monkeypatch.setattr(w,'load_bank',lambda *x:pytest.fail('bank numeric before authorities'))
    monkeypatch.setattr(w.semantic_io,'load_base_network',lambda *x,**kw:pytest.fail('base numeric before authorities'))
    with pytest.raises((ValueError,KeyError)):execute(tmp_path,config)
    assert not Path(config['output_dir']).exists()


def rebind_semantics(tmp,config,monkeypatch):
    refs=config['semantic_inputs'];master=w.p.read_json(refs['prepared_master']['path'])
    root=Path(refs['prepared_master']['path']).parent
    master['artifacts']={n:w.p.sha(root/n)for n in w.semantic_io.SEM_FILES}
    refs['prepared_master']=write(tmp,refs['prepared_master']['path'],master)
    seal=w.p.read_json(refs['root_acceptance']['path']);seal['prepared_master_ref']=refs['prepared_master']
    producer=w.p.read_json(seal['producer_closure_ref']['path']);producer['master_ref']=refs['prepared_master']
    seal['producer_closure_ref']=write(tmp,seal['producer_closure_ref']['path'],producer)
    refs['root_acceptance']=write(tmp,refs['root_acceptance']['path'],seal)
    study=w.p.read_json(config['road_speed_study_registration_ref']['path'])
    study.update(semantic_root_acceptance_ref=refs['root_acceptance'],semantic_prepared_master_ref=refs['prepared_master'])
    r=write(tmp,config['road_speed_study_registration_ref']['path'],study)
    config['road_speed_study_registration_ref']=r;monkeypatch.setattr(w,'ROAD_SPEED_STUDY_REF',r)


def test_numerical_phase_never_opens_stats_or_hashes_historical_originals(tmp_path,monkeypatch):
    config,_,_=fixture(tmp_path,monkeypatch,numeric=False)
    prepared=w.p.read_json(config['inputs']['prepared']['path'])
    forbidden={Path(p).resolve()for p in prepared['protected_hashes']if not p.endswith('.py')}
    # These include the actual private raw map, inventory, OCR and prepare config.
    old_open=Path.open;old_stat=Path.stat;old_resolve=Path.resolve
    def deny(path):
        text=str(path).replace('\\','/').casefold()
        assert not text.startswith('z:/'), 'opaque original role was touched'
        assert path not in forbidden,'original prepare asset was touched'
    def guarded_open(path,*a,**kw):deny(path);return old_open(path,*a,**kw)
    def guarded_stat(path,*a,**kw):deny(path);return old_stat(path,*a,**kw)
    def guarded_resolve(path,*a,**kw):deny(path);return old_resolve(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded_open);monkeypatch.setattr(Path,'stat',guarded_stat);monkeypatch.setattr(Path,'resolve',guarded_resolve)
    result=execute(tmp_path,config)
    assert result['numerical_status']=='complete_diagnostic_only'
    assert not forbidden.intersection(map(Path,result['protected_hashes']))
    with np.load(Path(config['output_dir'])/'result.npz',allow_pickle=False)as z:
        assert np.all(z['road_context_count']==0)and np.all(z['road_component_count']==0)
        assert np.all(z['road_witness_uniform']==-1)


@pytest.mark.parametrize('name',['network.json','matched.json','topology_receipt.json','plan.json'])
def test_full_old_metadata_must_equal_before_context(tmp_path,monkeypatch,name):
    config,_,_=fixture(tmp_path,monkeypatch)
    root=Path(config['semantic_inputs']['prepared_master']['path']).parent
    obj=w.p.read_json(root/name);obj['private_extra_mismatch']=True;write(tmp_path,root/name,obj)
    rebind_semantics(tmp_path,config,monkeypatch)
    monkeypatch.setattr(w.context_core,'prepare',lambda *a,**kw:pytest.fail('context before fullold equality'))
    with pytest.raises(ValueError,match='metadata differs'):execute(tmp_path,config)
    assert not (Path(config['output_dir'])/'frozen_run.json').exists()


@pytest.mark.parametrize('target',['road_study','seal','semantic_array','fresh_array','catalog','sealed_extra','fresh_extra','result','output_extra'])
def test_late_input_drift_or_membership_refuses_master(tmp_path,monkeypatch,target):
    config,_,_=fixture(tmp_path,monkeypatch)
    output=Path(config['output_dir']);root=Path(config['semantic_inputs']['prepared_master']['path']).parent
    fresh=Path(config['inputs']['prepared']['path']).parent;fired=[]
    paths={'road_study':Path(config['road_speed_study_registration_ref']['path']),
        'seal':Path(config['semantic_inputs']['root_acceptance']['path']),
        'semantic_array':root/'segment_semantics.npz','fresh_array':fresh/'network.npz',
        'catalog':root/'raw_edge_catalog.jsonl','sealed_extra':root/'unexpected','fresh_extra':fresh/'unexpected',
        'result':output/'result.json','output_extra':output/'unexpected'}
    def check():
        if fired or not (output/'result.json').exists():return
        owned(tmp_path,paths[target]).write_bytes(b'private final-boundary mutation');fired.append(True)
    with pytest.raises(ValueError):execute(tmp_path,config,check=check)
    assert fired and not (output/'frozen_run.json').exists()


def test_later_unavailable_retains_computed_witness_prefix(tmp_path,monkeypatch):
    config,_,_=fixture(tmp_path,monkeypatch)
    cls=w.p.dependencies()[2]._RoadSpeedFutureGuideFilter
    original=cls.predict
    def later(self,*a,**kw):
        original(self,*a,**kw)
        if self.index==2:raise w.p.dependencies()[2].Unavailable('private failure after second prediction')
    monkeypatch.setattr(cls,'predict',later)
    result=execute(tmp_path,config)
    assert result['numerical_status']=='unavailable'
    metadata=w.p.read_json(Path(config['output_dir'])/'result.json')
    assert metadata['road_speed_completed_prediction_steps']==2
    with np.load(Path(config['output_dir'])/'result.npz',allow_pickle=False)as z:
        assert np.all(z['road_context_count'][3:]==0)
        assert np.all(z['road_witness_steps']<=2)
        assert len(z['road_witness_steps'])==len(metadata['road_speed_witness_rng_before'])


def test_early_unavailable_does_not_invent_uncomputed_witnesses(tmp_path,monkeypatch):
    config,_,_=fixture(tmp_path,monkeypatch)
    seq=w.p.dependencies()[2]
    monkeypatch.setattr(seq,'coarsen_motion',lambda *a,**kw:(_ for _ in ()).throw(seq.Unavailable('private early refusal')))
    result=execute(tmp_path,config)
    assert result['numerical_status']=='unavailable'
    with np.load(Path(config['output_dir'])/'result.npz',allow_pickle=False)as z:
        assert not any(n.startswith('road_witness_')or n=='road_context_count'for n in z.files)


def base_archive(tmp):
    root=owned(tmp,tmp/'base');root.mkdir();_,net=roads();w.p.save_network(root,net)
    return root,w.p.read_json(root/'network.json')


def test_all_zip_caps_precede_first_npy_header(tmp_path,monkeypatch):
    root,meta=base_archive(tmp_path);path=root/'network.npz'
    with np.load(path,allow_pickle=False)as z:arrays={n:z[n]for n in z.files}
    with zipfile.ZipFile(owned(tmp_path,path),'w',compression=zipfile.ZIP_DEFLATED)as archive:
        for n,a in arrays.items():
            if n=='origin':continue
            out=bytesio.BytesIO();np.save(out,a);archive.writestr(n+'.npy',out.getvalue())
        archive.writestr('origin.npy',b'0'*70000)
    monkeypatch.setattr(w.semantic_io.storage,'_npy_header',lambda *a:pytest.fail('header before last ZIP cap'))
    with pytest.raises(ValueError,match='ALL base ZIP caps'):w.semantic_io.load_base_network(path,meta,w.p.dependencies()[3])


@pytest.mark.parametrize('fault',['shape','dtype','member','trailing','typed_hash'])
def test_finite_base_loader_refuses_before_network_construction(tmp_path,monkeypatch,fault):
    root,meta=base_archive(tmp_path);path=root/'network.npz'
    with np.load(path,allow_pickle=False)as z:arrays={n:z[n]for n in z.files}
    if fault=='shape':arrays['origin']=np.zeros((1,2),np.float64)
    if fault=='dtype':arrays['edge_len']=arrays['edge_len'].astype(np.float32)
    if fault=='member':arrays['extra']=np.zeros(1)
    if fault=='typed_hash':meta['array_sha256']['origin']='f'*64
    if fault=='trailing':
        with zipfile.ZipFile(owned(tmp_path,path),'w')as archive:
            for n,a in arrays.items():
                out=bytesio.BytesIO();np.save(out,a);archive.writestr(n+'.npy',out.getvalue()+(b'x'if n=='origin'else b''))
    else:
        owned(tmp_path,path).unlink();w.p.save_npz(owned(tmp_path,path),arrays)
    monkeypatch.setattr(w.semantic_io.p,'load_network',lambda *a:pytest.fail('construction before all finite checks'))
    if fault!='typed_hash':monkeypatch.setattr(w.semantic_io.np,'frombuffer',lambda *a,**kw:pytest.fail('numeric materialization before all headers/shapes'))
    with pytest.raises(ValueError):w.semantic_io.load_base_network(path,meta,w.p.dependencies()[3])


def test_export_witness_validator_refuses_typed_prefix_count_and_rng_faults(tmp_path,monkeypatch):
    config,_,_=fixture(tmp_path,monkeypatch)
    execute(tmp_path,config);output=Path(config['output_dir'])
    result=w.p.read_json(output/'result.json')
    result['status']='complete_diagnostic_only'
    with np.load(output/'result.npz',allow_pickle=False)as z:result.update({n:z[n]for n in z.files})
    w.validate_witness(result)
    for fault in ('wrong_dtype','counts','suffix','uniform','rng','steps','missing','completion'):
        value=deepcopy(result)
        if fault=='wrong_dtype':value['road_witness_previous_edge']=value['road_witness_previous_edge'].astype(np.int64)
        if fault=='counts':value['road_component_count'][1]=50001
        if fault=='suffix':value['road_speed_epsilon_steps_computed']=value['road_speed_completed_prediction_steps']=0
        if fault=='uniform':value['road_witness_uniform'][0,0]=1.
        if fault=='rng':value['road_speed_witness_rng_before'][0]['bit_generator']='MT19937'
        if fault=='steps':value['road_witness_steps'][0]=0
        if fault=='missing':value.pop('road_witness_innovation')
        if fault=='completion':value['road_speed_completed_prediction_steps']-=1
        with pytest.raises(ValueError):w.validate_witness(value)


def test_opaque_history_duplicate_path_is_rejected_without_path_access(monkeypatch):
    monkeypatch.setattr(Path,'resolve',lambda *a,**kw:pytest.fail('opaque path resolve'))
    monkeypatch.setattr(Path,'stat',lambda *a,**kw:pytest.fail('opaque path stat'))
    with pytest.raises(ValueError,match='duplicate historical'):
        w.semantic_io.opaque_map({'Z:/old/graph.graphml':'a'*64,'z:\\old\\graph.graphml':'a'*64})
