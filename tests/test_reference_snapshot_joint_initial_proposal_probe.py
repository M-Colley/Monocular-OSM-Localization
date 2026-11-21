"""Private four-control worker fixtures; no actual cases, labels or NN."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from scripts import reference_snapshot_joint_initial_proposal_probe as w
from tests import test_reference_snapshot_normalized_heading_probe as heading
from tests import test_reference_snapshot_junction_probe_v2 as snap
from tests import test_raw_vpr_junction_probe as io
from tests import test_raw_vpr_appearance as field_old
from tests.test_raw_vpr_bank import fixture as bank_fixture, reseal
from tests.test_reference_snapshot_no_adaptive_q_probe import informative_native_cache, owned


def case(bank_key):
    net,times,odom,status,motion,control,facts=field_old.case()
    keys=tuple(bank_key+'/'+name for name in (('a','b') if bank_key=='expanded' else ('a',)))
    common,records,_,regions=bank_fixture(keys=keys,names={k:control['ref_ids'].tolist() for k in keys})
    common['query']['times']=control['query_times'].tolist()
    for rank,record in enumerate(records):
        region=regions[record['key']];z,meta=region['arrays'],region['audit']
        for name in ('query_times','ref_latlon','raw_similarities'):z[name]=control[name].copy()
        if rank:
            region['descriptor_rows'][0,1]=np.float32(1e-5);z['raw_similarities'][:,0]+=np.float32(.005)
            meta['ref_embeddings_sha256']=w.bank_core.strict.tensor_sha(region['descriptor_rows'])
        z['decoded_cosines']=z['raw_similarities'][:,0].copy();z['returned_cosines']=z['decoded_cosines'].astype(np.float64)
        z['returned_latlon']=z['ref_latlon'][z['decoded_ref_indices']]
        meta['comparison_identity']['query_times_sha256']=w.native.typed(z['query_times'])
        meta['comparison_identity_sha256']=w.bank_core.strict.legacy.json_sha(meta['comparison_identity'])
        reseal(record,region)
    original=facts['all_physical_calls_and_aliases'][0]
    facts['bank_membership']={bank_key:list(keys)}
    facts['all_physical_calls_and_aliases']=[{**deepcopy(original),**deepcopy(r)} for r in records]
    evidence=w.bank_core.admit_bank(bank_key,records,common,facts,lambda r:regions[r['key']])
    assert evidence['status']=='complete_snapshot_research_evidence',evidence
    return net,times,odom,status,motion,evidence,facts


def fixture(tmp,monkeypatch,bank_key='baseline',target='legacy'):
    monkeypatch.setattr(heading,'w',w)
    monkeypatch.setattr(snap,'case',lambda versions=False:case(bank_key))
    config,report,qa=heading.fixture(tmp,monkeypatch)
    config.update(schema='reference-snapshot-joint-initial-proposal-probe-run-v1',bank_key=bank_key,heading_target=target)
    report['banks'][1]['bank_key']='baseline' if bank_key=='expanded' else 'expanded'
    qa['banks'][1]['bank_key']=report['banks'][1]['bank_key']
    root=io.write(tmp,tmp/'future_study.json',{'schema':'root-prospective-snapshot-future-appearance-proposal-study-v1'})
    config['future_guide_study_registration_ref']=root
    monkeypatch.setattr(w,'FUTURE_STUDY_REF',root)
    io.rebind(tmp,config,report,qa)
    ref=io.write(tmp,tmp/'initial_study.json',{'schema':'root-prospective-joint-speed-position-initial-proposal-study-v1'})
    config['joint_initial_proposal_study_registration_ref']=ref
    monkeypatch.setattr(w,'JOINT_INITIAL_PROPOSAL_STUDY_REF',ref)
    informative_native_cache(tmp,config,report,qa)
    return config,report,qa


@pytest.mark.parametrize('bank',['baseline','expanded'])
@pytest.mark.parametrize('target',['legacy','normalized-junction-v1'])
def test_all_four_fixed_controls_keep_every_target_event_and_data(tmp_path,monkeypatch,bank,target):
    config,report,_=fixture(tmp_path,monkeypatch,bank,target)
    monkeypatch.setattr(w.component,'load_descriptor_rows',lambda *x:pytest.fail('descriptor reload'))
    result=io.execute(tmp_path,config)
    assert result['numerical_status']=='complete_diagnostic_only' and result['proposal_only'] is True
    assert result['heading_target']==target and result['future_guide_study_registration_ref']==config['future_guide_study_registration_ref']
    assert result['heading_predictive_kernel_normalized']==(target!='legacy')
    saved=w.admission.read_snapshot_report(w.Ledger(tmp_path,lambda:None),io.ref(Path(config['output_dir'])/'result.json'))
    assert saved['accepted_bank_metadata']==report['banks'][0]
    assert [r['query_row'] for r in saved['appearance_events']]==list(range(80))
    assert [r['power'] for r in saved['appearance_events']]==saved['appearance']['group_powers']
    assert saved['future_guide']['target_changed'] is False and saved['descriptor_bodies_reloaded'] is False
    assert result['accuracy_admitted'] is False and result['selected_answer'] is None
    assert result['initial_witness_exported'] is True and result['target_changed'] is False
    assert result['joint_initial_proposal_mode']=='speed-conditioned-initial-v1'
    assert result['joint_initial_proposal_profile_sha256']==w.initial_core.PROFILE_SHA256
    assert result['joint_initial_proposal_study_registration_ref']==config['joint_initial_proposal_study_registration_ref']
    assert len(config['source_hashes'])==len(w.SOURCE_PATHS)==33
    assert all('enabled'not in e and 'effective_power'not in e for e in saved['appearance_events'])
    with np.load(Path(config['output_dir'])/'result.npz',allow_pickle=False)as z:
        bins=w.initial_core.bin_indices(z['initial_particle_logk'],z['initial_particle_eps'],.01)
        np.testing.assert_array_equal(z['initial_particle_speed_bin'],bins)
        np.testing.assert_array_equal(z['initial_speed_bin_counts'],np.bincount(bins,minlength=16))
        assert z['initial_particle_cell'].shape==(50000,) and z['initial_particle_cell'].dtype==np.int32
        assert np.any(z['native_heading_var']>0) and np.any(z['motion_status']=='unknown')
        assert not {'raw_query_enabled','effective_native_heading_var','resampled_after_step'}&set(z.files)


@pytest.mark.parametrize('fault',['old_ref','future_schema','heading_ref','DATAref','enum','knob','helper_source','old_prepare'])
def test_all_studies_and_fixed_controls_gate_before_numeric_arrays(tmp_path,monkeypatch,fault):
    config,_,_=fixture(tmp_path,monkeypatch)
    if fault=='old_ref':config['future_guide_study_registration_ref']={**config['future_guide_study_registration_ref'],'sha256':'0300fcccb0fa4798374e39402bcb9f3d07f944aad6b24cda521936f29e5d77cf'}
    if fault=='future_schema':
        ref=io.write(tmp_path,config['future_guide_study_registration_ref']['path'],{'schema':'old-future-study'})
        config['future_guide_study_registration_ref']=ref;monkeypatch.setattr(w,'FUTURE_STUDY_REF',ref)
    if fault=='heading_ref':config['heading_study_registration_ref']={**config['heading_study_registration_ref'],'sha256':'f'*64}
    if fault=='DATAref':config['study_registration_ref']={**config['study_registration_ref'],'sha256':'f'*64}
    if fault=='enum':config['heading_target']='normalized'
    if fault=='knob':config['diffusion_m_per_s']=25.
    if fault=='helper_source':config['source_hashes']['monocular_osm/snapshot_future_guide.py']='f'*64
    if fault=='old_prepare':
        ref=config['inputs']['prepared'];prepared=w.p.read_json(owned(tmp_path,ref['path']))
        prepared['protected_hashes'].pop(str((w.ROOT/'monocular_osm/snapshot_future_guide.py').resolve()))
        config['inputs']['prepared']=io.write(tmp_path,ref['path'],prepared)
    monkeypatch.setattr(w.p,'load_npz',lambda *x:pytest.fail('numeric before three declarations'))
    monkeypatch.setattr(w,'load_bank',lambda *x:pytest.fail('snapshot before three declarations'))
    with pytest.raises(ValueError):io.execute(tmp_path,config)
    assert not Path(config['output_dir']).exists()


def test_final_future_declaration_mutation_refuses_complete_master(tmp_path,monkeypatch):
    config,_,_=fixture(tmp_path,monkeypatch)
    output=Path(config['output_dir']);fired=[]
    def check():
        if fired or not (output/'result.json').exists():return
        target=Path(config['future_guide_study_registration_ref']['path'])
        assert target.resolve().is_relative_to(tmp_path.resolve())
        target.write_bytes(b'private late future declaration mutation');fired.append(True)
    with pytest.raises(ValueError):io.execute(tmp_path,config,check=check)
    assert fired and not (output/'frozen_run.json').exists()


@pytest.mark.parametrize('fault',['missing','ref','schema','helper_source','helper_prepare','veto_key','knob'])
def test_new_initial_declaration_and_source_before_values(tmp_path,monkeypatch,fault):
    config,_,_=fixture(tmp_path,monkeypatch);key='joint_initial_proposal_study_registration_ref'
    if fault=='missing':config.pop(key)
    if fault=='ref':config[key]={**config[key],'sha256':'f'*64}
    if fault=='schema':
        config[key]=io.write(tmp_path,config[key]['path'],{'schema':'old-initial-study'})
        monkeypatch.setattr(w,'JOINT_INITIAL_PROPOSAL_STUDY_REF',config[key])
    if fault=='helper_source':config['source_hashes']['monocular_osm/joint_initial_proposal.py']='f'*64
    if fault=='helper_prepare':
        ref=config['inputs']['prepared'];prepared=w.p.read_json(owned(tmp_path,ref['path']))
        prepared['protected_hashes'].pop(str((w.ROOT/'monocular_osm/joint_initial_proposal.py').resolve()))
        config['inputs']['prepared']=io.write(tmp_path,ref['path'],prepared)
    if fault=='veto_key':config['raw_query_veto_study_registration_ref']=config[key]
    if fault=='knob':config['initial_speed_bins']=8
    monkeypatch.setattr(w.p,'load_npz',lambda *a:pytest.fail('numeric before fourth declaration'))
    monkeypatch.setattr(w,'load_bank',lambda *a:pytest.fail('bank before fourth declaration'))
    with pytest.raises((ValueError,KeyError)):io.execute(tmp_path,config)
    assert not Path(config['output_dir']).exists()


@pytest.mark.parametrize('kind',['declaration','output','membership'])
def test_final_new_declaration_or_output_drift_refuses_master(tmp_path,monkeypatch,kind):
    config,_,_=fixture(tmp_path,monkeypatch);output=Path(config['output_dir']);fired=[]
    def check():
        if fired or not(output/'result.json').exists():return
        target=config['joint_initial_proposal_study_registration_ref']['path']if kind=='declaration'else output/('result.npz'if kind=='output'else'extra.txt')
        owned(tmp_path,target).write_bytes(b'private late initial-proposal drift');fired.append(True)
    with pytest.raises(ValueError):io.execute(tmp_path,config,check=check)
    assert fired and not(output/'frozen_run.json').exists()


def synthetic_witness():
    from types import SimpleNamespace
    net=SimpleNamespace(edge_len=np.array([150.,250.]))
    n=50000;nc=np.array([2,3]);widths=np.repeat(net.edge_len/nc,nc)
    starts=np.concatenate([np.arange(k)*(length/k)for k,length in zip(nc,net.edge_len)])
    cells=(np.arange(n)%5).astype(np.int32);logk=np.full(n,np.log(.08));eps=np.linspace(-2.,2.,n)
    bins=w.initial_core.bin_indices(logk,eps,.01);counts=np.bincount(bins,minlength=16).astype(np.int32)
    result={'status':'complete_diagnostic_only','initial_speed_representatives_mps':np.array(w.initial_core.SPEED_REPRESENTATIVES),
        'initial_speed_log_boundaries':np.array(w.initial_core.LOG_SPEED_BOUNDARIES),'initial_speed_bin_counts':counts,
        'initial_particle_cell':cells,'initial_particle_speed_bin':bins,'initial_particle_edge':np.repeat(np.arange(2),nc)[cells].astype(np.int32),
        'initial_particle_offset_m':starts[cells]+widths[cells]/2,'initial_particle_logk':logk,'initial_particle_eps':eps,
        'initial_particle_logw':np.zeros(n),'joint_initial_proposal_receipt':{'schema':'joint-speed-position-initial-proposal-receipt-v1',
        'mode':w.INITIAL_PROPOSAL_MODE,'profile_sha256':w.initial_core.PROFILE_SHA256,'bin_counts':counts.tolist(),
        'particle_count':n,'cell_count':5,'occupied_bin_count':int(np.count_nonzero(counts))}}
    return result,net


@pytest.mark.parametrize('fault',[None,'missing','int_dtype','float_dtype','short','nonfinite','representatives','boundaries','bins','counts','cell','edge','offset','logw_upper','logw_lower','receipt','particle_count','cell_count','occupied_count'])
def test_typed_witness_bins_partition_and_weight_bounds(fault):
    result,net=synthetic_witness()
    if fault=='missing':result.pop('initial_particle_eps')
    if fault=='int_dtype':result['initial_particle_cell']=result['initial_particle_cell'].astype(np.int64)
    if fault=='float_dtype':result['initial_particle_offset_m']=result['initial_particle_offset_m'].astype(np.float32)
    if fault=='short':result['initial_particle_logw']=result['initial_particle_logw'][:-1]
    if fault=='nonfinite':result['initial_particle_eps'][0]=np.nan
    if fault=='representatives':result['initial_speed_representatives_mps'][0]+=.01
    if fault=='boundaries':result['initial_speed_log_boundaries'][0]+=.01
    if fault=='bins':result['initial_particle_speed_bin'][0]=15
    if fault=='counts':result['initial_speed_bin_counts'][0]+=1
    if fault=='cell':result['initial_particle_cell'][0]=5
    if fault=='edge':result['initial_particle_edge'][0]=1
    if fault=='offset':result['initial_particle_offset_m'][0]=-1.
    if fault=='logw_upper':result['initial_particle_logw'][0]=np.log(4.)+.01
    if fault=='logw_lower':result['initial_particle_logw'][0]=-100.
    if fault=='receipt':result['joint_initial_proposal_receipt']['profile_sha256']='f'*64
    if fault in ('particle_count','cell_count','occupied_count'):
        field={'particle_count':'particle_count','cell_count':'cell_count','occupied_count':'occupied_bin_count'}[fault]
        result['joint_initial_proposal_receipt'][field]+=1
    if fault is None:assert w.initial_diagnostics(result,net,.01) is True
    else:
        with pytest.raises(ValueError):w.initial_diagnostics(result,net,.01)


def test_partition_cap_before_cell_allocation(monkeypatch):
    result,net=synthetic_witness();net.edge_len=np.array([20_000_100.])
    monkeypatch.setattr(w.np,'repeat',lambda *a:pytest.fail('oversize partition allocated'))
    with pytest.raises(ValueError,match='partition'):w.initial_diagnostics(result,net,.01)


def test_honest_uncomputed_and_partial_refusal():
    assert w.initial_diagnostics({'status':'unavailable'},None,None) is False
    with pytest.raises(ValueError):w.initial_diagnostics({'status':'complete_diagnostic_only'},None,None)
    with pytest.raises(ValueError):w.initial_diagnostics({'status':'unavailable','initial_particle_cell':np.zeros(1,np.int32)},None,None)


@pytest.mark.parametrize('later',[False,True])
def test_unavailable_preserves_computed_initialization_only(tmp_path,monkeypatch,later):
    from monocular_osm import junction_sequence as sequence
    config,_,_=fixture(tmp_path,monkeypatch)
    if later:
        from monocular_osm.graph_filter import GraphParticleFilter
        def fail(*a,**k):raise sequence.Unavailable('private later summary refusal')
        monkeypatch.setattr(GraphParticleFilter,'summary',fail)
    else:
        def fail(*a,**k):raise ValueError('private before initialization future refusal')
        monkeypatch.setattr(w.future_core,'build',fail)
    result=io.execute(tmp_path,config)
    assert result['numerical_status']=='unavailable' and result['initial_witness_exported'] is later
    saved=w.admission.read_snapshot_report(w.Ledger(tmp_path,lambda:None),io.ref(Path(config['output_dir'])/'result.json'))
    assert saved['route_latlon'] is None and 'appearance_events'not in saved
    assert ('joint_initial_proposal_receipt'in saved) is later
    with np.load(Path(config['output_dir'])/'result.npz',allow_pickle=False)as z:
        assert (w.INITIAL_WITNESS_KEYS<=set(z.files)) is later
