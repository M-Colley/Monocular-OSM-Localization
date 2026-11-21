"""Private four-control worker fixtures; no actual cases, labels or NN."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from scripts import reference_snapshot_high_particle_probe as w
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
    config.update(schema='reference-snapshot-high-particle-probe-run-v1',bank_key=bank_key,heading_target=target)
    report['banks'][1]['bank_key']='baseline' if bank_key=='expanded' else 'expanded'
    qa['banks'][1]['bank_key']=report['banks'][1]['bank_key']
    root=io.write(tmp,tmp/'future_study.json',{'schema':'root-prospective-snapshot-future-appearance-proposal-study-v1'})
    config['future_guide_study_registration_ref']=root
    monkeypatch.setattr(w,'FUTURE_STUDY_REF',root)
    adaptive=io.write(tmp,tmp/'high_particle_study.json',{'schema':'root-prospective-high-particle-final-summary-snapshot-study-v1'})
    config['high_particle_study_registration_ref']=adaptive
    monkeypatch.setattr(w,'HIGH_PARTICLE_STUDY_REF',adaptive)
    io.rebind(tmp,config,report,qa)
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
    assert result['resampling_mode']=='target-ess-half-v1'
    assert result['high_particle_budget_profile_sha256']==w.budget_core.PROFILE_SHA256
    assert result['high_particle_study_registration_ref']==config['high_particle_study_registration_ref']
    assert result['resampling_diagnostics_exported'] is True and result['native_heading_variance_unchanged'] is True
    assert result['particle_budget_mode']=='fixed-500k-final-summary-v1'
    assert result['sparse_final_summary_exported'] is True
    assert result['config']==w.asdict(w.budget_core.HighParticleConfig())
    with np.load(Path(config['output_dir'])/'result.npz',allow_pickle=False) as arrays:
        flags,counts=arrays['resampled_after_step'],arrays['initial_ancestor_count']
        assert flags.dtype==np.bool_ and counts.dtype==np.int32 and flags.shape==counts.shape==arrays['times'].shape
        assert not flags[-1] and counts[0]==500_000 and np.all(counts[1:]<=counts[:-1])
        np.testing.assert_array_equal(counts[1:][~flags[:-1]],counts[:-1][~flags[:-1]])
        np.testing.assert_array_equal(flags[:-1],arrays['ess'][:-1]<250_000.)
        assert np.any(arrays['native_heading_var']>0) and np.any(arrays['motion_status']=='unknown')
        assert 'effective_native_heading_var' not in arrays  # this is not the Q ablation
        assert arrays['summary_step_indices'].dtype==np.int32
        np.testing.assert_array_equal(arrays['summary_step_indices'],[len(arrays['times'])-1])
        assert arrays['mode_xy'].shape==(1,2) and arrays['n_modes'].shape==(1,)
        assert arrays['mode_mass_uncalibrated'].shape==arrays['mode_metres_per_vo_unit'].shape==(1,)
        assert arrays['ess'].shape==arrays['route_edge'].shape==arrays['route_offset_m'].shape==arrays['times'].shape
        assert saved['genealogy_payload_bytes']==16*500_000*len(arrays['times'])


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
        ref=config['inputs']['prepared'];prepared=w.p.read_json(ref['path'])
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


@pytest.mark.parametrize('fault',['missing','ref','schema','helper_source','old_prepare','new_knob','old_run12'])
def test_high_particle_declaration_and_source_before_any_numeric_member(tmp_path,monkeypatch,fault):
    config,_,_=fixture(tmp_path,monkeypatch);key='high_particle_study_registration_ref'
    if fault=='missing':config.pop(key)
    if fault=='ref':config[key]={**config[key],'sha256':'f'*64}
    if fault=='schema':
        config[key]=io.write(tmp_path,config[key]['path'],{'schema':'old-resampling-study'})
        monkeypatch.setattr(w,'HIGH_PARTICLE_STUDY_REF',config[key])
    if fault=='helper_source':config['source_hashes']['monocular_osm/high_particle_budget.py']='f'*64
    if fault=='old_prepare':
        ref=config['inputs']['prepared'];prepared=w.p.read_json(owned(tmp_path,ref['path']))
        prepared['protected_hashes'].pop(str((w.ROOT/'monocular_osm/high_particle_budget.py').resolve()))
        config['inputs']['prepared']=io.write(tmp_path,ref['path'],prepared)
    if fault=='new_knob':config['resample_ess_fraction']=.6
    if fault=='old_run12':config[key]={
        'path':'E:/monocular-osm-joint-vpr-proof/run12/adaptive_resampling_study_contract.json',
        'sha256':'e38f59c9c682955a2de4e5a3874860bb00b710b7979333b4df7323ffe61fa990'}
    monkeypatch.setattr(w.p,'load_npz',lambda *a:pytest.fail('NPZ before fourth declaration/source'))
    monkeypatch.setattr(w,'load_bank',lambda *a:pytest.fail('bank before fourth declaration/source'))
    with pytest.raises((ValueError,KeyError)):io.execute(tmp_path,config)
    assert not Path(config['output_dir']).exists()


def test_final_high_particle_declaration_mutation_refuses_master(tmp_path,monkeypatch):
    config,_,_=fixture(tmp_path,monkeypatch);output=Path(config['output_dir']);fired=[]
    def check():
        if fired or not (output/'result.json').exists():return
        path=owned(tmp_path,config['high_particle_study_registration_ref']['path'])
        path.write_bytes(b'private late high-particle declaration mutation');fired.append(True)
    with pytest.raises(ValueError):io.execute(tmp_path,config,check=check)
    assert fired and not (output/'frozen_run.json').exists()


@pytest.mark.parametrize('fault',[None,'flag_type','count_type','final_resample','count_increase','skip_loss',
    'zero_count','wrong_start','short','bad_clock'])
def test_typed_complete_population_diagnostics(fault):
    result={'status':'complete_diagnostic_only','times':np.array([0.,.5,1.]),
        'resampled_after_step':np.array([True,False,False]),'initial_ancestor_count':np.array([3,2,2],np.int32)}
    if fault=='flag_type':result['resampled_after_step']=result['resampled_after_step'].astype(np.int32)
    if fault=='count_type':result['initial_ancestor_count']=result['initial_ancestor_count'].astype(np.float64)
    if fault=='final_resample':result['resampled_after_step'][-1]=True
    if fault=='count_increase':result['initial_ancestor_count'][-1]=3
    if fault=='skip_loss':result['initial_ancestor_count'][-1]=1
    if fault=='zero_count':result['initial_ancestor_count'][-1]=0
    if fault=='wrong_start':result['initial_ancestor_count'][0]=2
    if fault=='short':result['initial_ancestor_count']=result['initial_ancestor_count'][:-1]
    if fault=='bad_clock':result['times'][1]=result['times'][0]
    if fault is None:assert w.resampling_diagnostics(result,3) is True
    else:
        with pytest.raises(ValueError):w.resampling_diagnostics(result,3)


def test_unavailable_does_not_invent_population_diagnostics():
    assert w.resampling_diagnostics({'status':'unavailable'},3) is False
    with pytest.raises(ValueError):w.resampling_diagnostics({'status':'unavailable','initial_ancestor_count':np.array([3])},3)


@pytest.mark.parametrize('fault',[None,'indices_type','wrong_endpoint','two_summaries','xy_type',
    'nonfinite_xy','count_type','negative_count','bad_mass','bad_scale','thinned_ess','wrong_genealogy'])
def test_sparse_final_summary_is_typed_and_names_only_last_endpoint(fault):
    result={'status':'complete_diagnostic_only','times':np.array([0.,.5,1.]),
        'summary_step_indices':np.array([2],np.int32),'mode_xy':np.array([[10.,20.]]),
        'n_modes':np.array([1],np.int64),'mode_mass_uncalibrated':np.array([.8]),
        'mode_metres_per_vo_unit':np.array([1.2]),'ess':np.array([500_000.,100_000.,300_000.]),
        'genealogy_payload_bytes':16*500_000*3}
    if fault=='indices_type':result['summary_step_indices']=np.array([2],np.int64)
    if fault=='wrong_endpoint':result['summary_step_indices'][0]=1
    if fault=='two_summaries':result['summary_step_indices']=np.array([1,2],np.int32)
    if fault=='xy_type':result['mode_xy']=result['mode_xy'].astype(np.float32)
    if fault=='nonfinite_xy':result['mode_xy'][0,0]=np.nan
    if fault=='count_type':result['n_modes']=np.array([1.])
    if fault=='negative_count':result['n_modes'][0]=-1
    if fault=='bad_mass':result['mode_mass_uncalibrated'][0]=1.01
    if fault=='bad_scale':result['mode_metres_per_vo_unit'][0]=0.
    if fault=='thinned_ess':result['ess']=result['ess'][-1:]
    if fault=='wrong_genealogy':result['genealogy_payload_bytes']-=1
    if fault is None:assert w.final_summary_diagnostics(result) is True
    else:
        with pytest.raises(ValueError):w.final_summary_diagnostics(result)


def test_unavailable_has_no_uncomputed_summary():
    assert w.final_summary_diagnostics({'status':'unavailable'}) is False
    with pytest.raises(ValueError):
        w.final_summary_diagnostics({'status':'unavailable','summary_step_indices':np.array([2],np.int32)})


def test_whole_control_unavailable_is_saved_without_partial_summary(tmp_path,monkeypatch):
    config,_,_=fixture(tmp_path,monkeypatch)
    _,_,sequence,_,_,_=w.p.dependencies()
    def refuse(*args,**kwargs):raise ValueError('private complete appearance unavailable')
    monkeypatch.setattr(w.appearance,'prepare_snapshot',refuse)
    result=io.execute(tmp_path,config)
    assert result['numerical_status']=='unavailable'
    assert result['sparse_final_summary_exported'] is False
    assert result['resampling_diagnostics_exported'] is False
    with np.load(Path(config['output_dir'])/'result.npz',allow_pickle=False) as arrays:
        assert 'summary_step_indices' not in arrays and 'resampled_after_step' not in arrays
        assert 'native_heading_var' in arrays and np.any(arrays['native_heading_var']>0)
