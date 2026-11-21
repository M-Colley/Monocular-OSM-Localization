"""Private four-control worker fixtures; no actual cases, labels or NN."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from scripts import reference_snapshot_future_guide_probe as w
from tests import test_reference_snapshot_normalized_heading_probe as heading
from tests import test_reference_snapshot_junction_probe_v2 as snap
from tests import test_raw_vpr_junction_probe as io
from tests import test_raw_vpr_appearance as field_old
from tests.test_raw_vpr_bank import fixture as bank_fixture, reseal


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
    config.update(schema='reference-snapshot-future-guide-probe-run-v1',bank_key=bank_key,heading_target=target)
    report['banks'][1]['bank_key']='baseline' if bank_key=='expanded' else 'expanded'
    qa['banks'][1]['bank_key']=report['banks'][1]['bank_key']
    root=io.write(tmp,tmp/'future_study.json',{'schema':'root-prospective-snapshot-future-appearance-proposal-study-v1'})
    config['future_guide_study_registration_ref']=root
    monkeypatch.setattr(w,'FUTURE_STUDY_REF',root)
    io.rebind(tmp,config,report,qa)
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
