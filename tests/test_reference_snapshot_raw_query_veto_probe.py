"""Owned R15 disk fixtures; no actual case, OCR, map, model or labels."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from scripts import reference_snapshot_raw_query_veto_probe as w
from tests import test_reference_snapshot_future_guide_probe as parent
from tests import test_raw_vpr_junction_probe as io
from tests.test_reference_snapshot_no_adaptive_q_probe import informative_native_cache, owned


def fixture(tmp, monkeypatch, bank_key="baseline", target="legacy", *, disabled=False):
    monkeypatch.setattr(parent, "w", w)
    if disabled:
        genuine = parent.case
        def case(key):
            # Change private score-generating fixture BEFORE original per-pool admission.
            from tests import test_raw_vpr_appearance as field
            original = field.case
            def low_case(*args, **kwargs):
                values = original(*args, **kwargs)
                values[5]["raw_similarities"][:] = [.03, .04, .01]
                return values
            with monkeypatch.context() as local:
                local.setattr(field, "case", low_case)
                return genuine(key)
        monkeypatch.setattr(parent, "case", case)
    config, report, qa = parent.fixture(tmp, monkeypatch, bank_key, target)
    config["schema"] = "reference-snapshot-raw-query-veto-probe-run-v1"
    ref = io.write(tmp, tmp / "veto_study.json", {"schema": "root-prospective-absolute-query-veto-snapshot-study-v1"})
    config["raw_query_veto_study_registration_ref"] = ref
    monkeypatch.setattr(w, "RAW_QUERY_VETO_STUDY_REF", ref)
    informative_native_cache(tmp, config, report, qa)
    return config, report, qa


@pytest.mark.parametrize("bank", ["baseline", "expanded"])
@pytest.mark.parametrize("target", ["legacy", "normalized-junction-v1"])
def test_four_controls_all80_events_and_original_native_q(tmp_path, monkeypatch, bank, target):
    config, report, _ = fixture(tmp_path, monkeypatch, bank, target)
    monkeypatch.setattr(w.component, "load_descriptor_rows", lambda *a: pytest.fail("descriptor reload"))
    result = io.execute(tmp_path, config)
    assert result["numerical_status"] == "complete_diagnostic_only"
    assert result["proposal_only"] is False and result["future_guide_is_proposal_only"] is True
    assert result["target_changed"] is True and result["raw_query_gate_exported"] is True
    assert result["raw_query_veto_mode"] == "absolute-max-dev-veto-v1"
    assert result["raw_query_veto_profile_sha256"] == w.veto_core.PROFILE_SHA256
    assert result["raw_query_veto_study_registration_ref"] == config["raw_query_veto_study_registration_ref"]
    assert len(config["source_hashes"]) == len(w.SOURCE_PATHS) == 33
    assert result["threshold_source_policy"]["validation_status"] == "posthoc_dev_unvalidated"
    assert result["threshold_source_policy"]["baseline_median_min"] == .162
    saved = w.admission.read_snapshot_report(w.Ledger(tmp_path, lambda: None), io.ref(Path(config["output_dir"])/"result.json"))
    assert saved["accepted_bank_metadata"] == report["banks"][0]
    assert [e["query_row"] for e in saved["appearance_events"]] == list(range(80))
    assert [e["power"] for e in saved["appearance_events"]] == saved["appearance"]["group_powers"]
    assert all(e["enabled"] is True and e["effective_power"] == e["power"] for e in saved["appearance_events"])
    assert saved["future_guide"]["target_changed"] is False  # guide remains run9, not vetoed
    assert not saved["descriptor_bodies_reloaded"] and not saved["confidence_calibrated"]
    assert result["accuracy_admitted"] is False and result["selected_answer"] is None
    with np.load(Path(config["output_dir"])/"result.npz", allow_pickle=False) as z:
        assert z["raw_query_enabled"].dtype == np.bool_ and z["raw_query_enabled"].all()
        assert z["raw_query_maxima"].dtype == np.float64 and z["raw_query_maxima"].shape == (80,)
        assert np.any(z["native_heading_var"] > 0) and np.any(z["motion_status"] == "unknown")
        assert "effective_native_heading_var" not in z and "resampled_after_step" not in z


def test_disabled_targets_still_keep_all80_events_and_ungated_appearance(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch, disabled=True)
    result = io.execute(tmp_path, config)
    assert result["numerical_status"] == "complete_diagnostic_only"
    saved = w.admission.read_snapshot_report(w.Ledger(tmp_path,lambda:None), io.ref(Path(config["output_dir"])/"result.json"))
    assert saved["raw_query_veto_receipt"]["disabled_count"] == 80
    assert len(saved["appearance_events"]) == 80
    assert all(e["enabled"] is False and e["effective_power"] == 0. and e["power"] > 0 for e in saved["appearance_events"])
    assert saved["future_guide"]["target_changed"] is False
    with np.load(Path(config["output_dir"])/"result.npz",allow_pickle=False) as z:
        assert not z["raw_query_enabled"].any() and np.all(z["raw_query_maxima"] < .162)


@pytest.mark.parametrize("fault", ["missing", "ref", "schema", "DATAref", "heading_ref", "future_ref", "enum", "knob", "helper_source", "policy_source", "helper_prepare", "policy_prepare"])
def test_four_declarations_and_sources_before_any_values(tmp_path, monkeypatch, fault):
    config, _, _ = fixture(tmp_path, monkeypatch)
    key = "raw_query_veto_study_registration_ref"
    if fault == "missing": config.pop(key)
    if fault == "ref": config[key] = {**config[key], "sha256": "f"*64}
    if fault == "schema":
        config[key] = io.write(tmp_path, config[key]["path"], {"schema": "old-study"})
        monkeypatch.setattr(w, "RAW_QUERY_VETO_STUDY_REF", config[key])
    for name, refkey in (("DATAref", "study_registration_ref"), ("heading_ref", "heading_study_registration_ref"), ("future_ref", "future_guide_study_registration_ref")):
        if fault == name: config[refkey] = {**config[refkey], "sha256": "f"*64}
    if fault == "enum": config["heading_target"] = "normalized"
    if fault == "knob": config["raw_query_threshold"] = .1
    for name, source in (("helper", "monocular_osm/raw_query_veto.py"), ("policy", "monocular_osm/assisted_policy.py")):
        if fault == name+"_source": config["source_hashes"][source] = "f"*64
        if fault == name+"_prepare":
            ref = config["inputs"]["prepared"]
            prepared = w.p.read_json(owned(tmp_path, ref["path"]))
            prepared["protected_hashes"].pop(str((w.ROOT/source).resolve()))
            config["inputs"]["prepared"] = io.write(tmp_path,ref["path"],prepared)
    monkeypatch.setattr(w.p,"load_npz",lambda *a:pytest.fail("NPZ before study/source"))
    monkeypatch.setattr(w,"load_bank",lambda *a:pytest.fail("bank before study/source"))
    with pytest.raises((ValueError,KeyError)): io.execute(tmp_path,config)
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("key,value", [("baseline_median_min",.163),("policy_version","validated"),("validation_status","validated")])
def test_policy_threshold_and_posthoc_provenance_before_arrays(tmp_path,monkeypatch,key,value):
    config,_,_=fixture(tmp_path,monkeypatch)
    policy=w.policy.policy_config();policy[key]=value
    monkeypatch.setattr(w.policy,"policy_config",lambda:deepcopy(policy))
    monkeypatch.setattr(w.p,"load_npz",lambda *a:pytest.fail("arrays before policy provenance"))
    monkeypatch.setattr(w,"load_bank",lambda *a:pytest.fail("bank before policy provenance"))
    with pytest.raises(ValueError,match="provenance"):io.execute(tmp_path,config)
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("kind",["declaration","output","membership"])
def test_final_callback_mutation_refuses_complete_master(tmp_path,monkeypatch,kind):
    config,_,_=fixture(tmp_path,monkeypatch);output=Path(config["output_dir"]);fired=[]
    def check():
        if fired or not (output/"result.json").exists():return
        path=(config["raw_query_veto_study_registration_ref"]["path"] if kind=="declaration" else output/("result.npz" if kind=="output" else "extra.txt"))
        owned(tmp_path,path).write_bytes(b"private late callback mutation");fired.append(True)
    with pytest.raises(ValueError):io.execute(tmp_path,config,check=check)
    assert fired and not (output/"frozen_run.json").exists()


def diagnostic_case():
    raw=np.full((80,3),.15,np.float32);raw[::2,1]=.2
    enabled,maxima,receipt=w.veto_core.build_gate(raw)
    powers=[.25]*80
    result={"status":"complete_diagnostic_only","raw_query_enabled":enabled,"raw_query_maxima":maxima,
        "raw_query_veto_receipt":receipt,"appearance":{"group_powers":powers},
        "appearance_events":[{"query_row":q,"power":powers[q],"enabled":bool(enabled[q]),"effective_power":powers[q] if enabled[q] else 0.} for q in range(80)]}
    return result,{"raw_similarities":raw}


@pytest.mark.parametrize("fault",[None,"enabled_type","enabled_value","maxima_type","maxima_value","receipt","event_missing","event_row","event_enabled_type","event_enabled","original_power","effective_power"])
def test_complete_gate_and_all80_effective_powers_exact(fault):
    result,bank=diagnostic_case()
    if fault=="enabled_type":result["raw_query_enabled"]=result["raw_query_enabled"].astype(np.int32)
    if fault=="enabled_value":result["raw_query_enabled"][0]=False
    if fault=="maxima_type":result["raw_query_maxima"]=result["raw_query_maxima"].astype(np.float32)
    if fault=="maxima_value":result["raw_query_maxima"][79]+=.001
    if fault=="receipt":result["raw_query_veto_receipt"]["enabled_count"]+=1
    if fault=="event_missing":result["appearance_events"].pop()
    if fault=="event_row":result["appearance_events"][0]["query_row"]=1
    if fault=="event_enabled_type":result["appearance_events"][0]["enabled"]=1
    if fault=="event_enabled":result["appearance_events"][0]["enabled"]=False
    if fault=="original_power":result["appearance_events"][0]["power"]=.5
    if fault=="effective_power":result["appearance_events"][1]["effective_power"]=.25
    if fault is None:assert w.veto_diagnostics(result,bank) is True
    else:
        with pytest.raises(ValueError):w.veto_diagnostics(result,bank)


def test_unavailable_does_not_invent_gate_diagnostics():
    assert w.veto_diagnostics({"status":"unavailable"},{}) is False
    with pytest.raises(ValueError):w.veto_diagnostics({"status":"unavailable","raw_query_enabled":np.ones(80,bool)},{})


def test_explicit_later_unavailable_keeps_only_genuinely_computed_gate(tmp_path,monkeypatch):
    config,_,_=fixture(tmp_path,monkeypatch)
    def refuse(*a,**k):raise ValueError("private whole future field unavailable")
    monkeypatch.setattr(w.future_core,"build",refuse)
    result=io.execute(tmp_path,config)
    assert result["numerical_status"]=="unavailable" and result["raw_query_gate_exported"] is True
    saved=w.admission.read_snapshot_report(w.Ledger(tmp_path,lambda:None),io.ref(Path(config["output_dir"])/"result.json"))
    assert saved["route_latlon"] is None and "appearance_events" not in saved
    with np.load(Path(config["output_dir"])/"result.npz",allow_pickle=False) as z:
        assert z["raw_query_enabled"].all() and z["raw_query_maxima"].shape==(80,)


@pytest.mark.parametrize("target",["legacy","normalized-junction-v1"])
def test_worker_all_enabled_is_exact_run9_array_noop(tmp_path,monkeypatch,target):
    from monocular_osm import junction_sequence as sequence
    config,_,_=fixture(tmp_path,monkeypatch,target=target)
    newname=("align_with_snapshot_future_guide_raw_query_veto" if target=="legacy" else
             "align_with_snapshot_normalized_heading_future_guide_raw_query_veto")
    old=(sequence.align_with_snapshot_future_guide if target=="legacy" else
         sequence.align_with_snapshot_normalized_heading_future_guide)
    genuine=getattr(sequence,newname);observed=[]
    def compare(*a,**k):
        reference=old(*a,**k);actual=genuine(*a,**k)
        assert reference["status"]==actual["status"]=="complete_diagnostic_only"
        for name,value in reference.items():
            if isinstance(value,np.ndarray):np.testing.assert_array_equal(actual[name],value)
        for name in ("appearance","future_guide","cues"):assert actual[name]==reference[name]
        for before,after in zip(reference["appearance_events"],actual["appearance_events"]):
            assert {key:after[key] for key in before}==before
        observed.append(True);return actual
    monkeypatch.setattr(sequence,newname,compare)
    result=io.execute(tmp_path,config)
    assert observed and result["numerical_status"]=="complete_diagnostic_only"
