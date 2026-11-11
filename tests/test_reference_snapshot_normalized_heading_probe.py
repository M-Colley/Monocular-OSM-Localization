"""Owned snapshot/heading-study fixtures; no actual data or labels."""
from pathlib import Path
import pytest
from scripts import reference_snapshot_normalized_heading_probe as w
from tests import test_reference_snapshot_junction_probe_v2 as old


def fixture(tmp, monkeypatch, *, versions=False):
    monkeypatch.setattr(old, "w", w)
    config, report, qa = old.fixture(tmp, monkeypatch, versions=versions)
    config["schema"] = "reference-snapshot-normalized-heading-probe-run-v1"
    target = old.old.write(tmp, tmp / "heading_study.json", {
        "schema": "root-prospective-normalized-junction-heading-snapshot-study-v1"})
    config["heading_study_registration_ref"] = target
    monkeypatch.setattr(w, "HEADING_STUDY_REF", target)
    return config, report, qa


@pytest.mark.parametrize("versions", [False, True])
def test_genuine_fixed_worker_keeps_data_clock_and_approximation_caveat(tmp_path, monkeypatch, versions):
    config, report, _ = fixture(tmp_path, monkeypatch, versions=versions)
    monkeypatch.setattr(w.component, "load_descriptor_rows", lambda *x: pytest.fail("descriptor reload"))
    result = old.old.execute(tmp_path, config)
    assert result["schema"] == "reference-snapshot-normalized-heading-probe-run-freeze-v1"
    assert result["numerical_status"] == "complete_diagnostic_only"
    assert result["heading_study_registration_ref"] == config["heading_study_registration_ref"]
    assert result["study_registration_ref"] == config["study_registration_ref"]
    assert result["heading_mode"] == "normalized-junction-v1"
    assert result["heading_profile_sha256"] == w.heading_core.PROFILE_SHA256
    assert result["exact_bias_posterior_moments"] is result["exact_marginalized_arc_convolution"] is False
    saved = w.admission.read_snapshot_report(w.Ledger(tmp_path, lambda: None), old.old.ref(Path(config["output_dir"]) / "result.json"))
    assert saved["accepted_bank_metadata"] == report["banks"][0]
    assert saved["descriptor_bodies_reloaded"] is False and saved["new_neural_forwards"] is False
    assert [r["query_row"] for r in saved["appearance_events"]] == list(range(80))
    assert result["config"]["particles"] == 50000 and result["config"]["seed"] == 0
    assert result["accuracy_admitted"] is False and result["selected_answer"] is None


@pytest.mark.parametrize("fault", ["missing", "ref", "schema", "DATAref", "helper_source", "new_knob", "old_prepare"])
def test_target_and_source_gates_before_any_numeric_member(tmp_path, monkeypatch, fault):
    config, _, _ = fixture(tmp_path, monkeypatch)
    if fault == "missing": config.pop("heading_study_registration_ref")
    if fault == "ref": config["heading_study_registration_ref"] = {**config["heading_study_registration_ref"], "sha256": "f"*64}
    if fault == "schema":
        ref = old.old.write(tmp_path, config["heading_study_registration_ref"]["path"], {"schema": "old-target"})
        config["heading_study_registration_ref"] = ref
        monkeypatch.setattr(w, "HEADING_STUDY_REF", ref)
    if fault == "DATAref": config["study_registration_ref"] = {**config["study_registration_ref"], "sha256": "f"*64}
    if fault == "helper_source": config["source_hashes"]["monocular_osm/junction_heading_likelihood.py"] = "f"*64
    if fault == "new_knob": config["heading_sigma_deg"] = 8.
    if fault == "old_prepare":
        ref = config["inputs"]["prepared"]; prepared = w.p.read_json(ref["path"])
        prepared["protected_hashes"].pop(str((w.ROOT / "monocular_osm/junction_heading_likelihood.py").resolve()))
        config["inputs"]["prepared"] = old.old.write(tmp_path, ref["path"], prepared)
    monkeypatch.setattr(w.p, "load_npz", lambda *x: pytest.fail("numeric before target/source gates"))
    monkeypatch.setattr(w, "load_bank", lambda *x: pytest.fail("snapshot before target/source gates"))
    with pytest.raises(ValueError): old.old.execute(tmp_path, config)
    assert not Path(config["output_dir"]).exists()


def test_final_target_declaration_mutation_refuses_master(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch)
    output = Path(config["output_dir"]); fired = []
    def check():
        if fired or not (output / "result.json").exists(): return
        path = Path(config["heading_study_registration_ref"]["path"])
        assert path.resolve().is_relative_to(tmp_path.resolve())
        path.write_bytes(b"private late target declaration mutation"); fired.append(True)
    with pytest.raises(ValueError): old.old.execute(tmp_path, config, check=check)
    assert fired and not (output / "frozen_run.json").exists()
