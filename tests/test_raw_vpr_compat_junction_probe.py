"""Synthetic compatible data product and Root-wire fixtures, never real cases."""
from pathlib import Path
import numpy as np
import pytest
from scripts import raw_vpr_compat_junction_probe as w
from tests import test_raw_vpr_junction_probe as old
from tests.test_raw_vpr_compat_appearance import case
from tests.test_admit_raw_vpr_compat_bank import study


def fixture(tmp, monkeypatch):
    # Common fixture writes a NEW preparation from ALL new sources. The bank
    # itself is freshly produced by genuine compatible.admit_bank in case().
    monkeypatch.setattr(old, "w", w)
    monkeypatch.setattr(old, "appearance_case", lambda: case()[:7])
    config, report, qa = old.fixture(tmp, monkeypatch)
    registration, contract = study(tmp, monkeypatch)
    config.update(schema="raw-vpr-compatible-junction-probe-run-v1", study_registration_ref=registration)
    report.update(schema="root-native-accepted-compatible-raw-vpr-banks-v1", study_registration_ref=registration,
                  declaration_ref=contract["declaration_ref"], compatibility_not_bitwise_identity=True,
                  historical_encoder_verified=False)
    qa.update(schema="root-closed-raw-vpr-compatible-bank-acceptance-v1",
              status="ACCEPTED_COMPLETE_COMPATIBLE_BANKS_OR_EXPLICIT_UNAVAILABLE_NO_POLICY",
              study_registration_ref=registration)
    closure = w.p.read_json(qa["external_closure_ref"]["path"])
    closure["schema"] = "root-raw-vpr-compatible-bank-external-closure-v1"
    qa["external_closure_ref"] = old.write(tmp, qa["external_closure_ref"]["path"], closure)
    old.rebind(tmp, config, report, qa)
    return config, report, qa


def test_fixed_compatible_worker_keeps_study_and_unchanged_math(tmp_path, monkeypatch):
    config, report, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(w.component, "load_descriptor_rows", lambda *x: pytest.fail("descriptor reload"))
    result = old.execute(tmp_path, config)
    assert result["schema"] == "raw-vpr-compatible-junction-probe-run-freeze-v1"
    assert result["numerical_status"] == "complete_diagnostic_only"
    assert result["study_registration_ref"] == config["study_registration_ref"]
    assert result["compatibility_not_bitwise_identity"] is True and result["accuracy_admitted"] is False
    metadata = w.p.read_json(Path(config["output_dir"]) / "result.json")
    assert metadata["accepted_bank_metadata"] == report["banks"][0]
    assert metadata["descriptor_bodies_reloaded"] is False and metadata["historical_encoder_verified"] is False
    assert [x["query_row"] for x in metadata["appearance_events"]] == list(range(80))
    assert result["config"]["particles"] == 50000 and result["config"]["junction_prior"] == "conditional-uniform-v3"


@pytest.mark.parametrize("fault", ["study", "qa_study", "strict_qa", "strict_report", "identity", "unavailable"])
def test_new_contract_and_closure_cannot_be_cast_from_old_before_npz(tmp_path, monkeypatch, fault):
    config, report, qa = fixture(tmp_path, monkeypatch)
    if fault == "study": config["study_registration_ref"] = {**config["study_registration_ref"], "sha256": "f" * 64}
    if fault == "qa_study": qa["study_registration_ref"] = {**qa["study_registration_ref"], "sha256": "f" * 64}
    if fault == "strict_qa": qa["schema"] = "root-closed-raw-vpr-bank-acceptance-v1"
    if fault == "strict_report": report["schema"] = "root-native-accepted-raw-vpr-banks-v1"
    if fault == "identity": report["banks"][0]["compatibility_not_bitwise_identity"] = False
    if fault == "unavailable": config["bank_key"] = "expanded"
    old.rebind(tmp_path, config, report, qa)
    old.forbid_values(monkeypatch)
    with pytest.raises(ValueError): old.execute(tmp_path, config)
    assert not Path(config["output_dir"]).exists()


def test_10001_column_shape_is_only_in_explicit_compatible_loader(tmp_path):
    from scripts import raw_vpr_junction_probe as strict
    n = 10001
    arrays = {"ref_ids": np.array([f"mapillary:{i}" for i in range(n)]),
              "ref_latlon": np.zeros((n, 2), np.float64), "raw_similarities": np.zeros((80, n), np.float32),
              "query_indices": np.arange(80, dtype=np.int64), "query_times": np.arange(80, dtype=np.float64)}
    archive = tmp_path / "private_union.npz"
    w.p.save_npz(archive, arrays)
    # Low-level IO shape fixture only; authority admission is tested above.
    chosen = {"status": "complete_compatible_research_bank", "mapping": [{"ref_id": v} for v in arrays["ref_ids"]],
        "array_artifact": {**old.ref(archive), "arrays": {k: {
            "shape": list(v.shape), "dtype": v.dtype.str, "sha256": w.native.typed(v)} for k, v in arrays.items()}}}
    loaded, _ = w.load_bank(chosen, w.Ledger(tmp_path, lambda: None))
    assert loaded["raw_similarities"].shape == (80, 10001)
    with pytest.raises(ValueError): strict.load_bank(chosen, strict.Ledger(tmp_path, lambda: None))


def test_final_study_bytes_are_checked_after_numerical_run(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch)
    output = Path(config["output_dir"])
    fired = []
    def check():
        if fired or not (output / "result.json").exists(): return
        path = Path(config["study_registration_ref"]["path"])
        assert path.resolve().is_relative_to(tmp_path.resolve())
        path.write_bytes(b"private late study mutation")
        fired.append(True)
    with pytest.raises(ValueError): old.execute(tmp_path, config, check=check)
    assert fired and not (output / "frozen_run.json").exists()
