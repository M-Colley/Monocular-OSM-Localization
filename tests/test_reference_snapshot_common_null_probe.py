"""Owned common-null worker fixtures; no actual cases, video, models or GT."""
from pathlib import Path
import pytest
from scripts import reference_snapshot_common_null_probe as w
from tests import test_reference_snapshot_normalized_heading_probe as heading
from tests import test_reference_snapshot_junction_probe_v2 as snap
from tests import test_raw_vpr_junction_probe as io
from tests.test_reference_snapshot_future_guide_probe import case


def fixture(tmp, monkeypatch, bank_key="baseline", target="legacy"):
    monkeypatch.setattr(heading, "w", w)
    monkeypatch.setattr(snap, "case", lambda versions=False: case(bank_key))
    config, report, qa = heading.fixture(tmp, monkeypatch)
    config.update(schema="reference-snapshot-common-null-probe-run-v1", bank_key=bank_key,
                  heading_target=target)
    report["banks"][1]["bank_key"] = "baseline" if bank_key == "expanded" else "expanded"
    qa["banks"][1]["bank_key"] = report["banks"][1]["bank_key"]
    root = io.write(tmp, tmp / "common_null_study.json", {
        "schema": "root-prospective-common-null-snapshot-visual-target-study-v1"})
    config["common_null_study_registration_ref"] = root
    monkeypatch.setattr(w, "COMMON_NULL_STUDY_REF", root)
    io.rebind(tmp, config, report, qa)
    return config, report, qa


@pytest.mark.parametrize("bank", ["baseline", "expanded"])
@pytest.mark.parametrize("target", ["legacy", "normalized-junction-v1"])
def test_four_controls_direct_log_ratio_every_original_target_and_no_future(tmp_path, monkeypatch, bank, target):
    config, report, _ = fixture(tmp_path, monkeypatch, bank, target)
    from monocular_osm import snapshot_future_guide as future, raw_vpr_appearance as strict
    monkeypatch.setattr(future, "build", lambda *a, **k: pytest.fail("future guide in EXIT-only control"))
    monkeypatch.setattr(future, "defensive_auxiliary_guide", lambda *a, **k: pytest.fail("future auxiliary defense"))
    monkeypatch.setattr(strict.Appearance, "log_factor", lambda *a, **k: pytest.fail("old rowmax/outlier wrapper"))
    monkeypatch.setattr(w.component, "load_descriptor_rows", lambda *a, **k: pytest.fail("descriptor reload"))
    result = io.execute(tmp_path, config)
    assert result["schema"] == "reference-snapshot-common-null-probe-run-freeze-v1"
    assert result["numerical_status"] == "complete_diagnostic_only"
    assert result["common_null_study_registration_ref"] == config["common_null_study_registration_ref"]
    assert result["common_null_profile_sha256"] == w.null_core.PROFILE_SHA256
    assert result["visual_target"] == "common-null-v1" and result["direct_log_ratio_sampling"] is True
    assert result["old_row_max_or_outlier_wrapper"] is False and result["future_proposal"] is False
    assert result["heading_target"] == target
    saved = w.admission.read_snapshot_report(w.Ledger(tmp_path, lambda: None),
        io.ref(Path(config["output_dir"]) / "result.json"))
    assert saved["accepted_bank_metadata"] == report["banks"][0]
    assert [r["query_row"] for r in saved["appearance_events"]] == list(range(80))
    assert [r["power"] for r in saved["appearance_events"]] == saved["appearance"]["group_powers"]
    assert any(r["log_factor_max"] > 0 for r in saved["appearance_events"])
    assert "future_guide" not in saved and saved["descriptor_bodies_reloaded"] is False
    assert saved["historical_encoder_verified"] is False and saved["confidence_calibrated"] is False
    assert result["config"]["particles"] == 50_000 and result["config"]["seed"] == 0
    assert result["accuracy_admitted"] is False and result["selected_answer"] is None


@pytest.mark.parametrize("fault", ["missing", "ref", "schema", "heading_ref", "DATAref", "enum",
    "knob", "future_knob", "helper_source", "old_prepare", "native_vo", "Root_flag"])
def test_all_three_declarations_and_source_native_gates_before_npz(tmp_path, monkeypatch, fault):
    config, report, qa = fixture(tmp_path, monkeypatch)
    if fault == "missing": config.pop("common_null_study_registration_ref")
    if fault == "ref": config["common_null_study_registration_ref"] = {**config["common_null_study_registration_ref"], "sha256": "f" * 64}
    if fault == "schema":
        ref = io.write(tmp_path, config["common_null_study_registration_ref"]["path"], {"schema": "old-visual-target"})
        config["common_null_study_registration_ref"] = ref
        monkeypatch.setattr(w, "COMMON_NULL_STUDY_REF", ref)
    if fault == "heading_ref": config["heading_study_registration_ref"] = {**config["heading_study_registration_ref"], "sha256": "f" * 64}
    if fault == "DATAref": config["study_registration_ref"] = {**config["study_registration_ref"], "sha256": "f" * 64}
    if fault == "enum": config["heading_target"] = "normalized"
    if fault == "knob": config["null_outlier"] = .2
    if fault == "future_knob": config["future_proposal"] = False
    if fault == "helper_source": config["source_hashes"]["monocular_osm/common_null_appearance.py"] = "f" * 64
    if fault == "old_prepare":
        ref = config["inputs"]["prepared"]
        prepared = w.p.read_json(ref["path"])
        prepared["protected_hashes"].pop(str((w.ROOT / "monocular_osm/common_null_appearance.py").resolve()))
        config["inputs"]["prepared"] = io.write(tmp_path, ref["path"], prepared)
    if fault == "native_vo": config["inputs"]["native_vo"] = {**config["inputs"]["native_vo"], "sha256": "f" * 64}
    if fault == "Root_flag":
        qa["all_observed_worker_births_closed"] = False
        io.rebind(tmp_path, config, report, qa)
    monkeypatch.setattr(w.p, "load_npz", lambda *a, **k: pytest.fail("numeric before complete metadata gates"))
    monkeypatch.setattr(w, "load_bank", lambda *a, **k: pytest.fail("snapshot before complete metadata gates"))
    with pytest.raises((ValueError, KeyError)):
        io.execute(tmp_path, config)
    assert not Path(config["output_dir"]).exists()


def test_final_common_null_declaration_mutation_refuses_master(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch)
    output, fired = Path(config["output_dir"]), []
    def check():
        if fired or not (output / "result.json").exists():
            return
        path = Path(config["common_null_study_registration_ref"]["path"]).resolve()
        assert path.is_relative_to(tmp_path.resolve())
        path.write_bytes(b"private late common-null declaration mutation")
        fired.append(True)
    with pytest.raises(ValueError):
        io.execute(tmp_path, config, check=check)
    assert fired and not (output / "frozen_run.json").exists()
