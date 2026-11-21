"""Owned no-extra-Q worker fixtures; all files private and temporary."""
from pathlib import Path
import math
import numpy as np
import pytest
from scripts import reference_snapshot_no_adaptive_q_probe as w
from tests import test_reference_snapshot_future_guide_probe as parent
from tests import test_raw_vpr_junction_probe as io


def owned(tmp, path):
    result = Path(path).resolve()
    assert result.is_relative_to(tmp.resolve()), "all private mutations require tmp containment"
    return result


def informative_native_cache(tmp, config, report, qa):
    """Persist a synthetic yaw glitch + held unknown pose, then rebind authority.

    This makes original variance NONZERO and proves the composed unknown term
    survives. Every role is a synthetic fixture, never historical evidence.
    """
    path = owned(tmp, config["inputs"]["native_vo"]["path"])
    arrays = w.p.load_npz(path)
    n = len(arrays["valid"])
    missing, turn = n // 3, n // 2
    arrays["centers"][missing:, 0] -= 1.
    arrays["xz"] = arrays["centers"][:, [0, 2]].copy()
    arrays["valid"][missing] = False
    arrays["n_inliers"][missing] = 0
    angle = math.radians(8.)
    rotation = np.array([[math.cos(angle), 0., math.sin(angle)], [0., 1., 0.],
                         [-math.sin(angle), 0., math.cos(angle)]])
    arrays["rotations"][turn:] = rotation
    arrays["translations"] = -np.einsum("nij,nj->ni", arrays["rotations"], arrays["centers"])
    path = owned(tmp, path.with_name("motion_Q_fixture.npz"))
    w.p.save_npz(path, arrays)
    motion_ref = io.ref(path)
    clock_path = owned(tmp, config["inputs"]["native_clock"]["path"])
    clock = w.p.load_npz(clock_path)
    clock.update(xz=arrays["xz"].copy(), valid=arrays["valid"].copy())
    clock_path = owned(tmp, clock_path.with_name("clock_Q_fixture.npz"))
    w.p.save_npz(clock_path, clock)
    clock_ref = io.ref(clock_path)
    facts_path = owned(tmp, config["inputs"]["native_facts"]["path"])
    facts = w.p.read_json(facts_path)
    facts.update(signed_vo_cache=motion_ref, cues_vo=clock_ref)
    facts_ref = io.write(tmp, facts_path, facts)
    native_qa_path = owned(tmp, report["native_root_acceptance_ref"]["path"])
    native_qa = w.p.read_json(native_qa_path)
    native_qa["native_facts_ref"] = facts_ref
    native_qa_ref = io.write(tmp, native_qa_path, native_qa)
    config["inputs"].update(native_vo=motion_ref, native_clock=clock_ref, native_facts=facts_ref)
    report.update(native_facts_ref=facts_ref, native_root_acceptance_ref=native_qa_ref)
    qa.update(native_facts_ref=facts_ref, native_root_acceptance_ref=native_qa_ref)
    io.rebind(tmp, config, report, qa)


def fixture(tmp, monkeypatch, bank="baseline", target="legacy"):
    monkeypatch.setattr(parent, "w", w)
    config, report, qa = parent.fixture(tmp, monkeypatch, bank, target)
    config["schema"] = "reference-snapshot-no-adaptive-q-probe-run-v1"
    study = io.write(tmp, tmp / "heading_variance_study.json", {
        "schema": "root-prospective-no-adaptive-heading-Q-future-snapshot-study-v1"})
    config["heading_variance_study_registration_ref"] = study
    monkeypatch.setattr(w, "HEADING_VARIANCE_STUDY_REF", study)
    informative_native_cache(tmp, config, report, qa)
    return config, report, qa


@pytest.mark.parametrize("bank", ["baseline", "expanded"])
@pytest.mark.parametrize("target", ["legacy", "normalized-junction-v1"])
def test_four_controls_keep_original_means_and_export_three_variance_roles(tmp_path, monkeypatch, bank, target):
    config, report, _ = fixture(tmp_path, monkeypatch, bank, target)
    _, _, sequence, gf, vo, _ = w.p.dependencies()
    original, status, times = w.p.load_motion(w.p.load_npz(config["inputs"]["native_vo"]["path"]),
        w.p.load_npz(config["inputs"]["native_clock"]["path"]), "timestamps", config["video_binding"], gf, vo)
    assert np.any(original.heading_var > 0) and np.any(status == "unknown")
    monkeypatch.setattr(w.component, "load_descriptor_rows", lambda *a, **k: pytest.fail("descriptor reload"))
    result = io.execute(tmp_path, config)
    assert result["schema"] == "reference-snapshot-no-adaptive-q-probe-run-freeze-v1"
    assert result["numerical_status"] == "complete_diagnostic_only" and result["heading_target"] == target
    assert result["heading_variance_study_registration_ref"] == config["heading_variance_study_registration_ref"]
    assert result["heading_variance_profile_sha256"] == w.variance_core.PROFILE_SHA256
    assert result["proposal_only"] is False and result["future_guide_is_proposal_only"] is True
    saved = w.admission.read_snapshot_report(w.Ledger(tmp_path, lambda: None),
        io.ref(Path(config["output_dir"]) / "result.json"))
    assert saved["accepted_bank_metadata"] == report["banks"][0]
    assert [r["query_row"] for r in saved["appearance_events"]] == list(range(80))
    assert [r["power"] for r in saved["appearance_events"]] == saved["appearance"]["group_powers"]
    with np.load(Path(config["output_dir"]) / "result.npz", allow_pickle=False) as arrays:
        for name, wanted in (("native_heading_var", original.heading_var), ("native_heading", original.heading),
                             ("native_step", original.step), ("native_dt", original.dt),
                             ("motion_status", status), ("native_times", times)):
            np.testing.assert_array_equal(arrays[name], wanted)
        np.testing.assert_array_equal(arrays["effective_native_heading_var"], np.zeros(len(original)))
        np.testing.assert_array_equal(arrays["composed_heading_var"], math.radians(5.)**2 * arrays["unknown_seconds"])
        assert np.any(arrays["composed_heading_var"] > 0)
    assert saved["historical_encoder_verified"] is False and saved["confidence_calibrated"] is False
    assert result["accuracy_admitted"] is False and result["selected_answer"] is None


@pytest.mark.parametrize("fault", ["missing", "ref", "schema", "heading_ref", "DATAref", "future_ref",
    "enum", "knob", "common_null_knob", "helper_source", "old_prepare", "native_vo", "Root_flag"])
def test_all_four_studies_and_source_native_gates_precede_numeric_arrays(tmp_path, monkeypatch, fault):
    config, report, qa = fixture(tmp_path, monkeypatch)
    if fault == "missing": config.pop("heading_variance_study_registration_ref")
    if fault == "ref": config["heading_variance_study_registration_ref"] = {**config["heading_variance_study_registration_ref"], "sha256": "f"*64}
    if fault == "schema":
        ref = io.write(tmp_path, config["heading_variance_study_registration_ref"]["path"], {"schema": "old-process-target"})
        config["heading_variance_study_registration_ref"] = ref
        monkeypatch.setattr(w, "HEADING_VARIANCE_STUDY_REF", ref)
    if fault == "heading_ref": config["heading_study_registration_ref"] = {**config["heading_study_registration_ref"], "sha256": "f"*64}
    if fault == "DATAref": config["study_registration_ref"] = {**config["study_registration_ref"], "sha256": "f"*64}
    if fault == "future_ref": config["future_guide_study_registration_ref"] = {**config["future_guide_study_registration_ref"], "sha256": "f"*64}
    if fault == "enum": config["heading_target"] = "normalized"
    if fault == "knob": config["delta_rw_deg"] = 0.
    if fault == "common_null_knob": config["common_null"] = False
    if fault == "helper_source": config["source_hashes"]["monocular_osm/no_adaptive_heading_variance.py"] = "f"*64
    if fault == "old_prepare":
        ref = config["inputs"]["prepared"]
        prepared = w.p.read_json(owned(tmp_path, ref["path"]))
        prepared["protected_hashes"].pop(str((w.ROOT / "monocular_osm/no_adaptive_heading_variance.py").resolve()))
        config["inputs"]["prepared"] = io.write(tmp_path, ref["path"], prepared)
    if fault == "native_vo": config["inputs"]["native_vo"] = {**config["inputs"]["native_vo"], "sha256": "f"*64}
    if fault == "Root_flag":
        qa["all_observed_worker_births_closed"] = False
        io.rebind(tmp_path, config, report, qa)
    monkeypatch.setattr(w.p, "load_npz", lambda *a, **k: pytest.fail("numeric before all four metadata declarations"))
    monkeypatch.setattr(w, "load_bank", lambda *a, **k: pytest.fail("snapshot before all four metadata declarations"))
    with pytest.raises((ValueError, KeyError)):
        io.execute(tmp_path, config)
    assert not Path(config["output_dir"]).exists()


def test_final_heading_variance_declaration_mutation_refuses_master(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch)
    output, fired = Path(config["output_dir"]), []
    def check():
        if fired or not (output/"result.json").exists():
            return
        path = owned(tmp_path, config["heading_variance_study_registration_ref"]["path"])
        path.write_bytes(b"private late heading-variance declaration mutation")
        fired.append(True)
    with pytest.raises(ValueError):
        io.execute(tmp_path, config, check=check)
    assert fired and not (output/"frozen_run.json").exists()
