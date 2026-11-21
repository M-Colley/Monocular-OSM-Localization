"""Study authority and compatible IO only; owned synthetic caches."""
from copy import deepcopy
from pathlib import Path
import pytest
from scripts import admit_raw_vpr_compat_bank as a
from tests import test_admit_raw_vpr_bank as old


def study(tmp, monkeypatch):
    numerical = old.write(tmp / "numerical_profile.json", {"schema": "private-unchanged-fusion-profile"})
    declaration = old.write(tmp / "declaration.json", {"schema": "root-prospective-full-bank-numerical-compatibility-study-v1"})
    contract = {"schema": "root-raw-vpr-compatible-study-contract-v1", **deepcopy(a.STUDY_VALUES),
                "declaration_ref": declaration, "numerical_fusion_profile_ref": numerical}
    record = old.write(tmp / "study.json", contract)
    monkeypatch.setattr(a, "STUDY_REF", record)
    monkeypatch.setattr(a, "DECLARATION_REF", declaration)
    return record, contract


def fixture(tmp, monkeypatch):
    config, _ = old.fixture(tmp)
    config["schema"] = "raw-vpr-compatible-bank-admission-config-v1"
    config["source_hashes"] = {name: a.native.sha(a.ROOT / name) for name in a.SOURCES}
    config["study_registration_ref"], contract = study(tmp, monkeypatch)
    return config, contract


def run(tmp, config, **kwargs):
    record = old.write(tmp / "compatible_config.json", config)
    return a.execute(record["path"], record["sha256"], numerical_context=old.threads,
                     resource_gate=lambda: {}, **kwargs)


def test_genuine_receipts_and_separate_complete_status(tmp_path, monkeypatch):
    config, _ = fixture(tmp_path, monkeypatch)
    result = run(tmp_path, config)
    assert result["schema"] == "root-native-accepted-compatible-raw-vpr-banks-v1"
    assert [r["status"] for r in result["banks"]] == ["complete_compatible_research_bank"] * 2
    assert result["study_registration_ref"] == config["study_registration_ref"]
    assert result["compatibility_not_bitwise_identity"] is True and result["historical_encoder_verified"] is False
    assert result["reference_encoding_identity"] == "UNVERIFIED" and result["policy_admission"] is False
    assert all(r["maximum_descriptor_l2_from_canonical"] == 0. for r in result["banks"])
    assert all(r["original_persisted_receipt_present"] is False for r in result["descriptor_receipts"].values())
    assert a.original.PROFILE["bank_profile_sha256"] != a.PROFILE["bank_profile_sha256"]


@pytest.mark.parametrize("fault", ["ref", "schema", "declaration", "threshold", "boolean", "cap", "claims"])
def test_study_contract_precedes_any_npz(tmp_path, monkeypatch, fault):
    config, contract = fixture(tmp_path, monkeypatch)
    if fault == "ref": config["study_registration_ref"] = {**config["study_registration_ref"], "sha256": "f" * 64}
    else:
        if fault == "schema": contract["schema"] = "old-study"
        if fault == "declaration": contract["declaration_ref"] = {**contract["declaration_ref"], "sha256": "f" * 64}
        if fault == "threshold": contract["descriptor_l2_max_f64"] = 2e-6
        if fault == "boolean": contract["require_exact_coordinates"] = 1
        if fault == "cap": contract["maximum_unique_references_per_bank"] = 30001
        if fault == "claims": contract["bitwise_descriptor_identity_claimed"] = True
        record = old.write(Path(config["study_registration_ref"]["path"]), contract)
        config["study_registration_ref"] = record
        monkeypatch.setattr(a, "STUDY_REF", record)
    monkeypatch.setattr(a.native.Ledger, "members", lambda *x: pytest.fail("NPZ before study"))
    monkeypatch.setattr(a.component, "_arrays_bound", lambda *x: pytest.fail("tensor before study"))
    with pytest.raises(ValueError): run(tmp_path, config)
    assert not Path(config["output_dir"]).exists()


def test_only_derived_report_gets64mib_and_shared_counter_counts_every_read(tmp_path):
    target = tmp_path / "compatible_report.json"
    with target.open("wb") as stream:
        stream.write(b'{"private":true}')
        for _ in range(17): stream.write(b" " * (1 << 20))
    record = {"path": str(target), "sha256": a.native.sha(target)}
    with pytest.raises(ValueError, match="metadata budget"):
        a.Ledger().json(record)  # Original/request/native roles retain16MiB.
    ledger = a.Ledger()
    assert a.read_compatible_report(ledger, record) == {"private": True}
    assert a.read_compatible_report(ledger, record) == {"private": True}
    assert ledger.total_json == target.stat().st_size * 2


@pytest.mark.parametrize("fault", ["role_limit", "remaining"])
def test_compatible_report_budget_refuses_before_hash_or_body(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace
    target = tmp_path / "derived.json"
    target.write_text("{}")
    record = {"path": str(target), "sha256": a.native.sha(target)}
    ledger = a.Ledger()
    if fault == "remaining": ledger.total_json = 256 << 20
    original = Path.stat
    def stat(path, *args, **kwargs):
        return SimpleNamespace(st_size=(64 << 20) + 1) if fault == "role_limit" and path == target else original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(a.native, "sha", lambda *x: pytest.fail("hash after budget refusal"))
    with pytest.raises(ValueError, match="before body/hash"):
        a.read_compatible_report(ledger, record)


@pytest.mark.parametrize("body", ['{"a":1,"a":2}', '{"a":NaN}'])
def test_compatible_report_keeps_duplicate_nonfinite_refusal(tmp_path, body):
    target = tmp_path / "derived.json"
    target.write_text(body)
    with pytest.raises(ValueError):
        a.read_compatible_report(a.Ledger(), {"path": str(target), "sha256": a.native.sha(target)})


def test_compatible_report_rechecks_bounded_body_after_protection(tmp_path, monkeypatch):
    target = tmp_path / "derived.json"
    target.write_bytes(b'{"a":1}')
    record = {"path": str(target), "sha256": a.native.sha(target)}
    ledger = a.Ledger()
    protect = ledger.protect
    def changed(value):
        path = protect(value)
        assert path.resolve().is_relative_to(tmp_path.resolve())
        path.write_bytes(b'{"a":2}')  # Same size: expected-body hash still binds.
        return path
    monkeypatch.setattr(ledger, "protect", changed)
    with pytest.raises(ValueError, match="bounded body"):
        a.read_compatible_report(ledger, record)


def test_oversize_complete_report_refuses_instead_of_truncating_banks(tmp_path, monkeypatch):
    config, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(a, "MAX_COMPAT_REPORT_JSON", 64)
    with pytest.raises(ValueError, match="no truncation"): run(tmp_path, config)
    output = Path(config["output_dir"])
    assert len(list(output.glob("*.npz"))) == 2
    assert not (output / "bank_admission.json").exists()


def test_late_study_mutation_cannot_publish(tmp_path, monkeypatch):
    config, _ = fixture(tmp_path, monkeypatch)
    output = Path(config["output_dir"])
    fired = []
    def check():
        if fired or len(list(output.glob("*.npz"))) != 2: return
        target = Path(config["study_registration_ref"]["path"])
        assert target.resolve().is_relative_to(tmp_path.resolve())
        target.write_bytes(b"private late study mutation")
        fired.append(True)
    with pytest.raises(ValueError): run(tmp_path, config, check=check)
    assert fired and not (output / "bank_admission.json").exists()
