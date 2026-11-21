"""Private snapshot/Root-wire fixtures only; no real cases, NN or labels."""
from copy import deepcopy
from pathlib import Path
import io
import zipfile
import numpy as np
import pytest
from scripts import reference_snapshot_junction_probe_v2 as w
from tests import test_raw_vpr_junction_probe as old
from tests import test_raw_vpr_appearance as field_old
from tests.test_raw_vpr_bank import fixture as bank_fixture, reseal
from tests.test_admit_reference_snapshots import study


def case(versions=False):
    net, times, odom, status, motion, control, facts = field_old.case()
    keys = ("baseline/a", "baseline/b") if versions else ("baseline/a",)
    common, records, _, regions = bank_fixture(keys=keys, names={k: control["ref_ids"].tolist() for k in keys})
    common["query"]["times"] = control["query_times"].tolist()
    for record in records:
        region = regions[record["key"]]
        z, meta = region["arrays"], region["audit"]
        for name in ("query_times", "ref_latlon", "raw_similarities"): z[name] = control[name].copy()
        if versions and record["key"] == "baseline/b":
            region["descriptor_rows"][0, 1] = np.float32(1e-5)  # Distinct protected version, no equivalence bound.
            z["raw_similarities"][:, 0] += np.float32(.005)
            meta["ref_embeddings_sha256"] = w.bank_core.strict.tensor_sha(region["descriptor_rows"])
        z["decoded_cosines"] = z["raw_similarities"][:, 0].copy()
        z["returned_cosines"] = z["decoded_cosines"].astype(np.float64)
        z["returned_latlon"] = z["ref_latlon"][z["decoded_ref_indices"]]
        meta["comparison_identity"]["query_times_sha256"] = w.native.typed(z["query_times"])
        meta["comparison_identity_sha256"] = w.bank_core.strict.legacy.json_sha(meta["comparison_identity"])
        reseal(record, region)
    original = facts["all_physical_calls_and_aliases"][0]
    facts["bank_membership"]["baseline"] = list(keys)
    facts["all_physical_calls_and_aliases"] = [{**deepcopy(original), **deepcopy(r)} for r in records]
    evidence = w.bank_core.admit_bank("baseline", records, common, facts, lambda r: regions[r["key"]])
    assert evidence["status"] == "complete_snapshot_research_evidence", evidence
    return net, times, odom, status, motion, evidence, facts


def fixture(tmp, monkeypatch, *, versions=False):
    monkeypatch.setattr(old, "w", w)
    monkeypatch.setattr(old, "appearance_case", lambda: case(versions))
    config, report, qa = old.fixture(tmp, monkeypatch)
    registration, contract = study(tmp, monkeypatch)
    config.update(schema="reference-snapshot-junction-probe-run-v1", study_registration_ref=registration)
    report.update(schema="root-native-accepted-reference-snapshots-v1", study_registration_ref=registration,
        declaration_ref=contract["declaration_ref"], descriptor_equivalence_assumed=False,
        historical_encoder_verified=False, old_strict_or_numeric_status_upgraded=False)
    qa.update(schema="root-closed-reference-snapshot-evidence-acceptance-v1",
        status="ACCEPTED_COMPLETE_SNAPSHOT_EVIDENCE_OR_EXPLICIT_UNAVAILABLE_NO_POLICY",
        study_registration_ref=registration, descriptor_equivalence_assumed=False,
        historical_encoder_verified=False, old_strict_or_numeric_status_upgraded=False)
    closure = w.p.read_json(qa["external_closure_ref"]["path"])
    closure["schema"] = "root-reference-snapshot-evidence-external-closure-v1"
    qa["external_closure_ref"] = old.write(tmp, qa["external_closure_ref"]["path"], closure)
    old.rebind(tmp, config, report, qa)
    return config, report, qa


@pytest.mark.parametrize("versions", [False, True])
def test_fixed_worker_keeps_all_snapshots_and_no_equivalence_claim(tmp_path, monkeypatch, versions):
    config, report, _ = fixture(tmp_path, monkeypatch, versions=versions)
    monkeypatch.setattr(w.component, "load_descriptor_rows", lambda *x: pytest.fail("descriptor reload"))
    result = old.execute(tmp_path, config)
    assert result["schema"] == "reference-snapshot-junction-probe-run-freeze-v1"
    assert result["numerical_status"] == "complete_diagnostic_only"
    assert result["study_registration_ref"] == config["study_registration_ref"]
    assert all(result[k] is False for k in ("descriptor_equivalence_assumed", "historical_encoder_verified", "old_strict_or_numeric_status_upgraded", "accuracy_admitted"))
    saved = w.admission.read_snapshot_report(w.Ledger(tmp_path, lambda: None), old.ref(Path(config["output_dir"]) / "result.json"))
    assert saved["accepted_bank_metadata"] == report["banks"][0]
    assert saved["accepted_bank_metadata"]["snapshot_count"] == (4 if versions else 3)
    assert saved["descriptor_bodies_reloaded"] is False
    assert [r["query_row"] for r in saved["appearance_events"]] == list(range(80))
    assert result["config"]["particles"] == 50000 and result["config"]["junction_prior"] == "conditional-uniform-v3"


@pytest.mark.parametrize("fault", ["study", "qa_study", "strict_qa", "numeric_report", "equivalence", "history", "upgrade", "unavailable"])
def test_separate_study_root_contract_before_npz(tmp_path, monkeypatch, fault):
    config, report, qa = fixture(tmp_path, monkeypatch)
    if fault == "study": config["study_registration_ref"] = {**config["study_registration_ref"], "sha256": "f" * 64}
    if fault == "qa_study": qa["study_registration_ref"] = {**qa["study_registration_ref"], "sha256": "f" * 64}
    if fault == "strict_qa": qa["schema"] = "root-closed-raw-vpr-bank-acceptance-v1"
    if fault == "numeric_report": report["schema"] = "root-native-accepted-compatible-raw-vpr-banks-v1"
    if fault == "equivalence": qa["descriptor_equivalence_assumed"] = True
    if fault == "history": report["historical_encoder_verified"] = True
    if fault == "upgrade": report["banks"][0]["old_strict_or_numeric_status_upgraded"] = True
    if fault == "unavailable": config["bank_key"] = "expanded"
    old.rebind(tmp_path, config, report, qa)
    monkeypatch.setattr(w, "snapshot_arrays_bound", lambda *x: pytest.fail("snapshot body before authority"))
    old.forbid_values(monkeypatch)
    with pytest.raises(ValueError): old.execute(tmp_path, config)
    assert not Path(config["output_dir"]).exists()


def arrays(n=2):
    return {"provider_ids": np.array(["mapillary:0"] * n), "ref_latlon": np.zeros((n, 2), np.float64),
        "raw_similarities": np.zeros((80, n), np.float32), "snapshot_weights": np.full(n, 1. / n, np.float64),
        "query_indices": np.arange(80, dtype=np.int64), "query_times": np.arange(80, dtype=np.float64)}


@pytest.mark.parametrize("fault", ["extra", "missing", "object", "F64_scores", "overshape", "trailing"])
def test_all_snapshot_headers_before_first_array_body(tmp_path, monkeypatch, fault):
    values = arrays()
    if fault == "extra": values["extra"] = np.zeros(1)
    if fault == "missing": values.pop("snapshot_weights")
    if fault == "object": values["provider_ids"] = np.array([{}], dtype=object)
    if fault == "F64_scores": values["raw_similarities"] = values["raw_similarities"].astype(np.float64)
    if fault == "overshape": values["query_times"] = np.zeros(81, np.float64)
    path = tmp_path / "snapshot.npz"
    with zipfile.ZipFile(path, "w") as z:
        for name, value in values.items():
            stream = io.BytesIO(); np.save(stream, value, allow_pickle=True)
            body = stream.getvalue() + (b"trailing" if fault == "trailing" and name == "query_times" else b"")
            z.writestr(name + ".npy", body)
    monkeypatch.setattr(np, "load", lambda *x, **k: pytest.fail("array body before all headers"))
    with pytest.raises(ValueError): w.snapshot_arrays_bound(path, w.native.sha(path))


def test128mib_archive_refuses_before_hash_or_body(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.npz"
    with path.open("wb") as f: f.truncate((128 << 20) + 1)
    monkeypatch.setattr(w.native, "sha", lambda *x: pytest.fail("hash after archive cap"))
    monkeypatch.setattr(np, "load", lambda *x, **k: pytest.fail("body after archive cap"))
    with pytest.raises(ValueError, match="128MiB"): w.snapshot_arrays_bound(path, "f" * 64)


def test_snapshot_payload_crc_is_checked_after_header_admission(tmp_path):
    import struct
    path = tmp_path / "snapshot.npz"
    np.savez(path, **arrays())
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("raw_similarities.npy")
    body = bytearray(path.read_bytes())
    name_bytes, extra_bytes = struct.unpack_from("<HH", body, info.header_offset + 26)
    start = info.header_offset + 30 + name_bytes + extra_bytes
    body[start + info.compress_size - 1] ^= 1
    path.write_bytes(body)
    with pytest.raises(zipfile.BadZipFile):
        w.snapshot_arrays_bound(path, w.native.sha(path))


def test_only_snapshot_derived_archive_can_exceed_original64mib(tmp_path):
    # Derived-header/body ABI fixture only, not an invented Root bank receipt.
    value = arrays(90000)
    value["provider_ids"] = value["provider_ids"].astype("U128")
    path = tmp_path / "private_stored_snapshot.npz"
    np.savez(path, **value)
    assert 64 << 20 < path.stat().st_size < 128 << 20
    loaded = w.snapshot_arrays_bound(path, w.native.sha(path))
    assert all(np.array_equal(loaded[k], v) for k, v in value.items())
    with pytest.raises(ValueError): w.component._arrays_bound(path, {path: w.native.sha(path)}, w.UNION_SPECS)


def test_last_callback_study_mutation_refuses_publication(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch)
    output = Path(config["output_dir"]); fired = []
    def check():
        if fired or not (output / "result.json").exists(): return
        path = Path(config["study_registration_ref"]["path"])
        assert path.resolve().is_relative_to(tmp_path.resolve())
        path.write_bytes(b"private late study mutation"); fired.append(True)
    with pytest.raises(ValueError): old.execute(tmp_path, config, check=check)
    assert fired and not (output / "frozen_run.json").exists()


def test_all_zip_declared_members_bound_before_first_header_reader(tmp_path, monkeypatch):
    path = tmp_path / "private_compressed_oversize_header.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in arrays().items():
            stream = io.BytesIO(); np.save(stream, value, allow_pickle=False)
            body = b"x" * 70000 if name == "query_times" else stream.getvalue()
            archive.writestr(name + ".npy", body)
    assert path.stat().st_size < 4096
    monkeypatch.setattr(np.lib.format, "read_magic", lambda *x: pytest.fail("header reader before all declaration caps"))
    with pytest.raises(ValueError, match="before header read"):
        w.snapshot_arrays_bound(path, w.native.sha(path))
