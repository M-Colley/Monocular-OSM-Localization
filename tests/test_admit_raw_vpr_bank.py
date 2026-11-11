"""Owned tiny disk fixtures only; no real input, model, native video or GT."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from scripts import admit_raw_vpr_bank as a
from tests.test_raw_vpr_bank import fixture as bank_fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": a.native.sha(path)}


def fixture(tmp):
    from monocular_osm import kartaview_vpr as kv
    comparison, _, facts, regions = bank_fixture(keys=("baseline/a", "expanded/a"))
    historical = tmp / "historical"; historical.mkdir()
    protected, bound, calls, banks, directories, cache_calls = [], [], [], [], [], {}
    for key, region in regions.items():
        kind = key.split("/")[0]
        output = historical / "output" / kind; output.mkdir(parents=True)
        cache = historical / "data" / kind / "mapillary"; cache.mkdir(parents=True)
        request = {"kind": kind, "execution_id": "private-execution", "video_sha256": "d" * 64,
            "source_sha256": {"monocular_osm/stages/prior.py": "e" * 64},
            "config": {"output_dir": "output/" + kind, "data_dir": "data/" + kind,
                "vpr_region_cache_dir": None, "vo_start_sec": 0., "vo_end_sec": 40., "frame_stride": 3,
                "max_frames": None, "frame_max_width": None, "retrieval_only": True,
                "answer": "track", "use_vpr_prior": True, "use_ipm_scale": False}}
        rr = write(output / "request.json", request)
        ids, ll = region["arrays"]["ref_ids"], region["arrays"]["ref_latlon"]
        refs = [{"id": str(ids[0]).split(":", 1)[1], "lat": float(ll[0, 0]), "lon": float(ll[0, 1]),
                 "url": "https://example.invalid/private"}]
        public_request = {"source": "mapillary", "center": [40., -80.], "radius_m": 1000., "cap": 1500,
            "model_name": "megaloc", "device": "cpu", "query_mode": "warp", "query_batch_size": 1,
            "cache_dir": str(cache)}
        signature = kv._fetch_signature(public_request["center"], 1000., 1500)
        meta_path = cache / ("mly_ref_meta_" + kv._sig_tag(signature) + ".json")
        write(meta_path, {"signature": signature, "refs": refs})
        fingerprint = kv._refs_fingerprint(refs)
        np.savez_compressed(cache / "ref_img_store.npz", ids=np.array([refs[0]["id"]]), xy=ll,
                            raw=np.zeros((1, 512, 512, 3), np.uint8))
        np.savez_compressed(cache / ("ref_emb_megaloc_" + fingerprint[:10] + ".npz"),
                            emb=region["descriptor_rows"], fingerprint=np.array(fingerprint))
        audit = region["audit"]
        audit["public_cache_request"] = public_request
        audit["reference_pool"].update({k: public_request[k] for k in ("source", "center", "radius_m", "cap")})
        array_path = output / "retrieval_audit/0000.npz"; array_path.parent.mkdir()
        np.savez_compressed(array_path, **region["arrays"])
        ar = {"path": str(array_path), "sha256": a.native.sha(array_path)}
        audit["array_artifact"].update(ar)
        mr = write(array_path.with_suffix(".json"), audit)
        call = {"key": key, "request_ref": rr, "metadata_ref": mr, "array_ref": ar}
        calls.append(call)
        own_members = [rr, mr, ar]
        owner = write(output / "child_frozen.json", {"artifacts": {
            str(Path(v["path"]).relative_to(output)): v["sha256"] for v in own_members}})
        protected.extend([mr, ar])
        bound.append({**call, "owner_freeze_ref": owner})
        app = deepcopy(facts["appearance_identity"])
        banks.append({"key": kind, "pair_records": [{"role": "a", "metadata": mr["path"],
            "metadata_sha256": mr["sha256"], "npz": ar["path"], "npz_sha256": ar["sha256"],
            "issues": [], "raw_headers": audit["array_artifact"]["arrays"], "appearance_identity": app}]})
        files = [{"path": str(p), "name": p.name, "sha256": a.native.sha(p), "bytes": p.stat().st_size}
                 for p in sorted(cache.iterdir())]
        directories.append({"path": str(cache), "files": files}); cache_calls[key] = str(cache)
    inv_ref = write(tmp / "inventory.json", {"schema": "closed-raw-vpr-metadata-bank-inventory-v1",
        "banks": banks, "protected_files": protected})
    binding_files = protected + [v[role] for v in bound for role in ("request_ref", "owner_freeze_ref")]
    binding_ref = write(tmp / "bindings.json", {"schema": "root-exact-raw-bank-own-request-bindings-v1",
        "inventory_ref": inv_ref, "all_inventory_rows_bound": 2, "calls": bound, "protected_files": binding_files})
    origin_ref = write(tmp / "origin.json", {"source": str(historical), "source_clean": True, "source_commit": "d" * 40})
    facts.update(metadata_inventory=inv_ref, own_request_bindings=binding_ref,
        request_execution_origin_ref=origin_ref, request_execution_root=str(historical),
        bank_membership={kind: [kind + "/a"] for kind in ("baseline", "expanded")},
        all_physical_calls_and_aliases=[{**deepcopy(call), "profile_key": "f" * 64,
            "query_pixels_sha256": "a" * 64,
            "query_indices_sha256": facts["appearance_identity"]["query_indices_header"]["sha256"],
            "query_times_sha256": facts["appearance_identity"]["query_times_header"]["sha256"],
            "query_to_vo": [{} for _ in range(80)]} for call in calls],
        protected_hashes={v["path"]: v["sha256"] for v in binding_files + [inv_ref, binding_ref, origin_ref]},
        external_actor_closure_verified=False)
    facts_ref = write(tmp / "native_facts.json", facts)
    registration_ref = write(tmp / "native_guard_registration.json", {"schema": "private-guard-registration"})
    result_ref = write(tmp / "native_guard_result.json", {"status": "complete", "returncode": 0, "reason": None})
    closure_ref = write(tmp / "native_closure.json", {"schema": "root-native-replay-external-closure-v1",
        "native_facts_ref": facts_ref, "registration_ref": registration_ref, "result_ref": result_ref,
        "worker_births_closed": True, "protected_inputs_unchanged": True})
    qa = {"schema": "root-closed-raw-vpr-native-acceptance-v1", "status": "ACCEPTED_INPUT_PIXELS_CLOCK_ONLY_NO_NN",
        "native_facts_ref": facts_ref, "inventory_ref": inv_ref, "own_request_bindings_ref": binding_ref,
        "external_guard_registration_ref": registration_ref, "external_guard_result_ref": result_ref,
        "external_closure_ref": closure_ref, "all_observed_worker_births_closed": True,
        "all_protected_input_hashes_unchanged": True, "native_input_pixels_clock_verified": True,
        "reference_encoding_identity": "UNVERIFIED", "model_inference_run": False,
        "reference_encoding_upgraded": False, "original_child_completion_certified": False,
        "ground_truth_used": False, "selected_answer": None}
    qa_ref = write(tmp / "native_root_qa.json", qa)
    payload_ref = write(tmp / "public_payload_inventory.json", {
        "schema": "root-raw-vpr-reference-payload-inventory-v1", "metadata_inventory_ref": inv_ref,
        "calls": cache_calls, "directories": directories})
    config = {"schema": "raw-vpr-bank-admission-config-v1",
        "source_hashes": {name: a.native.sha(a.ROOT / name) for name in a.SOURCES},
        "inventory_ref": inv_ref, "own_request_bindings_ref": binding_ref, "calls": calls,
        "banks": [{"key": kind, "call_keys": [kind + "/a"]} for kind in ("baseline", "expanded")],
        "request_execution_root": str(historical), "request_execution_origin_ref": origin_ref,
        "native_facts_ref": facts_ref, "native_root_acceptance_ref": qa_ref,
        "reference_payload_inventory": payload_ref, "output_dir": str(tmp / "output")}
    config_ref = write(tmp / "config.json", config)
    return config, config_ref


@contextmanager
def threads(): yield [{"num_threads": 2, "private_fixture": True}]


def run(ref, **kwargs):
    return a.execute(ref["path"], ref["sha256"], numerical_context=threads, resource_gate=lambda: {}, **kwargs)


def change(config, key, mutate):
    path = Path(config[key]["path"])
    value = json.loads(path.read_text(encoding="utf-8")); mutate(value)
    config[key] = write(path, value)


def test_complete_real_genuine_descriptor_loader_from_private_caches(tmp_path):
    config, ref = fixture(tmp_path)
    result = run(ref)
    assert [v["status"] for v in result["banks"]] == ["complete_research_bank"] * 2
    assert len(list(Path(config["output_dir"]).glob("*.npz"))) == 2
    assert result["external_actor_closure_verified"] is False and not result["field_built"]
    assert result["reference_encoding_identity"] == "UNVERIFIED"
    for proof in result["descriptor_receipts"].values():
        assert proof["current_descriptor_receipt"]["status"] == "complete"
        assert proof["original_persisted_receipt_present"] is False
    for row in result["banks"]:
        with np.load(row["array_artifact"]["path"], allow_pickle=False) as saved:
            assert set(saved.files) == set(a.ARRAY_NAMES)
            assert saved["raw_similarities"].shape == (80, 1)


@pytest.mark.parametrize("fault", ["qa_false", "guard_failed", "closure_false", "fact_ref", "missing_root"])
def test_positive_root_native_acceptance_precedes_even_query_grid_load(tmp_path, monkeypatch, fault):
    config, _ = fixture(tmp_path)
    if fault == "qa_false": change(config, "native_root_acceptance_ref", lambda q: q.update(all_observed_worker_births_closed=False))
    elif fault == "fact_ref": change(config, "native_root_acceptance_ref", lambda q: q["native_facts_ref"].update(sha256="f" * 64))
    elif fault == "missing_root": config.pop("native_root_acceptance_ref")
    else:
        qa = json.loads(Path(config["native_root_acceptance_ref"]["path"]).read_text())
        role = "external_guard_result_ref" if fault == "guard_failed" else "external_closure_ref"
        value = json.loads(Path(qa[role]["path"]).read_text())
        value.update(returncode=1) if fault == "guard_failed" else value.update(worker_births_closed=False)
        qa[role] = write(Path(qa[role]["path"]), value)
        config["native_root_acceptance_ref"] = write(Path(config["native_root_acceptance_ref"]["path"]), qa)
    ref = write(tmp_path / "config.json", config)
    monkeypatch.setattr(a.native.Ledger, "members", lambda *args: pytest.fail("NPZ member before native Root gate"))
    monkeypatch.setattr(a.component, "_arrays_bound", lambda *args: pytest.fail("raw arrays before native Root gate"))
    with pytest.raises((ValueError, KeyError)): run(ref)
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("fault", ["sha", "size", "new_file"])
def test_all_public_payloads_are_frozen_before_values(tmp_path, monkeypatch, fault):
    config, _ = fixture(tmp_path)
    if fault == "new_file":
        payload = json.loads(Path(config["reference_payload_inventory"]["path"]).read_text())
        (Path(payload["directories"][0]["path"]) / "ref_emb_unregistered.npz").write_bytes(b"private unknown")
    else:
        def mutate(value):
            item = value["directories"][0]["files"][0]
            item.update(sha256="f" * 64) if fault == "sha" else item.update(bytes=item["bytes"] + 1)
        change(config, "reference_payload_inventory", mutate)
    ref = write(tmp_path / "config.json", config)
    monkeypatch.setattr(a.native.Ledger, "members", lambda *args: pytest.fail("query values before payload freeze"))
    with pytest.raises(ValueError): run(ref)
    assert not Path(config["output_dir"]).exists()


def test_one_bank_loader_failure_is_retained_without_salvaging_its_columns(tmp_path):
    config, ref = fixture(tmp_path)
    def loader(region, cfg):
        if region["key"].startswith("baseline/"): raise ValueError("private baseline pool unavailable")
        return a.component.load_descriptor_rows(region, cfg)
    result = run(ref, descriptor_loader=loader)
    assert [v["status"] for v in result["banks"]] == ["UNAVAILABLE", "complete_research_bank"]
    assert "array_artifact" not in result["banks"][0]
    assert not (Path(config["output_dir"]) / "bank_00.npz").exists()
    assert (Path(config["output_dir"]) / "bank_01.npz").exists()


@pytest.mark.parametrize("target", ["request", "output", "public_payload", "membership"])
def test_last_callback_cannot_rebless_changed_sources_or_outputs(tmp_path, target):
    config, ref = fixture(tmp_path); output = Path(config["output_dir"])
    fired = []
    def check():
        if fired or len(list(output.glob("*.npz"))) != 2: return
        if target == "request": path = Path(config["calls"][0]["request_ref"]["path"])
        elif target == "output": path = output / "bank_00.npz"
        else:
            payload = json.loads(Path(config["reference_payload_inventory"]["path"]).read_text())
            path = Path(payload["directories"][0]["files"][0]["path"])
            if target == "membership": path = path.parent / "ref_emb_late.npz"
        assert path.resolve().is_relative_to(tmp_path.resolve())
        path.write_bytes(b"private last-callback mutation"); fired.append(str(path))
    with pytest.raises(ValueError): run(ref, check=check)
    assert fired and not (output / "bank_admission.json").exists()


def test_exclusive_output_is_not_retried(tmp_path):
    config, ref = fixture(tmp_path)
    run(ref)
    with pytest.raises(FileExistsError): run(ref)
