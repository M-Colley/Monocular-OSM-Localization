"""Frozen-source runner proofs; synthetic matrices, no models or labels."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from scripts import replay_metadata_union as r
from tests.test_metadata_union import case, make_region, comparison, summary, coords


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    r.write(path, value)


def freeze(path, **extra):
    write(path, {**extra, "artifacts": {p.relative_to(path.parent).as_posix(): r.sha(p)
        for p in path.parent.rglob("*") if p.is_file() and p != path}})


def store_region(root, index, region):
    root.mkdir(parents=True, exist_ok=True)
    arrays = root/f"{index:04d}.npz"; np.savez(arrays, **region["arrays"])
    audit = deepcopy(region["audit"])
    audit["array_artifact"] = {"path":str(arrays.resolve()), "sha256": r.sha(arrays), "arrays": {k: {"shape": list(v.shape),
        "dtype": v.dtype.str, "sha256": r.core.full._array_sha(v)} for k, v in region["arrays"].items()}}
    write(root/f"{index:04d}.json", audit)


def source_fixture(tmp_path, monkeypatch, *, two_baseline=False, mixed_baseline=False):
    monkeypatch.setattr(r, "source_identity", lambda *_: {})
    monkeypatch.setattr(r.utility, "check_headroom", lambda: {"available_physical_bytes": 20*1024**3,
        "available_commit_bytes": 20*1024**3})
    monkeypatch.setattr(r.subprocess, "check_output", lambda args, **_: "a"*40 if "rev-parse" in args else "")
    old = tmp_path/"old"; base = old/"baseline"; expanded = old/"expanded"
    q = 80; regions = []
    for i in range(6):
        n = 1200 if i == 0 else 1500
        region = make_region(f"region_{i:04d}", np.full((q,n), .2, np.float32),
            coords(np.arange(n)+i*2000), [f"{i}_{j}" for j in range(n)], chosen=np.zeros(q, np.int64))
        store_region(expanded/"metadata_admission/retrieval_audit", i, region); regions.append(region)
    baseline_region = make_region("original_baseline", np.full((q,2), .1, np.float32),
        coords([-10000,-9000]), ["baseline_a","baseline_b"], chosen=np.zeros(q, np.int64))
    store_region(base/"retrieval_audit", 0, baseline_region)
    def selected(region):
        a = region["arrays"]; selected = a["decoded_ref_indices"]
        return comparison(a["raw_similarities"][np.arange(q),selected], a["ref_latlon"][selected], a["ref_ids"][selected])
    bc, mc = selected(baseline_region), selected(regions[0])
    if two_baseline:
        fine=make_region("original_baseline_0001",np.full((q,2),.11,np.float32),coords([-8000,-7000]),
            ["fine_a","fine_b"],chosen=np.zeros(q,np.int64))
        store_region(base/"retrieval_audit",1,fine)
        fine_comparison=selected(fine)
        for row in range(q):
            audit_index=row%2 if mixed_baseline else 1
            bc["selected_audit_rows"][row]=[audit_index,row]
            if audit_index==1:
                for key in ("decoded_reference_cosines","decoded_reference_latlon","decoded_reference_ids"):
                    bc[key][row]=fine_comparison[key][row]
    bc["audit_artifacts"]=[{"path":str(path.resolve()),"sha256":r.sha(path)} for path in sorted((base/"retrieval_audit").iterdir())]
    write(base/"candidate.json", {"kind": "baseline", "status": "complete", "comparison": bc})
    evidence = {"baseline": summary(bc), "expanded": summary(mc, True), "cues": {"status": "failed"}}
    write(expanded/"metadata_admission/admission.json", {"evidence": evidence, "comparison": mc,
        "decision": r.core.policy.decide_assisted_policy(evidence)})
    freeze(base/"child_frozen.json", status="complete", kind="baseline")
    freeze(expanded/"child_frozen.json", status="partial", kind="expanded")
    freeze(old/"frozen_prediction.json", status="complete", scoring_has_not_run=True)
    upstream = tmp_path/"source"/"input_pixels.dat"; upstream.parent.mkdir(); upstream.write_bytes(b"native video/pixel identity")
    audit = tmp_path/"audit"/"input_audit.json"
    write(audit, {"status": "PASS", "freeze_sha256": r.sha(old/"frozen_prediction.json"), "source_commit": "a"*40,
        "source": str(upstream.parent), "ground_truth_read": False, "numeric_accuracy_scores_read": False,
        "new_models_or_forwards": False, "protected_unchanged": True, "source_clean": True,
        "protected_hashes": {str(p):r.sha(p) for p in [*old.rglob("*.json"),*old.rglob("*.npz"),upstream]}})
    proposal = tmp_path/"proposal"/"proposal.json"
    write(proposal, {"proposal_sha256": r.PROFILE["proposal_profile_sha256"]})
    monkeypatch.setitem(r.PROFILE, "proposal_sha256", r.sha(proposal))
    monkeypatch.setattr(r, "PROFILE_SHA256", r.core.json_sha(r.PROFILE))
    spec = tmp_path/"spec.json"
    write(spec, {"parent": {"freeze":str(old/"frozen_prediction.json"),"sha256":r.sha(old/"frozen_prediction.json")},
        "baseline": {"freeze":str(base/"child_frozen.json"),"sha256":r.sha(base/"child_frozen.json")},
        "expanded": {"freeze":str(expanded/"child_frozen.json"),"sha256":r.sha(expanded/"child_frozen.json")},
        "audit": {"path":str(audit),"sha256":r.sha(audit)}, "proposal": {"path":str(proposal),"sha256":r.sha(proposal)},
        "original_source_commit":"a"*40})
    return spec, old, upstream


def test_import_never_loads_neural_runtime():
    subprocess.run([sys.executable,"-c","import sys; from scripts import replay_metadata_union; assert 'torch' not in sys.modules"],
        cwd=r.ROOT, check=True)


def test_preparation_reads_all_six_full_matrices_and_independent_baseline_without_DP(tmp_path, monkeypatch):
    spec, _, _ = source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(r.core.full,"decode_full_tracks",lambda *_:pytest.fail("prepare cannot decode"))
    result = r.verified_source(r.specification(spec))
    assert sum(len(region["arrays"]["ref_ids"]) for region in result["regions"]) == 8700
    assert len(result["originals"]) == 16 and result["baseline_regions"][0]["key"] == "original_baseline_0000"
    assert result["baseline"]["comparison"]["decoded_reference_ids"][0] not in result["regions"][0]["arrays"]["ref_ids"]


@pytest.mark.parametrize("mixed",[False,True])
def test_all_frozen_baseline_audits_preserved_and_selected_second_or_mixed_rows_bound(tmp_path,monkeypatch,mixed):
    spec,_,_=source_fixture(tmp_path,monkeypatch,two_baseline=True,mixed_baseline=mixed)
    bundle=r.verified_source(r.specification(spec))
    assert len(bundle["baseline_regions"])==2 and len(bundle["originals"])==18
    mapping=bundle["identity"]["baseline_selected_sources"]
    assert [row["audit_index"] for row in mapping]==[i%2 if mixed else 1 for i in range(80)]
    assert all(row["audit_query_row"]==i for i,row in enumerate(mapping))


def test_baseline_comparison_cannot_omit_an_existing_frozen_audit_pool(tmp_path,monkeypatch):
    spec,_,_=source_fixture(tmp_path,monkeypatch,two_baseline=True)
    declared=r.specification(spec);path=Path(declared["baseline"]["freeze"])
    frozen=r.utility.freeze(path,declared["baseline"]["sha256"])
    candidate=r.load(path.parent/"candidate.json")["comparison"]
    candidate["audit_artifacts"]=candidate["audit_artifacts"][:2]
    with pytest.raises(ValueError,match="omitted"):r.read_baseline_pools(path,frozen,candidate)


@pytest.mark.parametrize("flag",["status","freeze_sha256","source_commit","ground_truth_read","numeric_accuracy_scores_read",
    "new_models_or_forwards","protected_unchanged","source_clean"])
def test_unaccepted_audit_flags_decline_before_DP(tmp_path,monkeypatch,flag):
    spec,_,_=source_fixture(tmp_path,monkeypatch); declared=r.specification(spec)
    path=Path(declared["audit"]["path"]); audit=r.load(path)
    audit[flag]="bad" if flag in {"status","freeze_sha256","source_commit"} else not audit[flag]
    path.unlink(); write(path,audit); declared["audit"]["sha256"]=r.sha(path)
    with pytest.raises(ValueError,match="audit"):r.verified_source(declared)


def test_mutable_original_cache_or_native_bytes_cannot_change_after_accepted_audit(tmp_path,monkeypatch):
    spec,_,upstream=source_fixture(tmp_path,monkeypatch); upstream.write_bytes(b"changed pixels")
    with pytest.raises(ValueError,match="destination/source"):r.verified_source(r.specification(spec))


@pytest.mark.parametrize("bad",["ground_truth/a.json","saved_official_score.json","offline_scores.json"])
def test_input_roles_decline_labels_before_hash(tmp_path,bad):
    value={role:{key:str(tmp_path/"safe.json"),"sha256":"a"*64} for role,key in r.ROLES.items()}
    value["original_source_commit"]="a"*40;value["parent"]["freeze"]=str(tmp_path/bad)
    path=tmp_path/"spec.json";write(path,value)
    with pytest.raises(ValueError,match="labels|scores"):r.specification(path)


def test_canonical_frozen_control_members_accept_windows_and_refuse_aliases(tmp_path):
    assert r.artifacts({"artifacts":{"controls\\a.json":"a"*64}})=={"controls/a.json":"a"*64}
    for members in ({"a.json":"a"*64,"A.json":"b"*64},{"../escape.json":"a"*64}):
        with pytest.raises(ValueError):r.artifacts({"artifacts":members})


@pytest.mark.parametrize("bad",["extra","object","float64","oversize"])
def test_npz_headers_and_roles_fail_closed(tmp_path,bad):
    region=case()[0][0];store_region(tmp_path,0,region);path=tmp_path/"0000.npz"
    arrays=deepcopy(region["arrays"])
    if bad=="extra":arrays["new"]=np.ones(2)
    elif bad=="object":arrays["ref_ids"]=np.array([{}],object)
    elif bad=="float64":arrays["raw_similarities"]=arrays["raw_similarities"].astype(float)
    else:arrays["ref_ids"]=np.array(["x"]*1501)
    np.savez(path,**arrays);meta=r.load(tmp_path/"0000.json");meta["array_artifact"]["sha256"]=r.sha(path)
    (tmp_path/"0000.json").unlink();write(tmp_path/"0000.json",meta)
    with pytest.raises(ValueError):r.read_region(tmp_path/"0000.json",path)


def small_prepared(tmp_path,monkeypatch):
    spec,_,_=source_fixture(tmp_path,monkeypatch)
    original=r.verified_source
    def small(value):
        bundle=original(value);regions,bc,bs,mc,ms,base=case()
        bundle.update(regions=regions,baseline={"comparison":bc},baseline_regions=base,
            admission={"comparison":mc,"evidence":{"baseline":bs,"expanded":ms}})
        return bundle
    monkeypatch.setattr(r,"verified_source",small)
    family=tmp_path/"new"
    frozen=r.prepare(spec,family,"new_pin")
    return family,frozen["input_freeze_sha256"]


def test_complete_new_freeze_keeps_both_variants_fixed_primary_and_original_controls(tmp_path,monkeypatch):
    family,digest=small_prepared(tmp_path,monkeypatch)
    assert not (family/"predictions").exists()
    result=r.infer(family,"new_pin",digest)
    master=r.utility.freeze(family/"predictions/frozen_predictions.json",result["freeze_sha256"])
    assert master["both_research_outputs"] is True and master["sole_new_primary"]=="reference_mean"
    assert master["scoring_has_not_run"] is master["original_controls_preserved_as_source_artifacts"] is True
    assert {f"prediction_{v}.json" for v in ("raw","reference_mean")} <= set(master["artifacts"])
    for row in r.load(family/"plan.json")["originals"]:
        assert Path(row["source"]).read_bytes()==(family/"inputs"/row["copy"]).read_bytes()
    with pytest.raises(FileExistsError):r.infer(family,"new_pin",digest)


def test_final_external_input_sha_refuses_in_run_plan_reblessing(tmp_path,monkeypatch):
    family,digest=small_prepared(tmp_path,monkeypatch);original=r.core.replay_metadata_union
    def mutate(*args):
        result=original(*args);plan=r.load(family/"plan.json");plan["edited"]=True
        (family/"plan.json").unlink();write(family/"plan.json",plan)
        frozen=r.load(family/"inputs_frozen.json");frozen["artifacts"]["plan.json"]=r.sha(family/"plan.json")
        (family/"inputs_frozen.json").unlink();write(family/"inputs_frozen.json",frozen)
        return result
    monkeypatch.setattr(r.core,"replay_metadata_union",mutate)
    with pytest.raises(ValueError,match="freeze changed"):r.infer(family,"new_pin",digest)
    assert not (family/"predictions/frozen_predictions.json").exists()


def test_decode_failure_preserves_partial_without_success_master(tmp_path,monkeypatch):
    family,digest=small_prepared(tmp_path,monkeypatch)
    def fail(*_):raise RuntimeError("synthetic resource refusal")
    monkeypatch.setattr(r.core,"replay_metadata_union",fail)
    with pytest.raises(RuntimeError,match="resource"):r.infer(family,"new_pin",digest)
    assert (family/"predictions/failure.json").exists() and not (family/"predictions/frozen_predictions.json").exists()


def test_resource_gate_enforces_registered_wall_and_both_memory_floors(monkeypatch):
    monkeypatch.setattr(r.time,"monotonic",lambda:3601.)
    with pytest.raises(ValueError,match="wall"):r.resource_gate(0)
    monkeypatch.setattr(r.time,"monotonic",lambda:1.)
    def fail():raise ValueError("physical/actual commit below8GiB")
    monkeypatch.setattr(r.utility,"check_headroom",fail)
    with pytest.raises(ValueError,match="commit"):r.resource_gate(0)


def test_existing_output_family_never_overwrites(tmp_path,monkeypatch):
    spec,_,_=source_fixture(tmp_path,monkeypatch);family=tmp_path/"existing";family.mkdir()
    with pytest.raises(ValueError,match="isolated"):r.prepare(spec,family,"new_pin")
