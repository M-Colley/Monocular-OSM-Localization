"""Historical configs and frozen runtime receipts; synthetic files only."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import assisted_goal_audit as audit
from monocular_osm.stages import assisted as stage
from tests import test_check_assisted_goal as legacy


def child_case(tmp_path, monkeypatch, *, absent=(), deleted=None, changed=None, request_deleted=None):
    case = legacy.fixture(tmp_path, monkeypatch)
    inputs = deepcopy(case.ctx.result["assisted_inputs"])
    response = case.children["baseline"]
    root = Path(response["candidate_artifact"]["path"]).parent
    candidate = audit.load_json(root / "candidate.json")
    request = audit.load_json(root / "request.json")
    for field in absent:
        inputs["config"].pop(field, None)
        candidate["config"].pop(field, None)
        request["config"].pop(field, None)
    if deleted:
        candidate["config"].pop(deleted)
        request["config"].pop(deleted)
    if changed:
        field, value = changed
        candidate["config"][field] = request["config"][field] = value
    if request_deleted:
        request["config"].pop(request_deleted)
    legacy.write(root / "request.json", request)
    legacy.write(root / "candidate.json", candidate)
    frozen = audit.load_json(root / "child_frozen.json")
    frozen["artifacts"] = {name: audit.file_sha(root / name) for name in frozen["artifacts"]}
    legacy.write(root / "child_frozen.json", frozen)
    child = {**candidate, "worker_returncode": 0}
    for role, name in (("candidate_artifact", "candidate.json"), ("result_artifact", "result.json"), ("frozen_artifact", "child_frozen.json")):
        child[role] = {"path": str(root / name), "sha256": audit.file_sha(root / name)}
    record = stage.child_record("baseline", child, case.output, expected=inputs)
    return audit.Files(case.output), record, inputs


@pytest.mark.parametrize("absent", [audit.RUNTIME_CONFIG_FIELDS, *[(field,) for field in audit.RUNTIME_CONFIG_FIELDS]])
def test_real_frozen_child_accepts_historical_runtime_fields_absent_in_both_parent_and_child(tmp_path, monkeypatch, absent):
    files, record, inputs = child_case(tmp_path, monkeypatch, absent=absent)
    child, ledger = audit.verify_child(files, record, inputs, {})
    assert child["status"] == "complete" and ledger
    assert all(field not in child["config"] and field not in inputs["config"] for field in absent)


@pytest.mark.parametrize("field", audit.RUNTIME_CONFIG_FIELDS)
def test_explicit_child_runtime_field_deletion_is_not_treated_as_historical(tmp_path, monkeypatch, field):
    values = child_case(tmp_path, monkeypatch, deleted=field)
    with pytest.raises(ValueError, match="component/baseline configuration"):
        audit.verify_child(*values, {})


@pytest.mark.parametrize("field,value", [("runtime_budget_s", 1000.), ("runtime_deadline_monotonic_s", 1000.),
                                         ("runtime_finalization_reserve_s", 121.), ("runtime_cleanup_reserve_s", 11.)])
def test_rehashed_request_and_child_cannot_change_explicit_runtime_fields(tmp_path, monkeypatch, field, value):
    values = child_case(tmp_path, monkeypatch, changed=(field, value))
    with pytest.raises(ValueError, match="component/baseline configuration"):
        audit.verify_child(*values, {})


def test_child_request_runtime_deletion_still_breaks_exact_request_config_binding(tmp_path, monkeypatch):
    values = child_case(tmp_path, monkeypatch, request_deleted="runtime_budget_s")
    with pytest.raises(ValueError, match="requested configuration"):
        audit.verify_child(*values, {})


def runtime_case(tmp_path):
    source = tmp_path / "source_snapshot/monocular_osm/runtime_budget.py"
    source.parent.mkdir(parents=True);source.write_text("# synthetic runtime implementation")
    cfg = dict(zip(audit.RUNTIME_CONFIG_FIELDS, (900., 1000., 120., 10.)))
    inputs = {"execution_id": "a" * 32, "source_sha256": {"monocular_osm/runtime_budget.py": audit.file_sha(source)}}
    event = {"kind": "baseline", "status": "process_complete", "monotonic_s": 200., "worker_returncode": 0,
             "owned_births": [{"pid": 71, "create_time": 50.}], "all_tracked_owned_births_gone": True,
             "ownership_scope": "observed_pid_births_only; unobserved_orphans_not_certified"}
    record = {"schema": "assisted-runtime-budget-v1", "origin": "benchmark", "deadline_monotonic_s": 1000.,
              "started_monotonic_s": 100., "work_deadline_monotonic_s": 870., "finalization_reserve_s": 120.,
              "cleanup_reserve_s": 10., "events": [event]}
    transcript = {"runtime_budget": record, "reached_stages": [{"kind": "baseline", "status": "complete", "worker_returncode": 0, "artifacts": []}]}
    path = tmp_path / "assisted_runtime.json"
    legacy.write(path, record)
    return SimpleNamespace(files=audit.Files(tmp_path), cfg=cfg, inputs=inputs, transcript=transcript,
                           ledger={path: audit.file_sha(path)}, path=path, source=source)


def seal_runtime(case):
    legacy.write(case.path, case.transcript["runtime_budget"])
    case.ledger[case.path] = audit.file_sha(case.path)


def check_runtime(case):
    audit.runtime_evidence(case.files, case.cfg, case.inputs, case.transcript, case.ledger)


def test_budgeted_parent_binds_source_exact_frozen_receipt_and_reserved_work_clock(tmp_path):
    check_runtime(runtime_case(tmp_path))


def test_old_unbudgeted_parent_needs_no_new_runtime_source_or_receipt(tmp_path):
    audit.runtime_evidence(audit.Files(tmp_path), {}, {"source_sha256": {}}, {}, {})


@pytest.mark.parametrize("fault", ["source_omitted", "source_changed", "receipt_omitted", "receipt_changed", "audit_omitted",
    "config_field_omitted", "schema", "origin", "clock", "reserve", "longer_absolute", "longer_relative",
    "complete_event_omitted", "wrong_exit", "missing_births", "duplicate_births", "nonfinite_birth", "live_birth_claim", "ownership_scope", "past_deadline"])
def test_rehashed_missing_stale_or_extended_runtime_claims_refuse(tmp_path, fault):
    case = runtime_case(tmp_path);record = case.transcript["runtime_budget"];event = record["events"][0]
    if fault == "source_omitted":case.inputs["source_sha256"].clear()
    elif fault == "source_changed":case.source.write_text("changed runtime implementation")
    elif fault == "receipt_omitted":case.ledger.clear()
    elif fault == "receipt_changed":case.path.write_text("{}")
    elif fault == "audit_omitted":case.transcript.pop("runtime_budget")
    elif fault == "config_field_omitted":case.cfg.pop("runtime_budget_s")
    elif fault == "schema":record["schema"] = "other";seal_runtime(case)
    elif fault == "origin":record["origin"] = "unknown";seal_runtime(case)
    elif fault == "clock":record["work_deadline_monotonic_s"] += 1;seal_runtime(case)
    elif fault == "reserve":case.cfg["runtime_cleanup_reserve_s"] = 11.
    elif fault == "longer_absolute":case.cfg["runtime_deadline_monotonic_s"] = 999.
    elif fault == "longer_relative":case.cfg["runtime_budget_s"] = 899.
    elif fault == "complete_event_omitted":record["events"].clear();seal_runtime(case)
    elif fault == "wrong_exit":event["worker_returncode"] = 1;seal_runtime(case)
    elif fault == "missing_births":event.pop("owned_births");seal_runtime(case)
    elif fault == "duplicate_births":event["owned_births"].append(deepcopy(event["owned_births"][0]));seal_runtime(case)
    elif fault == "nonfinite_birth":event["owned_births"][0]["create_time"] = -1;seal_runtime(case)
    elif fault == "live_birth_claim":event["all_tracked_owned_births_gone"] = False;seal_runtime(case)
    elif fault == "past_deadline":event["monotonic_s"] = 1000.001;seal_runtime(case)
    else:event["ownership_scope"] = "all_orphans_certified";seal_runtime(case)
    with pytest.raises((ValueError, KeyError, OSError)):
        check_runtime(case)


def test_frozen_receipt_must_equal_audit_record_even_after_individual_file_rehash(tmp_path):
    case = runtime_case(tmp_path);other = deepcopy(case.transcript["runtime_budget"]);other["origin"] = "config"
    legacy.write(case.path, other);case.ledger[case.path] = audit.file_sha(case.path)
    with pytest.raises(ValueError, match="frozen/equal runtime receipt"):
        check_runtime(case)


def timeout_event(case):
    path = case.files.root / "children" / case.inputs["execution_id"] / "cues";path.mkdir(parents=True)
    case.transcript["reached_stages"].append({"kind": "cues", "status": "failed", "artifacts": []})
    event = {"kind": "cues", "status": "timed_out", "monotonic_s": 880., "worker_returncode": -9,
             "reason": "child_work_deadline_expired", "request_sha256": "b" * 64, "partial_output": str(path),
             "partial_output_is_selected_or_complete": False, "owned_births": [{"pid": 72, "create_time": 60.}],
             "all_tracked_owned_births_gone": True, "ownership_scope": "observed_pid_births_only; unobserved_orphans_not_certified"}
    case.transcript["runtime_budget"]["events"].append(event);seal_runtime(case)
    return event


def test_timed_out_optional_child_stays_failed_and_explicitly_unselected(tmp_path):
    case = runtime_case(tmp_path);timeout_event(case);check_runtime(case)


@pytest.mark.parametrize("fault", ["promoted", "artifacts", "selected_partial", "wrong_execution", "wrong_request", "wrong_time"])
def test_timeout_receipt_cannot_promote_partial_or_mix_child_identity(tmp_path, fault):
    case = runtime_case(tmp_path);event = timeout_event(case)
    if fault == "promoted":case.transcript["reached_stages"][-1]["status"] = "complete"
    elif fault == "artifacts":case.transcript["reached_stages"][-1]["artifacts"] = [{"role": "candidate_artifact"}]
    elif fault == "selected_partial":event["partial_output_is_selected_or_complete"] = True
    elif fault == "wrong_execution":event["partial_output"] = str(case.files.root)
    elif fault == "wrong_request":event["request_sha256"] = "unknown"
    else:event["monotonic_s"] = 199.
    seal_runtime(case)
    with pytest.raises(ValueError):check_runtime(case)


def selected_mapping_case(*, ulp=True):
    import numpy as np
    from tests.test_metadata_union import duplicate_occurrence_case
    args = duplicate_occurrence_case(q=80, changed_rows=np.linspace(0, 79, 22).astype(int) if ulp else np.empty(0, dtype=int))
    regions, comparison = args[0], args[3]
    descriptors = {r["key"]: r["descriptor_rows"] for r in regions}
    union = audit.metadata_union_rebuild(regions, descriptors)
    rich, legacy_mapping = audit.metadata_selected_mappings(regions, comparison, union)
    expected = {"fixed_other_admission_fields": {"sole_primary": "reference_mean"},
                "original_selected_sources": {"baseline": [{"fixed": True}], "metadata": rich}}
    return regions, comparison, union, expected, legacy_mapping


def test_independent80_query22_ULP_union_proof_accepts_exact_rich_occurrences_only():
    regions, comparison, union, expected, legacy_mapping = selected_mapping_case()
    assert legacy_mapping is None
    rows = expected["original_selected_sources"]["metadata"]
    assert len(rows) == 80 and sum(r["raw_cosine"] != r["canonical_union_source"]["raw_cosine"] for r in rows) == 22
    assert all(r["region"] == "b_selected" and r["raw_cosine"] == comparison["decoded_reference_cosines"][q] for q, r in enumerate(rows))
    assert audit.metadata_admission_matches(deepcopy(expected), expected, legacy_mapping)
    compact = deepcopy(expected)
    compact["original_selected_sources"]["metadata"] = [{k: r[k] for k in ("query_row", "global_column", "ref_id")} for r in rows]
    assert not audit.metadata_admission_matches(compact, expected, legacy_mapping)


def test_historical_compact_mapping_is_accepted_only_with_exact_original_canonical_invariant():
    _, _, _, expected, legacy_mapping = selected_mapping_case(ulp=False)
    assert legacy_mapping is not None and len(legacy_mapping) == 80
    old = deepcopy(expected);old["original_selected_sources"]["metadata"] = legacy_mapping
    assert audit.metadata_admission_matches(old, expected, legacy_mapping)
    old["ignored_extra_claim"] = True
    assert not audit.metadata_admission_matches(old, expected, legacy_mapping)


@pytest.mark.parametrize("fault", ["scalar", "coordinate", "pool", "query", "missing_rows", "source_occurrence"])
def test_independent_union_source_proof_refuses_genuine_source_mismatch(fault):
    regions, comparison, union, _, _ = selected_mapping_case()
    if fault == "scalar":comparison["decoded_reference_cosines"][0] += 1e-12
    elif fault == "coordinate":comparison["decoded_reference_latlon"][0][0] += 1e-12
    elif fault == "pool":comparison["selected_audit_rows"][0] = [0, 0]
    elif fault == "query":comparison["selected_audit_rows"][1] = [1, 0]
    elif fault == "missing_rows":comparison.pop("selected_audit_rows")
    else:union["mapping"][0]["occurrences"] = [r for r in union["mapping"][0]["occurrences"] if r["region"] != "b_selected"]
    with pytest.raises(ValueError):audit.metadata_selected_mappings(regions, comparison, union)


@pytest.mark.parametrize("fault", ["extra", "missing", "canonical", "original", "baseline", "numeric_alias"])
def test_rich_admission_requires_exact_allowed_structure_not_subset_claims(fault):
    _, _, _, expected, legacy_mapping = selected_mapping_case();actual = deepcopy(expected)
    row = actual["original_selected_sources"]["metadata"][0]
    if fault == "extra":row["ignored_extra_claim"] = True
    elif fault == "missing":row.pop("audit_index")
    elif fault == "canonical":row["canonical_union_source"]["raw_cosine"] += 1e-12
    elif fault == "original":row["raw_cosine"] += 1e-12
    elif fault == "baseline":actual["original_selected_sources"]["baseline"] = []
    else:row["audit_index"] = 1.0
    assert not audit.metadata_admission_matches(actual, expected, legacy_mapping)
