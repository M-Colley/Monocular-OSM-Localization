"""Tiny persisted bytes and fake workers only; no models or real inputs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from monocular_osm import assisted_worker as worker, retrieval_catalog as catalog
from monocular_osm.pipeline import PipelineConfig
from monocular_osm.stages import assisted


def audit(index=0):
    return {"call_index": index, "status": "complete", "request": {"center": [1., 2.]},
        "ref_ids": np.array(["mapillary:1", "mapillary:2"]), "ref_latlon": np.array([[1., 2.], [1., 2.1]]),
        "raw_similarities": np.array([[.1, .2], [.3, .4]], np.float32),
        "query_indices": np.array([0, 1]), "query_times": np.array([0., 1.]),
        "ref_embeddings_sha256": "a" * 64, "query_pixels_sha256": "b" * 64}


def fixture(tmp_path, verified=True):
    root = tmp_path / "out/children/execution/expanded"
    root.mkdir(parents=True)
    request = {"kind": "expanded", "execution_id": "execution", "video_sha256": "c" * 64,
               "source_sha256": {"source.py": "d" * 64}, "config": {"vo_start_sec": 0., "vo_end_sec": 1.}}
    (root / "request.json").write_text(json.dumps(request))
    worker._save_audits([audit()], root)
    closure = {"all_tracked_owned_births_gone": verified, "ownership_scope": "private fake observed actors"}
    return root, request, closure


def published_parent_fixture(tmp_path):
    root, request, closure = fixture(tmp_path)
    output = tmp_path / 'out'
    record = catalog.write(output / 'retrieval_catalogs/execution/expanded.json', catalog.collect(root, request, closure))
    audit = {'reached_stages': [], 'persisted_retrieval_catalogs': {
        'expanded': {'status': 'closure_verified', 'terminal_authoritative': True, 'artifact': record}}}
    result = {'assisted_inputs': {'execution_id': 'execution', 'video': {'sha256': request['video_sha256']},
        'source_sha256': request['source_sha256'], 'registered_window': [0., 1.]}, 'assisted_audit': audit,
        'answer': {'route_latlon': [[1., 2.], [1., 2.1]]}, 'provenance': {'assisted_audit': deepcopy(audit)}}
    (output / 'result.json').write_text(json.dumps(result))
    return root, output, SimpleNamespace(cfg=SimpleNamespace(output_dir=output),
        assisted_child_root=output / 'children/execution', result=result)


def test_every_persisted_alias_is_retained_without_child_seal_or_response(tmp_path):
    root, request, closure = fixture(tmp_path)
    scope = root / "metadata_admission"
    scope.mkdir()
    worker._save_audits([audit()], scope)
    value = catalog.collect(root, request, closure)
    assert value["terminal_authoritative"] and value["child_seal"]["status"] == "absent"
    assert [r["status"] for r in value["pairs"]] == ["complete_pair", "complete_pair"]
    assert value["logical_calls"] == [{"logical_id": "expanded:execution:0000", "physical_pair_indices": [0, 1],
        "scope_aliases": ["metadata_admission/retrieval_audit", "retrieval_audit"], "conflict": False}]
    assert not value["RAM_only_or_full_acquisition_completeness_claim"] and not value["array_values_checked"]
    path = tmp_path / "out/retrieval_catalogs/execution/expanded.json"
    catalog.write(path, value)
    with pytest.raises(ValueError):
        catalog.write(path, value)


@pytest.mark.parametrize('index', ['absent', None])
def test_kv_hit_payload_has_writer_identity_without_invented_acquisition_index(tmp_path, index):
    root, request, closure = fixture(tmp_path)
    payload = audit()
    # Real KV hit paths clear/repopulate this acquisition dictionary. The
    # persistence writer must supply its own identity, without filling the gap.
    payload.pop('call_index')
    if index is None:
        payload['call_index'] = None
    worker._save_audits([payload], root)
    metadata = catalog.read(root / 'retrieval_audit/0000.json')
    assert metadata['persisted_writer']['ordinal'] == 0
    assert ('call_index' in metadata) is (index is None)
    row = catalog.collect(root, request, closure)['pairs'][0]
    assert row['status'] == 'complete_pair' and row['logical_id'] == 'expanded:execution:0000'
    assert row['acquisition_call_index'] == {'present': index is None, 'value': None}


@pytest.mark.parametrize('fault', ['missing_writer', 'ordinal', 'scope', 'array_path'])
def test_logical_identity_requires_writer_record_not_filename_only(tmp_path, fault):
    root, request, closure = fixture(tmp_path)
    path = root / 'retrieval_audit/0000.json'
    metadata = catalog.read(path)
    if fault == 'missing_writer': metadata.pop('persisted_writer')
    elif fault == 'ordinal': metadata['persisted_writer']['ordinal'] = 0.0
    else: metadata['persisted_writer'][fault] = str(root / 'wrong')
    path.write_text(json.dumps(metadata))
    row = catalog.collect(root, request, closure)['pairs'][0]
    assert row['status'] == 'incomplete' and row['logical_id'] is None


@pytest.mark.parametrize("fault", ["tensor", "typed_header", "call_index"])
def test_alias_conflicts_or_invalid_call_identity_never_disappear(tmp_path, fault):
    root, request, closure = fixture(tmp_path)
    scope = root / "metadata_admission"
    scope.mkdir()
    worker._save_audits([audit()], scope)
    path = scope / "retrieval_audit/0000.json"
    value = json.loads(path.read_text())
    if fault == "tensor":
        value["ref_embeddings_sha256"] = "e" * 64
    elif fault == "typed_header":
        value["array_artifact"]["arrays"]["ref_ids"]["sha256"] = "f" * 64
    else:
        value["call_index"] = 3
    path.write_text(json.dumps(value))
    result = catalog.collect(root, request, closure)
    assert len(result["pairs"]) == 2
    if fault == "call_index":
        assert result["pairs"][0]["status"] == "incomplete" and result["pairs"][0]["logical_id"] is None
    else:
        assert all(r["status"] == "conflict" for r in result["pairs"])
        assert result["logical_calls"][0]["conflict"]


def test_orphan_and_incomplete_pairs_are_explicit_not_filtered(tmp_path):
    root, request, closure = fixture(tmp_path)
    directory = root / "retrieval_audit"
    (directory / "0001.npz").write_bytes(b"unfinished standalone array bytes")
    (directory / "0002.json").write_text("{}")
    meta = json.loads((directory / "0000.json").read_text())
    meta["array_artifact"]["sha256"] = "0" * 64
    (directory / "0000.json").write_text(json.dumps(meta))
    result = catalog.collect(root, request, closure)
    assert len(result["pairs"]) == 3
    assert [r["status"] for r in result["pairs"]][1:] == ["orphan", "orphan"]
    assert result["pairs"][0]["issues"] and not result["child_status_or_selection_promoted"]


@pytest.mark.parametrize("changed", ["npz", "metadata", "request", "new_member", "new_seal"])
def test_final_drift_prevents_catalog_publication(tmp_path, changed):
    root, request, closure = fixture(tmp_path)
    value = catalog.collect(root, request, closure)
    if changed == "new_member":
        path = root / "retrieval_audit/0001.npz"
    elif changed == "new_seal":
        path = root / "child_frozen.json"
    else:
        path = root / {"npz": "retrieval_audit/0000.npz", "metadata": "retrieval_audit/0000.json", "request": "request.json"}[changed]
    path.write_bytes(b"changed after snapshot")
    target = tmp_path / "catalog.json"
    with pytest.raises(ValueError):
        catalog.write(target, value)
    assert not target.exists()


def test_parent_freeze_binds_unsealed_pair_and_refuses_late_drift(tmp_path):
    root, request, closure = fixture(tmp_path)
    output = tmp_path / "out"
    record = catalog.write(output / "retrieval_catalogs/execution/expanded.json", catalog.collect(root, request, closure))
    ctx = SimpleNamespace(cfg=SimpleNamespace(output_dir=output), assisted_child_root=output / "children/execution",
        result={"assisted_inputs": {"execution_id": "execution", "video": {"sha256": request["video_sha256"]},
                                   "source_sha256": request["source_sha256"], "registered_window": [0., 1.]},
                "assisted_audit": {"reached_stages": [], "persisted_retrieval_catalogs": {
                    "expanded": {"status": "closure_verified", "terminal_authoritative": True, "artifact": record}}}})
    assisted.freeze(ctx)
    frozen = json.loads((output / "frozen_prediction.json").read_text())
    assert {"children/execution/expanded/request.json", "children/execution/expanded/retrieval_audit/0000.json",
            "children/execution/expanded/retrieval_audit/0000.npz", "retrieval_catalogs/execution/expanded.json"} <= {
                k.replace("\\", "/") for k in frozen["artifacts"]}
    (output / "frozen_prediction.json").unlink()
    (root / "retrieval_audit/0000.npz").write_bytes(b"late mutation")
    with pytest.raises(ValueError):
        assisted.freeze(ctx)
    assert not (output / "frozen_prediction.json").exists()


def test_unbudgeted_worker_keeps_status_but_never_claims_verified_closure(tmp_path, monkeypatch):
    video = tmp_path / "video"; video.write_bytes(b"private")
    cfg = PipelineConfig(url="", city="Private", data_dir=tmp_path / "data", output_dir=tmp_path / "out")
    parent = SimpleNamespace(cfg=cfg, video_path=video, assisted_child_root=cfg.output_dir / "children/execution",
        assisted_evidence={}, result={"assisted_inputs": {"execution_id": "execution", "video": {"sha256": worker.file_sha256(video)}, "source_sha256": {}}})
    monkeypatch.setattr(worker, "_commit_headroom", lambda: 9 * 1024**3)
    def spawn(command, **kwargs):
        root = Path(command[-1]).parent
        worker._save_audits([audit()], root)
        (root / "response.json").write_text(json.dumps({"status": "unavailable"}))
        return SimpleNamespace(wait=lambda: 0)
    monkeypatch.setattr(worker.subprocess, "Popen", spawn)
    assert worker.run_child("baseline", parent)["status"] == "unavailable"
    row = parent.assisted_retrieval_catalogs["baseline"]
    assert row["status"] == "closure_unverified" and row["terminal_authoritative"] is False
    assert catalog.read(row["artifact"]["path"])["actor_closure"]["all_tracked_owned_births_gone"] is False


def test_catalog_failure_is_explicit_without_changing_child_selection_inputs(tmp_path, monkeypatch):
    root, request, closure = fixture(tmp_path)
    parent = SimpleNamespace(cfg=SimpleNamespace(output_dir=tmp_path / "out"),
        assisted_evidence={"selected": "baseline"}, result={"answer": "unchanged"})
    cfg = SimpleNamespace(output_dir=root)
    def cannot_publish(*_):
        raise OSError("private publication failure")
    monkeypatch.setattr(catalog, "write", cannot_publish)
    worker._record_terminal_retrieval_catalog(parent, cfg, request, closure)
    record = parent.assisted_retrieval_catalogs["expanded"]
    assert record["status"] == "unavailable" and record["artifact"] is None
    assert record["actor_closure"]["all_tracked_owned_births_gone"] is True
    assert "OSError" in record["error"] and not record["terminal_authoritative"]
    assert parent.result == {"answer": "unchanged"} and parent.assisted_evidence == {"selected": "baseline"}


def test_unverified_catalog_does_not_freeze_mutable_child_evidence(tmp_path):
    root, request, closure = fixture(tmp_path, verified=False)
    output = tmp_path / "out"
    record = catalog.write(output / "retrieval_catalogs/execution/expanded.json", catalog.collect(root, request, closure))
    ctx = SimpleNamespace(cfg=SimpleNamespace(output_dir=output), assisted_child_root=output / "children/execution",
        result={"assisted_inputs": {"execution_id": "execution", "video": {"sha256": request["video_sha256"]},
                                   "source_sha256": request["source_sha256"], "registered_window": [0., 1.]},
                "assisted_audit": {"reached_stages": [], "persisted_retrieval_catalogs": {
                    "expanded": {"status": "closure_unverified", "terminal_authoritative": False, "artifact": record}}}})
    (root / "retrieval_audit/0000.npz").write_bytes(b"a still-unverified descendant could write")
    assisted.freeze(ctx)
    frozen = json.loads((output / "frozen_prediction.json").read_text())
    assert {k.replace("\\", "/") for k in frozen["artifacts"]} == {"retrieval_catalogs/execution/expanded.json"}


@pytest.mark.parametrize('cap', ['file_bytes', 'total_bytes', 'pairs', 'entries'])
def test_optional_catalog_caps_decline_without_partial_artifact(tmp_path, monkeypatch, cap):
    root, request, closure = fixture(tmp_path)
    if cap == 'file_bytes': monkeypatch.setattr(catalog, 'MAX_FILE_BYTES', 1)
    elif cap == 'total_bytes': monkeypatch.setattr(catalog, 'MAX_HASH_BYTES', 1)
    elif cap == 'pairs': monkeypatch.setattr(catalog, 'MAX_PAIRS', 0)
    else: monkeypatch.setattr(catalog, 'MAX_ENTRIES', 0)
    parent = SimpleNamespace(cfg=SimpleNamespace(output_dir=tmp_path / 'out'))
    worker._record_terminal_retrieval_catalog(parent, SimpleNamespace(output_dir=root), request, closure)
    row = parent.assisted_retrieval_catalogs['expanded']
    assert row['status'] == 'unavailable' and row['artifact'] is None and not row['terminal_authoritative']
    assert not list((tmp_path / 'out/retrieval_catalogs').rglob('*.json'))


def test_optional_time_allowance_is_shared_across_phases_and_preserves_publication_reserve():
    clock = SimpleNamespace(now=0.)
    budget = SimpleNamespace(clock=lambda: clock.now, deadline=100., finalization_reserve_s=8.)
    limit = catalog.Limit(budget)
    assert limit.maximum_s == 2. and limit.publication_boundary == 94.
    with limit.phase() as check:
        clock.now += 1.25; check()
    with pytest.raises(catalog.Unavailable, match='time budget'):
        with limit.phase() as check:
            clock.now += .8; check()
    with pytest.raises(catalog.Unavailable):
        with limit.phase(): pytest.fail('shared time allowance reset')
    other = catalog.Limit(budget); clock.now = 94.
    with pytest.raises(catalog.Unavailable):
        with other.phase(): pytest.fail('mandatory publication reserve was consumed')


def test_metadata_bytes_must_match_the_previously_bound_hash(tmp_path):
    path = tmp_path / 'public.json'; path.write_text('{"value":1}')
    expected = catalog.sha(path)
    path.write_text('{"value":2}')
    with pytest.raises(ValueError, match='between hash and read'):
        catalog.read(path, expected_hash=expected)


def test_final_catalog_verification_follows_mandatory_hash_callbacks(tmp_path, monkeypatch):
    root, output, ctx = published_parent_fixture(tmp_path)
    path = root / 'retrieval_audit/0000.npz'; original = assisted.file_sha256; changed = []
    def mutate(p):
        if Path(p) == output / 'result.json' and not changed:
            path.write_bytes(b'changed after mandatory result hash started'); changed.append(True)
        return original(p)
    monkeypatch.setattr(assisted, 'file_sha256', mutate)
    with pytest.raises(ValueError, match='terminal retrieval evidence changed'):
        assisted.freeze(ctx)
    assert changed and not (output / 'frozen_prediction.json').exists()


def test_budget_downgrade_synchronizes_all_audit_copies_and_final_mandatory_hash(tmp_path):
    _, output, ctx = published_parent_fixture(tmp_path)
    before = deepcopy(ctx.result['answer'])
    ctx.runtime_budget = SimpleNamespace(clock=lambda: 94., deadline=100., finalization_reserve_s=8.)
    assisted.freeze(ctx)
    saved = json.loads((output / 'result.json').read_text())
    standalone = json.loads((output / 'assisted_audit.json').read_text())
    assert saved['assisted_audit'] == saved['provenance']['assisted_audit'] == standalone
    row = standalone['persisted_retrieval_catalogs']['expanded']
    assert row['status'] == 'unavailable' and row['artifact'] is None and not row['terminal_authoritative']
    assert saved['answer'] == before
    frozen = json.loads((output / 'frozen_prediction.json').read_text())
    assert frozen['prediction_available']
    assert frozen['artifacts']['result.json'] == catalog.sha(output / 'result.json')
    assert frozen['artifacts']['assisted_audit.json'] == catalog.sha(output / 'assisted_audit.json')
    assert not any('retrieval_catalogs' in name or 'retrieval_audit' in name for name in frozen['artifacts'])
