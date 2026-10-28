import json
from pathlib import Path
import threading
from urllib.parse import quote
import numpy as np
import pytest
from scripts import fetch_reference_metadata as r


@pytest.fixture(autouse=True)
def forbid_real_http(monkeypatch):
    import requests
    def forbidden(*args, **kwargs):
        pytest.fail("actual HTTP is forbidden in metadata unit tests")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


class Clock:
    def __init__(self): self.now = 0.; self.lock = threading.Lock()
    def time(self):
        with self.lock: return self.now
    def sleep(self, seconds):
        with self.lock: self.now += seconds


def plan(n=120):
    ids = np.array([f"mapillary:{i+1}" for i in range(n)])
    ll = np.column_stack([np.full(n, 42.), -83.+np.arange(n)*.0001])
    return r.plan_requests(ids, ll)


def ok(batch, token, allowance):
    return r.Response(200, json.dumps({"data": [{"id": value.split(":")[1]} for value in batch["reference_ids"]]}).encode())


def run(tmp_path, p=None, transport=ok, token="SECRET_MLY_TEST_VALUE"):
    clock = Clock()
    return r.run_batches(p or plan(), tmp_path/"attempts", token, transport=transport, clock=clock.time, sleep=clock.sleep)


def test_every_batch_id_attempt_and_nullable_result_is_retained(tmp_path):
    result = run(tmp_path)
    assert result["all_batches_successful"] and result["requests"] == 3
    assert len(result["records"]) == 120 and len(result["batches"]) == 3
    assert result["summary"]["available_images"] == 120
    assert result["summary"]["field_status_counts"]["sequence"]["absent"] == 120
    paths = list((tmp_path/"attempts").glob("*/request.json"))
    starts = sorted(r.read(path)["admission"]["start_elapsed_s"] for path in paths)
    assert all(b-a >= 1 for a, b in zip(starts, starts[1:]))
    for path in paths:
        record = r.read(path)
        assert "access_token" not in record["params"] and len(record["params"]["image_ids"].split(",")) <= 50
    with pytest.raises(FileExistsError): run(tmp_path)


@pytest.mark.parametrize("status", [400, 401, 403])
def test_first_batch_fatal_stops_all_requests_with_every_unknown_id_retained(tmp_path, status):
    calls = []
    def fail(batch, token, allowance):
        calls.append(batch["index"]); return r.Response(status, b'{"error":"unavailable"}')
    result = run(tmp_path, transport=fail)
    assert calls == [0] and result["requests"] == 1 and len(result["batches"]) == 3
    assert len(result["records"]) == result["summary"]["unknown_images"] == 120
    assert result["stop_reason"] == f"fatal_http_{status}"
    assert result["all_batches_successful"] is False


@pytest.mark.parametrize("response", [r.Response(503, b"temporary"), r.Response(None, b"", error="ReadTimeout")])
def test_one_transient_retry_is_preserved_with_fixed_global_backoff(tmp_path, response):
    calls = []
    def transport(batch, token, allowance):
        calls.append(batch["index"])
        return response if len(calls) == 1 else ok(batch, token, allowance)
    result = run(tmp_path, p=plan(3), transport=transport)
    assert calls == [0, 0] and result["requests"] == 2 and result["all_batches_successful"]
    first = r.read(tmp_path/"attempts/batch_0000_attempt_1/request.json")
    second = r.read(tmp_path/"attempts/batch_0000_attempt_2/request.json")
    assert second["admission"]["start_elapsed_s"]-first["admission"]["start_elapsed_s"] >= 2


def test_long_retry_after_and_exhausted_429_stop_instead_of_hopping_endpoints(tmp_path):
    result = run(tmp_path, transport=lambda *args: r.Response(429, b"limited", retry_after="61"))
    assert result["requests"] == 1 and result["stop_reason"] == "rate_limit_wait_exceeds_budget"
    response = r.read(tmp_path/"attempts/batch_0000_attempt_1/response.json")
    assert response["server_retry_after_s"] == 61
    other = tmp_path/"second"; other.mkdir()
    result = run(other, transport=lambda *args: r.Response(429, b"limited", retry_after="2"))
    assert result["requests"] == 2 and result["stop_reason"] == "rate_limit_retry_exhausted"


def test_missing_images_are_unknown_even_when_all_requests_succeed(tmp_path):
    result = run(tmp_path, transport=lambda *args: r.Response(200, b'{"data":[]}'))
    assert result["all_batches_successful"] and result["requests"] == 3
    assert result["summary"]["unknown_images"] == 120 and result["summary"]["available_images"] == 0


def test_bad_first_schema_is_not_a_license_to_change_fields_or_continue(tmp_path):
    result = run(tmp_path, transport=lambda *args: r.Response(200, b'{"data":[{"id":"999999"}]}'))
    assert result["requests"] == 1 and result["stop_reason"] == "first_batch_compatibility_failed"
    assert len(result["records"]) == 120 and result["summary"]["available_images"] == 0


def test_echoed_credentials_are_redacted_everywhere_and_not_parsed_as_metadata(tmp_path):
    token = "MLY|SENSITIVE_TEST_+/VALUE"
    body = json.dumps({"data": [{"id": "1", "sequence": token}],
        "url": "https://example.test?access_token="+quote(token, safe=""), "authorization": "OAuth "+token}).encode()
    result = run(tmp_path, p=plan(1), token=token, transport=lambda *args: r.Response(200, body))
    assert result["requests"] == 1 and result["summary"]["unknown_images"] == 1
    for path in (tmp_path/"attempts").rglob("*"):
        if path.is_file():
            assert token.encode() not in path.read_bytes() and quote(token, safe="").encode() not in path.read_bytes()
    outcome = r.read(tmp_path/"attempts/batch_0000_attempt_1/response.json")
    assert outcome["body_redacted_or_reencoded"] and outcome["decoded_body_sha256"] != outcome["stored_body_sha256"]


def test_budget_rejects_request_count_deadline_and_reserves_partial_terminal_body():
    clock = Clock(); budget = r.Budget(clock.time, clock.sleep)
    budget.used = r.PROFILE["max_total_decoded_body_bytes"]-17001
    admitted = budget.admit()
    assert admitted["allowance"] == 17001
    budget.finish(17001, 17001)
    assert budget.admit() is None and budget.used == r.PROFILE["max_total_decoded_body_bytes"]
    budget = r.Budget(clock.time, clock.sleep); budget.requests = 390
    assert budget.admit() is None
    budget = r.Budget(clock.time, clock.sleep); clock.sleep(1200)
    assert budget.admit() is None and budget.requests == 0


def test_deadline_preserves_unknowns_for_all_not_started_batches(tmp_path):
    clock = Clock()
    def transport(batch, token, allowance):
        clock.sleep(1200); return ok(batch, token, allowance)
    result = r.run_batches(plan(), tmp_path/"attempts", "SECRET", transport=transport, clock=clock.time, sleep=clock.sleep)
    assert result["requests"] == 1 and len(result["records"]) == 120
    assert result["summary"]["available_images"] == 50 and result["summary"]["unknown_images"] == 70


def test_http_dynamic_reads_never_decode_past_nonmultiple_terminal_allowance(monkeypatch):
    import requests
    calls = {}; payload = b"x"*20003
    class Raw:
        offset = 0
        def read(self, size, decode_content):
            assert decode_content is True
            calls.setdefault("sizes", []).append(size)
            chunk = payload[self.offset:self.offset+size]; self.offset += len(chunk); return chunk
    raw = Raw()
    class Reply:
        status_code = 200; headers = {}; raw = None
        def __enter__(self): self.raw = raw; return self
        def __exit__(self, *args): pass
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, endpoint, **kwargs): calls.update(endpoint=endpoint, kwargs=kwargs); return Reply()
    monkeypatch.setattr(requests, "Session", Session)
    response = r.http_get(plan(1)["batches"][0], "SECRET", 17001)
    assert len(response.body) == raw.offset == 17001 and calls["sizes"] == [16384, 617]
    assert response.error == "body_limit"
    assert calls["kwargs"]["headers"]["Authorization"] == "OAuth SECRET"
    assert "access_token" not in calls["kwargs"]["params"] and calls["kwargs"]["allow_redirects"] is False
    assert calls["kwargs"]["timeout"] == (10., 30.)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path/"original"; source.mkdir()
    ids = np.array(["mapillary:1", "mapillary:2", "mapillary:3"])
    ll = np.array([[42., -83.], [42., -83.001], [42.001, -83.]])
    np.savez(source/"gallery.npz", ref_ids=ids, ref_latlon=ll, never_load_query_or_scores=np.array([object()], dtype=object))
    r.write_new(source/"frozen.json", {"status": "complete", "scoring_has_not_run": True,
        "artifacts": {"gallery.npz": r.sha(source/"gallery.npz")}})
    monkeypatch.setattr(r, "source_identity", lambda *args: {})
    output = tmp_path/"metadata"
    result = r.prepare(source/"frozen.json", r.sha(source/"frozen.json"), "gallery.npz", output, "synthetic-pin")
    return source, output, result["plan_freeze_sha256"]


def use_mock_network(monkeypatch, callback=ok):
    original = r.run_batches
    clock = Clock()
    monkeypatch.setattr(r, "load_credential", lambda: ("SECRET_MLY_TEST_VALUE", "synthetic_test"))
    monkeypatch.setattr(r, "run_batches", lambda plan, root, token:
        original(plan, root, token, transport=callback, clock=clock.time, sleep=clock.sleep))


def test_prepare_freezes_only_id_geotag_membership_without_HTTP_or_query_arrays(prepared):
    source, output, digest = prepared
    freeze = r.verify_freeze(output/"plan_frozen.json", digest)
    record = r.read(output/"metadata_plan.json")
    assert freeze["http_has_not_run"] and record["reference_count"] == 3
    assert record["provenance"]["evidence_sha256"] == r.sha(source/"gallery.npz")
    assert not (output/"attempts").exists()


def test_mocked_complete_acquisition_freezes_plan_attempts_results_and_unknown_fields(prepared, monkeypatch):
    _, output, digest = prepared
    use_mock_network(monkeypatch)
    result = r.fetch(output, digest, "synthetic-pin")
    assert result["status"] == "complete" and result["requests"] == 1
    records = r.read(output/"all_reference_metadata.json")
    assert set(records) == {"mapillary:1", "mapillary:2", "mapillary:3"}
    freeze = r.verify_freeze(output/"acquisition_frozen.json", result["freeze_sha256"])
    assert freeze["all_ids_retained"] and freeze["selected_answer"] is None
    assert freeze["models_or_labels"] is False and freeze["metadata_is_city_inventory"] is False
    assert all(value["fields"]["sequence"] is None for value in records.values())
    before = r.sha(output/"acquisition_frozen.json")
    with pytest.raises(ValueError, match="resume"): r.fetch(output, digest, "synthetic-pin")
    assert r.sha(output/"acquisition_frozen.json") == before


def test_mocked_fatal_freezes_every_unknown_and_truthful_partial_status(prepared, monkeypatch):
    _, output, digest = prepared
    use_mock_network(monkeypatch, lambda *args: r.Response(403, b'{"error":"permission"}'))
    result = r.fetch(output, digest, "synthetic-pin")
    assert result["status"] == "partial_unknown" and result["summary"]["unknown_images"] == 3
    frozen = r.verify_freeze(output/"acquisition_frozen.json", result["freeze_sha256"])
    assert frozen["status"] == "partial_unknown" and frozen["all_ids_retained"]


def test_missing_token_cannot_send_or_create_attempts(prepared, monkeypatch):
    _, output, digest = prepared
    monkeypatch.setattr(r, "load_credential", lambda: ("", "missing"))
    monkeypatch.setattr(r, "run_batches", lambda *args: pytest.fail("network stage despite missing token"))
    with pytest.raises(ValueError, match="missing"): r.fetch(output, digest, "synthetic-pin")
    assert not (output/"attempts").exists() and not (output/"acquisition_frozen.json").exists()


def test_reblessed_plan_cannot_remove_ids_or_alter_request_fields(prepared, monkeypatch):
    _, output, _ = prepared
    plan = r.read(output/"metadata_plan.json")
    plan["batches"][0]["params"]["fields"] = "id"
    (output/"metadata_plan.json").write_text(json.dumps(plan))
    frozen = r.read(output/"plan_frozen.json"); frozen["artifacts"]["metadata_plan.json"] = r.sha(output/"metadata_plan.json")
    (output/"plan_frozen.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(r, "run_batches", lambda *args: pytest.fail("requests changed after plan"))
    with pytest.raises(ValueError, match="membership/requests"):
        r.fetch(output, r.sha(output/"plan_frozen.json"), "synthetic-pin")


def test_changed_source_after_transport_prevents_final_freeze(prepared, monkeypatch):
    _, output, digest = prepared
    use_mock_network(monkeypatch)
    calls = []
    def source(*args):
        calls.append(True)
        if len(calls) == 3: raise ValueError("changed metadata source")
        return {}
    monkeypatch.setattr(r, "source_identity", source)
    with pytest.raises(ValueError, match="changed metadata source"): r.fetch(output, digest, "synthetic-pin")
    assert (output/"all_reference_metadata.json").exists() and not (output/"acquisition_frozen.json").exists()


def test_original_input_changed_during_transport_cannot_be_frozen(prepared, monkeypatch):
    source, output, digest = prepared
    def mutate(batch, token, allowance):
        response = ok(batch, token, allowance)
        (source/"gallery.npz").write_bytes(b"changed source")
        return response
    use_mock_network(monkeypatch, mutate)
    with pytest.raises(ValueError, match="frozen member changed"): r.fetch(output, digest, "synthetic-pin")
    assert not (output/"acquisition_frozen.json").exists()


def test_missing_membership_or_bad_header_declines_before_numpy_load(tmp_path, monkeypatch):
    source = tmp_path/"source"; source.mkdir()
    np.savez(source/"bad.npz", ref_ids=np.array(["mapillary:1"], dtype=object), ref_latlon=np.zeros((1, 2)))
    r.write_new(source/"frozen.json", {"status": "complete", "scoring_has_not_run": True,
        "artifacts": {"bad.npz": r.sha(source/"bad.npz")}})
    monkeypatch.setattr(r.np, "load", lambda *args, **kwargs: pytest.fail("unsafe header materialized"))
    with pytest.raises(ValueError, match="typed allocation"):
        r.frozen_gallery(source/"frozen.json", r.sha(source/"frozen.json"), "bad.npz")
    (source/"frozen.json").write_text(json.dumps({"status": "complete", "scoring_has_not_run": True, "artifacts": {}}))
    with pytest.raises(ValueError, match="membership"):
        r.frozen_gallery(source/"frozen.json", r.sha(source/"frozen.json"), "bad.npz")


def test_foreign_executed_driver_and_final_source_digest_mismatch_decline(monkeypatch, tmp_path):
    monkeypatch.setattr(r.subprocess, "check_output", lambda args, **kwargs: "pin" if "rev-parse" in args else "")
    with pytest.raises(ValueError, match="source changed"): r.source_identity("pin", {})
    monkeypatch.setattr(r, "__file__", str(tmp_path/"foreign.py"))
    with pytest.raises(ValueError, match="foreign executed"): r.source_identity("pin")


@pytest.mark.parametrize("relative", ["Ground_Truth/route.json", "sCoReS.JSON", "nested/SCORE.JSONL",
    "offline_scores.json", "run2_official_registered_score.json", "score_run2.json", "SCORING_frozen.JSON"])
def test_label_score_members_are_rejected_before_hashing_their_bytes(tmp_path, monkeypatch, relative):
    forbidden = tmp_path/relative; forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_bytes(b"label values must never be read")
    frozen = tmp_path/"frozen.json"
    r.write_new(frozen, {"artifacts": {relative: "never-read"}})
    expected = r.sha(frozen); original = r.sha
    def spy(path):
        assert Path(path) != forbidden, "label bytes were hashed"
        return original(path)
    monkeypatch.setattr(r, "sha", spy)
    with pytest.raises(ValueError, match="outside metadata inputs"): r.verify_freeze(frozen, expected)


def test_forbidden_evidence_path_refused_before_any_manifest_or_evidence_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "sha", lambda *args: pytest.fail("hash before label evidence guard"))
    with pytest.raises(ValueError, match="outside metadata inputs"):
        r.frozen_gallery(tmp_path/"frozen.json", "irrelevant", "GROUND_TRUTH/route.npz")


def test_registry_fallback_stays_in_child_environment_and_records_origin_only(monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.delenv("MLY_TOKEN", raising=False)
    class Key:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    registry = SimpleNamespace(HKEY_CURRENT_USER="user", OpenKey=lambda *args: Key(),
        QueryValueEx=lambda *args: (" SECRET_FROM_TEST_REGISTRY ", 1))
    monkeypatch.setitem(sys.modules, "winreg", registry)
    # Only Windows consults its existing user registry; no write API exists in the fake.
    if r.os.name == "nt":
        token, origin = r.load_credential()
        assert token == "SECRET_FROM_TEST_REGISTRY" and origin == "HKCU_Environment"
        assert r.os.environ["MLY_TOKEN"] == token
        assert r.load_credential() == (token, "process_environment")


def test_resolved_alias_to_ground_truth_is_refused_before_member_hash(tmp_path, monkeypatch):
    target = tmp_path/"ground_truth"/"route.json"; target.parent.mkdir(); target.write_bytes(b"never read")
    alias = tmp_path/"benign.json"; alias.write_bytes(b"placeholder")
    frozen = tmp_path/"frozen.json"; r.write_new(frozen, {"artifacts": {"benign.json": "not-read"}})
    expected = r.sha(frozen); resolve = Path.resolve; original = r.sha
    monkeypatch.setattr(Path, "resolve", lambda path, *a, **kw: target if path == alias else resolve(path, *a, **kw))
    def spy(path):
        assert path != target, "resolved label member was hashed"
        return original(path)
    monkeypatch.setattr(r, "sha", spy)
    with pytest.raises(ValueError, match="outside metadata inputs"): r.verify_freeze(frozen, expected)


def test_final_artifacts_refuse_saved_scores_before_hashing_any_family_member(tmp_path, monkeypatch):
    (tmp_path/"owned.json").write_text("{}")
    (tmp_path/"offline_scores.json").write_text("never read")
    monkeypatch.setattr(r, "sha", lambda *args: pytest.fail("family member hashed before complete name validation"))
    with pytest.raises(ValueError, match="outside metadata inputs"): r.artifact_hashes(tmp_path)


def test_unexpected_saved_score_prevents_fetch_before_transport(prepared, monkeypatch):
    _, output, digest = prepared
    (output/"offline_scores.json").write_bytes(b"must never be read")
    monkeypatch.setattr(r, "run_batches", lambda *args: pytest.fail("HTTP before family label guard"))
    with pytest.raises(ValueError, match="outside metadata inputs"): r.fetch(output, digest, "synthetic-pin")
    assert not (output/"attempts").exists()
