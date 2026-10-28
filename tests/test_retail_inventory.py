"""Fixed complete-city fallback checks, synthetic HTTP only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from monocular_osm import retail_inventory as retail


def _plan():
    return retail.plan_city_retail_tiles("Synthetic City", [1., 2., 1.2, 2.4], bbox_source={
        "city": "Synthetic City", "scope": "named_city_bbox", "source_kind": "public_geocode",
        "source_sha256": "a" * 64})


def _poi(identifier, *, kind="node", name="Aster", lat=2.1, lon=1.1):
    element = {"type": kind, "id": identifier, "tags": {"name": name, "shop": "supermarket"}}
    element.update({"lat": lat, "lon": lon} if kind == "node" else {"center": {"lat": lat, "lon": lon}})
    return element


def _response(elements=(), **extra):
    return {"status": 200, "complete": True, "body": json.dumps({"elements": list(elements), **extra}).encode()}


def test_plan_covers_the_entire_city_with_shared_boundaries_before_calls():
    plan = _plan()
    assert len(plan["tiles"]) == 4 and plan["grid"] == [2, 2]
    boxes = [tile["bbox_wsen"] for tile in plan["tiles"]]
    assert boxes == [[1.,2.,1.1,2.2],[1.1,2.,1.2,2.2],[1.,2.2,1.1,2.4],[1.1,2.2,1.2,2.4]]
    assert all('nwr["shop"]' in tile["query"] and 'out center tags;' in tile["query"] for tile in plan["tiles"])
    assert plan["max_elements_before_merge"] == 100_000 and plan["transport"] == "get"


def test_complete_tiles_deduplicate_only_osm_type_id_and_return_valid_offline_cache(tmp_path, monkeypatch):
    plan, calls = _plan(), []
    def request(endpoint, query):
        assert json.loads((tmp_path / "plan.json").read_text()) == plan
        calls.append(query)
        return _response([_poi(2), _poi(1, kind="way"), _poi(1)])
    monkeypatch.setattr(retail, "_request_tile", request)
    snapshot = retail.fetch_tiled_city_retail(plan, tmp_path)
    assert len(calls) == 4 and snapshot["complete_parent_bbox_coverage"]
    assert [poi["id"] for poi in snapshot["pois"]] == ["node/1", "node/2", "way/1"]
    assert snapshot["elements_before_merge"] == 12
    assert len(snapshot["tile_responses"]) == 4 and len(snapshot["response_sha256"]) == 64
    monkeypatch.setattr(retail, "_request_tile", lambda *args: pytest.fail("valid offline cache made a network request"))
    assert retail.fetch_tiled_city_retail(plan, tmp_path, offline=True) == snapshot


@pytest.mark.parametrize("fault", ["remark", "http", "truncated", "malformed"])
def test_any_failed_tile_keeps_inventory_unknown_and_preserves_all_four_attempts(tmp_path, monkeypatch, fault):
    plan, calls = _plan(), []
    bad_query = plan["tiles"][1]["query"]
    def request(endpoint, query):
        calls.append(query)
        if query != bad_query: return _response([_poi(1)])
        if fault == "remark": return _response([_poi(2)], remark="runtime error: timeout")
        if fault == "http": return {**_response(), "status": 504}
        if fault == "truncated": return {**_response(), "complete": False}
        return {"status": 200, "complete": True, "body": b"not JSON"}
    monkeypatch.setattr(retail, "_request_tile", request)
    monkeypatch.setattr(retail.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="all four"):
        retail.fetch_tiled_city_retail(plan, tmp_path)
    assert len(calls) == 5 and set(calls) == {tile["query"] for tile in plan["tiles"]}
    assert not (tmp_path / "inventory.json").exists() and not (tmp_path / "complete.json").exists()
    assert len(list(tmp_path.glob("*_attempts.json"))) == 4
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError, match="preserved"):
        retail.fetch_tiled_city_retail(plan, tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def test_conflicting_duplicate_ids_are_refused_instead_of_choosing_a_tile(tmp_path, monkeypatch):
    plan, calls = _plan(), []
    def request(endpoint, query):
        calls.append(query)
        return _response([_poi(1, name="Aster" if len(calls) < 3 else "Beryl")])
    monkeypatch.setattr(retail, "_request_tile", request)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        retail.fetch_tiled_city_retail(plan, tmp_path)
    assert len(calls) == 4 and not (tmp_path / "inventory.json").exists()


def test_aggregate_budget_counts_boundary_duplicates_before_any_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(retail, "MAX_ELEMENTS", 3)
    plan = _plan()
    monkeypatch.setattr(retail, "_request_tile", lambda *args: _response([_poi(1)]))
    monkeypatch.setattr(retail, "_merge_elements", lambda *args: pytest.fail("over-budget data reached merge"))
    with pytest.raises(ValueError, match="pre-merge"):
        retail.fetch_tiled_city_retail(plan, tmp_path)
    assert not (tmp_path / "inventory.json").exists()
    assert len(list(tmp_path.glob("*_attempts.json"))) == 4


@pytest.mark.parametrize("target", ["plan.json", "r0c0_attempt1.response", "inventory.json"])
def test_completed_cache_checks_plan_raw_responses_and_snapshot(tmp_path, monkeypatch, target):
    plan = _plan()
    monkeypatch.setattr(retail, "_request_tile", lambda *args: _response([_poi(1)]))
    retail.fetch_tiled_city_retail(plan, tmp_path)
    (tmp_path / target).write_bytes(b"changed")
    monkeypatch.setattr(retail, "_request_tile", lambda *args: pytest.fail("corrupt cache was overwritten by network"))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        retail.fetch_tiled_city_retail(plan, tmp_path)


@pytest.mark.parametrize("box", [[1.,2.,3.,2.1], [2.,2.,1.,2.1], [1.,float('nan'),1.1,2.1],
                                 [True,2.,1.1,2.1], [180.,2.,180.1,2.1]])
def test_invalid_or_unbounded_city_boxes_decline(box):
    source = _plan()["bbox_source"]
    with pytest.raises(ValueError, match="bounded full named-city"):
        retail.plan_city_retail_tiles("Synthetic City", box, bbox_source=source)


def test_changed_plan_cannot_select_or_skip_one_tile(tmp_path, monkeypatch):
    plan = _plan()
    plan["tiles"].pop()
    monkeypatch.setattr(retail, "_request_tile", lambda *args: pytest.fail("changed plan made a request"))
    with pytest.raises(ValueError, match="plan/config/query changed"):
        retail.fetch_tiled_city_retail(plan, tmp_path)


def test_existing_default_cache_outputs_are_preserved(tmp_path):
    from monocular_osm.poi_clues import fetch_city_retail
    snapshot = {"schema": 1, "city": "Synthetic City", "pois": [_poi(1)]}
    path = tmp_path / "synthetic-city.json"
    path.write_text(json.dumps(snapshot))
    original = path.read_bytes()
    assert fetch_city_retail("Synthetic City", tmp_path, offline=True) == snapshot
    assert path.read_bytes() == original


def test_http_body_is_streamed_and_stops_at_the_declared_limit(monkeypatch):
    from contextlib import nullcontext
    import requests
    import osmnx._http
    from monocular_osm import net
    monkeypatch.setattr(retail, "MAX_BODY_BYTES", 128)
    monkeypatch.setattr(net, "prefer_reachable_host", lambda endpoint: nullcontext())
    monkeypatch.setattr(osmnx._http, "_config_dns", lambda endpoint: None)
    seen = []
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): seen.append("closed")
        def iter_content(self, chunk_size):
            for number in range(3):
                seen.append(number)
                yield b"x" * 100
    def get(endpoint, **kwargs):
        assert kwargs["stream"] is True and kwargs["timeout"] == (8,110)
        assert kwargs["params"] == {"data": "fixed public query"}
        return Response()
    monkeypatch.setattr(requests, "get", get)
    response = retail._request_tile(retail.ENDPOINT, "fixed public query")
    assert len(response["body"]) == 128 and response["complete"] is False
    assert seen == [0,1,"closed"]
