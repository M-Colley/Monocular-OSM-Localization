"""Pure synthetic road signs; no clip IDs, coordinates from labels or IO."""
from copy import deepcopy
import json
import pytest
from monocular_osm.road_sign_clues import (
    RoadSignConfig, build_motorway_junction_inventory, extract_road_sign_clues,
    match_motorway_junctions, plan_motorway_junction_inventory,
)


def _row(text, *, t=10., confidence=.99, box=(100.,100.,140.,120.)):
    return {"text":text,"t_sec":t,"confidence":confidence,"bbox":list(box)}


def _clues():
    return extract_road_sign_clues([_row("EXIT"),_row("12A",box=(100.,120.,140.,140.))],[480,640])


def _plan():
    return plan_motorway_junction_inventory("Synthetic City",[1.,2.,1.2,2.2],bbox_source={
        "city":"Synthetic City","scope":"named_city_bbox","source_kind":"public_geocode","source_sha256":"a"*64})


def _body(elements): return json.dumps({"elements":elements}).encode()


def _node(identifier,ref,*,lat=2.1,lon=1.1,kind="node"):
    row = {"id":identifier,"type":kind,"tags":{"highway":"motorway_junction","ref":ref,"name":"Synthetic Junction"}}
    row.update({"lat":lat,"lon":lon} if kind == "node" else {"center":{"lat":lat,"lon":lon}})
    return row


def test_same_frame_native_exit_pair_keeps_literal_target_without_camera_fix():
    clues = _clues()
    assert [row["ref"] for row in clues["observations"]] == ["12A"]
    assert clues["observations"][0]["pairing"] == "same_frame_nearby_native_boxes"
    assert clues["selected_answer"] is None and clues["camera_fix"] is None
    assert "not camera position" in clues["interpretation"]


@pytest.mark.parametrize("text,expected",[("EXIT7B",["7B"]),("Exit 7 B",["7B"]),("EXIT 7A;7B",["7A","7B"])])
def test_inline_exit_refs_and_explicit_alternatives_are_retained(text,expected):
    result = extract_road_sign_clues([_row(text)],[480,640])
    assert [row["ref"] for row in result["observations"]] == expected


@pytest.mark.parametrize("fault",["anchor_confidence","ref_confidence","different_frame","far_box","missing_box"])
def test_unproven_pairs_are_not_promoted(fault):
    rows = [_row("EXIT"),_row("12A",box=(100.,120.,140.,140.))]
    if fault == "anchor_confidence": rows[0]["confidence"] = .899
    elif fault == "ref_confidence": rows[1]["confidence"] = .899
    elif fault == "different_frame": rows[1]["t_sec"] += .001
    elif fault == "far_box": rows[1]["bbox"] = [400.,300.,440.,320.]
    else: rows[0]["bbox"] = None
    assert extract_road_sign_clues(rows,[480,640])["observations"] == []


def test_nearby_ambiguous_readings_are_all_retained_without_substitution():
    rows = [_row("EXIT"),_row("12A",box=(100.,120.,140.,140.)),_row("128",box=(140.,120.,180.,140.))]
    clues = extract_road_sign_clues(rows,[480,640])
    assert [row["ref"] for row in clues["observations"]] == ["12A","128"]


def test_route_lists_addresses_phones_dates_and_direction_context_cannot_become_exits():
    rows = [_row(text) for text in ("75 65","I-75","A1058","30","1775","678-672-1300","07/25/2023","NORTH","SE","Greenville")]
    clues = extract_road_sign_clues(rows,[480,640])
    assert not clues["observations"] and not clues["exit_anchors"]
    assert {row["text"] for row in clues["context"]} == {row["text"] for row in rows}
    assert all(not row["usable_as_camera_heading"] and not row["usable_as_camera_position"] for row in clues["context"])
    with_exit = extract_road_sign_clues([_row("EXIT"),_row("75 65")],[480,640])
    assert not with_exit["observations"]


def test_pair_and_input_budgets_refuse_before_unbounded_pairing():
    rows = [_row("EXIT"),_row("12"),_row("13")]
    with pytest.raises(ValueError,match="pair budget"):
        extract_road_sign_clues(rows,[480,640],config=RoadSignConfig(max_pairs=1))
    with pytest.raises(ValueError,match="detection budget"):
        extract_road_sign_clues(iter(rows),[480,640],config=RoadSignConfig(max_detections=2))


def test_clue_extraction_is_pure_and_does_not_mutate_inputs(monkeypatch):
    rows = [_row("EXIT"),_row("12A")]
    original = deepcopy(rows)
    def forbid(*args,**kwargs): pytest.fail("pure road sign code attempted IO")
    monkeypatch.setattr("builtins.open",forbid)
    monkeypatch.setattr("socket.create_connection",forbid)
    extract_road_sign_clues(rows,[480,640])
    assert rows == original


def test_public_query_plans_all_junctions_without_filtering_by_observed_ref():
    plan = _plan()
    assert '["highway"="motorway_junction"]["ref"]' in plan["query"]
    assert '"ref"=' not in plan["query"] and "12A" not in plan["query"]
    assert plan["bbox_wsen"] == [1.,2.,1.2,2.2]
    assert plan["endpoint"] == "https://maps.mail.ru/osm/tools/overpass/api/interpreter"


def test_inventory_retains_all_osm_refs_and_every_exact_ambiguous_match():
    inventory = build_motorway_junction_inventory(_body([_node(1,"12A;12B"),_node(2,"12A"),
        _node(3,"12AB"),_node(4,"other"),_node(5,"12A",kind="way")]),_plan())
    assert len(inventory["junctions"]) == 5
    matches = match_motorway_junctions(_clues(),inventory)
    assert {row["id"] for row in matches["matches"][0]["candidates"]} == {"node/1","node/2","way/5"}
    assert matches["selected_answer"] is None and matches["camera_fix"] is None
    assert matches["matches"][0]["camera_fix"] is None


def test_missing_inventory_is_unknown_and_does_not_become_zero_matches():
    result = match_motorway_junctions(_clues(),None)
    assert result["status"] == "unavailable" and "matches" not in result
    assert len(result["observations"]) == 1


def test_junction_name_and_directions_never_substitute_for_exit_reference():
    node = _node(1,"99")
    node["tags"].update(name="12A",destination="12A",exit_to="12A")
    inventory = build_motorway_junction_inventory(_body([node]),_plan())
    result = match_motorway_junctions(_clues(),inventory)
    assert result["matches"][0]["candidates"] == []


@pytest.mark.parametrize("fault",["remark","missing_elements","duplicate_conflict","bad_location","changed_plan"])
def test_incomplete_or_conflicting_metadata_cannot_publish_inventory(fault):
    plan = _plan()
    payload = {"elements":[_node(1,"12A")]}
    if fault == "remark": payload["remark"] = "runtime timeout"
    elif fault == "missing_elements": payload.pop("elements")
    elif fault == "duplicate_conflict": payload["elements"].append(_node(1,"12B"))
    elif fault == "bad_location": payload["elements"][0]["lat"] = 91.
    else: plan["query"] += '["ref"="12A"]'
    with pytest.raises(ValueError): build_motorway_junction_inventory(json.dumps(payload).encode(),plan)


def test_frozen_observations_and_inventory_digests_are_checked():
    clues = _clues()
    inventory = build_motorway_junction_inventory(_body([_node(1,"12A")]),_plan())
    clues["observations"][0]["ref"] = "12B"
    with pytest.raises(ValueError,match="changed"): match_motorway_junctions(clues,inventory)
    clues = _clues()
    inventory["junctions"][0]["ref"] = "12B"
    with pytest.raises(ValueError,match="changed"): match_motorway_junctions(clues,inventory)


def test_map_matching_refuses_excessive_candidates_without_truncating_or_choosing(monkeypatch):
    from monocular_osm import road_sign_clues as core
    clues = _clues()
    inventory = build_motorway_junction_inventory(_body([_node(1,"12A"),_node(2,"12A")]),_plan())
    monkeypatch.setattr(core,"MAX_MATCH_CANDIDATES",1)
    with pytest.raises(ValueError,match="before candidate copies"):
        match_motorway_junctions(clues,inventory)


@pytest.mark.parametrize("value",[-.1,1.1,float('nan')])
def test_bad_confidence_settings_decline(value):
    with pytest.raises(ValueError): RoadSignConfig(exit_confidence=value)
