"""Offline OSM-node adapter checks; synthetic XML uses the real GDAL OSM driver."""
import hashlib
from html import escape
import json
from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import box,mapping,Point

from monocular_osm import pbf_junction_inventory as pbf
from monocular_osm.road_sign_clues import extract_road_sign_clues


def _source(path):
    return {"verified":True,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size,
        "provider":"Synthetic public OSM fixture","source_url":"https://example.test/public-map.osm",
        "snapshot_date":"2026-01-01","license":"https://opendatacommons.org/licenses/odbl/1-0/"}


def _plan(path,*,region=None,city=None):
    return pbf.plan_pbf_junction_inventory("Synthetic City",[1.,2.,1.2,2.2],bbox_source={
        "city":"Synthetic City","scope":"named_city_bbox","source_kind":"public_geocode","source_sha256":"a"*64},
        extract_source=_source(path),extract_polygon={"geometry":mapping(region or box(.9,1.9,1.3,2.3)),"source_sha256":"b"*64},
        city_geometry=mapping(city) if city is not None else None)


def _node(identifier,lat,lon,tags):
    values=''.join(f'<tag k="{escape(k,quote=True)}" v="{escape(v,quote=True)}"/>' for k,v in tags.items())
    return f'<node id="{identifier}" lat="{lat}" lon="{lon}">{values}</node>'


def _file(tmp_path,nodes):
    path=tmp_path/"public.osm"
    path.write_text('<osm version="0.6">'+''.join(nodes)+'</osm>',encoding="utf-8")
    return path


def _read(path,plan,tmp_path):
    pyogrio=pytest.importorskip("pyogrio")
    if pyogrio.list_drivers().get("OSM") not in {"r","rw"}:pytest.skip("GDAL OSM reader unavailable")
    return pbf.read_city_junction_nodes(path,plan,tmp_path/"output",tmp_path/"gdal_tmp")


def test_real_osm_reader_preserves_typed_large_node_ids_all_tags_and_unnamed_junctions(tmp_path):
    path=_file(tmp_path,[_node(9007199254740993,2.1,1.1,{"highway":"motorway_junction","ref":"12A;12B",
        "name":'Straße "North" & South',"destination":"Everywhere"}),
        _node(9007199254740994,2.12,1.12,{"highway":"motorway_junction","ref":"99"}),
        _node(9007199254740995,2.12,1.12,{"shop":"supermarket","name":"Unrelated"})])
    plan=_plan(path);inventory=_read(path,plan,tmp_path)
    assert inventory["status"]=="complete" and inventory["complete_for_public_city_scope"]
    assert inventory["node_only"] and not inventory["ways_relations_included"] and not inventory["retail_inventory"]
    assert [row["id"] for row in inventory["junctions"]]==["node/9007199254740993","node/9007199254740994"]
    assert inventory["junctions"][0]["tags"]["name"]=='Straße "North" & South'
    assert inventory["junctions"][0]["normalized_refs"]==["12A","12B"]
    assert "name" not in inventory["junctions"][1]["tags"]
    assert hashlib.sha256((tmp_path/"output/node_osmconf.ini").read_bytes()).hexdigest()==plan["osm_config_sha256"]


def test_public_admin_polygon_scope_is_distinct_from_incomplete_enclosing_bbox(tmp_path):
    path=_file(tmp_path,[_node(1,2.1,1.05,{"highway":"motorway_junction","ref":"12A"}),
                          _node(2,2.1,1.18,{"highway":"motorway_junction","ref":"12A"})])
    plan=_plan(path,region=box(.9,1.9,1.16,2.3),city=box(1.,2.,1.15,2.2))
    assert plan["complete_public_scope_coverage"] and not plan["complete_bbox_coverage"]
    inventory=_read(path,plan,tmp_path)
    assert inventory["scope"]=="named_city_admin_polygon_junction_nodes"
    assert inventory["complete_for_public_city_scope"] and not inventory["complete_parent_bbox_coverage"]
    assert inventory["counts"]=={"bbox_node_rows":2,"outside_public_scope_rows":1,"unique_scope_nodes":1}
    assert inventory["junctions"][0]["id"]=="node/1"


def test_extract_missing_part_of_public_scope_remains_partial_not_zero_global_matches(tmp_path):
    path=_file(tmp_path,[_node(1,2.1,1.05,{"highway":"motorway_junction","ref":"12A"})])
    inventory=_read(path,_plan(path,region=box(.9,1.9,1.1,2.3)),tmp_path)
    assert inventory["status"]=="partial" and inventory["complete_within_extract"]
    assert not inventory["complete_for_public_city_scope"]
    clues=extract_road_sign_clues([{"text":"EXIT 12A","confidence":.99,"t_sec":1.,"bbox":[1.,1.,20.,10.]}],[100,100])
    targets=pbf.match_junction_targets(clues,inventory)
    assert targets["status"]=="partial" and len(targets["matches"][0]["candidates"])==1
    assert targets["camera_fix"] is None and targets["selected_answer"] is None


def test_real_points_only_scan_does_not_lose_late_nodes_to_other_layer_accumulation(tmp_path):
    path=tmp_path/"interleaved.osm"
    with path.open("w",encoding="utf-8") as handle:
        handle.write('<osm version="0.6">')
        for index in range(1,10006):handle.write(_node(index,2.1,1.1,{"shop":"yes"}))
        handle.write(_node(10006,2.1,1.1,{"highway":"motorway_junction","ref":"12A"}))
        for index in range(1,100006):handle.write(f'<way id="{index}"><nd ref="1"/><nd ref="2"/><tag k="highway" v="residential"/></way>')
        handle.write(_node(10007,2.12,1.12,{"highway":"motorway_junction","ref":"12B"}))
        handle.write('</osm>')
    inventory=_read(path,_plan(path),tmp_path)
    assert [row["id"] for row in inventory["junctions"]]==["node/10006","node/10007"]


def test_complete_empty_node_inventory_is_distinct_from_incomplete_extract(tmp_path):
    path=_file(tmp_path,[_node(1,2.1,1.1,{"shop":"supermarket"})])
    inventory=_read(path,_plan(path),tmp_path)
    assert inventory["status"]=="complete" and inventory["junctions"]==[]


def test_extract_hash_change_refuses_before_any_gdal_read(tmp_path,monkeypatch):
    pyogrio=pytest.importorskip("pyogrio.raw")
    path=_file(tmp_path,[]);plan=_plan(path);path.write_text('changed')
    monkeypatch.setattr(pyogrio,"read",lambda *a,**k:pytest.fail("changed source reached GDAL"))
    with pytest.raises(ValueError,match="hash changed"):_read(path,plan,tmp_path)
    assert not(tmp_path/"output").exists()


def test_gdal_configuration_is_bounded_and_restored_on_reader_failure(tmp_path,monkeypatch):
    pyogrio=pytest.importorskip("pyogrio");import pyogrio.raw
    path=_file(tmp_path,[]);plan=_plan(path)
    keys=("CPL_TMPDIR","OSM_MAX_TMPFILE_SIZE","OSM_INDEX_POINTS","OSM_USE_POINTS_INDEX","OSM_INDEX_WAYS","OSM_USE_WAYS_INDEX")
    before={key:pyogrio.get_gdal_config_option(key) for key in keys}
    def fail(*args,**kwargs):
        assert kwargs["max_features"]==10001 and kwargs["layer"]=="points"
        assert kwargs["TAGS_FORMAT"]=="JSON" and kwargs["MAX_TMPFILE_SIZE"]==32 and kwargs["INTERLEAVED_READING"]=="NO"
        assert kwargs["where"]==pbf.SQL_WHERE and kwargs["bbox"]==(1.,2.,1.2,2.2)
        assert pyogrio.get_gdal_config_option("CPL_TMPDIR")==str((tmp_path/"gdal_tmp").resolve())
        assert pyogrio.get_gdal_config_option("OSM_INDEX_POINTS")=="NO"
        raise RuntimeError("synthetic GDAL read failure")
    monkeypatch.setattr(pyogrio.raw,"read",fail)
    with pytest.raises(RuntimeError,match="synthetic"):_read(path,plan,tmp_path)
    assert {key:pyogrio.get_gdal_config_option(key) for key in keys}==before
    assert (tmp_path/"output/failed.json").exists() and not(tmp_path/"output/junction_inventory.json").exists()


@pytest.mark.parametrize("value",[1.,True,np.bool_(True),0,-1,"001","node/1","9223372036854775808"])
def test_unsafe_or_ambiguous_node_ids_decline(value):
    with pytest.raises(ValueError):pbf._node_identifier(value)


def test_poly_coverage_parser_respects_holes_and_requires_complete_end():
    raw=b'x\n1\n0 0\n4 0\n4 4\n0 4\n0 0\nEND\n!2\n1 1\n2 1\n2 2\n1 2\n1 1\nEND\nEND\n'
    from shapely.geometry import shape
    polygon=shape(pbf.parse_extract_poly(raw)["geometry"])
    assert polygon.covers(Point(.5,.5)) and not polygon.covers(Point(1.5,1.5))
    with pytest.raises(ValueError,match="incomplete"):pbf.parse_extract_poly(raw.rsplit(b'END',1)[0])


def test_cap_plus_one_refuses_instead_of_publishing_a_truncated_inventory(tmp_path,monkeypatch):
    raw=pytest.importorskip("pyogrio.raw")
    path=_file(tmp_path,[]);plan=_plan(path)
    def oversized(*args,**kwargs):
        assert kwargs["max_features"]==10001
        meta={"fields":["osm_id","highway","ref","all_tags"],"crs":"EPSG:4326","geometry_type":"Point"}
        return meta,None,np.empty(10001,object),[np.empty(10001,object) for i in range(4)]
    monkeypatch.setattr(raw,"read",oversized)
    with pytest.raises(ValueError,match="truncated reads"):_read(path,plan,tmp_path)
    assert not(tmp_path/"output/junction_inventory.json").exists()


def test_gdal_warning_refuses_instead_of_hiding_feature_loss(tmp_path,monkeypatch):
    raw=pytest.importorskip("pyogrio.raw");import warnings
    path=_file(tmp_path,[]);plan=_plan(path)
    def warn(*args,**kwargs):
        warnings.warn("features accumulated before consumption",RuntimeWarning)
        return {},None,[],[]
    monkeypatch.setattr(raw,"read",warn)
    with pytest.raises(RuntimeError,match="warning"):_read(path,plan,tmp_path)
    assert not(tmp_path/"output/junction_inventory.json").exists()


def test_changed_public_extract_during_read_does_not_gain_a_snapshot(tmp_path,monkeypatch):
    raw=pytest.importorskip("pyogrio.raw")
    path=_file(tmp_path,[_node(1,2.1,1.1,{"highway":"motorway_junction","ref":"12A"})]);plan=_plan(path)
    original=raw.read
    def changed(*args,**kwargs):
        result=original(*args,**kwargs)
        path.write_bytes(path.read_bytes()+b' ')
        return result
    monkeypatch.setattr(raw,"read",changed)
    with pytest.raises(ValueError,match="changed during"):_read(path,plan,tmp_path)


def test_observed_ref_filter_or_false_full_scope_claim_cannot_change_plan(tmp_path,monkeypatch):
    raw=pytest.importorskip("pyogrio.raw")
    path=_file(tmp_path,[]);plan=_plan(path)
    plan["where"]+=" AND ref = '12A'"
    monkeypatch.setattr(raw,"read",lambda *a,**k:pytest.fail("changed ref-filter plan reached GDAL"))
    with pytest.raises(ValueError,match="plan changed"):_read(path,plan,tmp_path)


@pytest.mark.parametrize("fault",["hstore","promoted_tag_mismatch","line_geometry","wrong_crs"])
def test_schema_tag_or_geometry_mismatches_do_not_become_typed_nodes(tmp_path,fault):
    from shapely.geometry import LineString
    path=_file(tmp_path,[]);plan=_plan(path)
    meta={"fields":["osm_id","highway","ref","all_tags"],"crs":"EPSG:4326","geometry_type":"Point"}
    tags=json.dumps({"highway":"motorway_junction","ref":"12A"})
    fields=[np.array(['1'],object),np.array(['motorway_junction'],object),np.array(['12A'],object),np.array([tags],object)]
    geometry=[Point(1.1,2.1).wkb]
    if fault=="hstore":fields[3][0]='"ref"=>"12A"'
    elif fault=="promoted_tag_mismatch":fields[2][0]='12B'
    elif fault=="line_geometry":geometry=[LineString([(1.1,2.1),(1.12,2.12)]).wkb]
    else:meta['crs']='EPSG:3857'
    with pytest.raises(ValueError):pbf._normalize_rows(meta,geometry,fields,plan)
