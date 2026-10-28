import json
import sqlite3
import struct
import zlib
from pathlib import Path

import pytest
from shapely.geometry import LineString

from monocular_osm import road_geometry as road


class Guard:
    def __init__(self): self.calls = 0
    def check(self, **kwargs): self.calls += 1


def row(identity="41", tags=None, coordinates=None):
    return road.checked_way(identity, tags or {"highway": "residential"},
        LineString(coordinates or [(-83., 42.), (-82.99, 42.01), (-82.98, 42.02)]).wkb)


@pytest.mark.parametrize("value", road.PHYSICAL_HIGHWAYS)
def test_physical_classes_keep_access_and_semantic_tags(value):
    tags = {"highway": value, "access": "no", "motor_vehicle": "private", "oneway": "-1",
        "service": "driveway", "bridge": "yes", "layer": "2", "source": "survey", "note": 'quote"雪'}
    result = row(tags=tags)
    assert json.loads(result[2]) == tags


@pytest.mark.parametrize("tags", [{"highway": "footway"}, {"highway": "construction"},
    {"highway": "proposed"}, {"highway": "pedestrian", "motor_vehicle": "yes"},
    {"highway": "residential", "area": "yes"}, {}])
def test_fixed_nonmotor_or_area_predicate(tags):
    assert not road.physical_road(tags)


@pytest.mark.parametrize("identity", ["0", "-1", "1.0", "01", "bad", str(2**63)])
def test_way_ids_are_actual_positive_integers(identity):
    with pytest.raises(ValueError): row(identity)


def test_full_intermediate_vertices_and_closed_road_are_not_simplified():
    points = [(-83., 42.), (-82.9, 42.01), (-82.8, 42.), (-83., 42.)]
    result = row(coordinates=points)
    assert result[3] == LineString(points).wkb
    assert result[4] == 4


def test_index_full_geometry_tags_bounds_and_rtree(tmp_path):
    guard = Guard(); rows = [row(), row("42", coordinates=[(-84., 41.), (-84.1, 41.1)])]
    target = tmp_path/"roads.sqlite"
    report = road.build_index(iter(rows), target, guard)
    with sqlite3.connect(target) as db:
        assert db.execute("SELECT osm_id,wkb,n_vertices FROM ways ORDER BY osm_id").fetchall() == [(r[0], r[3], r[4]) for r in rows]
        assert db.execute("SELECT osm_id FROM way_bounds WHERE minx<=-82.98 AND maxx>=-83 AND miny<=42.02 AND maxy>=42").fetchall() == [(41,)]
    assert report["counts"]["ways"] == 2 and report["counts"]["vertices"] == 5
    assert report["topology"] is False and guard.calls >= 2


def test_duplicate_ids_refuse_and_preserve_partial_without_overwrite(tmp_path):
    target = tmp_path/"roads.sqlite"
    with pytest.raises(sqlite3.IntegrityError): road.build_index(iter([row(), row()]), target, Guard())
    with pytest.raises(ValueError, match="exists"): road.build_index(iter([row()]), target, Guard())


def test_bound_exceeded_refuses_not_truncates(tmp_path, monkeypatch):
    monkeypatch.setitem(road.PROFILE, "max_ways", 1)
    with pytest.raises(ValueError, match="inventory bound"):
        road.build_index(iter([row(), row("42")]), tmp_path/"roads.sqlite", Guard())


@pytest.mark.parametrize("name", ["offline_scores.json", "SCORES.JSON", "registered-scoring.jsonl", "ground_truth/roads.pbf",
    "ground_truth.json", "GT.json", "query-labels.txt", "raw-oxts.pbf"])
def test_forbidden_before_digest(tmp_path, name):
    path = tmp_path/name; path.parent.mkdir(exist_ok=True); path.write_text("do not read")
    with pytest.raises(ValueError, match="forbidden"): road.file_sha(path)


def test_resource_check_uses_true_commit_and_never_weakens_floor(tmp_path):
    temporary = tmp_path/"temporary"; temporary.mkdir()
    guard = road.ResourceGuard(tmp_path, temporary, snapshot=lambda: {
        "physical_available": 20*road.GIB, "true_commit_available": 7*road.GIB, "rss": 10_000})
    with pytest.raises(ValueError, match="true_commit"): guard.check(force=True)


def test_temporary_directory_outside_family_refused_before_resource_query(tmp_path):
    output, other = tmp_path/"output", tmp_path/"other"
    output.mkdir(); other.mkdir()
    with pytest.raises(ValueError, match="inside owned"):
        road.ResourceGuard(output, other, snapshot=lambda: pytest.fail("must not sample"))


def test_output_and_temporary_disk_budgets_are_separate(tmp_path, monkeypatch):
    from types import SimpleNamespace
    temporary = tmp_path/"temporary"; temporary.mkdir()
    monkeypatch.setattr(road.shutil, "disk_usage", lambda _: SimpleNamespace(free=100*road.GIB))
    snapshot = lambda: {"physical_available": 20*road.GIB, "true_commit_available": 20*road.GIB, "rss": 100}
    monkeypatch.setitem(road.PROFILE, "max_output_bytes", 8)
    (tmp_path/"data.bin").write_bytes(b"x"*9)
    with pytest.raises(ValueError, match="output disk"):
        road.ResourceGuard(tmp_path, temporary, snapshot=snapshot).check(force=True)
    (tmp_path/"data.bin").unlink()
    monkeypatch.setitem(road.PROFILE, "max_temp_bytes", 8)
    (temporary/"spill.bin").write_bytes(b"x"*9)
    with pytest.raises(ValueError, match="temporary disk"):
        road.ResourceGuard(tmp_path, temporary, snapshot=snapshot).check(force=True)


def synthetic_osm(path):
    path.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="synthetic">
<node id="1" lat="42.0" lon="-83.0"/>
<node id="2" lat="42.01" lon="-82.99"/>
<node id="3" lat="42.02" lon="-82.98"/>
<way id="101"><nd ref="1"/><nd ref="2"/><nd ref="3"/>
<tag k="highway" v="service"/><tag k="access" v="private"/><tag k="oneway" v="-1"/>
<tag k="service" v="driveway"/><tag k="bridge" v="yes"/><tag k="layer" v="1"/>
<tag k="source" v="fixture"/><tag k="note" v="snow雪"/></way>
<way id="102"><nd ref="3"/><nd ref="2"/></way>
<way id="103"><nd ref="1"/><nd ref="3"/><tag k="highway" v="footway"/></way>
</osm>''', encoding="utf-8")


def test_real_installed_gdal_streams_tiny_local_xml_whole_way(tmp_path):
    pytest.importorskip("pyogrio")
    source, config, temporary = tmp_path/"tiny.osm", tmp_path/"osmconf.ini", tmp_path/"temp"
    temporary.mkdir(); synthetic_osm(source); config.write_text(road.OSM_CONFIG, encoding="utf-8", newline="\r\n")
    reader = road.GdalWays()
    result = list(reader.rows(source, config, temporary, Guard()))
    assert len(result) == 1 and result[0][0] == 101 and result[0][4] == 3
    tags = json.loads(result[0][2])
    assert tags["source"] == "fixture" and tags["note"] == "snow雪" and tags["access"] == "private"
    assert reader.stats["layers_seen"]["lines"] == 3


def test_iterator_closes_dataset_on_early_consumer_close(tmp_path):
    pytest.importorskip("pyogrio")
    source, config, temporary = tmp_path/"tiny.osm", tmp_path/"osmconf.ini", tmp_path/"temp"
    temporary.mkdir(); synthetic_osm(source); config.write_text(road.OSM_CONFIG)
    rows = road.GdalWays().rows(source, config, temporary, Guard())
    next(rows); rows.close()
    # Windows refuses this replacement while the original GDAL source handle is open.
    source.replace(tmp_path/"closed.osm")


def test_gdal_index_and_temp_configuration_restored_after_close(tmp_path):
    pyogrio = pytest.importorskip("pyogrio")
    source, config, temporary = tmp_path/"tiny.osm", tmp_path/"osmconf.ini", tmp_path/"temp"
    temporary.mkdir(); synthetic_osm(source); config.write_text(road.OSM_CONFIG)
    keys = ("OSM_INDEX_POINTS", "OSM_INDEX_WAYS", "CPL_TMPDIR", "GDAL_NUM_THREADS")
    previous = {k: pyogrio.get_gdal_config_option(k) for k in keys}
    try:
        pyogrio.set_gdal_config_options({"OSM_INDEX_POINTS": "NO", "OSM_INDEX_WAYS": "NO"})
        before = {k: pyogrio.get_gdal_config_option(k) for k in keys}
        rows = road.GdalWays().rows(source, config, temporary, Guard())
        next(rows); rows.close()
        assert {k: pyogrio.get_gdal_config_option(k) for k in keys} == before
    finally:
        pyogrio.set_gdal_config_options(previous)


def test_silent_gdal_missing_node_truncation_refused_by_source_ledger(tmp_path):
    pytest.importorskip("pyogrio")
    source, config = tmp_path/"broken.osm", tmp_path/"osmconf.ini"
    synthetic_osm(source)
    body = source.read_text(encoding="utf-8").replace('<way id="101"><nd ref="1"/><nd ref="2"/><nd ref="3"/>',
        '<way id="101"><nd ref="1"/><nd ref="999"/><nd ref="3"/>')
    source.write_text(body, encoding="utf-8"); config.write_text(road.OSM_CONFIG)
    temporary = tmp_path/"temp"; temporary.mkdir()
    ledger = tmp_path/"expected.sqlite"
    assert road.source_way_ledger(source, ledger, Guard())["selected_vertices"] == 3
    rows = road.GdalWays().rows(source, config, temporary, Guard())
    with pytest.raises(ValueError, match="source-way reference ledger"):
        road.build_index(rows, tmp_path/"roads.sqlite", Guard(), source_ledger=ledger)


def vi(value):
    out = bytearray()
    while value >= 128: out.append((value & 127) | 128); value >>= 7
    out.append(value); return bytes(out)


def pb(number, value, wire=2):
    return vi((number << 3) | wire)+(vi(len(value))+value if wire == 2 else vi(value))


def zz(value): return value*2 if value >= 0 else -value*2-1


def pbf_file(path, *, compressed=True, history=False):
    def block(kind, raw):
        blob = pb(2, len(raw), 0)+pb(3, zlib.compress(raw)) if compressed else pb(1, raw)
        header = pb(1, kind)+pb(3, len(blob), 0)
        return struct.pack(">I", len(header))+header+blob
    table = b"".join(pb(1, v.encode()) for v in ("", "highway", "residential", "access", "private"))
    nodes = b""
    for i, (lat, lon) in enumerate(((42., -83.), (42.01, -82.99), (42.02, -82.98)), 1):
        node = pb(1, zz(i), 0)+pb(8, zz(round(lat*1e7)), 0)+pb(9, zz(round(lon*1e7)), 0)
        nodes += pb(1, node)
    way = pb(1, 101, 0)+pb(2, vi(1)+vi(3))+pb(3, vi(2)+vi(4))+pb(8, vi(2)*3)
    primitive = pb(1, table)+pb(2, nodes)+pb(2, pb(3, way))
    header = pb(4, b"OsmSchema-V0.6")+(pb(4, b"HistoricalInformation") if history else b"")
    path.write_bytes(block(b"OSMHeader", header)+block(b"OSMData", primitive))


@pytest.mark.parametrize("compressed", [True, False])
def test_independent_pbf_way_metadata_and_gdal_geometry_agree(tmp_path, compressed):
    pytest.importorskip("pyogrio")
    source, config = tmp_path/"tiny.pbf", tmp_path/"osmconf.ini"
    pbf_file(source, compressed=compressed); config.write_text(road.OSM_CONFIG)
    temporary = tmp_path/"temp"; temporary.mkdir(); ledger = tmp_path/"expected.sqlite"
    counts = road.source_way_ledger(source, ledger, Guard())
    assert counts == {"source_ways_seen": 1, "selected_ways": 1, "selected_vertices": 3}
    rows = road.GdalWays().rows(source, config, temporary, Guard())
    report = road.build_index(rows, tmp_path/"roads.sqlite", Guard(), source_ledger=ledger)
    assert report["source_way_counts_and_tags_verified"]
    assert report["counts"]["ways"] == 1


def test_history_pbf_refused_not_silently_reduced(tmp_path):
    source = tmp_path/"history.pbf"; pbf_file(source, history=True)
    with pytest.raises(ValueError, match="history"):
        road.source_way_ledger(source, tmp_path/"expected.sqlite", Guard())


def test_source_ledger_final_partial_batch_checks_disk_bound(tmp_path, monkeypatch):
    source = tmp_path/"tiny.osm"; synthetic_osm(source)
    monkeypatch.setitem(road.PROFILE, "max_output_bytes", 1)
    with pytest.raises(ValueError, match="source ledger size bound"):
        road.source_way_ledger(source, tmp_path/"expected.sqlite", Guard())


def test_source_ledger_bytes_flush_before_row_limit(tmp_path, monkeypatch):
    tags = {"highway": "residential", "note": "x"*100}
    monkeypatch.setattr(road, "_source_ways_xml", lambda *args: (v for v in [
        (41, tags, 2, "a"*64), (42, tags, 2, "b"*64)]))
    monkeypatch.setitem(road.PROFILE, "batch_bytes", 256)
    class BatchGuard(Guard):
        def __init__(self): super().__init__(); self.forced = 0
        def check(self, **kwargs):
            self.forced += bool(kwargs.get("force")); super().check(**kwargs)
    guard = BatchGuard()
    report = road.source_way_ledger(tmp_path/"fixture.osm", tmp_path/"expected.sqlite", guard)
    assert report["selected_ways"] == 2 and guard.forced >= 3


@pytest.mark.parametrize("body", [b"\x00", b"\x00\x00\x00\x02x", b"\xff\xff\xff\xff"])
def test_malformed_or_unbounded_pbf_refused(tmp_path, body):
    source = tmp_path/"bad.pbf"; source.write_bytes(body)
    with pytest.raises(ValueError): road.source_way_ledger(source, tmp_path/"expected.sqlite", Guard())
