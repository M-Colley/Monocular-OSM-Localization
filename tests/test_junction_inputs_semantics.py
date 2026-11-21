"""Opt-in preparation uses genuine simplification on tiny invented maps."""
from copy import deepcopy
import gc
import json
import weakref

import networkx as nx
import numpy as np
import osmnx as ox
import pytest
from pyproj import Transformer
from shapely.geometry import LineString

from monocular_osm import junction_inputs as inputs
from monocular_osm import road_segment_semantics as sem
from tests.test_graph_filter_construction_trace import exact_network
from tests.test_junction_inputs import fixture


def case():
    graph, inv, matched, plan = fixture()
    for u, v, key, row in graph.edges(keys=True, data=True):
        row.update(highway="primary", ref=f"R{u}", maxspeed="50 mph",
                   oneway=True, reversed=False)
    return graph, inv, matched, plan


def run(graph, inv, matched, plan, *, copy_graph=True, project_fn=lambda g: g, records=None):
    records = [] if records is None else records
    result = inputs.prepare_graph_with_semantics(
        graph, inv, matched, plan, emit_record=records.append, copy_graph=copy_graph,
        simplify_fn=ox.simplification.simplify_graph, project_fn=project_fn)
    return (*result, records)


@pytest.mark.parametrize("variant", ["basic", "reverse", "chain", "parallel", "zero", "tiny", "loop", "raw_geometry"])
def test_exact_old18_receipt_and_every_raw_record_survive_optin(variant):
    graph, inv, matched, plan = case()
    if variant == "reverse":
        for u, v, key, d in list(graph.edges(keys=True, data=True)):
            graph.add_edge(v, u, key=key + 9, **{**d, "reversed": True})
    if variant in {"chain", "parallel", "zero", "tiny"}:
        x, y = graph.nodes[3]["x"], graph.nodes[3]["y"]
        graph.add_node(4, x=x + (0. if variant == "zero" else 1e-7 if variant == "tiny" else 50.), y=y)
        graph.add_node(5, x=x + 100., y=y)
        for u, v in ((3, 4), (4, 5)):
            graph.add_edge(u, v, key=8, osmid=99, length=50., highway="residential",
                           maxspeed="30", ref="B", oneway=True, reversed=False)
            if variant == "parallel":
                graph.add_edge(u, v, key=11, osmid=100, length=50., highway="primary",
                               maxspeed="70 mph", oneway=True, reversed=False)
    if variant == "loop":
        graph.add_edge(3, 3, key=6, osmid=11, length=0., highway="service",
                       oneway=True, reversed=False)
    if variant == "raw_geometry":
        a, b = graph.nodes[1], graph.nodes[2]
        graph.edges[1, 2, 0]["geometry"] = LineString(
            [(a["x"], a["y"]), (a["x"] + 30., a["y"] + 10.), (b["x"], b["y"])])
    before = deepcopy(graph)
    ordinary, receipt = inputs.prepare_graph(
        graph, inv, matched, plan, simplify_fn=ox.simplification.simplify_graph, project_fn=lambda g: g)
    net, actual, packet, records = run(graph, inv, matched, plan)
    exact_network(ordinary, net)
    assert actual == receipt
    assert nx.utils.graphs_equal(before, graph)
    assert len(records) == graph.number_of_edges()
    arrays = packet["arrays"]
    np.testing.assert_array_equal(arrays["raw_edge_keys"], np.asarray(list(graph.edges(keys=True)), dtype="int64"))
    np.testing.assert_array_equal(arrays["compact_edge_keys"], np.asarray(net.edge_keys, dtype="int64"))
    assert arrays["segment_raw_edge_index"].shape == net.seg_gstart.shape
    assert arrays["compact_path_indptr"].shape == (net.n_edges + 1,)
    assert len(arrays) == 7
    assert not any(a.dtype.hasobject for a in arrays.values())
    states, flags = arrays["segment_semantic_status"], arrays["raw_edge_disposition"]
    reason = {"parallel": "ambiguous_parallel_neutral", "tiny": "tiny_replacement_neutral",
              "loop": "synthetic_neutral", "raw_geometry": "raw_geometry_neutral"}.get(variant)
    if reason is not None:
        selected = states == sem.SEG_STATUS[reason]
        assert np.any(selected)
        assert np.all(arrays["segment_raw_edge_index"][selected] == -1)
    if variant == "parallel":
        assert np.count_nonzero(flags & sem.RAW_STATUS["omitted_parallel"]) == 2
    if variant == "zero":
        assert np.count_nonzero(flags & sem.RAW_STATUS["zero_chord"]) == 1
    # Streaming serialization remains the producer's sink, not this wrapper.
    assert all(json.loads(sem.record_bytes(r)) for r in records)


def test_catalog_is_emitted_before_any_edge_token_mutation_and_original_is_private():
    graph, inv, matched, plan = case()
    records = []
    def emit(row):
        assert all(sem.TOKEN not in d for _, _, d in graph.edges(data=True))
        records.append(row)
    net, receipt, packet = inputs.prepare_graph_with_semantics(
        graph, inv, matched, plan, emit_record=emit, copy_graph=False,
        simplify_fn=ox.simplification.simplify_graph, project_fn=lambda g: g)
    assert len(records) == 2
    assert all(sem.TOKEN in d for _, _, d in graph.edges(data=True))
    assert receipt["private_raw_ownership_consumed"] is True
    assert receipt["raw_graph_caller_owned_and_not_modified"] is False


def test_real_projection_carries_exact_ordered_lineage_without_changing_network():
    graph, inv, matched, plan = case()
    x, y = graph.nodes[3]["x"], graph.nodes[3]["y"]
    graph.add_node(4, x=x + 50., y=y + 10.)
    graph.add_node(5, x=x + 100., y=y)
    graph.add_edge(3, 4, key=11, osmid=90, length=51., highway="primary", maxspeed="30")
    graph.add_edge(4, 5, key=12, osmid=91, length=51., highway="primary", maxspeed="50 mph")
    tf = Transformer.from_crs(graph.graph["crs"], "EPSG:4326", always_xy=True)
    for _, row in graph.nodes(data=True):
        row["x"], row["y"] = tf.transform(row["x"], row["y"])
    graph.graph["crs"] = "EPSG:4326"
    ordinary, receipt = inputs.prepare_graph(graph, inv, matched, plan)
    net, actual, packet, _ = run(graph, inv, matched, plan, project_fn=ox.project_graph)
    exact_network(ordinary, net)
    assert actual == receipt
    assert packet["arrays"]["segment_raw_edge_index"].shape == net.seg_gstart.shape
    edge = next(i for i, key in enumerate(net.edge_keys) if key[:2] == (3, 5))
    path = packet["arrays"]["compact_path_indptr"]
    assert path[edge + 1] - path[edge] == 2  # merged tuple survives genuine GeoDataFrame projection


@pytest.mark.parametrize("fault", ["plan", "subset", "simplified", "copy_type", "missing_node"])
def test_original_guards_refuse_before_first_catalog_record(fault):
    graph, inv, matched, plan = case()
    copy = True
    if fault == "plan": plan["graph_request"]["retain_all"] = False
    if fault == "subset": matched["matches"][0]["candidates"] = []
    if fault == "simplified": graph.graph["simplified"] = True
    if fault == "copy_type": copy = 1
    if fault == "missing_node": graph.remove_node(2)
    with pytest.raises(ValueError):
        inputs.prepare_graph_with_semantics(
            graph, inv, matched, plan, emit_record=lambda _: pytest.fail("catalog before original guards"),
            copy_graph=copy, simplify_fn=lambda *a, **k: pytest.fail("simplifier before guard"),
            project_fn=lambda g: g)


def test_explicit_sink_required_before_work():
    with pytest.raises(ValueError, match="record sink"):
        inputs.prepare_graph_with_semantics(*case(), emit_record=[])


def test_simplifier_receives_only_original_options_plus_exact_public_lineage_seam():
    graph, inv, matched, plan = case()
    seen = []
    def simplify(g, **kwargs):
        assert all(sem.TOKEN in d for _, _, d in g.edges(data=True))
        assert kwargs["node_attrs_include"] == [inputs.KEEP_NODE]
        assert kwargs["remove_rings"] is False and kwargs["track_merged"] is True
        assert kwargs["edge_attr_aggs"] == {"length": sum, "travel_time": sum, sem.TOKEN: sem.concat_paths}
        seen.append(True)
        return ox.simplification.simplify_graph(g, **kwargs)
    inputs.prepare_graph_with_semantics(graph, inv, matched, plan, emit_record=lambda _: None,
                                       simplify_fn=simplify, project_fn=lambda g: g)
    assert seen == [True]


@pytest.mark.parametrize("semantics", [False, True])
def test_disposable_original_graph_is_released_before_projection(semantics):
    graph, inv, matched, plan = case()
    ref = weakref.ref(graph)
    owned = {"raw": graph}
    del graph
    checked = []
    def project(prepared):
        gc.collect()
        assert ref() is None, "outer wrapper retained disposed raw graph"
        checked.append(True)
        return prepared
    kwargs = {"copy_graph": False, "simplify_fn": ox.simplification.simplify_graph,
              "project_fn": project}
    function = inputs.prepare_graph_with_semantics if semantics else inputs.prepare_graph
    if semantics: kwargs["emit_record"] = lambda _: None
    function(owned.pop("raw"), inv, matched, plan, **kwargs)
    assert checked == [True]
