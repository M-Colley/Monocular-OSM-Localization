"""Single-city topology and correlation fixtures; no HTTP/video/models/GT."""
from copy import deepcopy
import json

import networkx as nx
import numpy as np
import osmnx as ox
import pytest
from pyproj import Transformer

from monocular_osm import junction_inputs as inputs
from monocular_osm import junction_sequence as sequence
from monocular_osm.graph_filter import Odometry
from monocular_osm.road_sign_clues import (
    build_motorway_junction_inventory, extract_road_sign_clues, match_motorway_junctions,
)


def fixture():
    plan = inputs.plan_inputs("Synthetic City", [8., 49., 9., 50.], bbox_source={
        "city": "Synthetic City", "scope": "named_city_bbox", "source_kind": "public_geocode",
        "source_sha256": "a" * 64})
    graph = nx.MultiDiGraph(crs="EPSG:32632")
    for node in (1, 2, 3):
        graph.add_node(node, x=455000. + 100. * node, y=5428000.)
    graph.add_edge(1, 2, osmid=10, length=100.)
    graph.add_edge(2, 3, osmid=10, length=100.)
    tf = Transformer.from_crs(graph.graph["crs"], "EPSG:4326", always_xy=True)
    elements = []
    for node, ref in ((2, "12A"), (3, "99"), (999, "77")):
        lon, lat = tf.transform(455000. + 100. * node, 5428000.)
        elements.append({"type": "node", "id": node, "lat": lat, "lon": lon,
                         "tags": {"highway": "motorway_junction", "ref": ref}})
    inventory = build_motorway_junction_inventory(json.dumps({"elements": elements}).encode(), plan["inventory_plan"])
    matched = timed_matches(inventory, [1.])
    return graph, inventory, matched, plan


def timed_matches(inventory, times, refs=None):
    refs = refs or ["12A"] * len(times)
    rows = [{"text": f"EXIT {ref}", "confidence": .99, "t_sec": t,
             "bbox": [100., 100., 180., 130.]} for t, ref in zip(times, refs)]
    clues = extract_road_sign_clues(rows, [480, 640])
    return match_motorway_junctions(clues, inventory)


def test_plan_uses_same_unchanged_single_city_bbox_without_ref_filter():
    _, _, _, plan = fixture()
    request = plan["graph_request"]
    assert request == {"bbox": [8., 49., 9., 50.], "network_type": "drive",
                       "simplify": False, "retain_all": True, "truncate_by_edge": True}
    assert plan["inventory_plan"]["scope"] == "named_city_bbox"
    assert 'motorway_junction' in plan["inventory_plan"]["query"]
    assert plan["observed_ref_filter"] is False and plan["around"] is None


def test_real_simplifier_preserves_all_present_inventory_nodes_and_one_way_graph():
    graph, inv, matched, plan = fixture()
    assert 2 not in ox.simplification.simplify_graph(graph).nodes  # actual previous hazard
    original = deepcopy(graph)
    net, proof = inputs.prepare_graph(graph, inv, matched, plan,
        simplify_fn=ox.simplification.simplify_graph, project_fn=lambda g: g)
    assert proof["required_hypotheses"] == ["node/2"]
    assert proof["preserved_inventory_ids"] == ["node/2", "node/3"]
    assert proof["missing_unmatched_inventory_ids"] == ["node/999"]
    assert proof["nearest_edge_projection_used"] is False
    assert {node for e in net.edge_keys for node in e[:2]} == {1, 2, 3}
    assert (2, 1, 0) not in net.edge_keys  # no fabricated reverse road
    assert all(inputs.KEEP_NODE not in graph.nodes[n] for n in graph)
    assert nx.utils.graphs_equal(graph, original)


def test_missing_required_node_refuses_before_simplification_no_subset():
    graph, inv, matched, plan = fixture()
    graph.remove_node(2)
    with pytest.raises(ValueError, match="hypotheses unavailable"):
        inputs.prepare_graph(graph, inv, matched, plan,
            simplify_fn=lambda *a, **k: pytest.fail("partial topology accepted"), project_fn=lambda g: g)


@pytest.mark.parametrize("fault", ["subset", "coordinate", "inventory", "scope", "simplified"])
def test_original_plan_all_candidate_and_exact_node_guards(fault):
    graph, inv, matched, plan = fixture()
    if fault == "subset": matched["matches"][0]["candidates"] = []
    if fault == "coordinate": graph.nodes[2]["x"] += 10.
    if fault == "inventory": inv["junctions"][0]["ref"] = "99"
    if fault == "scope": plan["graph_request"]["retain_all"] = False
    if fault == "simplified": graph.graph["simplified"] = True
    with pytest.raises(ValueError):
        inputs.prepare_graph(graph, inv, matched, plan,
            simplify_fn=ox.simplification.simplify_graph, project_fn=lambda g: g)


def test_bad_simplifier_cannot_remove_even_unmatched_present_inventory_node():
    graph, inv, matched, plan = fixture()
    def drop(g, **kwargs):
        g.remove_node(3)
        return g
    with pytest.raises(ValueError, match="removed retained"):
        inputs.prepare_graph(graph, inv, matched, plan, simplify_fn=drop, project_fn=lambda g: g)


def test_explicit_disposable_graph_mode_avoids_helper_copy(monkeypatch):
    graph, inv, matched, plan = fixture()
    original_copy = nx.MultiDiGraph.copy
    copies = []
    def copy(self, *args, **kwargs):
        copies.append(self)
        return original_copy(self, *args, **kwargs)
    monkeypatch.setattr(nx.MultiDiGraph, "copy", copy)
    _, receipt = inputs.prepare_graph(graph, inv, matched, plan, copy_graph=False,
        simplify_fn=ox.simplification.simplify_graph, project_fn=lambda g: g)
    # Genuine simplifier still owns its one copy; no preceding helper copy.
    assert len(copies) == 1 and copies[0] is graph
    assert graph.nodes[2][inputs.KEEP_NODE] is True
    assert receipt["private_raw_ownership_consumed"] is True


@pytest.mark.parametrize("cap", ["segments", "degree", "entries"])
def test_allocation_upper_bounds_refuse_before_roadnetwork_dense_arrays(monkeypatch, cap):
    graph, inv, matched, plan = fixture()
    if cap == "segments": monkeypatch.setattr(inputs, "MAX_SEGMENTS", 1)
    if cap == "degree": monkeypatch.setattr(inputs, "MAX_DEGREE", 0)
    if cap == "entries": monkeypatch.setattr(inputs, "MAX_SUCCESSOR_ENTRIES", 0)
    monkeypatch.setattr(sequence.RoadNetwork, "from_graph", lambda *a, **k: pytest.fail("dense allocation before bound"))
    with pytest.raises(ValueError):
        inputs.prepare_graph(graph, inv, matched, plan,
            simplify_fn=ox.simplification.simplify_graph, project_fn=lambda g: g)


def test_same_ref_groups_use_first_time_maxspan_not_transitive_chain():
    _, inv, _, _ = fixture()
    original = timed_matches(inv, [14., 0., 7., 8., 23.], ["12A", "12A", "12A", "12A", "99"])
    before = deepcopy(original)
    annotated = inputs.annotate_correlations(original)
    expected = inputs.validate_correlations(annotated)
    assert original == before
    assert [a["count"] for a in expected] == [1, 3, 3, 3, 1]
    assert expected[0]["group_id"] != expected[1]["group_id"]
    assert expected[1]["group_id"] == expected[2]["group_id"] == expected[3]["group_id"]
    assert [r["observation"]["t_sec"] for r in annotated["matches"]] == [14., 0., 7., 8., 23.]
    assert [r["observation"] for r in annotated["matches"]] == [r["observation"] for r in original["matches"]]
    assert [r["candidates"] for r in annotated["matches"]] == [r["candidates"] for r in original["matches"]]


def test_empty_candidate_observation_is_not_dropped_before_group_count():
    _, inv, _, _ = fixture()
    matched = timed_matches(inv, [0., 1.], ["88", "88"])
    assert all(r["candidates"] == [] for r in matched["matches"])
    annotated = inputs.annotate_correlations(matched)
    assert [a["power"] for a in inputs.validate_correlations(annotated)] == [.5, .5]


@pytest.mark.parametrize("fault", ["power", "bool", "count", "partial", "group", "profile"])
def test_core_recomputes_full_group_annotations_before_graph_callbacks(monkeypatch, fault):
    graph, inv, _, _ = fixture()
    matched = inputs.annotate_correlations(timed_matches(inv, [1., 2.]))
    row = matched["matches"][0]
    if fault == "power": row["correlation"]["power"] = 1.
    if fault == "bool": row["correlation"]["count"] = True
    if fault == "count": row["correlation"]["count"] = 1
    if fault == "partial": del row["correlation"]
    if fault == "group": row["correlation"]["group_id"] = "other"
    if fault == "profile": matched["correlation_profile"] = "transitive"
    monkeypatch.setattr(sequence, "_target", lambda *a: pytest.fail("map before full annotations"))
    with pytest.raises(ValueError, match="correlat"):
        sequence.prepare_cues(None, matched, np.array([0., 3.]))


def test_power_is_applied_to_whole_mixture_logfactor_and_legacy_remains_one(monkeypatch):
    graph, inv, _, plan = fixture()
    legacy = timed_matches(inv, [.5, .5])
    net, _ = inputs.prepare_graph(graph, inv, legacy, plan,
        simplify_fn=ox.simplification.simplify_graph, project_fn=lambda g: g)
    odom = Odometry([0.], [0.], [.5], unit_seconds=.5)
    monkeypatch.setattr(sequence._OfflineFilter, "guide", lambda self, e, s, k: np.ones(np.shape(e)))
    monkeypatch.setattr(sequence, "sign_factor", lambda *a, **k: np.array([.25, .75]))
    seen = []
    original_summary = sequence._OfflineFilter.summary
    def capture(self, weights):
        seen.append(weights.copy())
        return original_summary(self, weights)
    monkeypatch.setattr(sequence._OfflineFilter, "summary", capture)
    out = sequence.align(net, odom, ["stationary"], np.array([0., .5]),
                         inputs.annotate_correlations(legacy), config=sequence.Config(particles=2))
    assert out["status"] == "complete_diagnostic_only"
    np.testing.assert_allclose(seen[-1], [.25, .75])
    assert [r["factor_power"] for r in out["cues"]] == [.5, .5]
    assert [r["group_count"] for r in out["cues"]] == [2, 2]
    sequence.align(net, odom, ["stationary"], np.array([0., .5]), legacy,
                   config=sequence.Config(particles=2))
    np.testing.assert_allclose(seen[-1], [.1, .9])
    assert all(a["power"] == 1. for a in inputs.validate_correlations(legacy))
