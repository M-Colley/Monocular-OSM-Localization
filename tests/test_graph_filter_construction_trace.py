"""Exact construction witnesses on invented graphs, never map assets."""
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from monocular_osm import graph_filter as g


ARRAYS = ("origin", "edge_len", "edge_base", "seg_lo", "seg_hi", "seg_gstart",
          "seg_xy0", "seg_dir", "seg_heading", "edge_head_start", "edge_head_end",
          "edge_mid", "reverse_of", "succ_table", "succ_logprior", "succ_head",
          "xy_max", "edge_keys")


def graph(coords=None):
    net = nx.MultiDiGraph(crs="EPSG:32632")
    coords = coords or [(455000., 5428000.), (455100., 5428000.)]
    net.add_node(1, x=coords[0][0], y=coords[0][1])
    net.add_node(2, x=coords[-1][0], y=coords[-1][1])
    net.add_edge(1, 2, key=7, geometry=LineString(coords))
    return net


def exact_network(a, b):
    assert a.crs == b.crs
    assert a.stats() == b.stats()
    for name in ARRAYS:
        np.testing.assert_array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)))


@pytest.mark.parametrize("case", ["straight", "curve", "reverse", "repeat", "tiny", "synthetic", "loop", "parallel", "no_geometry"])
def test_trace_records_exact_builder_geometry_and_preserves_all18(case):
    a, b = (455000., 5428000.), (455100., 5428000.)
    c = (455040., 5428010.)
    coords = {"curve": [a, c, b], "repeat": [a, a, c, b],
              "tiny": [a, (a[0] + 1e-7, a[1]), b], "synthetic": [a, a],
              "loop": [a, c, a]}.get(case, [a, b])
    raw = graph(coords)
    if case == "reverse": raw.edges[1, 2, 7]["geometry"] = LineString([b, a])
    if case == "loop":
        raw.remove_edge(1, 2, 7)
        raw.add_edge(1, 1, key=7, geometry=LineString(coords))
    if case == "parallel": raw.add_edge(1, 2, key=9, geometry=LineString([a, c, b]))
    if case == "no_geometry": del raw.edges[1, 2, 7]["geometry"]
    original = deepcopy(raw)
    ordinary = g.RoadNetwork.from_graph(raw)
    trace = []
    traced = g.RoadNetwork.from_graph(raw, construction_trace=trace)
    exact_network(ordinary, traced)
    assert nx.utils.graphs_equal(original, raw)
    assert len(trace) == ordinary.n_edges
    for index, row in enumerate(trace):
        assert row["edge_index"] == index
        assert row["edge_key"] == ordinary.edge_keys[index]
        assert row["schema"] == "road-network-construction-trace-v1"
        assert row["keep_indices"].dtype == np.dtype("int64")
        before = row["oriented_vertices"]
        keep = np.r_[True, np.hypot(*np.diff(before, axis=0).T) > 1e-6]
        np.testing.assert_array_equal(row["keep_indices"], np.flatnonzero(keep))
        expected = before[keep]
        if row["synthetic"]: expected = np.vstack([expected[:1], expected[:1] + [.05, 0.]])
        np.testing.assert_array_equal(row["compact_vertices"], expected)
        lo, hi = traced.seg_lo[index], traced.seg_hi[index]
        np.testing.assert_array_equal(traced.seg_xy0[lo:hi] + traced.origin, expected[:-1])
        assert len(expected) - 1 == hi - lo
    assert trace[0]["geometry_reversed"] is (case == "reverse")
    assert trace[0]["synthetic"] is (case == "synthetic")


@pytest.mark.parametrize("collector", [False, {}, (), [1], iter(())])
def test_trace_requires_exact_empty_list_before_geometry(collector, monkeypatch):
    monkeypatch.setattr(g, "_edge_coords", lambda *a: pytest.fail("invalid collector reached geometry"))
    with pytest.raises(ValueError, match="empty built-in list"):
        g.RoadNetwork.from_graph(graph(), construction_trace=collector)


def test_list_subclass_cannot_inject_publication_callback():
    class Sink(list):
        def extend(self, rows): pytest.fail("arbitrary callback")
    with pytest.raises(ValueError, match="empty built-in list"):
        g.RoadNetwork.from_graph(graph(), construction_trace=Sink())


def test_default_performs_no_trace_work(monkeypatch):
    monkeypatch.setattr(g, "_construction_trace_row", lambda *a: pytest.fail("default trace work"))
    assert g.RoadNetwork.from_graph(graph()).n_edges == 1


def test_trace_is_independent_and_committed_only_after_success():
    raw = graph()
    trace = []
    net = g.RoadNetwork.from_graph(raw, construction_trace=trace)
    vertices = trace[0]["compact_vertices"].copy()
    net.seg_xy0[:] = 0.
    raw.nodes[1]["x"] = -100.
    np.testing.assert_array_equal(trace[0]["compact_vertices"], vertices)
    bad = nx.MultiDiGraph(crs="EPSG:32632")
    empty = []
    with pytest.raises(ValueError, match="no edges"):
        g.RoadNetwork.from_graph(bad, construction_trace=empty)
    assert empty == []


def test_projected_trace_and_genuine_filter_state_rng_exact():
    raw = nx.MultiDiGraph(crs="EPSG:4326")
    raw.add_node(1, x=8., y=49.)
    raw.add_node(2, x=8.001, y=49.001)
    raw.add_edge(1, 2, key=4)
    raw.add_edge(2, 1, key=5)
    trace = []
    plain, traced = g.RoadNetwork.from_graph(raw), g.RoadNetwork.from_graph(raw, construction_trace=trace)
    exact_network(plain, traced)
    assert max(abs(trace[0]["oriented_vertices"].ravel())) > 1000.
    filters = [g.GraphParticleFilter(n, g.FilterConfig(n_particles=128, seed=9, path_storage=False))
               for n in (plain, traced)]
    for pf in filters:
        pf.initialise()
        for _ in range(3):
            pf.predict(.1, 0., .5)
            pf.update_heading(0., .5)
            weights, _ = pf.normalise()
            pf.resample(weights)
    for name, value in vars(filters[0]).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(filters[1], name))
    assert filters[0].rng.bit_generator.state == filters[1].rng.bit_generator.state


def test_trace_capacity_checks_precede_copies_and_partial_publication(monkeypatch):
    raw = graph()
    monkeypatch.setattr(raw, "number_of_edges", lambda: 100_001)
    monkeypatch.setattr(g, "_edge_coords", lambda *a: pytest.fail("edge cap before geometry"))
    with pytest.raises(ValueError, match="edge cap"):
        g.RoadNetwork.from_graph(raw, construction_trace=[])


def test_trace_vertex_cap_precedes_geometry_body(monkeypatch):
    class HugeGeometry:
        coords = range(1_100_001)
    raw = graph()
    raw.edges[1, 2, 7]["geometry"] = HugeGeometry()
    monkeypatch.setattr(g, "_edge_coords", lambda *a: pytest.fail("vertex cap before geometry"))
    trace = []
    with pytest.raises(ValueError, match="vertex/member cap"):
        g.RoadNetwork.from_graph(raw, construction_trace=trace)
    assert trace == []


def test_trace_segment_cap_precedes_witness_copies(monkeypatch):
    coords = np.column_stack((np.arange(500_002, dtype=np.float64), np.zeros(500_002)))
    monkeypatch.setattr(g, "_edge_coords", lambda *a: coords)
    monkeypatch.setattr(g, "_construction_trace_row", lambda *a: pytest.fail("segment cap before trace copy"))
    trace = []
    with pytest.raises(ValueError, match="segment cap"):
        g.RoadNetwork.from_graph(graph(), construction_trace=trace)
    assert trace == []


def test_empty_stored_geometry_fallback_is_distinct_from_synthetic():
    raw = graph()
    raw.edges[1, 2, 7]["geometry"] = LineString()
    trace = []
    net = g.RoadNetwork.from_graph(raw, construction_trace=trace)
    row = trace[0]
    assert row["geometry_present"] is True and row["geometry_fallback"] is True
    assert row["synthetic"] is False and row["raw_vertices"].shape == (0, 2)
    assert row["oriented_vertices"].shape == (2, 2) and net.n_edges == 1
