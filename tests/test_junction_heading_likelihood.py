"""Private scalar/fork geometry and angular integration, no case inputs."""
from copy import deepcopy
import math
import networkx as nx
import numpy as np
import pytest
from scipy.integrate import quad
from shapely.geometry import LineString
from monocular_osm import junction_heading_likelihood as h
from monocular_osm.graph_filter import (FilterConfig, GraphParticleFilter,
    RoadNetwork, _interval_residual, wrap_angle)


def fork(current=0., successors=(), length=100.):
    graph = nx.MultiDiGraph(crs="EPSG:32632")
    graph.add_node(0, x=455000.-length*math.cos(current), y=5428000.-length*math.sin(current))
    graph.add_node(1, x=455000., y=5428000.)
    graph.add_edge(0, 1)
    for i, angle in enumerate(successors, 2):
        graph.add_node(i, x=455000.+100*math.cos(angle), y=5428000.+100*math.sin(angle))
        graph.add_edge(1, i)
    net = RoadNetwork.from_graph(graph)
    edge = next(i for i, key in enumerate(net.edge_keys) if key[:2] == (0, 1))
    return net, edge


def genuine(net, edges, offsets, previous, headings, zone):
    pf = GraphParticleFilter.__new__(GraphParticleFilter)
    pf.net, pf.cfg = net, FilterConfig(junction_zone_m=zone)
    pf.edge, pf.offset, pf.prev_head = np.broadcast_arrays(
        np.asarray(edges), np.asarray(offsets), np.asarray(previous))
    pf.edge, pf.offset, pf.prev_head = (v.reshape(-1) for v in (pf.edge, pf.offset, pf.prev_head))
    pf.road_heading = net.locate(pf.edge, pf.offset)[2]
    return pf._heading_innovation(np.asarray(headings).reshape(-1))


def union_reference_residual(headings, current, previous, successors, after, before):
    """Scalar shorter-arc minimum, preserving the actual signed tie order."""
    answer = wrap_angle(current - np.asarray(headings))
    if after and np.isfinite(previous):
        value = _interval_residual(headings, previous, current)
        answer = np.where(np.abs(value) < np.abs(answer), value, answer)
    if before:
        for successor in successors:
            value = _interval_residual(headings, current, successor)
            answer = np.where(np.abs(value) < np.abs(answer), value, answer)
    return answer


@pytest.mark.parametrize("current,successors,length,offset,previous,zone,expected", [
    (0., [1.], 100., 50., np.nan, 0., 0.),
    (0., [], 100., 5., math.pi/2, 12., math.pi/2),
    (0., [-math.pi/3, math.pi/3], 100., 95., np.nan, 12., 2*math.pi/3),
    (0., [-math.pi/2], 10., 5., 2*math.pi/3, 12., 7*math.pi/6),
    (math.radians(170), [math.radians(-170)], 100., 95., np.nan, 12., math.radians(20)),
])
def test_scalar_vector_union_matches_genuine_and_quadrature(
        current, successors, length, offset, previous, zone, expected):
    net, edge = fork(current, successors, length)
    width = h.arc_union_width(net, edge, offset, previous, zone)
    assert width.shape == () and width.dtype == np.float64
    assert float(width) == pytest.approx(expected, abs=2e-11)
    headings = np.linspace(-math.pi, math.pi, 20_001, endpoint=False)
    edges = np.full(len(headings), edge)
    widths = h.arc_union_width(net, edges, offset, previous, zone)
    np.testing.assert_array_equal(widths, np.full(len(headings), width))
    residual = genuine(net, edges, np.full(len(headings), offset),
                       np.full(len(headings), previous), headings, zone)
    tangent = float(net.locate(np.array([edge]), np.array([offset]))[2][0])
    succ = net.succ_head[edge][net.succ_table[edge] >= 0]
    independent = union_reference_residual(headings, tangent, previous, succ,
                                           offset < zone, net.edge_len[edge]-offset < zone)
    np.testing.assert_array_equal(residual, independent)
    extents = [0.]
    if offset < zone and np.isfinite(previous): extents.append(float(-wrap_angle(tangent-previous)))
    if net.edge_len[edge]-offset < zone: extents.extend(wrap_angle(succ-tangent))
    lower, upper = min(extents), max(extents)
    relative = wrap_angle(headings-tangent)
    inside = (relative >= lower) & (relative <= upper)
    distance = np.where(inside, 0., np.minimum(np.abs(wrap_angle(relative-lower)),
                                              np.abs(wrap_angle(relative-upper))))
    np.testing.assert_allclose(np.abs(residual), distance, atol=2e-15)
    for variance in (.01, .3, 3., 1e12):
        mass = np.exp(h.normalized_loglike(residual, variance, widths, .2)).mean()*2*math.pi
        assert mass == pytest.approx(1., abs=2e-8)


def test_oriented_antipodes_cover_full_circle():
    net, edge = fork(0., [math.pi], length=10.)
    assert h.arc_union_width(net, edge, 5., np.nan, 12.) == pytest.approx(math.pi)
    both = h.arc_union_width(net, edge, 5., math.pi, 12.)
    assert both == pytest.approx(2*math.pi)
    saved = net.succ_table[edge].copy(); net.succ_table[edge] = -1
    assert h.arc_union_width(net, edge, 5., math.pi, 12.) == math.pi
    net.succ_table[edge] = saved
    theta = np.linspace(-math.pi, math.pi, 10_001)
    residual = genuine(net, np.full(len(theta), edge), np.full(len(theta), 5.),
                       np.full(len(theta), math.pi), theta, 12.)
    np.testing.assert_array_equal(residual, np.zeros_like(theta))
    for variance in (np.nextafter(0., 1.), 1e-300, 1., 1e300, np.finfo(float).max):
        np.testing.assert_allclose(h.normalized_loglike(residual, variance, both, .2),
                                   -math.log(2*math.pi), rtol=0, atol=5e-16)


def test_width_does_not_claim_signed_successor_tie_order():
    net, edge = fork(0., [-math.pi/2, math.pi/2])
    width = h.arc_union_width(net, edge, 95., np.nan, 12.)
    first = genuine(net, [edge], [95.], [np.nan], [math.pi], 12.)[0]
    net.succ_table[edge] = net.succ_table[edge][::-1]
    net.succ_head[edge] = net.succ_head[edge][::-1]
    reverse = genuine(net, [edge], [95.], [np.nan], [math.pi], 12.)[0]
    assert first == -reverse and abs(first) == math.pi/2
    assert h.arc_union_width(net, edge, 95., np.nan, 12.) == width
    assert h.normalized_loglike(first, .3, width, .2) == h.normalized_loglike(reverse, .3, width, .2)
    assert not h.PROFILE["signed_innovation_ties_preserved_by_width"]


def test_strict_zone_boundaries_and_no_input_mutation():
    net, edge = fork(0., [math.pi/2])
    offsets = np.array([0., 12., 50., net.edge_len[edge]-12., net.edge_len[edge]])
    previous = np.full(len(offsets), -math.pi/3)
    before = deepcopy((net.__dict__, offsets, previous))
    widths = h.arc_union_width(net, edge, offsets, previous, 12.)
    np.testing.assert_allclose(widths, [math.pi/3, 0., 0., 0., math.pi/2])
    for key, value in before[0].items():
        if isinstance(value, np.ndarray): np.testing.assert_array_equal(net.__dict__[key], value)
    np.testing.assert_array_equal(offsets, before[1]); np.testing.assert_array_equal(previous, before[2])


def test_current_tangent_is_polyline_local_not_edge_start():
    graph = nx.MultiDiGraph(crs="EPSG:32632")
    for i, (x, y) in enumerate([(0., 0.), (10., 10.), (0., 10.)]):
        graph.add_node(i, x=455000.+x, y=5428000.+y)
    graph.add_edge(0, 1, geometry=LineString([(455000., 5428000.),
                    (455010., 5428000.), (455010., 5428010.)]))
    graph.add_edge(1, 2)
    net = RoadNetwork.from_graph(graph)
    edge = next(i for i, key in enumerate(net.edge_keys) if key[:2] == (0, 1))
    assert h.arc_union_width(net, edge, 19., np.nan, 2.) == math.pi/2


@pytest.mark.parametrize("width", [0., .3, math.pi, 2*math.pi])
@pytest.mark.parametrize("outlier", [0., .2, 1.])
def test_normalized_density_by_independent_tail_quadrature(width, outlier):
    # Integrate allowed interval and BOTH complement tails in distance space.
    # This avoids missing a narrow point peak on an arbitrary fixed angle grid.
    for variance in (1e-8, .03, 1., 1e6):
        peak = math.exp(float(h.normalized_loglike(0., variance, width, outlier)))
        half_gap = (2*math.pi-width)/2
        sigma = math.sqrt(variance)
        upper = min(half_gap/sigma, 12.)
        inlier_tail = quad(lambda u: math.exp(float(h.normalized_loglike(u*sigma,
                    variance, width, 0.)))*sigma, 0., upper, epsabs=1e-11)[0]
        # Beyond12 sigma is negligible; uniform part is integrated exactly.
        mass = width*peak + (1-outlier)*2*inlier_tail + outlier*2*half_gap/(2*math.pi)
        assert mass == pytest.approx(1., abs=2e-10)


def test_extreme_variance_limits_and_exact_uniform_outlier():
    variance = np.array([np.nextafter(0., 1.), 1e-300, 1e-8, 1., 1e300, np.finfo(float).max])
    at_peak = h.normalized_loglike(0., variance, 0., 0.)
    assert np.isfinite(at_peak).all()
    np.testing.assert_allclose(at_peak[-2:], -math.log(2*math.pi), rtol=0, atol=1e-15)
    tails = h.normalized_loglike(math.pi, variance, 0., .2)
    assert np.isfinite(tails).all()
    assert not np.isnan(h.normalized_loglike(math.pi, variance, 0., 0.)).any()
    np.testing.assert_array_equal(h.normalized_loglike(math.pi, variance, 0., 1.),
                                  np.full(len(variance), -math.log(2*math.pi)))


def test_heterogeneous_particle_vector_equals_scalar_calls():
    net, _ = fork(.3, [-2., 1.7])
    rng = np.random.default_rng(624)
    edges = rng.integers(0, net.n_edges, 50)
    offsets = rng.random(50)*net.edge_len[edges]
    previous = rng.uniform(-math.pi, math.pi, 50); previous[::4] = np.nan
    vector = h.arc_union_width(net, edges, offsets, previous, 12.)
    scalar = np.array([h.arc_union_width(net, int(e), float(o), float(p), 12.)
                       for e, o, p in zip(edges, offsets, previous)])
    np.testing.assert_array_equal(vector, scalar)
    residual = np.zeros(50); variance = np.linspace(.01, 20., 50)
    before = (residual.copy(), variance.copy(), vector.copy())
    logs = h.normalized_loglike(residual, variance, vector, .2)
    assert logs.dtype == np.float64 and logs.shape == (50,)
    for actual, expected in zip((residual, variance, vector), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("residual,variance,width,outlier", [
    (np.nan, 1., 0., .2), (4., 1., 0., .2), (0., 0., 0., .2),
    (0., -1., 0., .2), (0., np.inf, 0., .2), (0., 1., -.1, .2),
    (0., 1., 7., .2), (0., 1., 0., -1.), (0., 1., 0., np.nan),
    (True, 1., 0., .2), (0., True, 0., .2), (0., 1., 0., True),
    ([[0.]], 1., 0., .2), ([0., 0.], [1., 1., 1.], 0., .2),
])
def test_invalid_likelihood_inputs_refuse(residual, variance, width, outlier):
    with pytest.raises(ValueError): h.normalized_loglike(residual, variance, width, outlier)


@pytest.mark.parametrize("edge,offset,previous,zone", [
    (True, 1., np.nan, 12.), (0., 1., np.nan, 12.), (-1, 1., np.nan, 12.),
    (np.uint64(2**64-1), 1., np.nan, 12.), (0, -1., np.nan, 12.),
    (0, 1000., np.nan, 12.), (0, 1., np.inf, 12.), (0, 1., np.nan, -1.),
    (0, 1., np.nan, True), (0, 1., np.nan, np.inf),
])
def test_invalid_geometry_inputs_refuse(edge, offset, previous, zone):
    net, _ = fork(0., [1.])
    with pytest.raises(ValueError): h.arc_union_width(net, edge, offset, previous, zone)


def test_bounds_refuse_before_locate_and_allow_empty_arrays(monkeypatch):
    net, edge = fork(0., [1.])
    empty = h.arc_union_width(net, np.array([], dtype=np.int64), np.array([]), np.array([]), 12.)
    assert empty.shape == (0,)
    monkeypatch.setattr(net, "locate", lambda *args: pytest.fail("overcap geometry callback"))
    with pytest.raises(ValueError):
        h.arc_union_width(net, np.full(50_001, edge), 1., np.nan, 12.)
    with pytest.raises(ValueError): h.normalized_loglike(np.zeros(50_001), 1., 0., .2)
    net.succ_table = np.zeros((net.n_edges, 33), dtype=np.int64)
    net.succ_head = np.zeros((net.n_edges, 33))
    with pytest.raises(ValueError): h.arc_union_width(net, edge, 1., np.nan, 12.)
