"""Tests for the road-graph particle filter (monocular_osm.graph_filter).

Everything runs on small synthetic street graphs (no network): a drive is
simulated along a known node path, turned into noisy *relative* odometry
(unknown scale, arbitrary yaw origin, slow yaw drift, per-step heading noise)
and fed to the filter.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from monocular_osm.graph_filter import (
    FilterConfig,
    FilterResult,
    GraphParticleFilter,
    HeadingObservation,
    Odometry,
    PositionObservation,
    RoadNetwork,
    _systematic_resample,
    aggregate_odometry,
    compass_deg_to_heading_rad,
    evaluate_against_gt,
    odometry_from_arrays,
    odometry_from_trajectory,
    run_filter,
    wrap_angle,
)

X0, Y0 = 455_000.0, 5_428_000.0   # somewhere in UTM 32N (Karlsruhe-ish)
CRS = "EPSG:32632"


# --------------------------------------------------------------------------
# synthetic maps and drives
# --------------------------------------------------------------------------

def grid_graph(xs, ys, *, oneway_rows: dict | None = None) -> nx.MultiDiGraph:
    """Street grid with vertical streets at ``xs`` and horizontal at ``ys``.

    ``oneway_rows`` maps a row index j to +1 (eastbound only) or -1
    (westbound only). Node ids are i * 1000 + j.
    """
    oneway_rows = oneway_rows or {}
    G = nx.MultiDiGraph(crs=CRS)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            G.add_node(i * 1000 + j, x=X0 + x, y=Y0 + y)

    def add(a, b):
        ax, ay = G.nodes[a]["x"], G.nodes[a]["y"]
        bx, by = G.nodes[b]["x"], G.nodes[b]["y"]
        G.add_edge(a, b, length=math.hypot(bx - ax, by - ay))

    for j in range(len(ys)):
        for i in range(len(xs) - 1):
            a, b = i * 1000 + j, (i + 1) * 1000 + j
            ow = oneway_rows.get(j, 0)
            if ow >= 0:
                add(a, b)
            if ow <= 0:
                add(b, a)
    for i in range(len(xs)):
        for j in range(len(ys) - 1):
            a, b = i * 1000 + j, i * 1000 + j + 1
            add(a, b)
            add(b, a)
    return G


def simulate_drive(G, node_path, *, dt=0.1, speed=lambda t: 8.0 + 2.0 * np.sin(t / 6.0),
                   corner_m=6.0):
    """Ground-truth positions/headings every ``dt`` s along ``node_path``.

    Corners are rounded by a moving average (a car does not turn on a point).
    Returns (positions (T+1, 2) in CRS metres, headings (T,), dt).
    """
    pts = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_path])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s_dense = np.arange(0.0, cum[-1], 0.5)
    dense = np.column_stack([np.interp(s_dense, cum, pts[:, 0]),
                             np.interp(s_dense, cum, pts[:, 1])])
    w = max(1, int(corner_m / 0.5))
    ker = np.ones(2 * w + 1) / (2 * w + 1)
    pad = np.vstack([np.repeat(dense[:1], w, 0), dense, np.repeat(dense[-1:], w, 0)])
    smooth = np.column_stack([np.convolve(pad[:, 0], ker, "valid"),
                              np.convolve(pad[:, 1], ker, "valid")])
    sc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(smooth, axis=0).T))])
    t, s, out = 0.0, 0.0, [0.0]
    while True:
        s += speed(t) * dt
        t += dt
        if s >= sc[-1]:
            break
        out.append(s)
    s_t = np.array(out)
    pos = np.column_stack([np.interp(s_t, sc, smooth[:, 0]),
                           np.interp(s_t, sc, smooth[:, 1])])
    d = np.diff(pos, axis=0)
    heading = np.arctan2(d[:, 1], d[:, 0])
    return pos, heading, dt


def vo_like_odometry(heading, dt, rng, *, yaw0=1.234, drift_deg_s=0.3,
                     noise_deg=3.0, metric_steps=None) -> Odometry:
    """Unit-norm steps (monocular VO: scale unknown), heading in an arbitrary
    frame with a linear drift and white noise."""
    T = len(heading)
    t = np.arange(1, T + 1) * dt
    h = (heading + yaw0 + np.deg2rad(drift_deg_s) * t
         + rng.normal(0.0, np.deg2rad(noise_deg), T))
    if metric_steps is None:
        return Odometry(step=np.ones(T), heading=wrap_angle(h), dt=np.full(T, dt),
                        metric=False, unit_seconds=dt)
    return odometry_from_arrays(metric_steps, wrap_angle(h), np.full(T, dt), metric=True)


def gt_at(pos, odom: Odometry):
    """Ground truth aligned with the filter's T+1 estimates."""
    idx = np.concatenate([[0], odom.frame_index])
    return pos[idx]


def to_latlon(net: RoadNetwork, xy_crs):
    return net.local_to_latlon(net.crs_to_local(xy_crs))


IRREG_X = np.cumsum([0, 95, 140, 70, 160, 85, 120, 150, 75])
IRREG_Y = np.cumsum([0, 120, 80, 145, 95, 170, 70, 110])
# a route with five turns (L, R, L, L, R), ~1.35 km
ROUTE = [1, 1001, 2001, 2002, 2003, 3003, 4003, 5003, 5004, 5005, 4005, 3005, 3006]


def jittered_grid(seed=42, spacing=110.0, jitter=30.0):
    """A 9 x 8 street network with every junction moved by up to +-jitter m:
    streets meet at varied angles, like a real town, but every junction
    still looks alike (degree 4, similar spacing) — ambiguous locally."""
    G = grid_graph(np.arange(9) * spacing, np.arange(8) * spacing)
    rng = np.random.default_rng(seed)
    for n in G.nodes:
        G.nodes[n]["x"] += rng.uniform(-jitter, jitter)
        G.nodes[n]["y"] += rng.uniform(-jitter, jitter)
    for u, v, k, d in G.edges(keys=True, data=True):
        d["length"] = float(np.hypot(G.nodes[u]["x"] - G.nodes[v]["x"],
                                     G.nodes[u]["y"] - G.nodes[v]["y"]))
    return G


# --------------------------------------------------------------------------
# map representation
# --------------------------------------------------------------------------

def test_locate_follows_edge_geometry_and_heading():
    G = nx.MultiDiGraph(crs=CRS)
    G.add_node(1, x=X0, y=Y0)
    G.add_node(2, x=X0 + 100, y=Y0 + 100)
    # an L-shaped edge: east 100 m, then north 100 m
    geom = LineString([(X0, Y0), (X0 + 100, Y0), (X0 + 100, Y0 + 100)])
    G.add_edge(1, 2, length=200.0, geometry=geom)
    # the reverse direction with its geometry stored *end-first* (defensive)
    G.add_edge(2, 1, length=200.0, geometry=geom)
    net = RoadNetwork.from_graph(G)
    e12 = net.edge_keys.index((1, 2, 0))
    e21 = net.edge_keys.index((2, 1, 0))
    assert net.edge_len[e12] == pytest.approx(200.0)
    x, y, h, _ = net.locate(np.array([e12, e12, e12]), np.array([50.0, 100.0 + 1e-9, 150.0]))
    xy = net.local_to_crs(np.column_stack([x, y]))
    assert np.allclose(xy[0], [X0 + 50, Y0])
    assert np.allclose(xy[2], [X0 + 100, Y0 + 50])
    assert h[0] == pytest.approx(0.0)
    assert h[2] == pytest.approx(np.pi / 2)
    # reverse edge was re-oriented: starts at node 2 heading south
    x, y, h, _ = net.locate(np.array([e21]), np.array([10.0]))
    xy = net.local_to_crs(np.column_stack([x, y]))
    assert np.allclose(xy[0], [X0 + 100, Y0 + 90])
    assert h[0] == pytest.approx(-np.pi / 2)
    assert net.reverse_of[e12] == e21 and net.reverse_of[e21] == e12


def test_successors_exclude_uturn_except_at_dead_ends():
    # a T: 0 -- 1 -- 2 with a spur 1 -- 3 (3 is a dead end)
    G = grid_graph([0, 100, 200], [0])
    G.add_node(3, x=X0 + 100, y=Y0 + 100)
    G.add_edge(1000, 3, length=100.0)
    G.add_edge(3, 1000, length=100.0)
    net = RoadNetwork.from_graph(G)
    idx = {k[:2]: i for i, k in enumerate(net.edge_keys)}
    succ = {net.edge_keys[j][:2] for j in net.successors(idx[(0, 1000)])}
    assert succ == {(1000, 2000), (1000, 3)}          # no U-turn back to 0
    # dead end at 3: the only way on is to turn round
    assert {net.edge_keys[j][:2] for j in net.successors(idx[(1000, 3)])} == {(3, 1000)}
    net_u = RoadNetwork.from_graph(G, allow_uturn=True)
    idx_u = {k[:2]: i for i, k in enumerate(net_u.edge_keys)}
    succ_u = {net_u.edge_keys[j][:2] for j in net_u.successors(idx_u[(0, 1000)])}
    assert (1000, 0) in succ_u
    # an (optional) normalised turn prior: straight on is likelier than turning
    net_p = net.with_turn_prior(2.0, normalise=True)
    row = net_p.succ_table[idx[(0, 1000)]]
    lp = dict(zip([net_p.edge_keys[j][:2] for j in row if j >= 0],
                  net_p.succ_logprior[idx[(0, 1000)]][row >= 0]))
    assert lp[(1000, 2000)] > lp[(1000, 3)]
    assert np.exp(list(lp.values())).sum() == pytest.approx(1.0)
    assert net.succ_logprior is not net_p.succ_logprior      # copy, not mutation


def test_sample_uniform_respects_disc_prior():
    net = RoadNetwork.from_graph(grid_graph(IRREG_X, IRREG_Y))
    rng = np.random.default_rng(0)
    cx, cy, r = X0 + 300, Y0 + 300, 120.0
    e, s = net.sample_uniform(2000, rng, disc=(cx, cy, r))
    x, y, _, _ = net.locate(e, s)
    xy = net.local_to_crs(np.column_stack([x, y]))
    assert np.all(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) <= r + 1e-6)
    # uniform over road length: both directions of the streets are present
    assert len(np.unique(e)) > 10


def test_unprojected_graph_is_projected():
    G = grid_graph([0, 100], [0, 100])
    Gll = nx.MultiDiGraph(crs="EPSG:4326")
    from pyproj import Transformer
    tf = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    for n, d in G.nodes(data=True):
        lon, lat = tf.transform(d["x"], d["y"])
        Gll.add_node(n, x=lon, y=lat)
    for u, v, d in G.edges(data=True):
        Gll.add_edge(u, v, **d)
    net = RoadNetwork.from_graph(Gll)
    assert "4326" not in net.crs
    assert net.edge_len.max() == pytest.approx(100.0, rel=0.01)


# --------------------------------------------------------------------------
# odometry adapters
# --------------------------------------------------------------------------

def test_odometry_from_vo_trajectory_rotation_matches_translation():
    from monocular_osm.visual_odometry import Trajectory, _fit_plane_projection

    # camera driving a left-curving arc on the ground plane (OpenCV camera
    # convention: x right, y down, z forward); a stop in the middle
    n = 60
    yaw = np.linspace(0.0, np.deg2rad(60.0), n)            # turning left
    centers = np.zeros((n, 3))
    rot = np.zeros((n, 3, 3))
    for i in range(n):
        fwd = np.array([np.sin(-yaw[i]), 0.0, np.cos(-yaw[i])])
        right = np.array([np.cos(-yaw[i]), 0.0, -np.sin(-yaw[i])])
        down = np.array([0.0, 1.0, 0.0])
        rot[i] = np.vstack([right, down, fwd])              # world-to-camera rows
        if i:
            step = 0.0 if 30 <= i < 35 else 1.0
            centers[i] = centers[i - 1] + step * fwd
    centers[:, 1] += 0.01 * np.random.default_rng(0).normal(size=n)  # pitch jitter
    valid = np.ones(n, bool)
    valid[30:35] = False
    xz = _fit_plane_projection(centers, up=np.array([0.0, -1.0, 0.0]))
    traj = Trajectory(centers=centers, xz=xz, valid=valid, n_inliers=[0] * n,
                      rotations=rot, translations=np.zeros((n, 3)))
    o_rot = odometry_from_trajectory(traj, 0.1, heading_source="rotation")
    o_tr = odometry_from_trajectory(traj, 0.1, heading_source="translation")
    assert o_rot.step[30:34].sum() == 0.0                      # stop = no motion
    assert np.isnan(o_rot.heading[30:34]).all()
    assert not o_rot.metric and o_rot.unit_seconds == pytest.approx(0.1)
    ok = np.isfinite(o_rot.heading) & np.isfinite(o_tr.heading)
    diff = wrap_angle(o_rot.heading[ok] - o_tr.heading[ok])
    # same frame (no mirror): the two agree up to the half-step lag of a chord
    assert np.max(np.abs(diff)) < np.deg2rad(1.0)
    # turning left = heading increases (CCW, the map's convention)
    total = wrap_angle(o_rot.heading[ok][-1] - o_rot.heading[ok][0])
    assert total == pytest.approx(np.deg2rad(60.0) * (n - 2) / (n - 1), abs=np.deg2rad(2))
    metric = odometry_from_trajectory(traj, 0.1, step_lengths_m=np.full(n - 1, 2.0))
    assert metric.metric and metric.step[0] == 2.0 and metric.step[31] == 0.0


def test_aggregate_odometry_sums_and_averages():
    o = Odometry(step=[1, 1, 0, 1, 1, 1], heading=[0.1, 0.3, np.nan, -3.1, 3.1, 0.0],
                 dt=0.1, metric=False, unit_seconds=0.1)
    a = aggregate_odometry(o, 2)
    assert len(a) == 3
    assert np.allclose(a.step, [2, 1, 2])
    assert np.allclose(a.dt, [0.2, 0.2, 0.2])
    assert a.heading[0] == pytest.approx(0.2)
    assert abs(wrap_angle(a.heading[1] - (-3.1))) < 1e-9
    assert abs(wrap_angle(a.heading[2] - 3.1 / 2)) < 1e-6  # 3.1 and 0.0 average
    assert list(a.frame_index) == [2, 4, 6]


def test_compass_conversion():
    assert compass_deg_to_heading_rad(0.0) == pytest.approx(np.pi / 2)      # north
    assert compass_deg_to_heading_rad(90.0) == pytest.approx(0.0)           # east
    assert abs(compass_deg_to_heading_rad(270.0)) == pytest.approx(np.pi)   # west


def test_resampler_is_filterpy_systematic_and_side_effect_free():
    from filterpy.monte_carlo import systematic_resample

    rng = np.random.default_rng(3)
    w = rng.random(1000)
    w /= w.sum()
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    idx = _systematic_resample(w, np.random.default_rng(7))
    assert np.array_equal(np.random.get_state()[1], before)   # global RNG untouched
    # identical to filterpy when filterpy sees the same global seed
    seed = int(np.random.default_rng(7).integers(0, 2**32 - 1))
    np.random.seed(seed)
    assert np.array_equal(idx, systematic_resample(w))


# --------------------------------------------------------------------------
# filter behaviour
# --------------------------------------------------------------------------

def _drive(G, *, speed=lambda t: 8.0, seed=0, metric=False, **noise):
    pos, heading, dt = simulate_drive(G, ROUTE, speed=speed)
    steps = np.hypot(*np.diff(pos, axis=0).T) if metric else None
    odom = vo_like_odometry(heading, dt, np.random.default_rng(seed),
                            metric_steps=steps, **noise)
    return pos, aggregate_odometry(odom, 5)                   # 10 Hz -> 2 Hz


def _drive_irregular(seed=0):
    G = grid_graph(IRREG_X, IRREG_Y)
    pos, odom = _drive(G, seed=seed)
    return G, pos, odom


def test_unit_step_vo_localizes_on_ambiguous_street_network():
    """Monocular-VO-like input: unit steps (scale unknown), arbitrary yaw
    origin, 0.3 deg/s drift, 3 deg noise — on a town-like network where every
    junction looks alike. The heading sequence + map geometry pin it down."""
    G = jittered_grid()
    net = RoadNetwork.from_graph(G)
    pos, odom = _drive(G)
    gt = gt_at(pos, odom)
    res = run_filter(net, odom, FilterConfig(n_particles=20_000, seed=1), diagnostic_xy=gt)
    ev = evaluate_against_gt(res, to_latlon(net, gt))
    # ambiguous at the start (every straight street looks alike)...
    assert res.confidence[0] < 0.05
    assert res.gt_mass[0] < 0.05
    # ...then localized, and it stays localized
    assert ev["time_to_localize_s"] is not None and ev["time_to_localize_s"] < 120.0, ev
    assert ev["final_err_m"] < 20.0, ev
    assert res.confidence[-1] > 0.5
    assert res.n_modes[-1] == 1
    assert ev["final_mass_within_25m_of_gt"] > 0.5
    # the smoothed genealogy reproduces the whole route, start included
    assert ev["smoothed_mean_err_m"] < 15.0
    # the scale state learned metres per unit step (8 m/s at 10 Hz = 0.8)
    assert 0.65 < res.k_mode[-1] < 0.95


def test_unit_step_vo_with_variable_speed_finds_the_street():
    """Speed 6-10 m/s while the odometry steps stay unit length: the route is
    found, but along-track position lags — monocular steps carry no speed."""
    G = jittered_grid()
    net = RoadNetwork.from_graph(G)
    pos, odom = _drive(G, speed=lambda t: 8.0 + 2.0 * np.sin(t / 6.0))
    gt = gt_at(pos, odom)
    res = run_filter(net, odom, FilterConfig(n_particles=20_000, seed=2), diagnostic_xy=gt)
    ev = evaluate_against_gt(res, to_latlon(net, gt))
    assert ev["final_err_m"] < 40.0, ev
    assert np.nanmax(res.gt_mass[-20:]) > 0.3
    assert ev["smoothed_mean_err_m"] < 25.0


def test_metric_oracle_localizes_in_right_angle_grid():
    """Right-angle grid with irregular spacing: turn angles carry nothing
    (all 90 deg), so only the metric distances between turns identify the
    route. With metric odometry the filter must get there."""
    G = grid_graph(IRREG_X, IRREG_Y)
    net = RoadNetwork.from_graph(G)
    pos, odom = _drive(G, speed=lambda t: 8.0 + 2.0 * np.sin(t / 6.0), metric=True)
    gt = gt_at(pos, odom)
    res = run_filter(net, odom, FilterConfig(n_particles=60_000, seed=3))
    ev = evaluate_against_gt(res, to_latlon(net, gt))
    assert ev["final_err_m"] < 10.0, ev
    assert ev["time_to_localize_s"] is not None
    assert 0.95 < res.k_mode[-1] < 1.05                 # metres per metre


def test_straight_drive_in_symmetric_grid_stays_ambiguous():
    xs = np.arange(0, 1100, 100.0)
    G = grid_graph(xs, xs)
    path = [j * 1000 + 5 for j in range(2, 8)]       # straight east, 500 m
    pos, heading, dt = simulate_drive(G, path)
    odom = aggregate_odometry(vo_like_odometry(heading, dt, np.random.default_rng(0)), 5)
    net = RoadNetwork.from_graph(G)
    res = run_filter(net, odom, FilterConfig(n_particles=20_000, seed=2))
    assert res.confidence[-1] < 0.2
    # the posterior is spread over many separated places, not a false lock
    xy, w = res.final_xy, res.final_w
    cells = np.floor(xy / 100.0).astype(int)
    _, inv = np.unique(cells, axis=0, return_inverse=True)
    mass = np.bincount(inv.ravel(), w)
    assert (mass > 0.01).sum() >= 5
    assert res.n_modes[-1] != 1


def test_oneway_streets_are_respected():
    # two identical parallel streets 300 m apart; row 0 is westbound-only
    xs = np.cumsum([0, 130, 90, 160, 120, 100])
    ys = [0.0, 300.0]
    G_ow = grid_graph(xs, ys, oneway_rows={0: -1})
    G_tw = grid_graph(xs, ys)
    net_ow = RoadNetwork.from_graph(G_ow)
    eastbound_row0 = [k for k in net_ow.edge_keys if k[1] == k[0] + 1000 and k[0] % 1000 == 0]
    assert eastbound_row0 == []                       # no edge against the one-way
    assert (net_ow.reverse_of < 0).sum() == len(xs) - 1
    # drive east along the two-way street (row 1); an absolute heading fix
    # (e.g. the sun) says "east" — only one-way information can then decide
    path = [i * 1000 + 1 for i in range(len(xs))]
    pos, heading, dt = simulate_drive(G_tw, path)
    odom = aggregate_odometry(vo_like_odometry(heading, dt, np.random.default_rng(0)), 5)
    obs = [HeadingObservation(step=t, heading_rad=0.0, kappa=4.0) for t in range(0, len(odom), 4)]

    def mass_on_row1(G):
        net = RoadNetwork.from_graph(G)
        res = run_filter(net, odom, FilterConfig(n_particles=10_000, seed=4), heading_obs=obs)
        return float(res.final_w[res.final_xy[:, 1] > Y0 + 150].sum())

    assert mass_on_row1(G_ow) > 0.95                  # one-way rules out row 0
    assert 0.25 < mass_on_row1(G_tw) < 0.75           # control: symmetric


def test_position_observation_resolves_ambiguity():
    xs = np.arange(0, 1100, 100.0)
    G = grid_graph(xs, xs)
    path = [j * 1000 + 5 for j in range(2, 8)]
    pos, heading, dt = simulate_drive(G, path)
    odom = aggregate_odometry(vo_like_odometry(heading, dt, np.random.default_rng(0)), 5)
    net = RoadNetwork.from_graph(G)
    gt = gt_at(pos, odom)
    T = len(odom)
    # two noisy fixes (VPR-like, 20 m sigma) half-way and near the end, plus
    # one gross outlier in between that the uniform mixture must absorb
    a, b = T // 2, T - 4
    obs = [PositionObservation(a, *(gt[a] + [12.0, -8.0]), sigma_m=20.0),
           PositionObservation(b, *(gt[b] + [-10.0, 6.0]), sigma_m=20.0),
           PositionObservation(3 * T // 4, X0 + 50.0, Y0 + 950.0, sigma_m=20.0, outlier=0.3)]
    res = run_filter(net, odom, FilterConfig(n_particles=20_000, seed=5), position_obs=obs)
    ev = evaluate_against_gt(res, to_latlon(net, gt))
    assert res.confidence[a - 1] < 0.2                  # ambiguous before the first fix
    assert ev["errors_m"][a] < 30.0                     # pulled to the fix ...
    assert ev["final_err_m"] < 25.0, ev                 # ... and not to the outlier
    assert res.n_modes[-1] == 1


def test_normalised_turn_prior_is_available_and_proper():
    net = RoadNetwork.from_graph(grid_graph(IRREG_X, IRREG_Y), turn_prior_kappa=2.0,
                                 normalise_turn_prior=True)
    row_ok = net.succ_table >= 0
    tot = np.where(row_ok, np.exp(net.succ_logprior), 0.0).sum(axis=1)
    assert np.allclose(tot[row_ok.any(axis=1)], 1.0)
    flat = RoadNetwork.from_graph(grid_graph(IRREG_X, IRREG_Y))
    assert np.all(flat.succ_logprior[flat.succ_table >= 0] == 0.0)   # default: no tax


def test_determinism_with_fixed_seed():
    G, pos, odom = _drive_irregular()
    net = RoadNetwork.from_graph(G)
    odom = Odometry(step=odom.step[:40], heading=odom.heading[:40], dt=odom.dt[:40],
                    metric=False, unit_seconds=0.1, frame_index=odom.frame_index[:40])
    cfg = FilterConfig(n_particles=5_000, seed=11)
    np.random.seed(0)
    g0 = np.random.random()
    a = run_filter(net, odom, cfg)
    b = run_filter(net, odom, cfg)
    np.random.seed(0)
    assert np.random.random() == g0                     # no global-RNG coupling
    assert np.array_equal(a.mode_xy, b.mode_xy)
    assert np.array_equal(a.confidence, b.confidence)
    assert np.array_equal(a.smoothed_xy, b.smoothed_xy, equal_nan=True)
    c = run_filter(net, odom, FilterConfig(n_particles=5_000, seed=12))
    assert not np.array_equal(a.final_w, c.final_w)


def test_all_particles_dead_triggers_reinitialisation():
    # a single one-way street: every particle reaches its end and dies
    G = grid_graph([0, 50], [0], oneway_rows={0: +1})
    net = RoadNetwork.from_graph(G)
    odom = Odometry(step=np.ones(20) * 10, heading=np.zeros(20), dt=0.5,
                    metric=True)
    pf = GraphParticleFilter(net, FilterConfig(n_particles=500, seed=0,
                                               metric_logk_sigma=0.01), metric=True)
    res = pf.run(odom)
    assert res.n_reinit >= 1
    assert np.all(np.isfinite(res.confidence))


def test_evaluate_against_gt_time_to_localize():
    T = 6
    gt = np.tile([[49.0, 8.4]], (T, 1))
    # errors (deg lat * 111 km): 500, 50, 5, 30, 5, 5 m -> localized only at t=4
    off = np.array([500, 50, 5, 30, 5, 5]) / 111_195.0
    mode = gt + np.column_stack([off, np.zeros(T)])
    res = FilterResult(
        times=np.arange(T, dtype=float), frame_index=np.arange(T),
        mode_xy=np.zeros((T, 2)), mode_latlon=mode, confidence=np.linspace(0, 1, T),
        n_modes=np.ones(T, int), ess=np.ones(T), resampled=np.zeros(T, bool),
        k_mode=np.ones(T), smoothed_xy=None, smoothed_latlon=None,
        final_xy=None, final_w=None, gt_mass=None, runtime_s=0.0, n_reinit=0,
        config={}, map_stats={})
    ev = evaluate_against_gt(res, gt)
    assert ev["time_to_localize_s"] == pytest.approx(4.0)
    assert ev["final_err_m"] == pytest.approx(5.0, abs=0.1)
    assert ev["frac_steps_lt10m"] == pytest.approx(3 / 6)
