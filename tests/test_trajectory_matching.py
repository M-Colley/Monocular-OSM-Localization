"""Tests for trajectory matching: Procrustes math and end-to-end on
synthetic graphs with known ground truth."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from src.osm_data import RoadGraph, _build_polyline_view
from src.trajectory_matching import (
    MatchCandidate,
    _candidates_overlap,
    match_trajectory,
    procrustes_similarity,
    score_candidates_with_sliding_windows,
)


def test_procrustes_recovers_known_similarity() -> None:
    rng = np.random.default_rng(0)
    src = rng.uniform(-1, 1, size=(20, 2))
    th = np.deg2rad(37.0)
    R_true = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    s_true = 4.5
    t_true = np.array([10.0, -3.0])
    dst = s_true * src @ R_true.T + t_true

    R, scale, residual, src_aligned = procrustes_similarity(src, dst)
    assert scale == pytest.approx(s_true, rel=1e-6)
    assert residual < 1e-6
    # R should equal R_true (Procrustes also handles reflections; we asked
    # for orientation-preserving and the test data is orientation-preserving).
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(src_aligned, dst, atol=1e-6)


def test_procrustes_residual_grows_with_noise() -> None:
    rng = np.random.default_rng(1)
    src = rng.uniform(-1, 1, size=(30, 2))
    dst_clean = 2.0 * src
    dst_noisy = dst_clean + rng.normal(0, 0.05, size=src.shape)

    _, _, r_clean, _ = procrustes_similarity(src, dst_clean)
    _, _, r_noisy, _ = procrustes_similarity(src, dst_noisy)
    assert r_clean < 1e-9
    assert r_noisy > 0.01


def _l_shape_graph() -> RoadGraph:
    """Two streets that share a start node:

        A -- B -- C  (the 'L1' street)
        |
        D
        |
        E             (the 'L2' street, perpendicular)

    plus a third decoy street far away with the same shape as L1 but
    different heading; the matcher should still find the right spot when
    given an L1-shaped trajectory.
    """
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"

    nodes = {
        "A": (0, 0), "B": (100, 0), "C": (200, 0),
        "D": (0, -100), "E": (0, -200),
        # Decoy street 1 km north, oriented the same as A-B-C so it has
        # the same straight bearing signature. It's a separate component.
        "F": (10000, 0), "G": (10100, 0), "H": (10200, 0),
    }
    for k, (x, y) in nodes.items():
        g.add_node(k, x=float(x), y=float(y))

    def add(a: str, b: str) -> None:
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        length = float(np.hypot(bx - ax, by - ay))
        g.add_edge(a, b, length=length, geometry=LineString([(ax, ay), (bx, by)]),
                   name=f"{a}{b}")
        g.add_edge(b, a, length=length, geometry=LineString([(bx, by), (ax, ay)]),
                   name=f"{a}{b}")

    add("A", "B")
    add("B", "C")
    add("A", "D")
    add("D", "E")
    add("F", "G")
    add("G", "H")

    return _build_polyline_view(g)


def test_match_finds_correct_walk_on_l_shape() -> None:
    """A trajectory that turns 90° left should match A→D→E (a left turn),
    not the straight A→B→C or the decoy F→G→H."""
    road = _l_shape_graph()

    # Build a synthetic trajectory shaped like A → D → E: go south 100 m,
    # then continue south another 100 m. (No turn, it's actually
    # A→D→E going straight south.)
    south_traj = np.array(
        [[0, -i] for i in range(0, 201, 5)]
    , dtype=float)
    # Apply an arbitrary similarity (rotate, scale, translate) — the
    # matcher must be invariant to this.
    th = np.deg2rad(42.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    south_traj = 0.013 * south_traj @ R.T + np.array([7.5, -2.1])

    cands = match_trajectory(
        south_traj,
        road,
        n_samples=48,
        walks_per_node=4,
        walk_depth=6,
        bearing_top_k=20,
        final_top_k=5,
        estimated_length_m=180.0,
        progress=False,
    )
    assert cands, "matcher returned no candidates"
    best = cands[0]

    # Trajectory is a perfectly straight south path. Many walks in this
    # graph are straight (A→B→C, A→D→E, B→C, D→E, ...) and they all map
    # onto the trajectory under a 2-D similarity with essentially zero
    # residual — they're tied valid matches. The matcher should rank
    # ANY of these straight walks above the L-walks (B→A→D, D→A→B). We
    # check that property directly: the best walk's geometry must be
    # straight.
    poly = best.walk_xy
    if len(poly) >= 3:
        d0 = poly[1] - poly[0]
        dN = poly[-1] - poly[-2]
        cos_turn = float(d0 @ dN / (np.linalg.norm(d0) * np.linalg.norm(dN) + 1e-9))
        assert cos_turn > 0.9, (
            f"best walk has a turn (cos={cos_turn:.2f}); a straight walk "
            "should have ranked higher for a straight trajectory"
        )
    # And: the residual must be effectively zero — these are exact shape
    # matches modulo floating-point noise.
    assert best.score < 1.0, f"unexpectedly large residual {best.score}"


def test_match_l_turn_picks_l_turn_over_straight() -> None:
    """Trajectory with a clear 90° left turn should rank A→...→E (straight
    south, no turn) BELOW a walk that actually contains a 90° turn."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    # An L-walk and a straight-walk sharing the same start node 0.
    coords = {
        0: (0, 0),
        1: (100, 0),    # straight east
        2: (200, 0),
        3: (0, -100),   # south
        4: (-100, -100),  # then west — 90° right turn
    }
    for n, (x, y) in coords.items():
        g.add_node(n, x=float(x), y=float(y))

    def add(a: int, b: int) -> None:
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        length = float(np.hypot(bx - ax, by - ay))
        g.add_edge(a, b, length=length, geometry=LineString([(ax, ay), (bx, by)]))
        g.add_edge(b, a, length=length, geometry=LineString([(bx, by), (ax, ay)]))

    add(0, 1)
    add(1, 2)
    add(0, 3)
    add(3, 4)

    road = _build_polyline_view(g)

    # Trajectory: south 100, then a sharp right turn, west 100.
    traj = np.array(
        [[0, -i] for i in range(0, 101, 5)]
        + [[-i, -100] for i in range(5, 101, 5)]
    , dtype=float)

    cands = match_trajectory(
        traj,
        road,
        n_samples=48,
        walks_per_node=6,
        walk_depth=6,
        bearing_top_k=30,
        final_top_k=4,
        estimated_length_m=190.0,
        progress=False,
    )
    assert cands, "no candidates"
    best = cands[0]

    # The best walk must contain a real ~90° turn — the trajectory's shape
    # is an L, so the matcher should prefer any L-walk over the straight
    # walk 0→1→2. Multiple L-walks exist in this graph (0→3→4, 1→0→3,
    # etc.) and they're all valid by symmetry; the test just checks the
    # matcher picks one of them and not the straight road.
    poly = best.walk_xy
    d0 = poly[1] - poly[0]
    d1 = poly[-1] - poly[-2]
    cos_turn = float(d0 @ d1 / (np.linalg.norm(d0) * np.linalg.norm(d1)))
    assert cos_turn < 0.5, (
        f"best walk is too straight (cos={cos_turn:.2f}); matcher failed to "
        "prefer the L-shaped walk over the straight one"
    )

    # And: a straight walk should not be the top result.
    nodes = [best.walk[0][0]] + [v for _, v, _ in best.walk]
    assert nodes != [0, 1, 2], "straight walk ranked above L-walk"


def test_sliding_window_support_scores_candidates() -> None:
    road = _l_shape_graph()
    near_candidate = match_trajectory(
        np.array([[0, -i] for i in range(0, 201, 5)], dtype=float),
        road,
        n_samples=48,
        walks_per_node=4,
        walk_depth=6,
        bearing_top_k=20,
        final_top_k=1,
        estimated_length_m=180.0,
        progress=False,
    )[0]
    far_walk = [("F", "G", 0), ("G", "H", 0)]
    far_poly = np.array([[10000.0, 0.0], [10100.0, 0.0], [10200.0, 0.0]])
    full_candidates = [
        near_candidate,
        type(near_candidate)(
            score=20.0,
            bearing_corr=0.1,
            start_node="F",
            walk=far_walk,
            walk_xy=far_poly,
            aligned_traj_xy=far_poly.copy(),
            walk_length_m=200.0,
        ),
    ]

    def fake_match(_window: np.ndarray, _road: RoadGraph, **_kwargs: object) -> list:
        return [full_candidates[0]]

    results = score_candidates_with_sliding_windows(
        np.array([[0, -i] for i in range(0, 201, 5)], dtype=float),
        road,
        full_candidates[:2],
        window_size=32,
        step=16,
        match_fn=fake_match,
    )
    assert len(results) == 2
    assert results[0].support_count == results[0].n_windows
    assert results[0].sliding_score > results[1].sliding_score


def _dense_grid_road(size: int, spacing: float) -> RoadGraph:
    """A size x size two-way grid in meters, as a RoadGraph."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    for i in range(size):
        for j in range(size):
            g.add_node(i * size + j, x=j * spacing, y=i * spacing)

    def add(a: int, b: int) -> None:
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        L = float(np.hypot(bx - ax, by - ay))
        g.add_edge(a, b, length=L, geometry=LineString([(ax, ay), (bx, by)]))
        g.add_edge(b, a, length=L, geometry=LineString([(bx, by), (ax, ay)]))

    for i in range(size):
        for j in range(size):
            n = i * size + j
            if j + 1 < size:
                add(n, n + 1)
            if i + 1 < size:
                add(n, n + size)
    return _build_polyline_view(g)


def test_long_length_prior_on_dense_grid_yields_candidates() -> None:
    """The walk-depth cap must scale with the length prior. With ~60 m
    edges, the old fixed depth of 40 edges tops out at 2 400 m — every
    walk fails the 0.5 * 6 000 m length filter and the pool is silently
    EMPTY for a long clip through a dense grid."""
    road = _dense_grid_road(size=22, spacing=60.0)  # 484 nodes
    # An L-shaped trajectory (shape doesn't matter much; the pool must
    # simply be non-empty).
    traj = np.array(
        [[0, i] for i in range(0, 3001, 50)]
        + [[i, 3000] for i in range(50, 3001, 50)]
    , dtype=float)

    cands = match_trajectory(
        traj,
        road,
        n_samples=32,
        walks_per_node=4,
        bearing_top_k=10,
        final_top_k=3,
        sample_every=50,
        estimated_length_m=6000.0,
        progress=False,
    )
    assert cands, (
        "no candidates for a 6 km prior on a dense grid — walk depth "
        "did not scale with the length prior"
    )


def _stub_candidate(graph: nx.MultiDiGraph, walk: list[tuple], walk_xy: np.ndarray) -> MatchCandidate:
    start = walk[0][0] if walk else next(iter(graph.nodes))
    return MatchCandidate(
        score=1.0, bearing_corr=0.5, start_node=start, walk=walk,
        walk_xy=np.asarray(walk_xy, dtype=float),
        aligned_traj_xy=np.asarray(walk_xy, dtype=float).copy(),
        walk_length_m=float(np.linalg.norm(np.diff(walk_xy, axis=0), axis=1).sum()),
    )


def _named_two_street_graph() -> nx.MultiDiGraph:
    """Two disjoint streets 4 km apart sharing the SAME name."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    coords = {"A": (0, 0), "B": (400, 0), "C": (4000, 3000), "D": (4400, 3000)}
    for k, (x, y) in coords.items():
        g.add_node(k, x=float(x), y=float(y))
    for a, b in (("A", "B"), ("C", "D")):
        ax, ay = coords[a]
        bx, by = coords[b]
        g.add_edge(a, b, length=400.0,
                   geometry=LineString([(ax, ay), (bx, by)]), name="Hauptstraße")
        g.add_edge(b, a, length=400.0,
                   geometry=LineString([(bx, by), (ax, ay)]), name="Hauptstraße")
    return g


def test_overlap_supports_window_on_long_route() -> None:
    """A window covering the first quarter of a long route lies ON the
    candidate polyline and must count as support. The old centroid-to-
    centroid test put the two centroids ~550 m apart and denied it."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    g.add_node(0, x=0.0, y=0.0)
    full_route = _stub_candidate(g, [], np.array([[0.0, 0.0], [1500.0, 0.0]]))
    window_walk = _stub_candidate(g, [], np.array([[0.0, 0.0], [400.0, 0.0]]))
    assert _candidates_overlap(
        full_route, window_walk, g, support_radius_m=250.0
    ), "geometric support denied to a window lying exactly on the route"


def test_overlap_rejects_same_name_across_town() -> None:
    """A street-name match 4 km away must NOT count as support: the name
    shortcut has to be spatially scoped."""
    g = _named_two_street_graph()
    near = _stub_candidate(g, [("A", "B", 0)], np.array([[0.0, 0.0], [400.0, 0.0]]))
    far = _stub_candidate(g, [("C", "D", 0)], np.array([[4000.0, 3000.0], [4400.0, 3000.0]]))
    assert not _candidates_overlap(near, far, g, support_radius_m=250.0), (
        "city-wide same-name street granted support"
    )


def test_overlap_accepts_nearby_same_name() -> None:
    """The name shortcut still works when the named window walk is close
    (within ~500 m) to the candidate polyline."""
    g = _named_two_street_graph()
    near = _stub_candidate(g, [("A", "B", 0)], np.array([[0.0, 0.0], [400.0, 0.0]]))
    # Same street name, parallel 400 m away: outside the geometric
    # radius but inside the name-scoped one.
    shifted = _stub_candidate(g, [("A", "B", 0)], np.array([[0.0, 400.0], [400.0, 400.0]]))
    assert _candidates_overlap(near, shifted, g, support_radius_m=250.0)


def test_match_trajectory_empty_input_returns_empty() -> None:
    road = _l_shape_graph()
    assert match_trajectory(np.zeros((0, 2)), road, progress=False) == []


def test_match_trajectory_stationary_input_returns_empty() -> None:
    """VO emitting 300 identical poses (a car parked for the whole
    segment) must yield 'no matches', not a resample crash."""
    road = _l_shape_graph()
    traj = np.tile(np.array([[5.0, 5.0]]), (300, 1))
    assert match_trajectory(traj, road, progress=False) == []


def test_match_trajectory_drops_non_finite_scores(monkeypatch) -> None:
    """Candidates whose alignment failed (inf residual, unaligned
    trajectory) must be dropped, not ranked."""
    import src.trajectory_matching as tm

    def broken_procrustes(src, dst):
        return np.eye(2), 1.0, float("inf"), src.copy()

    monkeypatch.setattr(tm, "procrustes_similarity", broken_procrustes)
    road = _l_shape_graph()
    traj = np.array([[0, -i] for i in range(0, 201, 5)], dtype=float)
    cands = tm.match_trajectory(
        traj, road, n_samples=48, walks_per_node=4, walk_depth=6,
        bearing_top_k=20, final_top_k=5, estimated_length_m=180.0,
        progress=False,
    )
    assert cands == [], "inf-score candidates leaked into the ranking"


def test_restrict_to_start_nodes_gates_enumeration() -> None:
    """With restrict_to_start_nodes, only walks rooted at the given nodes
    are considered — the anchor-gate behavior. A trajectory shaped like
    the far decoy F-G-H must still be matched there when F is the only
    seed, proving enumeration was confined to the seed vicinity."""
    road = _l_shape_graph()
    # Straight east trajectory matching the decoy F->G->H.
    traj = np.array([[i, 0.0] for i in range(0, 201, 5)], dtype=float)

    gated = match_trajectory(
        traj, road, n_samples=48, walks_per_node=4, walk_depth=6,
        bearing_top_k=20, final_top_k=5, estimated_length_m=180.0,
        progress=False, extra_start_nodes=["F"], restrict_to_start_nodes=True,
    )
    assert gated, "gated match returned nothing"
    # Every returned candidate must start within the seeded component
    # (F/G/H), never from the A/B/C/D/E component.
    allowed = {"F", "G", "H"}
    for c in gated:
        nodes = {c.walk[0][0]} | {v for _, v, _ in c.walk}
        assert nodes <= allowed, f"gate leaked to non-seed nodes: {nodes}"


def test_restrict_to_start_nodes_noop_without_seeds() -> None:
    """restrict_to_start_nodes with no seeds must fall back to a full scan,
    not return empty."""
    road = _l_shape_graph()
    traj = np.array([[0, -i] for i in range(0, 201, 5)], dtype=float)
    cands = match_trajectory(
        traj, road, n_samples=48, walks_per_node=4, walk_depth=6,
        bearing_top_k=20, final_top_k=5, estimated_length_m=180.0,
        progress=False, extra_start_nodes=None, restrict_to_start_nodes=True,
    )
    assert cands, "fallback to full scan should still produce candidates"


def test_procrustes_fixed_scale_preserves_extent() -> None:
    """Fixed-scale Procrustes must NOT shrink a path to fit a smaller dst;
    it keeps the prescribed metric extent (the anti-compression property)."""
    from src.trajectory_matching import procrustes_fixed_scale
    # src spans 100 units; dst is a compact 10 m blob.
    src = np.array([[0, 0], [50, 0], [100, 0]], dtype=float)
    dst = np.array([[0, 0], [5, 0], [10, 0]], dtype=float)
    resid, aligned, _R = procrustes_fixed_scale(src, dst, scale=1.0)
    span = np.linalg.norm(aligned[-1] - aligned[0])
    assert span == pytest.approx(100.0, rel=1e-6)   # extent preserved, not shrunk
    assert resid > 30.0                              # and it fits the blob poorly


def test_procrustes_fixed_scale_recovers_rotation_translation() -> None:
    from src.trajectory_matching import procrustes_fixed_scale
    src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    s = 3.0
    th = np.deg2rad(40.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = s * src @ R.T + np.array([100.0, -50.0])
    resid, aligned, _R = procrustes_fixed_scale(src, dst, scale=s)
    assert resid < 1e-6
    assert np.allclose(aligned, dst, atol=1e-6)


def test_procrustes_fixed_scale_no_reflection() -> None:
    from src.trajectory_matching import procrustes_fixed_scale
    # dst is a mirrored src; fixed-scale fit must not use a reflection,
    # so it cannot match perfectly.
    src = np.array([[0, 0], [10, 0], [10, 5]], dtype=float)
    dst = np.array([[0, 0], [10, 0], [10, -5]], dtype=float)
    resid, aligned, _R = procrustes_fixed_scale(src, dst, scale=1.0)
    assert resid > 1.0


def test_match_trajectory_locked_scale_runs() -> None:
    road = _l_shape_graph()
    traj = np.array([[0, -i] for i in range(0, 201, 5)], dtype=float)
    cands = match_trajectory(
        traj, road, n_samples=48, walks_per_node=4, walk_depth=6,
        bearing_top_k=20, final_top_k=5, estimated_length_m=180.0,
        progress=False, locked_scale=1.0,
    )
    assert cands  # produces candidates with the fixed-scale path


def test_anchor_pinned_route_places_anchor_exactly() -> None:
    """The VO point at the anchor time must map exactly onto the anchor's
    world location, and the route keeps the locked metric extent."""
    from src.trajectory_matching import anchor_pinned_route
    # Straight 100-unit VO path; matched walk is a straight 500 m segment.
    vo = np.array([[0, 0], [25, 0], [50, 0], [75, 0], [100, 0]], dtype=float)
    walk = np.array([[0, 0], [500, 0]], dtype=float)
    s = 5.0  # 500 m / 100 units
    anchor_vo = vo[2]                      # midpoint, "seen" here
    anchor_world = np.array([9000.0, 7000.0])  # geocoded location
    route = anchor_pinned_route(vo, walk, s, anchor_vo, anchor_world, n_samples=32)
    # midpoint of the route must sit exactly on the anchor
    assert np.allclose(route[2], anchor_world, atol=1e-6)
    # extent preserved (100 units * scale 5 = 500 m)
    assert np.linalg.norm(route[-1] - route[0]) == pytest.approx(500.0, rel=1e-6)
