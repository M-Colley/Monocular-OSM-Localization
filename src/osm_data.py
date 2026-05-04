"""Fetch and cache the OSM driving road graph for a city.

We project to a metric CRS (UTM zone for the city's centroid) so that
edge geometries are in meters and Euclidean comparison against the
visual-odometry trajectory is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import LineString


@dataclass
class RoadGraph:
    """Lightweight container holding the projected OSMnx graph and a flat
    list of polylines (one per edge) in metric coordinates.

    `polylines[i]` is an Nx2 numpy array of (x, y) in meters; `edge_keys[i]`
    is the (u, v, key) tuple of the corresponding edge in `graph`.
    """
    graph: nx.MultiDiGraph
    polylines: list[np.ndarray]
    edge_keys: list[tuple]
    crs: str  # the EPSG / wkt string of the projection


def fetch_city_graph(
    place: str,
    cache_path: Path | None = None,
    *,
    network_type: str = "drive",
) -> RoadGraph:
    """Get the driving road graph for `place`. Caches to GraphML if asked."""
    cache_path = Path(cache_path) if cache_path else None

    if cache_path and cache_path.exists():
        graph = ox.load_graphml(cache_path)
    else:
        graph = ox.graph_from_place(place, network_type=network_type)
        graph = ox.project_graph(graph)  # UTM in meters
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ox.save_graphml(graph, cache_path)

    return _build_polyline_view(graph)


def _build_polyline_view(graph: nx.MultiDiGraph) -> RoadGraph:
    polylines: list[np.ndarray] = []
    edge_keys: list[tuple] = []

    for u, v, k, data in graph.edges(keys=True, data=True):
        geom: LineString | None = data.get("geometry")
        if geom is None:
            ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
            vx, vy = graph.nodes[v]["x"], graph.nodes[v]["y"]
            coords = np.array([[ux, uy], [vx, vy]], dtype=np.float64)
        else:
            coords = np.asarray(geom.coords, dtype=np.float64)
        if len(coords) < 2:
            continue
        polylines.append(coords)
        edge_keys.append((u, v, k))

    crs = str(graph.graph.get("crs", ""))
    return RoadGraph(graph=graph, polylines=polylines, edge_keys=edge_keys, crs=crs)


def _edge_length(graph: nx.MultiDiGraph, u: int, v: int, k: int) -> float:
    return float(graph.edges[u, v, k].get("length", 0.0))


def _edge_end_heading(graph: nx.MultiDiGraph, u: int, v: int, k: int) -> float:
    d = graph.edges[u, v, k]
    geom: LineString | None = d.get("geometry")
    if geom is not None and len(geom.coords) >= 2:
        (x0, y0), (x1, y1) = geom.coords[-2], geom.coords[-1]
    else:
        x0, y0 = graph.nodes[u]["x"], graph.nodes[u]["y"]
        x1, y1 = graph.nodes[v]["x"], graph.nodes[v]["y"]
    return float(np.arctan2(y1 - y0, x1 - x0))


def _edge_start_heading(graph: nx.MultiDiGraph, u: int, v: int, k: int) -> float:
    d = graph.edges[u, v, k]
    geom: LineString | None = d.get("geometry")
    if geom is not None and len(geom.coords) >= 2:
        (x0, y0), (x1, y1) = geom.coords[0], geom.coords[1]
    else:
        x0, y0 = graph.nodes[u]["x"], graph.nodes[u]["y"]
        x1, y1 = graph.nodes[v]["x"], graph.nodes[v]["y"]
    return float(np.arctan2(y1 - y0, x1 - x0))


def _heading_dev(h: float, ref: float) -> float:
    return abs(((h - ref + np.pi) % (2 * np.pi)) - np.pi)


def _build_walk(
    graph: nx.MultiDiGraph,
    first_edge: tuple,
    target_length_m: float,
    max_depth: int,
    *,
    turn_at: int | None = None,
    turn_rank: int = 0,
) -> tuple[list[tuple], float]:
    """Build a walk starting from `first_edge`, extending by smallest
    heading-deviation choice at each intersection.

    If `turn_at` is set, at that walk index (1-based: 1 = right after
    the first edge, 2 = after two edges, ...) we pick the `turn_rank`-th
    sorted option instead of the greedy zero-th. This lets the caller
    deliberately introduce a single turn at a known walk depth — which
    is what gives the matcher diversity in turn positions.
    """
    walk: list[tuple] = [first_edge]
    length = _edge_length(graph, *first_edge)
    heading = _edge_end_heading(graph, *first_edge)
    # Track visited nodes so we don't drive in circles. Without this, the
    # greedy chooser will happily lap a roundabout or block forever.
    visited = {first_edge[0], first_edge[1]}

    while length < target_length_m and len(walk) < max_depth:
        last_u, last_v, _ = walk[-1]
        out = list(graph.out_edges(last_v, keys=True))
        # Drop immediate reversal AND any revisit of an earlier node.
        out = [
            e for e in out
            if not (e[0] == last_v and e[1] == last_u)
            and e[1] not in visited
        ]
        if not out:
            break
        out.sort(key=lambda e: _heading_dev(_edge_start_heading(graph, *e), heading))
        if turn_at is not None and len(walk) == turn_at and turn_rank < len(out):
            choice = out[turn_rank]
        else:
            choice = out[0]
        walk.append(choice)
        length += _edge_length(graph, *choice)
        heading = _edge_end_heading(graph, *choice)
        visited.add(choice[1])

    return walk, length


def walks_from_node(
    graph: nx.MultiDiGraph,
    start: int,
    target_length_m: float,
    *,
    max_walks: int = 6,
    max_depth: int = 80,
) -> list[list[tuple]]:
    """Enumerate plausible driven walks rooted at `start`.

    Search strategy is **bounded** so it stays linear in `max_depth` per
    walk (no exponential DFS). At the start node we branch into up to
    `max_walks` first edges; each branch then **greedily** continues by
    smallest heading deviation at every subsequent intersection until
    cumulative length ≥ `target_length_m`.

    A walk is kept if its length is at least 50% of the target. We don't
    require an exact match because Procrustes downstream re-scales — the
    target is mainly a way to bound how far we walk before scoring.
    """
    walks: list[list[tuple]] = []

    out0 = list(graph.out_edges(start, keys=True))
    if not out0:
        return walks
    out0_sorted = sorted(out0, key=lambda e: (-_edge_length(graph, *e), e))

    seen: set[tuple] = set()

    def try_walk(first: tuple, *, turn_at: int | None, turn_rank: int) -> bool:
        """Build a walk and append it if it's new and long enough."""
        walk, length = _build_walk(
            graph, first, target_length_m, max_depth,
            turn_at=turn_at, turn_rank=turn_rank,
        )
        if length < 0.5 * target_length_m:
            return False
        key = tuple(walk)
        if key in seen:
            return False
        seen.add(key)
        walks.append(walk)
        return True

    # Per first edge, generate one greedy walk plus a few "branching"
    # walks that turn off the greedy continuation at successively deeper
    # intersections. The branching walks are what let us match
    # trajectories that contain real turns.
    branch_indices = (1, 2, 3, 4, 6, 8)
    for first in out0_sorted[:3]:
        if len(walks) >= max_walks:
            break
        try_walk(first, turn_at=None, turn_rank=0)
        for bi in branch_indices:
            if len(walks) >= max_walks:
                break
            # Take the second-best option (rank=1) — i.e., a real turn
            # away from the greedy path at intersection `bi`.
            try_walk(first, turn_at=bi, turn_rank=1)

    return walks


def walk_to_polyline(graph: nx.MultiDiGraph, walk: list[tuple]) -> np.ndarray:
    """Concatenate the geometries of `walk`'s edges into a single polyline."""
    pts: list[np.ndarray] = []
    for (u, v, k) in walk:
        d = graph.edges[u, v, k]
        geom: LineString | None = d.get("geometry")
        if geom is None:
            ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
            vx, vy = graph.nodes[v]["x"], graph.nodes[v]["y"]
            seg = np.array([[ux, uy], [vx, vy]], dtype=np.float64)
        else:
            seg = np.asarray(geom.coords, dtype=np.float64)
        if pts and len(pts[-1]):
            # Avoid duplicating the join point.
            if np.allclose(pts[-1][-1], seg[0]):
                seg = seg[1:]
        pts.append(seg)
    return np.vstack(pts) if pts else np.zeros((0, 2))
