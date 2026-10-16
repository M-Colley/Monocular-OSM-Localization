"""Fetch and cache the OSM driving road graph for a city.

We project to a metric CRS (UTM zone for the city's centroid) so that
edge geometries are in meters and Euclidean comparison against the
visual-odometry trajectory is meaningful.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import LineString

from .net import prefer_reachable_host, transient_retry


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


def save_graphml_atomic(graph: nx.MultiDiGraph, path: Path | str) -> None:
    """``ox.save_graphml`` via a temporary file and ``os.replace``: a full disk
    or a killed process leaves the previous cache (or none), never a truncated
    file that every later run would load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.part{os.getpid()}")
    try:
        ox.save_graphml(graph, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_cached_graphml(path: Path | str) -> nx.MultiDiGraph | None:
    """Load a GraphML cache, or ``None`` if it is missing or unreadable.

    An unreadable file (e.g. truncated when the disk filled during an earlier
    write: "unclosed token") is renamed to ``*.corrupt`` rather than deleted,
    and the caller refetches. Before this, one truncated cache failed every
    later run of the same clip in the same way.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        return ox.load_graphml(path)
    except (ET.ParseError, ValueError, KeyError, EOFError) as e:
        bad = path.with_name(f"{path.name}.corrupt")
        os.replace(path, bad)
        print(f"      -> cached road graph unreadable ({type(e).__name__}: {e}); "
              f"moved to {bad.name}, refetching")
        return None


@transient_retry()
def _download_graph(place: str, network_type: str,
                    around: tuple[float, float, float] | None):
    """The Overpass download, retried on transient network errors only (a
    geocode failure for a misspelt city is not transient and fails at once)."""
    # Pin osmnx's single DNS resolution to a reachable Overpass machine (see
    # net.prefer_reachable_host); re-probed on every retry attempt.
    with prefer_reachable_host(ox.settings.overpass_url):
        if around is not None:
            lat, lon, radius_m = around
            return ox.graph_from_point(
                (lat, lon), dist=float(radius_m), network_type=network_type,
            )
        return ox.graph_from_place(place, network_type=network_type)


def fetch_city_graph(
    place: str,
    cache_path: Path | None = None,
    *,
    network_type: str = "drive",
    around: tuple[float, float, float] | None = None,
) -> RoadGraph:
    """Get the driving road graph for `place`. Caches to GraphML if asked.

    ``around=(lat, lon, radius_m)`` fetches a disc of that radius around
    the point instead of the whole named place — essential for
    mega-cities (e.g. London), where ``graph_from_place`` would pull
    millions of edges and the matcher's full scan would be infeasible.
    ``place`` is then used only for cache naming / downstream geocoding.
    """
    cache_path = Path(cache_path) if cache_path else None

    graph = load_cached_graphml(cache_path) if cache_path else None
    if graph is None:
        graph = _download_graph(place, network_type, around)
        graph = ox.project_graph(graph)  # UTM in meters
        if cache_path:
            save_graphml_atomic(graph, cache_path)

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
    turns: dict[int, int] | None = None,
) -> tuple[list[tuple], float]:
    """Build a walk starting from `first_edge`, extending by smallest
    heading-deviation choice at each intersection.

    `turns` maps walk indices (1-based: 1 = right after the first edge,
    2 = after two edges, ...) to the rank of the sorted out-edge to take
    there instead of the greedy zero-th. This lets the caller introduce
    deliberate turns at known walk depths — which is what gives the
    matcher diversity in turn positions. Multiple entries produce walks
    with multiple turns; real urban routes routinely contain two or more
    non-greedy turns, and a route with more turns than the enumeration
    allows is simply unmatchable.
    """
    walk: list[tuple] = [first_edge]
    length = _edge_length(graph, *first_edge)
    heading = _edge_end_heading(graph, *first_edge)
    # Track visited (undirected) EDGES so we don't drive in circles.
    # Without this, the greedy chooser will happily lap a roundabout or
    # block forever. Edges — not nodes — because real routes revisit
    # intersections all the time: loops back to the start, figure-eights,
    # and crossings of a previously-driven street all pass through an
    # already-seen node, and a node-based set made those true routes
    # structurally unenumerable (KITTI drive_0033 is a loop). Each
    # roundabout edge is still consumed at most once, so the anti-lap
    # protection is preserved.
    visited_edges = {frozenset((first_edge[0], first_edge[1]))}

    while length < target_length_m and len(walk) < max_depth:
        last_u, last_v, _ = walk[-1]
        out = list(graph.out_edges(last_v, keys=True))
        # Drop immediate reversal AND any re-drive of an earlier edge.
        out = [
            e for e in out
            if not (e[0] == last_v and e[1] == last_u)
            and frozenset((e[0], e[1])) not in visited_edges
        ]
        if not out:
            break
        out.sort(key=lambda e: _heading_dev(_edge_start_heading(graph, *e), heading))
        rank = turns.get(len(walk), 0) if turns else 0
        choice = out[rank] if rank < len(out) else out[0]
        walk.append(choice)
        length += _edge_length(graph, *choice)
        heading = _edge_end_heading(graph, *choice)
        visited_edges.add(frozenset((choice[0], choice[1])))

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
    _EXHAUSTED = object()

    out0 = list(graph.out_edges(start, keys=True))
    if not out0:
        return walks
    out0_sorted = sorted(out0, key=lambda e: (-_edge_length(graph, *e), e))

    seen: set[tuple] = set()

    def try_walk(first: tuple, turns: dict[int, int] | None) -> bool:
        """Build a walk and append it if it's new and long enough."""
        walk, length = _build_walk(
            graph, first, target_length_m, max_depth, turns=turns,
        )
        if length < 0.5 * target_length_m:
            return False
        key = tuple(walk)
        if key in seen:
            return False
        seen.add(key)
        walks.append(walk)
        return True

    # Per first edge, generate:
    #   1. the greedy walk (no deliberate turns),
    #   2. single-turn walks at successively deeper intersections — at
    #      ranks 1 AND 2, because at a 4-way intersection rank 1 is only
    #      one of left/right; without rank 2 the other direction is
    #      unreachable,
    #   3. two-turn walks over a bounded set of index pairs.
    # The multi-turn walks are what let us match real urban routes: the
    # Ulm GT route (Neutorstrasse south, then left onto Olgastrasse)
    # needs two non-greedy turns and was structurally impossible to
    # enumerate under the previous single-turn scheme — the matcher
    # could only ever offer parallel-street approximations.
    branch_indices = (1, 2, 3, 4, 6, 8)
    pair_indices = (
        (1, 3), (1, 4), (1, 6), (2, 4), (2, 6), (3, 6), (3, 8), (4, 8),
    )
    # Turn specs grouped in generations of increasing complexity.
    single_turn_specs = [
        {bi: rank} for bi in branch_indices for rank in (1, 2)
    ]
    two_turn_specs = [
        spec for bi, bj in pair_indices
        for spec in ({bi: 1, bj: 1}, {bi: 1, bj: 2})
    ]
    # Enumerate ALL first edges (out-degree is small), and round-robin
    # the walk budget across them one spec at a time. Each direction
    # draws from the same ordered mix: greedy first, then single-turn
    # specs interleaved 2:1 with two-turn specs (shallow branch indices
    # first). The interleave matters: iterating generation-by-generation
    # would let the 12x single-turn generation exhaust a 40-walk budget
    # at a 4-way intersection before ANY two-turn walk is tried — and
    # real urban routes (the Ulm GT route above) need two turns. The
    # previous scheme ([:3] first-edge cap + sequential exhaustion)
    # never tried the 4th direction of a standard 4-way intersection at
    # all, and let the first edge burn the whole budget on multi-turn
    # variants while later directions got nothing.
    spec_mix: list[dict[int, int] | None] = [None]
    st_iter, tt_iter = iter(single_turn_specs), iter(two_turn_specs)
    while True:
        chunk = list(islice(st_iter, 2)) + list(islice(tt_iter, 1))
        if not chunk:
            break
        spec_mix.extend(chunk)

    pending = [(first, iter(spec_mix)) for first in out0_sorted]
    while pending and len(walks) < max_walks:
        still_pending = []
        for first, it in pending:
            spec = next(it, _EXHAUSTED)
            if spec is _EXHAUSTED:
                continue
            if len(walks) >= max_walks:
                return walks
            try_walk(first, spec)
            still_pending.append((first, it))
        pending = still_pending

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
