"""Probabilistic map matching: a particle filter over position on the OSM graph.

The rest of the pipeline localises a drive by *enumerating* candidate walks on
the road graph and scoring their shape (``osm_data.walks_from_node`` +
``trajectory_matching``). Three measured properties make that brittle: the true
route can be missing from the bounded walk pool, shape fit is uncorrelated with
geographic error in dense grids, and a monocular trajectory has no metric speed
so its arc length is moving *time*, not distance. This module replaces
enumeration-then-scoring with sequential Bayesian inference, following the
odometry-only map-localisation formulation of

    M. A. Brubaker, A. Geiger, R. Urtasun. "Lost! Leveraging the Crowd for
    Probabilistic Visual Self-Localization", CVPR 2013; extended as
    "Map-Based Probabilistic Visual Self-Localization", TPAMI 38(4), 2016.

and the particle-filter variant of Floros, van der Zander, Leibe,
"OpenStreetSLAM", ICRA 2013. Every hypothesis the map admits is represented
(no pool to fall out of), the road constrains the motion (one-way streets are
directed edges), and the unknown monocular scale is a *state* that the map
itself pins down — the distance between two turns on the true route is fixed by
OSM, so only the right (position, scale) pairs keep explaining the heading
sequence.

State per particle
------------------
``edge``      directed edge index (one-way streets have no reverse edge)
``offset``    metres along that edge's polyline
``logk, eps`` scale: metres per odometry unit is ``exp(logk + eps)``. Monocular
              VO steps have unit norm, so this is speed x frame interval (for
              metric odometry ~1). ``logk`` is the hypothesis' own cruising
              scale (slow random walk, log-normal urban-speed prior); ``eps`` a
              mean-reverting Ornstein-Uhlenbeck speed deviation. A free random
              walk on log k would let every hypothesis absorb the
              distance-between-turns information.
``delta``     heading bias, world heading = odometry heading + delta (the VO
              yaw has an arbitrary origin and drifts). Rao-Blackwellised: each
              particle keeps a Gaussian (mean, variance) for it, updated by a
              scalar robust Kalman filter, instead of a sample.

Motion: advance ``offset`` by ``k * step`` (multiplicative noise); at the end of
an edge pick a successor (immediate U-turns excluded except at dead ends). The
transition weight is flat by default — a *normalised* turn prior makes the
junction topology evidence (forced turns at T-junctions and map borders beat
optional turns at crossroads by ~1-1.5 nats each), which on synthetic grids was
what lost the true route. Successors are drawn from a defensive mixture of the
prior and softmaxes guided by the odometry heading now and 1.5 s ahead, with
the exact prior/proposal weight correction (the vehicle is mid-turn when a
particle crosses the node, so the current heading alone starves the true
branch).

Measurement: the road tangent at the particle's new position against odometry
heading + delta — Gaussian predictive density of the wrapped innovation mixed
with a uniform outlier, measurement variance inflated by
``heading_corr_time_s / dt`` so the evidence per *second* does not depend on the
frame rate (consecutive VO headings are strongly correlated). Generic absolute
observations — position fixes with a Gaussian/uniform-outlier mixture (VPR
retrievals, OCR street anchors) and absolute headings (the sun) — plug in at any
step.

Output per step: the weighted posterior mode (grid binning + local centroid),
the posterior mass within ``confidence_radius_m`` of it (a calibrated-ish
confidence), the number of separated modes, the effective sample size; at the
end the particle genealogy gives a smoothed most-supported trajectory (path
storage with pruning, Jacob, Murray & Rubenthaler 2015, "Path storage in the
particle filter", Stat. Comput. 25).

Everything per step is vectorised numpy over the particle arrays; the map is
flattened into segment arrays so locating 10^5 particles is one
``searchsorted``. Resampling is filterpy's systematic resampler.

Particle budget: the true hypothesis must hold a few particles within ~10 m
along-track when the first turn happens, i.e. N >~ 5 x (directed road length)
/ 16 m — ~100k for a 350 km map, ~500k for a city of 1,600 km.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Iterable, Sequence

import networkx as nx
import numpy as np
from scipy.special import i0e, logsumexp

__all__ = [
    "RoadNetwork",
    "Odometry",
    "PositionObservation",
    "HeadingObservation",
    "FilterConfig",
    "FilterResult",
    "GraphParticleFilter",
    "odometry_from_trajectory",
    "odometry_from_arrays",
    "aggregate_odometry",
    "heading_increment_noise",
    "compass_deg_to_heading_rad",
    "evaluate_against_gt",
    "fetch_drive_graph",
    "run_filter",
    "wrap_angle",
]

_TWO_PI = 2.0 * np.pi


def wrap_angle(a):
    """Wrap angles (rad) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % _TWO_PI - np.pi


def compass_deg_to_heading_rad(compass_deg):
    """Compass bearing (deg, clockwise from north) -> math heading (rad, CCW
    from east), the convention of the projected map and of KITTI OXTS yaw.
    ``sun_heading.estimate_heading`` reports compass degrees."""
    return wrap_angle(np.pi / 2.0 - np.deg2rad(np.asarray(compass_deg, float)))


# --------------------------------------------------------------------------
# Map
# --------------------------------------------------------------------------

def fetch_drive_graph(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    place: str | None = None,
    cache_path: Path | str | None = None,
    network_type: str = "drive",
) -> nx.MultiDiGraph:
    """Fetch (or load from GraphML) a projected osmnx driving graph.

    ``bbox`` is ``(west, south, east, north)`` in degrees (osmnx 2 order);
    ``place`` a geocodable name. osmnx keeps one-way streets as a single
    directed edge, which is exactly what the filter needs.
    """
    import osmnx as ox

    if cache_path is not None and Path(cache_path).exists():
        return ox.load_graphml(Path(cache_path))
    if bbox is not None:
        G = ox.graph_from_bbox(bbox=tuple(bbox), network_type=network_type)
    elif place is not None:
        G = ox.graph_from_place(place, network_type=network_type)
    else:
        raise ValueError("need bbox or place")
    G = ox.project_graph(G)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(G, Path(cache_path))
    return G


def _edge_coords(G: nx.MultiDiGraph, u, v, data) -> np.ndarray:
    """Polyline of a directed edge, oriented u -> v."""
    geom = data.get("geometry")
    pu = (float(G.nodes[u]["x"]), float(G.nodes[u]["y"]))
    pv = (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))
    if geom is None:
        return np.array([pu, pv], dtype=np.float64)
    c = np.asarray(geom.coords, dtype=np.float64)[:, :2]
    if len(c) < 2:
        return np.array([pu, pv], dtype=np.float64)
    # Defensive: a geometry stored end-first would invert the travel direction.
    d_fwd = math.hypot(c[0, 0] - pu[0], c[0, 1] - pu[1]) + math.hypot(
        c[-1, 0] - pv[0], c[-1, 1] - pv[1])
    d_rev = math.hypot(c[-1, 0] - pu[0], c[-1, 1] - pu[1]) + math.hypot(
        c[0, 0] - pv[0], c[0, 1] - pv[1])
    if d_rev + 1e-6 < d_fwd:
        c = c[::-1].copy()
    return c


def _turn_logprior(succ_table, succ_head, head_end, kappa: float,
                   normalise: bool = False) -> np.ndarray:
    """Log transition weight of each successor; -inf on padding.

    ``kappa * (cos(turn) - 1)``: 0 for straight on, ``-kappa`` for a right
    angle, ``-2 kappa`` for a reversal. With ``normalise`` it becomes a proper
    distribution p(successor | edge) ∝ exp(kappa cos(turn)).

    Unnormalised is the default on purpose. A normalised prior turns the
    junction *topology* into evidence: a turn at a four-way junction costs
    log(1/3) (uniform) or log 0.11 (kappa=2) while a forced turn at a
    T-junction or map border costs log(1/2), so wrong hypotheses at
    T-junctions/borders win by ~1-1.5 nats per turn, and every straight
    junction crossing is taxed, which favours slow hypotheses that cross
    fewer junctions. Measured on synthetic irregular grids, that bias, not the
    odometry, was what lost the true route. Route choice is purposeful, so
    the default lets the heading likelihood alone select the branch.
    """
    valid = succ_table >= 0
    turn = wrap_angle(succ_head - head_end[:, None])
    logit = np.where(valid, kappa * (np.cos(turn) - 1.0), -np.inf)
    if not normalise:
        return logit
    with np.errstate(invalid="ignore"):
        norm = logsumexp(logit, axis=1, keepdims=True)
    return np.where(valid, logit - np.where(np.isfinite(norm), norm, 0.0), -np.inf)


@dataclass
class RoadNetwork:
    """A directed road graph flattened into vectorisable arrays.

    Coordinates are metres in ``crs`` minus ``origin`` (kept small so float32
    path storage stays sub-millimetre). Edge ``e`` owns segments
    ``seg_lo[e]:seg_hi[e]``; the global offset ``edge_base[e] + s`` of a point
    ``s`` metres along it is searchsorted into ``seg_gstart``.
    """

    crs: str
    origin: np.ndarray
    edge_keys: list
    edge_len: np.ndarray
    edge_base: np.ndarray
    seg_lo: np.ndarray
    seg_hi: np.ndarray
    seg_gstart: np.ndarray
    seg_xy0: np.ndarray
    seg_dir: np.ndarray
    seg_heading: np.ndarray
    edge_head_start: np.ndarray
    edge_head_end: np.ndarray
    edge_mid: np.ndarray
    reverse_of: np.ndarray
    succ_table: np.ndarray       # (E, D) successor edge ids, -1 padded
    succ_logprior: np.ndarray    # (E, D) log transition prior, -inf padded
    succ_head: np.ndarray        # (E, D) start heading of each successor
    allow_uturn: bool = False
    turn_prior_kappa: float = 0.0
    normalise_turn_prior: bool = False
    xy_max: np.ndarray | None = None   # max local coordinate of any vertex
    _transformers: dict = field(default_factory=dict, repr=False)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_graph(
        cls,
        G: nx.MultiDiGraph,
        *,
        allow_uturn: bool = False,
        uturn_at_dead_ends: bool = True,
        turn_prior_kappa: float = 0.0,
        normalise_turn_prior: bool = False,
    ) -> "RoadNetwork":
        """Build the arrays from an osmnx-style MultiDiGraph.

        Nodes need ``x``/``y``; edges may carry a shapely ``geometry``. An
        unprojected (lat/lon) graph is projected with osmnx first. Edge lengths
        are the projected polyline lengths (consistent with the geometry the
        particles move on), not the geodesic ``length`` attribute.

        ``allow_uturn=False`` removes an edge's own reverse twin from its
        successors (an immediate U-turn), except at dead ends when
        ``uturn_at_dead_ends`` — a vehicle can turn round at the end of a
        cul-de-sac or at the map border. ``turn_prior_kappa`` and
        ``normalise_turn_prior`` set the transition weight
        (:func:`_turn_logprior`); the default (0, unnormalised) weighs every
        feasible branch equally, so junction topology is not evidence.
        """
        crs = G.graph.get("crs")
        projected = True
        try:
            import osmnx as ox
            projected = bool(crs) and ox.projection.is_projected(crs)
        except Exception:  # pragma: no cover - osmnx missing
            pass
        if not projected:
            import osmnx as ox
            G = ox.project_graph(G)
            crs = G.graph.get("crs")

        keys: list = []
        polys: list[np.ndarray] = []
        for u, v, k, data in G.edges(keys=True, data=True):
            c = _edge_coords(G, u, v, data)
            # drop repeated vertices (zero-length segments)
            d = np.hypot(*np.diff(c, axis=0).T)
            keep = np.concatenate([[True], d > 1e-6])
            c = c[keep]
            if len(c) < 2:  # degenerate edge: give it a tiny length
                c = np.vstack([c[:1], c[:1] + [0.05, 0.0]])
            keys.append((u, v, k))
            polys.append(c)
        if not keys:
            raise ValueError("graph has no edges")

        allxy = np.vstack(polys)
        origin = np.floor(allxy.min(axis=0) / 1000.0) * 1000.0
        E = len(keys)
        nseg = np.array([len(p) - 1 for p in polys])
        seg_lo = np.concatenate([[0], np.cumsum(nseg)[:-1]]).astype(np.int64)
        seg_hi = (seg_lo + nseg).astype(np.int64)
        xy0 = np.vstack([p[:-1] for p in polys]) - origin
        xy1 = np.vstack([p[1:] for p in polys]) - origin
        dvec = xy1 - xy0
        slen = np.hypot(dvec[:, 0], dvec[:, 1])
        sdir = dvec / slen[:, None]
        shead = np.arctan2(dvec[:, 1], dvec[:, 0])
        edge_len = np.add.reduceat(slen, seg_lo)
        edge_base = np.concatenate([[0.0], np.cumsum(edge_len)[:-1]])
        # segment start offset within its edge, then globally
        seg_edge = np.repeat(np.arange(E), nseg)
        cum = np.cumsum(slen)
        local = cum - slen - (cum[seg_lo] - slen[seg_lo])[seg_edge]
        seg_gstart = edge_base[seg_edge] + local

        head_start = shead[seg_lo]
        head_end = shead[seg_hi - 1]
        # midpoint (half arc length) per edge, for reverse-twin detection
        mids = np.empty((E, 2))
        for e, p in enumerate(polys):
            seg = np.hypot(*np.diff(p, axis=0).T)
            c = np.concatenate([[0.0], np.cumsum(seg)])
            half = c[-1] / 2.0
            mids[e, 0] = np.interp(half, c, p[:, 0])
            mids[e, 1] = np.interp(half, c, p[:, 1])
        mids -= origin

        out_by_node: dict = {}
        for i, (u, v, k) in enumerate(keys):
            out_by_node.setdefault(u, []).append(i)

        reverse_of = np.full(E, -1, dtype=np.int64)
        for i, (u, v, k) in enumerate(keys):
            best, best_d = -1, np.inf
            for j in out_by_node.get(v, []):
                if keys[j][1] != u or j == i and u != v:
                    continue
                if j == i:  # a self-loop is not its own reverse
                    continue
                dl = abs(edge_len[j] - edge_len[i])
                dm = float(np.hypot(*(mids[j] - mids[i])))
                if dl <= 1.0 + 0.02 * edge_len[i] and dm <= 2.0 + 0.02 * edge_len[i]:
                    if dm < best_d:
                        best, best_d = j, dm
            reverse_of[i] = best

        succ_lists: list[list[int]] = []
        for i, (u, v, k) in enumerate(keys):
            out = list(out_by_node.get(v, []))
            if not allow_uturn and reverse_of[i] >= 0:
                no_u = [j for j in out if j != reverse_of[i]]
                if no_u or not uturn_at_dead_ends:
                    out = no_u
            succ_lists.append(out)
        D = max(1, max(len(s) for s in succ_lists))
        succ_table = np.full((E, D), -1, dtype=np.int64)
        for i, s in enumerate(succ_lists):
            succ_table[i, : len(s)] = s
        valid = succ_table >= 0
        succ_head = np.where(valid, head_start[np.maximum(succ_table, 0)], 0.0)
        succ_logprior = _turn_logprior(succ_table, succ_head, head_end, turn_prior_kappa,
                                       normalise_turn_prior)

        return cls(
            crs=str(crs), origin=origin, edge_keys=keys, edge_len=edge_len,
            edge_base=edge_base, seg_lo=seg_lo, seg_hi=seg_hi,
            seg_gstart=seg_gstart, seg_xy0=xy0, seg_dir=sdir, seg_heading=shead,
            edge_head_start=head_start, edge_head_end=head_end, edge_mid=mids,
            reverse_of=reverse_of, succ_table=succ_table,
            succ_logprior=succ_logprior, succ_head=succ_head,
            allow_uturn=allow_uturn, turn_prior_kappa=turn_prior_kappa,
            normalise_turn_prior=normalise_turn_prior,
            xy_max=allxy.max(axis=0) - origin,
        )

    def with_turn_prior(self, kappa: float, normalise: bool = False) -> "RoadNetwork":
        """A copy sharing all arrays but with a different transition weight."""
        from dataclasses import replace
        return replace(self, turn_prior_kappa=float(kappa), normalise_turn_prior=normalise,
                       _transformers={},
                       succ_logprior=_turn_logprior(self.succ_table, self.succ_head,
                                                    self.edge_head_end, kappa, normalise))

    # ---- queries ----------------------------------------------------------
    @property
    def n_edges(self) -> int:
        return len(self.edge_len)

    @property
    def total_length_m(self) -> float:
        """Directed road length (a two-way street counts twice)."""
        return float(self.edge_len.sum())

    @property
    def bbox_area_m2(self) -> float:
        lo = self.seg_xy0.min(axis=0)
        hi = self.seg_xy0.max(axis=0)
        return float(max(1.0, (hi[0] - lo[0]) * (hi[1] - lo[1])))

    def successors(self, e: int) -> list[int]:
        row = self.succ_table[e]
        return [int(j) for j in row if j >= 0]

    def locate(self, edge: np.ndarray, offset: np.ndarray):
        """(x, y, heading, segment) of points ``offset`` m along ``edge``.

        x, y are local (``origin`` subtracted); heading is the travel-direction
        tangent in radians, CCW from the CRS x axis (east).
        """
        g = self.edge_base[edge] + offset
        seg = np.searchsorted(self.seg_gstart, g, side="right") - 1
        seg = np.clip(seg, self.seg_lo[edge], self.seg_hi[edge] - 1)
        t = g - self.seg_gstart[seg]
        x = self.seg_xy0[seg, 0] + self.seg_dir[seg, 0] * t
        y = self.seg_xy0[seg, 1] + self.seg_dir[seg, 1] * t
        return x, y, self.seg_heading[seg], seg

    def sample_uniform(self, n: int, rng: np.random.Generator,
                       disc: tuple[float, float, float] | None = None):
        """``n`` points uniform over directed road length, optionally restricted
        to a disc ``(x, y, radius_m)`` given in CRS coordinates."""
        if disc is None:
            w = self.edge_len / self.edge_len.sum()
            e = rng.choice(self.n_edges, size=n, p=w)
            return e, rng.random(n) * self.edge_len[e]
        cx, cy, r = disc
        cx, cy = cx - self.origin[0], cy - self.origin[1]
        # candidate edges: any part of the edge could be inside the disc
        dm = np.hypot(self.edge_mid[:, 0] - cx, self.edge_mid[:, 1] - cy)
        cand = np.nonzero(dm <= r + self.edge_len / 2.0 + 1.0)[0]
        if len(cand) == 0:
            raise ValueError("disc prior contains no road")
        w = self.edge_len[cand] / self.edge_len[cand].sum()
        out_e = np.empty(0, dtype=np.int64)
        out_s = np.empty(0)
        for _ in range(200):
            m = max(4 * (n - len(out_e)), 1024)
            e = cand[rng.choice(len(cand), size=m, p=w)]
            s = rng.random(m) * self.edge_len[e]
            x, y, _, _ = self.locate(e, s)
            ok = np.hypot(x - cx, y - cy) <= r
            out_e = np.concatenate([out_e, e[ok]])
            out_s = np.concatenate([out_s, s[ok]])
            if len(out_e) >= n:
                return out_e[:n], out_s[:n]
        raise ValueError("disc prior too small to sample from")

    # ---- coordinates --------------------------------------------------------
    def _tf(self, direction: str):
        if direction not in self._transformers:
            from pyproj import Transformer
            if direction == "inv":
                self._transformers[direction] = Transformer.from_crs(
                    self.crs, "EPSG:4326", always_xy=True)
            else:
                self._transformers[direction] = Transformer.from_crs(
                    "EPSG:4326", self.crs, always_xy=True)
        return self._transformers[direction]

    def local_to_latlon(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2) + self.origin
        lon, lat = self._tf("inv").transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lat, lon])

    def latlon_to_local(self, latlon: np.ndarray) -> np.ndarray:
        ll = np.asarray(latlon, float).reshape(-1, 2)
        x, y = self._tf("fwd").transform(ll[:, 1], ll[:, 0])
        return np.column_stack([x, y]) - self.origin

    def crs_to_local(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, float).reshape(-1, 2) - self.origin

    def local_to_crs(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, float).reshape(-1, 2) + self.origin

    def stats(self) -> dict:
        lo = self.seg_xy0.min(axis=0)
        hi = self.seg_xy0.max(axis=0)
        oneway = int((self.reverse_of < 0).sum())
        return {
            "crs": self.crs,
            "n_directed_edges": self.n_edges,
            "n_segments": int(len(self.seg_gstart)),
            "directed_length_km": self.total_length_m / 1000.0,
            "oneway_edges": oneway,
            "bbox_km": [float((hi[0] - lo[0]) / 1000.0), float((hi[1] - lo[1]) / 1000.0)],
            "max_successors": int(self.succ_table.shape[1]),
        }


# --------------------------------------------------------------------------
# Odometry
# --------------------------------------------------------------------------

@dataclass
class Odometry:
    """Per-step relative motion fed to the filter.

    ``step[t]``     distance travelled in step t, in odometry units (VO units,
                    or metres when ``metric``); 0 = stationary / no estimate.
    ``heading[t]``  travel heading at the end of step t, radians, in the
                    odometry's own (arbitrary) frame; NaN = unknown.
    ``dt[t]``       seconds covered by step t.
    ``unit_seconds`` for non-metric odometry: seconds of driving one unit
                    represents (the frame interval for unit-norm VO steps). The
                    speed prior is converted to a log_k prior with it.
    ``frame_index[t]`` raw frame index at the end of step t (GT alignment).
    ``heading_var[t]`` optional extra variance (rad^2) of the heading-increment
                    error in step t (see :func:`heading_increment_noise`); added
                    to the heading-bias process noise for that step.
    """

    step: np.ndarray
    heading: np.ndarray
    dt: np.ndarray
    metric: bool = False
    unit_seconds: float | None = None
    frame_index: np.ndarray | None = None
    heading_var: np.ndarray | None = None

    def __post_init__(self):
        self.step = np.clip(np.asarray(self.step, float), 0.0, None)
        self.heading = np.asarray(self.heading, float)
        self.dt = np.broadcast_to(np.asarray(self.dt, float), self.step.shape).copy()
        if self.frame_index is None:
            self.frame_index = np.arange(1, len(self.step) + 1)
        self.frame_index = np.asarray(self.frame_index, int)
        if self.heading_var is not None:
            self.heading_var = np.asarray(self.heading_var, float)
            if len(self.heading_var) != len(self.step):
                raise ValueError("heading_var must have one entry per step")
        if not (len(self.step) == len(self.heading) == len(self.dt) == len(self.frame_index)):
            raise ValueError("odometry arrays must have equal length")

    def __len__(self) -> int:
        return len(self.step)

    @property
    def times(self) -> np.ndarray:
        """Time at the end of each step (s), with t=0 before the first step."""
        return np.concatenate([[0.0], np.cumsum(self.dt)])


def odometry_from_arrays(step_length, heading_rad, dt, *, metric: bool = True,
                         unit_seconds: float | None = None,
                         frame_index=None) -> Odometry:
    """Generic odometry (e.g. wheel/INS/OXTS-derived): per-step length (metres
    if ``metric``), heading (rad, any fixed frame), dt (s)."""
    return Odometry(step=np.asarray(step_length, float),
                    heading=np.asarray(heading_rad, float), dt=dt, metric=metric,
                    unit_seconds=unit_seconds, frame_index=frame_index)


def _vo_plane_axes(centers: np.ndarray, xz: np.ndarray) -> np.ndarray:
    """The 3x2 linear map the VO used to project camera centres to ``xz``.

    ``visual_odometry._fit_plane_projection`` computes ``(C - mean) @ A`` but
    does not return ``A``; recover it by least squares so direction vectors
    (camera forward axes) can be projected into the very same plane. Falls back
    to the raw x/z drop when the fit is degenerate or does not reproduce ``xz``.
    """
    raw = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    C = np.asarray(centers, float)
    X = np.asarray(xz, float)
    if len(C) < 4:
        return raw
    Cc = C - C.mean(axis=0)
    Xc = X - X.mean(axis=0)
    A, *_ = np.linalg.lstsq(Cc, Xc, rcond=None)
    resid = np.linalg.norm(Cc @ A - Xc) / max(np.linalg.norm(Xc), 1e-9)
    if not np.all(np.isfinite(A)) or resid > 1e-3:
        return raw
    # orthonormalise the columns (the true map is a pair of orthonormal axes)
    q, r = np.linalg.qr(A)
    q = q * np.sign(np.diag(r))[None, :]
    return q


def odometry_from_trajectory(
    traj,
    frame_dt: float,
    *,
    heading_source: str = "rotation",
    step_lengths_m: np.ndarray | None = None,
    min_step: float = 1e-6,
    adaptive_heading_noise: bool = True,
) -> Odometry:
    """Per-step odometry from ``visual_odometry.Trajectory``.

    Step length is ``|xz[i] - xz[i-1]|`` (unit-norm VO steps; 0 where the pair
    was invalid/stationary — the chain held the pose). ``heading_source``:

    * ``"rotation"`` — yaw of the camera's optical axis projected into the VO
      ground plane. Rotation from an essential matrix is far better
      conditioned than its translation direction, so this is the less noisy
      choice; a fixed camera-mount yaw offset is absorbed by ``delta``.
    * ``"translation"`` — direction of the xz step itself.

    ``step_lengths_m`` (e.g. ``ground_flow_scale.step_lengths``) replaces the
    unit steps with metric ones and marks the odometry metric.

    ``adaptive_heading_noise`` attaches :func:`heading_increment_noise`.
    """
    xz = np.asarray(traj.xz, float)
    n = len(xz)
    if n < 2:
        raise ValueError("trajectory needs at least 2 poses")
    d = np.diff(xz, axis=0)
    step = np.hypot(d[:, 0], d[:, 1])
    valid = np.asarray(traj.valid, bool)[1:] & (step > min_step)
    step = np.where(valid, step, 0.0)
    if heading_source == "translation":
        heading = np.where(valid, np.arctan2(d[:, 1], d[:, 0]), np.nan)
    elif heading_source == "rotation":
        A = _vo_plane_axes(traj.centers, xz)
        fwd = np.asarray(traj.rotations, float)[:, 2, :]  # optical axis in world
        f2 = fwd @ A
        yaw = np.arctan2(f2[:, 1], f2[:, 0])
        heading = np.where(valid, yaw[1:], np.nan)
    else:
        raise ValueError(f"unknown heading_source {heading_source!r}")
    metric = step_lengths_m is not None
    if metric:
        sl = np.asarray(step_lengths_m, float)
        if len(sl) != len(step):
            raise ValueError("step_lengths_m must have len(traj.xz) - 1 entries")
        step = np.where(valid, np.clip(sl, 0.0, None), 0.0)
    odom = Odometry(step=step, heading=heading, dt=np.full(len(step), float(frame_dt)),
                    metric=metric, unit_seconds=None if metric else float(frame_dt),
                    frame_index=np.arange(1, n))
    if adaptive_heading_noise:
        odom.heading_var = heading_increment_noise(odom)
    return odom


def heading_increment_noise(odom: Odometry, window_s: float = 1.0) -> np.ndarray:
    """Per-step variance (rad^2) of the odometry's heading-increment error,
    estimated WITHOUT ground truth.

    A car's yaw rate is smooth on a ~1 s scale; what a moving average of the
    increments does not explain is odometry noise. Monocular VO yaw errors are
    heavy-tailed (on KITTI the increments' std is 2-6x their MAD): quiet steps
    plus occasional glitches that rotate the whole VO frame. Using each step's
    own squared residual as that step's process noise lets the heading bias
    jump exactly where the VO glitched instead of everywhere. On KITTI this
    GT-free estimate equals the error measured against OXTS (0.37 vs 0.44
    deg/step on drive 0009, 0.90 vs 0.90 on 0033).
    """
    from scipy.ndimage import uniform_filter1d

    T = len(odom)
    out = np.zeros(T)
    known = np.flatnonzero(np.isfinite(odom.heading))
    if len(known) < 5:
        return out
    dh = wrap_angle(np.diff(odom.heading[known]))
    dt = float(np.median(odom.dt)) if T else 0.1
    win = max(3, int(round(window_s / max(dt, 1e-6))) | 1)
    r = dh - uniform_filter1d(dh, win, mode="nearest")
    r *= math.sqrt(win / (win - 1.0))        # residual-from-local-mean shrinkage
    out[known[1:]] = r * r
    return out


def aggregate_odometry(odom: Odometry, factor: int) -> Odometry:
    """Merge every ``factor`` consecutive steps into one (fewer, longer filter
    steps). Distance is summed (path length); the heading is the
    length-weighted circular mean of the group's known headings."""
    factor = int(factor)
    if factor <= 1:
        return odom
    T = len(odom)
    groups = np.arange(T) // factor
    G = int(groups[-1]) + 1 if T else 0
    step = np.bincount(groups, odom.step, minlength=G)
    dt = np.bincount(groups, odom.dt, minlength=G)
    known = np.isfinite(odom.heading) & (odom.step > 0)
    w = np.where(known, odom.step, 0.0)
    h = np.where(known, odom.heading, 0.0)
    sx = np.bincount(groups, w * np.cos(h), minlength=G)
    sy = np.bincount(groups, w * np.sin(h), minlength=G)
    heading = np.where(np.hypot(sx, sy) > 0, np.arctan2(sy, sx), np.nan)
    last = np.minimum(np.arange(G) * factor + factor - 1, T - 1)
    hv = (None if odom.heading_var is None
          else np.bincount(groups, odom.heading_var, minlength=G))
    return Odometry(step=step, heading=heading, dt=dt, metric=odom.metric,
                    unit_seconds=odom.unit_seconds,
                    frame_index=odom.frame_index[last], heading_var=hv)


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------

@dataclass
class PositionObservation:
    """An absolute position fix at filter index ``step`` (0 = before the first
    odometry step, t = after step t-1's motion). ``xy`` in the map CRS, e.g. a
    VPR retrieval's reference position or an OCR street anchor's location.
    ``outlier`` is the probability the fix is unrelated to the true position
    (uniform over the map)."""

    step: int
    x: float
    y: float
    sigma_m: float = 30.0
    outlier: float = 0.2


@dataclass
class HeadingObservation:
    """An absolute travel heading (rad, CCW from east) at filter index
    ``step`` — e.g. the sun heading (use :func:`compass_deg_to_heading_rad`)."""

    step: int
    heading_rad: float
    kappa: float = 4.0
    outlier: float = 0.1


# --------------------------------------------------------------------------
# Filter
# --------------------------------------------------------------------------

@dataclass
class FilterConfig:
    n_particles: int = 50_000
    seed: int = 0
    # Relative-heading likelihood. delta (world = odometry heading + delta) is
    # Rao-Blackwellised: every particle carries a Gaussian (mean, var) for it,
    # updated by a scalar robust Kalman filter, and is weighted by the
    # predictive likelihood (Gaussian on the wrapped residual + uniform
    # outlier). Sampling delta instead lets it lag the VO drift and soak up
    # mid-turn residuals, which costs the true hypothesis nats on every
    # straight. The measurement variance is inflated by corr_time/dt so the
    # evidence per *second* is independent of the step rate (consecutive VO
    # headings are strongly correlated).
    heading_sigma_deg: float = 10.0
    heading_outlier: float = 0.1
    heading_corr_time_s: float = 1.0
    # Junction zone: within this many metres before/after a node, the vehicle
    # is on its turning arc, so the expected heading is the whole interval
    # between the incoming and outgoing road headings (residual = distance to
    # the interval, 0 inside). Without it the road heading jumps at the OSM
    # node while the car is mid-arc; on KITTI 0009 (a slow 90 deg turn over
    # ~15 m) that killed the true hypothesis even with oracle odometry.
    junction_zone_m: float = 12.0
    # process noise
    step_noise: float = 0.05          # multiplicative (log-normal) per step
    # log k = log k_bar + eps: k_bar is the hypothesis' own cruising scale
    # (slow random walk), eps a mean-reverting (Ornstein-Uhlenbeck) speed
    # deviation. A free random walk on log k lets every hypothesis absorb the
    # distance-between-turns information (and drift slow to dodge junctions).
    logk_rw: float = 0.01             # random walk of log k_bar, per sqrt(s)
    speed_dev_sigma: float = 0.3      # stationary std of eps (unit-step odometry)
    metric_speed_dev_sigma: float = 0.05  # same, metric odometry
    speed_dev_tau_s: float = 10.0     # OU time constant of eps
    delta_rw_deg: float = 1.0         # heading-bias random walk, deg per sqrt(s)
    delta_init_deg: float = 10.0      # prior std of delta around the road tangent
    # priors
    # Non-metric odometry: cruising speed ~ log-normal(median, sigma), clipped
    # to speed_range_mps (95 % of the default prior lies in 3-21 m/s). An
    # urban-driving prior, not a fitted value: k_bar is pinned only at the
    # *second* turn, so prior mass at implausible speeds is wasted particles.
    speed_median_mps: float = 8.0
    speed_log_sigma: float = 0.5
    speed_range_mps: tuple = (1.5, 30.0)
    metric_logk_sigma: float = 0.15        # metric odometry: log k_bar ~ N(0, .)
    logk_bounds_margin: float = 0.5        # clip log k_bar to prior range +- this
    # transitions
    turn_prior_kappa: float | None = None  # None: keep the RoadNetwork's weights
    normalise_turn_prior: bool = False     # see _turn_logprior
    # Importance proposal at junctions: a mixture of the prior and softmax
    # choices guided by the odometry heading now and ``proposal_lookahead_s``
    # later. A particle crosses the node while the car is still mid-turn, so
    # the current heading alone proposes "straight on" and the true hypothesis
    # starves; the look-ahead heading (a fixed lag of odometry) fixes that. The
    # weights are corrected by prior/proposal, so the target posterior is
    # unchanged — only the Monte Carlo variance is.
    guided_turns: bool = True
    turn_proposal_kappa: float = 4.0
    proposal_lookahead_s: float = 1.5
    defensive_mix: float = 0.2
    # resampling / summaries
    resample_ess_frac: float = 0.5
    mode_grid_m: float = 20.0
    confidence_radius_m: float = 25.0
    mode_sep_m: float = 100.0
    min_mode_mass: float = 0.05
    # Genealogy for the smoothed route. Ancestors are stored every path_stride
    # filter steps (ancestry across skipped steps is composed through the
    # resampling indices, so nothing is lost but temporal resolution; the route
    # is linearly interpolated between stored points). None = automatic: the
    # smallest stride that keeps the un-pruned store under path_budget_mb —
    # without a bound, 1.5e6 particles x 330 steps needed ~10 GB per process.
    path_storage: bool = True
    path_stride: int | None = None
    path_budget_mb: float = 1500.0
    prune_every: int = 20             # in stored generations
    # initialisation
    init_disc: tuple | None = None    # (x, y, r) in the map CRS
    max_transitions: int = 200


@dataclass
class FilterResult:
    times: np.ndarray            # (T+1,) s
    frame_index: np.ndarray      # (T+1,) raw frame index of each estimate
    mode_xy: np.ndarray          # (T+1, 2) map CRS
    mode_latlon: np.ndarray      # (T+1, 2)
    confidence: np.ndarray       # (T+1,) posterior mass within r of the mode
    n_modes: np.ndarray          # (T+1,) separated modes holding >= min mass
    ess: np.ndarray              # (T+1,)
    resampled: np.ndarray        # (T+1,) bool
    k_mode: np.ndarray           # (T+1,) mean metres/unit of mode particles
    smoothed_xy: np.ndarray | None
    smoothed_latlon: np.ndarray | None
    final_xy: np.ndarray         # (N, 2) particles, map CRS
    final_w: np.ndarray          # (N,)
    gt_mass: np.ndarray | None   # (T+1,) mass within r of diagnostic points
    runtime_s: float
    n_reinit: int
    config: dict
    map_stats: dict

    def to_json(self) -> dict:
        def arr(a):
            return None if a is None else np.asarray(a).tolist()
        return {
            "times": arr(self.times), "frame_index": arr(self.frame_index),
            "mode_latlon": arr(self.mode_latlon), "confidence": arr(self.confidence),
            "n_modes": arr(self.n_modes), "ess": arr(self.ess),
            "k_mode": arr(self.k_mode), "smoothed_latlon": arr(self.smoothed_latlon),
            "gt_mass": arr(self.gt_mass), "runtime_s": self.runtime_s,
            "n_reinit": self.n_reinit, "config": self.config,
            "map_stats": self.map_stats,
        }


class _PathStore:
    """Particle genealogy with pruning of dead branches (Jacob et al. 2015).

    Nodes are appended per stored generation in order, so a parent id is
    always smaller than its child's; dead lineages are dropped every
    ``prune_every`` generations. Old generations coalesce to a single
    ancestor, keeping memory ~O(T + N). 16 bytes per stored node.
    """

    def __init__(self, n: int):
        self.parent = np.empty(0, np.int32)
        self.x = np.empty(0, np.float32)
        self.y = np.empty(0, np.float32)
        self.step = np.empty(0, np.int32)
        self._chunks: list = []
        self.node_of = np.full(n, -1, np.int64)
        self.n = 0

    def add(self, step: int, x: np.ndarray, y: np.ndarray) -> None:
        ids = np.arange(self.n, self.n + len(x), dtype=np.int64)
        self._chunks.append((self.node_of.astype(np.int32), x.astype(np.float32),
                             y.astype(np.float32), np.full(len(x), step, np.int32)))
        self.node_of = ids
        self.n += len(x)

    def _flush(self) -> None:
        if self._chunks:
            p, x, y, s = zip(*self._chunks)
            self.parent = np.concatenate([self.parent, *p])
            self.x = np.concatenate([self.x, *x])
            self.y = np.concatenate([self.y, *y])
            self.step = np.concatenate([self.step, *s])
            self._chunks = []

    def reindex(self, idx: np.ndarray) -> None:
        self.node_of = self.node_of[idx]

    def restart(self) -> None:
        """Every particle was re-drawn from the prior: new lineages, no parents."""
        self.node_of = np.full_like(self.node_of, -1)

    def prune(self) -> None:
        self._flush()
        if self.n == 0:
            return
        alive = np.zeros(self.n, bool)
        alive[self.node_of[self.node_of >= 0]] = True
        # steps are contiguous, increasing blocks; walk them backwards
        bounds = np.flatnonzero(np.diff(self.step)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [self.n]])
        for a, b in zip(starts[::-1], ends[::-1]):
            m = alive[a:b]
            if m.any():
                par = self.parent[a:b][m]
                alive[par[par >= 0]] = True
        new_id = np.cumsum(alive) - 1
        par = self.parent[alive]
        self.parent = np.where(par >= 0, new_id[np.maximum(par, 0)], -1).astype(np.int32)
        self.x = self.x[alive]
        self.y = self.y[alive]
        self.step = self.step[alive]
        self.node_of = np.where(self.node_of >= 0, new_id[np.maximum(self.node_of, 0)], -1)
        self.n = int(alive.sum())

    def lineage(self, particle: int, n_steps: int) -> np.ndarray:
        self._flush()
        out = np.full((n_steps, 2), np.nan)
        node = int(self.node_of[particle])
        while node >= 0:
            out[self.step[node]] = (self.x[node], self.y[node])
            node = int(self.parent[node])
        return out

    @property
    def size(self) -> int:
        return self.n + sum(len(c[1]) for c in self._chunks)


@contextlib.contextmanager
def _global_numpy_seed(seed: int):
    """filterpy's resamplers draw from numpy's *global* RNG. Seed it from the
    filter's own Generator for reproducibility, and restore the caller's
    global state afterwards so the filter has no side effects."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _interval_residual(h, a, b):
    """Signed angle from heading ``h`` to the shorter arc from ``a`` to ``b``
    (0 inside the arc), in the convention expected - observed."""
    span = wrap_angle(b - a)
    rel = wrap_angle(h - a)
    inside = np.where(span >= 0, (rel >= 0) & (rel <= span), (rel <= 0) & (rel >= span))
    da = wrap_angle(a - h)
    db = wrap_angle(b - h)
    out = np.where(np.abs(da) <= np.abs(db), da, db)
    return np.where(inside, 0.0, out)


def _systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from filterpy.monte_carlo import systematic_resample

    w = np.asarray(w, float)
    w = w / w.sum()
    with _global_numpy_seed(int(rng.integers(0, 2**32 - 1))):
        try:
            return np.asarray(systematic_resample(w), dtype=np.int64)
        except IndexError:
            # filterpy's loop can run past the end when cumsum(w)[-1] rounds
            # just below the last stratum; same algorithm, clipped.
            u = np.random.random()
    pos = (u + np.arange(len(w))) / len(w)
    return np.minimum(np.searchsorted(np.cumsum(w), pos, side="right"), len(w) - 1)


class GraphParticleFilter:
    """Sequential Monte Carlo localisation on a directed road graph."""

    def __init__(self, net: RoadNetwork, cfg: FilterConfig | None = None, *,
                 metric: bool = False, unit_seconds: float = 0.1):
        self.cfg = cfg or FilterConfig()
        if self.cfg.turn_prior_kappa is not None and (
                self.cfg.turn_prior_kappa != net.turn_prior_kappa
                or self.cfg.normalise_turn_prior != net.normalise_turn_prior):
            net = net.with_turn_prior(self.cfg.turn_prior_kappa, self.cfg.normalise_turn_prior)
        self.net = net
        self.metric = metric
        self.unit_seconds = unit_seconds
        self.rng = np.random.default_rng(self.cfg.seed)
        n = self.cfg.n_particles
        self.edge = np.zeros(n, np.int64)
        self.offset = np.zeros(n)
        self.logk = np.zeros(n)       # log k_bar: the hypothesis' cruising scale
        self.eps = np.zeros(n)        # OU speed deviation, log k = logk + eps
        self.delta = np.zeros(n)      # posterior mean of the heading bias
        self.delta_var = np.zeros(n)  # ... and its variance (Rao-Blackwellised)
        self.logw = np.full(n, -math.log(n))
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.road_heading = np.zeros(n)
        self.prev_head = np.full(n, np.nan)   # end heading of the previous edge
        self.delta_ready = False
        self.n_reinit = 0
        self.store = _PathStore(n) if self.cfg.path_storage else None
        self._log_out_density = -math.log(net.bbox_area_m2)
        self._k_bounds = self._logk_prior_bounds()
        # dense binning grid for the per-step summary when the map is small
        # enough (local coordinates are >= 0 by construction of the origin)
        hi = net.xy_max if net.xy_max is not None else (
            net.seg_xy0.max(axis=0) + net.edge_len.max())
        shape = (int(hi[0] / self.cfg.mode_grid_m) + 3, int(hi[1] / self.cfg.mode_grid_m) + 3)
        self._dense_grid = shape if shape[0] * shape[1] <= 20_000_000 else None

    # ---- priors -----------------------------------------------------------
    def _logk_prior_bounds(self) -> tuple[float, float]:
        c = self.cfg
        if self.metric:
            lo, hi = -3.0 * c.metric_logk_sigma, 3.0 * c.metric_logk_sigma
        else:
            vmin, vmax = c.speed_range_mps
            lo = math.log(vmin * self.unit_seconds)
            hi = math.log(vmax * self.unit_seconds)
        return lo, hi

    def _sample_logk(self, n: int) -> np.ndarray:
        lo, hi = self._k_bounds
        c = self.cfg
        if self.metric:
            return self.rng.normal(0.0, c.metric_logk_sigma, n)
        mu = math.log(c.speed_median_mps * self.unit_seconds)
        return np.clip(self.rng.normal(mu, c.speed_log_sigma, n), lo, hi)

    @property
    def _dev_sigma(self) -> float:
        return self.cfg.metric_speed_dev_sigma if self.metric else self.cfg.speed_dev_sigma

    # ---- lifecycle --------------------------------------------------------
    def initialise(self, idx: np.ndarray | None = None) -> None:
        """Uniform over road length (or the disc prior), both directions,
        k from its prior. ``delta`` is set lazily at the first known heading."""
        n = self.cfg.n_particles
        if idx is None:
            idx = np.arange(n)
        e, s = self.net.sample_uniform(len(idx), self.rng, self.cfg.init_disc)
        self.edge[idx] = e
        self.offset[idx] = s
        self.logk[idx] = self._sample_logk(len(idx))
        self.eps[idx] = self.rng.normal(0.0, self._dev_sigma, len(idx))
        self.delta[idx] = 0.0
        self.delta_var[idx] = 0.0
        self.prev_head[idx] = np.nan
        self.logw[:] = -math.log(n)
        self._relocate()

    def _relocate(self) -> None:
        self.x, self.y, self.road_heading, _ = self.net.locate(self.edge, self.offset)

    def _init_delta(self, heading: float) -> None:
        """delta ~ N(road tangent - odometry heading, delta_init^2): the first
        heading only fixes the unknown yaw origin, it is not evidence."""
        self.delta = wrap_angle(self.road_heading - heading)
        self.delta_var = np.full(len(self.delta), np.deg2rad(self.cfg.delta_init_deg) ** 2)
        self.delta_ready = True

    # ---- motion -------------------------------------------------------------
    def _choose_successors(self, e: np.ndarray, h_preds: Sequence[np.ndarray]):
        """Sample successor edges; return (edges, log weight correction, dead).

        ``h_preds`` are predicted world headings (one array per guidance
        component, NaN = no guidance). q = mix * prior + (1 - mix) * mean of
        prior-times-von-Mises softmaxes; the weight is multiplied by prior / q.
        """
        net, c = self.net, self.cfg
        cand = net.succ_table[e]
        lp = net.succ_logprior[e]          # transition weight (maybe unnormalised)
        dead = cand[:, 0] < 0
        with np.errstate(invalid="ignore"):
            lpn = lp - logsumexp(lp, axis=1, keepdims=True)
        p = np.nan_to_num(np.exp(lpn))      # normalised, for the defensive part
        comps = [h for h in h_preds if h is not None] if c.guided_turns else []
        if comps:
            guided = np.zeros_like(p)
            for h in comps:
                known = np.isfinite(h)
                lg = lp + np.where(known[:, None],
                                   c.turn_proposal_kappa * np.cos(
                                       np.nan_to_num(h)[:, None] - net.succ_head[e]),
                                   0.0)
                with np.errstate(invalid="ignore"):
                    lg = lg - logsumexp(lg, axis=1, keepdims=True)
                guided += np.nan_to_num(np.exp(lg))
            q = c.defensive_mix * p + (1.0 - c.defensive_mix) * guided / len(comps)
        else:
            q = p
        q = np.nan_to_num(q, nan=0.0)
        cq = np.cumsum(q, axis=1)
        u = self.rng.random(len(e))[:, None] * cq[:, -1:]
        choice = np.minimum((cq <= u).sum(axis=1), cand.shape[1] - 1)
        rows = np.arange(len(e))
        # guard against picking a padded slot (u at the very top)
        deg = (cand >= 0).sum(axis=1)
        choice = np.minimum(choice, np.maximum(deg - 1, 0))
        new_e = cand[rows, choice]
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = lp[rows, choice] - np.log(q[rows, choice] / np.maximum(cq[:, -1], 1e-300))
        corr = np.where(dead, -np.inf, corr)
        return np.where(dead, e, new_e), corr, dead

    def predict(self, step: float, heading: float, dt: float,
                heading_ahead: float = float("nan"), heading_var: float = 0.0) -> None:
        """Propagate every particle by one odometry step.

        ``heading_ahead`` (odometry heading ``proposal_lookahead_s`` later)
        only shapes the junction proposal; it never enters the weights except
        through the exact prior/proposal correction.
        """
        c, n = self.cfg, self.cfg.n_particles
        sq = math.sqrt(max(dt, 0.0))
        lo, hi = self._k_bounds
        m = c.logk_bounds_margin
        if dt > 0:
            self.logk = np.clip(self.logk + self.rng.normal(0.0, c.logk_rw * sq, n),
                                lo - m, hi + m)
            a = math.exp(-dt / c.speed_dev_tau_s)
            self.eps = a * self.eps + self.rng.normal(
                0.0, self._dev_sigma * math.sqrt(1.0 - a * a), n)
        if step <= 0.0:
            return
        if self.delta_ready and dt > 0:
            self.delta_var = (self.delta_var + np.deg2rad(c.delta_rw_deg) ** 2 * dt
                              + max(heading_var, 0.0))
        dist = np.exp(self.logk + self.eps) * step * np.exp(self.rng.normal(0.0, c.step_noise, n))
        self.offset = self.offset + dist
        h_now = (heading + self.delta) if (self.delta_ready and np.isfinite(heading)) else None
        h_ahead = ((heading_ahead + self.delta)
                   if (self.delta_ready and np.isfinite(heading_ahead)) else None)
        for _ in range(c.max_transitions):
            over = np.flatnonzero(self.offset > self.net.edge_len[self.edge])
            if len(over) == 0:
                break
            self.offset[over] -= self.net.edge_len[self.edge[over]]
            new_e, corr, dead = self._choose_successors(
                self.edge[over],
                [None if h_now is None else h_now[over],
                 None if h_ahead is None else h_ahead[over]])
            self.prev_head[over] = self.net.edge_head_end[self.edge[over]]
            self.edge[over] = new_e
            self.logw[over] += corr
            if dead.any():
                d = over[dead]
                self.offset[d] = self.net.edge_len[self.edge[d]]  # park at the end
        else:
            over = self.offset > self.net.edge_len[self.edge]
            self.logw[over] = -np.inf
            self.offset[over] = self.net.edge_len[self.edge[over]]
        self._relocate()

    # ---- measurements -------------------------------------------------------
    def update_heading(self, heading: float, dt: float) -> None:
        """Relative-heading likelihood with delta marginalised per particle.

        Observation model: road tangent = odometry heading + delta + v,
        v ~ (1-eps) N(0, R) + eps U(-pi, pi). Each particle is weighted by the
        predictive density of its innovation under N(0, P + R) (mixed with the
        outlier floor), then its delta is updated by a Kalman step whose gain
        is scaled by the inlier responsibility (a robust KF).
        """
        c = self.cfg
        if not np.isfinite(heading) or dt <= 0:
            return
        if not self.delta_ready:
            self._init_delta(heading)
            return
        inflate = max(1.0, c.heading_corr_time_s / dt) if c.heading_corr_time_s > 0 else 1.0
        R = np.deg2rad(c.heading_sigma_deg) ** 2 * inflate
        nu = self._heading_innovation(heading + self.delta)
        S = self.delta_var + R
        l_in = -0.5 * nu * nu / S - 0.5 * np.log(_TWO_PI * S)
        eps = c.heading_outlier
        if eps > 0:
            a = math.log1p(-eps) + l_in
            ll = np.logaddexp(a, math.log(eps / _TWO_PI))
            rho = np.exp(a - ll)
        else:
            ll, rho = l_in, np.ones_like(l_in)
        K = rho * self.delta_var / S
        self.delta = wrap_angle(self.delta + K * nu)
        self.delta_var = (1.0 - K) * self.delta_var
        self.logw = self.logw + ll

    def _heading_innovation(self, h_vehicle: np.ndarray) -> np.ndarray:
        """Signed angular distance from the road's expected heading to the
        predicted vehicle heading; inside a junction zone the expectation is the
        arc between incoming and outgoing headings (see ``junction_zone_m``)."""
        net, W = self.net, self.cfg.junction_zone_m
        nu = wrap_angle(self.road_heading - h_vehicle)
        if W <= 0:
            return nu
        best = np.abs(nu)
        # just after a node: arc from the previous edge's end heading
        after = np.flatnonzero((self.offset < W) & np.isfinite(self.prev_head))
        if len(after):
            d = _interval_residual(h_vehicle[after], self.prev_head[after],
                                   self.road_heading[after])
            take = np.abs(d) < best[after]
            nu[after[take]] = d[take]
            best[after[take]] = np.abs(d[take])
        # just before a node: arc towards any successor
        before = np.flatnonzero(net.edge_len[self.edge] - self.offset < W)
        if len(before):
            e = self.edge[before]
            hv = h_vehicle[before]
            for j in range(net.succ_table.shape[1]):
                ok = net.succ_table[e, j] >= 0
                if not ok.any():
                    continue
                d = _interval_residual(hv, self.road_heading[before], net.succ_head[e, j])
                take = ok & (np.abs(d) < best[before])
                nu[before[take]] = d[take]
                best[before[take]] = np.abs(d[take])
        return nu

    @staticmethod
    def _vm_mixture(r: np.ndarray, kappa: float, outlier: float) -> np.ndarray:
        lvm = kappa * (np.cos(r) - 1.0) + math.log(1.0 / (_TWO_PI * i0e(kappa)))
        if outlier <= 0:
            return lvm
        return np.logaddexp(math.log1p(-outlier) + lvm, math.log(outlier / _TWO_PI))

    def update_position(self, obs: PositionObservation) -> None:
        ox_, oy = obs.x - self.net.origin[0], obs.y - self.net.origin[1]
        d2 = (self.x - ox_) ** 2 + (self.y - oy) ** 2
        s2 = obs.sigma_m ** 2
        lg = -d2 / (2 * s2) - math.log(_TWO_PI * s2)
        if obs.outlier > 0:
            lg = np.logaddexp(math.log1p(-obs.outlier) + lg,
                              math.log(obs.outlier) + self._log_out_density)
        self.logw = self.logw + lg

    def update_abs_heading(self, obs: HeadingObservation) -> None:
        r = wrap_angle(obs.heading_rad - self.road_heading)
        self.logw = self.logw + self._vm_mixture(r, obs.kappa, obs.outlier)

    # ---- resampling ---------------------------------------------------------
    def normalise(self) -> tuple[np.ndarray, float]:
        lse = logsumexp(self.logw)
        if not np.isfinite(lse):
            # every hypothesis died (e.g. all driven off the map): restart
            self.n_reinit += 1
            self.initialise()
            self.delta_ready = False
            if self.store is not None:
                self.store.restart()
            lse = logsumexp(self.logw)
        self.logw = self.logw - lse
        w = np.exp(self.logw)
        ess = 1.0 / float(np.sum(w * w))
        return w, ess

    def resample(self, w: np.ndarray) -> np.ndarray:
        idx = _systematic_resample(w, self.rng)
        for name in ("edge", "offset", "logk", "eps", "delta", "delta_var", "x", "y",
                     "road_heading", "prev_head"):
            setattr(self, name, getattr(self, name)[idx])
        self.logw = np.full(len(idx), -math.log(len(idx)))
        if self.store is not None:
            self.store.reindex(idx)
        return idx

    # ---- summaries ----------------------------------------------------------
    _OFFS = np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)], np.int64)

    def _block_stats(self, w: np.ndarray, K: int = 64):
        """Bin particles on a ``mode_grid_m`` grid; for the K heaviest cells
        return the (K, 9) masses and weighted x/y sums of their 3x3 blocks."""
        g = self.cfg.mode_grid_m
        if self._dense_grid is not None:
            nx_, ny_ = self._dense_grid
            ix = (self.x / g).astype(np.int64) + 1      # +1: a one-cell border
            iy = (self.y / g).astype(np.int64) + 1
            flat = ix * ny_ + iy
            size = nx_ * ny_
            cm = np.bincount(flat, w, minlength=size)
            K = min(K, int(np.count_nonzero(cm)) or 1)
            top = np.argpartition(-cm, K - 1)[:K]
            nb = top[:, None] + self._OFFS[None, :, 0] * ny_ + self._OFFS[None, :, 1]
            nb = np.clip(nb, 0, size - 1)
            sx = np.bincount(flat, w * self.x, minlength=size)
            sy = np.bincount(flat, w * self.y, minlength=size)
            return cm[nb], sx[nb], sy[nb]
        cx = np.floor(self.x / g).astype(np.int64)
        cy = np.floor(self.y / g).astype(np.int64)
        key = (cx << 32) + (cy + (1 << 31))
        uk, inv = np.unique(key, return_inverse=True)
        cm = np.bincount(inv, w, minlength=len(uk))
        sx = np.bincount(inv, w * self.x, minlength=len(uk))
        sy = np.bincount(inv, w * self.y, minlength=len(uk))
        ucx = uk >> 32
        ucy = (uk & 0xFFFFFFFF) - (1 << 31)
        K = min(K, len(uk))
        top = np.argpartition(-cm, K - 1)[:K]
        offs = self._OFFS
        nk = ((ucx[top, None] + offs[None, :, 0]) << 32) + (ucy[top, None] + offs[None, :, 1] + (1 << 31))
        pos = np.minimum(np.searchsorted(uk, nk), len(uk) - 1)
        hit = uk[pos] == nk
        return (np.where(hit, cm[pos], 0.0), np.where(hit, sx[pos], 0.0),
                np.where(hit, sy[pos], 0.0))

    def summary(self, w: np.ndarray) -> dict:
        """Weighted posterior mode, its mass, and the number of separated modes."""
        c = self.cfg
        bm, bx, by = self._block_stats(w)
        block_mass = bm.sum(axis=1)
        order = np.argsort(-block_mass)
        best = order[0]
        mx = bx[best].sum() / max(block_mass[best], 1e-300)
        my = by[best].sum() / max(block_mass[best], 1e-300)
        r2 = c.confidence_radius_m ** 2
        near = (self.x - mx) ** 2 + (self.y - my) ** 2 <= r2
        if near.any() and w[near].sum() > 0:
            # one mean-shift refinement inside the confidence radius
            mx = float(np.sum(w[near] * self.x[near]) / w[near].sum())
            my = float(np.sum(w[near] * self.y[near]) / w[near].sum())
            near = (self.x - mx) ** 2 + (self.y - my) ** 2 <= r2
        conf = float(w[near].sum())
        kmode = float(np.sum(w[near] * np.exp(self.logk[near] + self.eps[near]))
                      / max(w[near].sum(), 1e-300)) \
            if near.any() else float("nan")
        # greedy non-maximum suppression for distinct modes
        peaks: list[tuple[float, float]] = []
        for j in order:
            if block_mass[j] < c.min_mode_mass:
                break
            px = bx[j].sum() / block_mass[j]
            py = by[j].sum() / block_mass[j]
            if all((px - qx) ** 2 + (py - qy) ** 2 > c.mode_sep_m ** 2 for qx, qy in peaks):
                peaks.append((px, py))
        return {"mode": (float(mx), float(my)), "confidence": conf,
                "n_modes": len(peaks), "k_mode": kmode}

    def mass_near(self, w: np.ndarray, x: float, y: float, r: float) -> float:
        lx, ly = x - self.net.origin[0], y - self.net.origin[1]
        return float(w[(self.x - lx) ** 2 + (self.y - ly) ** 2 <= r * r].sum())

    # ---- driver -------------------------------------------------------------
    def run(
        self,
        odom: Odometry,
        *,
        position_obs: Iterable[PositionObservation] = (),
        heading_obs: Iterable[HeadingObservation] = (),
        diagnostic_xy: np.ndarray | None = None,
        diagnostic_radius_m: float = 25.0,
    ) -> FilterResult:
        """Filter a whole odometry sequence and summarise every step.

        ``diagnostic_xy`` ((T+1, 2) map-CRS points, e.g. ground truth) is used
        *only* to record the posterior mass near it; it never enters the
        filter.
        """
        t0 = time.perf_counter()
        T = len(odom)
        pos_by = _group(position_obs)
        head_by = _group(heading_obs)
        self.initialise()
        out = {k: np.full(T + 1, np.nan) for k in ("conf", "ess", "k")}
        mode = np.full((T + 1, 2), np.nan)
        n_modes = np.zeros(T + 1, int)
        resampled = np.zeros(T + 1, bool)
        gt_mass = np.full(T + 1, np.nan) if diagnostic_xy is not None else None

        ahead = lookahead_headings(odom, self.cfg.proposal_lookahead_s)
        stride = 1
        if self.store is not None:
            stride = self.cfg.path_stride or max(1, math.ceil(
                self.cfg.n_particles * (T + 1) * 16.0 / (self.cfg.path_budget_mb * 1e6)))
        n_stored = 0
        for t in range(T + 1):
            if t > 0:
                st, hd, dt = float(odom.step[t - 1]), float(odom.heading[t - 1]), float(odom.dt[t - 1])
                hv = 0.0 if odom.heading_var is None else float(odom.heading_var[t - 1])
                self.predict(st, hd, dt, heading_ahead=float(ahead[t - 1]), heading_var=hv)
                if st > 0:
                    self.update_heading(hd, dt)
            for o in pos_by.get(t, ()):
                self.update_position(o)
            for o in head_by.get(t, ()):
                self.update_abs_heading(o)
            w, ess = self.normalise()
            if self.store is not None and (t % stride == 0 or t == T):
                self.store.add(t, self.x, self.y)
                n_stored += 1
                if n_stored % self.cfg.prune_every == 0:
                    self.store.prune()
            s = self.summary(w)
            mode[t] = s["mode"]
            out["conf"][t] = s["confidence"]
            out["k"][t] = s["k_mode"]
            n_modes[t] = s["n_modes"]
            out["ess"][t] = ess
            if gt_mass is not None and np.all(np.isfinite(diagnostic_xy[t])):
                gt_mass[t] = self.mass_near(w, diagnostic_xy[t, 0], diagnostic_xy[t, 1],
                                            diagnostic_radius_m)
            if ess < self.cfg.resample_ess_frac * self.cfg.n_particles:
                self.resample(w)
                resampled[t] = True

        w = np.exp(self.logw - logsumexp(self.logw))
        smoothed = None
        if self.store is not None:
            fs = self.summary(w)
            mx, my = fs["mode"]
            d2 = (self.x - mx) ** 2 + (self.y - my) ** 2
            inside = d2 <= self.cfg.confidence_radius_m ** 2
            cand = np.flatnonzero(inside) if inside.any() else np.arange(len(w))
            best = cand[np.argmax(w[cand])]
            smoothed = _fill_gaps(self.store.lineage(int(best), T + 1))
        mode_crs = self.net.local_to_crs(mode)
        sm_crs = None if smoothed is None else self.net.local_to_crs(smoothed)
        cfg = asdict(self.cfg)
        cfg["path_stride_used"] = stride if self.store is not None else None
        return FilterResult(
            times=odom.times, frame_index=np.concatenate([[0], odom.frame_index]),
            mode_xy=mode_crs, mode_latlon=self.net.local_to_latlon(mode),
            confidence=out["conf"], n_modes=n_modes, ess=out["ess"],
            resampled=resampled, k_mode=out["k"],
            smoothed_xy=sm_crs,
            smoothed_latlon=None if smoothed is None else self.net.local_to_latlon(smoothed),
            final_xy=self.net.local_to_crs(np.column_stack([self.x, self.y])),
            final_w=w, gt_mass=gt_mass, runtime_s=time.perf_counter() - t0,
            n_reinit=self.n_reinit, config=cfg, map_stats=self.net.stats(),
        )


def _fill_gaps(path: np.ndarray) -> np.ndarray:
    """Linearly interpolate the NaN rows of an (n, 2) path between its finite
    rows (a lineage stored every few steps); leading/trailing NaNs stay."""
    ok = np.isfinite(path).all(axis=1)
    if ok.all() or ok.sum() < 2:
        return path
    idx = np.arange(len(path))
    out = path.copy()
    for k in range(2):
        out[:, k] = np.interp(idx, idx[ok], path[ok, k], left=np.nan, right=np.nan)
    return out


def lookahead_headings(odom: Odometry, lookahead_s: float) -> np.ndarray:
    """Per step, the first known odometry heading at least ``lookahead_s``
    later (the last known one near the end; NaN if none). Feeds only the
    junction proposal, i.e. a fixed lag of ``lookahead_s`` when run online."""
    T = len(odom)
    out = np.full(T, np.nan)
    if T == 0 or lookahead_s <= 0:
        return out
    t_end = np.cumsum(odom.dt)
    known = np.flatnonzero(np.isfinite(odom.heading) & (odom.step > 0))
    if len(known) == 0:
        return out
    j = np.searchsorted(t_end[known], t_end + lookahead_s - 1e-9)
    j = np.minimum(j, len(known) - 1)
    return odom.heading[known[j]]


def _group(obs: Iterable) -> dict:
    out: dict = {}
    for o in obs:
        out.setdefault(int(o.step), []).append(o)
    return out


def run_filter(net: RoadNetwork, odom: Odometry, cfg: FilterConfig | None = None,
               **kwargs) -> FilterResult:
    """Convenience: build a filter matching ``odom``'s units and run it."""
    pf = GraphParticleFilter(net, cfg, metric=odom.metric,
                             unit_seconds=odom.unit_seconds or 0.1)
    return pf.run(odom, **kwargs)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _haversine_m(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la1, lo1 = np.radians(a[:, 0]), np.radians(a[:, 1])
    la2, lo2 = np.radians(b[:, 0]), np.radians(b[:, 1])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def evaluate_against_gt(
    result: FilterResult,
    gt_latlon: np.ndarray,
    *,
    loc_threshold_m: float = 10.0,
    stay_threshold_m: float = 20.0,
    mass_radius_m: float = 25.0,
) -> dict:
    """Score a filter run against per-estimate ground truth ((T+1, 2) lat/lon).

    time_to_localize_s: first time the mode error is < ``loc_threshold_m``
    and stays < ``stay_threshold_m`` for the rest of the run (None = never).
    """
    gt = np.asarray(gt_latlon, float)
    err = _haversine_m(result.mode_latlon, gt)
    t = result.times
    ok = err < loc_threshold_m
    stay = err < stay_threshold_m
    # suffix-all of `stay`
    stay_suffix = np.flip(np.cumprod(np.flip(stay)).astype(bool))
    good = np.flatnonzero(ok & stay_suffix)
    ttl = float(t[good[0]]) if len(good) else None
    out = {
        "mean_err_m": float(np.mean(err)),
        "median_err_m": float(np.median(err)),
        "final_err_m": float(err[-1]),
        "time_to_localize_s": ttl,
        "frac_steps_lt10m": float(np.mean(err < 10.0)),
        "frac_steps_lt25m": float(np.mean(err < 25.0)),
        "final_confidence": float(result.confidence[-1]),
        "final_n_modes": int(result.n_modes[-1]),
        "duration_s": float(t[-1]),
    }
    if ttl is not None:
        after = t >= ttl
        out["mean_err_after_localized_m"] = float(np.mean(err[after]))
    if result.smoothed_latlon is not None:
        se = _haversine_m(result.smoothed_latlon, gt)
        se = se[np.isfinite(se)]
        out["smoothed_mean_err_m"] = float(np.mean(se)) if len(se) else None
        out["smoothed_median_err_m"] = float(np.median(se)) if len(se) else None
    # posterior mass within r of GT at the end (from the final particle cloud)
    if result.final_xy is not None:
        tf = None
        try:
            from pyproj import Transformer
            crs = result.map_stats.get("crs")
            tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        except Exception:
            tf = None
        if tf is not None:
            gx, gy = tf.transform(gt[-1, 1], gt[-1, 0])
            d2 = (result.final_xy[:, 0] - gx) ** 2 + (result.final_xy[:, 1] - gy) ** 2
            out["final_mass_within_25m_of_gt"] = float(result.final_w[d2 <= mass_radius_m ** 2].sum())
    if result.gt_mass is not None:
        out["mean_gt_mass"] = float(np.nanmean(result.gt_mass))
    out["errors_m"] = err.tolist()
    return out
