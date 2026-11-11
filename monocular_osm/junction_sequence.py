"""Experimental OFFLINE road/OCR sequence importance sampler; no admission API.

All signposted junction alternatives are soft upstream-reachability factors, NOT
camera fixes or assertions that an exit is taken. Future evidence guides proposals
only; initial/turn importance ratios and auxiliary-resampling corrections are
explicit. Finite particles, a shortest-road-distance observation model, VO and
sign timing can all be wrong. Returned mass is not calibrated accuracy confidence.
No downloads, images, model execution, GT, policy mutation or file IO occur here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
import time

import numpy as np
from scipy.special import logsumexp

from .graph_filter import FilterConfig, GraphParticleFilter, Odometry, RoadNetwork
from .road_sign_clues import _EXIT, _ref
from .junction_inputs import CORRELATION_PROFILE, validate_correlations


PROFILE = {
    "method": "offline-all-junction-upstream-sequence-prototype-correlation-v2",
    "junction_prior": "legacy",
    "turn_target_semantics": "unit weight per legal successor; unnormalised outgoing mass equals degree",
    "correlation_profile": CORRELATION_PROFILE,
    "legacy_unannotated_factor_power": 1.,
    "max_particles": 50_000, "max_steps": 1200, "composition_hz": 2.0,
    "max_native_increments": 100_000,
    "max_cues": 8, "max_hypotheses": 64, "max_edges": 100_000,
    "max_segments": 500_000, "max_successor_entries": 1_000_000,
    "retained_segment_arrays_max_bytes": 24_000_000,
    "graph_preparation_separate_from_runtime": True,
    "max_guide_cells": 2_000_000, "max_initial_cells": 200_000,
    "genealogy_bytes": 1 << 30, "proposal_block_particles": 512,
    "proposal_scratch_bytes": 64 << 20, "inner_s": 1800., "outer_s": 1830.,
    "external_cpu_threads": 2, "external_rss_bytes": 3 << 30,
    "external_physical_and_commit_floor_bytes": 8 << 30,
    "advance_distance_scale_m": 750., "advance_time_scale_s": 45.,
    "sign_outlier": .15, "defensive_proposal": .25,
    "initial_cell_m": 100., "junction_node_coordinate_tolerance_m": 5.,
    "junction_binding": "exact retained OSM node ID; all incident directed endpoints; no nearest-road fallback",
    "max_attachments": 64,
    "speed_prior_median_mps": 8., "speed_prior_log_sigma": .75,
    "speed_prior_range_mps": [.5, 55.], "heading_sigma_deg": 15.,
    "speed_range_is_hard_cap": False,
    "speed_bound_semantics": "initial/base logk bounds plus logk margin; Gaussian OU eps and lognormal step noise are unbounded",
    "heading_outlier": .2, "heading_bias_rw_deg_sqrt_s": 2.,
    "unknown_extra_heading_rw_deg_sqrt_s": 5.,
    "cue_timing": "native read time retained; factor at next composed boundary with explicit lag",
    "cue_requirement": "road_sign_clues literal EXIT pairing; bare references forbidden",
    "turn_prior_kappa": 0., "normalise_turn_prior": False, "allow_uturn": False,
    "all_inputs_retained": True, "online": False, "accuracy_admitted": False,
}

PROFILE_V3 = {**PROFILE,
    "method": "offline-all-junction-upstream-sequence-prototype-conditional-uniform-v3",
    "junction_prior": "conditional-uniform-v3", "normalise_turn_prior": True,
    "turn_target_semantics": "proper conditional uniform over legal successors; not uniform over complete routes",
}


class Unavailable(ValueError):
    """Bound/eligibility failure, never a successful partial-window route."""


@dataclass(frozen=True)
class Config:
    # Lower values are useful for pure fixtures; actual profile must be declared.
    particles: int = 50_000
    seed: int = 0
    genealogy_bytes: int = 1 << 30
    inner_s: float = 1800.
    junction_prior: str = "legacy"

    def validate(self):
        if (type(self.particles) is not int or not 1 <= self.particles <= 50_000
                or type(self.seed) is not int or self.seed < 0
                or type(self.genealogy_bytes) is not int
                or not 1 <= self.genealogy_bytes <= PROFILE["genealogy_bytes"]
                or isinstance(self.inner_s, bool) or not math.isfinite(self.inner_s)
                or not 0 < self.inner_s <= 1800
                or type(self.junction_prior) is not str
                or self.junction_prior not in {"legacy", "conditional-uniform-v3"}):
            raise ValueError("invalid bounded junction-sequence configuration")


@dataclass
class Motion:
    times: np.ndarray
    step: np.ndarray
    heading: np.ndarray
    heading_var: np.ndarray
    unknown_seconds: np.ndarray
    stationary_seconds: np.ndarray
    native_end_indices: np.ndarray
    counts: np.ndarray  # measured, unknown, stationary native increments per bin
    se2: np.ndarray  # supplied-heading SE2 composition; NaN across unknown motion
    unit_seconds: float


def compose_se2(increments):
    """Compose body-frame transforms in acquisition order (no similarity fit)."""
    a = np.asarray(increments, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (3, 3) or not np.isfinite(a).all():
        raise ValueError("finite SE2 increments required")
    out = np.eye(3)
    for m in a:
        if (not np.allclose(m[2], [0, 0, 1], atol=1e-12, rtol=0)
                or not np.allclose(m[:2, :2].T @ m[:2, :2], np.eye(2), atol=1e-12)
                or not np.isclose(np.linalg.det(m[:2, :2]), 1., atol=1e-12)):
            raise ValueError("increment is not a proper SE2 transform")
        out = out @ m
    return out


def coarsen_motion(odom: Odometry, status, timestamps) -> Motion:
    """Full-window <=2Hz target composition, never a first-1200 truncation.

    ``status`` MUST independently distinguish measured / unknown / stationary;
    a held VO pose alone cannot establish a stop. Arc lengths sum; yaw rotations
    compose, rather than averaging angles. SE2 is the embedding of the supplied
    travel-heading/arc increments, NOT recovery of unprovided 3D translations.
    Unknown portions retain elapsed time and yield unknown composed SE2, not I.
    The final partial bin is retained even when shorter than half a second.
    """
    n = len(odom)
    if n > PROFILE["max_native_increments"]:
        raise Unavailable("complete native motion inventory exceeds fixed bound")
    ts = np.asarray(timestamps)
    st = np.asarray(status)
    if (odom.metric or odom.unit_seconds is None or not math.isfinite(odom.unit_seconds)
            or odom.unit_seconds <= 0 or n == 0 or ts.dtype != np.float64
            or ts.shape != (n + 1,) or not np.isfinite(ts).all()
            or not np.all(ts[1:] > ts[:-1]) or st.shape != (n,)
            or not np.isin(st, ["measured", "unknown", "stationary"]).all()
            or not np.allclose(np.diff(ts), odom.dt, atol=1e-9, rtol=1e-10)
            or not np.isfinite(odom.step).all() or not np.isfinite(odom.dt).all()
            or not np.all(odom.dt > 0) or np.isinf(odom.heading).any()
            or np.any((st == "measured") & (odom.step <= 0))
            or np.any((st == "stationary") & (odom.step != 0))):
        raise ValueError("explicit nonmetric full native clock/motion semantics required")
    hv = np.zeros(n) if odom.heading_var is None else np.asarray(odom.heading_var)
    if hv.shape != (n,) or not np.isfinite(hv).all() or np.any(hv < 0):
        raise ValueError("invalid heading uncertainty")
    ends, last = [], 0
    for i in range(1, n + 1):
        if ts[i] - ts[last] >= .5:
            ends.append(i)
            last = i
    if last != n:
        ends.append(n)
    if len(ends) > PROFILE["max_steps"]:
        raise Unavailable("complete native window exceeds 1200 composed steps")
    steps, heads, variances, unknown, stopped, counts, transforms = ([] for _ in range(7))
    previous_heading = 0.
    lo = 0
    for hi in ends:
        status_bin = st[lo:hi]
        measured = status_bin == "measured"
        unk = status_bin == "unknown"
        stop = status_bin == "stationary"
        steps.append(float(odom.step[lo:hi][measured].sum(dtype=np.float64)))
        u = float(odom.dt[lo:hi][unk].sum(dtype=np.float64))
        unknown.append(u)
        stopped.append(float(odom.dt[lo:hi][stop].sum(dtype=np.float64)))
        counts.append([int(measured.sum()), int(unk.sum()), int(stop.sum())])
        known_h = np.flatnonzero(~unk & np.isfinite(odom.heading[lo:hi]))
        heads.append(float(odom.heading[lo + known_h[-1]]) if len(known_h) else np.nan)
        variances.append(float(hv[lo:hi].sum()) + math.radians(5.) ** 2 * u)
        inc = []
        valid_se2 = not unk.any() and np.isfinite(odom.heading[lo:hi]).all()
        for j in range(lo, hi):
            h = float(odom.heading[j])
            if st[j] != "unknown" and math.isfinite(h):
                dh = h - previous_heading
                cs, sn = math.cos(dh), math.sin(dh)
                d = float(odom.step[j]) if st[j] == "measured" else 0.
                inc.append(np.array([[cs, -sn, d * cs], [sn, cs, d * sn], [0, 0, 1.]]))
                previous_heading = h
        transforms.append(compose_se2(inc) if valid_se2 else np.full((3, 3), np.nan))
        lo = hi
    return Motion(ts[np.r_[0, ends]].copy(), np.asarray(steps), np.asarray(heads),
                  np.asarray(variances), np.asarray(unknown), np.asarray(stopped),
                  np.asarray(ends, dtype=np.int64), np.asarray(counts),
                  np.asarray(transforms), float(odom.unit_seconds))


@dataclass
class Target:
    identity: str
    edges: np.ndarray
    offsets: np.ndarray
    tail_distance: np.ndarray

    def remaining(self, net, edge, offset):
        d = net.edge_len[edge] - offset + self.tail_distance[edge]
        for e, s in zip(self.edges, self.offsets):
            d = np.minimum(d, np.where((edge == e) & (offset <= s), s - offset, np.inf))
        return d


def _target(net, identity, xy, reverse, poll):
    # A geometric crossing can be a bridge. Only the actual OSM topology node
    # establishes junction reachability; the future map adapter must preserve
    # these nodes through simplification. Never replace/drop a missing target.
    try:
        node = int(identity.removeprefix("node/"))
    except ValueError:
        raise Unavailable("junction is not a retained OSM topology node") from None
    if identity != f"node/{node}" or node <= 0:
        raise Unavailable("junction is not a retained OSM topology node")
    attachments = {}
    for e, (u, v, _) in enumerate(net.edge_keys):
        if e % 4096 == 0:
            poll()
        if u == node:
            attachments[(e, 0.)] = None
        if v == node:
            attachments[(e, float(net.edge_len[e]))] = None
    if not attachments:
        raise Unavailable("junction missing from retained OSM topology; preserve it before graph simplification")
    if len(attachments) > PROFILE["max_attachments"]:
        raise Unavailable("junction attachment bound exceeded; no alternative dropped")
    ae = np.asarray([e for e, _ in attachments])
    ao = np.asarray([s for _, s in attachments])
    x, y, _, _ = net.locate(ae, ao)
    if np.any(np.hypot(x - xy[0], y - xy[1]) > PROFILE["junction_node_coordinate_tolerance_m"]):
        raise Unavailable("junction topology node coordinate disagrees with public inventory")
    start = np.full(net.n_edges, np.inf)
    heap = []
    for e, s in attachments:
        if s < start[e]:
            start[e] = s
            heapq.heappush(heap, (s, e))
    visits = 0
    while heap:
        d, e = heapq.heappop(heap)
        if d != start[e]:
            continue
        visits += 1
        if visits % 512 == 0:
            poll()
        for prev in reverse[e]:
            value = d + net.edge_len[prev]
            if value < start[prev]:
                start[prev] = value
                heapq.heappush(heap, (value, prev))
    tail = np.full(net.n_edges, np.inf)
    for k in range(net.succ_table.shape[1]):
        suc = net.succ_table[:, k]
        tail = np.minimum(tail, np.where(suc >= 0, start[np.maximum(suc, 0)], np.inf))
    pairs = sorted(attachments)
    return Target(identity, np.asarray([a for a, _ in pairs]), np.asarray([b for _, b in pairs]), tail)


@dataclass
class Cue:
    time: float
    step: int
    targets: tuple[Target, ...]
    observation: dict
    power: float = 1.
    group_id: str | None = None
    group_count: int = 1


def prepare_cues(net, matched, times, poll=lambda: None):
    """Consume road_sign_clues.match_motorway_junctions output; retain all rows."""
    rows = matched.get("matches")
    if (matched.get("status") != "complete" or matched.get("selected_answer") is not None
            or matched.get("camera_fix") is not None or not isinstance(rows, list) or not rows):
        raise Unavailable("complete all-hypothesis road-sign match inventory required")
    if len(rows) > 8 or sum(len(r["candidates"]) for r in rows) > 64:
        raise Unavailable("complete OCR hypothesis inventory exceeds fixed bounds")
    annotations = validate_correlations(matched)
    if net.n_edges * sum(len(r["candidates"]) for r in rows) > 2_000_000:
        raise Unavailable("road/junction distance table budget exceeded")
    reverse = [[] for _ in range(net.n_edges)]
    for e, row in enumerate(net.succ_table):
        for nxt in row[row >= 0]:
            reverse[int(nxt)].append(e)
    targets, descriptions, cues = {}, {}, []
    for row, annotation in zip(rows, annotations):
        obs = row["observation"]
        anchor = obs.get("anchor") or {}
        if (obs.get("pairing") not in {"inline_EXIT_reference", "same_frame_nearby_native_boxes"}
                or not isinstance(anchor.get("text"), str) or not _EXIT.search(anchor["text"])
                or not isinstance(obs.get("ref"), str)
                or _ref(obs.get("ref")) != obs.get("ref")):
            raise ValueError("literal EXIT pairing from road_sign_clues is required; bare refs are not anchors")
        stamp = obs.get("t_sec")
        if (isinstance(stamp, bool) or not isinstance(stamp, (int, float))
                or not math.isfinite(stamp) or not times[0] <= stamp <= times[-1]
                or row.get("camera_fix") is not None):
            raise ValueError("OCR times must belong to the complete native window")
        candidate_ids = []
        for candidate in row["candidates"]:
            identity = candidate["id"]
            ll = np.asarray([candidate["lat"], candidate["lon"]], dtype=np.float64)
            if (not isinstance(identity, str) or not identity or not np.isfinite(ll).all()
                    or abs(ll[0]) > 90 or abs(ll[1]) > 180 or identity in candidate_ids):
                raise ValueError("invalid or duplicated junction hypothesis")
            if identity in descriptions and not np.array_equal(ll, descriptions[identity]):
                raise ValueError("conflicting junction identity")
            if identity not in targets:
                descriptions[identity] = ll.copy()
                targets[identity] = _target(net, identity, net.latlon_to_local(ll)[0], reverse, poll)
            candidate_ids.append(identity)
        cues.append(Cue(float(stamp), int(np.searchsorted(times, stamp, side="left")),
                        tuple(targets[k] for k in sorted(candidate_ids)), dict(obs),
                        annotation["power"], annotation["group_id"], annotation["count"]))
    return sorted(cues, key=lambda c: (c.time, str(c.observation.get("ref", ""))))


def sign_factor(net, targets, edge, offset, speed, *, travel=0.):
    """Dimensionless event compatibility, NOT a calibrated position density.

    Exponential advance-distance and advance-time survival components allow
    signs before junctions; an outlier floor admits unrelated/missing exits.
    Directed reachability concerns the signposted target, not the taken route.
    """
    support = np.zeros(np.shape(edge), dtype=np.float64)
    speed = np.maximum(np.asarray(speed), .5)
    for target in targets:
        remaining = target.remaining(net, edge, offset) - travel
        valid = (remaining >= 0) & np.isfinite(remaining)
        d = np.where(valid, remaining, np.inf)
        support += .5 * np.exp(-d / 750.) + .5 * np.exp(-d / (speed * 45.))
    if targets:
        support /= len(targets)
    return .15 + .85 * support


def initial_proposal(lengths, guide, defensive=.25):
    """Piecewise road-length prior p, proposal q, exact p/q per cell."""
    lengths, guide = np.asarray(lengths), np.asarray(guide)
    if (lengths.ndim != 1 or lengths.shape != guide.shape or not len(lengths)
            or not np.isfinite(lengths).all() or not np.isfinite(guide).all()
            or np.any(lengths <= 0) or np.any(guide <= 0) or not 0 < defensive <= 1):
        raise ValueError("positive bounded proposal cells required")
    p = lengths / lengths.sum()
    tilted = p * guide
    q = defensive * p + (1 - defensive) * tilted / tilted.sum()
    return q, np.log(p) - np.log(q)


class _OfflineFilter(GraphParticleFilter):
    def __init__(self, net, cfg, motion, cues, poll):
        super().__init__(net, cfg, metric=False, unit_seconds=motion.unit_seconds)
        self.motion, self.cues, self.poll, self.index = motion, cues, poll, 0
        self.units = np.r_[0., np.cumsum(motion.step + motion.unknown_seconds / motion.unit_seconds)]

    def guide(self, edge, offset, logk):
        # Nearest future observation is a proposal heuristic; ALL observations
        # still enter the target factors at their own times, without a winner.
        future = next((c for c in self.cues if c.step > self.index and c.targets), None)
        if future is None:
            return np.ones(np.shape(edge))
        travel = np.exp(logk) * (self.units[future.step] - self.units[self.index])
        speed = np.exp(logk) / self.unit_seconds
        return sign_factor(self.net, future.targets, edge, offset, speed, travel=travel)

    def initialise_offline(self, appearance=None):
        net = self.net
        nc = np.ceil(net.edge_len / 100.).astype(np.int64)
        if int(nc.sum()) > 200_000:
            raise Unavailable("complete road-length proposal partition exceeds bound")
        edges = np.repeat(np.arange(net.n_edges), nc)
        starts = np.concatenate([np.arange(k) * (length / k) for k, length in zip(nc, net.edge_len)])
        lengths = np.repeat(net.edge_len / nc, nc)
        guide = self.guide(edges, starts + lengths / 2, np.full(len(edges), math.log(8. * self.unit_seconds)))
        if appearance is not None:
            guide = guide * appearance.initial_weight(net, edges, starts + lengths / 2)
        q, ratio = initial_proposal(lengths, guide)
        pick = self.rng.choice(len(q), size=self.cfg.n_particles, p=q)
        self.edge = edges[pick]
        self.offset = starts[pick] + self.rng.random(len(pick)) * lengths[pick]
        self.logk = self._sample_logk(len(pick))
        self.eps = self.rng.normal(0., self._dev_sigma, len(pick))
        self.logw = ratio[pick]
        self._relocate()

    def _choose_successors(self, e, h_preds):
        # Bound temporary candidate tensors independently of particle count.
        out, correction, dead = np.empty_like(e), np.empty(len(e)), np.empty(len(e), bool)
        for lo in range(0, len(e), 512):
            self.poll()
            sl = slice(lo, lo + 512)
            old = e[sl]
            cand = self.net.succ_table[old]
            lp = self.net.succ_logprior[old]
            no_exit = cand[:, 0] < 0
            safe = np.maximum(cand, 0)
            with np.errstate(invalid="ignore"):
                p = np.nan_to_num(np.exp(lp - logsumexp(lp, axis=1, keepdims=True)))
            # h_preds and e are subsets of crossing particles; no false mapping
            # to the full logk array. Proposal uses broad fixed speed; target
            # motion scale remains particle-specific in predict().
            guide = self.guide(safe, np.zeros_like(safe, dtype=float),
                               np.full(safe.shape, math.log(8. * self.unit_seconds)))
            logits = lp + np.log(guide)
            for h in h_preds:
                if h is not None:
                    hs = h[sl]
                    logits += np.where(np.isfinite(hs)[:, None],
                                       2. * np.cos(np.nan_to_num(hs)[:, None] - self.net.succ_head[old]), 0.)
            with np.errstate(invalid="ignore"):
                guided = np.nan_to_num(np.exp(logits - logsumexp(logits, axis=1, keepdims=True)))
            q = .25 * p + .75 * guided
            cdf = np.cumsum(q, axis=1)
            degree = (cand >= 0).sum(axis=1)
            u = self.rng.random(len(old))[:, None] * cdf[:, -1:]
            choice = np.minimum((cdf <= u).sum(axis=1), np.maximum(degree - 1, 0))
            rows = np.arange(len(old))
            chosen = cand[rows, choice]
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = lp[rows, choice] - np.log(q[rows, choice] / np.maximum(cdf[:, -1], 1e-300))
            out[sl] = np.where(no_exit, old, chosen)
            correction[sl] = np.where(no_exit, -np.inf, corr)
            dead[sl] = no_exit
        return out, correction, dead


class _NormalizedHeadingOfflineFilter(_OfflineFilter):
    """Normalized predictive kernel with the existing approximate bias KF.

    Nearest-arc signed innovation/ties and first-heading initialization remain.
    This is not exact marginalization or an exact posterior bias-moment update.
    """
    def update_heading(self, heading, dt):
        from . import junction_heading_likelihood as heading_core
        from .graph_filter import wrap_angle
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
        width = heading_core.arc_union_width(self.net, self.edge, self.offset,
                                            self.prev_head, c.junction_zone_m)
        l_in = heading_core.normalized_loglike(nu, S, width, 0.)
        eps = c.heading_outlier
        if eps > 0:
            a = math.log1p(-eps) + l_in
            ll = np.logaddexp(a, math.log(eps / (2 * math.pi)))
            rho = np.exp(a - ll)
        else:
            ll, rho = l_in, np.ones_like(l_in)
        # NEW normalized responsibility enters the retained robust-KF update.
        gain = rho * self.delta_var / S
        self.delta = wrap_angle(self.delta + gain * nu)
        self.delta_var = (1.0 - gain) * self.delta_var
        self.logw = self.logw + ll


class _SnapshotFutureGuideFilter(_OfflineFilter):
    """RAW guide only; existing initial/turn defenses remain in their callers."""
    def guide(self, edge, offset, logk):
        old_exit = super().guide(edge, offset, logk)
        return old_exit * self.future_guide.sample(self.index, self.net, edge, offset)


class _NormalizedHeadingSnapshotFutureGuideFilter(_SnapshotFutureGuideFilter, _NormalizedHeadingOfflineFilter):
    """Same proposal guide with the separately declared normalized target."""


def auxiliary_resample(pf, weights, guide):
    """Sample ancestors q_i∝w_i*g_i, then correct by 1/g_i (up to common Z)."""
    tilted = weights * guide
    if not np.isfinite(tilted).all() or np.any(guide <= 0) or tilted.sum() <= 0:
        raise ValueError("invalid future-cue auxiliary proposal")
    idx = pf.resample(tilted / tilted.sum())
    pf.logw = -np.log(guide[idx])
    pf.logw -= logsumexp(pf.logw)
    return idx


def align(net: RoadNetwork, odometry: Odometry, motion_status, native_times, matched,
          *, config: Config | None = None, check=lambda: None, clock=time.monotonic):
    """Original road/OCR-only entry; appearance fusion is explicitly separate."""
    return _align(net, odometry, motion_status, native_times, matched,
                  config=config, check=check, clock=clock)


@dataclass(frozen=True)
class CommonNullAppearance:
    """Sample the separately normalized visual log-ratio without an old wrapper."""
    field: object
    groups: np.ndarray
    powers: np.ndarray
    steps: np.ndarray
    receipt: dict

    def log_factor(self, query, net, local_xy, *, time):
        from .raw_vpr_appearance import need
        need(type(query) is int and 0 <= query < 80 and time == self.field.query_times[query],
             "common-null appearance must sample exact original query PTS")
        local = np.asarray(local_xy)
        need(local.ndim == 2 and local.shape[1] == 2 and np.isfinite(local).all(),
             "finite local PF positions required")
        return self.field.sample(net.local_to_crs(local)[:, None, :],
                                 np.array([time], np.float64))[:, 0]

    def initial_weight(self, net, edge, offset):
        x, y, _, _ = net.locate(edge, offset)
        log_factor = self.log_factor(0, net, np.column_stack([x, y]), time=self.field.query_times[0])
        return np.exp(self.powers[0] * log_factor)


def prepare_common_null_appearance(evidence, net, native_facts, native_times, composed_times, *, check=lambda: None):
    """Authenticate the unchanged snapshot/schedule contract, then build logpsi."""
    from . import common_null_appearance as null_core
    from . import reference_snapshot_appearance as snapshot_core
    strict = snapshot_core.strict
    check()
    providers = snapshot_core._validate(evidence)
    steps, _ = strict.verify_schedule(evidence, native_facts, native_times, composed_times)
    times = evidence["query_times"]
    groups, powers = strict.temporal_groups(times)
    refs = net.local_to_crs(net.latlon_to_local(evidence["ref_latlon"]))
    origin, shape, extent = strict.full_extent(net, refs)
    field = null_core.build_common_null_field(refs, evidence["raw_similarities"],
        evidence["snapshot_weights"], times, origin_xy=origin, shape=shape, check=check)
    strict.need(field.values.dtype == np.float32 and field.values.size <= 16_000_000
                and field.outside_value == 0., "complete common-null F32 log-ratio field required")
    receipt = {"profile": dict(null_core.PROFILE), "profile_sha256": null_core.PROFILE_SHA256,
        "visual_target": "common-null-v1", "field": field.source_record(), "crs": net.crs, **extent,
        "snapshot_count": len(evidence["snapshot_weights"]), "unique_provider_count": providers,
        "all_references": len(evidence["snapshot_weights"]), "all_physical_calls": evidence["physical_calls_validated"],
        "reference_snapshot_profile_sha256": evidence["profile_sha256"],
        "appearance_identity": evidence["appearance_identity"],
        "native_clock_sha256": native_facts["signed_vo_clock_sha256"],
        "snapshot_weights_sha256": strict.native.typed(evidence["snapshot_weights"]),
        "query_times": times.tolist(), "group_ids": groups.tolist(), "group_powers": powers.tolist(),
        "scheduled_steps": steps.tolist(), "all_queries_once": True,
        "direct_log_ratio_sampling": True, "old_row_max_or_outlier_wrapper": False,
        "outside_log_factor": 0., "model_and_reference_encoding_verified": False,
        "descriptor_equivalence_assumed": False, "historical_encoder_verified": False}
    check()
    return CommonNullAppearance(field, groups, powers, steps, receipt)


def align_with_snapshot_common_null_appearance(net: RoadNetwork, odometry: Odometry, motion_status,
                                              native_times, matched, evidence, native_facts, *,
                                              config: Config | None = None, check=lambda: None, clock=time.monotonic):
    """Explicit common-null visual target with unchanged legacy heading/EXIT guides."""
    config = config or Config(junction_prior="conditional-uniform-v3")
    config.validate()
    if config.junction_prior != "conditional-uniform-v3" or config.particles != 50_000 or config.seed != 0:
        raise ValueError("common-null requires conditional-uniform-v3 / 50k / seed0")
    return _align(net, odometry, motion_status, native_times, matched, config=config, check=check, clock=clock,
                  appearance_input=(evidence, native_facts), snapshot_appearance=True, common_null=True)


def align_with_snapshot_common_null_normalized_heading(net: RoadNetwork, odometry: Odometry, motion_status,
                                                      native_times, matched, evidence, native_facts, *,
                                                      config: Config | None = None, check=lambda: None, clock=time.monotonic):
    """Explicit common-null visual target with normalized heading/approximate bias KF."""
    config = config or Config(junction_prior="conditional-uniform-v3")
    config.validate()
    if config.junction_prior != "conditional-uniform-v3" or config.particles != 50_000 or config.seed != 0:
        raise ValueError("common-null requires conditional-uniform-v3 / 50k / seed0")
    return _align(net, odometry, motion_status, native_times, matched, config=config, check=check, clock=clock,
                  appearance_input=(evidence, native_facts), snapshot_appearance=True,
                  normalized_heading=True, common_null=True)


def align_with_appearance(net: RoadNetwork, odometry: Odometry, motion_status, native_times, matched,
                          bank, native_facts, *, config: Config | None = None,
                          check=lambda: None, clock=time.monotonic):
    """Off-default fixed raw-appearance research experiment; never an Answer."""
    config = config or Config(junction_prior="conditional-uniform-v3")
    config.validate()
    if config.junction_prior != "conditional-uniform-v3" or config.particles != 50_000 or config.seed != 0:
        raise ValueError("appearance experiment requires conditional-uniform-v3 / 50k / seed0")
    return _align(net, odometry, motion_status, native_times, matched, config=config,
                  check=check, clock=clock, appearance_input=(bank, native_facts))


def align_with_compatible_appearance(net: RoadNetwork, odometry: Odometry, motion_status, native_times, matched,
                                    bank, native_facts, *, config: Config | None = None,
                                    check=lambda: None, clock=time.monotonic):
    """Separate numeric-compatible data study; identical fixed fusion math."""
    config = config or Config(junction_prior="conditional-uniform-v3")
    config.validate()
    if config.junction_prior != "conditional-uniform-v3" or config.particles != 50_000 or config.seed != 0:
        raise ValueError("compatible appearance requires conditional-uniform-v3 / 50k / seed0")
    return _align(net, odometry, motion_status, native_times, matched, config=config,
                  check=check, clock=clock, appearance_input=(bank, native_facts), compatible_appearance=True)


def align_with_snapshot_appearance(net: RoadNetwork, odometry: Odometry, motion_status, native_times, matched,
                                   evidence, native_facts, *, config: Config | None = None,
                                   check=lambda: None, clock=time.monotonic):
    """Separate provider-normalized snapshot alternatives; never an Answer."""
    config = config or Config(junction_prior="conditional-uniform-v3")
    config.validate()
    if config.junction_prior != "conditional-uniform-v3" or config.particles != 50_000 or config.seed != 0:
        raise ValueError("snapshot appearance requires conditional-uniform-v3 / 50k / seed0")
    return _align(net, odometry, motion_status, native_times, matched, config=config,
                  check=check, clock=clock, appearance_input=(evidence, native_facts), snapshot_appearance=True)


def align_with_snapshot_normalized_heading(net: RoadNetwork, odometry: Odometry, motion_status,
                                           native_times, matched, evidence, native_facts, *,
                                           config: Config | None = None, check=lambda: None,
                                           clock=time.monotonic):
    """Explicit off-default target study; same snapshot inputs and fixed sampler."""
    config = config or Config(junction_prior="conditional-uniform-v3")
    config.validate()
    if config.junction_prior != "conditional-uniform-v3" or config.particles != 50_000 or config.seed != 0:
        raise ValueError("normalized heading requires conditional-uniform-v3 / 50k / seed0")
    return _align(net, odometry, motion_status, native_times, matched, config=config,
                  check=check, clock=clock, appearance_input=(evidence, native_facts),
                  snapshot_appearance=True, normalized_heading=True)


def align_with_snapshot_future_guide(net: RoadNetwork, odometry: Odometry, motion_status, native_times,
                                    matched, evidence, native_facts, *, config: Config | None = None,
                                    check=lambda: None, clock=time.monotonic):
    """Explicit future-proposal study with the unchanged legacy heading target."""
    config = config or Config(junction_prior="conditional-uniform-v3")
    config.validate()
    if config.junction_prior != "conditional-uniform-v3" or config.particles != 50_000 or config.seed != 0:
        raise ValueError("future guide requires conditional-uniform-v3 / 50k / seed0")
    return _align(net, odometry, motion_status, native_times, matched, config=config, check=check, clock=clock,
                  appearance_input=(evidence, native_facts), snapshot_appearance=True, future_proposal=True)


def align_with_snapshot_normalized_heading_future_guide(net: RoadNetwork, odometry: Odometry, motion_status,
                                                       native_times, matched, evidence, native_facts, *,
                                                       config: Config | None = None, check=lambda: None,
                                                       clock=time.monotonic):
    """Explicit future proposal plus normalized kernel/approximate bias KF."""
    config = config or Config(junction_prior="conditional-uniform-v3")
    config.validate()
    if config.junction_prior != "conditional-uniform-v3" or config.particles != 50_000 or config.seed != 0:
        raise ValueError("future guide requires conditional-uniform-v3 / 50k / seed0")
    return _align(net, odometry, motion_status, native_times, matched, config=config, check=check, clock=clock,
                  appearance_input=(evidence, native_facts), snapshot_appearance=True,
                  normalized_heading=True, future_proposal=True)


def align_with_snapshot_future_guide_no_adaptive_q(net: RoadNetwork, odometry: Odometry, motion_status,
                                                 native_times, matched, evidence, native_facts, *,
                                                 config: Config | None = None, check=lambda: None, clock=time.monotonic):
    """Explicit legacy-heading future proposal with zero native extra-Q only."""
    config = config or Config(junction_prior="conditional-uniform-v3")
    config.validate()
    if config.junction_prior != "conditional-uniform-v3" or config.particles != 50_000 or config.seed != 0:
        raise ValueError("no-adaptive-Q requires conditional-uniform-v3 / 50k / seed0")
    return _align(net, odometry, motion_status, native_times, matched, config=config, check=check, clock=clock,
                  appearance_input=(evidence, native_facts), snapshot_appearance=True,
                  future_proposal=True, no_adaptive_heading_q=True)


def align_with_snapshot_normalized_heading_future_guide_no_adaptive_q(net: RoadNetwork, odometry: Odometry,
                                                                   motion_status, native_times, matched,
                                                                   evidence, native_facts, *, config: Config | None = None,
                                                                   check=lambda: None, clock=time.monotonic):
    """Explicit normalized-heading future proposal with zero native extra-Q only."""
    config = config or Config(junction_prior="conditional-uniform-v3")
    config.validate()
    if config.junction_prior != "conditional-uniform-v3" or config.particles != 50_000 or config.seed != 0:
        raise ValueError("no-adaptive-Q requires conditional-uniform-v3 / 50k / seed0")
    return _align(net, odometry, motion_status, native_times, matched, config=config, check=check, clock=clock,
                  appearance_input=(evidence, native_facts), snapshot_appearance=True,
                  normalized_heading=True, future_proposal=True, no_adaptive_heading_q=True)


def align_with_snapshot_future_guide_adaptive_resampling(net: RoadNetwork, odometry: Odometry, motion_status,
                                                        native_times, matched, evidence, native_facts, *,
                                                        config: Config | None = None, check=lambda: None, clock=time.monotonic):
    """Explicit legacy-heading future target with target-ESS-triggered auxiliary resampling."""
    config = config or Config(junction_prior="conditional-uniform-v3")
    config.validate()
    if config.junction_prior != "conditional-uniform-v3" or config.particles != 50_000 or config.seed != 0:
        raise ValueError("adaptive resampling requires conditional-uniform-v3 / 50k / seed0")
    return _align(net, odometry, motion_status, native_times, matched, config=config, check=check, clock=clock,
                  appearance_input=(evidence, native_facts), snapshot_appearance=True,
                  future_proposal=True, adaptive_resampling=True)


def align_with_snapshot_normalized_heading_future_guide_adaptive_resampling(net: RoadNetwork, odometry: Odometry,
                                                                          motion_status, native_times, matched,
                                                                          evidence, native_facts, *, config: Config | None = None,
                                                                          check=lambda: None, clock=time.monotonic):
    """Explicit normalized-heading future target with target-ESS-triggered auxiliary resampling."""
    config = config or Config(junction_prior="conditional-uniform-v3")
    config.validate()
    if config.junction_prior != "conditional-uniform-v3" or config.particles != 50_000 or config.seed != 0:
        raise ValueError("adaptive resampling requires conditional-uniform-v3 / 50k / seed0")
    return _align(net, odometry, motion_status, native_times, matched, config=config, check=check, clock=clock,
                  appearance_input=(evidence, native_facts), snapshot_appearance=True,
                  normalized_heading=True, future_proposal=True, adaptive_resampling=True)


def _align(net: RoadNetwork, odometry: Odometry, motion_status, native_times, matched,
           *, config: Config | None = None, check=lambda: None, clock=time.monotonic,
           appearance_input=None, compatible_appearance=False, snapshot_appearance=False, normalized_heading=False, future_proposal=False,
           common_null=False, no_adaptive_heading_q=False, adaptive_resampling=False):
    """Return a diagnostic offline lineage or explicit unavailable; never select.

    Genealogy is EXACTLY 16*P*(T+1) payload bytes: int32 parent, int32 road
    edge, float64 offset. Worst admitted 50k*1201 =960,800,000 B (<1GiB).
    No thinning/interpolating of the composed path, automatic restart or MAP
    optimality claim. Root must externally enforce CPU/RSS/headroom/outer time.
    """
    config = config or Config()
    config.validate()
    if common_null and (appearance_input is None or not snapshot_appearance or compatible_appearance or future_proposal):
        raise ValueError("common-null requires separate snapshot inputs and EXIT-only proposals")
    if no_adaptive_heading_q and (appearance_input is None or not snapshot_appearance or not future_proposal
                                 or common_null or compatible_appearance):
        raise ValueError("no-adaptive-Q requires original snapshot visual target and explicit future proposal")
    if adaptive_resampling and (appearance_input is None or not snapshot_appearance or not future_proposal
                               or common_null or compatible_appearance or no_adaptive_heading_q):
        raise ValueError("adaptive resampling requires original run9 snapshot/future/native-Q target")
    profile = PROFILE_V3 if config.junction_prior == "conditional-uniform-v3" else PROFILE
    if appearance_input is not None:
        if common_null:
            from . import common_null_appearance as appearance_core
            prepare_appearance = prepare_common_null_appearance
        elif snapshot_appearance:
            from . import reference_snapshot_appearance as appearance_core
            prepare_appearance = appearance_core.prepare_snapshot
        elif compatible_appearance:
            from . import raw_vpr_compat_appearance as appearance_core
            prepare_appearance = appearance_core.prepare_compatible
        else:
            from . import raw_vpr_appearance as appearance_core
            prepare_appearance = appearance_core.prepare
        profile = {**profile, "method": ("offline-all-junction-reference-snapshot-appearance-v1" if snapshot_appearance else
                                         "offline-all-junction-compatible-raw-vpr-appearance-v1" if compatible_appearance else
                                         "offline-all-junction-raw-vpr-appearance-v1"),
                   "base_method": PROFILE_V3["method"], "appearance_profile": dict(appearance_core.PROFILE),
                   "appearance_profile_sha256": appearance_core.PROFILE_SHA256,
                   "appearance_initial_guide": "EXIT guide times first appearance factor to group power; exact initial p/q",
                   "turn_and_ancestor_guides": "unchanged EXIT-only guides with exact corrections"}
        if compatible_appearance:
            profile["compatibility_not_bitwise_identity"] = True
        if snapshot_appearance:
            profile["descriptor_equivalence_assumed"] = False
            profile["historical_encoder_verified"] = False
    if normalized_heading:
        if not snapshot_appearance:
            raise ValueError("normalized-heading target requires explicit snapshot entry")
        from . import junction_heading_likelihood as heading_core
        profile = {**profile, "method": "offline-all-junction-snapshot-normalized-heading-v1",
            "heading_mode": "normalized-junction-v1", "heading_profile": dict(heading_core.PROFILE),
            "heading_profile_sha256": heading_core.PROFILE_SHA256,
            "heading_predictive_kernel_normalized": True,
            "heading_bias_update": "nearest-branch robust scalar KF approximation with normalized responsibility",
            "exact_marginalized_arc_convolution": False, "exact_bias_posterior_moments": False}
    if common_null:
        profile = {**profile,
            "method": ("offline-snapshot-common-null-normalized-heading-v1" if normalized_heading else
                       "offline-snapshot-common-null-legacy-heading-v1"),
            "visual_target": "common-null-v1", "base_heading_target": "normalized-junction-v1" if normalized_heading else "legacy",
            "visual_target_changed": True, "direct_log_ratio_sampling": True,
            "old_row_max_or_outlier_wrapper": False, "future_proposal": False,
            "appearance_initial_guide": "EXIT guide times exp(beta0*logpsi0); exact initial p/q",
            "turn_and_ancestor_guides": "unchanged EXIT-only guides with exact corrections"}
    if future_proposal:
        if not snapshot_appearance:
            raise ValueError("future proposal requires explicit snapshot entry")
        from . import snapshot_future_guide as future_core
        profile = {**profile,
            "method": ("offline-snapshot-normalized-heading-future-proposal-v1" if normalized_heading else
                       "offline-snapshot-legacy-heading-future-proposal-v1"),
            "base_heading_target": "normalized-junction-v1" if normalized_heading else "legacy",
            "future_guide_profile": dict(future_core.PROFILE),
            "future_guide_profile_sha256": future_core.PROFILE_SHA256,
            "proposal_only": True, "appearance_target_unchanged": True,
            "appearance_initial_guide": "old EXIT times future times original G0 power; exact initial p/q",
            "turn_and_ancestor_guides": "RAW EXIT times future for turns; defensive effectiveg for auxiliary; exact corrections",
            "initial_and_turn_guide": "RAW old EXIT times future; existing .25 defenses unchanged; initial G0 retained",
            "auxiliary_guide": ".25+.75*raw/(w@raw) ONLY at auxiliary caller; exact1/effectiveg correction"}
    if no_adaptive_heading_q:
        from . import no_adaptive_heading_variance as variance_core
        profile = {**profile,
            "method": ("offline-snapshot-normalized-heading-future-no-adaptive-q-v1" if normalized_heading else
                       "offline-snapshot-legacy-heading-future-no-adaptive-q-v1"),
            "variance_mode": "no-adaptive-extra-Q-v1", "heading_variance_profile": dict(variance_core.PROFILE),
            "heading_variance_profile_sha256": variance_core.PROFILE_SHA256,
            "proposal_only": False, "future_guide_is_proposal_only": True,
            "native_heading_extra_Q_changed": True, "motion_means_status_clock_unchanged": True,
            "unknown_motion_variance_unchanged": True, "fixed_bias_random_walk_unchanged": True,
            "measured_covariance": False, "calibrated_covariance": False}
    if adaptive_resampling:
        from . import adaptive_auxiliary_resampling as resampling_core
        profile = {**profile,
            "method": ("offline-snapshot-normalized-heading-future-adaptive-resampling-v1" if normalized_heading else
                       "offline-snapshot-legacy-heading-future-adaptive-resampling-v1"),
            "resampling_mode": "target-ess-half-v1", "adaptive_resampling_profile": dict(resampling_core.PROFILE),
            "adaptive_resampling_profile_sha256": resampling_core.PROFILE_SHA256,
            "target_ESS_precedes_guide": True, "native_heading_variance_unchanged": True,
            "no_trigger": "identity ancestry; PF/logw/RNG unchanged; no auxiliary guide computation",
            "initial_ancestor_count_scope": "distinct initial particle labels in stored cloud BEFORE resampling-after-step",
            "resampling_diagnostics_extra_payload": "P int32 labels plus O(P) bincount scratch; two O(T) output arrays"}
    cfg = FilterConfig(n_particles=config.particles, seed=config.seed, path_storage=False,
                       turn_prior_kappa=0., normalise_turn_prior=profile["normalise_turn_prior"],
                       speed_median_mps=8., speed_log_sigma=.75, speed_range_mps=(.5, 55.),
                       heading_sigma_deg=15., heading_outlier=.2, delta_rw_deg=2.)
    started = clock()
    def poll():
        check()
        if clock() - started > config.inner_s:
            raise Unavailable("complete offline sequence exceeded inner time budget")
    base = {"method": profile["method"], "profile": dict(profile), "config": asdict(config),
            "effective_filter_config": asdict(cfg), "base_turn_profile_verified": False,
            "online": False, "accuracy_admitted": False, "selection": None,
            "confidence_calibrated": False, "proposal_corrections": "initial p/q; turn prior/q; auxiliary 1/g"}
    variance_arrays = {}
    try:
        poll()
        if no_adaptive_heading_q:
            effective_odometry = variance_core.without_adaptive_heading_variance(odometry)
            motion = coarsen_motion(effective_odometry, motion_status, native_times)
            variance_arrays = {"effective_native_heading_var": effective_odometry.heading_var.copy(),
                               "composed_heading_var": motion.heading_var.copy()}
            del effective_odometry
        else:
            motion = coarsen_motion(odometry, motion_status, native_times)
        t_count, p = len(motion.step), config.particles
        genealogy_bytes = 16 * p * (t_count + 1)
        if genealogy_bytes > config.genealogy_bytes:
            raise Unavailable("complete full-window genealogy cannot fit declared byte bound")
        if (net.n_edges > 100_000 or len(net.seg_gstart) > 500_000
                or np.count_nonzero(net.succ_table >= 0) > 1_000_000
                or net.succ_table.shape[1] > 32):
            raise Unavailable("complete road graph exceeds declared bounded prototype")
        if net.turn_prior_kappa != 0 or net.normalise_turn_prior or net.allow_uturn:
            raise ValueError("profile requires default unnormalised directed-road turn prior")
        valid_turn = net.succ_table >= 0
        if (not np.all(net.succ_logprior[valid_turn] == 0.)
                or not np.all(np.isneginf(net.succ_logprior[~valid_turn]))):
            raise ValueError("base compact graph no longer carries original unit legal-turn weights")
        base["base_turn_profile_verified"] = True
        if config.junction_prior == "conditional-uniform-v3":
            # New TARGET, not deletion of an importance correction. All geometry,
            # node IDs and successor topology are shared with original inputs.
            net = net.with_turn_prior(0., normalise=True)
        cues = prepare_cues(net, matched, motion.times, poll)
        appearance = None
        appearance_records = []
        if appearance_input is not None:
            try:
                appearance = prepare_appearance(appearance_input[0], net, appearance_input[1],
                    native_times, motion.times, check=poll)
            except ValueError as error:
                raise Unavailable("appearance inputs refused: " + str(error)) from error
            base["appearance"] = appearance.receipt
        future = None
        if future_proposal:
            try:
                future = future_core.build(appearance, motion.times, check=poll)
                base["future_guide"] = future.receipt
            except ValueError as error:
                raise Unavailable("future proposal refused: " + str(error)) from error
            filter_class = _NormalizedHeadingSnapshotFutureGuideFilter if normalized_heading else _SnapshotFutureGuideFilter
        else:
            filter_class = _NormalizedHeadingOfflineFilter if normalized_heading else _OfflineFilter
        pf = filter_class(net, cfg, motion, cues, poll)
        if future_proposal:
            pf.future_guide = future
        parent = np.empty((t_count + 1, p), dtype=np.int32)
        edges = np.empty_like(parent)
        offsets = np.empty((t_count + 1, p), dtype=np.float64)
        resampling_arrays = {}
        if adaptive_resampling:
            initial_labels = np.arange(p, dtype=np.int32)
            resampling_arrays = {"resampled_after_step": np.zeros(t_count + 1, dtype=np.bool_),
                                 "initial_ancestor_count": np.zeros(t_count + 1, dtype=np.int32)}
        if appearance is None:
            pf.initialise_offline()
        else:
            pf.initialise_offline(appearance)
        ancestry = np.full(p, -1, dtype=np.int32)
        ess, cue_records, modes, mode_counts, mode_mass, scale_modes = ([] for _ in range(6))
        for t in range(t_count + 1):
            poll()
            pf.index = t
            if t:
                step = motion.step[t - 1] + motion.unknown_seconds[t - 1] / motion.unit_seconds
                dt = float(motion.times[t] - motion.times[t - 1])
                pf.predict(float(step), float(motion.heading[t - 1]), dt,
                           heading_var=float(motion.heading_var[t - 1]))
                if step > 0:
                    pf.update_heading(float(motion.heading[t - 1]), dt)
            for cue in (c for c in cues if c.step == t):
                factor = sign_factor(net, cue.targets, pf.edge, pf.offset,
                                     np.exp(pf.logk + pf.eps) / motion.unit_seconds)
                pf.logw += cue.power * np.log(factor)
                cue_records.append({"time": cue.time, "applied_time": float(motion.times[t]),
                                    "composition_lag_s": float(motion.times[t] - cue.time),
                                    "correlation_group_id": cue.group_id, "group_count": cue.group_count,
                                    "factor_power": cue.power,
                                    "junction_ids": [j.identity for j in cue.targets],
                                    "no_matches_outlier_only": not bool(cue.targets),
                                    "factor_min": float(factor.min()), "factor_max": float(factor.max())})
            if appearance is not None:
                for query in np.flatnonzero(appearance.steps == t):
                    query = int(query)
                    stamp = appearance.field.query_times[query]
                    log_factor = appearance.log_factor(query, net, np.column_stack([pf.x, pf.y]), time=stamp)
                    pf.logw += appearance.powers[query] * log_factor
                    appearance_records.append({"query_row": query, "t_sec": float(stamp),
                        "applied_time": float(motion.times[t]), "composition_lag_s": float(motion.times[t] - stamp),
                        "group_id": int(appearance.groups[query]), "power": float(appearance.powers[query]),
                        "log_factor_min": float(log_factor.min()), "log_factor_max": float(log_factor.max())})
            norm = logsumexp(pf.logw)
            if not math.isfinite(norm):
                raise Unavailable("all full-window graph hypotheses died; no automatic restart")
            pf.logw -= norm
            weights = np.exp(pf.logw)
            ess.append(float(1 / np.sum(weights * weights)))
            summary = pf.summary(weights)
            modes.append(summary["mode"])
            mode_counts.append(summary["n_modes"])
            mode_mass.append(summary["confidence"])
            scale_modes.append(summary["k_mode"])
            edges[t], offsets[t], parent[t] = pf.edge, pf.offset, ancestry
            if adaptive_resampling:
                resampling_arrays["initial_ancestor_count"][t] = np.count_nonzero(np.bincount(initial_labels, minlength=p))
            if t < t_count:
                if adaptive_resampling and not resampling_core.decision(weights)["resample"]:
                    ancestry = np.arange(p, dtype=np.int32)
                else:
                    guide = pf.guide(pf.edge, pf.offset, pf.logk + pf.eps)
                    if future_proposal:
                        guide = future_core.defensive_auxiliary_guide(weights, guide)
                    ancestry = auxiliary_resample(pf, weights, guide).astype(np.int32)
                    if adaptive_resampling:
                        resampling_arrays["resampled_after_step"][t] = True
                if adaptive_resampling:
                    initial_labels = initial_labels[ancestry]
        # A representative lineage in the heaviest final spatial mode, not
        # a MAP path or an isolated high-importance-weight tail particle.
        mx, my = modes[-1]
        near = (pf.x - mx) ** 2 + (pf.y - my) ** 2 <= cfg.confidence_radius_m ** 2
        candidates = np.flatnonzero(near) if near.any() else np.arange(p)
        choice = int(candidates[np.argmax(pf.logw[candidates])])
        route_edge, route_offset = np.empty(t_count + 1, np.int32), np.empty(t_count + 1)
        for t in range(t_count, -1, -1):
            route_edge[t], route_offset[t] = edges[t, choice], offsets[t, choice]
            choice = int(parent[t, choice])
        x, y, heading, _ = net.locate(route_edge, route_offset)
        if appearance is not None and [row["query_row"] for row in appearance_records] != list(range(80)):
            raise Unavailable("appearance did not consume every original query exactly once")
        poll()
        return {**base, "status": "complete_diagnostic_only", "times": motion.times,
                **variance_arrays,
                **resampling_arrays,
                **({"appearance_events": appearance_records} if appearance is not None else {}),
                "route_latlon": net.local_to_latlon(np.column_stack([x, y])),
                "route_edge": route_edge, "route_offset_m": route_offset,
                "road_heading": heading, "mode_xy": net.local_to_crs(np.asarray(modes)),
                "ess": np.asarray(ess), "n_modes": np.asarray(mode_counts),
                "mode_mass_uncalibrated": np.asarray(mode_mass),
                "mode_metres_per_vo_unit": np.asarray(scale_modes), "cues": cue_records,
                "native_end_indices": motion.native_end_indices, "motion_counts": motion.counts,
                "unknown_seconds": motion.unknown_seconds, "stationary_seconds": motion.stationary_seconds,
                "composed_se2": motion.se2, "full_window_endpoints_preserved": True,
                "genealogy_payload_bytes": genealogy_bytes, "elapsed_s": clock() - started,
                "approximation": "finite-particle offline importance genealogy; no exact MAP or calibration"}
    except Unavailable as exc:
        return {**base, "status": "unavailable", "reason": str(exc), "route_latlon": None, **variance_arrays,
                "full_window_endpoints_preserved": False, "elapsed_s": clock() - started}
