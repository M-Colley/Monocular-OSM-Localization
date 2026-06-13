"""Offline benchmark for the trajectory-matching cost function.

Replays enumeration + scoring + GT evaluation from the *cached* VO
trajectory and OSM graph of each validation clip — no VO, no download,
so iterating on the matching cost is seconds, not minutes. For each clip
it reports, per cost variant, the GT mean-route-error of the #1 pick,
plus the best-achievable error in the pool and the rank the variant
gives that best candidate. Lower pick-error = better selection.

    python scripts/bench_matching.py            # all variants, all clips
    python scripts/bench_matching.py --variant turn_weighted
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluator import evaluate_candidates_against_waypoints, load_gt_waypoints
from src.osm_data import fetch_city_graph
from src.trajectory_matching import MatchCandidate, match_trajectory
from src.visual_odometry import trajectory_arc_length

DATA = Path("data")


@dataclass
class Clip:
    name: str
    npz: str
    graphml: str
    gt: str
    est_len_m: float          # the prior the production run used
    scale_lock: bool = True


CLIPS = [
    Clip("ulm",
         "ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/trajectory_v2_0-420.0_s3_fauto.npz",
         "ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/Ulm_Germany.graphml",
         "ground_truth/ulm_ULl8s4qydrk.json", 2310.0),
    Clip("kitti_0033",
         "local-36a50c34107a-drive-0033-karlsruhe-germany/trajectory_v2_0-166.0_s3_fauto.npz",
         "local-36a50c34107a-drive-0033-karlsruhe-germany/Karlsruhe_Germany_around_48.9702_8.4788_968.graphml",
         "ground_truth/kitti_drive_0033.json", 876.0),
    Clip("comma_148",
         "local-88d9fe89bc4d-route-148-san-francisco-california-usa/trajectory_v2_0-240.0_s3_fauto.npz",
         "local-88d9fe89bc4d-route-148-san-francisco-california-usa/San_Francisco_California_USA_around_37.6725_-122.4656_1272.graphml",
         "ground_truth/comma_148.json", 1319.0),
]


# --- extra per-candidate features computed from the aligned geometry -------


def _heading_series(xy: np.ndarray) -> np.ndarray:
    d = np.diff(xy, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def _turn_weighted_corr(traj_xy: np.ndarray, walk_xy: np.ndarray, n: int = 128) -> float:
    """Bearing-signature correlation, but weighting each sample by how
    much the *trajectory* is turning there. Straights are ubiquitous and
    non-discriminative; the turns are the signature. Weighting the
    Pearson correlation by |curvature| makes matching the turn pattern
    dominate the score."""
    from src.visual_odometry import resample_uniform
    a = resample_uniform(traj_xy, n)
    b = resample_uniform(walk_xy, n)
    ha = np.unwrap(_heading_series(a))
    hb = np.unwrap(_heading_series(b))
    # turn magnitude of the trajectory at each step
    w = np.abs(np.diff(ha, prepend=ha[0]))
    w = w + 1e-3
    ha = ha - np.average(ha, weights=w)
    hb = hb - np.average(hb, weights=w)
    num = np.sum(w * ha * hb)
    den = np.sqrt(np.sum(w * ha * ha) * np.sum(w * hb * hb)) + 1e-12
    return float(np.clip(num / den, -1.0, 1.0))


def _endpoint_consistency(traj_xy: np.ndarray, walk_xy: np.ndarray) -> float:
    """How well the start→end displacement direction & magnitude (as a
    fraction of arc length) agree. A loop (end≈start) only matches walks
    that also close; a straight only matches straights. Scale-free."""
    def feat(xy):
        arc = float(trajectory_arc_length(xy)[-1]) + 1e-9
        disp = xy[-1] - xy[0]
        return np.linalg.norm(disp) / arc, np.arctan2(disp[1], disp[0])
    ra, aa = feat(traj_xy)
    rb, ab = feat(walk_xy)
    dang = abs((aa - ab + np.pi) % (2 * np.pi) - np.pi)
    return float(abs(ra - rb) + 0.3 * dang)   # lower = more consistent


# --- cost variants: map a candidate (+features) to a sort key (lower=better)


def make_cost(variant: str, pool: list[MatchCandidate]):
    rms = np.array([c.score for c in pool])
    corr = np.array([c.bearing_corr for c in pool])
    rms_spread = float(np.std(rms)) + 1e-6
    if variant == "baseline":
        return rms - 400.0 * corr
    if variant == "turn_weighted":
        tw = np.array([_turn_weighted_corr(c.aligned_traj_xy, c.walk_xy) for c in pool])
        return rms - 400.0 * corr - 400.0 * tw
    if variant == "endpoint":
        ep = np.array([_endpoint_consistency(c.aligned_traj_xy, c.walk_xy) for c in pool])
        return rms - 400.0 * corr + 800.0 * ep
    if variant == "turn+endpoint":
        tw = np.array([_turn_weighted_corr(c.aligned_traj_xy, c.walk_xy) for c in pool])
        ep = np.array([_endpoint_consistency(c.aligned_traj_xy, c.walk_xy) for c in pool])
        return rms - 400.0 * corr - 300.0 * tw + 600.0 * ep
    if variant == "consensus_rank":
        # Rank-fuse independent geometric sub-scores. If the true route is
        # "consistently decent" on all while each look-alike is great on
        # only one, summed ranks should lift it.
        tw = np.array([_turn_weighted_corr(c.aligned_traj_xy, c.walk_xy) for c in pool])
        ep = np.array([_endpoint_consistency(c.aligned_traj_xy, c.walk_xy) for c in pool])
        def ranks(x, ascending=True):
            order = np.argsort(x if ascending else -x)
            r = np.empty(len(x), int); r[order] = np.arange(len(x))
            return r
        return ranks(rms) + ranks(-corr) + ranks(-tw) + ranks(ep)
    raise SystemExit(f"unknown variant {variant}")


VARIANTS = ["baseline", "turn_weighted", "endpoint", "turn+endpoint", "consensus_rank"]


args_diag = False
args_hyp = False


def run_clip(clip: Clip, variants: list[str]) -> None:
    xz = np.load(DATA / clip.npz)["xz"]
    road = fetch_city_graph(clip.name, cache_path=DATA / clip.graphml)
    locked = None
    if clip.scale_lock:
        arc = float(trajectory_arc_length(xz)[-1])
        locked = clip.est_len_m / arc if arc > 1e-6 else None
    pool = match_trajectory(
        xz, road, final_top_k=400, sample_every=1,
        estimated_length_m=clip.est_len_m, locked_scale=locked, progress=False)
    wp = load_gt_waypoints(Path(clip.gt))
    evals = evaluate_candidates_against_waypoints(pool, road, wp)
    gt_err = np.array([e.mean_route_error_m for e in evals])
    best_i = int(np.argmin(gt_err))
    rms = np.array([c.score for c in pool])
    corr = np.array([c.bearing_corr for c in pool])
    print(f"\n=== {clip.name}  (pool {len(pool)}, locked_scale "
          f"{locked:.3f})  best-in-pool {gt_err[best_i]:.0f} m ===")
    if args_diag:
        # Is shape a good proxy for GT error at all?
        rrms = np.corrcoef(rms, gt_err)[0, 1]
        rcomb = np.corrcoef(rms - 400 * corr, gt_err)[0, 1]
        rms_rank = int(np.where(np.argsort(rms) == best_i)[0][0]) + 1
        corr_rank = int(np.where(np.argsort(-corr) == best_i)[0][0]) + 1
        print(f"  [diag] pool RMS range {rms.min():.0f}-{rms.max():.0f} m; "
              f"corr(RMS,GTerr)={rrms:+.2f}  corr(combined,GTerr)={rcomb:+.2f}")
        print(f"  [diag] GT-best (#{best_i}): GTerr {gt_err[best_i]:.0f}m  "
              f"RMS {rms[best_i]:.0f}m (rank {rms_rank}/{len(pool)})  "
              f"corr {corr[best_i]:+.2f} (rank {corr_rank})")
        # how good is the *best RMS* candidate, geographically?
        print(f"  [diag] lowest-RMS cand: GTerr {gt_err[int(np.argmin(rms))]:.0f}m; "
              f"highest-corr cand: GTerr {gt_err[int(np.argmax(corr))]:.0f}m")
    if args_hyp:
        from src.hypotheses import (cluster_candidates, distinct_hypotheses,
                                     hypothesis_confidence)
        hyps = distinct_hypotheses(pool, road, top_n=8)
        conf = hypothesis_confidence(pool, hyps)
        clusters = cluster_candidates(pool)
        # distinct-hypothesis rank of the cluster containing the GT-best candidate
        best_hyp_rank = next((r for r, cl in enumerate(clusters, 1) if best_i in cl), -1)
        in_top5 = best_hyp_rank != -1 and best_hyp_rank <= 5
        print(f"  [hyp] {len(clusters)} distinct places; GT-best's place ranks "
              f"#{best_hyp_rank} -> in top-5? {in_top5}")
        print(f"  [hyp] confidence={conf['level']} "
              f"(concentration {conf['concentration']}, spread {conf['spread_m']} m, "
              f"support {conf['support']})")
        for h in hyps[:5]:
            print(f"        hyp#{h.rank} GTerr {gt_err[h.candidate_index]:5.0f} m  "
                  f"support {h.support:3d}  {', '.join(h.street_names) or '(unnamed)'}")
    for v in variants:
        cost = make_cost(v, pool)
        order = np.argsort(cost)
        pick = order[0]
        rank_of_best = int(np.where(order == best_i)[0][0]) + 1
        print(f"  {v:16s} pick #1 -> {gt_err[pick]:6.0f} m   "
              f"(GT-best ranked #{rank_of_best})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default=None, help="run one variant (default: all)")
    p.add_argument("--clip", default=None, help="run one clip by name")
    p.add_argument("--diag", action="store_true", help="print cost-vs-GT diagnostics")
    p.add_argument("--hyp", action="store_true", help="print top-N hypothesis report")
    args = p.parse_args()
    global args_diag, args_hyp
    args_diag = args.diag
    args_hyp = args.hyp
    variants = [args.variant] if args.variant else VARIANTS
    clips = [c for c in CLIPS if (not args.clip or c.name == args.clip)]
    for clip in clips:
        run_clip(clip, variants)


if __name__ == "__main__":
    main()
