"""Stage 10/10b: optional in-run evaluation against ground truth."""

from __future__ import annotations

from ..evaluator import (
    best_rank_for_gt,
    best_rank_for_waypoints,
    evaluate_candidates,
    evaluate_candidates_against_waypoints,
    load_gt_waypoints,
)
from .context import RunContext


def evaluate_ground_truth_streets(ctx: RunContext) -> None:
    """Optional evaluation of the candidates against ground-truth street names."""
    cfg = ctx.cfg
    result = ctx.result
    road = ctx.road
    candidates = ctx.candidates

    # 10. Optional: ground-truth evaluation.
    if cfg.ground_truth_streets and candidates:
        gt = list(cfg.ground_truth_streets)
        print(f"[10] Evaluating against ground-truth streets: {gt}")
        gt_results = evaluate_candidates(candidates, road, gt)
        best_dist_rank, name_rank = best_rank_for_gt(gt_results)
        print(f"      -> closest-to-GT candidate is shape #{best_dist_rank}; "
              f"first GT-named candidate is rank {name_rank}")
        for i, r in enumerate(gt_results):
            tag = f"  ON-GT({', '.join(r.matching_gt_names)})" if r.on_gt_street else ""
            print(f"        #{i+1}  d_to_GT={r.nearest_distance_m:7.1f} m{tag}")
            result["matches"][i]["gt_distance_m"] = r.nearest_distance_m
            result["matches"][i]["gt_on_street"] = r.on_gt_street
            result["matches"][i]["gt_matching_names"] = r.matching_gt_names
        result["ground_truth"] = {
            "streets": gt,
            "best_distance_rank": best_dist_rank,
            "first_named_rank": name_rank,
        }


def evaluate_waypoints(ctx: RunContext) -> None:
    """Optional metric evaluation against GPS waypoint ground truth."""
    cfg = ctx.cfg
    result = ctx.result
    road = ctx.road
    candidates = ctx.candidates
    anchored_cand = ctx.anchored_cand

    # 10b. Optional: metric evaluation against GPS waypoint ground truth.
    ctx.waypoint_evals = None
    ctx.anchored_eval = None
    if cfg.ground_truth_waypoints and candidates:
        print(f"[10b] Evaluating against GT waypoints: {cfg.ground_truth_waypoints}")
        try:
            waypoints = load_gt_waypoints(cfg.ground_truth_waypoints)
            ctx.waypoint_evals = evaluate_candidates_against_waypoints(
                candidates, road, waypoints
            )
        except (OSError, ValueError) as e:
            print(f"      -> waypoint evaluation failed: {e}")
            result["ground_truth_waypoints_error"] = str(e)
        if ctx.waypoint_evals:
            wp_best_rank = best_rank_for_waypoints(ctx.waypoint_evals)
            print(f"      -> best candidate by mean route error: #{wp_best_rank}")
            for i, ev in enumerate(ctx.waypoint_evals):
                m = result["matches"][i]
                m["gt_waypoint_start_error_m"] = round(ev.start_error_m, 1)
                m["gt_waypoint_mean_route_error_m"] = round(ev.mean_route_error_m, 1)
                m["gt_waypoint_max_route_error_m"] = round(ev.max_route_error_m, 1)
                print(f"        #{i+1}  start={ev.start_error_m:7.1f} m  "
                      f"mean={ev.mean_route_error_m:7.1f} m  "
                      f"max={ev.max_route_error_m:7.1f} m")
            result["ground_truth_waypoints"] = {
                "file": str(cfg.ground_truth_waypoints),
                "n_waypoints": int(len(waypoints)),
                "best_mean_error_rank": wp_best_rank,
                "final_pick_start_error_m": round(ctx.waypoint_evals[0].start_error_m, 1),
                "final_pick_mean_route_error_m": round(
                    ctx.waypoint_evals[0].mean_route_error_m, 1
                ),
            }
            if anchored_cand is not None:
                aev = evaluate_candidates_against_waypoints(
                    [anchored_cand], road, waypoints)[0]
                ctx.anchored_eval = aev
                result["anchored_position"]["gt_start_error_m"] = round(
                    aev.start_error_m, 1)
                result["anchored_position"]["gt_mean_route_error_m"] = round(
                    aev.mean_route_error_m, 1)
                print(f"      -> ANCHOR-PRIMARY (headline) vs GT: start "
                      f"{aev.start_error_m:.1f} m, mean route "
                      f"{aev.mean_route_error_m:.1f} m   (matcher pick: start "
                      f"{ctx.waypoint_evals[0].start_error_m:.1f} m, mean route "
                      f"{ctx.waypoint_evals[0].mean_route_error_m:.1f} m)")
