"""Conditional saved-pool ablation before extracting a reusable POI aligner.

The pool was proposed by the original appearance-assisted search. Removing
appearance here tests selection within that pool, not appearance-free recall.
All fixed predictions are saved by infer before the separate score phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import pilot_joint_alignment as joint

FORMULAS = {"full": "appearance - road - length - POI",
            "no_appearance": "-road - length - POI", "poi_only": "-POI"}


def infer(args):
    from pyproj import Transformer

    args.out.mkdir(parents=True, exist_ok=True)
    registration = {"status": "post-hoc conditional candidate-pool ablation",
                    "source": str(args.input.resolve()), "formulas": FORMULAS,
                    "selection_uses_gt": False,
                    "limitation": "source candidate proposals used appearance; does not measure appearance-free recall",
                    "tie_break": "hypothesis, x, y, yaw, length; independent of original appearance ranking",
                    "script_sha256": joint.sha256(__file__)}
    joint.write_json(args.out / "experiment.json", registration)
    for source in sorted(args.input.glob("*__distance_bearing.npz")):
        with np.load(source, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            fine, unit, times = z["fine_candidates"], z["unit_shape"], z["query_times"]
        if meta.get("ground_truth_used") is not False:
            raise ValueError("source predictions must declare GT-free inference")
        scores = {"full": fine[:, 10], "no_appearance": -fine[:, 6] - fine[:, 7] - fine[:, 8],
                  "poi_only": -fine[:, 8]}
        inverse = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True)
        for mode, objective in scores.items():
            order = np.lexsort((fine[:, 3], fine[:, 2], fine[:, 1], fine[:, 0], fine[:, 9], -objective))
            pick = int(order[0])
            xy = joint.transformed_routes(fine[pick:pick + 1], unit)[0]
            lon, lat = inverse.transform(xy[:, 0], xy[:, 1])
            label = f"{meta['variant']}__{mode}"
            metadata = {"variant": label, "source": str(source.resolve()),
                        "source_sha256": joint.sha256(source), "mode": mode,
                        "formula": FORMULAS[mode], "source_candidate_count": len(fine),
                        "selected_source_rank": pick + 1, "selected_hypothesis": int(fine[pick, 9]),
                        "selected_parameters": fine[pick].tolist(), "window": meta["window"],
                        "crs": meta["crs"], "ground_truth_used": False}
            np.savez_compressed(args.out / f"{label}.npz", route_latlon=np.c_[lat, lon], query_times=times,
                                objective=objective, ranking=order, metadata=np.array(json.dumps(metadata)))
            print(f"saved {label}, branch {metadata['selected_hypothesis']}, source rank {pick + 1}")


def score(args):
    from monocular_osm.metrics import score_route
    from pyproj import Transformer

    truth = json.loads(args.gt.read_text(encoding="utf-8"))["waypoints"]
    results = {}
    for source in sorted(args.out.glob("*.npz")):
        with np.load(source, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            route, times = z["route_latlon"], z["query_times"]
        fixes = [p for p in truth if meta["window"][0] <= p["t_sec"] <= meta["window"][1]]
        ll = np.array([[p["lat"], p["lon"]] for p in fixes])
        tt = np.array([p["t_sec"] for p in fixes])
        at_query = np.c_[np.interp(times, tt, ll[:, 0]), np.interp(times, tt, ll[:, 1])]
        tf = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
        prediction_xy = np.column_stack(tf.transform(route[:, 1], route[:, 0]))
        gt_xy = np.column_stack(tf.transform(at_query[:, 1], at_query[:, 0]))
        metrics = score_route(route, ll)
        results[meta["variant"]] = {"symmetric_mean_m": metrics.symmetric_mean_m,
                                    "start_error_m": metrics.start_error_m,
                                    "mean_time_error_m": float(np.linalg.norm(prediction_xy - gt_xy, axis=1).mean()),
                                    "selected_hypothesis": meta["selected_hypothesis"],
                                    "selected_source_rank": meta["selected_source_rank"]}
    joint.write_json(args.out / "scores.json", results)
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inf = sub.add_parser("infer")
    inf.add_argument("--input", type=Path, default=ROOT / "output/joint_poi_alignment_pilot")
    inf.add_argument("--out", type=Path, default=ROOT / "output/poi_alignment_ablation")
    scr = sub.add_parser("score")
    scr.add_argument("--out", type=Path, default=ROOT / "output/poi_alignment_ablation")
    scr.add_argument("--gt", type=Path, default=ROOT / "ground_truth/kitti_drive_0009.json")
    args = parser.parse_args()
    (infer if args.command == "infer" else score)(args)


if __name__ == "__main__":
    main()
