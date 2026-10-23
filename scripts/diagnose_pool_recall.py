"""Is the true route even in the matcher's candidate pool? (uses ground truth)

The matcher enumerates walks on the OSM graph, keeps the top 500 by
bearing correlation, Procrustes-aligns those and returns the top-k. No
ranking channel can recover a route that never entered the pool, yet pool
recall was never measured. This script replays the matcher on a finished
run's cached graph and VO trajectory and reports, per stage, how close the
best walk comes to the ground-truth route — for one or more walk budgets
(--walks-per-node), with timings.

It is an offline diagnostic: ground truth is used here only, never fed to
the pipeline.

    python scripts/diagnose_pool_recall.py \\
        --result output/<slug>/result.json --gt ground_truth/<clip>.json \\
        --walks-per-node 40 120

Distances (metres, in the graph's projected CRS):
    cover  = mean distance from the densified GT route to the walk
    excess = mean distance from the densified walk to the GT route
    sym    = (cover + excess) / 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monocular_osm.evaluator import load_gt_waypoints  # noqa: E402
from monocular_osm.osm_data import fetch_city_graph  # noqa: E402
from monocular_osm.position import latlon_to_xy  # noqa: E402
from monocular_osm.trajectory_matching import match_trajectory  # noqa: E402

STAGES = ("enumerated", "prefiltered", "final")
THRESHOLDS_M = (50.0, 100.0, 250.0)


def _densified_points(xy: np.ndarray, spacing_m: float = 10.0):
    if len(xy) == 1:
        return shapely.points(xy)
    line = shapely.segmentize(shapely.LineString(xy), spacing_m)
    return shapely.points(shapely.get_coordinates(line))


class RecallObserver:
    """Accumulates, per stage, the best walk-to-GT distances seen."""

    def __init__(self, gt_xy: np.ndarray):
        self.gt_line = shapely.LineString(gt_xy)
        self.gt_pts = _densified_points(gt_xy)
        self.stats = {s: {"n": 0, "best_sym_m": np.inf, "best_cover_m": np.inf,
                          **{f"n_sym_le_{int(t)}m": 0 for t in THRESHOLDS_M}}
                      for s in STAGES}

    def __call__(self, stage: str, walk_xy: np.ndarray) -> None:
        walk_xy = np.asarray(walk_xy, dtype=np.float64)
        if len(walk_xy) < 2:
            return
        walk_line = shapely.LineString(walk_xy)
        cover = float(np.mean(shapely.distance(self.gt_pts, walk_line)))
        excess = float(np.mean(shapely.distance(_densified_points(walk_xy),
                                                self.gt_line)))
        sym = 0.5 * (cover + excess)
        st = self.stats[stage]
        st["n"] += 1
        st["best_sym_m"] = min(st["best_sym_m"], sym)
        st["best_cover_m"] = min(st["best_cover_m"], cover)
        for t in THRESHOLDS_M:
            if sym <= t:
                st[f"n_sym_le_{int(t)}m"] += 1

    def report(self) -> dict:
        return {s: {k: (round(v, 1) if isinstance(v, float) and np.isfinite(v)
                        else (None if isinstance(v, float) else v))
                    for k, v in st.items()}
                for s, st in self.stats.items()}


def _find_graph(data_dir: Path) -> Path:
    graphs = sorted(data_dir.glob("*.graphml"), key=lambda q: q.stat().st_mtime)
    if not graphs:
        raise SystemExit(f"no cached graph (*.graphml) in {data_dir}")
    return graphs[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--result", type=Path, required=True,
                    help="result.json of a finished run (provides data_dir, "
                         "VO cache and the length prior)")
    ap.add_argument("--gt", type=Path, required=True, help="ground-truth JSON")
    ap.add_argument("--graph", type=Path, default=None,
                    help="graphml to use (default: newest in the run's data_dir)")
    ap.add_argument("--walks-per-node", type=int, nargs="+", default=[40])
    ap.add_argument("--sample-every", type=int, default=None,
                    help="start-node subsampling (default: the run's config)")
    ap.add_argument("--out", type=Path, default=None, help="write JSON here")
    args = ap.parse_args(argv)

    result = json.loads(args.result.read_text(encoding="utf-8"))
    prov = result.get("provenance") or {}
    cfg = prov.get("config") or {}
    data_dir = Path(cfg.get("data_dir", args.result.parent))
    vo_cache = prov.get("vo_cache")
    if not vo_cache or not Path(vo_cache).exists():
        raise SystemExit("result.json has no usable provenance.vo_cache; "
                         "re-run the clip with the current pipeline")
    with np.load(vo_cache) as z:
        traj_xz = np.asarray(z["xz"], dtype=np.float64)
    length_m = float(result.get("estimated_length_m") or 1500.0)
    sample_every = args.sample_every or int(cfg.get("sample_every") or 1)

    graph_path = args.graph or _find_graph(data_dir)
    road = fetch_city_graph(cfg.get("city", ""), cache_path=graph_path)
    gt_xy = latlon_to_xy(load_gt_waypoints(args.gt), road.crs)

    out = {"result": str(args.result), "gt": str(args.gt), "graph": str(graph_path),
           "estimated_length_m": length_m, "sample_every": sample_every,
           "note": "full-scan replay (no OCR-anchor gating), free Procrustes scale",
           "budgets": {}}
    for wpn in args.walks_per_node:
        obs = RecallObserver(gt_xy)
        t0 = time.perf_counter()
        match_trajectory(traj_xz, road, final_top_k=int(cfg.get("top_k") or 10),
                         sample_every=sample_every, estimated_length_m=length_m,
                         walks_per_node=wpn, walk_observer=obs, progress=False)
        rep = obs.report()
        rep["seconds"] = round(time.perf_counter() - t0, 1)
        out["budgets"][str(wpn)] = rep
        print(f"walks_per_node={wpn}  ({rep['seconds']} s)")
        for s in STAGES:
            r = rep[s]
            print(f"  {s:12s} n={r['n']:>7}  best sym={r['best_sym_m']} m  "
                  f"best cover={r['best_cover_m']} m  "
                  f"<=100 m: {r['n_sym_le_100m']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
