"""Offline diagnostics for frozen POI hypotheses; never selects an answer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(path):
    """Validate *every* inference artifact before any scoring/GT access."""
    frozen = json.loads(Path(path).read_text(encoding="utf-8"))
    if frozen.get("ground_truth_used") is not False:
        raise ValueError("hypotheses must declare ground_truth_used=false")
    records = []
    seen = set()
    for artifact in frozen["artifacts"]:
        if artifact["id"] in seen:
            raise ValueError("duplicate frozen clip")
        seen.add(artifact["id"])
        source = Path(artifact["path"])
        if digest(source) != artifact["sha256"]:
            raise ValueError(f"{artifact['id']}: frozen artifact fingerprint changed")
        record = json.loads(source.read_text(encoding="utf-8"))
        if (record["id"] != artifact["id"] or record.get("selected_answer") is not None
                or record.get("script_sha256") != frozen["script_sha256"]):
            raise ValueError("artifact identity or inference provenance mismatch")
        records.append(record)
    return frozen, records


def interpolate_gt(waypoints, t_sec):
    """Geodesic interpolation at a video timestamp; refuse extrapolation."""
    import numpy as np
    from monocular_osm.geo import WGS84

    times = np.array([w["t_sec"] for w in waypoints], float)
    if (len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(t_sec) or t_sec < times[0] or t_sec > times[-1]):
        raise ValueError("observation must lie within finite, strictly ordered GT times")
    index = min(int(np.searchsorted(times, t_sec, side="right")) - 1, len(times) - 2)
    a, b = waypoints[index:index + 2]
    fraction = (t_sec - times[index]) / (times[index + 1] - times[index])
    bearing, _, length = WGS84.inv(a["lon"], a["lat"], b["lon"], b["lat"])
    lon, lat, _ = WGS84.fwd(a["lon"], a["lat"], bearing, length * fraction)
    return float(lat), float(lon)


def score_record(record, waypoints):
    from monocular_osm.geo import geodesic_distance_m
    from scripts.probe_region_plans import _route_samples, _coverage

    points, weights, times = _route_samples(waypoints, 20.0)
    candidates = []
    visibility_m = float(record["bounds"]["visibility_m"])
    for index, cluster in enumerate(record["candidate_clusters"]):
        witnesses = []
        for conjunction in cluster["conjunctions"]:
            pois = {p["id"]: p for p in conjunction["pois"]}
            for witness in conjunction["witnesses"]:
                observations = []
                for observation in witness["observations"]:
                    poi = pois[observation["poi_id"]]
                    camera = interpolate_gt(waypoints, observation["t_sec"])
                    distance = float(geodesic_distance_m(*camera, poi["lat"], poi["lon"]))
                    observations.append({"poi_id": poi["id"], "t_sec": observation["t_sec"],
                                         "camera_to_poi_m": distance,
                                         "within_frozen_visibility": distance <= visibility_m})
                witnesses.append({"observations": observations,
                                  "both_within_frozen_visibility": all(
                                      o["within_frozen_visibility"] for o in observations)})
        coverage = _coverage([{"center": cluster["centre_latlon"], "radius_m": 3000.0}],
                             points, weights, times, waypoints)
        candidates.append({"candidate_index": index, "poi_ids": cluster["poi_ids"],
                           "witnesses": witnesses, "three_km_route_coverage": coverage})
    return {"id": record["id"], "selected_answer": None, "candidates": candidates}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path,
                        default=ROOT / "output/poi_clue_probe/frozen_candidates.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "output/poi_clue_probe/offline_scores.json")
    args = parser.parse_args(argv)
    _, records = load_frozen(args.frozen)
    manifest = json.loads((args.root / "benchmarks/manifest.json").read_text(encoding="utf-8"))
    entries = {c["id"]: c for c in manifest["clips"]}
    results = []
    for record in records:
        entry = entries[record["id"]]
        gt_path = args.root / entry["gt"]
        waypoints = json.loads(gt_path.read_text(encoding="utf-8"))["waypoints"]
        rows = [[round(float(w["t_sec"]), 3), round(float(w["lat"]), 7),
                 round(float(w["lon"]), 7)] for w in waypoints]
        sha = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
        if sha != entry["gt_waypoints_sha256"]:
            raise ValueError(f"{record['id']}: registered GT fingerprint changed")
        results.append({**score_record(record, waypoints), "gt_file_sha256": digest(gt_path)})
    result = {"phase": "offline diagnostics after frozen inference", "selected_answer": None,
              "frozen_sha256": digest(args.frozen),
              "caveat": "All branches retained. Coverage and visibility compatibility do not establish localization accuracy. GT is interpolated along geodesic chords.",
              "clips": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for clip in results:
        print(json.dumps({"id": clip["id"], "candidates": [
            {"index": c["candidate_index"], "witness_distances_m": [
                [round(o["camera_to_poi_m"], 1) for o in w["observations"]]
                for w in c["witnesses"]],
             "route_coverage": c["three_km_route_coverage"]["arc_fraction_inside"]}
            for c in clip["candidates"]]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
