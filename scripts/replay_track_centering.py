"""Offline replay of the retrieval track decoder on saved similarity matrices:
raw Viterbi (the pipeline's) vs reference-mean-centred Viterbi, per pass.

Pass 2 (fixed pool): symmetric error of the smoothed track route.
Pass 1: distance from the robust centre to the GT route (which area pass 1
would hand to pass 2). Dev clips, in-sample; GT used for scoring only.

    python scripts/replay_track_centering.py <run_dir> <out.json>

<run_dir> is an assisted benchmark run whose baseline children saved
``retrieval_audit/*.npz`` (full similarity matrices).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import psutil

psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from monocular_osm.geo import geodesic_distance_m  # noqa: E402
from monocular_osm.kartaview_vpr import _robust_center, _viterbi_decode  # noqa: E402
from monocular_osm.metrics import score_route  # noqa: E402

RUN = sys.argv[1]
man = {c["id"]: c for c in json.load(open(ROOT + "/benchmarks/manifest.json", encoding="utf-8"))["clips"]}


def smooth(ll, w=5):
    h = w // 2
    return np.array([np.median(ll[max(0, i - h):i + h + 1], axis=0) for i in range(len(ll))])


def decode(z, centred):
    s = z["raw_similarities"].astype(np.float32)
    if centred:
        s = s - s.mean(axis=0, keepdims=True)
    t = z["query_times"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 4.0
    path = _viterbi_decode(s, z["ref_latlon"], dt)
    return path, z["raw_similarities"][np.arange(len(path)), path]


rows = []
for cid in sorted(man):
    if man[cid]["split"] != "dev":
        continue
    audit = sorted(glob.glob(f"{RUN}/{cid}/rep0/pipeline_out/*/children/*/baseline/retrieval_audit/*.npz"))
    if not audit:
        continue
    gt = np.array([[w["lat"], w["lon"]] for w in json.load(open(ROOT + "/" + man[cid]["gt"], encoding="utf-8"))["waypoints"]])
    row = {"clip": cid, "n_passes": len(audit)}
    for k, f in enumerate(audit):
        z = np.load(f, allow_pickle=False)
        raw_path, raw_sim = decode(z, False)
        row[f"p{k}_replay_matches"] = bool(np.array_equal(raw_path, z["decoded_ref_indices"]))
        for name, centred in (("raw", False), ("centred", True)):
            path, sim = decode(z, centred) if centred else (raw_path, raw_sim)
            ll = z["ref_latlon"][path]
            sym = score_route(smooth(ll), gt).symmetric_mean_m
            c = _robust_center(ll, sim)
            dc = min(geodesic_distance_m(c[0], c[1], a, b) for a, b in gt)
            row[f"p{k}_{name}_track_m"] = round(float(sym), 1)
            row[f"p{k}_{name}_centre_to_route_m"] = round(float(dc), 1)
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
