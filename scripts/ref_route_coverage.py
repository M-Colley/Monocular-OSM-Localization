"""Pass-1 reference coverage of the route (diagnostic).

    python scripts/ref_route_coverage.py OUT.json CLIP=PATH/mly_ref_store_*.json [...]

Diagnostic (uses GT of already-scored clips; no method is tuned on it):
how much of the GT route has a pass-1 Mapillary reference nearby, for the
served hash subsample vs a grid-stratified subsample of the SAME listing and cap."""
import glob, hashlib, heapq, json, sys
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from monocular_osm.geo import LocalFrame
CAP = 1200
def gt_points(clip):
    g = json.load(open(ROOT / "ground_truth" / f"{clip}.json", encoding="utf-8"))
    wps = g.get("waypoints") or g.get("route") or g
    pts = [(w["lat"], w["lon"]) if isinstance(w, dict) else (w[0], w[1]) for w in wps]
    return np.array(pts, float)
def densify(xy, step=10.0):
    out = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(1, int(np.hypot(*(b - a)) // step))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)
def hash_sample(ids):
    return heapq.nsmallest(CAP, ids, key=lambda i: hashlib.sha1(i.encode()).hexdigest())
def grid_sample(ids, xy, cell):
    # one id per occupied cell (lowest sha1 inside the cell), cells visited in sha1 order of their
    # representative, until CAP: spatially even, deterministic, still GT-free
    cells = {}
    for i, (x, y) in zip(ids, xy):
        k = (int(np.floor(x / cell)), int(np.floor(y / cell)))
        h = hashlib.sha1(i.encode()).hexdigest()
        if k not in cells or h < cells[k][0]: cells[k] = (h, i)
    reps = sorted(cells.values())[:CAP]
    return [i for _, i in reps], len(cells)
rows = []
OUT = Path(sys.argv[1])
for clip, store in [(a, b) for a, b in (s.split("=", 1) for s in sys.argv[2:])]:
    union = json.load(open(store))
    ids = list(union.keys())
    gt = gt_points(clip)
    f = LocalFrame(*gt.mean(0))
    route = densify(np.array([f.to_xy(*p) for p in gt]))
    xy = np.array([f.to_xy(union[i][0], union[i][1]) for i in ids])
    idx = {i: k for k, i in enumerate(ids)}
    res = {"clip": clip, "listed": len(ids)}
    for name, sel in [("hash", hash_sample(ids))] + [(f"grid{c}", grid_sample(ids, xy, c)[0]) for c in (250, 400)]:
        P = xy[[idx[i] for i in sel]]
        d = cKDTree(P).query(route)[0]
        res[name] = {"within50": round(float(np.mean(d <= 50)), 3), "within150": round(float(np.mean(d <= 150)), 3),
                     "median_m": round(float(np.median(d)))}
    dall = cKDTree(xy).query(route)[0]
    res["all_listed_within50"] = round(float(np.mean(dall <= 50)), 3)
    rows.append(res); print(json.dumps(res)); sys.stdout.flush()
OUT.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
