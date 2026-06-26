"""Does map-semantic (traffic-signals + road-class) discriminate the true route
from the wrong pick, inside the VPR gate?

(1) YOLO detects traffic-light APPROACH events in the video (a large, central light).
(2) From the OSM graph: traffic_signals nodes + edge highway class.
(3) Compare the video's signal count + road character against the GT (Olgastrasse)
    route vs the blind pipeline's wrong final pick. If the GT route matches the
    video's semantics much better, the cue is worth integrating as a re-ranker.
"""

from __future__ import annotations

import glob
import json

import cv2
import numpy as np
import osmnx as ox
from pyproj import Transformer
from ultralytics import YOLO

VIDEO = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4"
GT = "ground_truth/ulm_ULl8s4qydrk.json"
TF = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)


def densify(lls, step=15.0):
    xy = np.array([TF.transform(lo, la) for la, lo in lls])
    out = [xy[0]]
    for i in range(1, len(xy)):
        d = np.linalg.norm(xy[i] - xy[i - 1]); n = max(1, int(d / step))
        for k in range(1, n + 1):
            out.append(xy[i - 1] + (xy[i] - xy[i - 1]) * k / n)
    return np.array(out)


def signals_near(poly, sig_xy, thr=28.0):
    if len(sig_xy) == 0:
        return 0
    hit = 0
    for s in sig_xy:
        if np.min(np.linalg.norm(poly - s, axis=1)) < thr:
            hit += 1
    return hit


def road_classes(poly, G, edge_xy_cls):
    cls = []
    for (mx, my, c) in edge_xy_cls:
        if np.min(np.linalg.norm(poly - np.array([mx, my]), axis=1)) < 25.0:
            cls.append(c)
    from collections import Counter
    return Counter(cls)


def main():
    # ---- (1) YOLO traffic-light approach events ----
    print("[1] YOLO traffic-light detection on the video")
    model = YOLO("yolov8s.pt")
    cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    times = np.arange(0, min(dur, 420), 2.5)
    promin = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000); ok, f = cap.read()
        if not ok:
            promin.append(0.0); continue
        h, w = f.shape[:2]
        r = model(f, verbose=False, classes=[9], conf=0.30)[0]
        best = 0.0
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx = (x1 + x2) / 2 / w
            area = (x2 - x1) * (y2 - y1) / (w * h)
            if 0.2 < cx < 0.8:                       # central lights = ones we approach
                best = max(best, area)
        promin.append(best)
    cap.release()
    promin = np.array(promin)
    # an "approach event" = a peak where a central light gets large then passes
    big = promin > 0.0008
    events = int(np.sum((big[1:] & ~big[:-1])))      # rising edges
    ev_t = times[1:][(big[1:] & ~big[:-1])]
    print(f"  video: {events} traffic-light approach events at t={[int(x) for x in ev_t]}s "
          f"(max prominence {promin.max():.4f})")

    # ---- (2) OSM signals + road class from the VPR-gated graph ----
    gp = glob.glob("data/ull8s4qydrk-*/Ulm_Germany_around_*900.graphml")
    G = ox.load_graphml(gp[0])
    sig_xy = np.array([[float(d["x"]), float(d["y"])] for n, d in G.nodes(data=True)
                       if str(d.get("highway", "")).find("traffic_signals") >= 0])
    edge_xy_cls = []
    for u, v, d in G.edges(data=True):
        hw = d.get("highway"); hw = hw[0] if isinstance(hw, list) else hw
        mx = (float(G.nodes[u]["x"]) + float(G.nodes[v]["x"])) / 2
        my = (float(G.nodes[u]["y"]) + float(G.nodes[v]["y"])) / 2
        edge_xy_cls.append((mx, my, hw))
    print(f"[2] OSM gate: {len(sig_xy)} traffic_signals nodes, {G.number_of_edges()} edges")

    # ---- (3) GT route vs wrong final pick ----
    wps = json.load(open(GT))["waypoints"]
    gt_poly = densify([(w["lat"], w["lon"]) for w in wps])
    res = json.load(open(glob.glob("output/blind_vpr_chain/*/result.json")[0], encoding="utf-8"))["position"]
    wrong_poly = densify([(a, b) for a, b in res["route_latlon"]])

    print("\n[3] discrimination:")
    for name, poly in [("GT route (Olgastrasse)", gt_poly), ("wrong final pick", wrong_poly)]:
        ns = signals_near(poly, sig_xy)
        rc = road_classes(poly, G, edge_xy_cls)
        arterial = sum(v for k, v in rc.items() if k in ("primary", "secondary", "tertiary",
                       "primary_link", "secondary_link"))
        resid = sum(v for k, v in rc.items() if k in ("residential", "living_street", "service",
                    "unclassified"))
        frac = arterial / max(1, arterial + resid)
        print(f"  {name:24s}: {ns} signals on route | arterial-fraction {frac:.0%} "
              f"| classes {dict(rc.most_common(4))}")
    print(f"\n  video shows {events} signal approaches on a tram arterial -> "
          f"compare which route matches.")


if __name__ == "__main__":
    main()
