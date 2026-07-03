"""Offline validation of the local OSM gazetteer anchor source.

For Ulm and London: load cached OCR detections + the road graph, build
the gazetteer (osmnx download, cached), fuzzy-match, and report for each
matched anchor its distance to the GT route polyline and how many land
within 300 m. No pipeline run and no Nominatim needed.

Run:
    C:/Users/localadmin/AppData/Local/Programs/Python/Python312/python.exe \
        scripts/test_gazetteer_offline.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import osmnx as ox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.osm_data import _build_polyline_view  # noqa: E402
from src.osm_gazetteer import build_gazetteer, match_texts  # noqa: E402
from src.scene_text import SceneText  # noqa: E402


def _load_detections(cache_path: Path) -> list[SceneText]:
    d = json.loads(cache_path.read_text(encoding="utf-8"))
    return [SceneText(x["text"], float(x["confidence"]), float(x["t_sec"]))
            for x in d["detections"]]


def _gt_polyline(gt_path: Path) -> list[tuple[float, float]]:
    d = json.loads(gt_path.read_text(encoding="utf-8"))
    return [(w["lat"], w["lon"]) for w in d["waypoints"]]


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _dist_to_polyline_m(lat, lon, poly: list[tuple[float, float]]) -> float:
    """Min distance from a point to a densified GT waypoint polyline."""
    if len(poly) == 1:
        return _haversine_m(lat, lon, poly[0][0], poly[0][1])
    best = math.inf
    lat0 = poly[0][0]
    mlon = 111320.0 * math.cos(math.radians(lat0))
    mlat = 111320.0

    def to_xy(la, lo):
        return np.array([(lo - poly[0][1]) * mlon, (la - lat0) * mlat])

    p = to_xy(lat, lon)
    for (la1, lo1), (la2, lo2) in zip(poly[:-1], poly[1:]):
        a, b = to_xy(la1, lo1), to_xy(la2, lo2)
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom == 0 else float(np.clip((p - a) @ ab / denom, 0, 1))
        proj = a + t * ab
        best = min(best, float(np.linalg.norm(p - proj)))
    return best


def _run(label, graphml, cache, gt, cache_out):
    print(f"\n=== {label} ===")
    graph = _build_polyline_view(ox.load_graphml(graphml))
    dets = _load_detections(cache)
    poly = _gt_polyline(gt)
    print(f"detections: {len(dets)}  graph nodes: {len(graph.graph.nodes)}  "
          f"GT waypoints: {len(poly)}")

    gaz = build_gazetteer(graph, cache_path=cache_out)
    print(f"gazetteer entries: {len(gaz['entries'])}")

    anchors = match_texts(dets, gaz, min_confidence=0.5)
    within = 0
    for a in sorted(anchors, key=lambda a: -a.confidence):
        d = _dist_to_polyline_m(a.lat, a.lon, poly)
        flag = "OK <=300m" if d <= 300 else ""
        if d <= 300:
            within += 1
        print(f"  {a.name!r:32s} conf={a.confidence:.2f} "
              f"@({a.lat:.5f},{a.lon:.5f})  {d:7.0f} m to GT  {flag}")
    print(f"--> {len(anchors)} matched anchors, {within} within 300 m of GT")
    return len(anchors), within


def main():
    data = ROOT / "data"
    scratch = Path(
        "C:/Users/LOCALA~1/AppData/Local/Temp/claude/"
        "C--Users-localadmin-Documents-Monocular-OSM-Localization/"
        "19e9dfcd-21b8-4761-a3ff-4e00ae57a8f1/scratchpad"
    )
    scratch.mkdir(parents=True, exist_ok=True)

    n_ulm, w_ulm = _run(
        "Ulm 4K",
        data / "ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/Ulm_Germany.graphml",
        data / "ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/scene_text_cache_4k.json",
        ROOT / "ground_truth/ulm_ULl8s4qydrk.json",
        scratch / "gaz_ulm.json",
    )
    n_lon, w_lon = _run(
        "London",
        data / "local-73200bdd8068-input-london-uk/London_UK_around_51.5200_-0.1290_2500.graphml",
        data / "london_T4wTL3LpLqU/scene_text_cache_4k.json",
        ROOT / "ground_truth/london_T4wTL3LpLqU.json",
        scratch / "gaz_london.json",
    )

    print("\n=== SUMMARY ===")
    print(f"Ulm:    {n_ulm} anchors, {w_ulm} within 300 m")
    print(f"London: {n_lon} anchors, {w_lon} within 300 m "
          f"(SUCCESS needs >= 2)")


if __name__ == "__main__":
    main()
