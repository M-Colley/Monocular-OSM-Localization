"""Feasibility kill experiments for the house-number + street-pair anchor idea.

KILL A (OSM density): count addr:housenumber features within ~100 m of the Ulm GT
route. Need > 50 (and report how many also have addr:street).

KILL B (OCR readability): count plausible digit-only house-number reads. First check
the cached OCR detections; if they lack enough, run easyocr (gpu=False) over ~20 frames
sampled across input_4k.webm. Need >= 5 plausible reads.

Run with the Python312 interpreter:
  C:/Users/localadmin/AppData/Local/Programs/Python/Python312/python.exe \
    scripts/test_housenumber_feasibility.py
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(REPO, "ground_truth", "ulm_ULl8s4qydrk.json")
DATA_DIR = os.path.join(
    REPO, "data", "ull8s4qydrk-ulm-germany-4k-drive-ulm-germany"
)
CACHE = os.path.join(DATA_DIR, "scene_text_cache_4k.json")
VIDEO = os.path.join(DATA_DIR, "input_4k.webm")


def load_route_line():
    from shapely.geometry import LineString

    gt = json.load(open(GT, encoding="utf-8"))
    pts = [(w["lon"], w["lat"]) for w in gt["waypoints"]]
    return LineString(pts)


def kill_a():
    """OSM address density within ~100 m of the GT route."""
    import osmnx as ox
    from shapely.geometry import LineString  # noqa: F401

    line = load_route_line()
    # Buffer ~100 m. Route centroid latitude ~48.4 -> 1 deg lat ~= 111.1 km,
    # 1 deg lon ~= 74 km. Use a mean-degree buffer of ~100 m.
    mean_lat = sum(y for _, y in line.coords) / len(line.coords)
    import math

    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(mean_lat))
    m_per_deg = (m_per_deg_lat + m_per_deg_lon) / 2.0
    buf_deg = 100.0 / m_per_deg
    poly = line.buffer(buf_deg)

    tags = {"addr:housenumber": True}
    gdf = ox.features_from_polygon(poly, tags)
    n_total = len(gdf)
    has_street = 0
    if "addr:street" in gdf.columns:
        has_street = int(gdf["addr:street"].notna().sum())
    examples = []
    if n_total:
        cols = [c for c in ("addr:street", "addr:housenumber") if c in gdf.columns]
        sub = gdf[cols].dropna()
        for _, row in sub.head(12).iterrows():
            examples.append(
                f"{row.get('addr:street','?')} {row.get('addr:housenumber','?')}"
            )
    print("=== KILL A: OSM address density (<=100 m of GT route) ===")
    print(f"addr:housenumber features: {n_total}")
    print(f"  ...also with addr:street: {has_street}")
    print("examples:", examples)
    passed = n_total > 50
    print(f"KILL A {'PASS' if passed else 'FAIL'} (need > 50)")
    return passed, n_total, has_street, examples


def _digit_reads_from_cache():
    d = json.load(open(CACHE, encoding="utf-8"))
    reads = []
    for x in d.get("detections", []):
        t = x["text"].strip()
        if re.fullmatch(r"\d{1,4}", t) and x.get("confidence", 0) >= 0.5:
            reads.append((t, round(x["confidence"], 3), round(x["t_sec"], 1)))
    return reads, d.get("signature", {})


def kill_b(force_ocr=False):
    print("\n=== KILL B: OCR house-number readability ===")
    cache_reads, sig = _digit_reads_from_cache()
    print(f"cache signature min_len={sig.get('min_len')} "
          f"interval={sig.get('sample_interval_sec')}s")
    print(f"cache digit-only 1-4char conf>=0.5: {len(cache_reads)} -> {cache_reads}")
    # The cache uses min_len=3 so 1-2 digit numbers are pre-filtered; we must
    # run our own easyocr pass to fairly count short house numbers.
    reads = list(cache_reads)
    if force_ocr or len(cache_reads) < 5:
        print("Running fresh easyocr (gpu=False) over ~20 sampled frames...")
        reads = _easyocr_pass()
    passed = len(reads) >= 5
    print(f"total plausible reads: {len(reads)}")
    print(f"KILL B {'PASS' if passed else 'FAIL'} (need >= 5)")
    return passed, reads


def _easyocr_pass(n_frames=20):
    import cv2
    import easyocr

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        # webm sometimes reports 0 frame count; fall back to duration by seeking ms
        dur = 420.0
        idxs_ms = [int(dur * 1000 * i / (n_frames - 1)) for i in range(n_frames)]
        use_ms = True
    else:
        idxs = [int(total * i / (n_frames - 1)) for i in range(n_frames)]
        use_ms = False
    reader = easyocr.Reader(["de", "en"], gpu=False)
    reads = []
    for i in range(n_frames):
        if use_ms:
            cap.set(cv2.CAP_PROP_POS_MSEC, idxs_ms[i])
            t_sec = idxs_ms[i] / 1000.0
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idxs[i])
            t_sec = idxs[i] / fps
        ok, frame = cap.read()
        if not ok:
            continue
        for _box, text, conf in reader.readtext(frame):
            t = text.strip()
            if re.fullmatch(r"\d{1,4}", t) and conf >= 0.5:
                reads.append((t, round(float(conf), 3), round(t_sec, 1)))
    cap.release()
    print(f"easyocr digit-only 1-4char conf>=0.5: {len(reads)}")
    for r in reads:
        print("  ", r)
    return reads


if __name__ == "__main__":
    force = "--force-ocr" in sys.argv
    a_pass, *_ = kill_a()
    b_pass, _ = kill_b(force_ocr=force)
    print("\n=== VERDICT ===")
    print(f"KILL A pass: {a_pass}   KILL B pass: {b_pass}")
    print(f"PROCEED: {a_pass and b_pass}")
