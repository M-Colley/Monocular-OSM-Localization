"""Second test2 screen: targeted queries, then full-resolution verification.

Stage 1 (titles + YouTube's own frame thumbnails hq1-3): any parsed stamp.
Stage 2: three 2-second sections (25/50/75 % of the video) downloaded at
<=1080p and read with the ground-truth builder's own band OCR; a candidate
passes with >= 2 fixes within 40 km of the city named in its title.
Exclusions decided from the title alone, before any pipeline run: timelapse /
hyperlapse / sped-up footage, rear-facing cameras, videos under 7 min, clips
already in the manifest.

    python screen2.py screen2.json
"""
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\markc\Desktop\Monocular-OSM-Localization")
sys.path.insert(0, str(ROOT))
from monocular_osm.geo import geodesic_distance_m  # noqa: E402
from monocular_osm.gps_overlay import _prepare_band, parse_latlon  # noqa: E402
from monocular_osm.scene_text import RapidOcrReader  # noqa: E402

# (city string as titled, title keys, queries) - fixed before screening.
CITIES = [
    ("Vienna, Austria", ["wien", "vienna"], ["Wien Rundfahrt durch die Innenstadt", "Vienna dashcam GPS"]),
    ("Munich, Germany", ["münchen", "munchen", "munich"], ["München Rundfahrt durch die Innenstadt", "Munich dashcam GPS"]),
    ("Hamburg, Germany", ["hamburg"], ["Hamburg Rundfahrt durch die Innenstadt", "Hamburg dashcam GPS"]),
    ("Cologne, Germany", ["köln", "koln", "cologne"], ["Köln Rundfahrt durch die Innenstadt", "Köln dashcam"]),
    ("Stuttgart, Germany", ["stuttgart"], ["Stuttgart Rundfahrt durch die Innenstadt", "Stuttgart dashcam"]),
    ("Dresden, Germany", ["dresden"], ["Dresden Rundfahrt durch die Innenstadt", "Dresden dashcam"]),
    ("Nuremberg, Germany", ["nürnberg", "nurnberg", "nuremberg"], ["Nürnberg Rundfahrt durch die Innenstadt"]),
    ("Hannover, Germany", ["hannover", "hanover"], ["Hannover Rundfahrt durch die Innenstadt"]),
    ("Mainz, Germany", ["mainz"], ["Mainz Rundfahrt durch die Innenstadt"]),
    ("Wiesbaden, Germany", ["wiesbaden"], ["Wiesbaden Rundfahrt durch die Innenstadt"]),
    ("Heidelberg, Germany", ["heidelberg"], ["Heidelberg Rundfahrt durch die Innenstadt"]),
    ("Freiburg im Breisgau, Germany", ["freiburg"], ["Freiburg Rundfahrt durch die Innenstadt"]),
    ("Basel, Switzerland", ["basel"], ["Basel Rundfahrt durch die Innenstadt", "Basel dashcam"]),
    ("Zurich, Switzerland", ["zürich", "zurich"], ["Zürich Rundfahrt durch die Innenstadt"]),
    ("Lucerne, Switzerland", ["luzern", "lucerne"], ["Luzern Rundfahrt durch die Innenstadt"]),
    ("Salzburg, Austria", ["salzburg"], ["Salzburg Rundfahrt durch die Innenstadt"]),
    ("Luxembourg City, Luxembourg", ["luxembourg", "luxemburg"], ["Luxemburg Rundfahrt durch die Innenstadt"]),
    ("Metz, France", ["metz"], ["Metz tour around the city dashcam"]),
    ("Paris, France", ["paris"], ["Paris dashcam GPS coordinates", "Paris Rundfahrt"]),
    ("Lyon, France", ["lyon"], ["Lyon dashcam GPS"]),
    ("Amsterdam, Netherlands", ["amsterdam"], ["Amsterdam dashcam GPS", "Amsterdam dashcam"]),
    ("Warsaw, Poland", ["warsaw", "warszawa"], ["Warsaw driving tour 2K", "Warszawa dashcam"]),
    ("Birmingham, UK", ["birmingham"], ["Birmingham city centre drive Nextbase", "Birmingham dash cam drive"]),
    ("Leeds, UK", ["leeds"], ["Leeds city centre drive dash cam"]),
    ("Glasgow, UK", ["glasgow"], ["Glasgow city centre drive dash cam"]),
    ("Edinburgh, UK", ["edinburgh"], ["Edinburgh drive dash cam"]),
    ("Liverpool, UK", ["liverpool"], ["Liverpool city centre drive dash cam"]),
    ("Toronto, Canada", ["toronto"], ["Toronto dashcam GPS", "Driving Toronto downtown dashcam"]),
    ("Montreal, Canada", ["montreal", "montréal"], ["Montreal dashcam GPS"]),
    ("Boston, USA", ["boston"], ["Driving Boston dashcam ROVE", "Boston dashcam GPS"]),
    ("San Francisco, USA", ["san francisco"], ["San Francisco dashcam GPS"]),
    ("Washington, D.C., USA", ["washington"], ["Driving Washington DC dashcam"]),
    ("Philadelphia, USA", ["philadelphia"], ["Driving Philadelphia dashcam"]),
    ("Pittsburgh, USA", ["pittsburgh"], ["Driving Pittsburgh dashcam"]),
    ("Minneapolis, USA", ["minneapolis"], ["Driving Minneapolis dashcam"]),
    ("Perth, Australia", ["perth"], ["Perth driving dashcam Navman"]),
    ("Adelaide, Australia", ["adelaide"], ["Adelaide driving dashcam"]),
]
EXCLUDE = re.compile(r"time\s*-?lapse|hyper\s*-?lapse|\b\d+x\b|sped\s*up|speed\s*up|rear\s*-?view|rear camera|backwards", re.I)
MIN_S, PER_QUERY = 420, 12
used = {c["video"].get("url", "").split("v=")[-1]
        for c in json.loads((ROOT / "benchmarks/manifest.json").read_text(encoding="utf-8"))["clips"]}
reader = RapidOcrReader()


def geocode(q, _c={}):
    if q not in _c:
        import osmnx as ox
        try:
            _c[q] = ox.geocode(q)
        except Exception:
            _c[q] = None
    return _c[q]


def read_stamp(img):
    for region in ("bottom", "top"):
        for frac, scale in ((0.18, 1.0), (0.18, 2.0), (0.10, 3.0)):
            band = _prepare_band(img, region, frac, scale)
            texts = [str(t) for (_b, t, c) in reader.readtext(band) if float(c) >= 0.3]
            for cand in [" ".join(texts), *texts]:
                ll = parse_latlon(cand)
                if ll:
                    return region, ll
    return None


def thumb_fixes(vid):
    out = []
    for k in (1, 2, 3):
        try:
            raw = urllib.request.urlopen(f"https://i.ytimg.com/vi/{vid}/hq{k}.jpg", timeout=20).read()
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            continue
        got = read_stamp(img) if img is not None else None
        if got:
            out.append(got)
    return out


def section_fixes(vid, dur):
    """Seek the <=1080p stream directly (no segment download) to 25/50/75 %."""
    out = []
    try:
        url = subprocess.run(["yt-dlp", "-g", "-f", "bv*[height<=1080][ext=mp4]/bv*[height<=1080]",
                              f"https://www.youtube.com/watch?v={vid}"], capture_output=True, text=True,
                             timeout=90).stdout.strip().splitlines()[0]
    except Exception:
        return out
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    for frac in (0.25, 0.5, 0.75):
        t = int(dur * frac)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, img = cap.read()
        got = read_stamp(img) if ok else None
        if got:
            out.append({"t": t, "region": got[0], "lat": got[1][0], "lon": got[1][1]})
    cap.release()
    return out


EXTRA = [("Amsterdam, Netherlands", "2sS6glBBm8g"), ("Amsterdam, Netherlands", "dGUqr5cTcrE"),
         ("Amsterdam, Netherlands", "ru2WkNy8o8s"), ("Warsaw, Poland", "GnXsgpVPvww"),
         ("Paris, France", "HpBfjRA60ws")]   # round-1 thumbnail near-misses


rows, seen = [], set(used)
for city, keys, queries in CITIES:
    centre = geocode(city)
    for q in queries:
        p = subprocess.run(["yt-dlp", "--flat-playlist", "-J", f"ytsearch{PER_QUERY}:{q}"], capture_output=True, timeout=180)
        try:
            entries = json.loads(p.stdout.decode("utf-8", "replace")).get("entries", [])
        except ValueError:
            entries = []
        for rank, e in enumerate(entries):
            vid, title, dur = e.get("id"), e.get("title") or "", e.get("duration") or 0
            if not vid or vid in seen:
                continue
            seen.add(vid)
            low = title.casefold()
            if dur < MIN_S or EXCLUDE.search(title) or not any(k in low for k in keys):
                continue
            tf = thumb_fixes(vid)
            row = {"city": city, "id": vid, "title": title, "duration_s": dur, "query": q, "rank": rank,
                   "thumb_fixes": [list(x[1]) for x in tf], "section_fixes": [], "n_near": 0, "ok": False}
            if tf:
                sf = section_fixes(vid, dur)
                near = [f for f in sf if centre and geodesic_distance_m(centre[0], centre[1], f["lat"], f["lon"]) < 40_000]
                row.update(section_fixes=sf, n_near=len(near), ok=len(near) >= 2)
            rows.append(row)
            if tf:
                print(("OK  " if row["ok"] else "    ") + f"{city:28s} {vid} {dur:5.0f}s thumbs {len(tf)} sections {len(row['section_fixes'])}/{row['n_near']} near  {title[:60]}", flush=True)
        Path(sys.argv[1]).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
for city, vid in EXTRA:
    if vid in {r["id"] for r in rows}:
        continue
    meta = json.loads(subprocess.run(["yt-dlp", "-J", "--skip-download", f"https://www.youtube.com/watch?v={vid}"],
                                     capture_output=True, timeout=120).stdout.decode("utf-8", "replace") or "{}")
    title, dur, centre = meta.get("title", ""), meta.get("duration") or 0, geocode(city)
    sf = section_fixes(vid, dur) if dur and not EXCLUDE.search(title) else []
    near = [f for f in sf if centre and geodesic_distance_m(centre[0], centre[1], f["lat"], f["lon"]) < 40_000]
    rows.append({"city": city, "id": vid, "title": title, "duration_s": dur, "query": "round-1 near-miss", "rank": None,
                 "thumb_fixes": [], "section_fixes": sf, "n_near": len(near), "ok": len(near) >= 2})
    print(("OK  " if len(near) >= 2 else "    ") + f"{city:28s} {vid} {dur:5.0f}s sections {len(sf)}/{len(near)} near  {title[:60]}", flush=True)
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
print("passing:", sum(r["ok"] for r in rows), "of", len(rows), "screened")
