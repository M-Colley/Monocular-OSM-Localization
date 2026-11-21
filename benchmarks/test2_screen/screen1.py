"""Screen YouTube for test2 candidates: city-titled dashcam drives whose own
frame thumbnails (hq1-3, at 25/50/75 % of the video) show a GPS coordinate
stamp that parses near the named city. Selection reads only titles,
durations and those three thumbnails; nothing the pipeline produces.

    python screen.py out.json
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\markc\Desktop\Monocular-OSM-Localization")
sys.path.insert(0, str(ROOT))
from monocular_osm.geo import geodesic_distance_m  # noqa: E402
from monocular_osm.gps_overlay import _prepare_band, parse_latlon  # noqa: E402
from monocular_osm.scene_text import RapidOcrReader  # noqa: E402

# Fixed before screening: cities absent from dev and test1, in this order.
CITIES = [
    ("Vienna", "Vienna, Austria", ["vienna", "wien"]),
    ("Munich", "Munich, Germany", ["munich", "munchen", "muenchen", "münchen"]),
    ("Hamburg", "Hamburg, Germany", ["hamburg"]),
    ("Zurich", "Zurich, Switzerland", ["zurich", "zürich", "zuerich"]),
    ("Paris", "Paris, France", ["paris"]),
    ("Lyon", "Lyon, France", ["lyon"]),
    ("Amsterdam", "Amsterdam, Netherlands", ["amsterdam"]),
    ("Brussels", "Brussels, Belgium", ["brussels", "bruxelles", "brussel"]),
    ("Prague", "Prague, Czechia", ["prague", "praha"]),
    ("Warsaw", "Warsaw, Poland", ["warsaw", "warszawa"]),
    ("Copenhagen", "Copenhagen, Denmark", ["copenhagen", "kobenhavn", "københavn"]),
    ("Stockholm", "Stockholm, Sweden", ["stockholm"]),
    ("Madrid", "Madrid, Spain", ["madrid"]),
    ("Barcelona", "Barcelona, Spain", ["barcelona"]),
    ("Milan", "Milan, Italy", ["milan", "milano"]),
    ("Toronto", "Toronto, Canada", ["toronto"]),
    ("Montreal", "Montreal, Canada", ["montreal", "montréal"]),
    ("Vancouver", "Vancouver, Canada", ["vancouver"]),
    ("Boston", "Boston, USA", ["boston"]),
    ("Seattle", "Seattle, USA", ["seattle"]),
    ("San Francisco", "San Francisco, USA", ["san francisco"]),
    ("Los Angeles", "Los Angeles, USA", ["los angeles"]),
    ("Houston", "Houston, USA", ["houston"]),
    ("Denver", "Denver, USA", ["denver"]),
    ("Melbourne", "Melbourne, Australia", ["melbourne"]),
    ("Brisbane", "Brisbane, Australia", ["brisbane"]),
    ("Auckland", "Auckland, New Zealand", ["auckland"]),
]
QUERIES = ["{c} dashcam driving downtown", "{c} dash cam drive city"]
MIN_S, MAX_PER_QUERY = 420, 12

used = {c["video"].get("url", "").split("v=")[-1]
        for c in json.loads((ROOT / "benchmarks/manifest.json").read_text(encoding="utf-8"))["clips"]}
reader = RapidOcrReader()
geo_cache: dict[str, tuple] = {}


def geocode(q):
    if q not in geo_cache:
        import osmnx as ox
        try:
            geo_cache[q] = ox.geocode(q)
        except Exception:
            geo_cache[q] = None
    return geo_cache[q]


def fixes(vid):
    out = []
    for k in (1, 2, 3):
        try:
            raw = urllib.request.urlopen(f"https://i.ytimg.com/vi/{vid}/hq{k}.jpg", timeout=20).read()
        except Exception:
            continue
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        got = None
        for region in ("top", "bottom"):
            for frac, scale in ((0.18, 2.0), (0.10, 3.0)):
                band = _prepare_band(img, region, frac, scale)
                texts = [str(t) for (_b, t, c) in reader.readtext(band) if float(c) >= 0.3]
                for cand in [" ".join(texts), *texts]:
                    ll = parse_latlon(cand)
                    if ll:
                        got = (region, ll)
                        break
                if got:
                    break
            if got:
                break
        if got:
            out.append({"thumb": k, "region": got[0], "lat": got[1][0], "lon": got[1][1]})
    return out


results, seen = [], set(used)
for name, city, keys in CITIES:
    centre = geocode(city)
    for q in QUERIES:
        p = subprocess.run(["yt-dlp", "--flat-playlist", "-J", f"ytsearch{MAX_PER_QUERY}:{q.format(c=name)}"],
                           capture_output=True, timeout=180)
        try:
            entries = json.loads(p.stdout.decode("utf-8", "replace")).get("entries", [])
        except ValueError:
            entries = []
        for e in entries:
            vid, title, dur = e.get("id"), e.get("title") or "", e.get("duration") or 0
            if not vid or vid in seen:
                continue
            seen.add(vid)
            low = title.casefold()
            if dur < MIN_S or not any(k in low or k.replace(" ", "") in low.replace(" ", "") for k in keys):
                continue
            fx = fixes(vid)
            near = [f for f in fx if centre and geodesic_distance_m(centre[0], centre[1], f["lat"], f["lon"]) < 40_000]
            row = {"city": city, "id": vid, "title": title, "duration_s": dur, "query": q.format(c=name),
                   "fixes": fx, "n_near": len(near), "ok": len(near) >= 2}
            results.append(row)
            print(("OK  " if row["ok"] else "    ") + f"{city:24s} {vid} {dur:5.0f}s {len(fx)} fixes/{len(near)} near  {title[:70]}",
                  flush=True)
    Path(sys.argv[1]).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
print("candidates passing:", sum(r["ok"] for r in results), "of", len(results), "screened")
