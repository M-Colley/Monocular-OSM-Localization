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
    ('Rotterdam, Netherlands', ['rotterdam'], ['Rotterdam dashcam VIOFO', 'Rotterdam daily dashcam']),
    ('Utrecht, Netherlands', ['utrecht'], ['Utrecht dashcam VIOFO', 'Utrecht daily dashcam']),
    ('The Hague, Netherlands', ['hague', 'den haag'], ['The Hague dashcam VIOFO', 'The Hague daily dashcam']),
    ('Antwerp, Belgium', ['antwerp', 'antwerpen'], ['Antwerp dashcam VIOFO', 'Antwerp daily dashcam']),
    ('Ghent, Belgium', ['ghent', 'gent'], ['Ghent dashcam VIOFO', 'Ghent daily dashcam']),
    ('Krakow, Poland', ['krakow', 'kraków'], ['Krakow dashcam VIOFO', 'Krakow daily dashcam']),
    ('Wroclaw, Poland', ['wroclaw', 'wrocław'], ['Wroclaw dashcam VIOFO', 'Wroclaw daily dashcam']),
    ('Gdansk, Poland', ['gdansk', 'gdańsk'], ['Gdansk dashcam VIOFO', 'Gdansk daily dashcam']),
    ('Budapest, Hungary', ['budapest'], ['Budapest dashcam VIOFO', 'Budapest daily dashcam']),
    ('Vienna, Austria', ['vienna', 'wien'], ['Vienna dashcam VIOFO', 'Vienna daily dashcam']),
    ('Munich, Germany', ['munich', 'münchen', 'munchen'], ['Munich dashcam VIOFO', 'Munich daily dashcam']),
    ('Copenhagen, Denmark', ['copenhagen', 'københavn'], ['Copenhagen dashcam VIOFO', 'Copenhagen daily dashcam']),
    ('Oslo, Norway', ['oslo'], ['Oslo dashcam VIOFO', 'Oslo daily dashcam']),
    ('Helsinki, Finland', ['helsinki'], ['Helsinki dashcam VIOFO', 'Helsinki daily dashcam']),
    ('Dublin, Ireland', ['dublin'], ['Dublin dashcam VIOFO', 'Dublin daily dashcam']),
    ('Lisbon, Portugal', ['lisbon', 'lisboa'], ['Lisbon dashcam VIOFO', 'Lisbon daily dashcam']),
    ('Porto, Portugal', ['porto'], ['Porto dashcam VIOFO', 'Porto daily dashcam']),
    ('Valencia, Spain', ['valencia'], ['Valencia dashcam VIOFO', 'Valencia daily dashcam']),
    ('Rome, Italy', ['rome', 'roma'], ['Rome dashcam VIOFO', 'Rome daily dashcam']),
    ('Turin, Italy', ['turin', 'torino'], ['Turin dashcam VIOFO', 'Turin daily dashcam']),
    ('Quebec City, Canada', ['quebec'], ['Quebec City dashcam VIOFO', 'Quebec City daily dashcam']),
    ('Calgary, Canada', ['calgary'], ['Calgary dashcam VIOFO', 'Calgary daily dashcam']),
    ('Edmonton, Canada', ['edmonton'], ['Edmonton dashcam VIOFO', 'Edmonton daily dashcam']),
    ('Winnipeg, Canada', ['winnipeg'], ['Winnipeg dashcam VIOFO', 'Winnipeg daily dashcam']),
    ('Seattle, USA', ['seattle'], ['Seattle dashcam VIOFO', 'Seattle daily dashcam']),
    ('Portland, Oregon, USA', ['portland'], ['Portland dashcam VIOFO', 'Portland daily dashcam']),
    ('Dallas, Texas, USA', ['dallas'], ['Dallas dashcam VIOFO', 'Dallas daily dashcam']),
    ('Austin, Texas, USA', ['austin'], ['Austin dashcam VIOFO', 'Austin daily dashcam']),
    ('Miami, Florida, USA', ['miami'], ['Miami dashcam VIOFO', 'Miami daily dashcam']),
    ('Nashville, Tennessee, USA', ['nashville'], ['Nashville dashcam VIOFO', 'Nashville daily dashcam']),
    ('Baltimore, Maryland, USA', ['baltimore'], ['Baltimore dashcam VIOFO', 'Baltimore daily dashcam']),
    ('Cleveland, Ohio, USA', ['cleveland'], ['Cleveland dashcam VIOFO', 'Cleveland daily dashcam']),
    ('Salt Lake City, Utah, USA', ['salt lake'], ['Salt Lake City dashcam VIOFO', 'Salt Lake City daily dashcam']),
    ('San Diego, California, USA', ['san diego'], ['San Diego dashcam VIOFO', 'San Diego daily dashcam']),
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


EXTRA = []


import json as _j
prior = {r["id"] for f in ("screen1.json", "screen2.json") for r in _j.load(open(f, encoding="utf-8"))}
rows, seen = [], set(used) | prior
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
