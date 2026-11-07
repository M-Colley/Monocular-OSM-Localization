"""Screen YouTube for test3 candidates (third held-out split, 2026-11-07).

Same pass rule as the test2 screens (benchmarks/test2_screen/README.md), fixed
before screening: a video is eligible when its title names the city, it is at
least 7 minutes long and its title does not mark it as timelapse/hyperlapse,
sped-up or rear-facing footage. It passes when the coordinate stamp read from
three full-resolution frames (25/50/75 % of the video, seeked directly from the
stream) parses within 40 km of the named city in at least two of them. The
frames are only read when one of YouTube's own three frame thumbnails shows a
stamp. Selection reads titles, durations and those frames only; nothing the
localization pipeline produces.

Cities: none used by dev, test or test2, none screened in rounds 1-4, from the
same regions as the earlier splits (Europe, North America, Australia/New
Zealand). Three queries per city, in this order. YouTube calls are throttled
to one per 15 s; a bot check pauses the screen (it is not recorded as "no
result"), and the output is resumable.

    python screen5.py screen5.json [--workers N]

The thumbnail OCR of each query's eligible videos runs in N worker processes
(default 6; added after Düsseldorf, Dortmund and part of Essen had been screened
sequentially, which took ~13 min per city). Only the speed changes: the same
searches, filters, frames and pass rule.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from monocular_osm.geo import geodesic_distance_m  # noqa: E402
from monocular_osm.gps_overlay import _prepare_band, parse_latlon  # noqa: E402
from monocular_osm.scene_text import RapidOcrReader  # noqa: E402

# (city string, title keys) - fixed before screening.
CITIES = [
    # Germany, Austria, Switzerland
    ("Düsseldorf, Germany", ["düsseldorf", "dusseldorf", "duesseldorf"]),
    ("Dortmund, Germany", ["dortmund"]),
    ("Essen, Germany", ["essen"]),
    ("Leipzig, Germany", ["leipzig"]),
    ("Bremen, Germany", ["bremen"]),
    ("Bonn, Germany", ["bonn"]),
    ("Augsburg, Germany", ["augsburg"]),
    ("Münster, Germany", ["münster", "munster", "muenster"]),
    ("Graz, Austria", ["graz"]),
    ("Linz, Austria", ["linz"]),
    ("Innsbruck, Austria", ["innsbruck"]),
    ("Geneva, Switzerland", ["geneva", "genève", "geneve", "genf"]),
    ("Lausanne, Switzerland", ["lausanne"]),
    # France, Benelux
    ("Marseille, France", ["marseille"]),
    ("Toulouse, France", ["toulouse"]),
    ("Bordeaux, France", ["bordeaux"]),
    ("Nice, France", ["nice"]),
    ("Lille, France", ["lille"]),
    ("Nantes, France", ["nantes"]),
    ("Montpellier, France", ["montpellier"]),
    ("Eindhoven, Netherlands", ["eindhoven"]),
    ("Groningen, Netherlands", ["groningen"]),
    ("Liège, Belgium", ["liège", "liege", "luik"]),
    # Iberia, Italy, Greece
    ("Seville, Spain", ["seville", "sevilla"]),
    ("Bilbao, Spain", ["bilbao"]),
    ("Zaragoza, Spain", ["zaragoza"]),
    ("Naples, Italy", ["naples", "napoli"]),
    ("Florence, Italy", ["florence", "firenze"]),
    ("Bologna, Italy", ["bologna"]),
    ("Genoa, Italy", ["genoa", "genova"]),
    ("Athens, Greece", ["athens", "athina"]),
    # UK, Ireland
    ("Bristol, UK", ["bristol"]),
    ("Sheffield, UK", ["sheffield"]),
    ("Nottingham, UK", ["nottingham"]),
    ("Leicester, UK", ["leicester"]),
    ("Cardiff, UK", ["cardiff"]),
    ("Belfast, UK", ["belfast"]),
    ("Southampton, UK", ["southampton"]),
    ("Cork, Ireland", ["cork"]),
    # Nordics, Baltics
    ("Gothenburg, Sweden", ["gothenburg", "göteborg", "goteborg"]),
    ("Malmö, Sweden", ["malmö", "malmo"]),
    ("Aarhus, Denmark", ["aarhus", "århus"]),
    ("Bergen, Norway", ["bergen"]),
    ("Vilnius, Lithuania", ["vilnius"]),
    ("Riga, Latvia", ["riga"]),
    ("Tallinn, Estonia", ["tallinn"]),
    # Central and south-east Europe
    ("Brno, Czechia", ["brno"]),
    ("Bratislava, Slovakia", ["bratislava"]),
    ("Poznan, Poland", ["poznan", "poznań"]),
    ("Lodz, Poland", ["lodz", "łódź", "łodz"]),
    ("Katowice, Poland", ["katowice"]),
    ("Ljubljana, Slovenia", ["ljubljana"]),
    ("Zagreb, Croatia", ["zagreb"]),
    ("Bucharest, Romania", ["bucharest", "bucurești", "bucuresti"]),
    ("Sofia, Bulgaria", ["sofia"]),
    # Canada
    ("Hamilton, Ontario, Canada", ["hamilton"]),
    ("Halifax, Nova Scotia, Canada", ["halifax"]),
    ("Victoria, British Columbia, Canada", ["victoria"]),
    ("Kitchener, Ontario, Canada", ["kitchener"]),
    # USA
    ("Charlotte, North Carolina, USA", ["charlotte"]),
    ("Raleigh, North Carolina, USA", ["raleigh"]),
    ("Columbus, Ohio, USA", ["columbus"]),
    ("Cincinnati, Ohio, USA", ["cincinnati"]),
    ("Indianapolis, Indiana, USA", ["indianapolis"]),
    ("Milwaukee, Wisconsin, USA", ["milwaukee"]),
    ("Kansas City, Missouri, USA", ["kansas city"]),
    ("St. Louis, Missouri, USA", ["st. louis", "st louis", "saint louis"]),
    ("New Orleans, Louisiana, USA", ["new orleans"]),
    ("Orlando, Florida, USA", ["orlando"]),
    ("Tampa, Florida, USA", ["tampa"]),
    ("Jacksonville, Florida, USA", ["jacksonville"]),
    ("Las Vegas, Nevada, USA", ["las vegas"]),
    ("Sacramento, California, USA", ["sacramento"]),
    ("San Jose, California, USA", ["san jose"]),
    ("San Antonio, Texas, USA", ["san antonio"]),
    ("Albuquerque, New Mexico, USA", ["albuquerque"]),
    ("Tucson, Arizona, USA", ["tucson"]),
    ("Oklahoma City, Oklahoma, USA", ["oklahoma city"]),
    ("Louisville, Kentucky, USA", ["louisville"]),
    ("Memphis, Tennessee, USA", ["memphis"]),
    ("Richmond, Virginia, USA", ["richmond"]),
    ("Buffalo, New York, USA", ["buffalo"]),
    ("Omaha, Nebraska, USA", ["omaha"]),
    # Australia, New Zealand
    ("Perth, Australia", ["perth"]),
    ("Canberra, Australia", ["canberra"]),
    ("Gold Coast, Australia", ["gold coast"]),
    ("Hobart, Australia", ["hobart"]),
    ("Wellington, New Zealand", ["wellington"]),
    ("Christchurch, New Zealand", ["christchurch"]),
]
QUERIES = ["{c} dashcam GPS", "{c} dashcam VIOFO", "{c} dashcam drive"]
EXCLUDE = re.compile(r"time\s*-?lapse|hyper\s*-?lapse|\b\d+x\b|sped\s*up|speed\s*up|rear\s*-?view|rear camera|backwards", re.I)
BOT = re.compile(r"sign in to confirm|not a bot|http error 429|too many requests", re.I)
MIN_S, PER_QUERY, THROTTLE_S = 420, 12, 15
BOT_PAUSE_S, BOT_RETRIES = 1800, 4

used = {c["video"].get("url", "").split("v=")[-1]
        for c in json.loads((ROOT / "benchmarks/manifest.json").read_text(encoding="utf-8"))["clips"]}
prior = {r["id"] for f in ("screen1.json", "screen2.json", "screen3.json", "screen4.json")
         for r in json.loads((ROOT / "benchmarks/test2_screen" / f).read_text(encoding="utf-8"))}
reader = RapidOcrReader()


class BotCheck(RuntimeError):
    pass


def youtube(args, timeout):
    """One throttled yt-dlp call; raises BotCheck instead of returning nothing."""
    for attempt in range(BOT_RETRIES + 1):
        time.sleep(THROTTLE_S)
        p = subprocess.run(["yt-dlp", *args], capture_output=True, timeout=timeout)
        err = p.stderr.decode("utf-8", "replace")
        if not BOT.search(err):
            return p.stdout.decode("utf-8", "replace")
        print(f"bot check (attempt {attempt + 1}); pausing {BOT_PAUSE_S} s", flush=True)
        time.sleep(BOT_PAUSE_S)
    raise BotCheck(err[-300:])


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
        url = youtube(["-g", "-f", "bv*[height<=1080][ext=mp4]/bv*[height<=1080]",
                       f"https://www.youtube.com/watch?v={vid}"], 90).strip().splitlines()[0]
    except (IndexError, subprocess.TimeoutExpired):
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


def main(out_path, workers=6):
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    out = Path(out_path)
    rows = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {(r["city"], r["query"]) for r in rows if r.get("id") is None}   # marker rows = finished queries
    seen = set(used) | prior | {r["id"] for r in rows if r.get("id")}
    for city, keys in CITIES:
        centre = geocode(city)
        name = city.split(",")[0]
        for q in (t.format(c=name) for t in QUERIES):
            if (city, q) in done:
                continue
            try:
                entries = json.loads(youtube(["--flat-playlist", "-J", f"ytsearch{PER_QUERY}:{q}"], 180)
                                     or "{}").get("entries", [])
            except ValueError:
                entries = []
            eligible = []
            for rank, e in enumerate(entries):
                vid, title, dur = e.get("id"), e.get("title") or "", e.get("duration") or 0
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                low = title.casefold()
                if dur < MIN_S or EXCLUDE.search(title) or not any(
                        re.search(rf"(?<!\w){re.escape(k)}(?!\w)", low) for k in keys):
                    continue
                eligible.append((rank, vid, title, dur))
            vids = [v for _, v, _, _ in eligible]
            thumbs = list(pool.map(thumb_fixes, vids)) if pool else [thumb_fixes(v) for v in vids]
            for (rank, vid, title, dur), tf in zip(eligible, thumbs):
                row = {"city": city, "id": vid, "title": title, "duration_s": dur, "query": q, "rank": rank,
                       "thumb_fixes": [list(x[1]) for x in tf], "section_fixes": [], "n_near": 0, "ok": False}
                if tf:
                    sf = section_fixes(vid, dur)
                    near = [f for f in sf if centre and
                            geodesic_distance_m(centre[0], centre[1], f["lat"], f["lon"]) < 40_000]
                    row.update(section_fixes=sf, n_near=len(near), ok=len(near) >= 2)
                    print(("OK  " if row["ok"] else "    ") + f"{city[:28]:28s} {vid} {dur:5.0f}s thumbs {len(tf)} "
                          f"sections {len(sf)}/{len(near)} near  {title[:60]}", flush=True)
                rows.append(row)
            # a marker row records that this query was screened (also when nothing was eligible)
            rows.append({"city": city, "id": None, "query": q, "n_results": len(entries)})
            out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"-- {city}: {sum(r.get('ok', False) for r in rows if r['city'] == city)} passing", flush=True)
    print("passing:", sum(r.get("ok", False) for r in rows), "of", sum(1 for r in rows if r.get("id")), "screened")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6)
