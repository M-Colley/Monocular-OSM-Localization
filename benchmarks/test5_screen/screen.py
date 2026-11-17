"""Fixed test5 road-video screen; no localization or scoring.

The test4 V3 screen (benchmarks/test4_screen/screen_replacements_v3.py) with
the test5 city pool read from selection_preregistration.json. Same title
eligibility module, same test3 screen5 thumbnail/three-frame OCR helpers,
same 40 km rule. Exit codes: 0 a pass to build, 75 retry later
(bot check or unread frames), 2 pool exhausted.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "benchmarks" / "test4_screen"))
import title_eligibility_v3 as title_v3  # noqa: E402

ROAD = title_v3.ROAD
EXCLUDE = title_v3.EXCLUDE
NONROAD = re.compile(r"\bflying\b|\bflight\b|\batc\b|aircraft|airplane|aeroplane|helicopter|drone|cockpit|\btrain\b|\btram\b|railway|\bwalk(?:ing)?\b|cycling|bicycle", re.I)
PRIVATE = ROOT / "output" / "test5_data_acquisition_20261117" / "screen_evidence"
RESULTS = HERE / "screen_results.json"


def selection() -> dict:
    raw = (HERE / "selection_preregistration.json").read_bytes()
    receipt = json.loads((HERE / "preregistration_receipt.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(raw).hexdigest() == receipt["selection_sha256"], "selection changed after registration"
    return json.loads(raw)


def matched_key(title: str, keys: list[list[str]]) -> list[str] | None:
    folded = title.casefold()
    for key, display in keys:
        if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", folded):
            return [key, display]
    return None


def registered_city(city: str, display: str) -> str:
    return display + city[city.index(","):]


def load_screen5():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import psutil
    proc = psutil.Process()
    proc.cpu_affinity(proc.cpu_affinity()[:2])
    spec = importlib.util.spec_from_file_location("test5_screen5", ROOT / "benchmarks/test3_screen/screen5.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    s.cv2.setNumThreads(2)
    s.reader = s.RapidOcrReader(params={
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })
    return s


#: Operational amendment 01 (operational_amendment_01.json): seekable HTTPS
#: streams first. screen5 asks for "bv*[height<=1080][ext=mp4]/...", which
#: with yt-dlp 2026.10.26 can resolve to an HLS (m3u8) stream that OpenCV
#: cannot seek, so every frame read times out. Same frames, same criterion.
SECTION_FORMAT = ("bv*[height<=1080][ext=mp4][protocol=https]/bv*[height<=1080][protocol=https]/"
                  "bv*[height<=1080][ext=mp4]/bv*[height<=1080]")


def section_fixes(s, vid, dur):
    """screen5.section_fixes with SECTION_FORMAT; also counts frames actually read.

    Returns ``(fixes, n_frames_read)``. Amendment 02: screen5 marks a stamped
    thumbnail with no parsed frame stamp UNVERIFIED whether or not the frames
    were read, so a video whose frames read but show no parseable stamp would
    be retried forever. Only unread frames are unverified.
    """
    import shutil
    import subprocess as _sp
    out, n_read = [], 0
    # Amendment 03: without a JavaScript runtime yt-dlp warns that "some
    # formats may be missing"; the seekable HTTPS stream then sometimes is.
    js = ["--js-runtimes", "node"] if shutil.which("node") else []
    try:
        url = s.youtube([*js, "-g", "-f", SECTION_FORMAT, f"https://www.youtube.com/watch?v={vid}"],
                        90).strip().splitlines()[0]
    except (IndexError, _sp.TimeoutExpired):
        return out, n_read
    cap = s.cv2.VideoCapture(url, s.cv2.CAP_FFMPEG)
    for frac in (0.25, 0.5, 0.75):
        t = int(dur * frac)
        cap.set(s.cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, img = cap.read()
        n_read += bool(ok and img is not None)
        got = s.read_stamp(img) if ok else None
        if got:
            out.append({"t": t, "region": got[0], "lat": got[1][0], "lon": got[1][1]})
    cap.release()
    return out, n_read


def screen_one(s, centre, vid, duration) -> dict:
    thumbs = s.thumb_fixes(vid)
    fixes, n_frames_read = section_fixes(s, vid, duration) if thumbs else ([], 0)
    near = [f for f in fixes if s.geodesic_distance_m(centre[0], centre[1], f["lat"], f["lon"]) < 40000]
    row = {"thumbnail_stamp": bool(thumbs), "n_frames_read": n_frames_read,
           "n_sections_read": len(fixes), "n_near": len(near),
           "ok": len(near) >= 2, "region": fixes[0]["region"] if fixes else None,
           "status": "PASS" if len(near) >= 2 else (
               "UNVERIFIED_FRAMES" if thumbs and not n_frames_read else "SCREEN_NOT_PASSED")}
    evidence = {"thumbs": thumbs, "fixes": fixes}
    row["screen_evidence_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    n = 1
    while (PRIVATE / f"{vid}_{n:02d}.json").exists():
        n += 1
    (PRIVATE / f"{vid}_{n:02d}.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return row


def save(rows):
    RESULTS.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    sel = selection()
    rows = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else []
    s = load_screen5()
    # An unread frame is retried unchanged before the screen advances.
    resolved = {r["id"] for r in rows if r.get("id") and r.get("status") != "UNVERIFIED_FRAMES"}
    for r in [r for r in rows if r.get("status") == "UNVERIFIED_FRAMES" and r["id"] not in resolved]:
        centre = s.geocode(r["city"])
        if centre is None:
            raise RuntimeError("named-city geocode unavailable; screen not advanced")
        retry = {**{k: r[k] for k in ("city", "id", "title", "duration_s", "query", "rank", "eligible",
                                      "matched_key", "registered_city")},
                 **screen_one(s, centre, r["id"], r["duration_s"]), "retry_of_unverified": True}
        rows.append(retry)
        save(rows)
        print(json.dumps(retry, ensure_ascii=True), flush=True)
        return 0 if retry["ok"] else (75 if retry["status"] == "UNVERIFIED_FRAMES" else 1)
    done = {(r["city"], r["query"]) for r in rows if r.get("query_complete")}
    manifest = json.loads((ROOT / "benchmarks/manifest.json").read_text(encoding="utf-8"))["clips"]
    seen = {c["video"].get("url", "").split("v=")[-1] for c in manifest} | {r["id"] for r in rows if r.get("id")}
    snapshots = HERE / "queries"
    snapshots.mkdir(exist_ok=True)
    for entry in sel["city_order"]:
        city, keys = entry["city"], entry["title_keys"]
        word = city.split(",")[0].split()[0]
        centre = None
        for template in sel["queries"]:
            query = template.format(city=word)
            if (city, query) in done:
                continue
            snapshot = snapshots / (hashlib.sha256(query.encode()).hexdigest()[:16] + ".json")
            if not snapshot.exists():
                time.sleep(15)
                p = subprocess.run(["yt-dlp", "--flat-playlist", "-J", f"ytsearch{sel['results_per_query']}:{query}"],
                                   capture_output=True, timeout=180)
                if p.returncode:
                    err = p.stderr.decode("utf-8", "replace")
                    rows.append({"city": city, "query": query, "id": None, "status": "UNVERIFIED_INFRASTRUCTURE",
                                 "bot_check": bool(s.BOT.search(err)), "query_complete": False,
                                 "utc": dt.datetime.now(dt.timezone.utc).isoformat()})
                    save(rows)
                    return 75
                # Freeze the search order before inspecting any candidate.
                with snapshot.open("x", encoding="utf-8") as f:
                    json.dump(json.loads(p.stdout.decode("utf-8")), f, ensure_ascii=False, indent=2)
            entries = json.loads(snapshot.read_text(encoding="utf-8")).get("entries", [])
            for rank, e in enumerate(entries):
                vid, title, duration = e.get("id"), e.get("title") or "", e.get("duration") or 0
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                key = matched_key(title, keys)
                eligible = bool(duration >= 420 and not EXCLUDE.search(title) and not NONROAD.search(title)
                                and ROAD.search(title) and key)
                row = {"city": city, "id": vid, "title": title, "duration_s": duration, "query": query,
                       "rank": rank, "eligible": eligible, "ok": False,
                       "matched_key": key, "registered_city": registered_city(city, key[1]) if key else None,
                       "eligibility_implementation": title_v3.VERSION}
                if eligible:
                    if centre is None:
                        centre = s.geocode(city)
                        if centre is None:
                            raise RuntimeError("named-city geocode unavailable; screen not advanced")
                    row.update(screen_one(s, centre, vid, duration))
                else:
                    row["status"] = "TITLE_OR_DURATION_INELIGIBLE"
                rows.append(row)
                save(rows)
                print(json.dumps(row, ensure_ascii=True), flush=True)
                if row["ok"]:
                    return 0
                if row.get("status") == "UNVERIFIED_FRAMES":
                    return 75
            rows.append({"city": city, "query": query, "id": None, "query_complete": True, "n_results": len(entries)})
            save(rows)
    print("fixed test5 pool exhausted", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
