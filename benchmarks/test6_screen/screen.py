"""Fixed 6-continent test6 road-video screen; never executes localization or scoring.

Reads ``selection_preregistration.json`` (verified against ``preregistration_receipt.json``),
enforces ``continent_quotas`` across Asia, Latin America, Middle East & Africa, Europe,
North America, and Oceania, runs multilingual YouTube queries, and checks GPS coordinate
stamps on thumbnails + 25/50/75 % frames within 40 km of the city geocode.

Exit codes: 0 a pass to build, 75 pause/retry (bot check or unread stream), 2 pool/quotas done.
"""
from __future__ import annotations

from collections import Counter
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from screen_worldwide import (  # noqa: E402
    is_multilingual_road_title,
    load_preregistration,
    match_worldwide_city_key,
)

PRIVATE = Path("E:/claude-scoring/test6/private_screen_evidence")
RESULTS = HERE / "screen_results.json"
BUILD_DIR = HERE / "build_attempts"
SECTION_FORMAT = (
    "bv*[height<=1080][ext=mp4][protocol=https]/bv*[height<=1080][protocol=https]/"
    "bv*[height<=1080][ext=mp4]/bv*[height<=1080]"
)


def registered_city(city: str, display: str) -> str:
    return (display + city[city.index(","):]) if "," in city else display


def completed_region_counts(sel: dict) -> Counter:
    city_to_region = {c["city"]: c["region"] for c in sel["city_order"]}
    counts: Counter = Counter()
    if not BUILD_DIR.is_dir():
        return counts
    passed_ids: set[str] = set()
    for p in sorted(BUILD_DIR.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("status") == "AVAILABLE_GT_QC_PASSED" and rec.get("id") not in passed_ids:
            passed_ids.add(rec["id"])
            reg = rec.get("continent_region") or city_to_region.get(rec.get("pool_city", ""))
            if reg:
                counts[reg] += 1
    return counts


def load_screen5():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    spec = importlib.util.spec_from_file_location("test6_screen5", ROOT / "benchmarks/test3_screen/screen5.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    s.cv2.setNumThreads(2)
    s.reader = s.RapidOcrReader(params={
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })
    return s


def section_fixes(s, vid: str, dur: float):
    out, n_read = [], 0
    js = ["--js-runtimes", "node"] if shutil.which("node") else []
    cap = None
    for _ in range(3):
        try:
            url = s.youtube([*js, "-g", "-f", SECTION_FORMAT, f"https://www.youtube.com/watch?v={vid}"],
                            90).strip().splitlines()[0]
        except (IndexError, subprocess.TimeoutExpired):
            url = None
        if url:
            cap = s.cv2.VideoCapture(url, s.cv2.CAP_FFMPEG)
            if cap.isOpened():
                break
            cap.release()
            cap = None
        time.sleep(8)
    if cap is None:
        return out, n_read
    for idx_f, frac in enumerate((0.25, 0.5, 0.75)):
        # Under the >=2-of-3 stamp rule, if the first two readable frames yielded
        # zero stamps, the third frame cannot reach 2.
        if idx_f == 2 and n_read == 2 and len(out) == 0:
            break
        t = int(dur * frac)
        cap.set(s.cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, img = cap.read()
        n_read += bool(ok and img is not None)
        got = s.read_stamp(img) if (ok and img is not None) else None
        if got:
            out.append({"t": t, "region": got[0], "lat": got[1][0], "lon": got[1][1]})
    cap.release()
    return out, n_read


def screen_one(s, centre, vid: str, duration: float, *, thumb_gate: bool = False) -> dict:
    thumbs = s.thumb_fixes(vid)
    if thumb_gate and not thumbs:
        fixes, n_frames_read = [], 0
    else:
        fixes, n_frames_read = section_fixes(s, vid, duration)
    near = [f for f in fixes if s.geodesic_distance_m(centre[0], centre[1], f["lat"], f["lon"]) < 40000]
    status = (
        "PASS" if len(near) >= 2 else
        ("FRAMES_UNREADABLE" if (not thumb_gate or thumbs) and not n_frames_read else "SCREEN_NOT_PASSED")
    )
    row = {
        "thumbnail_stamp": bool(thumbs),
        "n_frames_read": n_frames_read,
        "n_sections_read": len(fixes),
        "n_near": len(near),
        "ok": len(near) >= 2,
        "region": fixes[0]["region"] if fixes else None,
        "status": status,
    }
    evidence = {"thumbs": thumbs, "fixes": fixes}
    row["screen_evidence_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    n = 1
    while (PRIVATE / f"{vid}_{n:02d}.json").exists():
        n += 1
    (PRIVATE / f"{vid}_{n:02d}.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return row


def save(rows: list[dict]) -> None:
    RESULTS.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    assert (HERE / "preregistration_receipt.json").is_file(), "freeze preregistration_receipt.json first"
    sel = load_preregistration()
    quotas: dict[str, int] = sel["continent_quotas"]
    have_counts = completed_region_counts(sel)
    if sum(have_counts.values()) >= int(sel["target"]):
        print("target met across continents", flush=True)
        return 2

    rows = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else []
    s = load_screen5()
    resolved = {r["id"] for r in rows if r.get("id") and r.get("status") != "UNVERIFIED_FRAMES"}

    for vid in dict.fromkeys(r["id"] for r in rows if r.get("status") == "UNVERIFIED_FRAMES"
                             and r["id"] not in resolved):
        tries = [r for r in rows if r.get("id") == vid and r.get("status") == "UNVERIFIED_FRAMES"
                 and "n_frames_read" in r]
        if len(tries) >= 3:
            last = tries[-1]
            rows.append({
                **{k: last[k] for k in ("city", "continent_region", "id", "title", "duration_s",
                                        "query", "rank", "eligible", "matched_key", "registered_city")},
                "ok": False, "status": "FRAMES_UNREADABLE", "unverified_attempts": len(tries),
            })
            save(rows)
            resolved.add(vid)

    for r in [r for r in rows if r.get("status") == "UNVERIFIED_FRAMES" and r["id"] not in resolved]:
        reg_name = r.get("continent_region")
        if reg_name and have_counts[reg_name] >= quotas.get(reg_name, 0):
            continue
        centre = s.geocode(r["city"])
        if centre is None:
            return 75
        retry = {
            **{k: r[k] for k in ("city", "continent_region", "id", "title", "duration_s",
                                 "query", "rank", "eligible", "matched_key", "registered_city")},
            **screen_one(s, centre, r["id"], r["duration_s"]),
            "retry_of_unverified": True,
        }
        rows.append(retry)
        save(rows)
        print(json.dumps(retry, ensure_ascii=True), flush=True)
        return 0 if retry["ok"] else (75 if retry["status"] == "UNVERIFIED_FRAMES" else 1)

    # Operational amendment 01: inspect 1080p frames in exact recorded order on
    # eligible candidates that were previously skipped solely by the low-res
    # thumbnail pre-filter (n_frames_read == 0 and status == SCREEN_NOT_PASSED).
    frame_checked_ids = {
        r["id"] for r in rows
        if r.get("id") and (
            r.get("n_frames_read", 0) > 0
            or r.get("status") in ("PASS", "FRAMES_UNREADABLE", "UNVERIFIED_FRAMES")
        )
    }
    for r in [r for r in rows
              if r.get("eligible")
              and r.get("status") == "SCREEN_NOT_PASSED"
              and r.get("n_frames_read", 0) == 0
              and r["id"] not in frame_checked_ids]:
        reg_name = r.get("continent_region")
        if reg_name and have_counts[reg_name] >= quotas.get(reg_name, 0):
            continue
        centre = s.geocode(r["city"])
        if centre is None:
            return 75
        recheck = {
            **{k: r[k] for k in ("city", "continent_region", "id", "title", "duration_s",
                                 "query", "rank", "eligible", "matched_key", "registered_city")},
            **screen_one(s, centre, r["id"], r["duration_s"]),
            "amendment": "operational_amendment_01",
        }
        rows.append(recheck)
        save(rows)
        print(json.dumps(recheck, ensure_ascii=True), flush=True)
        if recheck["ok"]:
            return 0
        if recheck.get("status") == "UNVERIFIED_FRAMES":
            return 75
        frame_checked_ids.add(r["id"])

    done = {(r["city"], r["query"]) for r in rows if r.get("query_complete")}
    manifest = json.loads((ROOT / "benchmarks/manifest.json").read_text(encoding="utf-8"))["clips"]
    seen = {c["video"].get("url", "").split("v=")[-1] for c in manifest} | {r["id"] for r in rows if r.get("id")}
    snapshots = HERE / "queries"
    snapshots.mkdir(exist_ok=True)
    road_terms = sel["multilingual_road_terms"]

    for entry in sel["city_order"]:
        city = entry["city"]
        creg = entry["region"]
        if have_counts[creg] >= quotas.get(creg, 0):
            continue
        keys = entry["title_keys"]
        centre = None
        for query in entry["local_queries"]:
            if (city, query) in done:
                continue
            snapshot = snapshots / (hashlib.sha256(f"{city}::{query}".encode()).hexdigest()[:16] + ".json")
            if not snapshot.exists():
                time.sleep(8)
                js = ["--js-runtimes", "node"] if shutil.which("node") else []
                p = subprocess.run(
                    ["yt-dlp", *js, "--flat-playlist", "-J", f"ytsearch{sel['results_per_query']}:{query}"],
                    capture_output=True, timeout=180,
                )
                if p.returncode:
                    err = p.stderr.decode("utf-8", "replace")
                    rows.append({
                        "city": city, "continent_region": creg, "query": query, "id": None,
                        "status": "UNVERIFIED_INFRASTRUCTURE", "bot_check": bool(s.BOT.search(err)),
                        "query_complete": False, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    })
                    save(rows)
                    return 75
                with snapshot.open("x", encoding="utf-8") as f:
                    json.dump(json.loads(p.stdout.decode("utf-8")), f, ensure_ascii=False, indent=2)
            entries = json.loads(snapshot.read_text(encoding="utf-8")).get("entries", [])
            for rank, e in enumerate(entries):
                vid, title, duration = e.get("id"), e.get("title") or "", e.get("duration") or 0
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                key = match_worldwide_city_key(title, keys)
                eligible = bool(
                    duration >= 300
                    and is_multilingual_road_title(title, road_terms)
                    and key is not None
                )
                row = {
                    "city": city, "continent_region": creg, "id": vid, "title": title,
                    "duration_s": duration, "query": query, "rank": rank,
                    "eligible": eligible, "ok": False, "matched_key": key,
                    "registered_city": registered_city(city, key[1]) if key else None,
                }
                if eligible:
                    if centre is None:
                        centre = s.geocode(city)
                        if centre is None:
                            print(f"geocode unavailable for {city}", flush=True)
                            return 75
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
            rows.append({
                "city": city, "continent_region": creg, "query": query,
                "id": None, "query_complete": True, "n_results": len(entries),
            })
            save(rows)

    # Operational amendment 02: fixed hardware GPS-dashcam model queries in the
    # exact registered city_order for continents still below continent_quotas.
    hw_templates = (
        "{word} VIOFO dashcam",
        "{word} 70mai dashcam",
        "{word} Garmin dashcam",
        "{word} Nextbase dashcam",
        "{word} Rove dashcam",
    )
    for entry in sel["city_order"]:
        city = entry["city"]
        creg = entry["region"]
        if have_counts[creg] >= quotas.get(creg, 0):
            continue
        word = city.split(",")[0].strip()
        keys = entry["title_keys"]
        centre = None
        for tmpl in hw_templates:
            query = tmpl.format(word=word)
            if (city, query) in done:
                continue
            snapshot = snapshots / (hashlib.sha256(f"{city}::hw::{query}".encode()).hexdigest()[:16] + ".json")
            if not snapshot.exists():
                time.sleep(5)
                js = ["--js-runtimes", "node"] if shutil.which("node") else []
                p = subprocess.run(
                    ["yt-dlp", *js, "--flat-playlist", "-J", f"ytsearch{sel['results_per_query']}:{query}"],
                    capture_output=True, timeout=180,
                )
                if p.returncode:
                    err = p.stderr.decode("utf-8", "replace")
                    rows.append({
                        "city": city, "continent_region": creg, "query": query, "id": None,
                        "status": "UNVERIFIED_INFRASTRUCTURE", "bot_check": bool(s.BOT.search(err)),
                        "query_complete": False, "amendment": "operational_amendment_02",
                        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    })
                    save(rows)
                    return 75
                with snapshot.open("x", encoding="utf-8") as f:
                    json.dump(json.loads(p.stdout.decode("utf-8")), f, ensure_ascii=False, indent=2)
            entries = json.loads(snapshot.read_text(encoding="utf-8")).get("entries", [])
            for rank, e in enumerate(entries):
                vid, title, duration = e.get("id"), e.get("title") or "", e.get("duration") or 0
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                key = match_worldwide_city_key(title, keys)
                eligible = bool(
                    duration >= 300
                    and is_multilingual_road_title(title, road_terms)
                    and key is not None
                )
                row = {
                    "city": city, "continent_region": creg, "id": vid, "title": title,
                    "duration_s": duration, "query": query, "rank": rank,
                    "eligible": eligible, "ok": False, "matched_key": key,
                    "registered_city": registered_city(city, key[1]) if key else None,
                    "amendment": "operational_amendment_02",
                }
                if eligible:
                    if centre is None:
                        centre = s.geocode(city)
                        if centre is None:
                            return 75
                    row.update(screen_one(s, centre, vid, duration, thumb_gate=True))
                else:
                    row["status"] = "TITLE_OR_DURATION_INELIGIBLE"
                rows.append(row)
                save(rows)
                print(json.dumps(row, ensure_ascii=True), flush=True)
                if row["ok"]:
                    return 0
            rows.append({
                "city": city, "continent_region": creg, "query": query,
                "id": None, "query_complete": True, "n_results": len(entries),
                "amendment": "operational_amendment_02",
            })
            save(rows)

    print("fixed test6 worldwide pool exhausted", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
