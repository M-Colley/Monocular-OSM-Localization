"""Fixed test4 road-video replacement screen; no localization or scoring.

Same thumbnail/three-frame OCR test as screen5. Only the preregistered city
order and explicit ground-motor-vehicle title eligibility are new. Frames
are sought in memory; only 0-600 s is downloaded for a full GT build.
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
CITIES = [("Trondheim, Norway", ["trondheim"]),
          ("Tampere, Finland", ["tampere"]),
          ("Rennes, France", ["rennes"]),
          ("Grenoble, France", ["grenoble"]),
          ("Dresden, Germany", ["dresden"])]
import title_eligibility_v3 as title_v3
ROAD = title_v3.ROAD
NONROAD = re.compile(r"\bflying\b|\bflight\b|\batc\b|aircraft|airplane|aeroplane|helicopter|drone|cockpit|\btrain\b|\btram\b|railway|\bwalk(?:ing)?\b|cycling|bicycle", re.I)


def main() -> int:
    profile_path = HERE / "metadata_eligibility_v3_registration.json"
    if len(sys.argv) != 2:
        raise ValueError("expected external title-profile SHA256")
    expected_profile = sys.argv[1]
    profile_bytes = profile_path.read_bytes()
    assert hashlib.sha256(profile_bytes).hexdigest() == expected_profile
    profile = json.loads(profile_bytes)
    assert profile["version"] == title_v3.VERSION
    assert profile["new_road_regex"] == title_v3.ROAD_PATTERN
    assert profile["eligibility_module_sha256"] == hashlib.sha256((HERE / "title_eligibility_v3.py").read_bytes()).hexdigest()
    assert profile["screen_helper_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import psutil
    proc = psutil.Process()
    proc.cpu_affinity(proc.cpu_affinity()[:2])
    source = ROOT / "benchmarks/test3_screen/screen5.py"
    spec = importlib.util.spec_from_file_location("test4_original_screen5", source)
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    s.cv2.setNumThreads(2)
    # Operational CPU-only session sizing; all OCR numerics/settings stay default.
    s.reader = s.RapidOcrReader(params={
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })
    # The original screen's direct seeks are reused at their true origin.
    out = HERE / "screen_results.json"
    rows = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {(r["city"], r["query"]) for r in rows if r.get("query_complete")}
    used = {c["video"].get("url", "").split("v=")[-1]
            for c in json.loads((ROOT / "benchmarks/manifest.json").read_text(encoding="utf-8"))["clips"]}
    used.update(json.loads((HERE / "selection_preregistration.json").read_text(encoding="utf-8"))["fixed_initial_order"])
    seen = used | {r["id"] for r in rows if r.get("id")}
    # A failed stream read is not permission to silently advance on resume.
    if any(r.get("status") == "UNVERIFIED_FRAMES" for r in rows):
        raise RuntimeError("unverified frame read remains pending; retry that exact candidate before advancing")
    # Replay only metadata implementation errors. Old rows stay byte-for-byte
    # represented in the pre-V3 snapshot; a separate correction ledger links
    # each superseded metadata verdict. Completed stamp/GT verdicts stay final.
    for correction in profile["metadata_corrections"]:
        vid = correction["id"]
        if any(r.get("id") == vid and r.get("eligible") is True for r in rows):
            continue
        original = correction["original_record"]
        assert original["status"] == "TITLE_OR_DURATION_INELIGIBLE" and not original["eligible"]
        assert hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest() == correction["original_record_sha256"]
        city = original["city"]
        centre = s.geocode(city)
        if centre is None:
            raise RuntimeError("named-city geocode unavailable; correction not advanced")
        row = {**original, "eligible": True, "ok": False,
               "supersedes_metadata_sha256": correction["original_record_sha256"],
               "eligibility_implementation": title_v3.VERSION,
               "title_profile_sha256": expected_profile}
        thumbs = s.thumb_fixes(vid)
        fixes = s.section_fixes(vid, original["duration_s"]) if thumbs else []
        near = [f for f in fixes if s.geodesic_distance_m(centre[0], centre[1], f["lat"], f["lon"]) < 40000]
        row.update(thumbnail_stamp=bool(thumbs), n_sections_read=len(fixes),
                   n_near=len(near), ok=len(near) >= 2,
                   region=fixes[0]["region"] if fixes else None,
                   status="PASS" if len(near) >= 2 else ("UNVERIFIED_FRAMES" if thumbs and not fixes else "SCREEN_NOT_PASSED"))
        evidence = {"thumbs": thumbs, "fixes": fixes}
        row["screen_evidence_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
        private = ROOT / "output/test4_data_acquisition_20261110/screen_evidence"
        private.mkdir(parents=True, exist_ok=True)
        with (private / f"{vid}.json").open("x", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)
        rows.append(row)
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(row, ensure_ascii=True), flush=True)
        if row["ok"]:
            return 0
        if row["status"] == "UNVERIFIED_FRAMES":
            return 75
    for city, keys in CITIES:
        centre = s.geocode(city)
        if centre is None:
            raise RuntimeError("named-city geocode unavailable; screen not advanced")
        for template in ("{city} dashcam GPS", "{city} dashcam VIOFO", "{city} dashcam drive"):
            query = template.format(city=city.split(",")[0])
            if (city, query) in done:
                continue
            snapshots = HERE / "queries"
            snapshots.mkdir(exist_ok=True)
            snapshot = snapshots / (hashlib.sha256(query.encode()).hexdigest()[:16] + ".json")
            if not snapshot.exists():
                time.sleep(15)
                p = subprocess.run(["yt-dlp", "--flat-playlist", "-J", f"ytsearch12:{query}"],
                                   capture_output=True, timeout=180)
                error = p.stderr.decode("utf-8", "replace")
                if p.returncode:
                    bot = bool(s.BOT.search(error))
                    failure = {"city": city, "query": query, "id": None,
                               "status": "UNVERIFIED_INFRASTRUCTURE", "bot_check": bot,
                               "query_complete": False, "utc": dt.datetime.now(dt.timezone.utc).isoformat()}
                    if bot:
                        failure["retry_not_before_utc"] = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=30)).isoformat()
                    rows.append(failure)
                    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    print(json.dumps(failure), flush=True)
                    return 75
                payload = json.loads(p.stdout.decode("utf-8"))
                # Freeze search order before inspecting any of its candidates.
                with snapshot.open("x", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            entries = json.loads(snapshot.read_text(encoding="utf-8")).get("entries", [])
            for rank, e in enumerate(entries):
                vid, title, duration = e.get("id"), e.get("title") or "", e.get("duration") or 0
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                eligible = duration >= 420 and not s.EXCLUDE.search(title) and not NONROAD.search(title) and ROAD.search(title) and any(
                    re.search(rf"(?<!\w){re.escape(k)}(?!\w)", title.casefold()) for k in keys)
                row = {"city": city, "id": vid, "title": title, "duration_s": duration,
                       "query": query, "rank": rank, "eligible": bool(eligible), "ok": False,
                       "eligibility_implementation": title_v3.VERSION, "title_profile_sha256": expected_profile}
                if eligible:
                    thumbs = s.thumb_fixes(vid)
                    fixes = s.section_fixes(vid, duration) if thumbs else []
                    near = [f for f in fixes if s.geodesic_distance_m(centre[0], centre[1], f["lat"], f["lon"]) < 40000]
                    row.update(thumbnail_stamp=bool(thumbs), n_sections_read=len(fixes),
                               n_near=len(near), ok=len(near) >= 2,
                               region=fixes[0]["region"] if fixes else None,
                               status="PASS" if len(near) >= 2 else ("UNVERIFIED_FRAMES" if thumbs and not fixes else "SCREEN_NOT_PASSED"))
                    # Do not put new coordinate labels into algorithm-visible records.
                    row["screen_evidence_sha256"] = hashlib.sha256(json.dumps(
                        {"thumbs": thumbs, "fixes": fixes}, sort_keys=True).encode()).hexdigest()
                    private = ROOT / "output/test4_data_acquisition_20261110/screen_evidence"
                    private.mkdir(parents=True, exist_ok=True)
                    with (private / f"{vid}.json").open("x", encoding="utf-8") as f:
                        json.dump({"thumbs": thumbs, "fixes": fixes}, f, indent=2)
                else:
                    row["status"] = "TITLE_OR_DURATION_INELIGIBLE"
                rows.append(row)
                out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                print(json.dumps(row, ensure_ascii=True), flush=True)
                if row["ok"]:
                    # Pause at the first pass for the unchanged GT builder.
                    # A definitive failed build permits a resumed fixed-order search.
                    return 0
                if row.get("status") == "UNVERIFIED_FRAMES":
                    return 75
            rows.append({"city": city, "query": query, "id": None,
                         "query_complete": True, "n_results": len(entries)})
            out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("fixed replacement pool exhausted", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
