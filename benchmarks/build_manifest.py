"""The committed benchmark definition: builds ``benchmarks/manifest.json``.

    python benchmarks/build_manifest.py            # (re)write the manifest
    python benchmarks/build_manifest.py --check    # exit 1 if it drifted
    python benchmarks/build_manifest.py --geocode  # snapshot city centroids (network)

The manifest is DERIVED, never hand-edited: core-fleet entries come from
:data:`CORE_CLIPS` below (strings only — no coordinates are maintained here;
each clip's analysis window and GT hash are read from its GT file) and the
overlay fleet from ``ground_truth/overlay_fleet.json`` +
``scripts/overlay_clips.py``. ``tests/test_benchmark_manifest.py`` fails when
the committed file and this generator disagree.

City strings are the ones a user would type: the place the video title or
the dataset names. Strings chosen with knowledge of where the GT track lies
(Daly City, Vaughan, Scottsdale) are kept only as labelled
``city_alternatives`` for sensitivity runs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MANIFEST_PATH = Path("benchmarks") / "manifest.json"
OVERLAY_INDEX = Path("ground_truth") / "overlay_fleet.json"
GEOCODE_SNAPSHOT = Path("benchmarks") / "geocode_snapshot.json"
SCHEMA_VERSION = 1

#: Controlled vocabularies, written into the manifest so it is self-describing.
GT_SOURCES = {
    "hand-labelled": "GPS fixes typed in by a person from map look-ups of the video",
    "kitti_oxts": "KITTI raw OXTS RT3003 GPS/IMU (INS) global pose",
    "comma_ins": "comma2k19 global_pose: INS/GNSS/vision fusion (ECEF -> WGS84)",
    "consumer_gps": "consumer GPS receiver @1 Hz (no IMU fusion)",
    "applanix": "Applanix POS LV post-processed GNSS/INS poses",
    "overlay_ocr": "OCR of the GPS stamp burned into the video frame (not hand-verified)",
}
WINDOW_SOURCES = {
    "full": "the whole recording / dataset extract as distributed",
    "prefix": "the first N seconds of a longer video; N is where the hand labels end",
    "operator_selected": "a sub-sequence of a longer recording picked by an operator "
                         "who had the GT track available; treat as potentially GT-informed",
    "gt_continuous": "window boundaries cut FROM the GT track (longest cut-free run "
                     "of OCR'd GPS fixes) -- GT-derived by construction",
}
SPLITS = {
    "dev": "Every clip that existed before this benchmark: all were used to tune the "
           "pipeline, so dev numbers are optimistic by construction.",
    "test": "Reserved for FRESH clips added after the method is frozen. Empty today. "
            "`run` skips split=test unless --split test is given explicitly.",
}

#: The core fleet. Strings only — no coordinates are maintained here; the
#: analysis window (vo_segment) and GT hash are read from each GT file.
CORE_CLIPS: tuple[dict, ...] = (
    {
        "id": "ulm4k_0-300", "name": "Ulm 4K (ULl8s4qydrk), 0-300 s",
        "gt": "ground_truth/ulm_ULl8s4qydrk_0-300.json", "gt_source": "hand-labelled",
        "video": {"url": "https://www.youtube.com/watch?v=ULl8s4qydrk"},
        "window_source": "prefix",
        "window_note": "First 300 s of a longer upload. The GT file is a trimmed copy "
                       "(t<=300 s) of the 0-420 s hand labels, cut for the tile3d "
                       "cross-check; 300 s is the README row's window.",
        "city": "Ulm, Germany",
        "city_basis": "the video title names Ulm ('Ulm Germany 4K Drive')",
        "city_alternatives": [], "legacy_city": "Ulm, Germany",
        "available_locally": True,
    },
    {
        "id": "ulm2", "name": "Ulm #2 (LC9Sa--u5KE)",
        "gt": "ground_truth/ulm_LC9Sa--u5KE.json", "gt_source": "hand-labelled",
        "video": {"url": "https://www.youtube.com/watch?v=LC9Sa--u5KE"},
        "window_source": "prefix",
        "window_note": "First 500 s; the hand labels cover exactly this span.",
        "city": "Ulm, Germany",
        "city_basis": "the video title names Ulm ('Ulm DE Centre-ville dashcam 4K ...')",
        "city_alternatives": [], "legacy_city": "Ulm, Germany",
        "available_locally": True,
    },
    {
        "id": "berlin_0-420", "name": "Berlin (lBlKR2ek0w4), 0-420 s",
        "gt": "ground_truth/berlin_lBlKR2ek0w4_0-420.json", "gt_source": "hand-labelled",
        "video": {"url": "https://www.youtube.com/watch?v=lBlKR2ek0w4"},
        "window_source": "prefix",
        "window_note": "First 420 s of a 41-min crossing (the pipeline's default 7-min "
                       "window); the GT file is a trimmed copy (t<=420 s) of 26 min of "
                       "1-minute hand labels.",
        "city": "Berlin, Germany",
        "city_basis": "the video title names Berlin ('Berlin Germany 4K Driving Tour ...')",
        "city_alternatives": [], "legacy_city": "Berlin, Germany",
        "available_locally": True,
    },
    {
        "id": "london", "name": "London (T4wTL3LpLqU)",
        "gt": "ground_truth/london_T4wTL3LpLqU.json", "gt_source": "hand-labelled",
        "video": {"url": "https://www.youtube.com/watch?v=T4wTL3LpLqU"},
        "window_source": "prefix",
        "window_note": "First 295 s; the hand labels cover exactly this span.",
        "city": "London, UK",
        "city_basis": "clip curated as a central-London drive; the upload title is not "
                      "recorded in-repo (verify it names London)",
        "city_alternatives": [], "legacy_city": "London, UK",
        "available_locally": True,
    },
    {
        "id": "kitti_0009", "name": "KITTI 2011_09_26 drive 0009",
        "gt": "ground_truth/kitti_drive_0009.json", "gt_source": "kitti_oxts",
        "video": {"path": "data/kitti/drive_0009.mp4"},
        "window_source": "full",
        "window_note": "The whole synced drive (~46 s).",
        "city": "Karlsruhe, Germany",
        "city_basis": "KITTI raw documentation: all drives recorded in Karlsruhe",
        "city_alternatives": [], "legacy_city": "Karlsruhe, Germany",
        "available_locally": True,
        "acquire": "Render image_02 of the raw drive: python scripts/make_kitti_gt.py "
                   "data/kitti/raw/2011_09_26/2011_09_26_drive_0009_sync "
                   "--out-video data/kitti/drive_0009.mp4 --out-gt <scratch>.json",
    },
    {
        "id": "kitti_0033", "name": "KITTI 2011_09_30 drive 0033",
        "gt": "ground_truth/kitti_drive_0033.json", "gt_source": "kitti_oxts",
        "video": {"path": "data/kitti/drive_0033.mp4"},
        "window_source": "full",
        "window_note": "The whole synced drive (~165 s).",
        "city": "Karlsruhe, Germany",
        "city_basis": "KITTI raw documentation: all drives recorded in Karlsruhe",
        "city_alternatives": [], "legacy_city": "Karlsruhe, Germany",
        "available_locally": True,
        "acquire": "Render image_02 of the raw drive: python scripts/make_kitti_gt.py "
                   "data/kitti/raw/2011_09_30/2011_09_30_drive_0033_sync "
                   "--out-video data/kitti/drive_0033.mp4 --out-gt <scratch>.json",
    },
    {
        "id": "comma_148", "name": "comma2k19 route b0c9d2329ad1606b (segments 1-4)",
        "gt": "ground_truth/comma_148.json", "gt_source": "comma_ins",
        "video": {"path": "data/comma/route_148.mp4"},
        "window_source": "operator_selected",
        "window_note": "Four consecutive 1-minute segments of a longer commute, chosen "
                       "for trajectory shape by an operator with the pose track at hand.",
        "city": "San Francisco Peninsula, California, USA",
        "city_basis": "comma2k19 documents its data as commutes on California's I-280 "
                      "(San Jose - San Francisco); the GT file written by "
                      "monocular_osm/comma2k19.py says 'San Francisco Peninsula'",
        "city_alternatives": ["San Francisco, California, USA",
                              "Daly City, California, USA"],
        "city_alternatives_notes": {
            "San Francisco, California, USA": "metro-level name a user might type; "
                                              "its centroid is ~14 km from this route",
            "Daly City, California, USA": "GT-INFORMED: chosen for the README's "
                                          "deployable row because the route is in Daly "
                                          "City -- not a blind choice",
        },
        "legacy_city": "Daly City, California, USA",
        "available_locally": False,
        "acquire": "comma2k19 Chunk_1 via scripts/extract_comma_segments.py "
                   "(route 'b0c9d2329ad1606b|2018-08-17--14-55-39', segments 1-4) then "
                   "scripts/make_comma_gt.py",
    },
    {
        "id": "malaga_07", "name": "Malaga Urban extract 07",
        "gt": "ground_truth/malaga_extract07.json", "gt_source": "consumer_gps",
        "video": {"path": "data/malaga-urban-extract07-spain/input.mp4"},
        "window_source": "full",
        "window_note": "The whole extract (~105 s; the builder's 200 s cap is not reached).",
        "city": "Málaga, Spain",
        "city_basis": "dataset name (Malaga Urban Dataset)",
        "city_alternatives": [], "legacy_city": "Málaga, Spain",
        "available_locally": False,
        "acquire": "Malaga Urban extract 07 via scripts/build_ext_clips.py (build_malaga)",
    },
    {
        "id": "boreas_glenshields", "name": "Boreas 2020-11-26-13-58 (Glen Shields)",
        "gt": "ground_truth/boreas_glenshields.json", "gt_source": "applanix",
        "video": {"path": "data/boreas-glenshields-vaughan-canada/input.mp4"},
        "window_source": "operator_selected",
        "window_note": "Seconds 300-480 of the sequence, picked by an operator with the "
                       "Applanix track available (scripts/build_ext_clips.py).",
        "city": "Toronto, Ontario, Canada",
        "city_basis": "the Boreas dataset describes its route as driven in Toronto",
        "city_alternatives": ["Vaughan, Ontario, Canada"],
        "city_alternatives_notes": {
            "Vaughan, Ontario, Canada": "GT-INFORMED: the town the GT track lies in; "
                                        "used by the earlier README rows",
        },
        "legacy_city": "Vaughan, Ontario, Canada",
        "available_locally": False,
        "acquire": "Boreas sequence boreas-2020-11-26-13-58 via scripts/_dl_boreas.py "
                   "then scripts/build_ext_clips.py (build_boreas)",
    },
)

#: The city each overlay upload's TITLE names (first place named). The
#: overlay index's own ``city`` is reverse-geocoded from the GT track, i.e.
#: GT-derived, so it is only kept as a labelled alternative.
OVERLAY_TITLE_CITY: dict[str, str] = {
    "vCSEG6KaFng": "Chicago, Illinois, USA",
    "LmIHvLMLqFk": "Chicago, Illinois, USA",
    "wJsEQTCAg1c": "Phoenix, Arizona, USA",
    "ZhGb8q1kliY": "Detroit, Michigan, USA",
    "1nF_7l07i-E": "Detroit, Michigan, USA",
    "g5lnpYCk1Ec": "Detroit, Michigan, USA",
    "y66ZkRpUh4k": "Atlanta, Georgia, USA",
    "kxEDNj5L_yQ": "Atlanta, Georgia, USA",
    "Wrn4_uCxRCQ": "Newcastle upon Tyne, England, UK",
}


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gt_waypoints_digest(gt: dict) -> str:
    """sha256 of the waypoint list only (t, lat, lon), so description edits
    do not change it but any coordinate/timing change does."""
    rows = [[round(float(w["t_sec"]), 3), round(float(w["lat"]), 7),
             round(float(w["lon"]), 7)] for w in gt["waypoints"]]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _core_entry(spec: dict, root: Path) -> dict:
    gt = _read_json(root / spec["gt"])
    entry = {
        "id": spec["id"], "name": spec["name"], "fleet": "core", "split": "dev",
        "gt": spec["gt"], "gt_source": spec["gt_source"],
        "gt_n_waypoints": len(gt["waypoints"]),
        "gt_waypoints_sha256": gt_waypoints_digest(gt),
        "video": dict(spec["video"]),
        "vo_segment": gt["vo_segment"], "vo_segment_source": "gt_file",
        "window_source": spec["window_source"], "window_note": spec["window_note"],
        "city": spec["city"], "city_basis": spec["city_basis"],
        "city_alternatives": list(spec["city_alternatives"]),
        "city_alternatives_notes": dict(spec.get("city_alternatives_notes", {})),
        "legacy_city": spec["legacy_city"],
        "available_locally": spec["available_locally"],
    }
    if "acquire" in spec:
        entry["acquire"] = spec["acquire"]
    return entry


def _overlay_entries(root: Path) -> list[dict]:
    from overlay_clips import BY_ID  # scripts/overlay_clips.py

    index_path = root / OVERLAY_INDEX
    if not index_path.exists():
        return []
    out: list[dict] = []
    for c in _read_json(index_path).get("clips", []):
        vid = c["video_id"]
        if vid not in OVERLAY_TITLE_CITY:
            raise KeyError(f"overlay clip {vid} has no title-named city in "
                           f"OVERLAY_TITLE_CITY; add the city its title names")
        clip = BY_ID[vid]
        gt = _read_json(root / c["ground_truth"])
        city = OVERLAY_TITLE_CITY[vid]
        alts, notes = [], {}
        if c["city"] != city:
            alts.append(c["city"])
            notes[c["city"]] = ("GT-INFORMED: reverse-geocoded from the OCR'd GT track "
                                "by scripts/build_overlay_fleet.py; used by "
                                "run_all_gt.py --overlay")
        out.append({
            "id": f"overlay_{vid}",
            "name": f"{city.split(',')[0]} ({vid})",
            "fleet": "overlay", "split": clip.split,
            "gt": c["ground_truth"], "gt_source": "overlay_ocr",
            "gt_n_waypoints": len(gt["waypoints"]),
            "gt_waypoints_sha256": gt_waypoints_digest(gt),
            "video": {"url": clip.url},
            "title": c.get("title"),
            "vo_segment": c["vo_segment"], "vo_segment_source": "overlay_index",
            "window_source": "gt_continuous",
            "window_note": (f"Longest cut-free run of OCR'd GPS fixes inside the "
                            f"{int(clip.window[0])}-{int(clip.window[1])} s download "
                            f"window ({c.get('n_dropped_to_cuts', 0)} fixes dropped "
                            f"to edit cuts)."),
            "city": city,
            "city_basis": "the first place the upload title names",
            "city_alternatives": alts, "city_alternatives_notes": notes,
            "legacy_city": c["city"],
            "available_locally": True,
        })
    return out


def build_manifest(root: Path = ROOT) -> dict:
    """The benchmark definition, derived from CORE_CLIPS + the overlay index."""
    clips = [_core_entry(s, root) for s in CORE_CLIPS] + _overlay_entries(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "GPS-free localisation benchmark. Every run gets ONLY the video, the "
            "`city` string and `vo_segment`; ground truth is used for offline "
            "scoring only (scripts/benchmark.py aggregate). Core and overlay "
            "fleets are reported separately and never pooled."),
        "generated_by": "python benchmarks/build_manifest.py",
        "vocabulary": {"gt_source": GT_SOURCES, "window_source": WINDOW_SOURCES},
        "splits": SPLITS,
        "clips": clips,
    }


def manifest_text(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def load_manifest(path: Path | None = None) -> dict:
    return _read_json(path or (ROOT / MANIFEST_PATH))


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def write_manifest(root: Path = ROOT) -> Path:
    path = root / MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(manifest_text(build_manifest(root)))
    return path


def check_manifest(root: Path = ROOT) -> bool:
    """True when the committed manifest equals the generator's output
    (parsed comparison: a Windows checkout may rewrite line endings)."""
    path = root / MANIFEST_PATH
    return path.exists() and _read_json(path) == build_manifest(root)


def geocode_snapshot(root: Path = ROOT) -> Path:
    """Geocode every manifest city string the committed snapshots lack and
    record it in benchmarks/geocode_snapshot.json (needs network, one-off),
    so the city-centroid baseline is reproducible offline afterwards."""
    import importlib.metadata as md

    from monocular_osm import baselines as B

    manifest = load_manifest(root / MANIFEST_PATH)
    path = root / GEOCODE_SNAPSHOT
    snap = _read_json(path) if path.exists() else {}
    strings = sorted({s for c in manifest["clips"]
                      for s in [c["city"], *c.get("city_alternatives", []),
                                c.get("legacy_city")] if s})
    for s in strings:
        if B.city_centroid(s, root=root, allow_network=False) is not None:
            continue
        try:
            lat, lon = B._osmnx_geocode(s)
        except Exception as e:
            print(f"  FAIL {s!r}: {e}")
            continue
        snap[s] = {"lat": lat, "lon": lon, "geocoder": "osmnx.geocode (Nominatim)",
                   "osmnx": md.version("osmnx"), "retrieved_utc": _now()}
        print(f"  {s!r}: {lat:.6f}, {lon:.6f}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(sorted(snap.items())), indent=2, ensure_ascii=False) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true",
                   help="exit 1 if the committed manifest is stale")
    g.add_argument("--geocode", action="store_true",
                   help="snapshot centroids of manifest city strings (network)")
    args = p.parse_args(argv)
    if args.check:
        ok = check_manifest()
        print(f"{MANIFEST_PATH.as_posix()} is "
              + ("up to date" if ok else "STALE; run: python benchmarks/build_manifest.py"))
        return 0 if ok else 1
    if args.geocode:
        print(f"wrote {geocode_snapshot().relative_to(ROOT).as_posix()}")
        return 0
    print(f"wrote {write_manifest().relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
