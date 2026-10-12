"""The committed benchmark definition (benchmarks/manifest.json)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_manifest_under_test", ROOT / "benchmarks" / "build_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_builder()
MANIFEST = json.loads((ROOT / "benchmarks" / "manifest.json").read_text(encoding="utf-8"))
CLIPS = MANIFEST["clips"]

CORE_GT = {
    "ground_truth/ulm_ULl8s4qydrk_0-300.json", "ground_truth/ulm_LC9Sa--u5KE.json",
    "ground_truth/berlin_lBlKR2ek0w4_0-420.json", "ground_truth/london_T4wTL3LpLqU.json",
    "ground_truth/kitti_drive_0009.json", "ground_truth/kitti_drive_0033.json",
    "ground_truth/comma_148.json", "ground_truth/malaga_extract07.json",
    "ground_truth/boreas_glenshields.json",
}


def test_committed_manifest_matches_its_generator() -> None:
    """The manifest is generated from CORE_CLIPS + the overlay index; a
    hand edit (or a GT/index change without regeneration) is drift."""
    # Parsed comparison: a Windows checkout (autocrlf) may rewrite line endings.
    assert M.build_manifest(ROOT) == MANIFEST


def test_ids_unique_and_fleets_valid() -> None:
    ids = [c["id"] for c in CLIPS]
    assert len(ids) == len(set(ids))
    assert {c["fleet"] for c in CLIPS} == {"core", "overlay"}
    assert {c["gt"] for c in CLIPS if c["fleet"] == "core"} == CORE_GT


def test_every_gt_file_exists_and_hash_matches() -> None:
    for c in CLIPS:
        path = ROOT / c["gt"]
        assert path.exists(), c["gt"]
        gt = json.loads(path.read_text(encoding="utf-8"))
        assert M.gt_waypoints_digest(gt) == c["gt_waypoints_sha256"], c["id"]
        assert len(gt["waypoints"]) == c["gt_n_waypoints"]


def test_every_overlay_index_entry_is_represented() -> None:
    index = json.loads((ROOT / "ground_truth" / "overlay_fleet.json").read_text(encoding="utf-8"))
    overlay = {c["id"]: c for c in CLIPS if c["fleet"] == "overlay"}
    for e in index["clips"]:
        c = overlay[f"overlay_{e['video_id']}"]
        assert c["gt"] == e["ground_truth"]
        assert c["vo_segment"] == e["vo_segment"]
        assert c["window_source"] == "gt_continuous"   # honest: cut from the GT track
        assert c["video"] == {"url": f"https://www.youtube.com/watch?v={e['video_id']}"}
    assert len(overlay) == len(index["clips"])


def test_gt_covers_exactly_the_analysis_window() -> None:
    for c in CLIPS:
        a, b = (float(x) for x in c["vo_segment"].split(":"))
        gt = json.loads((ROOT / c["gt"]).read_text(encoding="utf-8"))
        ts = [float(w["t_sec"]) for w in gt["waypoints"]]
        assert a - 2.0 <= min(ts) and max(ts) <= b + 2.0, (c["id"], c["vo_segment"], ts[0], ts[-1])


def test_vocabularies_and_splits() -> None:
    vocab = MANIFEST["vocabulary"]
    for c in CLIPS:
        assert c["gt_source"] in vocab["gt_source"], c["id"]
        assert c["window_source"] in vocab["window_source"], c["id"]
        assert c["split"] == "dev"          # every existing clip was used for tuning
    assert "test" in MANIFEST["splits"]     # documented, reserved, empty
    assert not [c for c in CLIPS if c["split"] == "test"]


def test_local_and_youtube_videos_are_well_formed() -> None:
    for c in CLIPS:
        v = c["video"]
        assert ("url" in v) != ("path" in v), c["id"]
        if "url" in v:
            assert v["url"].startswith("https://www.youtube.com/watch?v=")
        else:
            assert not Path(v["path"]).is_absolute()
    unavailable = {c["id"] for c in CLIPS if not c["available_locally"]}
    assert unavailable == {"comma_148", "malaga_07", "boreas_glenshields"}


def test_city_strings_are_not_gt_informed() -> None:
    by_id = {c["id"]: c for c in CLIPS}
    # The review's three GT-informed picks are demoted to labelled alternatives.
    assert by_id["comma_148"]["city"] == "San Francisco Peninsula, California, USA"
    assert "Daly City, California, USA" in by_id["comma_148"]["city_alternatives"]
    assert by_id["boreas_glenshields"]["city"] == "Toronto, Ontario, Canada"
    assert by_id["overlay_wJsEQTCAg1c"]["city"] == "Phoenix, Arizona, USA"
    assert "Scottsdale, Arizona, USA" in by_id["overlay_wJsEQTCAg1c"]["city_alternatives"]
    for c in CLIPS:
        for alt in c["city_alternatives"]:
            assert alt in c["city_alternatives_notes"], (c["id"], alt)
            assert alt != c["city"]
        if c["fleet"] == "overlay":
            # The city is the one the upload's title names.
            assert c["city"].split(",")[0].split()[0].casefold() in c["title"].casefold()


def test_every_city_string_has_an_offline_centroid() -> None:
    """The city-centroid baseline must be reproducible without network."""
    from monocular_osm.baselines import city_centroid

    for c in CLIPS:
        for city in [c["city"], *c["city_alternatives"], c["legacy_city"]]:
            assert city_centroid(city, allow_network=False) is not None, (c["id"], city)


def test_non_ocr_ground_truth_does_not_claim_ocr() -> None:
    for path in sorted((ROOT / "ground_truth").glob("*.json")):
        gt = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(gt, dict) or "waypoints" not in gt:
            continue
        src = gt.get("source", "")
        if src and src != "gps_overlay_ocr":
            assert "OCR" not in gt.get("description", "").replace("Not OCR", ""), path.name
