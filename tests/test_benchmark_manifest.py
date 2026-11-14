"""The committed benchmark definition (benchmarks/manifest.json)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

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
        assert c["video"] == {
            "url": f"https://www.youtube.com/watch?v={e['video_id']}",
            "local_path": e["video"],
        }
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
        assert c["split"] in ("dev", "test", "test2", "test3", "test4")
    assert "test" in MANIFEST["splits"]
    # Every clip that existed while the method was tuned is dev; the held-out
    # clips (added 2026-10-24, before anything ran on them) are the only test ones.
    held_out = {"overlay_nsONSyk9wtc", "overlay_bF5IfC2ZHjQ", "overlay_5b-tgjtwb2U",
                "overlay_zt8rHvHKBUU", "overlay_4LvkBo69jZo", "overlay_iJ2JvNk4Rmk",
                "overlay_-1xwoetuoOg"}
    assert {c["id"] for c in CLIPS if c["split"] == "test"} == held_out


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
            city_word = c["city"].split(",")[0].split()[0].casefold()
            title_names = {"bucharest": ("bucharest", "bucuresti", "bucurești")}
            assert any(name in c["title"].casefold()
                       for name in title_names.get(city_word, (city_word,)))


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


def test_overlay_clip_split_is_a_declared_field_defaulting_to_dev() -> None:
    """A held-out clip must be declared as such in the clip list (one place,
    before any result exists), never promoted by editing the manifest."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "overlay_clips_under_test",
        Path(__file__).resolve().parents[1] / "scripts" / "overlay_clips.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod          # @dataclass looks its module up here
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    assert {c.split for c in mod.CLIPS} <= {"dev", "test", "test2", "test3", "test4"}
    # The original fleet (used while the method was tuned) stays dev; held-out
    # clips added later are test.
    original = {"vCSEG6KaFng", "LmIHvLMLqFk", "wJsEQTCAg1c", "ZhGb8q1kliY", "1nF_7l07i-E",
                "g5lnpYCk1Ec", "y66ZkRpUh4k", "kxEDNj5L_yQ", "Wrn4_uCxRCQ"}
    assert all(c.split == "dev" for c in mod.CLIPS if c.video_id in original)
    held = mod.OverlayClip("abc", "bottom", (0, 600), "stamp", "note", split="test")
    assert held.split == "test"


@pytest.mark.parametrize("local_path", [
    "data/new-window/input_0-600.mp4",
    "E:/registered-inputs/clip/input_0-420.mkv",
])
def test_overlay_manifest_preserves_exact_registered_local_video(tmp_path, monkeypatch, local_path):
    """The local bytes and original URL are complementary identities.

    Synthetic index/labels only: do not regenerate or inspect real GT here.
    """
    index_path = tmp_path / M.OVERLAY_INDEX
    index_path.parent.mkdir(parents=True)
    index_path.touch()
    entry = {
        "video_id": "vCSEG6KaFng", "video": local_path,
        "ground_truth": "ground_truth/toy.json", "city": "Chicago, Illinois, USA",
        "vo_segment": "0:420", "title": "Chicago toy video",
    }
    toy_gt = {"waypoints": [{"t_sec": 0, "lat": 0, "lon": 0}]}
    monkeypatch.setattr(M, "_read_json", lambda path: (
        {"clips": [entry]} if path == index_path else toy_gt))
    [actual] = M._overlay_entries(tmp_path)
    assert actual["video"] == {
        "url": "https://www.youtube.com/watch?v=vCSEG6KaFng",
        "local_path": local_path,
    }
    assert actual["vo_segment"] == "0:420"
    assert actual["title"] == entry["title"]


def test_incremental_overlay_entries_do_not_open_historical_labels(tmp_path, monkeypatch):
    index_path = tmp_path / M.OVERLAY_INDEX
    index_path.parent.mkdir(parents=True)
    index_path.touch()
    selected = {
        "video_id": "vCSEG6KaFng", "video": "data/toy/input_0-600.mp4",
        "ground_truth": "ground_truth/new-toy.json", "city": "Chicago, Illinois, USA",
        "vo_segment": "0:420", "title": "Chicago toy video",
    }
    old = {"video_id": "LmIHvLMLqFk", "ground_truth": "ground_truth/old-forbidden.json"}
    opened = []

    def read(path):
        opened.append(path)
        if path == index_path:
            return {"clips": [old, selected]}
        assert path == tmp_path / selected["ground_truth"], "historical labels must stay closed"
        return {"waypoints": [{"t_sec": 0, "lat": 0, "lon": 0}]}

    monkeypatch.setattr(M, "_read_json", read)
    [entry] = M._overlay_entries(tmp_path, video_ids={"vCSEG6KaFng"})
    assert entry["id"] == "overlay_vCSEG6KaFng"
    assert opened == [index_path, tmp_path / selected["ground_truth"]]


def test_test4_manifest_matches_closed_data_qc_cohort() -> None:
    """Data eligibility is frozen before any test4 localization outcome."""
    here = ROOT / "benchmarks" / "test4_screen"
    selection = json.loads((here / "selection_preregistration.json").read_text(encoding="utf-8"))
    outcome = json.loads((here / "selection_outcome.json").read_text(encoding="utf-8"))
    availability = json.loads((here / "availability.json").read_text(encoding="utf-8"))
    assert selection["target"] == 7
    assert outcome["target_met"] is False
    assert outcome["status"] == "FIXED_POOL_EXHAUSTED_SIX_USABLE_TARGET_SEVEN_NOT_MET"
    assert outcome["localization_executed"] is False
    assert outcome["method_binding_status"] == "PENDING_SEPARATE_GLOBAL_METHOD_FREEZE"
    cohort = {"overlay_" + x["id"] for x in availability["candidates"] if x["available"]}
    actual = [c for c in CLIPS if c["split"] == "test4"]
    assert len(actual) == len(cohort) == 6
    assert {c["id"] for c in actual} == cohort
    assert [c["id"].removeprefix("overlay_") for c in actual] == outcome["eligible_video_ids"]
    assert "overlay_oPbnvEIRm3A" not in {c["id"] for c in CLIPS}
    assert {c["id"] for c in CLIPS if c["split"] == "dev" and c["available_locally"]} == {
        "ulm4k_0-300", "ulm2", "berlin_0-420", "london", "kitti_0009", "kitti_0033",
        "overlay_vCSEG6KaFng", "overlay_LmIHvLMLqFk", "overlay_wJsEQTCAg1c",
        "overlay_ZhGb8q1kliY", "overlay_1nF_7l07i-E", "overlay_g5lnpYCk1Ec",
        "overlay_y66ZkRpUh4k", "overlay_kxEDNj5L_yQ", "overlay_Wrn4_uCxRCQ",
    }
    for c in actual:
        assert c["available_locally"] is True
        assert c["video"]["local_path"].startswith("data/test4/")
        assert not Path(c["video"]["local_path"]).is_absolute()
