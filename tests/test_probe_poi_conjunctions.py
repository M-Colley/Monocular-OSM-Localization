"""POI hypotheses retain branch ambiguity and obey visibility/time constraints."""
import importlib.util
import json
from pathlib import Path
import sys


spec = importlib.util.spec_from_file_location("probe_poi", Path(__file__).resolve().parents[1] /
                                              "scripts" / "probe_poi_conjunctions.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def _poi(identifier, brand, east_m):
    frame = probe.LocalFrame(48.4, 10.0)
    lat, lon = frame.to_latlon(east_m, 0)
    return {"id": identifier, "lat": lat, "lon": lon, "tags": {"brand": brand, "shop": "yes"}}


def _detection(text, t=0, confidence=0.95):
    return {"text": text, "t_sec": t, "confidence": confidence, "bbox": [0, 0, 10, 10]}


def test_all_feasible_branches_are_retained_without_arbitrary_selection():
    pois = [_poi("a1", "Brand One", 0), _poi("b1", "Brand Two", 80),
            _poi("a2", "Brand One", 1000), _poi("b2", "Brand Two", 1080)]
    matches = probe.match_literal_clues([_detection("Brand One"), _detection("Brand Two")], pois)
    clusters = probe.candidate_conjunctions(matches, pois)
    assert {tuple(c["poi_ids"]) for c in clusters} == {("a1", "b1"), ("a2", "b2")}


def test_visibility_and_time_bounds_control_feasibility():
    pois = [_poi("a", "Brand One", 0), _poi("b", "Brand Two", 350)]
    simultaneous = probe.match_literal_clues([_detection("Brand One"), _detection("Brand Two")], pois)
    assert probe.candidate_conjunctions(simultaneous, pois) == []
    consecutive = probe.match_literal_clues([_detection("Brand One"), _detection("Brand Two", t=10)], pois)
    assert len(probe.candidate_conjunctions(consecutive, pois)) == 1
    distant_time = probe.match_literal_clues([_detection("Brand One"), _detection("Brand Two", t=30)], pois)
    assert probe.candidate_conjunctions(distant_time, pois) == []


def test_same_chain_or_low_confidence_is_not_independent_corroboration():
    pois = [_poi("a", "Brand One", 0), _poi("b", "Brand One", 20)]
    matches = probe.match_literal_clues([_detection("Brand One")], pois)
    assert probe.candidate_conjunctions(matches, pois) == []
    assert probe.match_literal_clues([_detection("Brand One", confidence=0.5)], pois) == []


def test_short_brands_need_exact_full_literal_matches():
    pois = [_poi("a", "XY", 0)]
    assert len(probe.match_literal_clues([_detection("XY")], pois)) == 1
    assert probe.match_literal_clues([_detection("XYthing")], pois) == []


def test_one_literal_matching_two_osm_aliases_is_still_one_clue():
    first, second = _poi("a", "Chain Coffee", 0), _poi("b", "Chain", 20)
    first["tags"]["name"] = "Chain"
    matches = probe.match_literal_clues([_detection("Chain")], [first, second])
    assert len(matches) == 2
    assert probe.candidate_conjunctions(matches, [first, second]) == []


def test_nearby_node_way_duplicates_preserve_source_ids_without_inflating_branches():
    pois = [_poi("node/1", "Brand One", 0), _poi("way/2", "Brand One", 10),
            _poi("node/3", "Brand One", 1000)]
    unique = probe.deduplicate_pois(pois)
    assert len(unique) == 2
    assert unique[0]["osm_ids"] == ["node/1", "way/2"]


def test_missing_inventory_is_explicit_and_reused_failure_does_not_erase_other_city_results(
        tmp_path, monkeypatch):
    clues, output = tmp_path / "clues", tmp_path / "output"
    clues.mkdir()
    for identifier, city in [("a", "City A"), ("b", "City A"), ("c", "City B")]:
        (clues / f"{identifier}.json").write_text(json.dumps(
            {"id": identifier, "city": city, "detections": []}))
    calls = []

    def fetch(city, directory, **kwargs):
        calls.append(city)
        if city == "City A":
            raise FileNotFoundError("no cached inventory")
        return {"pois": [_poi("a", "Brand One", 0)], "snapshot_sha256": "snapshot"}

    monkeypatch.setattr(probe, "fetch_city_retail", fetch)
    monkeypatch.setattr(sys, "argv", ["probe", "--clues-dir", str(clues), "--out", str(output), "--offline"])
    assert probe.main() == 1
    assert calls == ["City A", "City B"]
    summary = json.loads((output / "summary.json").read_text())
    assert [c["status"] for c in summary["clips"]] == ["retail_inventory_unavailable",
                                                       "retail_inventory_unavailable", "ok"]
    assert json.loads((output / "a.json").read_text())["candidate_clusters"] is None
