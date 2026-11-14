"""Public POI inference preserves ambiguity, provenance and overlay rejection."""
import pytest

from monocular_osm.poi_clues import PoiClueConfig, _digest, build_poi_hypotheses


def _poi(identifier, brand, lon):
    return {"id": identifier, "lat": 50.0, "lon": lon, "tags": {"brand": brand}}


def _observations():
    return [{"text": "Example Grocery", "t_sec": 0, "confidence": .95},
            {"text": "Example Pharmacy", "t_sec": 4, "confidence": .9}]


def test_public_inference_preserves_all_ambiguous_branches_and_content_hashes():
    pois = [_poi("a", "Example Grocery", 8), _poi("b", "Example Pharmacy", 8.001),
            _poi("c", "Example Grocery", 8.1), _poi("d", "Example Pharmacy", 8.101)]
    detections = _observations()
    result = build_poi_hypotheses(detections, {"city": "Example", "pois": pois})
    assert [c["poi_ids"] for c in result["candidate_clusters"]] == [["a", "b"], ["c", "d"]]
    assert result["selected_answer"] is None and result["confidence_calibrated"] is False
    assert result["inventory"]["snapshot_sha256"] == _digest(pois)
    assert result["detections_sha256"] == _digest(detections)
    assert "osm_ids" not in pois[0]


def test_public_boundary_rejects_position_stamps_before_matching():
    detections = _observations() + [{"text": "N 50.00000 E 8.00000", "t_sec": 0, "confidence": 1.0}]
    pois = [_poi("a", "Example Grocery", 8), _poi("b", "Example Pharmacy", 8.001),
            _poi("bad", "N 50.00000 E 8.00000", 8)]
    result = build_poi_hypotheses(detections, {"pois": pois})
    assert result["n_position_stamps_rejected"] == 1
    assert all(m["poi_id"] != "bad" for m in result["matched_observations"])


def test_declared_inventory_digest_is_validated():
    with pytest.raises(ValueError, match="fingerprint changed"):
        build_poi_hypotheses([], {"pois": [], "snapshot_sha256": "old"})


def test_deduplication_does_not_change_existing_source_aliases():
    from copy import deepcopy

    pois = [_poi("node/1", "Example Grocery", 8), _poi("way/2", "Example Grocery", 8.00001)]
    pois[0]["additional_aliases"] = [["grocery", "Grocery", "name"]]
    pois[1]["tags"]["name"] = "Other Label"
    original = deepcopy(pois)
    result = build_poi_hypotheses([], {"pois": pois})
    assert result["inventory"]["n_deduplicated_pois"] == 1
    assert pois == original


@pytest.mark.parametrize("config", [{"visibility_m": 0}, {"max_gap_s": float("inf")},
                                   {"min_confidence": 1.1}, {"max_speed_mps": -1}])
def test_nonphysical_settings_are_refused(config):
    with pytest.raises(ValueError):
        PoiClueConfig(**config)


def test_nonfinite_observation_cannot_become_a_position_hypothesis():
    detections = [{"text": "Example Grocery", "t_sec": float("nan"), "confidence": .9}]
    with pytest.raises(ValueError, match="OCR times"):
        build_poi_hypotheses(detections, {"pois": []})
