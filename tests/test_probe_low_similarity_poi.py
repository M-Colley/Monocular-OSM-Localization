"""Input-only selection/provenance checks; fake assets and no OCR/network."""
from copy import deepcopy
import json

import pytest

from scripts import probe_low_similarity_poi as probe


def inputs(median=.16199, kept="pass-2", rounded=.162):
    return {"vpr_prior": {"track": [[0, 1, 2, rounded], [1, 2, 3, rounded]]},
            "vpr_coarse_to_fine": {"kept": kept, "pass1_sim": .8, "pass2_sim": median}}


def test_selection_uses_unrounded_retained_pass_despite_rounded_threshold():
    assert probe.select_from_inputs(inputs())["selected"] is True
    at_cutoff = probe.select_from_inputs(inputs(.162))
    assert at_cutoff["selected"] is False
    assert at_cutoff["median_source"] == "vpr_coarse_to_fine.pass2_sim"


def test_kept_first_pass_not_discarded_second_pass_controls_selection():
    value = inputs(.1, "pass-1", .8)
    result = probe.select_from_inputs(value)
    assert result["median_unrounded"] == .8 and result["selected"] is False


@pytest.mark.parametrize("bad", [.1, .3, float("nan"), True])
def test_stale_or_malformed_raw_summary_cannot_select(bad):
    with pytest.raises(ValueError, match="stale or inconsistent"):
        probe.select_from_inputs(inputs(bad))


def test_missing_raw_value_can_prove_high_score_exclusion_but_never_omit_uncertain_clip():
    value = inputs(rounded=.2678)
    value.pop("vpr_coarse_to_fine")
    result = probe.select_from_inputs(value)
    assert result["median_unrounded"] is None and result["selected"] is False
    value["vpr_prior"]["track"][0][3] = .162
    value["vpr_prior"]["track"][1][3] = .162
    with pytest.raises(ValueError, match="potentially eligible"):
        probe.select_from_inputs(value)


def test_allowlist_excludes_labels_scores_answers_and_manual_prior_payloads():
    value = {**inputs(), "scores": {"answer": 0}, "ground_truth": {"path": "never.json"},
             "answer": {"route_latlon": [[11, 22]]},
             "provenance": {"config": {"city": "Example", "vo_start_sec": 0, "vo_end_sec": 10,
                                         "osm_around": [11, 22, 3], "ground_truth_path": "never.json"}}}
    selected = probe.project_inputs(json.dumps(value).encode())
    assert set(selected) == {"vpr_prior", "vpr_coarse_to_fine", "provenance"}
    assert selected["provenance"]["config"] == {"city": "Example", "vo_start_sec": 0, "vo_end_sec": 10}
    assert "ground_truth" not in json.dumps(selected)
    assert probe.select_from_inputs(selected)["selected"]


def test_clips_names_and_accuracy_fields_cannot_change_input_only_membership():
    a = inputs()
    b = {**deepcopy(a), "clip": "known-good", "scores": {"symmetric_mean_m": .001}}
    assert probe.select_from_inputs(a) == probe.select_from_inputs(b)


def inventory():
    pois = [{"id": "node/1", "lat": 1, "lon": 2, "tags": {"name": "Example"}}]
    return {"schema": 1, "city": "Example City", "scope": "named_city_bbox", "pois": pois,
            "bbox_wsen": [1, 0, 3, 2], "query": 'nwr["shop"](0,1,2,3);out center tags;',
            "response_sha256": "a" * 64, "snapshot_sha256": probe.json_sha(pois)}


@pytest.mark.parametrize("field,value", [("city", "Other"), ("scope", "per_clip"),
    ("response_sha256", ""), ("snapshot_sha256", "b" * 64), ("bbox_wsen", [3, 2, 1, 0])])
def test_inventory_rejects_wrong_or_stale_public_city_identity(field, value):
    value_to_check = inventory()
    probe.validate_inventory(value_to_check, "Example City")
    value_to_check[field] = value
    with pytest.raises(ValueError, match="identity/scope/content"):
        probe.validate_inventory(value_to_check, "Example City")


def test_copy_verifies_immutable_source_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"cached model bytes")
    target = tmp_path / "isolated/model"
    original = source.read_bytes()
    with pytest.raises(ValueError, match="hash changed"):
        probe.checked_copy(source, target, "0" * 64)
    result = probe.checked_copy(source, target, probe.sha(source), len(original))
    assert target.read_bytes() == source.read_bytes() == original
    assert result["sha256"] == probe.sha(target)
    with pytest.raises(ValueError, match="new target"):
        probe.checked_copy(source, target)


def test_torch_import_guard_refuses_torch_without_importing_it():
    guard = probe.NoTorch()
    assert guard.find_spec("numpy") is None
    with pytest.raises(RuntimeError, match="Torch is forbidden"):
        guard.find_spec("torch")
    with pytest.raises(RuntimeError, match="Torch is forbidden"):
        guard.find_spec("torch.nn")
