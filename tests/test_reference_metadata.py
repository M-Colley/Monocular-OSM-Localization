import json
import numpy as np
import pytest
from monocular_osm import reference_metadata as m


def gallery(n=120):
    return np.array([f"mapillary:{i+1}" for i in range(n)]), np.column_stack([np.full(n, 42.), -83.+np.arange(n)*.0001])


def body(rows, **extra):
    return json.dumps({"data": rows, **extra}).encode()


def test_all_ids_planned_once_in_uniform_fixed_batches_without_locations_or_scores():
    ids, ll = gallery(9735)
    before = ll.copy()
    plan = m.plan_requests(ids, ll, provenance={"freeze": "abc"})
    flattened = [value for batch in plan["batches"] for value in batch["reference_ids"]]
    assert len(plan["batches"]) == 195 and len(plan["batches"][-1]["reference_ids"]) == 35
    assert len(flattened) == len(set(flattened)) == 9735 and set(flattened) == set(ids)
    assert all(batch["params"]["fields"] == ",".join(m.FIELDS) for batch in plan["batches"])
    other = m.plan_requests(ids[::-1], ll[::-1]+[1., 1.])
    assert other["batches"] == plan["batches"]
    np.testing.assert_array_equal(ll, before)
    assert plan["source_ids_sha256"] == m.array_sha(ids) and plan["selected_answer"] is None


@pytest.mark.parametrize("fault", ["too_many", "duplicate", "wrong_provider", "bad_id", "object", "huge_id", "latitude", "int_min"])
def test_invalid_gallery_or_budget_refused(fault):
    ids, ll = gallery(2)
    if fault == "too_many": ids, ll = gallery(9751)
    if fault == "duplicate": ids[1] = ids[0]
    if fault == "wrong_provider": ids = np.array(["kartaview:1", "mapillary:2"])
    if fault == "bad_id": ids = np.array(["mapillary:1?access_token=x", "mapillary:2"])
    if fault == "object": ids = ids.astype(object)
    if fault == "huge_id": ids = ids.astype("U129")
    if fault == "latitude": ll[0, 0] = 91
    if fault == "int_min": ll = np.array([[42, 0], [42, np.iinfo(np.int64).min]], np.int64)
    with pytest.raises(ValueError): m.plan_requests(ids, ll)


def test_nullable_fields_and_sequence_shape_compatibility_keep_all_images():
    ids, _ = gallery(3)
    result = m.parse_response(body([{"id": "1", "sequence": {"id": "sequence_x"}, "captured_at": 123456,
        "camera_type": "perspective", "compass_angle": 0, "computed_compass_angle": 360,
        "computed_rotation": [0, 1, 2]}, {"id": "2", "sequence": "sequence_y", "compass_angle": None}]), ids)
    assert set(result) == set(ids)
    assert result[ids[0]]["fields"]["sequence"] == "sequence_x"
    assert result[ids[1]]["fields"]["sequence"] == "sequence_y"
    assert result[ids[0]]["fields"]["compass_angle"] == result[ids[0]]["fields"]["computed_compass_angle"] == 0
    assert result[ids[1]]["field_status"]["compass_angle"] == "null"
    assert result[ids[1]]["field_status"]["camera_type"] == "absent"
    assert result[ids[2]]["image_status"] == "unknown"
    assert result[ids[2]]["reason"] == "omitted_from_successful_response"


@pytest.mark.parametrize("field,value", [("sequence", {}), ("captured_at", True), ("captured_at", 1.5),
    ("compass_angle", -1), ("computed_compass_angle", float("nan")), ("computed_rotation", [1, 2]),
    ("computed_rotation", [True, 0, 1]), ("camera_type", "")])
def test_invalid_optional_field_does_not_discard_its_reference(field, value):
    result = m.parse_response(body([{"id": "1", field: value}]), ["mapillary:1"])["mapillary:1"]
    assert result["image_status"] == "available" and result["fields"][field] is None
    assert result["field_status"][field] == "invalid"


@pytest.mark.parametrize("value", [{"data": [{"id": "2"}]}, {"data": [{"id": "1"}, {"id": "1"}]},
    {"data": [{"id": 1}]}, {"error": {"message": "bad"}}, {"data": [], "paging": {"next": "https://example.com"}}])
def test_unknown_duplicate_numeric_ids_and_pagination_fail_closed(value):
    with pytest.raises(ValueError): m.parse_response(json.dumps(value).encode(), ["mapillary:1"])


def test_summary_is_input_only_and_does_not_create_edges_from_equal_timestamps():
    ids, ll = gallery(4); plan = m.plan_requests(ids, ll)
    rows = [{"id": str(i+1), "sequence": "same", "captured_at": 100 if i < 2 else 200,
             "camera_type": "spherical" if i == 0 else "perspective"} for i in range(3)]
    records = m.parse_response(body(rows), ids)
    stats = m.summarize(plan, records)
    assert stats["reference_count"] == 4 and stats["available_images"] == 3 and stats["unknown_images"] == 1
    assert stats["repeated_sequences"] == 1 and stats["strictly_timed_sampled_edges"] == 0
    assert stats["equal_time_adjacent_pairs"] == 1 and stats["spherical_images"] == 1
    assert stats["sampled_gap_m_quantiles"] is None and stats["adjacencies_touching_tied_timestamps"] == 1
    assert stats["selected_answer"] is None and stats["query_or_GT_comparison"] is False
    with pytest.raises(ValueError): m.summarize(plan, {key: value for key, value in records.items() if key != ids[-1]})
