"""Independent streaming/alias probes using private in-memory pools only."""
from copy import deepcopy
import gc
import weakref

import numpy as np
import pytest

from monocular_osm import raw_vpr_bank as bank
from tests.test_raw_vpr_bank import fixture, reseal, run


def test_last_physical_alias_is_semantically_checked_after_duplicate_id_seen():
    keys = ("baseline/00-original", "baseline/10-alias", "baseline/20-alias")
    case = fixture(keys=keys, names={key: ["mapillary:0"] for key in keys})
    last = case[3][keys[-1]]
    # Valid shape and freshly bound typed bytes, but an invalid original
    # returned control. Deduplication must not skip this physical alias.
    last["arrays"]["returned_cosines"][17] += .001
    reseal(case[1][-1], last)
    seen = []

    def load(record):
        seen.append(record["key"])
        return case[3][record["key"]]

    result = run(case, loader=load)
    assert seen == list(keys)
    assert result["status"] == "UNAVAILABLE"
    assert "original regional raw controls differ" in result["reason"]
    assert not {"raw_similarities", "ref_ids", "mapping"}.intersection(result)


@pytest.mark.parametrize("delta, accepted", [(3.75e-7, True), (7.5e-7, False)])
def test_three_occurrence_envelope_uses_all_extrema_not_only_canonical(delta, accepted):
    keys = ("baseline/00-middle", "baseline/10-high", "baseline/20-low")
    case = fixture(keys=keys, names={key: ["mapillary:0"] for key in keys})
    represented = np.asarray([.2, .2 + delta, .2 - delta], np.float32)
    # Both later observations are individually close to the canonical row.
    assert all(abs(float(x) - float(represented[0])) <= 1e-6 for x in represented)
    scalar_spread = max(map(float, represented)) - min(map(float, represented))
    assert (scalar_spread <= 1e-6) is accepted
    for i, key in enumerate(keys):
        region = case[3][key]
        arrays = region["arrays"]
        arrays["raw_similarities"][:] = represented[i]
        arrays["decoded_cosines"][:] = represented[i]
        arrays["returned_cosines"][:] = represented[i]
        reseal(case[1][i], region)
    result = run(case)
    assert (result["status"] == "complete_research_bank") is accepted
    if accepted:
        assert result["mapping"][0]["maximum_raw_score_spread"] == scalar_spread
        assert len(result["mapping"][0]["occurrences"]) == 3
        assert result["query_observations_per_time"] == 1
        assert np.array_equal(result["raw_similarities"][:, 0], np.full(80, represented[0], np.float32))
    else:
        assert "duplicate raw-score envelope conflict" in result["reason"]
        assert "raw_similarities" not in result


def test_loaded_descriptor_pool_and_array_views_are_released_before_next_callback():
    keys = ("baseline/00", "baseline/01", "baseline/02")
    case = fixture(keys=keys, names={key: ["mapillary:0"] for key in keys}, descriptors=False)
    previous = []
    seen = []

    def load(record):
        gc.collect()
        assert all(ref() is None for ref in previous), "a previous loaded pool is still retained"
        region = deepcopy(case[3][record["key"]])
        rows = np.zeros((1, 8448), np.float32)
        rows[0, 0] = 1.
        region["descriptor_rows"] = rows
        region["audit"]["ref_embeddings_sha256"] = bank.tensor_sha(rows)
        previous.extend(weakref.ref(array) for array in (
            rows, region["arrays"]["raw_similarities"], region["arrays"]["ref_latlon"]))
        seen.append(record["key"])
        return region

    result = run(case, loader=load)
    assert result["status"] == "complete_research_bank"
    assert seen == list(keys)
    gc.collect()
    assert all(ref() is None for ref in previous)
    assert result["physical_calls_validated"] == result["reference_occurrences"] == 3
