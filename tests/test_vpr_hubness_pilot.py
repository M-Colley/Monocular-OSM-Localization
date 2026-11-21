"""Hubness formulas operate only on saved similarities, with fixed parameters."""

import numpy as np
import pytest

from scripts.pilot_vpr_hubness import correct_similarities


def test_raw_control_is_unchanged_and_reference_means_are_removed():
    sims = np.array([[0.7, 0.3], [0.8, 0.6], [0.9, 0.3]], dtype=np.float32)
    np.testing.assert_array_equal(correct_similarities(sims, "raw"), sims)
    centered = correct_similarities(sims, "reference_mean")
    np.testing.assert_allclose(centered.mean(axis=0), 0, atol=1e-7)
    # A generally popular reference loses its generic advantage.
    assert centered[1].argmax() == 1 and sims[1].argmax() == 0


def test_csls_clamps_top_k_for_a_small_matrix_and_keeps_cosine_scale():
    sims = np.array([[0.7, 0.3], [0.8, 0.6], [0.9, 0.3]], dtype=np.float32)
    expected = sims - 0.5 * sims.mean(axis=0) - 0.5 * sims.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(correct_similarities(sims, "csls10"), expected)


def test_nonfinite_input_is_rejected_before_decoding():
    with pytest.raises(ValueError, match="finite matrix"):
        correct_similarities([[0.2, np.nan]], "reference_mean")
