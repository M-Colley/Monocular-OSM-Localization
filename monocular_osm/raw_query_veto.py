"""Pure fixed absolute raw-query veto for a distinct research target.

The inherited .162 cutoff was a posthoc DEV decoded-track median threshold,
not an independent or calibrated query inlier probability. No IO/admission,
appearance/proposal mutation, event removal or power renormalization occurs.
"""
import hashlib
import json
from types import MappingProxyType

import numpy as np


THRESHOLD = .162
MAX_SNAPSHOTS = 96_000
PROFILE = MappingProxyType({
    "method": "absolute-raw-query-veto-snapshot-v1",
    "queries": 80,
    "maximum_snapshots": MAX_SNAPSHOTS,
    "raw_input": "native C-contiguous finite Float32[80,S], 1<=S<=96000; unchanged",
    "maximum": "maximum ORIGINAL raw Float32 cosine over ALL admitted snapshots per query, then exact Float64 promotion",
    "enabled": "Float64 raw maximum >= Float64 literal 0.162; no Float32 threshold rounding",
    "threshold": THRESHOLD,
    "threshold_source": "monocular_osm/assisted_policy.py:_CONFIG.baseline_median_min",
    "threshold_origin": "inherited posthoc DEV decoded-track median cutoff; no new tuning",
    "target": "enabled[q] * original log_factor[q], then original power; all 80 events retained",
    "disabled_factor": "exactly neutral; no event dropping or power renormalization",
    "proposals": "original Appearance, initial G0, future/backward, turn and auxiliary guides and corrections unchanged",
    "query_inlier_probability": False,
    "statistically_independent_reliability": False,
    "calibrated": False,
    "descriptor_equivalence_assumed": False,
    "historical_reference_encoding_upgraded": False,
    "data_native_or_policy_admission": False,
    "input_payload_max_bytes": 80 * MAX_SNAPSHOTS * 4,
    "finite_validation_boolean_scratch_max_bytes": MAX_SNAPSHOTS,
    "target_changed": True,
})
PROFILE_SHA256 = hashlib.sha256(json.dumps(dict(PROFILE), sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def build_gate(raw):
    """Return independent ``(enabled_bool80, maxima_Float64_80, receipt)``.

    The caller must first admit every snapshot. No reference/version selection
    occurs: alias multiplicity and order cannot alter a maximum. Nonfinite
    values refuse the whole gate. Validation scratch is one row of booleans;
    no full Float64 matrix or boolean cube is made. Readonly input is allowed.
    """
    if not (isinstance(raw, np.ndarray) and raw.dtype == np.dtype(np.float32)
            and raw.dtype.isnative and raw.ndim == 2 and raw.shape[0] == 80
            and 1 <= raw.shape[1] <= MAX_SNAPSHOTS and raw.flags.c_contiguous):
        raise ValueError("native C-contiguous Float32[80,S], 1<=S<=96000 required")
    maxima = np.empty(80, dtype=np.float64)
    for q in range(80):
        row = raw[q]
        if not np.isfinite(row).all():
            raise ValueError("every original raw score must be finite")
        maxima[q] = float(np.max(row))
    enabled = maxima >= THRESHOLD
    receipt = {
        "schema": "absolute-raw-query-veto-receipt-v1",
        "status": "complete_raw_query_veto_research_gate",
        "profile_sha256": PROFILE_SHA256,
        "threshold": THRESHOLD,
        "threshold_source": PROFILE["threshold_source"],
        "query_count": 80,
        "snapshot_count": int(raw.shape[1]),
        "enabled_count": int(np.count_nonzero(enabled)),
        "disabled_count": int(np.count_nonzero(~enabled)),
        "all_queries_and_original_powers_retained": True,
        "target_only": True,
        "proposal_guides_unchanged": True,
        "query_inlier_probability": False,
        "calibrated": False,
        "data_native_or_policy_admission": False,
        "historical_reference_encoding_upgraded": False,
    }
    return enabled, maxima, receipt
