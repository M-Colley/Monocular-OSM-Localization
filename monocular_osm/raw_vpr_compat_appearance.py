"""Explicit numeric-compatible input contract; unchanged raw appearance math.

The caller authenticates the newly accepted compatible bank and native facts.
No bitwise descriptor identity, historical encoder or policy claim is made.
"""
from types import MappingProxyType
import numpy as np
from . import raw_vpr_appearance as strict
from . import raw_vpr_compat_bank as compatible

PROFILE = MappingProxyType({**dict(strict.PROFILE),
    "method": "numeric-compatible-raw-vpr-full-road-appearance-v1",
    "raw_bank_profile_sha256": compatible.PROFILE_SHA256,
    "maximum_references": 30000, "compatibility_not_bitwise_identity": True,
    "historical_encoder_verified": False,
    "strict_appearance_math_profile_sha256": strict.PROFILE_SHA256})
PROFILE_SHA256 = strict.poi._json_digest(dict(PROFILE))


def prepare_compatible(bank, net, native_facts, native_times, composed_times, *, check=lambda: None):
    """All references; genuine unchanged field/factor; separate authority."""
    check()
    strict.need(bank.get("status") == "complete_compatible_research_bank"
        and bank.get("profile") == dict(compatible.PROFILE) and bank.get("profile_sha256") == compatible.PROFILE_SHA256
        and bank.get("compatibility_not_bitwise_identity") is True
        and bank.get("reference_encoding_identity") == "UNVERIFIED"
        and bank.get("query_observations_per_time") == 1 and bank.get("policy_admission") is False,
        "complete genuine numeric-compatible bank required; strict banks are separate")
    n = len(bank["ref_ids"])
    scores, coordinates, times = bank["raw_similarities"], bank["ref_latlon"], bank["query_times"]
    strict.need(1 <= n <= 30000 and len(set(bank["ref_ids"].tolist())) == n
        and coordinates.shape == (n, 2) and coordinates.dtype == np.float64 and np.isfinite(coordinates).all()
        and scores.shape == (80, n) and scores.dtype == np.float32 and scores.flags.c_contiguous
        and np.isfinite(scores).all() and np.max(abs(scores)) <= 1.0001, "full bounded compatible bank required")
    strict.need(bank["physical_calls_validated"] == len(bank["inputs"]), "compatible membership differs")
    steps, _ = strict.verify_schedule(bank, native_facts, native_times, composed_times)
    groups, powers = strict.temporal_groups(times)
    refs = net.local_to_crs(net.latlon_to_local(coordinates))
    origin, shape, extent = strict.full_extent(net, refs)
    check()
    field = strict.poi.build_appearance_field(refs, scores, times, origin_xy=origin, shape=shape,
        reference_mean=False, config=strict.poi.POIAlignmentConfig(), provenance={
            "appearance_profile_sha256": PROFILE_SHA256, "bank_profile_sha256": bank["profile_sha256"],
            "appearance_identity": bank["appearance_identity"],
            "native_clock_sha256": native_facts["signed_vo_clock_sha256"], "crs": net.crs,
            "compatibility_not_bitwise_identity": True, "historical_encoder_verified": False})
    strict.need(field.values.dtype == np.float32 and field.values.size <= 16_000_000, "genuine F32 field cap differs")
    maxima = field.values.max(axis=(1, 2)).astype(np.float64)
    receipt = {"profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256, "crs": net.crs,
        "field": field.source_record(), **extent, "all_references": n,
        "all_physical_calls": bank["physical_calls_validated"], "query_times": times.tolist(),
        "group_ids": groups.tolist(), "group_powers": powers.tolist(), "scheduled_steps": steps.tolist(),
        "all_queries_once": True, "row_maxima_over_complete_grid": maxima.tolist(), "raw_mode": True,
        "model_and_reference_encoding_verified": False, "compatibility_not_bitwise_identity": True}
    check()
    return strict.Appearance(field, maxima, groups, powers, steps, receipt)
