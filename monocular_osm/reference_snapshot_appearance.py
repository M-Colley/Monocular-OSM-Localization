"""Provider-normalized alternative-snapshot appearance research field.

All original snapshots remain alternatives. Neither descriptor equivalence nor
a common historical encoder is asserted. The caller authenticates the separate
snapshot evidence and native proof; this module performs no IO or inference.
"""
from types import MappingProxyType
import math
import numpy as np
from . import raw_vpr_appearance as strict
from . import reference_snapshot_evidence as snapshot


PROFILE = MappingProxyType({
    **dict(strict.PROFILE),
    "method": "provider-normalized-reference-snapshot-appearance-v1",
    "reference_snapshot_profile_sha256": snapshot.PROFILE_SHA256,
    "raw_bank_profile_sha256": None,
    "maximum_providers": 30000, "maximum_snapshots": 96000,
    "maximum_raw_F32_values": 7680000,
    "snapshots": "ALL distinct provider/actual-descriptor-row/raw-column alternatives",
    "snapshot_weights": "uniform 1/K per provider; total provider weight1",
    "weighted_row_statistics": "Float64 population mean/std, one query scratch; std floor .005",
    "weighted_field": "log Gaussian(sum(weight*exp(z))) / Gaussian(sum(weight))",
    "all_one_weights": "genuine unchanged poi_alignment.build_appearance_field, raw mode",
    "weighted_histogram_dtype": "float64", "field_dtype": "float32",
    "descriptor_equivalence_claimed": False, "historical_encoder_verified": False,
    "strict_appearance_math_profile_sha256": strict.PROFILE_SHA256,
})
PROFILE_SHA256 = strict.poi._json_digest(dict(PROFILE))


def _weighted_values(scores, refs, weights, origin, shape, check):
    """Bounded F64 spatial sums with a single reusable snapshot-row scratch."""
    from scipy.ndimage import gaussian_filter
    strict.need(len(shape) == 2 and min(shape) >= 2 and 80 * math.prod(shape) <= 16_000_000,
                "whole snapshot field exceeds 16M F32 values; no crop")
    indices = np.rint((refs - origin) / 100.).astype(np.int64)
    strict.need(np.all(indices >= 0) and np.all(indices[:, 0] < shape[1])
                and np.all(indices[:, 1] < shape[0]), "every snapshot must lie inside full field")
    density = np.zeros(shape, np.float64)
    np.add.at(density, (indices[:, 1], indices[:, 0]), weights)
    gaussian_filter(density, sigma=1.5, output=density, mode="constant", truncate=3.)
    supported = density > 1e-6
    values = np.full((80, *shape), -6., np.float32)
    scratch = np.empty(len(weights), np.float64)
    mass = float(np.sum(weights, dtype=np.float64))
    for q in range(80):
        check()
        np.copyto(scratch, scores[q])
        mean = float(np.dot(weights, scratch) / mass)
        scratch -= mean
        np.square(scratch, out=scratch)
        std = math.sqrt(float(np.dot(weights, scratch) / mass))
        np.copyto(scratch, scores[q])
        scratch -= mean
        scratch /= max(std, .005)
        np.clip(scratch, -6., 6., out=scratch)
        np.exp(scratch, out=scratch)
        scratch *= weights
        numerator = np.zeros(shape, np.float64)
        np.add.at(numerator, (indices[:, 1], indices[:, 0]), scratch)
        gaussian_filter(numerator, sigma=1.5, output=numerator, mode="constant", truncate=3.)
        np.divide(numerator, density, out=numerator, where=supported)
        np.maximum(numerator, 1e-8, out=numerator)
        np.log(numerator, out=numerator)
        values[q, supported] = numerator[supported]
        del numerator
    return values


def _validate(evidence):
    strict.need(evidence.get("status") == "complete_snapshot_research_evidence"
        and evidence.get("profile") == dict(snapshot.PROFILE)
        and evidence.get("profile_sha256") == snapshot.PROFILE_SHA256
        and evidence.get("reference_encoding_identity") == "UNVERIFIED"
        and evidence.get("query_observations_per_time") == 1
        and evidence.get("policy_admission") is False,
        "complete genuine snapshot alternatives required; old banks are separate")
    ids, coords = evidence["provider_ids"], evidence["ref_latlon"]
    scores, weights = evidence["raw_similarities"], evidence["snapshot_weights"]
    strict.need(isinstance(ids, np.ndarray) and ids.ndim == 1 and ids.dtype.kind in "US"
        and ids.dtype.itemsize <= 512 and 1 <= len(ids) <= 96000,
        "bounded original provider IDs required for ALL snapshots")
    count = len(ids)
    strict.need(isinstance(coords, np.ndarray) and coords.dtype == np.float64
        and coords.shape == (count, 2) and np.isfinite(coords).all()
        and np.all(abs(coords[:, 0]) <= 90) and np.all(abs(coords[:, 1]) <= 180)
        and isinstance(scores, np.ndarray) and scores.shape == (80, count)
        and scores.dtype == np.float32 and scores.flags.c_contiguous
        and np.isfinite(scores).all() and np.max(abs(scores)) <= 1.0001
        and isinstance(weights, np.ndarray) and weights.shape == (count,)
        and weights.dtype == np.float64 and np.isfinite(weights).all() and np.all(weights > 0),
        "bounded finite F32 original columns/F64 coordinates and positive weights required")
    strict.need(all(evidence.get(k) is False for k in ("descriptor_equivalence_assumed",
        "historical_encoder_verified", "old_strict_or_numeric_status_upgraded")),
        "snapshot alternatives must not claim equivalence or historical upgrade")
    providers = {}
    for col, identifier in enumerate(ids.tolist()):
        strict.need(isinstance(identifier, str) and identifier.startswith("mapillary:"),
                    "provider-qualified snapshot ID required")
        providers.setdefault(identifier, []).append(col)
    strict.need(len(providers) <= 30000 and evidence.get("snapshot_count") == count
        and evidence.get("unique_provider_count") == len(providers), "complete provider/snapshot counts differ")
    for columns in providers.values():
        canonical = coords[columns[0]]
        strict.need(all(np.array_equal(coords[col], canonical) for col in columns),
                    "provider snapshots have different exact coordinates")
        expected = np.float64(1. / len(columns))
        strict.need(all(weights[col] == expected for col in columns)
            and math.isclose(math.fsum(float(weights[col]) for col in columns), 1.,
                             rel_tol=0., abs_tol=8 * np.finfo(np.float64).eps),
            "every provider must have uniform 1/K snapshot weights summing to1")
    inputs = evidence["inputs"]
    strict.need(isinstance(inputs, dict) and 1 <= len(inputs) <= 64
        and evidence["physical_calls_validated"] == len(inputs), "snapshot physical membership differs")
    expected_occurrences = set()
    for key, row in inputs.items():
        shape = row["typed_arrays"]["ref_ids"]["shape"]
        strict.need(isinstance(shape, list) and len(shape) == 1 and type(shape[0]) is int
            and 1 <= shape[0] <= 1500, "bounded complete original pool inventory required")
        expected_occurrences.update((key, col) for col in range(shape[0]))
    mapping, keys, occurrences = evidence["mapping"], set(), set()
    strict.need(isinstance(mapping, list) and len(mapping) == count, "FULL snapshot mapping required")
    for col, row in enumerate(mapping):
        key = (row["provider_id"], row["descriptor_row_sha256"], row["raw_column_sha256"])
        strict.need(key[0] == str(ids[col]) and key not in keys
            and strict.native.is_sha(key[1]) and key[2] == strict.bank_core.legacy.full._array_sha(scores[:, col]),
            "distinct snapshot key/actual raw column differs")
        keys.add(key)
        sources = row["occurrences"]
        strict.need(isinstance(sources, list) and 1 <= len(sources) <= 96000,
                    "ALL original snapshot occurrences required")
        actual = []
        for source in sources:
            pair = (source["key"], source["column"])
            strict.need(type(pair[1]) is int and pair in expected_occurrences and pair not in occurrences,
                        "snapshot source occurrence absent, repeated or outside original pool")
            occurrences.add(pair)
            actual.append(pair)
        canonical = (row["canonical"]["key"], row["canonical"]["column"])
        strict.need(canonical == min(actual), "snapshot canonical source must remain lexical")
    strict.need(occurrences == expected_occurrences and evidence["reference_occurrences"] == len(occurrences),
                "FULL original occurrence inventory differs")
    return len(providers)


def prepare_snapshot(evidence, net, native_facts, native_times, composed_times, *, check=lambda: None):
    """Build the complete provider-normalized field under a separate data type."""
    check()
    providers = _validate(evidence)
    steps, _ = strict.verify_schedule(evidence, native_facts, native_times, composed_times)
    times = evidence["query_times"]
    groups, powers = strict.temporal_groups(times)
    coords, scores, weights = (evidence[k] for k in ("ref_latlon", "raw_similarities", "snapshot_weights"))
    refs = net.local_to_crs(net.latlon_to_local(coords))
    origin, shape, extent = strict.full_extent(net, refs)
    provenance = {"appearance_profile_sha256": PROFILE_SHA256,
        "reference_snapshot_profile_sha256": evidence["profile_sha256"],
        "appearance_identity": evidence["appearance_identity"],
        "native_clock_sha256": native_facts["signed_vo_clock_sha256"], "crs": net.crs,
        "snapshot_weights_sha256": strict.native.typed(weights),
        "descriptor_equivalence_assumed": False, "historical_encoder_verified": False}
    check()
    all_one = bool(np.all(weights == 1.))
    if all_one:
        # Deliberately preserve genuine F32 row-statistic/histogram rounding.
        field = strict.poi.build_appearance_field(refs, scores, times, origin_xy=origin,
            shape=shape, reference_mean=False, config=strict.poi.POIAlignmentConfig(), provenance=provenance)
    else:
        values = _weighted_values(scores, refs, weights, origin, shape, check)
        source = {"kind": "all-snapshot provider-normalized Gaussian log-mean-exp",
            "snapshot_count": len(weights), "unique_provider_count": providers,
            "reference_xy_sha256": strict.poi._digest(refs),
            "similarities_sha256": strict.poi._digest(scores), "reference_mean": False,
            "config": strict.poi.POIAlignmentConfig().to_dict(), "caller": provenance}
        field = strict.poi.RasterField(values, tuple(origin), 100., -6., times, source)
    strict.need(field.values.dtype == np.float32 and field.values.size <= 16_000_000,
                "complete snapshot field exceeds F32 bound")
    maxima = field.values.max(axis=(1, 2)).astype(np.float64)
    receipt = {"profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256, "crs": net.crs,
        "field": field.source_record(), **extent, "all_references": len(weights),
        "snapshot_count": len(weights), "unique_provider_count": providers,
        "all_physical_calls": evidence["physical_calls_validated"],
        "query_times": times.tolist(), "group_ids": groups.tolist(), "group_powers": powers.tolist(),
        "scheduled_steps": steps.tolist(), "all_queries_once": True,
        "row_maxima_over_complete_grid": maxima.tolist(), "raw_mode": True,
        "all_one_weights_used_genuine_original_builder": all_one,
        "model_and_reference_encoding_verified": False, "descriptor_equivalence_assumed": False,
        "historical_encoder_verified": False, "snapshot_weights_sha256": strict.native.typed(weights)}
    check()
    return strict.Appearance(field, maxima, groups, powers, steps, receipt)
