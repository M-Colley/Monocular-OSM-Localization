"""Fixed dimensionless raw-VPR compatibility for a full-road research posterior.

The caller authenticates the complete admitted bank/native facts. No IO, model,
map acquisition, GT, calibrated confidence or policy admission occurs here.
"""
from dataclasses import dataclass
import math
from types import MappingProxyType
import numpy as np
from pyproj import CRS
from . import poi_alignment as poi
from . import raw_vpr_bank as bank_core
from . import raw_vpr_native as native

PROFILE = MappingProxyType({
    "method": "raw-vpr-full-road-appearance-compatibility-v1", "reference_mean": False,
    "field_cell_m": 100., "appearance_sigma_m": 150., "row_std_floor": .005,
    "z_clip": 6., "empty_appearance": -6., "extent_margin_m": 450.,
    "extent": "ALL directed-road polyline vertices and ALL admitted reference positions",
    "max_field_values": 16_000_000, "field_dtype": "float32", "max_field_bytes": 64_000_000,
    "factor": "log(.1+.9*exp(A_q(x)-max_over_complete_grid(A_q)))",
    "outlier_floor": .1, "group_max_span_s": 30., "group_power": "1/full_group_count",
    "query_rows": 80, "max_composition_lag_s": 1., "spatial_crs": "absolute net.crs",
    "reference_encoding_identity": "UNVERIFIED", "dimensionless_compatibility": True,
    "calibrated_confidence": False, "policy_admission": False,
    "raw_bank_profile_sha256": bank_core.PROFILE_SHA256,
})
PROFILE_SHA256 = poi._json_digest(dict(PROFILE))


def need(value, message):
    if not value:
        raise ValueError(message)


def temporal_groups(times):
    """Fixed maximum span from FIRST row; every exact time retained once."""
    t = np.asarray(times)
    need(t.dtype == np.float64 and t.shape == (80,) and np.isfinite(t).all()
         and np.all(np.diff(t) > 0), "exact ordered native80 appearance times required")
    groups, powers = np.empty(80, np.int64), np.empty(80, np.float64)
    lo, group = 0, 0
    while lo < 80:
        hi = lo + 1
        while hi < 80 and t[hi] - t[lo] <= 30.:
            hi += 1
        groups[lo:hi], powers[lo:hi] = group, 1. / (hi - lo)
        lo, group = hi, group + 1
    return groups, powers


def full_extent(net, reference_xy):
    """Include terminal endpoints too; seg_xy0 alone omits some dead ends."""
    crs = CRS.from_user_input(net.crs)
    need(crs.is_projected and len(crs.axis_info) >= 2
         and all(a.unit_conversion_factor == 1. for a in crs.axis_info[:2]), "projected metre net.crs required")
    refs = np.asarray(reference_xy)
    need(refs.ndim == 2 and refs.shape[1] == 2 and len(refs) and np.isfinite(refs).all()
         and net.seg_xy0.ndim == 2 and net.seg_xy0.shape[1] == 2 and len(net.seg_xy0)
         and np.isfinite(net.seg_xy0).all(), "finite complete reference/road geometry required")
    x, y, _, _ = net.locate(np.arange(net.n_edges), net.edge_len)
    endpoints = net.local_to_crs(np.column_stack([x, y]))
    need(np.isfinite(endpoints).all(), "finite complete road endpoints required")
    low = np.minimum(refs.min(axis=0), np.minimum(net.seg_xy0.min(axis=0) + net.origin, endpoints.min(axis=0)))
    high = np.maximum(refs.max(axis=0), np.maximum(net.seg_xy0.max(axis=0) + net.origin, endpoints.max(axis=0)))
    origin = np.floor((low - 450.) / 100.) * 100.
    upper = np.ceil((high + 450.) / 100.) * 100.
    dimensions = [int(math.ceil((upper[i] - origin[i]) / 100.)) + 1 for i in (1, 0)]
    values = 80 * math.prod(dimensions)
    need(min(dimensions) >= 2 and values <= 16_000_000,
         "whole full-road/reference appearance field exceeds 16M F32 values; no crop")
    return origin, tuple(dimensions), {"full_geometry_min_xy": low.tolist(), "full_geometry_max_xy": high.tolist(),
                                      "grid_upper_xy": upper.tolist(), "field_values": values, "field_bytes": 4 * values}


def verify_schedule(bank, facts, native_times, composed_times):
    records = [value["source"] for value in bank["inputs"].values()]
    flat = native.validate_native_facts(facts, records, bank["bank_key"])
    need(facts["signed_vo_clock_sha256"] == native.typed(native_times), "appearance proof belongs to another native VO clock")
    times, indices = bank["query_times"], bank["query_indices"]
    need(native.typed(times) == flat["query_times_header"]["sha256"]
         and native.typed(indices) == flat["query_indices_header"]["sha256"], "admitted bank and native query bytes differ")
    identity = bank["appearance_identity"]
    expected_flat = {"backbone": {k: v for k, v in identity["model"].items() if k != "weights_sha256"},
        "actual_model_state_sha256": identity["model"]["weights_sha256"],
        "preprocessing": identity["query"]["preprocessing"], "query_mode": identity["query"]["mode"],
        "query_pixels_sha256": identity["query"]["pixels_sha256"],
        "query_descriptors_sha256": identity["query"]["descriptors_sha256"],
        "query_indices_header": {"shape": list(indices.shape), "dtype": indices.dtype.str, "sha256": native.typed(indices)},
        "query_times_header": {"shape": list(times.shape), "dtype": times.dtype.str, "sha256": native.typed(times)}}
    need(flat == expected_flat and bank["appearance_identity_sha256"] == poi._json_digest(identity),
         "admitted bank model/query identity differs from signed native appearance")
    expected = native.query_vo_join(times, native_times, 1.)
    composed = np.asarray(composed_times)
    need(composed.dtype == np.float64 and composed.ndim == 1 and len(composed) >= 2
         and np.isfinite(composed).all() and np.all(np.diff(composed) > 0), "actual coarsen_motion endpoint clock required")
    steps = np.searchsorted(composed, times, side="left")
    need(np.all(steps < len(composed)), "appearance queries outside composed horizon")
    for i, row in enumerate(expected):
        step = int(steps[i])
        need(row["composed_step"] == step and row["composed_t_sec"] == float(composed[step])
             and 0 <= composed[step] - times[i] <= 1., "native proof and coarsen_motion step/time disagree")
    names = set(facts["bank_membership"][bank["bank_key"]])
    for physical in facts["all_physical_calls_and_aliases"]:
        if physical["key"] not in names:
            continue
        for i, (actual, wanted) in enumerate(zip(physical["query_to_vo"], expected)):
            need(isinstance(actual, dict) and all(type(actual.get(k)) is type(v) and actual[k] == v for k, v in wanted.items())
                 and type(actual.get("query_row")) is int and actual["query_row"] == i
                 and actual.get("local_query_index") == int(indices[i]), "signed query-to-VO time/bracket/lag differs")
    return steps.astype(np.int64), flat


@dataclass(frozen=True)
class Appearance:
    field: poi.RasterField
    maxima: np.ndarray
    groups: np.ndarray
    powers: np.ndarray
    steps: np.ndarray
    receipt: dict

    def log_factor(self, query, net, local_xy, *, time):
        need(type(query) is int and 0 <= query < 80 and time == self.field.query_times[query],
             "appearance must sample exact original query PTS")
        local = np.asarray(local_xy)
        need(local.ndim == 2 and local.shape[1] == 2 and np.isfinite(local).all(), "finite local PF positions required")
        absolute = net.local_to_crs(local)
        values = self.field.sample(absolute[:, None, :], np.array([time], np.float64))[:, 0]
        return np.logaddexp(math.log(.1), math.log(.9) + values - self.maxima[query])

    def initial_weight(self, net, edge, offset):
        x, y, _, _ = net.locate(edge, offset)
        log_factor = self.log_factor(0, net, np.column_stack([x, y]), time=self.field.query_times[0])
        return np.exp(self.powers[0] * log_factor)


def prepare(bank, net, native_facts, native_times, composed_times, *, check=lambda: None):
    """Build genuine RAW field after complete bank/clock/whole-extent guards."""
    check()
    need(bank.get("status") == "complete_research_bank" and bank.get("profile") == dict(bank_core.PROFILE)
         and bank.get("profile_sha256") == bank_core.PROFILE_SHA256
         and bank.get("reference_encoding_identity") == "UNVERIFIED"
         and bank.get("query_observations_per_time") == 1 and bank.get("policy_admission") is False,
         "complete genuine admitted research bank required")
    n = len(bank["ref_ids"])
    scores, coordinates, times = bank["raw_similarities"], bank["ref_latlon"], bank["query_times"]
    need(1 <= n <= 10_000 and len(set(bank["ref_ids"].tolist())) == n
         and coordinates.shape == (n, 2) and coordinates.dtype == np.float64 and np.isfinite(coordinates).all()
         and scores.shape == (80, n) and scores.dtype == np.float32 and scores.flags.c_contiguous
         and np.isfinite(scores).all() and np.max(abs(scores)) <= 1.0001, "full bounded raw bank arrays required")
    need(bank["physical_calls_validated"] == len(bank["inputs"]), "bank physical membership differs")
    steps, flat = verify_schedule(bank, native_facts, native_times, composed_times)
    groups, powers = temporal_groups(times)
    reference_xy = net.local_to_crs(net.latlon_to_local(coordinates))
    origin, shape, extent = full_extent(net, reference_xy)
    check()
    config = poi.POIAlignmentConfig()
    field = poi.build_appearance_field(reference_xy, scores, times, origin_xy=origin, shape=shape,
        reference_mean=False, config=config, provenance={"appearance_profile_sha256": PROFILE_SHA256,
        "bank_profile_sha256": bank["profile_sha256"], "appearance_identity": bank["appearance_identity"],
        "native_clock_sha256": native_facts["signed_vo_clock_sha256"], "crs": net.crs})
    need(field.values.dtype == np.float32 and field.values.size <= 16_000_000, "genuine F32 field bound differs")
    maxima = field.values.max(axis=(1, 2)).astype(np.float64)
    receipt = {"profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256, "crs": net.crs,
               "field": field.source_record(), **extent, "all_references": n,
               "all_physical_calls": bank["physical_calls_validated"],
               "query_times": times.tolist(), "group_ids": groups.tolist(), "group_powers": powers.tolist(),
               "scheduled_steps": steps.tolist(), "all_queries_once": True,
               "row_maxima_over_complete_grid": maxima.tolist(), "raw_mode": True,
               "model_and_reference_encoding_verified": False}
    check()
    return Appearance(field, maxima, groups, powers, steps, receipt)
