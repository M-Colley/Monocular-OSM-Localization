"""Pure fixed 500k budget/configuration and global auxiliary calculations.

No filtering, IO, state mutation, RNG, input admission or policy selection.
The caller owns bounded pointwise work, literal full-window genealogy, final-
only spatial summary, the existing target factors, and external enforcement.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np

MAX_PARTICLES = 500_000
BUDGET_MODE = "fixed-500k-final-summary-v1"
RESAMPLING_MODE = "target-ess-half-v1"
PROFILE = MappingProxyType({
    "method": "fixed-high-particle-final-summary-budget-v1",
    "budget_mode": BUDGET_MODE, "resampling_mode": RESAMPLING_MODE,
    "particles": MAX_PARTICLES, "seed": 0, "max_steps": 1200,
    "genealogy_budget_bytes": 12 << 30,
    "exact_worst_full_genealogy_payload_bytes": 16 * MAX_PARTICLES * 1201,
    "inner_s": 1800., "outer_s": 1830.,
    "external_cpu_threads": 2, "external_rss_bytes": 16 << 30,
    "external_physical_and_true_commit_floor_bytes": 8 << 30,
    "junction_prior": "conditional-uniform-v3",
    "summary": "one genuine complete spatial summary at final stored step; no intermediate summary",
    "ESS": "PRE-GUIDE target ESS=1/sum(weights*weights), Float64; strict <0.5*P",
    "normalized_weight_sum_absolute_tolerance": 1e-12,
    "skip": "identity ancestry; preserve all PF fields/logw/RNG; no guide or inverse-guide correction",
    "auxiliary_guide": ".25+.75*rawg/(ONE global Float64 dot(weights,rawg)); no per-chunk normalization",
    "auxiliary_correction": "unchanged genuine q proportional to w*effectiveg, corrected by 1/effectiveg",
    "target_changed": False, "proposal_and_numerical_budget_control": True,
    "native_heading_variance_unchanged": True,
    "full_window_genealogy_required": True,
    "actual_runtime_or_peak_memory_guaranteed": False,
    "native_data_or_policy_admission": False,
})
PROFILE_SHA256 = hashlib.sha256(json.dumps(dict(PROFILE), sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HighParticleConfig:
    """Dedicated fixed configuration; does not widen the original Config."""
    particles: int = MAX_PARTICLES
    seed: int = 0
    genealogy_bytes: int = 12 << 30
    inner_s: float = 1800.
    junction_prior: str = "conditional-uniform-v3"

    def validate(self):
        if not (type(self.particles) is int and self.particles == MAX_PARTICLES
                and type(self.seed) is int and self.seed == 0
                and type(self.genealogy_bytes) is int and self.genealogy_bytes == 12 << 30
                and type(self.inner_s) is float and self.inner_s == 1800.
                and type(self.junction_prior) is str
                and self.junction_prior == "conditional-uniform-v3"):
            raise ValueError("fixed 500k/seed0/12GiB/1800.0s/conditional-uniform-v3 configuration required")


def _shape(weights):
    return (isinstance(weights, np.ndarray) and weights.dtype == np.float64
            and weights.ndim == 1 and 1 <= weights.size <= MAX_PARTICLES)


def _weights(weights):
    if (not np.isfinite(weights).all() or np.any(weights < 0)
            or abs(float(np.sum(weights, dtype=np.float64)) - 1.) > 1e-12):
        raise ValueError("normalized finite nonnegative Float64 target weights required")


def decision(weights):
    """Fixed pre-guide target ESS rule; no callbacks or state/RNG operations."""
    if not _shape(weights):
        raise ValueError("bounded one-dimensional Float64 target weights required")
    _weights(weights)
    ess = float(1. / np.sum(weights * weights))
    return {"ess": ess, "resample": bool(ess < .5 * weights.size),
            "particle_count": int(weights.size)}


def defensive_auxiliary_guide(weights, raw_guide):
    """One GLOBAL parent formula, including when pointwise work was chunked.

    The returned effective guide, not the raw guide, enters both the ancestor
    proposal and the genuine inverse-guide correction. Never apply this to
    initial or turn proposals, which retain their existing defenses.
    """
    if not (_shape(weights) and isinstance(raw_guide, np.ndarray)
            and raw_guide.dtype == np.float64 and raw_guide.shape == weights.shape):
        raise ValueError("same-shaped bounded Float64 weights and raw guide required")
    _weights(weights)
    if not np.isfinite(raw_guide).all() or np.any(raw_guide <= 0):
        raise ValueError("positive finite raw guide required")
    mean = float(np.dot(weights, raw_guide))
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError("positive finite global weighted guide mean required")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        effective = raw_guide / mean
        effective *= .75
        effective += .25
    if not np.isfinite(effective).all():
        raise ValueError("global auxiliary effective guide overflow")
    return effective
