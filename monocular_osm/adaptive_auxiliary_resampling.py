"""Pure fixed target-ESS decision for conditional auxiliary resampling.

This helper never evaluates a guide, mutates a particle/log-weight, draws RNG,
or admits inputs. On a skip the caller MUST keep every particle and log-weight
unchanged and use identity ancestry. In particular, no inverse-guide factor is
introduced when no guided ancestor was sampled. On a trigger the caller keeps
the genuine existing guided resampler and its inverse-guide correction.
"""
import hashlib
import json
from types import MappingProxyType

import numpy as np


PROFILE = MappingProxyType({
    "method": "conditional-auxiliary-target-ess-half-v1",
    "maximum_particles": 50_000,
    "decision": "resample iff PRE-GUIDE normalized target ESS < 0.5 * particle_count",
    "ess": "1 / sum(weights * weights), Float64 NumPy reduction; no clipping",
    "normalized_weight_sum_absolute_tolerance": 1e-12,
    "skip": "identity ancestry; every PF field/log-weight/RNG unchanged; no guide evaluation or correction",
    "trigger": "unchanged genuine ancestor q proportional to w * effective_guide; new logw proportional to -log(effective_guide[ancestor])",
    "final_step": "caller never resamples after final stored step",
    "target_changed": False,
    "proposal_only": True,
    "native_or_data_admission": False,
    "policy_admission": False,
    "finite_particle_accuracy_guaranteed": False,
})
PROFILE_SHA256 = hashlib.sha256(json.dumps(dict(PROFILE), sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def decision(weights):
    """Return ``{ess, resample, particle_count}`` without state or callbacks.

    Weights are the current NORMALIZED TARGET weights, including all carried
    initial/turn/earlier auxiliary corrections and current target factors; they
    are not the guide-tilted ancestor distribution. The actual resampling and
    final-step decision remain the caller's responsibility.
    """
    if not (isinstance(weights, np.ndarray) and weights.dtype == np.float64
            and weights.ndim == 1 and 1 <= weights.size <= 50_000):
        raise ValueError("bounded one-dimensional Float64 target weights required")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("finite nonnegative target weights required")
    total = float(np.sum(weights, dtype=np.float64))
    if abs(total - 1.) > 1e-12:
        raise ValueError("normalized target weights required")
    ess = float(1. / np.sum(weights * weights))
    return {"ess": ess, "resample": bool(ess < .5 * weights.size),
            "particle_count": int(weights.size)}
