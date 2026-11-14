"""Pure fixed initial-proposal bins; no IO, RNG or latent replacement.

Caller draws genuine logk/eps, then uses genuine initial_proposal with
log(p_cell)-log(q_cell|bin). No speed/bin-mass ratio or target change.
"""
import hashlib
import json
import math
from types import MappingProxyType
import numpy as np

MODE = "speed-conditioned-initial-v1"
MAX_PARTICLES = 50_000
MAX_CELLS = 200_000
SCRATCH_BYTES = 64 << 20
SPEED_REPRESENTATIVES = tuple(float(x) for x in np.geomspace(.5, 55., 16))
LOG_SPEED_REPRESENTATIVES = tuple(float(x) for x in np.log(SPEED_REPRESENTATIVES))
LOG_SPEED_BOUNDARIES = tuple(float(x) for x in
    .5 * (np.asarray(LOG_SPEED_REPRESENTATIVES[:-1]) + np.asarray(LOG_SPEED_REPRESENTATIVES[1:])))
PROFILE = MappingProxyType({
    "method": MODE, "bins": 16, "representatives_mps": SPEED_REPRESENTATIVES,
    "representatives_construction": "Float64 NumPy geomspace(0.5,55.0,16), endpoints included",
    "log_speed_boundaries": LOG_SPEED_BOUNDARIES,
    "boundary_construction": "Float64 arithmetic midpoint of log(adjacent representative speeds)",
    "assignment": "searchsorted(log boundaries,logk+eps-log(unit_seconds),side=right); ties higher; total exterior tails",
    "maximum_particles": MAX_PARTICLES, "maximum_full_partition_cells": MAX_CELLS,
    "maximum_initial_guide_points": 16 * MAX_CELLS,
    "initial_scratch_bytes": SCRATCH_BYTES, "active_bin_layers": 1,
    "prior": "genuine clipped-normal logk and independent eps; all draws/atoms/tails preserved",
    "conditional_cell_proposal": ".25*p+.75*p*g_bin/sum(p*g_bin), genuine initial_proposal; uniform within cell",
    "initial_log_weight": "log(p_cell)-log(q_cell|bin); no speed/bin ratio or per-bin normalization",
    "guide": "same full100m partition; same EXIT/future at log(v_rep*unit_seconds) times SAME G0; one bin at a time",
    "rng": "none here; caller new draw order explicit; original entries unchanged",
    "latent_draws_replaced_or_clipped": False, "target_changed": False,
    "proposal_only": True, "native_data_or_policy_admission": False,
    "new_accuracy_guaranteed": False,
})
PROFILE_SHA256 = hashlib.sha256(json.dumps(dict(PROFILE), sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def bin_indices(logk, eps, unit_seconds):
    """Return independent int32[P] bins; readonly/strided F64 draws are valid.

    Finite tail draws map to bins 0/15 without clipping. Float64 combined-log
    arithmetic overflow refuses explicitly, rather than changing latent draws.
    """
    if not (isinstance(logk, np.ndarray) and isinstance(eps, np.ndarray)
            and logk.dtype == np.float64 and eps.dtype == np.float64
            and logk.ndim == eps.ndim == 1 and logk.shape == eps.shape
            and 1 <= logk.size <= MAX_PARTICLES):
        raise ValueError("same-shape bounded Float64 latent vectors, 1<=P<=50000 required")
    if (isinstance(unit_seconds, (bool, np.bool_))
            or not isinstance(unit_seconds, (int, float, np.integer, np.floating))):
        raise ValueError("finite positive unit_seconds required")
    try:
        unit = float(unit_seconds)
    except (OverflowError, ValueError) as error:
        raise ValueError("finite positive unit_seconds required") from error
    if not math.isfinite(unit) or unit <= 0:
        raise ValueError("finite positive unit_seconds required")
    if not np.isfinite(logk).all() or not np.isfinite(eps).all():
        raise ValueError("finite unchanged latent draws required")
    with np.errstate(over="ignore", invalid="ignore"):
        log_speed = logk + eps - math.log(unit)
    if not np.isfinite(log_speed).all():
        raise ValueError("combined log speed overflow; no latent clipping permitted")
    return np.searchsorted(LOG_SPEED_BOUNDARIES, log_speed, side="right").astype(np.int32)
