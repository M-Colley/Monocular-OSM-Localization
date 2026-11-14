"""Explicit version selection for reproducible assisted experiments."""
from importlib import import_module


def resolve_policy(profile):
    if profile not in {"v1", "v2", "v3", "v4"}:
        raise ValueError("assisted_profile must be v1, v2, v3 or v4")
    suffix = "" if profile == "v1" else "_" + profile
    return import_module(".assisted_policy" + suffix, __package__)


#: Profiles that share v3's metadata-union expansion and its proofs (v4 adds a
#: sign-site stage in front of it and changes nothing else).
V3_FAMILY = frozenset({"v3", "v4"})
