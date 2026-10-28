"""Explicit version selection for reproducible assisted experiments."""
from importlib import import_module


def resolve_policy(profile):
    if profile not in {"v1", "v2", "v3"}:
        raise ValueError("assisted_profile must be v1, v2 or v3")
    suffix = "" if profile == "v1" else "_" + profile
    return import_module(".assisted_policy" + suffix, __package__)
