"""Explicit version selection for reproducible assisted experiments."""
from importlib import import_module


def resolve_policy(profile):
    if profile not in {"v1", "v2"}:
        raise ValueError("assisted_profile must be v1 or v2")
    return import_module(".assisted_policy" + ("_v2" if profile == "v2" else ""), __package__)
