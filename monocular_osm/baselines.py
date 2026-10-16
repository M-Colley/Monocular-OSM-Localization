"""Baselines and ablations every benchmark row must be compared against.

Each function returns a :class:`Prediction` — a lat/lon route (or a single
point, scored by :mod:`monocular_osm.metrics` as a degenerate route) plus a
record of where it came from — or ``None`` when the input lacks what the
baseline needs (the aggregator reports that as "n/a", never as a score).

* :func:`city_centroid` — answer the geocoded centre of the ``--city``
  string and nothing else. The trivial floor: a GPS-free method that does
  not beat it has not localised anything. It beat the pipeline on 5 of 8
  core clips in the October 2026 review, so it is a standard row.
* :func:`vpr_prior_only` — the street-imagery retrieval centre alone
  (``result["vpr_prior"]``). Answers "does the trajectory add anything
  beyond image retrieval?".
* :func:`matcher_only` — the raw trajectory-to-OSM matcher pick
  (``result["matcher_position"]``), no anchor of any kind.
* :func:`pinned_only` — the matcher's candidate georeferenced by OCR / sign
  anchors (``result["pinned_position"]``), when the pipeline built one.
* :func:`pipeline_headline` — the pipeline's headline answer
  (``result["position"]``), for symmetry.
* :func:`pipeline_selective` — the pipeline when its own retrieval
  confidence clears a PRE-REGISTERED threshold
  (``benchmarks/preregistered_rules.json``), else the city centroid: does the
  pipeline know when it is wrong? Deployable (uses no ground truth).
* :func:`shortlist_oracle` — best of the top-k location hypotheses BY
  GROUND TRUTH. An oracle upper bound on what a perfect re-ranker could
  reach; it uses GT to choose, so it is never deployable and is labelled
  ``oracle=True`` everywhere it is reported.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Committed geocode snapshots, searched in order (exact-string keys).
#: data/geocode_cache.json is the project's tracked snapshot; the benchmark
#: keeps its own for city strings the project snapshot lacks.
#: Decision rules fixed before their held-out test (see that file's notes).
PREREGISTERED_RULES = Path("benchmarks") / "preregistered_rules.json"

DEFAULT_SNAPSHOTS: tuple[Path, ...] = (
    Path("data") / "geocode_cache.json",
    Path("benchmarks") / "geocode_snapshot.json",
)


@dataclass
class Prediction:
    """A baseline answer: ``route_latlon`` is Nx2 [lat, lon] (N == 1 for a
    point). ``source`` says exactly where the coordinates came from."""

    name: str
    route_latlon: list[list[float]]
    kind: str                      # "point" | "route"
    source: str
    oracle: bool = False
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# city centroid
# ---------------------------------------------------------------------------


def _read_snapshot(path: Path) -> dict:
    try:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return blob if isinstance(blob, dict) else {}


def _latlon_from_entry(entry) -> tuple[float, float] | None:
    """Snapshot values are ``[lat, lon]`` (project cache) or a dict with
    ``lat``/``lon`` (benchmark snapshot)."""
    try:
        if isinstance(entry, dict):
            lat, lon = float(entry["lat"]), float(entry["lon"])
        else:
            lat, lon = float(entry[0]), float(entry[1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if not (np.isfinite(lat) and np.isfinite(lon)
            and -90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def _osmnx_geocode(city: str) -> tuple[float, float]:
    import osmnx as ox  # heavy import: only on a snapshot miss
    lat, lon = ox.geocode(city)
    return float(lat), float(lon)


def city_centroid(
    city: str,
    *,
    snapshots: Iterable[Path] | None = None,
    root: Path = REPO_ROOT,
    allow_network: bool = True,
    geocoder: Callable[[str], tuple[float, float]] | None = None,
) -> Prediction | None:
    """The geocoded centre of the exact ``city`` string.

    Looks the string up in the committed snapshots first (exact key match,
    so a run is reproducible offline and immune to Nominatim drift), then —
    only if ``allow_network`` — asks ``osmnx.geocode`` (or ``geocoder``).
    ``source`` records which one answered; ``None`` if neither could.
    """
    snaps = list(snapshots) if snapshots is not None else list(DEFAULT_SNAPSHOTS)
    for snap in snaps:
        path = snap if Path(snap).is_absolute() else Path(root) / snap
        entry = _read_snapshot(path).get(city)
        ll = _latlon_from_entry(entry) if entry is not None else None
        if ll is not None:
            rel = Path(snap).as_posix()
            return Prediction("city_centroid", [[ll[0], ll[1]]], "point",
                              source=f"snapshot:{rel}", meta={"city": city})
    if not allow_network:
        return None
    try:
        lat, lon = (geocoder or _osmnx_geocode)(city)
    except Exception:
        return None
    ll = _latlon_from_entry([lat, lon])
    if ll is None:
        return None
    return Prediction("city_centroid", [[ll[0], ll[1]]], "point",
                      source="osmnx.geocode", meta={"city": city})


# ---------------------------------------------------------------------------
# result.json-derived ablations
# ---------------------------------------------------------------------------


def report_route(report: dict | None) -> list[list[float]] | None:
    """The route a position report actually answers with.

    OrienterNet refinement (opt-in) replaces the reported start but stores
    its refined route under ``orienternet_route_latlon``, leaving
    ``route_latlon`` coarse — so prefer the refined route when present.
    Falls back to ``start``/``end`` (or the headline lat/lon) for reports
    without a route.
    """
    if not isinstance(report, dict):
        return None
    for key in ("orienternet_route_latlon", "route_latlon"):
        route = report.get(key)
        if isinstance(route, list) and route:
            try:
                arr = np.asarray(route, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if arr.ndim == 2 and arr.shape[1] == 2 and np.isfinite(arr).all():
                return arr.tolist()
    pts = []
    for key in ("start", "end"):
        p = report.get(key)
        if isinstance(p, dict) and "latitude" in p and "longitude" in p:
            pts.append([float(p["latitude"]), float(p["longitude"])])
    if pts:
        return pts
    if "latitude" in report and "longitude" in report:
        return [[float(report["latitude"]), float(report["longitude"])]]
    return None


def _from_report(name: str, result: dict, key: str) -> Prediction | None:
    rep = (result or {}).get(key)
    route = report_route(rep)
    if route is None:
        return None
    kind = "route" if len(route) >= 2 else "point"
    return Prediction(name, route, kind, source=f"result.{key}",
                      meta={"report_source": rep.get("source"),
                            "ranking": rep.get("ranking"),
                            "refined_by_orienternet":
                                "orienternet_route_latlon" in rep})


def pipeline_headline(result: dict) -> Prediction | None:
    """``result["position"]`` — the answer the pipeline gives the user."""
    return _from_report("pipeline", result, "position")


def matcher_only(result: dict) -> Prediction | None:
    """``result["matcher_position"]`` — the raw matcher pick, no anchors."""
    return _from_report("matcher_only", result, "matcher_position")


def pinned_only(result: dict) -> Prediction | None:
    """``result["pinned_position"]`` — the matcher candidate georeferenced
    by OCR/sign anchors (source ``ocr_anchor_pin``/``anchor_georef``).
    Present only when the pipeline built one."""
    return _from_report("pinned_only", result, "pinned_position")


def graph_filter_only(result: dict) -> Prediction | None:
    """``result["filter_position"]`` — the opt-in road-graph particle filter's
    smoothed route (``--graph-filter``); absent unless the channel ran."""
    return _from_report("graph_filter_only", result, "filter_position")


def vpr_prior_only(result: dict) -> Prediction | None:
    """``result["vpr_prior"]`` — the retrieval centre as a single point."""
    vp = (result or {}).get("vpr_prior")
    if not isinstance(vp, dict):
        return None
    ll = _latlon_from_entry(vp)
    if ll is None:
        return None
    return Prediction("vpr_prior_only", [[ll[0], ll[1]]], "point",
                      source="result.vpr_prior",
                      meta={"origin": vp.get("origin"),
                            "track_frames": vp.get("track_frames")})


def load_preregistered_rule(name: str = "pipeline_selective", *,
                            root: Path = REPO_ROOT) -> dict | None:
    """One rule from ``benchmarks/preregistered_rules.json`` (``None`` if the
    file or rule is absent). Read, never fitted: the threshold was fixed on
    development clips before the held-out ones were run."""
    blob = _read_snapshot(Path(root) / PREREGISTERED_RULES)
    rule = (blob.get("rules") or {}).get(name)
    return rule if isinstance(rule, dict) and "tau" in rule else None


def retrieval_confidence(result: dict,
                         signals: Sequence[str] = ("vpr_coarse_to_fine.pass2_sim",
                                                   "vpr_prior.track_sim_median"),
                         ) -> tuple[float | None, str | None]:
    """The first finite ``section.key`` of ``signals`` present in ``result``
    and the name of the one used, or ``(None, None)``."""
    for sig in signals:
        section, _, key = sig.partition(".")
        v = ((result or {}).get(section) or {}).get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            return v, sig
    return None, None


def pipeline_selective(result: dict, fallback: Prediction | None,
                       rule: dict) -> Prediction | None:
    """Selective prediction with a pre-registered rule: the pipeline's answer
    when ``confidence >= rule["tau"]``, else ``fallback`` (the city centroid).

    A run without any confidence signal abstains. ``meta`` records the
    decision, so coverage (how often it answers) is reported beside the error.
    """
    conf, sig = retrieval_confidence(result, rule.get("confidence_signal")
                                     or ("vpr_coarse_to_fine.pass2_sim",
                                         "vpr_prior.track_sim_median"))
    tau = float(rule["tau"])
    answer = conf is not None and conf >= tau
    base = pipeline_headline(result) if answer else fallback
    if base is None:
        return None
    return Prediction("pipeline_selective", base.route_latlon, base.kind,
                      source=base.source,
                      meta={"abstained": not answer, "confidence": conf,
                            "signal": sig, "tau": tau})


def shortlist_hypotheses(result: dict) -> list[dict]:
    """The ranked location hypotheses carried by the result (may be [])."""
    for key in ("position", "matcher_position"):
        rep = (result or {}).get(key)
        if isinstance(rep, dict) and isinstance(rep.get("hypotheses"), list):
            return [h for h in rep["hypotheses"] if isinstance(h, dict)]
    return []


def shortlist_oracle(
    result: dict,
    gt_latlon: Sequence | np.ndarray,
    k: int = 5,
) -> Prediction | None:
    """ORACLE: of the first ``k`` hypotheses, the one whose point is nearest
    the GT start. Hypothesis coordinates are candidate START points
    (monocular_osm/hypotheses.py), so start error is the meaningful metric.

    Uses ground truth to choose — an upper bound on re-ranking, NOT a
    deployable method.
    """
    from .metrics import local_transformer, project

    hyps = shortlist_hypotheses(result)[: max(0, int(k))]
    pts = [ll for ll in (_latlon_from_entry([h.get("latitude"), h.get("longitude")])
                         for h in hyps) if ll is not None]
    if not pts:
        return None
    gt = np.asarray(gt_latlon, dtype=np.float64).reshape(-1, 2)
    tf = local_transformer(float(gt[0, 0]), float(gt[0, 1]))
    xy = project(np.asarray(pts), tf)
    g0 = project(gt[:1], tf)[0]
    best = int(np.argmin(np.hypot(*(xy - g0).T)))
    return Prediction(f"shortlist_oracle@{k}", [list(pts[best])], "point",
                      source="result.position.hypotheses", oracle=True,
                      meta={"k": int(k), "k_available": len(pts),
                            "chosen_rank": best + 1})
