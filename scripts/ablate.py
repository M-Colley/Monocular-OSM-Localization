"""Offline channel ablation and learned fusion over a benchmark run.

Every pipeline run writes ``fusion_features.json``: each matcher
candidate's per-channel ranks and penalties plus its route
(monocular_osm/fusion.py). ``scripts/benchmark.py run`` copies it into each
rep directory. This script re-runs the fusion rule offline under variants and
scores the candidate each variant would pick against ground truth — so the
question "what does channel X contribute?" costs seconds, not a fleet re-run.

Variants:
  recorded        the effective weights the run used (sanity: must pick the
                  run's own consensus #1)
  shape_only      shape rank alone
  no_<channel>    leave one active channel out (rank weight or penalty -> 0)
  no_double_shape sliding weight 0 — removes the shape double-count that
                  occurs whenever sliding windows did not run
  learned_loco    conditional-logit weights (statsmodels ConditionalLogit)
                  fitted on all OTHER clips, applied to the held-out clip
  oracle          the pool's best candidate (upper bound for any re-ranking)

Scope: this ablates the MATCHER pick (consensus #1), i.e. what
result["matcher_position"] reports. The anchor-primary headline places one
of the top-3 consensus candidates on the VPR track; that placement step is
not replayed here.

    python scripts/ablate.py --run-name r1
    python scripts/ablate.py --run-name r1 --metric start_error_m
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monocular_osm.fusion import (  # noqa: E402
    PENALTY_CHANNELS,
    RANK_CHANNELS,
    CandidateFeatures,
    consensus_order,
    load_feature_table,
)
from monocular_osm.metrics import score_route  # noqa: E402

RUNS_ROOT = ROOT / "output" / "benchmark_runs"
LOGIT_FEATURES = (
    "shape_rank", "sliding_rank", "turn_rank", "anchor_rank", "aerial_rank",
    "tile3d_rank", "plate_penalty", "vpr_penalty", "sun_penalty",
)


def _benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_for_ablate", ROOT / "scripts" / "benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _zero_weights() -> dict[str, float]:
    return {c: 0.0 for c in RANK_CHANNELS}


def _penalties_off(feats: list[CandidateFeatures], which: set[str]) -> list[CandidateFeatures]:
    out = []
    for f in feats:
        g = CandidateFeatures(**{**f.__dict__})
        for p in which:
            setattr(g, f"{p}_penalty", 0.0)
        out.append(g)
    return out


def variants_for(feats: list[CandidateFeatures], eff: dict) -> dict[str, tuple]:
    """name -> (features, weights) for every rule-based variant."""
    v = {"recorded": (feats, dict(eff)),
         "shape_only": (_penalties_off(feats, set(PENALTY_CHANNELS)),
                        {**_zero_weights(), "shape": 1.0})}
    for c in RANK_CHANNELS:
        if c != "shape" and eff.get(c, 0.0):
            v[f"no_{c}"] = (feats, {**eff, c: 0.0})
    for p in PENALTY_CHANNELS:
        if any(getattr(f, f"{p}_penalty") for f in feats):
            v[f"no_{p}"] = (_penalties_off(feats, {p}), dict(eff))
    if eff.get("sliding") and all(f.sliding_rank == f.shape_rank for f in feats):
        v["no_double_shape"] = (feats, {**eff, "sliding": 0.0})
    return v


def load_reps(run_dir: Path, manifest: dict, bm, metric: str) -> list[dict]:
    """One record per rep that has a feature table, with per-candidate scores."""
    by_id = {c["id"]: c for c in manifest["clips"]}
    reps = []
    for clip_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        entry = by_id.get(clip_dir.name)
        if entry is None:
            continue
        gt, _ = bm.load_gt(entry, ROOT)
        for rep_dir in sorted(clip_dir.glob("rep*")):
            path = rep_dir / "fusion_features.json"
            if not path.exists():
                continue
            feats, eff, payload = load_feature_table(path)
            routes = [row.get("route_latlon") for row in payload["candidates"]]
            if not feats or any(r is None for r in routes):
                continue
            errs = np.array([score_route(r, gt).to_dict()[metric] for r in routes])
            reps.append({"clip": entry["id"], "rep": rep_dir.name, "feats": feats,
                         "eff": eff, "errors": errs})
    return reps


def _logit_matrix(feats: list[CandidateFeatures]) -> np.ndarray:
    return np.array([[float(getattr(f, k)) for k in LOGIT_FEATURES] for f in feats])


def fit_conditional_logit(train: list[dict]):
    """Conditional logit: each rep is a choice set whose 'chosen' item is the
    candidate closest to GT. Returns (columns kept, params) or None."""
    from statsmodels.discrete.conditional_models import ConditionalLogit

    X, y, g = [], [], []
    for gi, r in enumerate(train):
        best = int(np.argmin(r["errors"]))
        X.append(_logit_matrix(r["feats"]))
        y.extend(int(i == best) for i in range(len(r["feats"])))
        g.extend([gi] * len(r["feats"]))
    X = np.vstack(X)
    # Drop columns with no within-group variation: unidentifiable.
    g_arr = np.asarray(g)
    keep = [j for j in range(X.shape[1])
            if any(np.ptp(X[g_arr == gi, j]) > 0 for gi in set(g))]
    if not keep:
        return None
    import warnings

    try:
        with warnings.catch_warnings():
            # Only point estimates are used; a singular Hessian (no standard
            # errors) is expected with few choice sets and is not an error.
            warnings.simplefilter("ignore")
            res = ConditionalLogit(np.asarray(y), X[:, keep], groups=g_arr).fit(
                disp=False, method="bfgs", maxiter=500)
    except Exception:
        return None
    params = np.asarray(res.params, dtype=np.float64)
    if not np.isfinite(params).all():
        return None
    return keep, params


def learned_picks(reps: list[dict]) -> dict[tuple, int | None]:
    """Leave-one-CLIP-out: fit on other clips, pick on the held-out one."""
    picks: dict[tuple, int | None] = {}
    for clip in sorted({r["clip"] for r in reps}):
        train = [r for r in reps if r["clip"] != clip]
        fit = fit_conditional_logit(train) if len({r["clip"] for r in train}) >= 2 else None
        for r in (r for r in reps if r["clip"] == clip):
            if fit is None:
                picks[(r["clip"], r["rep"])] = None
                continue
            keep, params = fit
            utility = _logit_matrix(r["feats"])[:, keep] @ params
            picks[(r["clip"], r["rep"])] = int(np.argmax(utility))
    return picks


def ablate(reps: list[dict]) -> dict:
    per_variant: dict[str, dict[str, list[float]]] = {}

    def add(name, clip, err):
        per_variant.setdefault(name, {}).setdefault(clip, []).append(float(err))

    for r in reps:
        for name, (feats, w) in variants_for(r["feats"], r["eff"]).items():
            add(name, r["clip"], r["errors"][consensus_order(feats, w)[0]])
        add("oracle", r["clip"], float(np.min(r["errors"])))
    for (clip, rep), pick in learned_picks(reps).items():
        if pick is not None:
            r = next(x for x in reps if x["clip"] == clip and x["rep"] == rep)
            add("learned_loco", clip, r["errors"][pick])

    clips = sorted({r["clip"] for r in reps})
    # Fleet statistics are taken over the clips EVERY variant covers: a
    # variant missing a hard clip (e.g. learned_loco when a held-out fit is
    # unidentifiable) would otherwise look better for averaging over less.
    common = set(clips)
    for by_clip in per_variant.values():
        common &= set(by_clip)
    table = {}
    for name, by_clip in per_variant.items():
        med = {c: float(np.median(v)) for c, v in by_clip.items()}
        shared = [med[c] for c in sorted(common)]
        table[name] = {
            "clips": len(med),
            "fleet_clips": len(shared),
            "fleet_mean_of_clip_medians_m": (round(float(np.mean(shared)), 1)
                                             if shared else None),
            "fleet_median_m": round(float(np.median(shared)), 1) if shared else None,
            "per_clip_median_m": {c: round(med[c], 1) for c in clips if c in med},
        }
    return table


def to_markdown(table: dict, metric: str) -> str:
    clips = sorted({c for v in table.values() for c in v["per_clip_median_m"]})
    order = ["recorded", "shape_only", "no_double_shape", "learned_loco", "oracle"]
    names = [n for n in order if n in table] + sorted(n for n in table if n not in order)
    head = ("| variant | clips | fleet mean (common clips) | fleet median | "
            + " | ".join(clips) + " |")
    sep = "|---" * (4 + len(clips)) + "|"
    rows = [f"Matcher-pick ablation — metric `{metric}` (m, median over reps per clip)",
            "", head, sep]
    for n in names:
        v = table[n]
        cells = [str(v["per_clip_median_m"].get(c, "–")) for c in clips]
        rows.append(f"| {n} | {v['clips']} | {v['fleet_mean_of_clip_medians_m']} | "
                    f"{v['fleet_median_m']} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--metric", default="symmetric_mean_m",
                    help="RouteScore field to minimise (default symmetric_mean_m)")
    args = ap.parse_args(argv)

    bm = _benchmark_module()
    run_dir = args.runs_root / args.run_name
    reps = load_reps(run_dir, bm.load_manifest(), bm, args.metric)
    if not reps:
        raise SystemExit(f"no rep with fusion_features.json under {run_dir}")
    table = ablate(reps)
    md = to_markdown(table, args.metric)
    (run_dir / "ablation.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (run_dir / "ablation.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
