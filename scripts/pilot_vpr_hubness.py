"""Post-hoc dev pilot: fixed hubness corrections on saved VPR matrices.

No models, downloads or GT enter ``infer``. The complete raw/reference-mean/
CSLS10 candidate set is saved before the separate offline ``score`` phase.
This experiment was designed after seeing the query-preprocessing results.

    python scripts/pilot_vpr_hubness.py infer
    python scripts/pilot_vpr_hubness.py score
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FORMULAS = {
    "raw": "S (unchanged cosine control)",
    "reference_mean": "S - mean(S, axis=query)",
    "csls10": "S - 0.5 * mean_top10_refs(S, per query) - 0.5 * mean_top10_queries(S, per ref)",
}


def correct_similarities(sims, mode):
    """Fixed transformations whose only input is an inference similarity matrix."""
    s = np.asarray(sims, dtype=np.float32)
    if s.ndim != 2 or 0 in s.shape or not np.isfinite(s).all():
        raise ValueError("similarities must be a non-empty finite matrix")
    if mode == "raw":
        return s.copy()
    if mode == "reference_mean":
        return s - s.mean(axis=0, keepdims=True)
    if mode == "csls10":
        nq, nr = min(10, s.shape[0]), min(10, s.shape[1])
        ref_hubness = np.partition(s, -nq, axis=0)[-nq:].mean(axis=0, keepdims=True)
        query_density = np.partition(s, -nr, axis=1)[:, -nr:].mean(axis=1, keepdims=True)
        # Half-CSLS keeps the original cosine scale for the unchanged Viterbi
        # transition penalty. The per-query term is constant across states.
        return s - 0.5 * ref_hubness - 0.5 * query_density
    raise ValueError(f"unknown hubness correction: {mode}")


def infer(args):
    from monocular_osm.baselines import vpr_track_only
    from monocular_osm.kartaview_vpr import _viterbi_decode

    args.out.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.input.glob("*.npz"))
    if not sources:
        raise FileNotFoundError(f"no saved inference matrices: {args.input}")
    registration = {"status": "post-hoc dev experiment; preprocessing scores were already seen",
                    "formulas": FORMULAS, "top_k": 10, "source": str(args.input.resolve()),
                    "gt_used_during_inference": False, "default_changed": False,
                    "decoding": "unchanged Viterbi, original >6000-ref argmax fallback",
                    "answer": "unchanged running median5 retrieval route"}
    registration_path = args.out / "experiment.json"
    if registration_path.exists() and json.loads(registration_path.read_text()) != registration:
        raise ValueError("output belongs to another registered experiment; choose a new --out")
    registration_path.write_text(json.dumps(registration, indent=2), encoding="utf-8")
    for source in sources:
        with np.load(source, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            original, ll, times = z["similarities"], z["ref_latlon"], z["query_times"]
        if meta.get("gt_used") is not False:
            raise ValueError(f"source does not declare GT-free inference: {source}")
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        for mode in FORMULAS:
            output = args.out / f"{source.stem}__{mode}.npz"
            corrected = correct_similarities(original, mode)
            chosen, decoding = corrected.argmax(axis=1), "argmax"
            try:
                chosen = _viterbi_decode(corrected, ll, float(np.mean(np.diff(times))))
                decoding = "viterbi"
            except ValueError as exc:
                decoding = f"argmax ({exc})"
            track = [[float(t), *map(float, ll[r]), float(corrected[q, r])]
                     for q, (t, r) in enumerate(zip(times, chosen))]
            route = vpr_track_only({"vpr_prior": {"track": track}}).route_latlon
            metadata = {**meta, "mode": f"{meta['mode']}__{mode}", "normalization": mode,
                        "normalization_formula": FORMULAS[mode], "decoding": decoding,
                        "source_prediction": str(source.resolve()), "source_sha256": source_hash,
                        "query_mode": meta["mode"], "gt_used": False,
                        "experiment": "post-hoc reference hubness correction"}
            temp = output.with_suffix(".tmp.npz")
            np.savez_compressed(temp, similarities=corrected, ref_latlon=ll, query_times=times,
                                chosen=chosen, route_latlon=np.asarray(route),
                                metadata=np.array(json.dumps(metadata)))
            temp.replace(output)
            print(f"saved {output.name}: {decoding}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["infer", "score"])
    parser.add_argument("--input", type=Path, default=ROOT / "output/query_preprocessing_pilot")
    parser.add_argument("--out", type=Path, default=ROOT / "output/vpr_hubness_pilot")
    args = parser.parse_args(argv)
    if args.command == "infer":
        infer(args)
    else:
        from pilot_query_preprocessing import score
        score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
