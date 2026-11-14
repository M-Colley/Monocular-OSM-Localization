"""Fixed query-only MegaLoc pilot on KITTI 0009 and Atlanta twilight.

Inference uses cached model weights and reference embeddings only. It never
opens GT. Run the separate ``score`` command after predictions are saved.
Defaults are deliberately unchanged by this experiment.

    python scripts/pilot_query_preprocessing.py infer --threads 4 --batch 2
    python scripts/pilot_query_preprocessing.py score
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Fixed before this pilot's scores: original warp; three square views with
# maximum cosine per reference; native aspect at approximately 512² pixels.
MODES = ("warp", "crops", "aspect")
CLIPS = {
    "kitti_0009": {
        "video": "data/kitti/drive_0009.mp4", "window": [0.0, 46.0],
        "references": "output/dense_pilot/dense_kitti_0009_75m.npz",
        "reference_label": "existing dense75m pilot, 8610 references; not v6",
    },
    "overlay_y66ZkRpUh4k": {
        "video": "data/y66zkrpuh4k-asmr-driving-in-atlanta-at-night-in-a-tesla-model-3-wolfbox-i07-dash/input.mp4",
        "window": [0.0, 295.0],
        "references": "y66zkrpuh4k-asmr-driving-in-atlanta-at-night-in-a-tesla-model-3-wolfbox-i07-dash/_benchmark_cold_vpr/v2/overlay_y66ZkRpUh4k/rep0/mapillary",
        "reference_label": "archived v2 seeded 3km pool, 1500 references; not v6",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 ** 2):
            h.update(block)
    return h.hexdigest()


def load_cached_references(path: Path):
    """Read embeddings/coordinates without loading pixels or accessing APIs."""
    from monocular_osm.kartaview_vpr import _refs_fingerprint

    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            ids, ll, emb = z["ids"].astype(str), z["latlon"].astype(float), z["emb"].astype(np.float32)
        fingerprint = None
        source = path
    else:
        selected = None
        for meta in sorted(path.glob("mly_ref_meta_*.json")):
            blob = json.loads(meta.read_text(encoding="utf-8"))
            refs = blob.get("refs", [])
            fingerprint = _refs_fingerprint(refs)
            for source in sorted(path.glob("ref_emb_megaloc*.npz")):
                with np.load(source, allow_pickle=False) as z:
                    if str(z["fingerprint"]) == fingerprint and len(z["emb"]) == len(refs):
                        selected = (refs, z["emb"].astype(np.float32), source, fingerprint)
                        break
            if selected is not None:
                break
        if selected is None:
            raise ValueError(f"no fingerprint-matched cached MegaLoc embeddings: {path}")
        refs, emb, source, fingerprint = selected
        ids = np.array([str(r["id"]) for r in refs])
        ll = np.array([[r["lat"], r["lon"]] for r in refs], dtype=float)
    if emb.ndim != 2 or ll.shape != (len(emb), 2) or len(ids) != len(emb):
        raise ValueError("cached embedding/reference identity or coordinate shapes disagree")
    return emb, ll, {"embedding_file": str(source.resolve()), "sha256": _sha256(source),
                     "fingerprint": fingerprint, "n_refs": len(emb)}


def _load_offline_megaloc(device):
    # Set before huggingface_hub is imported. Even the optional config.json
    # fetch in the cached hubconf must stay offline during this pilot.
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from monocular_osm import kartaview_vpr as kv

    sha = kv.HUB_MEGALOC.split(":", 1)[1]
    repo = Path(torch.hub.get_dir()) / f"gmberton_MegaLoc_{sha}"
    if not repo.is_dir():
        raise FileNotFoundError(f"pinned MegaLoc hub code is not cached: {repo}")
    model = torch.hub.load(str(repo), "get_trained_model", source="local",
                           verbose=False).to(device).eval()
    kv._MODEL, kv._MODEL_NAME, kv._MODEL_DEVICE = model, "megaloc", str(device)
    kv._MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    kv._STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return kv


def infer(args):
    import torch
    from compare_vpr_backbones import query_frames

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    experiment = {"modes": list(MODES), "clips": args.clips, "n_query": args.n_query,
                  "device": args.device, "threads": args.threads, "batch": args.batch,
                  "selection": "all fixed modes; no GT opened during inference",
                  "references": "fixed cached embeddings; no downloads; not the v6 pool",
                  "sequence": "existing Viterbi; existing argmax fallback above6000 references",
                  "answer": "existing running median5 retrieval route"}
    registration = args.out / "experiment.json"
    if registration.exists() and json.loads(registration.read_text()) != experiment:
        raise ValueError("output already belongs to a different pilot configuration; use a new --out")
    registration.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    kv = _load_offline_megaloc(args.device)
    for cid in args.clips:
        entry = CLIPS[cid]
        ref_path = Path(entry["references"])
        ref_path = ROOT / ref_path if cid == "kitti_0009" else args.archive_root / ref_path
        emb, ll, reference_info = load_cached_references(ref_path)
        frames, times = query_frames(ROOT / entry["video"], *entry["window"], args.n_query)
        if len(frames) < 3:
            raise RuntimeError(f"too few readable query frames: {cid}")
        indices = np.arange(len(frames))
        reference_info["label"] = entry["reference_label"]
        ref_emb = torch.from_numpy(emb)
        dt = float(np.mean(np.diff(times)))
        for mode in MODES:
            output = args.out / f"{cid}_{mode}.npz"
            if output.exists():
                with np.load(output, allow_pickle=False) as z:
                    prior = json.loads(str(z["metadata"]))
                if prior["references"]["sha256"] != reference_info["sha256"]:
                    raise ValueError(f"stale pilot reference embeddings: {output}")
                print(f"reusing {output.name}", flush=True)
                continue
            start = time.perf_counter()
            # Keep the control numerically identical while respecting this
            # pilot's smaller CPU batch budget (batch partition changes no math).
            if mode == "warp":
                q_emb = kv._embed(args.device, kv._QueryViews(frames, indices, mode),
                                  batch_size=args.batch, prepare=kv._prep)
                sims = (q_emb @ ref_emb.T).numpy()
            else:
                sims = kv._query_similarities(frames, indices, ref_emb, args.device,
                                              mode=mode, batch_size=args.batch)
            chosen, decoding = sims.argmax(axis=1), "argmax"
            try:
                chosen = kv._viterbi_decode(sims, ll, dt)
                decoding = "viterbi"
            except ValueError as exc:
                decoding = f"argmax ({exc})"
            metadata = {"clip": cid, "mode": mode, "references": reference_info,
                        "query_window": entry["window"], "query_frames": len(frames),
                        "input_size": list(frames[0].shape), "decoding": decoding,
                        "elapsed_s": time.perf_counter() - start,
                        "hub": kv.HUB_MEGALOC, "device": args.device,
                        "torch": torch.__version__, "gt_used": False}
            temp = output.with_suffix(".tmp.npz")
            np.savez_compressed(temp, similarities=sims, ref_latlon=ll,
                                query_times=times, chosen=chosen,
                                metadata=np.array(json.dumps(metadata)))
            temp.replace(output)
            print(f"{cid} {mode}: {metadata['elapsed_s']:.1f}s, {len(frames)} queries, "
                  f"{len(ll)} references; {decoding}", flush=True)


def score(args):
    """This is the only phase that opens ground truth, after inference files exist."""
    from compare_vpr_backbones import gt_at
    from monocular_osm.baselines import vpr_track_only
    from monocular_osm.geo import geodesic_distance_m
    from monocular_osm.metrics import score_route

    manifest = json.loads((ROOT / "benchmarks/manifest.json").read_text(encoding="utf-8"))
    entries = {e["id"]: e for e in manifest["clips"]}
    results = {}
    for file in sorted(args.out.glob("*.npz")):
        with np.load(file, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            sims, ll, tt, chosen = z["similarities"], z["ref_latlon"], z["query_times"], z["chosen"]
        gt_path = ROOT / entries[meta["clip"]]["gt"]
        waypoints = json.loads(gt_path.read_text(encoding="utf-8"))["waypoints"]
        # Only GT inside the inference window is scored, as in the benchmark.
        start, end = meta["query_window"]
        waypoints = [p for p in waypoints if start <= p["t_sec"] <= end]
        gt = np.array([[p["lat"], p["lon"]] for p in waypoints])
        at_query = gt_at(waypoints, tt)
        top = np.argsort(-sims, axis=1)
        nearest = geodesic_distance_m(at_query[:, 0, None], at_query[:, 1, None],
                                      ll[None, :, 0], ll[None, :, 1])
        retrieved = np.take_along_axis(nearest, top, axis=1)
        track = [[float(t), *map(float, ll[r]), float(sims[q, r])]
                 for q, (t, r) in enumerate(zip(tt, chosen))]
        prediction = vpr_track_only({"vpr_prior": {"track": track}})
        metrics = score_route(prediction.route_latlon, gt).to_dict()
        results.setdefault(meta["clip"], {})[meta["mode"]] = {
            "symmetric_mean_m": metrics["symmetric_mean_m"],
            "start_error_m": metrics["start_error_m"],
            "oracle_coverage_50m": float(np.mean(nearest.min(axis=1) <= 50)),
            "r1_50m": float(np.mean(retrieved[:, 0] <= 50)),
            "r1_250m": float(np.mean(retrieved[:, 0] <= 250)),
            "r5_50m": float(np.mean(retrieved[:, :5].min(axis=1) <= 50)),
            "elapsed_s": meta["elapsed_s"], "references": meta["references"],
        }
    (args.out / "scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["infer", "score"])
    parser.add_argument("--out", type=Path, default=ROOT / "output/query_preprocessing_pilot")
    parser.add_argument("--archive-root", type=Path, default=Path("E:/monocular-osm-archive"))
    parser.add_argument("--clips", nargs="+", choices=list(CLIPS), default=list(CLIPS))
    parser.add_argument("--n-query", type=int, default=80)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    (infer if args.command == "infer" else score)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
