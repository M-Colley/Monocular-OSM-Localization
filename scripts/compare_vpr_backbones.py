"""A/B test of the retrieval backbone: SAGE (ICLR 2026) against MegaLoc.

Both models retrieve from the exact reference sets benchmark run v2 searched:
the Mapillary metadata and pixels its cold-cache reps archived (read-only;
``--archive-root``). For each clip the query frames are the 80 the pipeline
samples evenly over the analysis window, and ground truth is interpolated at
each query's video time. Both models embed the same images on the same device
(CPU, fp32 by default) with their own published preprocessing (MegaLoc: the
pipeline's 512x512; SAGE: 322x322, its README).

Reported per model and pass (pass 1 = the disc that chooses the district;
pass 2 = the coarse-to-fine disc): top-1 / top-5 recall within 25 / 50 /
100 m, the median top-1 error, the error of the robust retrieval centre the
pipeline builds from the top-1 hits (``kartaview_vpr._robust_center``), and
the AUROC of the top-1 similarity for separating correct (< 50 m) from wrong
retrievals: the confidence the abstention rule thresholds. Nothing is tuned.

    python scripts/compare_vpr_backbones.py --models megaloc sage_vitb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

#: Pinned like the pipeline's MegaLoc hub repo (kartaview_vpr.HUB_MEGALOC).
HUB_SAGE = "chenshunpeng/SAGE:{sha}"
SAGE_SHA_DEFAULT = "b52f9fc24fd0bdc2cb574b1e54d99e5d664d01cd"   # main, 2026-09-20 (MIT)
PASSES = {"pass1": "mapillary", "pass2": "mapillary_c2f"}
CORRECT_M = 50.0


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def archived_refs(rep_dir: Path, sub: str):
    """(ids, raw[N,512,512,3] BGR, latlon[N,2], signature) of one archived pass,
    read-only from ``mly_ref_meta_*.json`` + ``ref_img_store.npz``."""
    d = rep_dir / sub
    metas = sorted(d.glob("mly_ref_meta_*.json"))
    store = d / "ref_img_store.npz"
    if not metas or not store.exists():
        return None
    blob = json.loads(metas[0].read_text(encoding="utf-8"))
    refs = blob.get("refs") or []
    with np.load(store, allow_pickle=True) as z:
        sid = [str(x) for x in z["ids"]]
        raw = z["raw"]
        pos = {k: i for i, k in enumerate(sid)}
        keep = [(r, pos[str(r["id"])]) for r in refs if str(r["id"]) in pos]
        if len(keep) < 30:
            return None
        img = np.stack([raw[i] for _, i in keep])
    latlon = np.array([[float(r["lat"]), float(r["lon"])] for r, _ in keep])
    return [str(r["id"]) for r, _ in keep], img, latlon, blob.get("signature"), [r for r, _ in keep]


def archived_megaloc(rep_dir: Path, sub: str, refs: list[dict]) -> np.ndarray | None:
    """MegaLoc reference embeddings exactly as the v2 run computed them (GPU,
    fp16 autocast), used only when their content fingerprint proves they
    belong to ``refs`` in this order."""
    from monocular_osm.kartaview_vpr import _refs_fingerprint

    fp = _refs_fingerprint(refs)
    for f in sorted((rep_dir / sub).glob("ref_emb_megaloc_*.npz")):
        with np.load(f, allow_pickle=True) as z:
            if str(z["fingerprint"]) == fp and len(z["emb"]) == len(refs):
                return np.asarray(z["emb"], np.float32)
    return None


def query_frames(video: Path, start: float, end: float, n: int):
    """``n`` frames evenly over [start, end) and their video times (s)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    times = np.linspace(start, end, n, endpoint=False)
    frames, kept = [], []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, f = cap.read()
        if ok and f is not None:
            frames.append(f)
            kept.append(float(t))
    cap.release()
    return frames, np.array(kept)


def gt_at(waypoints: list[dict], t: np.ndarray) -> np.ndarray:
    tt = np.array([w["t_sec"] for w in waypoints], float)
    la = np.array([w["lat"] for w in waypoints], float)
    lo = np.array([w["lon"] for w in waypoints], float)
    return np.column_stack([np.interp(t, tt, la), np.interp(t, tt, lo)])


def haversine_m(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.radians(np.atleast_2d(a)), np.radians(np.atleast_2d(b))
    dlat, dlon = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    h = np.sin(dlat / 2) ** 2 + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin(dlon / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

class Backbone:
    def __init__(self, name: str, sage_sha: str):
        import torch

        self.name = name
        if name == "megaloc":
            from monocular_osm.kartaview_vpr import HUB_MEGALOC
            self.model = torch.hub.load(HUB_MEGALOC, "get_trained_model", trust_repo=True,
                                        verbose=False).eval()
            self.size = 512
        elif name.startswith("sage"):
            self.model = torch.hub.load(HUB_SAGE.format(sha=sage_sha), name, pretrained=True,
                                        trust_repo=True, verbose=False).eval()
            self.size = 322
        else:
            raise ValueError(name)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def embed(self, bgr_images, batch: int = 16) -> np.ndarray:
        import cv2
        import torch

        out = []
        for i in range(0, len(bgr_images), batch):
            ts = []
            for im in bgr_images[i:i + batch]:
                rgb = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
                t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                ts.append((t - self.mean) / self.std)
            with torch.no_grad():
                f = self.model(torch.stack(ts)).float()
            out.append(torch.nn.functional.normalize(f, dim=1).numpy())
        return np.concatenate(out)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def auroc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    if len(pos) == 0 or len(neg) == 0:
        return None
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


def score(sims: np.ndarray, ref_ll: np.ndarray, q_gt: np.ndarray, gt_route: np.ndarray) -> dict:
    from monocular_osm.kartaview_vpr import _robust_center

    order = np.argsort(-sims, axis=1)
    top1 = order[:, 0]
    err1 = np.array([haversine_m(ref_ll[top1[q]], q_gt[q])[0] for q in range(len(q_gt))])
    err5 = np.array([haversine_m(ref_ll[order[q, :5]], np.repeat(q_gt[q:q + 1], 5, 0)).min()
                     for q in range(len(q_gt))])
    s1 = sims[np.arange(len(top1)), top1]
    centre = np.asarray(_robust_center(ref_ll[top1], s1), float).reshape(-1)[:2]
    centre_err = float(haversine_m(np.repeat(centre[None], len(gt_route), 0), gt_route).min())
    ok = err1 < CORRECT_M
    return {
        "n_query": int(len(q_gt)),
        **{f"r1@{d}m": float(np.mean(err1 < d)) for d in (25, 50, 100)},
        **{f"r5@{d}m": float(np.mean(err5 < d)) for d in (25, 50, 100)},
        "top1_err_median_m": float(np.median(err1)),
        "centre_err_m": centre_err,
        "top1_sim_median": float(np.median(s1)),
        "sim_auroc_correct_vs_wrong": auroc(s1[ok], s1[~ok]),
    }


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    import benchmark as B
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["megaloc", "sage_vitb"])
    ap.add_argument("--clips", nargs="*", default=None, help="default: core clips run in v2")
    ap.add_argument("--run-name", default="v2")
    ap.add_argument("--archive-root", type=Path, default=Path("E:/monocular-osm-archive"))
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--n-query", type=int, default=80)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--sage-sha", default=SAGE_SHA_DEFAULT)
    ap.add_argument("--passes", nargs="+", default=["pass1"], choices=list(PASSES))
    ap.add_argument("--no-reuse-archived", dest="reuse_archived", action="store_false",
                    help="recompute MegaLoc's reference embeddings instead of reusing the "
                         "fingerprint-matched ones the v2 run archived")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "vpr_backbones")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = B.load_manifest(ROOT / "benchmarks" / "manifest.json")
    clips = [c for c in manifest["clips"]
             if (c["id"] in args.clips if args.clips else c["fleet"] == "core")]
    models = {m: Backbone(m, args.sage_sha) for m in args.models}
    results: dict = {"meta": {"models": args.models, "run": args.run_name,
                              "n_query": args.n_query, "device": "cpu", "precision": "fp32",
                              "sage_hub": HUB_SAGE.format(sha=args.sage_sha),
                              "correct_m": CORRECT_M}, "clips": {}}
    for e in clips:
        slug_dirs = [d for d in B.candidate_slug_dirs(e, e["city"], args.data_dir)]
        rep_dir = next((args.archive_root / d.name / "_benchmark_cold_vpr" / args.run_name
                        / e["id"] / "rep0" for d in slug_dirs
                        if (args.archive_root / d.name / "_benchmark_cold_vpr" / args.run_name
                            / e["id"] / "rep0").is_dir()), None)
        if rep_dir is None:
            print(f"SKIP {e['id']}: no archived {args.run_name} references")
            continue
        video = (ROOT / e["video"]["path"]) if "path" in e["video"] else next(
            (d / "input.mp4" for d in slug_dirs if (d / "input.mp4").exists()), None)
        if video is None:
            print(f"SKIP {e['id']}: video not on disk")
            continue
        start, end = (float(x) for x in e["vo_segment"].split(":"))
        frames, qt = query_frames(video, start, end, args.n_query)
        wps = json.loads((ROOT / e["gt"]).read_text(encoding="utf-8"))["waypoints"]
        q_gt = gt_at(wps, qt)
        gt_route = np.array([[w["lat"], w["lon"]] for w in wps])
        out: dict = {}
        for pname, sub in ((k, v) for k, v in PASSES.items() if k in args.passes):
            refs = archived_refs(rep_dir, sub)
            if refs is None:
                continue
            ids, raw, ref_ll, sig, ref_dicts = refs
            out[pname] = {"n_refs": int(len(ids)), "signature": sig}
            for mname, model in models.items():
                cache = args.out / f"emb_{e['id']}_{pname}_{mname}.npz"
                t0 = time.perf_counter()
                if cache.exists():
                    z = np.load(cache)
                    r_emb, q_emb = z["refs"], z["queries"]
                else:
                    r_emb = (archived_megaloc(rep_dir, sub, ref_dicts)
                             if mname == "megaloc" and args.reuse_archived else None)
                    out[pname].setdefault("megaloc_refs", None)
                    if mname == "megaloc":
                        out[pname]["megaloc_refs"] = ("archived v2 (GPU fp16)"
                                                      if r_emb is not None else "recomputed (CPU fp32)")
                    if r_emb is None:
                        r_emb = model.embed(raw)
                    q_emb = model.embed(frames)
                    np.savez_compressed(cache, refs=r_emb, queries=q_emb, ids=np.array(ids))
                sc = score(q_emb @ r_emb.T, ref_ll, q_gt, gt_route)
                sc["embed_s"] = round(time.perf_counter() - t0, 1)
                out[pname][mname] = sc
                print(f"{e['id']:14s} {pname} {mname:9s} r1@50 {sc['r1@50m']:.2f} "
                      f"r5@50 {sc['r5@50m']:.2f} centre {sc['centre_err_m']:7.0f} m "
                      f"AUROC {sc['sim_auroc_correct_vs_wrong']}", flush=True)
        results["clips"][e["id"]] = out
        (args.out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
