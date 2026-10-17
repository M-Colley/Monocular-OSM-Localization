"""Pre-registered SafeVPR transfer test (benchmarks/preregistered_rules.json,
rule ``safevpr_frame_verification``): does a calibrated accept/abstain decision
on retrieval keep its false-discovery rate on footage it was not calibrated on?

Unit: one query frame and the MegaLoc top-1 reference of the first retrieval
pass, on the exact reference sets benchmark run v2 searched (read-only from its
cold-run archives) and the pipeline's 80 query frames per clip. Label: correct
iff that reference is within 50 m of the ground truth at the query's time.
Scores: SafeVPR's patch mutual-nearest-neighbour + Lowe-ratio score (its own
``extract_patch_features`` / ``mnn_lowe_ratio``, DINOv2 ViT-L/14, T=1 because
the references are single photos) and the MegaLoc cosine (baseline). Each is
calibrated once with SafeVPR's ``fit_mondrian_ltt`` (alpha 0.10) on the core
clips and applied unchanged to the overlay clips.

    python scripts/eval_safevpr.py --safevpr-root E:/monocular-osm-tools/SafeVPR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

ALPHA = 0.10
CORRECT_M = 50.0


def find_rep_dir(slug_dirs, clip_id: str, run: str, roots: list[Path]) -> Path | None:
    """The archived rep0 of ``clip_id`` under any of ``roots`` (E: archive or
    the clip's own data dir, where the older runner archived in place)."""
    for d in slug_dirs:
        for root in roots:
            base = (root / d.name) if root.name != d.name else root
            cand = base / "_benchmark_cold_vpr" / run / clip_id / "rep0"
            if cand.is_dir():
                return cand
        cand = d / "_benchmark_cold_vpr" / run / clip_id / "rep0"
        if cand.is_dir():
            return cand
    return None


def bgr_to_pil(img):
    import cv2
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB))


def clip_pairs(entry, args, megaloc, dino, verifier) -> list[dict] | None:
    import benchmark as B
    import compare_vpr_backbones as C

    slug_dirs = B.candidate_slug_dirs(entry, entry["city"], args.data_dir)
    rep = find_rep_dir(slug_dirs, entry["id"], args.run_name, [args.archive_root])
    if rep is None:
        print(f"SKIP {entry['id']}: no archived {args.run_name} references")
        return None
    refs = C.archived_refs(rep, "mapillary")
    if refs is None:
        print(f"SKIP {entry['id']}: archived pass-1 references unreadable")
        return None
    ids, raw, ref_ll, _sig, ref_dicts = refs
    video = (ROOT / entry["video"]["path"]) if "path" in entry["video"] else next(
        (d / "input.mp4" for d in slug_dirs if (d / "input.mp4").exists()), None)
    if video is None:
        print(f"SKIP {entry['id']}: video not on disk")
        return None
    start, end = (float(x) for x in entry["vo_segment"].split(":"))
    frames, qt = C.query_frames(video, start, end, args.n_query)
    wps = json.loads((ROOT / entry["gt"]).read_text(encoding="utf-8"))["waypoints"]
    q_gt = C.gt_at(wps, qt)

    cache = args.out / f"megaloc_{entry['id']}.npz"
    ab_cache = ROOT / "output" / "vpr_backbones" / f"emb_{entry['id']}_pass1_megaloc.npz"
    if cache.exists():
        z = np.load(cache)
        r_emb, q_emb = z["refs"], z["queries"]
    elif ab_cache.exists():                       # same refs + queries, from the A/B run
        z = np.load(ab_cache)
        r_emb, q_emb = z["refs"], z["queries"]
    else:
        r_emb = C.archived_megaloc(rep, "mapillary", ref_dicts)
        if r_emb is None:
            r_emb = megaloc.embed(raw)
        q_emb = megaloc.embed(frames)
    np.savez_compressed(cache, refs=r_emb, queries=q_emb)
    sims = q_emb @ r_emb.T
    top1 = sims.argmax(axis=1)

    feat_cache: dict = {}

    def feats(key, img):
        if key not in feat_cache:
            feat_cache[key] = verifier.extract_patch_features(dino, bgr_to_pil(img))
        return feat_cache[key]

    out = []
    for q in range(len(frames)):
        r = int(top1[q])
        err = float(C.haversine_m(ref_ll[r], q_gt[q])[0])
        s_patch = verifier.mnn_lowe_ratio(feats(("q", q), frames[q]), feats(("r", r), raw[r]))
        out.append({"clip": entry["id"], "fleet": entry["fleet"], "query": q,
                    "t_sec": float(qt[q]), "ref_id": ids[r], "err_m": err,
                    "correct": bool(err < CORRECT_M), "cosine": float(sims[q, r]),
                    "patch_mnn": float(s_patch)})
    print(f"{entry['id']:22s} {entry['fleet']:7s} correct {np.mean([p['correct'] for p in out]):.2f} "
          f"cos {np.median([p['cosine'] for p in out]):.3f} "
          f"patch {np.median([p['patch_mnn'] for p in out]):.3f}", flush=True)
    return out


def evaluate(pairs: list[dict], conformal, metrics) -> dict:
    cal = [p for p in pairs if p["fleet"] == "core"]
    test = [p for p in pairs if p["fleet"] == "overlay"]
    res: dict = {"alpha": ALPHA, "n_cal": len(cal), "n_test": len(test),
                 "cal_positive_rate": float(np.mean([p["correct"] for p in cal])) if cal else None,
                 "test_positive_rate": float(np.mean([p["correct"] for p in test])) if test else None,
                 "scores": {}}
    for score in ("patch_mnn", "cosine"):
        cs = np.array([p[score] for p in cal], float)
        cy = np.array([p["correct"] for p in cal], bool)
        ts = np.array([p[score] for p in test], float)
        ty = np.array([p["correct"] for p in test], bool)
        thr = conformal.fit_mondrian_ltt(cs, cy.astype(int), alpha=ALPHA)
        r: dict = {"thresholds": {"edges": [float(x) for x in thr["edges"]],
                                  "taus": [float(x) for x in thr["taus"]]}}
        for name, s, y in (("cal", cs, cy), ("test", ts, ty)):
            if len(s) == 0:
                continue
            acc = conformal.accept(s, thr)
            n_acc = int(acc.sum())
            r[name] = {"accepted": n_acc, "coverage": float(acc.mean()),
                       "fdr": (float(np.mean(~y[acc])) if n_acc else None),
                       "tpr": (float(acc[y].mean()) if y.any() else None),
                       "auroc": (float(metrics.auroc(s, y.astype(int)))
                                 if 0 < y.sum() < len(y) else None)}
        if test:
            per_clip = {}
            for cid in sorted({p["clip"] for p in test}):
                m = np.array([p["clip"] == cid for p in test])
                acc = conformal.accept(ts[m], thr)
                per_clip[cid] = {"accepted": int(acc.sum()), "correct_accepted": int((acc & ty[m]).sum()),
                                 "correct": int(ty[m].sum()), "n": int(m.sum())}
            r["test_per_clip"] = per_clip
        res["scores"][score] = r
    pm, co = res["scores"]["patch_mnn"].get("test", {}), res["scores"]["cosine"].get("test", {})
    res["hypotheses"] = {
        "S1": None if pm.get("fdr") is None else bool(pm["fdr"] <= ALPHA),
        "S2": (None if pm.get("accepted") is None else bool(
            (co.get("fdr") is not None and pm.get("fdr") is not None and co["fdr"] > pm["fdr"])
            or (co.get("accepted") == 0 and (pm.get("accepted") or 0) > 0
                and pm.get("fdr") is not None and pm["fdr"] <= ALPHA))),
        "S3": (None if pm.get("auroc") is None or co.get("auroc") is None
               else bool(pm["auroc"] > co["auroc"])),
    }
    return res


def main(argv=None) -> int:
    import benchmark as B
    import compare_vpr_backbones as C
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--safevpr-root", type=Path, default=Path("E:/monocular-osm-tools/SafeVPR"))
    ap.add_argument("--archive-root", type=Path, default=Path("E:/monocular-osm-archive"))
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--run-name", default="v2")
    ap.add_argument("--n-query", type=int, default=80)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "safevpr")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.safevpr_root))
    from safevpr import conformal, metrics, verifier

    manifest = B.load_manifest(ROOT / "benchmarks" / "manifest.json")
    megaloc = C.Backbone("megaloc", C.SAGE_SHA_DEFAULT)
    dino = verifier.load_dinov2(device=torch.device("cpu"))
    pairs: list[dict] = []
    for e in manifest["clips"]:
        got = clip_pairs(e, args, megaloc, dino, verifier)
        if got:
            pairs.extend(got)
            (args.out / "pairs.json").write_text(json.dumps(pairs, indent=0), encoding="utf-8")
    res = evaluate(pairs, conformal, metrics)
    res["safevpr_commit"] = "f11d5fd"
    (args.out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "scores"}, indent=1))
    for sname, r in res["scores"].items():
        print(sname, "cal", r.get("cal"), "\n   test", r.get("test"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
