"""Pilot: does a much denser first-pass reference set fix retrieval on a
failing clip?

The first (city-wide) retrieval pass keeps ~1,500 of the 200k-500k Mapillary
images in its disc (lowest sha1 of the id, i.e. uniform-random), which cover
3 % of KITTI 0009's route within 50 m although the full pool covers 92 %
(docs/experiments_2026-10.md, section 1). Here the same disc is searched with
one reference per ``--spacing`` metres instead: the pipeline's own Mapillary
listing (``kartaview_vpr._mly_tile_listing``, coverage tiles) lists the disc's
images, they are thinned deterministically (sha1 order, at most one per
``spacing`` disc; no ground truth involved), their thumbnail URLs are looked
up (``_mly_lookup_urls``), and they are downloaded and embedded with
MegaLoc exactly as the pipeline does (512x512, ImageNet normalisation) in
streaming batches without keeping pixels. The control is v2's archived sparse
set of the same disc, embedded the same way on the same device, against the
same 80 query frames. Ground truth scores only.

    python scripts/dense_retrieval_pilot.py --clip kitti_0009 --spacing 75
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


def thin(refs: list[dict], centre, spacing_m: float) -> list[dict]:
    """At most one reference within ``spacing_m`` of another, chosen in sha1
    order (deterministic, insertion-stable like the pipeline's pick)."""
    from monocular_osm.geo import LocalFrame

    frame = LocalFrame(centre[0], centre[1])
    order = sorted(refs, key=lambda r: hashlib.sha1(str(r["id"]).encode()).hexdigest())
    lat = np.array([r["lat"] for r in order])
    lon = np.array([r["lon"] for r in order])
    x, y = frame.to_xy(lat, lon)
    cells: dict[tuple, list] = {}
    keep = []
    for i, (xi, yi) in enumerate(zip(x, y)):
        cx, cy = int(np.floor(xi / spacing_m)), int(np.floor(yi / spacing_m))
        clash = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for (px, py) in cells.get((cx + dx, cy + dy), ()):
                    if (px - xi) ** 2 + (py - yi) ** 2 < spacing_m ** 2:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            cells.setdefault((cx, cy), []).append((xi, yi))
            keep.append(order[i])
    return keep


def fetch_image(sess, url: str):
    import cv2

    try:
        r = sess.get(url, timeout=25)
        if r.status_code == 200:
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            return None if img is None else cv2.resize(img, (512, 512))
    except Exception:
        return None
    return None


def embed_dense(refs: list[dict], model, batch: int, cache: Path):
    """Download + embed in batches, keeping only embeddings (float16) and the
    ids/positions of the images that downloaded. Resumable via ``cache``."""
    import requests

    done_ids, embs, lls = [], [], []
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        done_ids, embs, lls = list(z["ids"]), [z["emb"]], [z["latlon"]]
    done = set(map(str, done_ids))
    todo = [r for r in refs if str(r["id"]) not in done]
    sess = requests.Session()
    t0 = time.perf_counter()
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        with ThreadPoolExecutor(max_workers=16) as ex:
            imgs = list(ex.map(lambda r: fetch_image(sess, r["url"]), chunk))
        ok = [(r, im) for r, im in zip(chunk, imgs) if im is not None]
        if ok:
            e = model.embed([im for _, im in ok]).astype(np.float16)
            embs.append(e)
            lls.append(np.array([[r["lat"], r["lon"]] for r, _ in ok]))
            done_ids.extend(str(r["id"]) for r, _ in ok)
        n = len(done_ids)
        if (k // batch) % 10 == 0 or k + batch >= len(todo):
            np.savez(cache, ids=np.array(done_ids), emb=np.concatenate(embs),
                     latlon=np.concatenate(lls))
            rate = (k + len(chunk)) / max(time.perf_counter() - t0, 1e-6)
            print(f"  dense refs {n:6d}/{len(refs)} ({rate:.1f}/s)", flush=True)
    return np.array(done_ids), np.concatenate(embs).astype(np.float32), np.concatenate(lls)


def main(argv=None) -> int:
    import benchmark as B
    import compare_vpr_backbones as C
    import torch

    import threading

    import requests

    from monocular_osm.kartaview_vpr import _mly_lookup_urls, _mly_tile_listing

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clip", default="kitti_0009")
    ap.add_argument("--spacing", type=float, default=75.0)
    ap.add_argument("--run-name", default="v2", help="whose archived pass-1 disc/sparse set")
    ap.add_argument("--archive-root", type=Path, default=Path("E:/monocular-osm-archive"))
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "dense_pilot")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("MLY_TOKEN"):
        raise SystemExit("MLY_TOKEN is not set in this environment")

    manifest = B.load_manifest(ROOT / "benchmarks" / "manifest.json")
    e = next(c for c in manifest["clips"] if c["id"] == args.clip)
    slug_dirs = B.candidate_slug_dirs(e, e["city"], ROOT / "data")
    # Archives live on the archive drive (older runs) or in place in the clip's
    # data dir (the lean runner keeps metadata only, no pixels).
    rep = next((base / "_benchmark_cold_vpr" / args.run_name / e["id"] / "rep0"
                for d in slug_dirs for base in (args.archive_root / d.name, d)
                if (base / "_benchmark_cold_vpr" / args.run_name / e["id"] / "rep0"
                    / "mapillary").is_dir()), None)
    if rep is None:
        raise SystemExit(f"no archived {args.run_name} pass-1 set for {args.clip}")
    sparse = C.archived_refs(rep, "mapillary")
    meta = json.loads(sorted((rep / "mapillary").glob("mly_ref_meta_*.json"))[0]
                      .read_text(encoding="utf-8"))
    sig = meta["signature"]
    s_ids = [str(r["id"]) for r in meta["refs"]]
    centre, radius = sig["center"], float(sig["radius_m"])
    print(f"{args.clip}: pass-1 disc {centre} r={radius:.0f} m; sparse set {len(s_ids)} refs",
          flush=True)

    # 1. the disc's images, listed the way the pipeline lists them; URLs are
    #    looked up only for the thinned set (and the sparse control below)
    t0 = time.perf_counter()
    token, sess = os.environ["MLY_TOKEN"], requests.Session()
    stats, lock = {"tiles": 0, "split": 0, "lost": 0}, threading.Lock()

    def count(key, n=1):
        with lock:
            stats[key] += n

    listed = _mly_tile_listing(tuple(centre), radius, token, sess, count, stats)
    pool = [{"id": i, "lat": v[0], "lon": v[1]} for i, v in listed.items()]
    print(f"  listed {len(pool)} images ({time.perf_counter() - t0:.0f} s; {stats})", flush=True)
    dense = thin(pool, centre, args.spacing)
    urls = _mly_lookup_urls([r["id"] for r in dense], token, sess)
    dense = [{**r, "url": urls[r["id"]]} for r in dense if r["id"] in urls]
    print(f"  thinned to {len(dense)} at {args.spacing:.0f} m spacing (with URLs)", flush=True)

    model = C.Backbone("megaloc", C.SAGE_SHA_DEFAULT)
    d_ids, d_emb, d_ll = embed_dense(dense, model, args.batch,
                                     args.out / f"dense_{args.clip}_{int(args.spacing)}m.npz")
    if sparse is not None:                       # archived pixels: embed those
        _ids, s_raw, s_ll, _sig, _ = sparse
        s_emb = model.embed(s_raw).astype(np.float32)
    else:                                        # metadata-only archive: re-download
        by_id = {str(r["id"]): r for r in pool}
        s_urls = _mly_lookup_urls([i for i in s_ids if i in by_id], token, sess)
        s_refs = [{**by_id[i], "url": s_urls[i]} for i in s_ids if i in s_urls]
        print(f"  sparse set: {len(s_refs)}/{len(s_ids)} ids re-listed; downloading", flush=True)
        _i, s_emb, s_ll = embed_dense(s_refs, model, args.batch,
                                      args.out / f"sparse_{args.clip}.npz")

    video = (ROOT / e["video"]["path"]) if "path" in e["video"] else next(
        d / "input.mp4" for d in slug_dirs if (d / "input.mp4").exists())
    start, end = (float(x) for x in e["vo_segment"].split(":"))
    frames, qt = C.query_frames(video, start, end, 80)
    wps = json.loads((ROOT / e["gt"]).read_text(encoding="utf-8"))["waypoints"]
    q_gt = C.gt_at(wps, qt)
    route = np.array([[w["lat"], w["lon"]] for w in wps])
    q_emb = model.embed(frames).astype(np.float32)

    res = {"clip": args.clip, "disc": {"center": centre, "radius_m": radius},
           "spacing_m": args.spacing, "pool": len(pool), "listing": stats}
    for name, emb, ll in (("sparse_v2", s_emb, s_ll), ("dense", d_emb, d_ll)):
        sc = C.score(q_emb @ emb.T, ll, q_gt, route)
        sc["n_refs"] = int(len(ll))
        sc["oracle@50m"] = C.oracle_coverage(ll, q_gt, 50.0)
        res[name] = sc
        print(f"  {name:10s} refs {len(ll):6d}  oracle@50 {sc['oracle@50m']:.2f}  "
              f"R@1 50 m {sc['r1@50m']:.2f}  R@1 100 m {sc['r1@100m']:.2f}  "
              f"retrieval-centre error {sc['centre_err_m']:.0f} m", flush=True)
    (args.out / f"results_{args.clip}_{int(args.spacing)}m.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
