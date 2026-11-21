"""Pilot: verify literal shop-sign sites with dense local imagery.

When OCR finds a co-visible shop pair (a "conjunction") that the city has at
more than one site, the assisted image-only decode searches 3 km around each
site with a 1,500-image sample, which can leave the signed shops themselves
almost unsampled (Atlanta y66: one image within 200 m of the true site, 25 at
the false one). This pilot lists EVERY Mapillary image within ``radius`` of
each site's shops (coverage tiles), keeps a deterministic sha1 sample, embeds
it with MegaLoc exactly as the pipeline does, and reports per site:
the sign-window frames' best similarity, and the full-window image-only decode
(reference-mean emission, the policy's own decoder) over the union of the
sites' dense pools. No ground truth is used except to score.

    python scripts/pilot_sign_site_verification.py <video> <clip_id> <t0> <t1> <out.json> \
        --site NAME LAT LON [--site ...] [--radius 300] [--per-site 600] [--sign-window A B]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def mly_token() -> str:
    import os
    if os.environ.get("MLY_TOKEN"):
        return os.environ["MLY_TOKEN"]
    import winreg
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
    return winreg.QueryValueEx(key, "MLY_TOKEN")[0]


def site_pool(centre, radius, per_site, token, sess):
    """Deterministic sha1 sample of every image within ``radius`` (coverage tiles)."""
    import monocular_osm.kartaview_vpr as kv
    from monocular_osm.geo import geodesic_distance_m
    stats, lock = {"tiles": 0, "split": 0, "lost": 0}, threading.Lock()

    def count(key, n=1):
        with lock:
            stats[key] += n
    listed = kv._mly_tile_listing(centre, radius, token, sess, count, stats)
    near = [i for i, v in listed.items() if geodesic_distance_m(centre[0], centre[1], v[0], v[1]) <= radius]
    keep = sorted(near, key=lambda i: hashlib.sha1(i.encode()).hexdigest())[:per_site]
    urls = kv._mly_lookup_urls(keep, token, sess)
    return [(i, listed[i], urls[i]) for i in keep if i in urls], len(near), stats


def fetch(sess, url):
    import cv2
    try:
        r = sess.get(url, timeout=25)
        if r.status_code == 200:
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            return None if img is None else cv2.resize(img, (512, 512))
    except Exception:
        return None
    return None


def main(argv=None) -> int:
    import cv2
    import psutil
    import requests
    import torch

    import monocular_osm.kartaview_vpr as kv
    from monocular_osm import poi_image_track as T
    from monocular_osm.geo import LocalFrame
    from monocular_osm.metrics import score_route

    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video"); ap.add_argument("clip"); ap.add_argument("t0", type=float); ap.add_argument("t1", type=float)
    ap.add_argument("out", type=Path)
    ap.add_argument("--site", nargs=3, action="append", metavar=("NAME", "LAT", "LON"), required=True)
    ap.add_argument("--radius", type=float, default=300.0)
    ap.add_argument("--per-site", type=int, default=600)
    ap.add_argument("--sign-window", nargs=2, type=float, default=None)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    token, sess = mly_token(), requests.Session()
    kv._MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    kv._STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    kv._resolve_backbone("megaloc", "cpu")

    # 80 query frames over the window, as the pipeline samples them
    times = np.linspace(args.t0, args.t1, 80)
    cap = cv2.VideoCapture(args.video)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, img = cap.read()
        frames.append(img if ok else frames[-1])
    cap.release()
    q_warp = kv._embed("cpu", [kv._prep(f) for f in frames]).numpy()
    crops = [kv._prep(kv._square_query_crop(f, k)) for f in frames for k in range(3)]
    q_crop = kv._embed("cpu", crops).numpy().reshape(len(frames), 3, -1)

    ids, lls, embs, site_of, report = [], [], [], [], {"clip": args.clip, "sites": {}}
    for s, (name, lat, lon) in enumerate(args.site):
        pool, n_near, stats = site_pool((float(lat), float(lon)), args.radius, args.per_site, token, sess)
        with ThreadPoolExecutor(16) as ex:
            imgs = list(ex.map(lambda r: fetch(sess, r[2]), pool))
        ok = [(r, im) for r, im in zip(pool, imgs) if im is not None]
        e = kv._embed("cpu", np.stack([im for _, im in ok])).numpy()
        ids += [r[0] for r, _ in ok]; lls += [r[1] for r, _ in ok]; embs.append(e); site_of += [s] * len(ok)
        report["sites"][name] = {"centre": [float(lat), float(lon)], "images_within_radius": n_near,
                                 "embedded": len(ok), "listing": stats}
        print(f"{name}: {n_near} images within {args.radius:.0f} m, embedded {len(ok)}", flush=True)
    E, LL, site_of = np.concatenate(embs), np.array(lls, float), np.array(site_of)

    sign = (np.where((times >= args.sign_window[0]) & (times <= args.sign_window[1]))[0]
            if args.sign_window else np.arange(len(times)))
    for mode, Q in (("warp", q_warp), ("crops", None)):
        S = (q_warp @ E.T) if mode == "warp" else np.max(np.einsum("qkd,rd->qkr", q_crop, E), axis=1)
        C = S - S.mean(axis=0, keepdims=True)
        for s, (name, _, _) in enumerate(args.site):
            cols = np.where(site_of == s)[0]
            report["sites"][name][f"sign_frames_best_raw_{mode}"] = float(S[np.ix_(sign, cols)].max(axis=1).mean())
            report["sites"][name][f"sign_frames_best_centred_{mode}"] = float(C[np.ix_(sign, cols)].max(axis=1).mean())
        frame = LocalFrame(float(LL[:, 0].mean()), float(LL[:, 1].mean()))
        xy = np.column_stack(frame.to_xy(LL[:, 0], LL[:, 1]))
        path, objective = T._decode(np.ascontiguousarray(C), xy, times)
        sel = xy[path]
        sm = np.array([np.median(sel[max(0, i - 2):i + 3], axis=0) for i in range(len(sel))])
        route = np.column_stack(frame.to_latlon(sm[:, 0], sm[:, 1]))
        chosen_site = int(np.bincount(site_of[path]).argmax())
        entry = {"objective": objective, "chosen_site": args.site[chosen_site][0]}
        try:
            man = {c["id"]: c for c in json.loads((ROOT / "benchmarks/manifest.json").read_text(encoding="utf-8"))["clips"]}
            gt = np.array([[w["lat"], w["lon"]] for w in
                           json.loads((ROOT / man[args.clip]["gt"]).read_text(encoding="utf-8"))["waypoints"]])
            entry["route_symmetric_m"] = round(score_route(route, gt).symmetric_mean_m, 1)
        except Exception as exc:
            entry["score_error"] = str(exc)
        report[f"union_decode_{mode}"] = entry
        print(mode, json.dumps(entry), flush=True)
    for name, v in report["sites"].items():
        print(name, {k: round(x, 4) for k, x in v.items() if k.startswith("sign_frames")}, flush=True)

    # Two-step: choose the site whose dense imagery best matches the sign frames
    # (raw warp), then decode the whole window over that site's dense pool plus
    # the route-scale image-only pool (3 km, 1,500 images) around it.
    best = max(range(len(args.site)), key=lambda s: report["sites"][args.site[s][0]]["sign_frames_best_raw_warp"])
    name, lat, lon = args.site[best]
    cache = args.out.with_suffix("") / f"route_pool_{name}"
    route_refs = [r for r in kv._fetch_refs_mapillary((float(lat), float(lon)), 3000.0, str(cache), cap=1500, token=token)
                  if r.get("url")]
    with ThreadPoolExecutor(16) as ex:
        imgs = list(ex.map(lambda r: fetch(sess, r["url"]), route_refs))
    ok = [(r, im) for r, im in zip(route_refs, imgs) if im is not None]
    E2 = kv._embed("cpu", np.stack([im for _, im in ok])).numpy()
    dense = np.where(site_of == best)[0]
    E_all = np.concatenate([E[dense], E2]); LL_all = np.concatenate([LL[dense], np.array([[r["lat"], r["lon"]] for r, _ in ok])])
    report["two_step"] = {"chosen_site": name, "dense_refs": int(len(dense)), "route_pool_refs": int(len(ok))}
    frame = LocalFrame(float(LL_all[:, 0].mean()), float(LL_all[:, 1].mean()))
    xy = np.column_stack(frame.to_xy(LL_all[:, 0], LL_all[:, 1]))
    for mode in ("warp", "crops"):
        S = (q_warp @ E_all.T) if mode == "warp" else np.max(np.einsum("qkd,rd->qkr", q_crop, E_all), axis=1)
        C = S - S.mean(axis=0, keepdims=True)
        path, objective = T._decode(np.ascontiguousarray(C), xy, times)
        sel = xy[path]
        sm = np.array([np.median(sel[max(0, i - 2):i + 3], axis=0) for i in range(len(sel))])
        route = np.column_stack(frame.to_latlon(sm[:, 0], sm[:, 1]))
        entry = {"objective": objective}
        try:
            entry["route_symmetric_m"] = round(score_route(route, gt).symmetric_mean_m, 1)
        except Exception as exc:
            entry["score_error"] = str(exc)
        report["two_step"][mode] = entry
        print("two-step", name, mode, json.dumps(entry), flush=True)
    np.savez_compressed(args.out.with_suffix(".npz"), dense_emb=E, dense_latlon=LL, dense_site=site_of,
                        route_emb=E2, q_warp=q_warp, q_crop=q_crop, times=times)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
