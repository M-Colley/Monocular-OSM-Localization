"""Which VPR (model x aggregation) gives the tightest coarse prior on Ulm?

The anchor-primary placement is capped by the VPR prior's accuracy. Memory records
the OLD EigenPlaces prior at ~53 m from the GT route; the MegaLoc upgrade + robust
aggregation currently lands ~141-360 m — so the 'upgrade' may have regressed it.
This isolates model and aggregation on the CACHED KartaView refs + Ulm frames and
measures each prior's distance to the GT route, so we can lock in the best config.

The 'to route' metric is the point-to-segment distance to the GT route POLYLINE
(consecutive waypoint pairs, local metric projection) — NOT the nearest sparse
waypoint, whose ~250 m quantization noise is the same magnitude as the
differences being compared. Query frames mirror the production path:
``kartaview_vpr_prior`` samples ``n_query=40`` frames uniformly from the
analyzed vo-segment, so we sample 40 frames from the GT file's ``vo_segment``.
"""

from __future__ import annotations

import json
import math

import numpy as np

CACHE = "data/local-e341fb8389af-input-ulm-germany/kartaview"
VIDEO = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4"
GT = "ground_truth/ulm_ULl8s4qydrk.json"
N_QUERY = 40  # production default: kartaview_vpr_prior(n_query=40)


def _dist_m(a_lat, a_lon, b_lat, b_lon):
    return math.hypot((a_lat - b_lat) * 111320.0,
                      (a_lon - b_lon) * 111320.0 * math.cos(math.radians(a_lat)))


def route_polyline_dist_m(lat, lon, waypoints):
    """Distance (m) from a point to the GT route POLYLINE.

    Point-to-segment over consecutive waypoint pairs in a local
    equirectangular metric projection — reuses the evaluator's
    point-to-polyline helper so the metric matches the pipeline's GT
    evaluation instead of quantizing to the nearest sparse waypoint.
    """
    from src.evaluator import _segment_to_polyline_distance

    lat0 = float(np.mean([w["lat"] for w in waypoints]))
    c = 111320.0 * math.cos(math.radians(lat0))
    poly = np.array([[w["lon"] * c, w["lat"] * 111320.0] for w in waypoints])
    p = np.array([lon * c, lat * 111320.0])
    return float(_segment_to_polyline_distance(p, poly))


def _parse_segment(seg: str | None) -> tuple[float, float | None]:
    """'0:420' -> (0.0, 420.0); missing/open end -> None."""
    if not seg:
        return 0.0, None
    start_s, _, end_s = seg.partition(":")
    start = float(start_s) if start_s else 0.0
    end = float(end_s) if end_s else None
    return start, end


def _query_times(seg: str | None, duration_sec: float, n_query: int = N_QUERY):
    """Sample times the way production does: uniformly across the analyzed
    vo-segment (not across the whole video)."""
    start, end = _parse_segment(seg)
    if end is None or end > duration_sec > 0:
        end = duration_sec
    return np.linspace(start, max(start, end), n_query)


def _embed(model, imgs, device):
    import torch

    out = []
    for i in range(0, len(imgs), 24):
        batch = torch.stack(imgs[i:i + 24]).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            f = model(batch).float()
        out.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(out)


def _load(name, device):
    import torch

    if name == "megaloc":
        return torch.hub.load("gmberton/MegaLoc", "get_trained_model", verbose=False).to(device).eval()
    return torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                          backbone="ResNet50", fc_output_dim=2048, verbose=False).to(device).eval()


def _agg(ref_xy, top1, maxsim, kind):
    from src.kartaview_vpr import _geometric_median

    if kind == "median":
        return tuple(float(x) for x in np.median(ref_xy[top1], axis=0))
    if kind == "geomed":
        return tuple(float(x) for x in _geometric_median(ref_xy[top1]))
    # confidence-thresholded, weighted geometric median at percentile `pct`
    pct = {"conf": 60, "conf80": 80, "conf+mad": 60}[kind]
    keep = maxsim >= float(np.percentile(maxsim, pct))
    if int(keep.sum()) < 5:
        keep = np.ones(len(maxsim), bool)
    pts = ref_xy[top1[keep]]; wts = maxsim[keep]
    p = _geometric_median(pts, weights=wts)
    if kind == "conf+mad":
        # reject spatially isolated matches (> 2.5 MAD from the median), refit
        dm = 111320.0
        d = np.hypot((pts[:, 0] - p[0]) * dm,
                     (pts[:, 1] - p[1]) * dm * math.cos(math.radians(p[0])))
        mad = np.median(np.abs(d - np.median(d))) + 1e-6
        good = d <= np.median(d) + 2.5 * 1.4826 * mad
        if good.sum() >= 5:
            p = _geometric_median(pts[good], weights=wts[good])
    return float(p[0]), float(p[1])


def main():
    import cv2
    import torch

    import src.kartaview_vpr as _K
    from src.kartaview_vpr import _prep

    _K._MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _K._STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    device = "cuda"
    d = np.load(f"{CACHE}/ref_imgs.npz", allow_pickle=True)
    raw = d["raw"]
    if "ref_xy" in d.files:
        # current self-contained cache (stores kept photos' lat/lon directly)
        ref_xy = np.asarray(d["ref_xy"], float)
    else:
        # legacy cache: keep-indices into ref_meta.json (plain list, or the
        # newer {"signature":..., "refs":[...]} wrapper)
        keep = d["keep"].tolist()
        meta = json.load(open(f"{CACHE}/ref_meta.json"))
        refs = meta["refs"] if isinstance(meta, dict) else meta
        ref_xy = np.array([[refs[k]["lat"], refs[k]["lon"]] for k in keep])
    ref_imgs = [_prep(raw[i]) for i in range(len(raw))]

    gt = json.load(open(GT))
    wps = gt["waypoints"]
    glat = np.mean([w["lat"] for w in wps]); glon = np.mean([w["lon"] for w in wps])

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Query frames from the analyzed segment only (what the pipeline sees),
    # not the whole video.
    q_imgs = []
    for t in _query_times(gt.get("vo_segment"), n / fps if n else 0.0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(t * fps), max(n - 1, 0)))
        ok, f = cap.read()
        if ok:
            q_imgs.append(_prep(f))
    cap.release()

    def route_dist(lat, lon):
        return route_polyline_dist_m(lat, lon, wps)

    print(f"{len(ref_xy)} refs, {len(q_imgs)} query frames "
          f"(segment {gt.get('vo_segment')})\n")
    print(f"{'model':<12}{'agg':<10}{'to route':>10}{'to centre':>11}")
    for model_name in ["eigenplaces", "megaloc"]:
        model = _load(model_name, device)
        ref_emb = _embed(model, ref_imgs, device)
        q_emb = _embed(model, q_imgs, device)
        sims = (q_emb @ ref_emb.T).numpy()
        top1 = sims.argmax(1); maxsim = sims.max(1)
        for kind in ["conf", "conf80", "conf+mad"]:
            lat, lon = _agg(ref_xy, top1, maxsim, kind)
            print(f"{model_name:<12}{kind:<10}{route_dist(lat, lon):>8.0f} m{_dist_m(lat, lon, glat, glon):>9.0f} m")
        del model, ref_emb; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
