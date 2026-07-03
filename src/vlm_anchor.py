"""VLM district/landmark anchor (Gemma 4, multimodal).

A coarse, shape-INDEPENDENT location prior of a DIFFERENT class than VPR: instead
of image retrieval, a vision-language model reads dashcam frames and infers the
most likely neighbourhood/district plus any readable street/shop/landmark names,
from world knowledge. Geocoding the consensus gives a centre that feeds the same
anchor-primary path as VPR (src/pipeline.py [10a*]) — useful when VPR has no
coverage (e.g. low-res clips with no KartaView references nearby).

Per user: uses google/gemma-4-E2B-it (local multimodal Gemma 4), not Qwen3-VL.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# E2B ("effective 2B") fits a 16 GB GPU reliably; E4B (16 GB weights) OOMs the
# caching-allocator warmup on a 16 GB card. Both are Gemma 4.
MODEL_ID = "google/gemma-4-E2B-it"
# Ask for STREET (most specific -> route-accurate when right), DISTRICT (coarse
# fallback) and all readable TEXT (geocodable POI/street tokens). Rigid 3-line
# format so the parser is reliable; the parser is also lenient (markdown/preamble).
# V1 prompt (LIVE-validated on Ulm 4K, 2026-07): the original prompt made the
# model (a) loop-repeat a word to fill the TEXT budget (BÜCHEREI x19, "1,2,3…31")
# and (b) answer DISTRICT with the CITY itself (a tautology that just geocodes to
# the city centroid). This variant asks for DISTINCT words, one street PLATE only,
# and a *sub*-district (explicitly not the city name), and is generated with a
# repetition_penalty (see _ask) that eliminated the loops and recovered a genuine
# street ("Salzstadel") the original never produced.
_PROMPT = (
    "You are a geolocation expert reading a dashcam frame from {city}.\n"
    "Report only what is clearly legible; do NOT guess or repeat words.\n"
    "Give the SINGLE most specific street/square name and the neighbourhood.\n"
    "Output EXACTLY three lines:\n"
    "TEXT: <up to 5 distinct legible sign words, comma-separated, or none>\n"
    "STREET: <one street/square name if a street PLATE is legible, else unknown>\n"
    "DISTRICT: <the neighbourhood WITHIN {city} (not '{city}' itself), else unknown>"
)

_model = None
_proc = None
_BAD = {"", "unknown", "none", "n/a", "na", "not visible", "not identifiable"}


@dataclass
class VlmAnchor:
    lat: float
    lon: float
    label: str           # the geocoded place string that resolved
    street_votes: dict
    district_votes: dict
    text_votes: dict


def _load():
    global _model, _proc
    if _model is not None:
        return
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    _proc = AutoProcessor.from_pretrained(MODEL_ID)
    # device_map="auto" + low_cpu_mem_usage spreads onto CPU if the GPU is full
    # (E-series fits 16 GB, but a fragmented GPU forces partial offload).
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    try:
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, device_map="auto", **kw).eval()
    except Exception:
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, device_map="cpu", **kw).eval()


def _ask(pil: Image.Image, city: str) -> str:
    import torch
    dev = getattr(_model, "device", None) or next(_model.parameters()).device
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text", "text": _PROMPT.format(city=city)}]}]
    inp = _proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        # repetition_penalty > 1 is required: greedy decode alone loops a word
        # to fill the token budget (BÜCHEREI x19), poisoning the vote counts.
        out = _model.generate(**inp, max_new_tokens=120, do_sample=False,
                              repetition_penalty=1.3)
    return _proc.batch_decode(
        out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def _field(text: str, key: str) -> str | None:
    """Lenient extraction of a 'KEY: value' line (tolerates markdown/bullets)."""
    m = re.search(rf"(?im)^[\s*>-]*{key}\s*[:\-]\s*(.+)$", text)
    if not m:
        return None
    v = m.group(1).strip().strip("*` ").rstrip(".")
    return None if v.lower() in _BAD else v


def _parse(text: str) -> tuple[str | None, str | None, list[str]]:
    street = _field(text, "STREET")
    district = _field(text, "DISTRICT")
    raw = _field(text, "TEXT") or ""
    texts = [t.strip() for t in re.split(r"[,;/]", raw)
             if t.strip() and t.strip().lower() not in _BAD and len(t.strip()) > 2]
    return street, district, texts


def _latlon_dist_km(a_lat, a_lon, b_lat, b_lon) -> float:
    import math
    dy = (a_lat - b_lat) * 111.32
    dx = (a_lon - b_lon) * 111.32 * math.cos(math.radians(0.5 * (a_lat + b_lat)))
    return float(math.hypot(dx, dy))


# ---------------------------------------------------------------------------
# Here-sign vs direction-sign classification
# ---------------------------------------------------------------------------

# A tight, format-locked question. The model must decide whether the named
# text labels the place the camera IS (a street nameplate, a shopfront, a
# building name) or points ELSEWHERE (a wayfinding/motorway sign with arrows,
# town names, distances, on a green/blue panel). This is the London 'Holborn'
# fix: 'Holborn' on a directional finger-sign 1.5 km off-route must classify
# 'direction' so its geocoded anchor is down-weighted, while a genuine street
# plate on the route classifies 'here'.
#
# LIVE-tuned (2026-07, London+Ulm 4K). A neutral "HERE or DIRECTION?" was too
# HERE-biased: it read 'Holborn'/'Bloomsbury' on a wayfinding gantry as 'here'
# (the exact failure). This wording — "a PLACE THIS SIGN POINTS TO" vs "the name
# of THIS very spot", with an explicit tie-break toward DIRECTION for a place
# name that *could* be pointing — flips Holborn/Bloomsbury/Euston/Polizei-
# präsidium/Parkleitsystem to 'direction' while keeping genuine shopfronts
# ('HOTEL', 'FINE ART') 'here' (6/6 on the labeled probe, at crop margin 1.2).
_SIGN_PROMPT = (
    "This is a cropped dashcam image around the text \"{text}\".\n"
    "Is \"{text}\" the name of a PLACE THIS SIGN POINTS TO (a directional / "
    "wayfinding / motorway guide sign — has arrows, a coloured panel, lists towns "
    "or districts with directions), or is it the name of THIS very spot (a street "
    "nameplate fixed to a wall, or a shopfront)?\n"
    "If it is a place name on a sign that could be pointing somewhere, answer "
    "DIRECTION. Only answer HERE if it is clearly a nameplate or shopfront AT this "
    "location.\n"
    "Answer with EXACTLY one word: HERE or DIRECTION."
)

_HERE_RE = re.compile(r"\bhere\b", re.IGNORECASE)
_DIR_RE = re.compile(r"\bdirection\b", re.IGNORECASE)


def _crop_bbox(frame_bgr, bbox, *, margin: float = 1.2):
    """Crop ``frame_bgr`` around ``bbox`` (x_min,y_min,x_max,y_max) with a
    generous relative ``margin`` so the sign's shape/colour/arrows are visible,
    not just the glyphs. Returns a PIL RGB image, or the whole frame if the
    box is missing/degenerate."""
    h, w = frame_bgr.shape[:2]
    if bbox is None:
        return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    x0, y0, x1, y1 = (float(v) for v in bbox)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    mx, my = margin * bw, margin * bh
    x0 = int(max(0, x0 - mx)); y0 = int(max(0, y0 - my))
    x1 = int(min(w, x1 + mx)); y1 = int(min(h, y1 + my))
    if x1 <= x0 or y1 <= y0:
        return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    crop = frame_bgr[y0:y1, x0:x1]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def _ask_sign(pil: Image.Image, text: str) -> str:
    import torch
    dev = getattr(_model, "device", None) or next(_model.parameters()).device
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text", "text": _SIGN_PROMPT.format(text=text)}]}]
    inp = _proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = _model.generate(**inp, max_new_tokens=8, do_sample=False)
    return _proc.batch_decode(
        out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def _classify_reply(reply: str) -> str:
    """Lenient parse of a HERE/DIRECTION reply -> 'here'|'direction'|'other'.
    'direction' wins ties (the conservative choice: a misread that suppresses
    a genuine here-anchor is cheaper than trusting a direction sign)."""
    is_dir = bool(_DIR_RE.search(reply))
    is_here = bool(_HERE_RE.search(reply))
    if is_dir:
        return "direction"
    if is_here:
        return "here"
    return "other"


def classify_sign_types(frames_bgr, detections, *, ask_fn=None):
    """Classify each OCR detection as 'here' | 'direction' | 'other'.

    ``detections`` is a list of objects with ``.text``, ``.bbox`` (pixel
    ``(x_min,y_min,x_max,y_max)`` or ``None``) and either ``.frame_idx`` or
    ``.t_sec`` — anything with those attributes works (a :class:`SceneText`
    plus a resolved frame index, or a lightweight record). ``frames_bgr`` is
    the list the indices reference; when a detection carries a bbox but no
    valid frame index the whole first frame is used as a fallback.

    The SAME loaded Gemma model answers a tight HERE/DIRECTION question per
    detection (deterministic, ``do_sample=False``). ``ask_fn`` is an
    injection point for tests: ``ask_fn(pil, text) -> reply_str``. Returns a
    list of labels aligned to ``detections``.
    """
    if ask_fn is None:
        _load()
        ask_fn = _ask_sign
    out: list[str] = []
    for d in detections:
        text = getattr(d, "text", "") or ""
        bbox = getattr(d, "bbox", None)
        fi = getattr(d, "frame_idx", None)
        if fi is None or not (0 <= int(fi) < len(frames_bgr)):
            fi = 0
        frame = frames_bgr[int(fi)] if frames_bgr else None
        if frame is None:
            out.append("other")
            continue
        try:
            reply = ask_fn(_crop_bbox(frame, bbox), text)
        except Exception:
            out.append("other")
            continue
        out.append(_classify_reply(reply))
    return out


def vlm_district_anchor(frames_bgr, city: str, *, geocode_fn=None,
                        n_query: int = 6, min_votes: int = 2,
                        min_street_votes: int = 1,
                        use_text_fallback: bool = False,
                        max_km_from_city: float = 15.0) -> VlmAnchor | None:
    """Read frames with Gemma 4, vote on street/district/readable-text, then
    geocode the consensus in priority order (street = most specific/route-accurate
    first, then district, then text tokens), bounded to the city. None if nothing
    geocodes.

    Guards keep a hallucination from relocating the answer to another city
    (Nominatim's unstructured search drops unmatched tokens, so "Marktplatz,
    Erbach" happily resolves to the wrong Erbach), and any geocode farther than
    ``max_km_from_city`` from the bare city's centroid is rejected. Candidate
    priority, most trustworthy first:

    1. Streets with ``>= min_street_votes`` (default 1). A word in the STREET
       slot already passed the model's strict "is this a street PLATE" test and
       is geocoded *as a street*, so a single legible plate is worth more than a
       repeated free-text token. LIVE-validated: the Ulm run read "Salzstadel"
       on a parking sign exactly once — geocodes 81 m from the route — while the
       only multi-token noise ("WILL") sat 2.2 km off. A ``min_street_votes``
       of 1 keeps that win; raise it to demand corroboration.
    2. Districts with ``>= min_votes`` (default 2 — a district is coarse, so
       insist on agreement). The city name itself is never a useful district
       (it just geocodes to the centroid); the prompt tells the model not to
       emit it, and it is dropped here if it slips through.
    3. Raw TEXT tokens — only when ``use_text_fallback`` is True. These carry NO
       vote guard and are pure OCR-of-the-VLM's-reading, so on their own they
       relocate the anchor to whatever a hallucinated word geocodes to (the
       "WILL" → 2.2 km failure). Off by default; enable only where any prior
       beats none.
    """
    if not frames_bgr:
        return None
    _load()
    idx = np.linspace(0, len(frames_bgr) - 1, min(n_query, len(frames_bgr))).astype(int)
    streets: Counter = Counter()
    districts: Counter = Counter()
    texts: Counter = Counter()
    for i in idx:
        pil = Image.fromarray(cv2.cvtColor(frames_bgr[int(i)], cv2.COLOR_BGR2RGB))
        try:
            st, di, tx = _parse(_ask(pil, city))
        except Exception:
            continue
        if st:
            streets[st] += 1
        if di:
            districts[di] += 1
        for t in tx:
            texts[t] += 1

    if geocode_fn is None:
        import osmnx as ox

        def geocode_fn(q):
            return tuple(ox.geocode(q))

    # Reference centre for the "bounded to the city" check: geocode the bare
    # city once. If even that fails we cannot bound anything — bail out rather
    # than risk an unbounded hallucination becoming the anchor.
    try:
        city_lat, city_lon = geocode_fn(city)
    except Exception:
        return None

    # Priority: streets (>= min_street_votes, route-accurate), then districts
    # (>= min_votes, coarse so insist on agreement), then — only if opted in —
    # raw text tokens (no guard). Each is geocoded "<name>, <city>" so it
    # resolves to the local one, and anything > max_km_from_city away is
    # rejected. The bare city name is stripped from the district slot: it just
    # geocodes to the centroid and is never a useful sub-district anchor.
    _city_fold = city.split(",")[0].strip().casefold()
    candidates = ([s for s, c in streets.most_common(3) if c >= min_street_votes]
                  + [d for d, c in districts.most_common(2)
                     if c >= min_votes and d.casefold() != _city_fold])
    if use_text_fallback:
        candidates += [t for t, _ in texts.most_common(5)]
    seen = set()
    for cand in candidates:
        k = cand.casefold()
        if k in seen:
            continue
        seen.add(k)
        try:
            lat, lon = geocode_fn(f"{cand}, {city}")
        except Exception:
            continue
        if _latlon_dist_km(float(lat), float(lon), float(city_lat),
                           float(city_lon)) > max_km_from_city:
            continue  # resolved to a same-named place in another city
        return VlmAnchor(lat=float(lat), lon=float(lon), label=cand,
                         street_votes=dict(streets), district_votes=dict(districts),
                         text_votes=dict(texts))
    return None
