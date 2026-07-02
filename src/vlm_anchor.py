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
_PROMPT = (
    "This is a dashcam frame from {city}. Identify the location as precisely as "
    "possible from visible street-name plates, shop/business names and landmarks.\n"
    "Output EXACTLY these three lines and nothing else:\n"
    "TEXT: <every readable sign/shop/street word, comma-separated, or none>\n"
    "STREET: <the specific street or square name if identifiable, else unknown>\n"
    "DISTRICT: <the neighbourhood/district name, else unknown>"
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
        out = _model.generate(**inp, max_new_tokens=120, do_sample=False)
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


def vlm_district_anchor(frames_bgr, city: str, *, geocode_fn=None,
                        n_query: int = 6, min_votes: int = 2,
                        max_km_from_city: float = 15.0) -> VlmAnchor | None:
    """Read frames with Gemma 4, vote on street/district/readable-text, then
    geocode the consensus in priority order (street = most specific/route-accurate
    first, then district, then text tokens), bounded to the city. None if nothing
    geocodes.

    Two guards keep a hallucination from relocating the answer to another city
    (Nominatim's unstructured search drops unmatched tokens, so "Marktplatz,
    Erbach" happily resolves to the wrong Erbach): a street/district needs
    ``min_votes`` frames agreeing before it is geocoded at all, and any geocode
    farther than ``max_km_from_city`` from the bare city's centroid is rejected.
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

    # Priority: most-voted street (route-accurate), then district, then text
    # tokens. Streets/districts need >= min_votes — a single-frame hallucination
    # must not win. Each is geocoded "<name>, <city>" so it resolves to the
    # local one, and anything landing > max_km_from_city away is rejected.
    candidates = ([s for s, c in streets.most_common(3) if c >= min_votes]
                  + [d for d, c in districts.most_common(2) if c >= min_votes]
                  + [t for t, _ in texts.most_common(5)])
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
