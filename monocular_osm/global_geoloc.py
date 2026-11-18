"""Stage 0 planet-scale city inference for videos without a known ``--city``.

When a user supplies a raw ``--video`` (or an untitled URL) with
``--infer-city-from-video`` and no ``--city``, this module estimates the most
likely metropolitan area anywhere in the world before Stage 1 retrieval begins:

1. **Metadata fast path:** Multilingual title/description parsing
   (:func:`monocular_osm.city_inference.guess_city_from_title`).
2. **Zero-shot vision-language ranking (CLIP / StreetCLIP):** Scores sampled
   frames against a curated gazetteer of major world cities across all six
   inhabited continents (Europe, North America, Latin America, Asia, Middle
   East, Africa, Oceania), aggregating log-probabilities across the clip.
3. **Optional VLM visual forensics (Gemma 4):** Reads visible scripts, driving
   side (left-hand vs right-hand traffic), plate styles, road markings, and
   signage to vote on ``City, Country`` when VLM inference is requested.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import re

import cv2
import numpy as np


@dataclass(frozen=True)
class GlobalCityGuess:
    city: str
    lat: float | None
    lon: float | None
    confidence: float
    method: str
    top_candidates: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


#: Representative global metropolitan areas spanning all inhabited continents,
#: used as zero-shot candidate prompts ("A street dashcam photo in <City>.")
#: when no city name is supplied.
WORLD_METROS: tuple[tuple[str, float, float], ...] = (
    # Europe
    ("London, UK", 51.5074, -0.1278),
    ("Manchester, UK", 53.4808, -2.2426),
    ("Glasgow, UK", 55.8642, -4.2518),
    ("Dublin, Ireland", 53.3498, -6.2603),
    ("Paris, France", 48.8566, 2.3522),
    ("Lyon, France", 45.7640, 4.8357),
    ("Marseille, France", 43.2965, 5.3698),
    ("Berlin, Germany", 52.5200, 13.4050),
    ("Munich, Germany", 48.1351, 11.5820),
    ("Hamburg, Germany", 53.5511, 9.9937),
    ("Frankfurt am Main, Germany", 50.1109, 8.6821),
    ("Cologne, Germany", 50.9375, 6.9603),
    ("Ulm, Germany", 48.4011, 9.9876),
    ("Amsterdam, Netherlands", 52.3676, 4.9041),
    ("Rotterdam, Netherlands", 51.9244, 4.4777),
    ("Brussels, Belgium", 50.8503, 4.3517),
    ("Zurich, Switzerland", 47.3769, 8.5417),
    ("Vienna, Austria", 48.2082, 16.3738),
    ("Madrid, Spain", 40.4168, -3.7038),
    ("Barcelona, Spain", 41.3874, 2.1686),
    ("Lisbon, Portugal", 38.7223, -9.1393),
    ("Rome, Italy", 41.9028, 12.4964),
    ("Milan, Italy", 45.4642, 9.1900),
    ("Warsaw, Poland", 52.2297, 21.0122),
    ("Kraków, Poland", 50.0647, 19.9450),
    ("Prague, Czech Republic", 50.0755, 14.4378),
    ("Budapest, Hungary", 47.4979, 19.0402),
    ("Bucharest, Romania", 44.4268, 26.1025),
    ("Sofia, Bulgaria", 42.6977, 23.3219),
    ("Belgrade, Serbia", 44.7866, 20.4489),
    ("Zagreb, Croatia", 45.8150, 15.9819),
    ("Athens, Greece", 37.9838, 23.7275),
    ("Istanbul, Turkey", 41.0082, 28.9784),
    ("Stockholm, Sweden", 59.3293, 18.0686),
    ("Oslo, Norway", 59.9139, 10.7522),
    ("Copenhagen, Denmark", 55.6761, 12.5683),
    ("Helsinki, Finland", 60.1699, 24.9384),
    ("Riga, Latvia", 56.9496, 24.1052),
    ("Tallinn, Estonia", 59.4370, 24.7536),
    ("Vilnius, Lithuania", 54.6872, 25.2797),
    ("Moscow, Russia", 55.7558, 37.6173),
    ("Kyiv, Ukraine", 50.4501, 30.5234),
    # North America
    ("New York, New York, USA", 40.7128, -74.0060),
    ("Los Angeles, California, USA", 34.0522, -118.2437),
    ("Chicago, Illinois, USA", 41.8781, -87.6298),
    ("Houston, Texas, USA", 29.7604, -95.3698),
    ("Phoenix, Arizona, USA", 33.4484, -112.0740),
    ("Dallas, Texas, USA", 32.7767, -96.7970),
    ("San Francisco, California, USA", 37.7749, -122.4194),
    ("Seattle, Washington, USA", 47.6062, -122.3321),
    ("Portland, Oregon, USA", 45.5152, -122.6784),
    ("Denver, Colorado, USA", 39.7392, -104.9903),
    ("Atlanta, Georgia, USA", 33.7490, -84.3880),
    ("Miami, Florida, USA", 25.7617, -80.1918),
    ("Detroit, Michigan, USA", 42.3314, -83.0458),
    ("Boston, Massachusetts, USA", 42.3601, -71.0589),
    ("Toronto, Ontario, Canada", 43.6532, -79.3832),
    ("Montreal, Quebec, Canada", 45.5017, -73.5673),
    ("Vancouver, British Columbia, Canada", 49.2827, -123.1207),
    ("Calgary, Alberta, Canada", 51.0447, -114.0719),
    ("Mexico City, Mexico", 19.4326, -99.1332),
    ("Guadalajara, Mexico", 20.6597, -103.3496),
    ("Monterrey, Mexico", 25.6866, -100.3161),
    # Latin America
    ("São Paulo, Brazil", -23.5505, -46.6333),
    ("Rio de Janeiro, Brazil", -22.9068, -43.1729),
    ("Brasília, Brazil", -15.7975, -47.8919),
    ("Curitiba, Brazil", -25.4284, -49.2733),
    ("Buenos Aires, Argentina", -34.6037, -58.3816),
    ("Santiago, Chile", -33.4489, -70.6693),
    ("Bogotá, Colombia", 4.7110, -74.0721),
    ("Medellín, Colombia", 6.2442, -75.5812),
    ("Lima, Peru", -12.0464, -77.0428),
    ("Montevideo, Uruguay", -34.9011, -56.1645),
    # East, Southeast & South Asia
    ("Tokyo, Japan", 35.6762, 139.6503),
    ("Osaka, Japan", 34.6937, 135.5023),
    ("Nagoya, Japan", 35.1815, 136.9066),
    ("Fukuoka, Japan", 33.5902, 130.4017),
    ("Sapporo, Japan", 43.0618, 141.3545),
    ("Seoul, South Korea", 37.5665, 126.9780),
    ("Busan, South Korea", 35.1796, 129.0756),
    ("Taipei, Taiwan", 25.0330, 121.5654),
    ("Kaohsiung, Taiwan", 22.6273, 120.3014),
    ("Hong Kong", 22.3193, 114.1694),
    ("Beijing, China", 39.9042, 116.4074),
    ("Shanghai, China", 31.2304, 121.4737),
    ("Shenzhen, China", 22.5431, 114.0579),
    ("Guangzhou, China", 23.1291, 113.2644),
    ("Chengdu, China", 30.5728, 104.0668),
    ("Singapore", 1.3521, 103.8198),
    ("Bangkok, Thailand", 13.7563, 100.5018),
    ("Kuala Lumpur, Malaysia", 3.1390, 101.6869),
    ("Jakarta, Indonesia", -6.2088, 106.8456),
    ("Manila, Philippines", 14.5995, 120.9842),
    ("Ho Chi Minh City, Vietnam", 10.8231, 106.6297),
    ("Hanoi, Vietnam", 21.0278, 105.8342),
    ("Mumbai, India", 19.0760, 72.8777),
    ("New Delhi, India", 28.6139, 77.2090),
    ("Bengaluru, India", 12.9716, 77.5946),
    ("Hyderabad, India", 17.3850, 78.4867),
    ("Chennai, India", 13.0827, 80.2707),
    # Middle East & Africa
    ("Dubai, UAE", 25.2048, 55.2708),
    ("Abu Dhabi, UAE", 24.4539, 54.3773),
    ("Riyadh, Saudi Arabia", 24.7136, 46.6753),
    ("Tel Aviv, Israel", 32.0853, 34.7818),
    ("Doha, Qatar", 25.2854, 51.5310),
    ("Cairo, Egypt", 30.0444, 31.2357),
    ("Casablanca, Morocco", 33.5731, -7.5898),
    ("Johannesburg, South Africa", -26.2041, 28.0473),
    ("Cape Town, South Africa", -33.9249, 18.4241),
    ("Nairobi, Kenya", -1.2921, 36.8219),
    ("Lagos, Nigeria", 6.5244, 3.3792),
    # Oceania
    ("Sydney, Australia", -33.8688, 151.2093),
    ("Melbourne, Australia", -37.8136, 144.9631),
    ("Brisbane, Australia", -27.4705, 153.0260),
    ("Perth, Australia", -31.9505, 115.8605),
    ("Auckland, New Zealand", -36.8485, 174.7633),
)


def sample_video_frames(video_path: Path | str, n_frames: int = 8,
                        start_sec: float = 0.0, end_sec: float | None = None) -> list[np.ndarray]:
    """Uniformly sample ``n_frames`` BGR images from ``video_path``."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return []
        dur = total / max(fps, 1e-3)
        t0 = max(0.0, float(start_sec))
        t1 = min(dur, float(end_sec)) if end_sec is not None else dur
        if t1 <= t0:
            t0, t1 = 0.0, dur
        times = np.linspace(t0, max(t0, t1 - 0.1), max(1, int(n_frames)))
        out = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                out.append(frame)
        return out
    finally:
        cap.release()


def _clip_rank_cities(frames_bgr: list[np.ndarray],
                      metros: tuple[tuple[str, float, float], ...] = WORLD_METROS,
                      device: str | None = None,
                      top_k: int = 5) -> GlobalCityGuess | None:
    """Zero-shot StreetCLIP / CLIP frame-to-city ranking across ``metros``."""
    if not frames_bgr or not metros:
        return None
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "geolocal/StreetCLIP"
    try:
        proc = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(dev).eval()
    except Exception:
        model_id = "openai/clip-vit-base-patch32"
        proc = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(dev).eval()

    prompts = [f"A street-level dashcam photo in {name}." for name, _, _ in metros]
    pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]

    with torch.no_grad():
        txt_in = proc(text=prompts, return_tensors="pt", padding=True, truncation=True).to(dev)
        txt_feat = model.get_text_features(**txt_in)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        img_in = proc(images=pil_imgs, return_tensors="pt").to(dev)
        img_feat = model.get_image_features(**img_in)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        scale = model.logit_scale.exp().clamp(max=100.0)
        logits = scale * (img_feat @ txt_feat.T)
        log_probs = torch.log_softmax(logits, dim=-1)
        agg_probs = torch.softmax(log_probs.mean(dim=0), dim=-1).cpu().numpy()

    order = np.argsort(-agg_probs)[:max(1, int(top_k))]
    cands = [
        {
            "city": metros[int(i)][0],
            "lat": float(metros[int(i)][1]),
            "lon": float(metros[int(i)][2]),
            "probability": round(float(agg_probs[int(i)]), 4),
        }
        for i in order
    ]
    best = cands[0]
    return GlobalCityGuess(
        city=best["city"],
        lat=best["lat"],
        lon=best["lon"],
        confidence=best["probability"],
        method=f"zero_shot_clip:{model_id}",
        top_candidates=cands,
    )


_VLM_GLOBAL_PROMPT = (
    "You are an expert street-view geolocator analyzing a dashcam frame.\n"
    "Inspect driving side (left vs right), scripts/languages on signs, license "
    "plate shapes/colors, road markings, utility poles, vegetation, and architecture.\n"
    "Output EXACTLY two lines:\n"
    "DRIVING_SIDE: <LEFT or RIGHT or UNKNOWN>\n"
    "CITY: <most likely 'City, Country', e.g. 'Tokyo, Japan' or 'São Paulo, Brazil'>"
)
_CITY_LINE_RE = re.compile(r"(?im)^[\s*>-]*CITY\s*[:\-]\s*(.+)$")


def _vlm_vote_city(frames_bgr: list[np.ndarray], n_query: int = 5) -> GlobalCityGuess | None:
    """Fallback/complementary Gemma 4 VLM city inference across sampled frames."""
    if not frames_bgr:
        return None
    import torch
    from PIL import Image
    from . import vlm_anchor as va

    va._load()
    idx = np.linspace(0, len(frames_bgr) - 1, min(n_query, len(frames_bgr))).astype(int)
    dev = getattr(va._model, "device", None) or next(va._model.parameters()).device
    votes: Counter[str] = Counter()
    for i in idx:
        pil = Image.fromarray(cv2.cvtColor(frames_bgr[int(i)], cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": _VLM_GLOBAL_PROMPT},
        ]}]
        inp = va._proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(dev)
        with torch.no_grad():
            out = va._model.generate(**inp, max_new_tokens=48, do_sample=False, repetition_penalty=1.2)
        reply = va._proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = _CITY_LINE_RE.search(reply)
        if m:
            cand = m.group(1).strip().strip("*`\"' ").rstrip(".")
            if cand and cand.lower() not in {"unknown", "none", "n/a"}:
                votes[cand] += 1
    if not votes:
        return None
    ranked = votes.most_common(5)
    total = sum(votes.values())
    top_city, top_count = ranked[0]
    return GlobalCityGuess(
        city=top_city,
        lat=None,
        lon=None,
        confidence=round(top_count / max(1, total), 4),
        method=f"vlm_vote:{va.MODEL_ID}",
        top_candidates=[{"city": c, "votes": n} for c, n in ranked],
    )


def infer_city_worldwide(
    *,
    video_path: Path | str | None = None,
    frames_bgr: list[np.ndarray] | None = None,
    title: str | None = None,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    use_vlm: bool = False,
    device: str | None = None,
) -> GlobalCityGuess | None:
    """Infer ``City, Country`` from title metadata or visual content worldwide."""
    from .city_inference import guess_city_from_title

    guessed = guess_city_from_title(title)
    if guessed:
        return GlobalCityGuess(
            city=guessed, lat=None, lon=None, confidence=1.0,
            method="video_title", top_candidates=[{"city": guessed}],
        )
    imgs = list(frames_bgr) if frames_bgr is not None else (
        sample_video_frames(video_path, n_frames=8, start_sec=start_sec, end_sec=end_sec)
        if video_path is not None else []
    )
    if not imgs:
        return None
    if use_vlm:
        try:
            res = _vlm_vote_city(imgs)
            if res is not None:
                return res
        except Exception:
            pass
    try:
        return _clip_rank_cities(imgs, device=device)
    except Exception:
        return None
