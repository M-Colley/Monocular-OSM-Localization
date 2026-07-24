"""Deep image-embedding retrieval over candidate map patches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np
from pyproj import Transformer

from .aerial_match import render_osm_patch
from .osm_data import RoadGraph
from .trajectory_matching import MatchCandidate


class ImageEmbedder(Protocol):
    def encode(self, images: Sequence[np.ndarray]) -> np.ndarray: ...


@dataclass
class EmbeddingMatchResult:
    candidate_index: int
    source: str
    cosine_similarity: float
    image_path: Path | None
    error: str | None = None


def _normalize_rgb_image(img: np.ndarray, *, size: int = 224) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = img.copy()
    else:
        raise ValueError(f"unsupported image shape: {img.shape}")
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def _embedding_cube_to_rgb(embedding: np.ndarray, *, size: int = 224) -> np.ndarray:
    if embedding.ndim != 3:
        raise ValueError("expected HxWxC embedding cube")
    h, w, c = embedding.shape
    if c < 3:
        raise ValueError("embedding cube needs at least 3 channels")

    if max(h, w) > size:
        scale = max(1, int(np.ceil(max(h, w) / size)))
        embedding = embedding[::scale, ::scale]

    flat = embedding.reshape(-1, embedding.shape[-1]).astype(np.float32)
    flat -= flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    rgb = flat @ vt[:3].T
    rgb = rgb.reshape(embedding.shape[0], embedding.shape[1], 3)
    rgb -= rgb.min(axis=(0, 1), keepdims=True)
    denom = rgb.max(axis=(0, 1), keepdims=True)
    denom[denom < 1e-6] = 1.0
    rgb = np.clip(rgb / denom, 0.0, 1.0)
    rgb = (255.0 * rgb).astype(np.uint8)
    if rgb.shape[:2] != (size, size):
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb


def _embedding_cubes_to_rgb_shared(
    cubes: Sequence[np.ndarray], *, size: int = 224
) -> list[np.ndarray]:
    """PCA-render several embedding cubes with ONE shared basis.

    :func:`_embedding_cube_to_rgb` fits a fresh SVD basis per tile (plus
    a per-tile min/max stretch and PCA sign ambiguity), so two cubes are
    not comparable even when they cover adjacent ground. For candidate
    ranking we instead pool the pixels of *all* candidate crops, fit a
    single mean/basis, and normalize with a single global min/max — the
    resulting false-colour images live in one shared colour space.
    """
    if not cubes:
        return []
    prepped: list[np.ndarray] = []
    for embedding in cubes:
        if embedding.ndim != 3:
            raise ValueError("expected HxWxC embedding cube")
        if embedding.shape[2] < 3:
            raise ValueError("embedding cube needs at least 3 channels")
        h, w = embedding.shape[:2]
        if max(h, w) > size:
            scale = max(1, int(np.ceil(max(h, w) / size)))
            embedding = embedding[::scale, ::scale]
        prepped.append(np.asarray(embedding, dtype=np.float32))

    pooled = np.vstack([e.reshape(-1, e.shape[-1]) for e in prepped])
    mean = pooled.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(pooled - mean, full_matrices=False)
    basis = vt[:3].T                                   # (C, 3)

    projected = [
        ((e.reshape(-1, e.shape[-1]) - mean) @ basis).reshape(e.shape[0], e.shape[1], 3)
        for e in prepped
    ]
    lo = np.min([p.min(axis=(0, 1)) for p in projected], axis=0)
    hi = np.max([p.max(axis=(0, 1)) for p in projected], axis=0)
    denom = np.where(hi - lo < 1e-6, 1.0, hi - lo)

    out: list[np.ndarray] = []
    for p in projected:
        rgb = np.clip((p - lo) / denom, 0.0, 1.0)
        rgb = (255.0 * rgb).astype(np.uint8)
        if rgb.shape[:2] != (size, size):
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        out.append(rgb)
    return out


class TorchvisionImageEmbedder:
    """Deep image embedder for retrieval.

    Two backbone families are supported:

    * ``"resnet18"`` — ImageNet-pretrained ResNet18 (penultimate 512-d
      features). Cheap and offline-friendly, but ImageNet features carry
      a large domain gap when comparing a photographic IPM stitch to an
      OSM line-drawing — empirically it ranked wrong candidates to the
      top on GT-evaluated runs.
    * ``"dinov2_vits14"`` / ``"dinov2_vitb14"`` / ``"dinov2_vitl14"`` —
      self-supervised DINOv2 ViT features loaded via ``torch.hub``
      (``facebookresearch/dinov2``). DINOv2 is the backbone behind
      modern visual-place-recognition stacks (AnyLoc) precisely because
      its features are far more robust *across domains* — the right
      choice for cross-view (ground/IPM ↔ satellite) similarity. First
      use downloads the weights (~90 MB for ViT-S).
    """

    def __init__(self, *, model_name: str = "resnet18", device: str | None = None) -> None:
        import os

        try:
            import torch
            from torchvision import models
            from torchvision.transforms import functional as F
        except ImportError as exc:  # pragma: no cover - exercised in real envs
            raise RuntimeError(
                "Embedding retrieval requires optional dependencies: "
                "pip install torch torchvision"
            ) from exc

        # Auto-pick the device when not specified. DINOv2's attention uses
        # xformers' memory_efficient_attention, which only supports CUDA +
        # fp16/bf16 — on CPU/fp32 it raises NotImplementedError. So DINOv2
        # must run on CUDA; ResNet18 is fine on either.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._torch = torch
        self._F = F
        self._device = device
        self._model_name = model_name

        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
            model = torch.nn.Sequential(*(list(backbone.children())[:-1]))
            self._is_dinov2 = False
            self._feat_dim = 512
        elif model_name.startswith("dinov2"):
            # DINOv2 ViT loaded from torch.hub; calling model(x) returns the
            # [B, embed_dim] CLS embedding. Robust cross-domain features.
            #
            # Disable xformers' memory_efficient_attention so DINOv2 falls
            # back to torch's built-in scaled_dot_product_attention. The
            # xformers kernel rejects CPU/fp32 and is fragile across
            # torch/xformers version pairs; the math fallback is portable
            # and plenty fast for 224px tiles. Must be set before hub load.
            os.environ.setdefault("XFORMERS_DISABLED", "1")
            try:
                model = torch.hub.load(
                    "facebookresearch/dinov2", model_name, trust_repo=True
                )
            except Exception as exc:  # pragma: no cover - network/hub path
                raise RuntimeError(
                    f"Failed to load DINOv2 backbone {model_name!r} via torch.hub: {exc}"
                ) from exc
            self._is_dinov2 = True
            self._feat_dim = int(getattr(model, "embed_dim", 384))
        else:
            raise ValueError(
                "Supported embedding models: 'resnet18' or 'dinov2_vits14' / "
                f"'dinov2_vitb14' / 'dinov2_vitl14'. Got: {model_name}"
            )

        model.eval()
        self._model = model.to(device)

    def encode(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self._feat_dim), dtype=np.float32)
        # DINOv2 ViT-/14 needs side lengths that are multiples of 14; 224 works.
        size = 224
        batch = []
        for image in images:
            rgb = _normalize_rgb_image(image, size=size)
            tensor = self._F.to_tensor(rgb)
            tensor = self._F.normalize(
                tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            batch.append(tensor)
        stacked = self._torch.stack(batch).to(self._device)
        with self._torch.inference_mode():
            out = self._model(stacked)
            feats = out.flatten(1)
            feats = self._torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)


def _candidate_center_lonlat(road: RoadGraph, cand: MatchCandidate) -> tuple[float, float]:
    center_xy = cand.walk_xy.mean(axis=0)
    transformer = Transformer.from_crs(road.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(float(center_xy[0]), float(center_xy[1]))
    return float(lon), float(lat)


def _crop_embedding_cube(
    embedding: np.ndarray,
    crs,
    transform,
    lon: float,
    lat: float,
    *,
    half_extent_m: float,
    min_half_px: int = 4,
) -> np.ndarray:
    """Crop a tile-wide embedding cube to a window centred on ``(lon, lat)``.

    ``fetch_embedding`` returns the cube for the whole 0.1-degree
    (~11 km) registry tile *containing* the point, so without this crop
    every candidate in the same cell renders the byte-identical image.
    ``transform`` is the affine mapping pixel ``(col, row)`` to ``(x, y)``
    in ``crs`` (rasterio convention: ``x = a*col + b*row + c``,
    ``y = d*col + e*row + f``); we invert it to find the candidate pixel
    and window ``half_extent_m`` metres around it (clamped to the tile).
    Returns the full cube unchanged when ``crs``/``transform`` are missing.
    """
    if crs is None or transform is None:
        return embedding

    # Accept a rasterio Affine (attributes a..f) or a 6-sequence.
    if hasattr(transform, "a"):
        a, b, c = float(transform.a), float(transform.b), float(transform.c)
        d, e, f = float(transform.d), float(transform.e), float(transform.f)
    else:
        a, b, c, d, e, f = (float(v) for v in list(transform)[:6])

    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tr.transform(lon, lat)

    det = a * e - b * d
    if not np.isfinite(det) or abs(det) < 1e-12:
        return embedding
    col = (e * (x - c) - b * (y - f)) / det
    row = (-d * (x - c) + a * (y - f)) / det

    # Metres per pixel along each axis; geographic CRSs report degrees.
    px_x = float(np.hypot(a, d))
    px_y = float(np.hypot(b, e))
    try:
        from pyproj import CRS
        if CRS.from_user_input(crs).is_geographic:
            px_x *= 111_320.0 * max(np.cos(np.deg2rad(lat)), 1e-6)
            px_y *= 110_540.0
    except Exception:
        pass
    if px_x <= 0 or px_y <= 0:
        return embedding

    h, w = embedding.shape[:2]
    half_c = max(int(round(half_extent_m / px_x)), min_half_px)
    half_r = max(int(round(half_extent_m / px_y)), min_half_px)
    ci = int(round(np.clip(col, 0, w - 1)))
    ri = int(round(np.clip(row, 0, h - 1)))
    c0 = max(ci - half_c, 0)
    c1 = min(ci + half_c + 1, w)
    r0 = max(ri - half_r, 0)
    r1 = min(ri + half_r + 1, h)
    if c1 - c0 < 2 or r1 - r0 < 2:
        return embedding
    return embedding[r0:r1, c0:c1]


def _fetch_geotessera_cube(
    road: RoadGraph,
    cand: MatchCandidate,
    *,
    year: int,
    half_extent_m: float = 600.0,
) -> np.ndarray:
    """Fetch the GeoTessera embedding cube cropped around the candidate."""
    try:
        from geotessera import GeoTessera
    except ImportError as exc:  # pragma: no cover - exercised in real envs
        raise RuntimeError(
            "GeoTessera source requires optional dependency: pip install geotessera"
        ) from exc

    lon, lat = _candidate_center_lonlat(road, cand)
    client = GeoTessera()
    embedding, crs, transform = client.fetch_embedding(lon=lon, lat=lat, year=year)
    return _crop_embedding_cube(
        np.asarray(embedding), crs, transform, lon, lat, half_extent_m=half_extent_m,
    )


def _render_geotessera_patch(
    road: RoadGraph,
    cand: MatchCandidate,
    *,
    year: int,
    size: int = 224,
    half_extent_m: float = 600.0,
) -> np.ndarray:
    cube = _fetch_geotessera_cube(road, cand, year=year, half_extent_m=half_extent_m)
    return _embedding_cube_to_rgb(cube, size=size)


def _render_source_image(
    source: str,
    road: RoadGraph,
    cand: MatchCandidate,
    *,
    geotessera_year: int,
    size: int,
) -> np.ndarray:
    if source == "osm":
        gray = render_osm_patch(
            road,
            (float(cand.walk_xy.mean(axis=0)[0]), float(cand.walk_xy.mean(axis=0)[1])),
            resolution=size,
            half_extent_m=600.0,
        )
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if source == "geotessera":
        return _render_geotessera_patch(road, cand, year=geotessera_year, size=size)
    if source in ("esri", "satellite"):
        # Real RGB orthoimagery — the right domain to compare a photographic
        # IPM/ground query against (vs the OSM line-drawing or GeoTessera
        # PCA false-colour, both of which carry a large domain gap).
        from .satellite import satellite_tile_for_candidate

        return satellite_tile_for_candidate(
            road, cand, half_extent_m=600.0, size=size, provider=source,
        )
    raise ValueError(f"unsupported embedding source: {source}")


def score_candidates_by_embeddings(
    query_rgb: np.ndarray | None,
    road: RoadGraph,
    candidates: list[MatchCandidate],
    *,
    output_dir: Path,
    sources: Sequence[str] = ("osm",),
    model_name: str = "resnet18",
    geotessera_year: int = 2024,
    size: int = 224,
    embedder: ImageEmbedder | None = None,
) -> dict[str, list[EmbeddingMatchResult]]:
    if query_rgb is None or not candidates or not sources:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)

    embedder = embedder or TorchvisionImageEmbedder(model_name=model_name)
    query_vec = embedder.encode([query_rgb])[0]

    results: dict[str, list[EmbeddingMatchResult]] = {}
    for source in sources:
        source_dir = output_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)

        rendered: list[np.ndarray] = []
        rendered_indices: list[int] = []
        source_results: list[EmbeddingMatchResult] = [
            EmbeddingMatchResult(
                candidate_index=i,
                source=source,
                cosine_similarity=0.0,
                image_path=None,
                error=None,
            )
            for i in range(len(candidates))
        ]

        if source == "geotessera":
            # Fetch the candidate-centred cubes first, then PCA-render them
            # with ONE shared basis so the false-colour images are
            # comparable across candidates (a per-tile basis + per-tile
            # stretch makes even neighbouring crops non-comparable).
            cubes: list[np.ndarray] = []
            cube_indices: list[int] = []
            for i, cand in enumerate(candidates):
                try:
                    cubes.append(_fetch_geotessera_cube(
                        road, cand, year=geotessera_year,
                    ))
                    cube_indices.append(i)
                except Exception as exc:
                    source_results[i].error = str(exc)
            try:
                images = _embedding_cubes_to_rgb_shared(cubes, size=size)
            except Exception as exc:
                images = []
                for i in cube_indices:
                    source_results[i].error = str(exc)
                cube_indices = []
            for i, img in zip(cube_indices, images, strict=True):
                image_path = source_dir / f"{source}_candidate_{i + 1}.png"
                cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                rendered.append(img)
                rendered_indices.append(i)
                source_results[i].image_path = image_path
        else:
            for i, cand in enumerate(candidates):
                image_path = source_dir / f"{source}_candidate_{i + 1}.png"
                try:
                    img = _render_source_image(
                        source,
                        road,
                        cand,
                        geotessera_year=geotessera_year,
                        size=size,
                    )
                    cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    rendered.append(img)
                    rendered_indices.append(i)
                    source_results[i].image_path = image_path
                except Exception as exc:
                    source_results[i].error = str(exc)

        if rendered:
            candidate_vecs = embedder.encode(rendered)
            sims = candidate_vecs @ query_vec
            for idx, sim in zip(rendered_indices, sims, strict=True):
                source_results[idx].cosine_similarity = float(sim)

        results[source] = source_results

    return results
