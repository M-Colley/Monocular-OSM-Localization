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


class TorchvisionImageEmbedder:
    def __init__(self, *, model_name: str = "resnet18", device: str = "cpu") -> None:
        try:
            import torch
            from torchvision import models
            from torchvision.transforms import functional as F
        except ImportError as exc:  # pragma: no cover - exercised in real envs
            raise RuntimeError(
                "Embedding retrieval requires optional dependencies: "
                "pip install torch torchvision"
            ) from exc

        if model_name != "resnet18":
            raise ValueError(
                f"Only 'resnet18' is currently supported for embedding model, got: {model_name}"
            )

        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        model = torch.nn.Sequential(*(list(backbone.children())[:-1]))
        model.eval()

        self._torch = torch
        self._F = F
        self._model = model.to(device)
        self._device = device

    def encode(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, 512), dtype=np.float32)
        batch = []
        for image in images:
            rgb = _normalize_rgb_image(image)
            tensor = self._F.to_tensor(rgb)
            tensor = self._F.normalize(
                tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            batch.append(tensor)
        stacked = self._torch.stack(batch).to(self._device)
        with self._torch.inference_mode():
            feats = self._model(stacked).flatten(1)
            feats = self._torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)


def _candidate_center_lonlat(road: RoadGraph, cand: MatchCandidate) -> tuple[float, float]:
    center_xy = cand.walk_xy.mean(axis=0)
    transformer = Transformer.from_crs(road.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(float(center_xy[0]), float(center_xy[1]))
    return float(lon), float(lat)


def _render_geotessera_patch(
    road: RoadGraph,
    cand: MatchCandidate,
    *,
    year: int,
    size: int = 224,
) -> np.ndarray:
    try:
        from geotessera import GeoTessera
    except ImportError as exc:  # pragma: no cover - exercised in real envs
        raise RuntimeError(
            "GeoTessera source requires optional dependency: pip install geotessera"
        ) from exc

    lon, lat = _candidate_center_lonlat(road, cand)
    client = GeoTessera()
    embedding, _crs, _transform = client.fetch_embedding(lon=lon, lat=lat, year=year)
    return _embedding_cube_to_rgb(np.asarray(embedding), size=size)


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
