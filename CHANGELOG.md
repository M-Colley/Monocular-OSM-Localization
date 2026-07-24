# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/); while on `0.x` the public interface
(CLI flags, output schema) may still change between minor versions.

## [0.1.0] — 2026-07-24

First tagged release: a monocular, GPS-free video-to-map localizer with an
honest, reproducible benchmark. The pipeline takes a dashcam video plus a city
name and estimates where in that city the drive happened, drawing the route on
the OpenStreetMap road network — no GPS in the video.

### Headline results are GPS-free (deployable), not GT-seeded

The numbers this release reports as its headline are **fully GPS-free**: seeded
only from the city name plus what the pipeline reads off the video
(title/description place names, license-plate district, OCR'd signs), then dense
Mapillary VPR coarse-to-fine. On the 9-clip ground-truth fleet (default config,
`MLY_TOKEN` set):

| Outcome | Clips | GPS-free mean route error |
|---|---|---|
| ✅ Localizes | Málaga, Ulm 4K, Ulm #2 | 30 / 48 / 156 m |
| ⚠️ Right area, coarse | Berlin, London | 1.5 / 2.5 km |
| ❌ Collapses | comma2k19, KITTI 0009, KITTI 0033, Boreas | 5–7 km |

**This is a solid, honest baseline — not a solved problem.** GPS-free
localization works where the footage gives the coarse pass something to grab (a
named place, a legible sign, dense modern Mapillary coverage, a distinctive
streetscape) and fails on self-similar, signal-poor, or old low-res footage — an
input limitation, not an algorithm one. A separate *fine-localization ceiling*
(77 m fleet mean) is reported in the README, but it is GT-seeded (uses a
`--osm-around` location leak) and is explicitly **not** a deployment number.

### The Mapillary key is crucial

Dense street-level VPR against Mapillary imagery is the single biggest accuracy
lever and the one signal that works without a GPS leak. It requires a free
`MLY_TOKEN` (see the README). Without it the pipeline falls back to the sparse,
tokenless KartaView source and GPS-free localization mostly collapses to
kilometre-scale error.

### The best configuration is now the default

The best-performing blind configuration ships on by default, each with a `--no-*`
escape:

- **VPR prior** (`--use-vpr-prior`, on) — MegaLoc retrieval anchor.
- **Mapillary source** (`--vpr-source mapillary`, was `kartaview`) — dense
  references; needs `MLY_TOKEN`.
- **Coarse-to-fine VPR** (`--vpr-coarse-to-fine`, on) — wide pass → tight
  re-fetch for GPS-free runs.
- **Coarse-from-video** (`--coarse-from-video`, on) — seed the search disc from
  the title/description place names.
- **Coarse-from-frames** (`--coarse-from-frames`, on) — seed from license-plate
  district + OCR'd place names.
- **Scale-lock** (`--scale-lock`, on) — span the true route extent.

### Added

- **9-clip GT fleet** across six cities / two continents / four GPS sources: Ulm
  ×2, Berlin, KITTI ×2 (Karlsruhe), comma2k19 (Daly City), London, Málaga,
  Boreas (Vaughan). External adapters in `src/ext_datasets.py`,
  `src/kitti_raw.py`, `src/comma2k19.py`.
- **GPS-free coarse-prior stack**: video title/description place-name geocoding
  (`src/location_prior.py`), license-plate registration-district anchor, and
  OCR place-name seeds, all feeding the search disc when no location is given.
- **`--use-tile3d` skyline channel**: renders open LoD2 CityGML building models
  (Berlin, Baden-Württemberg, Bavaria, NRW) and re-ranks candidates by
  rendered-vs-observed skyline agreement (`src/citygml_lod2.py`,
  `src/tile3d_match.py`); a global OSM-buildings LoD1 fallback
  (`src/osm_buildings3d.py`); optional SegFormer sky segmentation
  (`src/sky_segmentation.py`). Safe but currently inert for the headline; opt-in
  margin-gated tie-breaker available.
- **Union VPR source** (`--vpr-source union`) concatenating Mapillary + KartaView
  + Panoramax references.
- Video metadata now round-trips `fps` and `description`.

### Changed

- Pipeline defaults flipped to the best blind configuration (see above).
- README restructured so the **GPS-free (deployable) results are the headline**,
  with the GT-seeded numbers demoted to a clearly-labeled fine-localization
  ceiling.
- tile3d performance: ~9× faster skyline channel via a mesh spatial grid,
  vectorized projection, and a per-tile parse cache.

### Fixed

- Auto-stride fps-probe asymmetry that made A/B runs diverge.
- Mesh `.npz` cache race + self-heal on corrupt caches.
- Dense-VPR coarse-pass timeout on mega-cities (pass-1 token download cap).

### Known limitations

- GPS-free localization collapses on self-similar suburban grids and old low-res
  footage (4 of 9 fleet clips) — the coarse-localization wall.
- Fine placement is scale/extent-limited on uncalibrated clips; metric scale is
  not recoverable from monocular shape alone.
- Straight-line drives carry no shape information; the window must contain a turn.

[0.1.0]: https://github.com/
