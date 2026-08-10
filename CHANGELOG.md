# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/); while on `0.x` the public interface
(CLI flags, output schema) may still change between minor versions.

## [Unreleased]

### Added

- **GPS-overlay clip fleet — ground truth with no manual labelling.** Consumer
  dashcams burn the live position into the frame, so a clip with such a stamp
  carries its own ground truth. `scripts/build_overlay_fleet.py` downloads each
  clip's analysis window, OCRs the stamp, reverse-geocodes the midpoint into a
  `--city` string and writes `ground_truth/overlay_<id>.json` — video URL in,
  evaluable ground truth out. The clip list lives in `scripts/overlay_clips.py`;
  adding a clip is one entry.
- **8 new evaluation clips** across four US cities and five camera/overlay
  formats: Chicago day + night (VIOFO A229 Pro), Phoenix/Tempe/Scottsdale (ROVE
  R2-4K), Detroit ×3 (DOD LS460W and YouTube Capture), Atlanta ×2 (WolfBox i07).
  They deliberately probe cases the existing fleet lacks: a night/wet drive
  against a near-identical daytime one, freeway stretches with no turns for the
  shape matcher, and US arterial grids.
- `scripts/run_all_gt.py --overlay` / `--all-fleets` runs the new fleet. It is a
  **separate** fleet: its GT is OCR-derived rather than hand-labelled or INS, so
  folding it into the core 9-clip mean would quietly change the published
  numbers. The sweep reads the index the builder writes, so slugs, video paths
  and GT-derived discs are never maintained by hand.
- `scripts/check_overlay_gt.py` audits auto-extracted ground truth: re-derives
  implied speeds, flags out-and-back detours (what a surviving digit misread
  looks like), and reports the coordinate precision actually present — the floor
  on any error measured against that clip. Non-zero exit, so it can gate a
  rebuild.
- **Edit-cut handling.** `gps_overlay.split_on_discontinuities` /
  `longest_continuous_run` split a track wherever it implies a speed no car
  reaches, and the builder keeps only the longest continuous stretch. Uploaded
  dashcam footage is frequently a compilation — one clip here jumped 19.5 km
  between consecutive samples — and a cut is equally fatal to the VO the ground
  truth is meant to score. The jump filter cannot help: both sides of a cut are
  perfectly valid fixes.
- Extracted tracks are cached beside each video, so re-running the fleet after a
  code change costs no OCR. Keyed on the video's identity, the sampling
  parameters *and* a parser version, so a parser change re-extracts.
- The video-metadata cache moved from `cli.py` into `download.py` next to
  `fetch_video_metadata`, and is now public (`cached_video_metadata`). A fleet
  build makes one metadata request per clip and YouTube answers a burst of them
  with "Sign in to confirm you're not a bot" — which used to strand eight
  fully-downloaded videos on a call none of them needed.
- `scripts/make_gt_from_overlay.py --url` for a single clip, with `--city` now
  optional (the track's midpoint is reverse-geocoded).
- `download_video(..., section=(start, end))` fetches only an analysis window.
- `easyocr` is now declared — `pip install -e ".[ocr]"`. It was imported by
  `monocular_osm/scene_text.py` and `gps_overlay.py` but appeared in no manifest.

### Fixed

- **A run-together DMS parsed 37 km off, and looked fine.** When OCR preserved
  the apostrophe, `4222'59"N` matched greedily as degrees=`422`, and the
  degree-overflow repair truncated to `42` while leaving minutes as `2` —
  yielding 42°2'59" instead of 42°22'59". The resulting ground truth passed
  every plausibility check (smooth legs, 18 km/h mean, sensible turns) while
  sitting 37 km south of the real route. Overflowing degrees have two causes
  needing opposite repairs — digits running together (shift the digit into the
  minutes) versus a degree sign read as a digit, `W83°` → `W838` (drop it) — and
  the discriminator is whether shifting keeps minutes under 60.
- The dropped-decimal-point repair (`W:87 6216` → `W:87.6216`) was eating DMS:
  `N 83 12109"W` became the decimal `83.12109`, losing 1.6 km. The
  run-together-DMS repairs now run first, and the rule refuses a digit run
  closed by a quote.
- `build_overlay_fleet.py` cross-checks the reverse-geocoded city against the
  video title and warns when they disagree. On the clip above this was the
  *only* visible symptom — a township the title never mentions.

### Changed

- **Downloads are ~13× faster.** `_format_selector` now prefers mp4/AVC over the
  higher-ranked VP9/webm rendition: measured 925 kB/s vs 71 kB/s on the same
  1080p upload, and cv2's bundled FFmpeg decodes AVC most reliably. The
  codec-agnostic choice remains as the next fallback.
- Node is enabled as a yt-dlp JavaScript runtime when on PATH, clearing the
  "extraction without a JS runtime is deprecated, some formats may be missing"
  warning.
- `gps_overlay.parse_latlon` handles the formats the new fleet actually uses:
  hemisphere-**prefix** DMS (`N42° 20' 18.07"`) as well as suffix, a missing or
  misread degree sign (`W83°` OCR'd as `W838`), and DMS whose punctuation the OCR
  ate entirely (`4222159"N`). A new `normalize_overlay_text` repairs the
  systematic damage — spaced separators, fractions split across detection boxes,
  dropped decimal points — before parsing.
- `extract_gps_track` OCRs each band under three preprocessing variants and takes
  a consensus where two agree. No single variant reads every camera: the native
  band preserves DOD DMS punctuation that upscaling smears, while 2× upscaling
  recovers the decimal point (and sometimes the whole hemisphere letter) on
  VIOFO/WolfBox stamps.

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
