# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/); while on `0.x` the public interface
(CLI flags, output schema) may still change between minor versions.

## [Unreleased]

### Reproducible, leak-free evaluation

Every published number should be traceable to a commit, a configuration and a
map snapshot, and comparable against a trivial baseline. A review found that
the 0.1.x GPS-free headline was none of these (one row had no committed source;
rows came from at least two code states; repeated runs of one clip differed by
hundreds of metres), and that guessing the **centre of the named city** beat
the pipeline on 5 of 8 clips. This release builds the machinery to measure
honestly; the re-measured results are in the README.

- **Provenance in every `result.json`**: git commit / branch / dirty flag,
  package and library versions, full config, argv, CUDA availability, VO
  algorithm version, the OSM graph's cache path / creation date / size, which
  retrieval backbone (pinned hub commit) and reference source were *actually*
  used, and whether `MLY_TOKEN` was set — never its value.
- **VO cache versioning.** Cache keys carry `VO_ALGO_VERSION`; trajectories
  written by older code (before the plane-handedness and burned-in-graphics
  fixes) are reported stale and never loaded. The flat-layout fallback that
  could hand one clip *another clip's* trajectory is gone, and the two
  May-2026 Ulm caches that were committed to `data/` are removed.
- **`benchmarks/manifest.json` + `scripts/benchmark.py`**: the committed
  benchmark definition (clips, user-level city strings, analysis windows, GT
  provenance, dev/test split) and a runner that passes the pipeline *only*
  video, city and window — ground truth is applied offline, so leakage is
  impossible by construction (every command is audited before launch, and
  every result's recorded argv/config is re-audited at aggregation, including
  whether the search disc came from the video rather than from GT).
  Repeated runs, a cold-cache option, bootstrap confidence intervals over
  clips, per-clip median/IQR, provenance consistency checks.
- **`monocular_osm/metrics.py`**: two-directional route metrics (start, end,
  GT→prediction, prediction→GT, symmetric mean, Hausdorff, Fréchet) and
  success rates at 25/50/100/250/500/1000 m. The legacy mean route error was
  one-directional, so a prediction that follows the route and then wanders 3 km
  away scored 0.
- **`monocular_osm/baselines.py`**: city centroid, retrieval-prior-only,
  matcher-only, pinned-only and best-of-shortlist oracle — the rows every
  result is compared against.
- **`scripts/ablate.py`**: re-scores channel ablations, the removed shape
  double-count, and leave-one-clip-out conditional-logit fusion offline from
  each run's `fusion_features.json`.
- **`scripts/diagnose_pool_recall.py`** (plus a `walk_observer` hook on the
  matcher): is the true route even in the candidate pool, per stage and walk
  budget?
- **`scripts/golden_pipeline.py`**: behaviour-preservation check for
  refactoring (full default run from warm caches, normalised diff).
- `scripts/fetch_kitti_raw.py`: fetches only the camera frames, OXTS and
  calibration out of KITTI's multi-GB archives with HTTP range requests.

### Measured: benchmark run `v2`, core fleet

Default configuration, `MLY_TOKEN` set (Mapillary + MegaLoc), three reps per
clip with cold retrieval caches, package code `18e65ae`, all 18 reps healthy.
Symmetric mean route error, median over reps:

| Clip | Pipeline | Centre of named city | Retrieval prior alone | Matcher alone |
|---|---|---|---|---|
| Ulm 4K, 0-300 s | 27 m | 445 m | 101 m | 1,219 m |
| London | 83 m | 1,501 m | 191 m | 719 m |
| Berlin, 0-420 s | 3,356 m | 909 m | 3,495 m | 4,058 m |
| KITTI 0009 | 5,346 m | 2,563 m | 5,533 m | 5,830 m |
| KITTI 0033 | 7,094 m | 6,709 m | 7,099 m | 6,921 m |
| Ulm #2 | 7,662 m | 350 m | 7,986 m | 8,406 m |

The pipeline beats the centre of the named city on 2 of 6 clips (sign test
p = 0.69), the retrieval prior alone on 6 of 6 (p = 0.03; by as little as 5 m
where both fail) and the matcher alone on 5 of 6. Repetitions agree to within
150 m (IQR). On the four failing clips
even the best candidate in the matcher's pool, chosen with ground truth, is
2.8–8.2 km off (`scripts/ablate.py`): the failures happen before matching.
comma2k19, Málaga and Boreas were not run (not on the benchmark machine).
Reproduce with `scripts/benchmark.py run --run-name v2 --fleet core --repeats 3
--cold-vpr`, then `aggregate --run-name v2 --offline` and
`scripts/ablate.py --run-name v2`.

### Selective prediction, pre-registered

- **`benchmarks/preregistered_rules.json`** (commit `1841961`) fixes one rule
  before its test data existed: answer with the pipeline only when its
  retrieval confidence (coarse-to-fine pass-2 similarity) is ≥ τ = 0.162,
  otherwise answer the centre of the named city. τ is the midpoint between the
  best losing (0.130) and worst winning (0.194) core rep; on those clips it
  separates perfectly, which is a fit, not evidence. The overlay fleet, which
  played no part in choosing τ, is the test; the hypotheses, the code version
  and the one test value seen beforehand are recorded in the file.
- `scripts/benchmark.py aggregate` scores the rule as `pipeline_selective`
  (read, never fitted), reports coverage and each rep's decision, runs the
  registered H3 test (one-sided Fisher exact) and labels the clips τ was fitted
  on as in-sample. `baselines.retrieval_confidence()` / `pipeline_selective()`.

### Added

- `monocular_osm/fusion.py`: the consensus rule and its weights as one explicit
  module; `--fusion-weights JSON` for ablations; each run writes
  `fusion_features.json` (every candidate's per-channel ranks, penalties and
  route). Two facts the README table hid are now stated and tested: with
  sliding windows off (the default) shape effectively weighs **2.0**, and the
  "aerial" channel re-scores the same trajectory/walk pair the shape fit scored.
- `result["channels"]`: ok / inactive / failed / disabled per channel, with the
  exception text on failure (18 channels used to fail into a `print()` only).
- `result["pinned_position"]`; position routes carry `route_t_sec` (video time
  per point) and keep up to 1000 points (was 50, ~250 m spacing on a 12 km
  clip); hypotheses carry their routes; `osm_around_source` records whether the
  search disc came from the user or the video title.
- CLI parity: `--walks-per-node`, `--scale-recovery-thresh-m`,
  `--scale-recovery-min-inliers`, `--scale-recovery-min-baseline-m`; a test fails
  if any `PipelineConfig` field has no CLI mapping.
- `monocular_osm/geo.py` (pyproj geodesy) and `monocular_osm/net.py`
  (tenacity-based bounded retries).
- `monocular_osm/data/kfz_codes.json`: 698 licence-plate district codes from
  Wikidata (was 311 hand-entered), refreshed by `scripts/update_kfz_codes.py`.
- `result["vpr_prior"]["track_sim_median"]`: the retrieval confidence of the
  final track, recorded on every run (Ulm #2 ran with none, because
  coarse-to-fine was skipped); `vpr_prior.backbone.device` records where the model ran.
- `scripts/benchmark.py run --cold-archive {metadata,full}`.

### Changed

- **`result["matcher_position"]` is now the raw matcher pick.** On any run where
  the scale-locked OCR anchor pin succeeded it used to carry the pinned route
  under `source: "matcher"`, so every "matcher-only" number on sign-rich clips
  was really matcher + pin. The pinned route is `pinned_position`; headline
  selection is unchanged (anchor-primary > pinned > matcher).
- **OCR coarse seeds need two place names inside the named city.** One
  geocoded fragment anywhere within 60 km used to become an 800 m hard gate on
  both retrieval and the road graph; on KITTI 0009 rapidocr read `KA.GC` (a
  plate fragment), the run searched 20 km away and finished 18.3 km off. The
  same clip now finishes 5.3 km off.
- **Title seeds need two place names too.** On Ulm #2 the single title word
  "Napoleon" became an 800 m hard gate centred 8–9 km from the route; all three
  benchmark reps answered 7.7 km off, where the centre of the city is 350 m
  off. Chosen after seeing that clip, so it is a dev-set fix to be judged on
  held-out clips. Of the core fleet it changes only Ulm #2 (Berlin, London and
  Ulm 4K each matched five places).
- Retrieval runs on the CPU (same model) when the GPU has under 4 GB free:
  with another process holding the GPU the model died natively, with no Python
  traceback.
- `PipelineConfig.top_k` defaults to 10 (the CLI default every published run
  used; the Python API silently ran 5).
- Evaluator distances are exact (shapely), not a 10 m densified approximation.
- The position summary prints `Shape fit`, not `Confidence` (it grades fit
  quality, which is not correctness); the JSON keeps `confidence.level` and adds
  `basis: "shape_fit"`.
- A VLM-fallback prior gets district-level (1.5 km), not retrieval-level
  (150 m), trust.
- Library reuse: `skimage.measure.ransac` for the anchor similarity fit (94.7%
  identical on 3000 synthetic cases; where it differs the scale error is lower:
  median 10.3% vs 12.2%); pyproj for all geodesy (distances shift 0.1–0.5%;
  `enu_to_latlon` was up to 0.54 m off at 1.8 km); rapidfuzz for name matching;
  exact shapely anchor distances; a cKDTree radius query for anchor seeding;
  osmnx's own Nominatim throttling instead of an extra sleep.
- `rapidfuzz` and `tenacity` are core dependencies; `fast-alpr` is the `[plates]`
  extra; `pycolmap` (used by one experiment script) is the `[sfm]` extra.
- Licence-plate table: RID was "Rottal-Inn" (now Riedenburg), LOS was "Frankfurt
  (Oder)" (now Beeskow), PM was "Brandenburg an der Havel" (now Bad Belzig).
  Codes of retired plate systems (occupation zones, GDR letters) are dropped so
  they cannot vote for the wrong city.
- GT provenance: KITTI and comma ground truth said "Auto-extracted from a
  burned-in GPS overlay via OCR"; they come from OXTS / the comma INS pose.
- `scripts/run_all_gt.py --blind` is now actually blind (London kept a
  ground-truth-centred disc); non-blind output is labelled GT-SEEDED.

### Fixed

- **Image retrieval silently never loaded on a fresh machine.** `torch.hub`
  asks on stdin whether to trust the MegaLoc/EigenPlaces repos; an unattended
  run read EOF as "no", so the retrieval channel — the main accuracy lever —
  dropped out with only "VPR unavailable" printed. The repos are now trusted
  explicitly and pinned to commits, a MegaLoc→EigenPlaces fallback is printed,
  and the failure reason is recorded in `result["channels"]`.
- `--skip-download` could analyse a yt-dlp intermediate (`input.temp.mp4`),
  which shares the `.mp4` suffix and is newer than the real file.
- `SimilarityTransform.from_estimate` reports failure with a *falsy* object,
  not `None`; degenerate Procrustes input raised instead of returning the
  inf-residual fallback.
- Sliding-window support counted any walk that crossed or touched a candidate
  (closest approach ≤ 250 m), so in a small search disc every window supported
  every candidate; it now requires ≥ 50% of the window walk within 60 m.
- The sun-heading clock reader imported `easyocr` directly and returned nothing
  on a default install (which ships rapidocr); it uses the shared engine factory.
- A Nominatim error body was cached forever as "not found" (osmnx raises the
  same exception for both); only a genuine "could not geocode" is cached.
- **A full disk poisoned later runs.** `ox.save_graphml` interrupted by a full
  disk left a truncated road-graph cache that every later run of the clip
  loaded and died on. Graph caches are now written to a temporary file and
  renamed; an unreadable cache is moved to `*.corrupt` and refetched.
- Cold benchmark runs archived ~1.2 GB of downloaded reference pixels and
  embeddings per clip and rep and filled the disk after ~50 reps (which then
  disabled retrieval and starved the pagefile: MemoryError, access
  violations). They now keep the reference id/position lists only (~3 %);
  `--cold-archive full` keeps everything.
- One transient Overpass `ConnectTimeout` or YouTube 403 killed runs that had
  already spent minutes on VO and retrieval; downloads and the graph fetch now
  retry with exponential backoff (bounded, announced, transient errors only).

### Added (OCR engine, earlier in this release cycle)

- **`--ocr-engine rapidocr`** — an alternative OCR backend (PP-OCRv6 on
  onnxruntime) behind the existing `OcrReader` Protocol, serving both OCR
  consumers: scene-text anchors and the GPS-overlay stamp reader.

  Measured against easyocr on 320 overlay-fleet frames, both engines seeing
  identical images and the same three preprocessing variants:

  | engine | parse yield | >100 m misreads | ms/img |
  |---|---|---|---|
  | easyocr (GPU) | 92.5% | 4.1% | 354 |
  | **rapidocr** (onnxruntime CPU) | **96.6%** | **3.2%** | 1829 |

  On the 80 frames where PaddleOCR also finished: yield 91.2 / **97.5** / 96.2%
  for easyocr / rapidocr / paddleocr. Median error is a three-way tie
  (7.8 / 8.0 / 8.2 m) and should not be read as accuracy — it is dominated by
  interpolating the 5 s reference across real vehicle motion, not by OCR. Only
  yield and the gross-misread rate carry signal.

  Its gains concentrate on the formats that need the most repair downstream —
  the DOD LS460W DMS clips and the low-contrast ROVE stamp. End to end through
  the repo on `1nF_7l07i-E` (easyocr's weakest clip): 96% of sampled frames
  yielded a fix, with median agreement of 0.0 m against the existing
  reference — the same coordinates, read on more frames.

  **rapidocr is now the default engine.** `pip install -e ".[ocr]"` installs
  it; easyocr moves to its own `[easyocr]` extra. That also makes the OCR
  channel much lighter — rapidocr pulls no torch.

  Whichever engine is requested, a missing backend **falls back to the other
  with a warning** rather than failing the run: OCR is one channel of several,
  and both engines are optional extras, so an install with only one of them
  must still work.

  **Two consequences to be aware of.** Every number published in this repo
  before 2026-08-12 was measured with easyocr — including the overlay fleet's
  `ground_truth/*.json`, which is committed as extracted and is NOT regenerated
  by this change. And because both OCR caches key on the engine, existing
  caches will regenerate on first run under the new default (correct, but it
  costs an OCR pass).

  Both OCR caches — scene-text and the overlay track — now key on the engine,
  so one engine's output can never be served to the other.

  **PaddleOCR was evaluated and rejected**: no accuracy edge over rapidocr,
  38x slower than easyocr (13.6 s/image), and on Windows it only runs with
  `enable_mkldnn=False` (otherwise paddlepaddle raises
  `ConvertPirAttribute2RuntimeAttribute` in its oneDNN path) — which is what
  makes it that slow.

## [0.1.2] — 2026-08-11

### Added


- **`--use-ground-flow-scale`: a real speed profile instead of a constant one.**
  The VO normalises every relative translation to unit length, so the recovered
  trajectory encodes speed as a one-bit moving/stopped flag — our step
  magnitudes have a coefficient of variation of 0.02–0.07 where the true ones
  have 1.0–1.4. That is a **constant-speed assumption imposed on the trajectory
  shape**, which is the one thing `trajectory_matching` scores.
  `monocular_osm/ground_flow_scale.py` measures a per-step length from
  road-plane optical flow and rescales the steps, keeping their directions.

  **The overlay-fleet numbers that used to sit here have been removed.** They
  were measured against ground truth that has since been re-extracted (see
  Unreleased), which moved several clips' analysis windows — one from 555 s to
  275 s once two real edit cuts became visible. The estimator is unchanged; its
  reference is not, so those figures are not re-quotable and need re-measuring.
  The oracle experiment behind the design still holds: substituting *true* step
  lengths for unit ones is worth a large fraction of the remaining shape error.

  Three measured facts shape the design. **Absolute scale is worth exactly
  zero** — a constant 30% error scores 0.0 m, because the matcher's similarity
  fit absorbs any global factor, so `camera_height_m` being a guess costs
  nothing. **The estimate need not be accurate** — corrupting the true lengths
  by 15–30% changes the score by under a metre; freedom from *bias* is what
  matters. And **only the horizon row matters geometrically**, so it is
  self-calibrated per clip from flow consistency rather than assumed. Recovered
  pitches across the fleet range −6.2° to +2.4°, which is why the fixed
  `ipm_scale_pitch_deg` was never going to serve every camera.

  It **abstains rather than guessing**, all-or-nothing: when the tracked
  points disagree about how far the car moved, or the horizon never resolves,
  the trajectory comes back untouched. That fires on 2 of 8 overlay clips and
  1 of 3 core-fleet clips, and no clip is made worse.

  **Also validated on the core fleet**, whose ground truth is hand-labelled or
  from an INS — independent of the estimator, unlike the overlay fleet where
  GT and evaluation share a source. Mean **134.3 → 123.2 m**:

  | clip | base | ground flow | oracle | |
  |---|---|---|---|---|
  | Ulm 4K (960×540 from 4K) | 345.8 | **319.7** | 285.9 | conf 1.00 |
  | KITTI 0009 (**1242×374**) | 17.3 | **9.9** | 10.2 | conf 1.00 |
  | London (720p) | 39.9 | 39.9 | 51.9 | abstained |

  KITTI settles the geometry question: at 3.3:1 the horizon lands at 0.43 of
  frame height, well inside the search band, and the estimate essentially
  reaches the oracle. Nothing about the method assumed 16:9.

  Off by default regardless. Three core clips is a thin basis, and the
  remaining fleet (comma2k19, Málaga, Boreas, Berlin) is untested.

  Costs a second, streaming pass over the video (~2 min per 4-minute clip,
  rolling `stride + 1` frame buffer). That pass is not avoidable: the estimator
  needs the frames *between* the VO's strided ones. Tracking only the strided
  pair — all the pipeline's frame list holds — measurably loses most of the
  benefit (−3.8% versus −9.3%) and regresses a clip.

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

### Changed

- Ground-flow scale applies **all-or-nothing** rather than blending at partial
  confidence. Blending was never exercised until the London clip scored 0.46 —
  and it turned a 39.9 m baseline into 45.3 m. Every clip that has ever helped
  scored a flat 1.00, so below the floor the trajectory is handed back
  untouched (core-fleet mean 125.0 → 123.2 m, no clip made worse; the overlay
  fleet is bit-identical, every clip there being already 1.00 or 0.00).
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

### Fixed

- `--help` crashed for every user. argparse `%`-formats help strings, so a
  literal `61%` made `% o` parse as the `%o` conversion. `tests/test_cli_help_strings.py`
  now checks all 86 help strings by reading `cli.py` as source, so the guard
  works even where the optional heavy dependencies block importing the CLI.


- **Burned-in graphics were blinding the visual odometry.** `_estimate_relative_pose`
  selected its 300 correspondences by *descriptor distance*, and descriptor
  distance is smallest exactly for the pixels that never move — a channel
  watermark, a dashcam's GPS/speed stamp, the dashboard, a hood reflection.
  The stationary guard then took the median over that selection, read 0.00 px
  while the car was doing 65 mph, and voided the pair; the essential matrix,
  when it was fitted at all, was fitted on points that satisfy *every* epipolar
  geometry. Now the motion statistic is the 90th percentile over ALL
  cross-checked matches, expressed as a fraction of focal length rather than in
  pixels, and only correspondences carrying real parallax are fitted.

  The per-clip figures previously quoted here were measured against ground
  truth that has since been re-extracted, so they have been removed rather
  than left looking current. The defect and the fix are unchanged and are
  covered by tests; what needs re-measuring is how much they are worth.

- **The loop-closure detector could fabricate a loop out of uninitialised
  memory.** `cv2.findFundamentalMat` returns `F=None` when RANSAC fails but
  still hands back a mask whose contents are undefined; `_inliers_from_features`
  checked only `mask is not None` and summed it. Measured returns of 2195 and
  1603 from a 40-point input — either clears `min_inliers=30` outright. Four
  guards, which must stay together:
  its exception (degenerate point sets *raise* rather than return, and that
  crash was the only thing preventing a false closure on one clip), the `F is
  None` check, a parallax gate (the 449-inlier "verified loop" on
  `ZhGb8q1kliY` had a median inlier displacement of 0.00 px — zero-disparity
  points are inliers to any fundamental matrix), and a refusal in
  `redistribute_drift` for any closure whose gap exceeds 45% of the arc it
  spans. Not one of the 8 clips is a loop, yet 6 of 8 fired; applying one would
  have folded the trajectory roughly in half. Thresholds are fractions of image
  width so KITTI's 640 px frames are judged like 1080p ones — that is where the
  one documented positive result lives (`--enable-loop-closure`, 144 → 77 m).

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
