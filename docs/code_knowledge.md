# Project code knowledge

Refreshed 10 November 2026 from the current checkout and closed experiment
records, including `d4c67ed`, `e292224` and `afb4b89`. Read this alongside the
dated records; a test passing or a candidate
route being produced does not establish localization accuracy.

## Active objective

Reach **15/15 original, available development clips within 200 m** and add
more videos. Score the actual selected `result.answer` with the unrounded
symmetric mean route error. Use one reproducible global policy across the
fifteen clips. Additional videos form a separate cohort; they do not replace
failed development clips or change that denominator.

## Current measured results

- Default retrieval baseline: **9/15**, median **135.10 m**, at `2b95f12`.
- Assisted v3: **12/15** in the completed `a8add7a` run at 7,200 seconds per
  clip. Five warm reruns at 14,400 seconds, combined with the unchanged ten
  outcomes, support **13/15**. This was not a fresh fifteen-clip repetition.
- Assisted v4: **14/15**, median **43 m**, in the complete `409a959` fleet,
  using 16 CPU threads and a 14,400-second budget. Every clip completed and
  passed the recorded evidence audit. Detroit `g5lnpYCk1Ec` remains at
  **1,387.24 m**. These are development results; v4 has no clean held-out
  validation.
- At the initial refresh the manifest contained 42 entries: 18 development clips, of which 15 are
  available, plus 24 available clips across `test`, `test2`, and `test3`.
  The three unavailable development entries remain documented.
- The retrieval track scored **5/7** on test1 and **2/7** on test2. Assisted
  v3 also scored **2/7** on test2. The opt-in metro-ring method scored
  **5/10**, equal to its ordinary-track control, on test3.
- Metro-ring's **20/29 versus 16/29** development comparison did not
  generalize to test3. It stays off by default. The denser first-pass pilot
  moved one Katowice result to 745 m; it crossed no clip below 200 m.

Sources: [assisted fleet](assisted_fleet_2026-11.md),
[test1](heldout_test_2026-10.md), [test2](heldout_test2_2026-11.md),
[test3](heldout_test3_2026-11.md), and
[benchmark declarations](../benchmarks/preregistered_rules.json).

## Code map

- `monocular_osm/pipeline.py` defines `PipelineConfig` and dispatches staged
  execution through `monocular_osm/stages/` and `RunContext`.
- `stages/prior.py`, `kartaview_vpr.py`, `regional_retrieval.py`, and
  `city_retrieval.py` implement the ordinary street-image retrieval track,
  region searches, and municipal coarse-to-fine search.
- `assisted_policy*.py` select among isolated children in
  `assisted_worker.py`. V4 adds literal shop-sign site verification in
  `poi_site_verify.py` to v3's retrieval, OCR/POI, and road-filter stages.
- Completed retrieval calls are checkpointed per call; terminal catalogs
  preserve persisted evidence after owned-child cleanup. A persisted call
  does not certify that its child completed or that its route is correct.
- `scripts/benchmark.py` runs frozen inputs and separately aggregates the
  selected Answer. `scripts/check_dev_goal.py` audits evidence and the
  fifteen-clip outcome. Component/oracle alternatives are diagnostics.
- `road_sign_clues.py` requires literal EXIT evidence and retains every
  matching public junction. `road_sign_context.py` plans bounded native OCR
  crops. `junction_sequence.py` is an experimental offline road sequence
  sampler; none of these modules establishes a newly admitted Answer.
- `scripts/overlay_clips.py`, `scripts/build_overlay_fleet.py`,
  `benchmarks/build_manifest.py`, and `scripts/check_overlay_gt.py` manage
  video registration and offline label acquisition/quality checks.

## Remaining development failure

Detroit's municipal search densified cells selected from already-decoded
coarse tracks. Cells needed later in the drive received no decoded support;
only the first portion was covered well. The current investigation is an
opt-in `--vpr-city-cell-selector=raw-votes` selector based on each coarse
cell's raw query/reference similarities. It retains the existing fine-cell
count, reference budgets, decoder, and admission policy for comparison.
`decoded` remains the default. The frozen `62868aa` Detroit pilot completed
all 27 coarse and three fine calls, selecting cells 1, 9, and 13. It retained
the baseline at **1,387.2377579105248 m**. Its expanded child was worse at
**1,714.6662206685678 m** in an offline component diagnostic, and failed the
existing raw-cosine admission margin. This is a negative experiment; no
fifteen-clip gain or adoption follows from it.

The simple two-exit VO similarity fit has already failed, including the
correct junction pair. Its correct-pair error was 1,923 m. A bounded full
road-graph sequence prototype with all OCR junction alternatives is now under
development. It must preserve uncertain advance-sign location/time, unknown
motion, scale, and competing modes; it has no validated result or policy
admission yet. See the Detroit discussion in the fleet record before
proposing a new pilot.

The original Detroit OCR has bare `186B` and `188`, with no literal EXIT.
The frozen `e292224` native-crop probe read all four eligible numeric seeds
and recovered no EXIT. A separately registered full-frame temporal probe
read all 29 planned native frames around those seeds, using fixed offsets
of +/-0.5, 1, 2 and 4 seconds and the same cached CPU OCR model. It recovered
literal EXIT references **186A, 186B and 190A** at their actual frame times.
The two 186B views are one correlated event. This verifies an input
prerequisite; no road sequence or new localization accuracy has been
measured. The graph adapter must preserve every matched OSM node ID and
directed incidence; nearest-road projection is insufficient at bridges.

The temporal probe closed in 49.562 s with 404,078,592 bytes peak sampled
aggregate RSS. Its immutable result is
`E:/monocular-osm-exit-temporal-proof/run1/temporal_result.json`, SHA256
`d86782eb13974c9874838fbd83a658f9d4f574352af971eb68e412be55e4a608`.

## Evaluation and operating constraints

- Inference uses the video and declared city/title cues. Burned-in coordinate
  stamps are excluded from OCR evidence. Offline ground truth never supplies
  inference seeds, a chosen candidate, or per-clip policy overrides.
- Used test1-test3 results have informed subsequent development. Preserve
  their historical held-out claims for the originally frozen methods; obtain
  a fresh cohort for a new generalization claim.
- New video selection must use fixed metadata/stamp/label-quality eligibility,
  record every rejection, and exclude aircraft footage from a road-driving
  cohort before localization. Freeze the tested method before new-cohort
  localization. Keep its labels out of algorithm development.
- Overlay acquisition writes window files such as `input_0-600.mp4` under
  title-derived directories. Existing test3 runs needed manual staging because
  `--skip-download` expected another directory/name. The benchmark now accepts
  registered `video.local_path` through `--video-path`, preserving URL identity
  and video timing. Relative paths resolve against the executing checkout;
  an isolated experiment can freeze an absolute-path manifest instead.
- Benchmark's ordinary timeout is shorter than the measured assisted budget.
  Set the intended CPU budget and CPU-only VRAM floor explicitly when
  reproducing an assisted run. Record warm caches and CPU contention.
- Since `d4c67ed`, benchmark score artifacts retain raw floating-point
  precision. A displayed 200.00 m can still exceed 200 m; use the unrounded
  metric and consistent threshold flags for goal checks.

## Stale state to avoid

Historical October 31 fields in `output/goal_progress.json`, including its
old `75406` live-session claim and 9/15-only status, are stale. Follow the
current `authoritative_checkpoint` and dated November records. The original
A8 run has finished and been scored. Preserve the old artifacts and verify
process state before resuming a job. The 15/15 objective is still incomplete.
