# Project code knowledge

Refreshed 11 November 2026 from the current checkout and closed experiment
records, including appearance prototype source `2e0f169` and dataset addition `fa7df5e`. Read this alongside the
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
- The manifest now contains **48 entries, 45 available**: 18 development
  clips (15 available), 24 across `test`, `test2`, and `test3`, and six new
  `test4` clips. The three unavailable development entries remain documented.
  TEST4's fixed metadata/QC search exhausted its pool at six usable videos;
  its planned target of seven was not met. No TEST4 localization has run.
  Its labels remain restricted to data QC until a global method is frozen.
  See [selection and acquisition](../benchmarks/test4_screen/README.md) and
  the [dataset freeze](../benchmarks/test4_screen/dataset_freeze.json).
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
  crops. `junction_inputs.py` preserves exact map topology and groups
  correlated sign views. `junction_sequence.py` is an experimental offline
  road sequence sampler; none establishes a newly admitted Answer.
- `scripts/junction_sequence_probe.py` separates guarded map preparation
  from the fixed full-window numerical trial. Compact network arrays,
  original native motion, literal signs, source hashes and video identity
  are bound before running; the helper never scores or selects an Answer.
- `raw_vpr_native.py` and `scripts/replay_raw_vpr_native.py` verify every
  frozen retrieval call against its OWN request, original native video pixels,
  and signed VO clock. Relative historical output paths require the pinned
  original execution root. This is input proof, not a descriptor/model replay
  or child-completion certificate.
- `raw_vpr_bank.py` validates an entire declared bank through the existing
  `metadata_union` identity/acquisition gates. Every physical copy is checked;
  overlapping IDs require exact coordinates, computed protected descriptor
  identity, and a full raw-score envelope within 1e-6. It streams descriptor
  pools and refuses the whole bank on conflict. It builds no likelihood field
  and changes no selection policy.
- `scripts/admit_raw_vpr_bank.py` consumes Root's closed native acceptance,
  the full frozen metadata/payload inventory and genuine descriptor receipts.
  It emits independent complete/unavailable bank outcomes before any field.
- `raw_vpr_appearance.py` and `junction_sequence.align_with_appearance`
  implement an explicit research prototype: one robust all-reference raster
  factor per query, fixed 30-second correlation groups and corrected initial
  proposal guidance. All visual alternatives remain in the field. The
  original road-only entry remains the default; no Answer policy uses this.
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
road-graph sequence prototype with all OCR junction alternatives was tested.
Its first frozen full-window trial completed but failed at
**6,323.285942994211 m**, worse than v4's retained Answer. It remains off
by default, with no policy admission. See the Detroit discussion in the fleet record before
proposing a new pilot.

The original Detroit OCR has bare `186B` and `188`, with no literal EXIT.
The frozen `e292224` native-crop probe read all four eligible numeric seeds
and recovered no EXIT. A separately registered full-frame temporal probe
read all 29 planned native frames around those seeds, using fixed offsets
of +/-0.5, 1, 2 and 4 seconds and the same cached CPU OCR model. It recovered
literal EXIT references **186A, 186B and 190A** at their actual frame times.
The two 186B views are one correlated event. This verifies an input
prerequisite; it did not produce a localization gain in the subsequent graph
trial. The graph adapter must preserve every matched OSM node ID and
directed incidence; nearest-road projection is insufficient at bridges.

The temporal probe closed in 49.562 s with 404,078,592 bytes peak sampled
aggregate RSS. Its immutable result is
`E:/monocular-osm-exit-temporal-proof/run1/temporal_result.json`, SHA256
`d86782eb13974c9874838fbd83a658f9d4f574352af971eb68e412be55e4a608`.

The failed `7ddea55` graph trial used all 4,440 native increments, composed
into 1,110 steps, all four literal observations and seven map hypotheses.
Its output was frozen and every worker closed before the one offline score.
The diagnostic score is
`E:/monocular-osm-junction-sequence-proof/run1/component_score.json`, SHA256
`ba9a65ba9f9a3b450d385bc6c390db7705182ba9ebb0af58ef0cdac9f99e6bd8`.
Its flat unnormalized turn potentials reward branching multiplicity; a new
conditional-uniform transition model was then tested at clean `72437d9`
with byte-identical typed map arrays, signs and hypotheses. It also failed:
**18,281.577297323704 m**, in 64.156 s. Neither graph-only variant is adopted.
The next investigation uses early raw VPR evidence as soft spatial mixtures
alongside road/sign constraints. The complete saved inventory contains two
baseline records and 40 expanded/control records, including five physical
copies. All 42 are retained; copies cannot create extra query observations.
Root's guarded native replay at `0695745` passed for every call: all 80 query
pixels and times match the original video, and each time is an exact member
of the signed VO clock. The half-second composed clock introduces at most
0.401956186775692 s of explicitly recorded lag. The replay closed in 55.110 s
with 147,611,648 bytes peak sampled aggregate RSS and no NN or GT use.

The native acceptance is
`E:/monocular-osm-joint-vpr-proof/run2/native_root_acceptance.json`, SHA256
`407e1495c0c4e729d9d5d6c7c842012dcf8c22c8a382232b2e7950f3e4cec5e5`.
Whole-bank admission at `b702c81` then accepted the complete baseline bank:
2,606 unique references from 2,699 occurrences, including 93 overlapping IDs.
Root independently checked every canonical coordinate/raw score and actual
descriptor row/tensor. Overlap coordinates/descriptors agree exactly; raw
score spread is zero. The expanded bank is **UNAVAILABLE** because a repeated
ID has conflicting protected descriptors. It was not rescued by dropping
cells or copies. The guarded job closed in 55.625 s with 151,584,768 bytes
peak sampled aggregate RSS. Root acceptance is
`E:/monocular-osm-joint-vpr-proof/run3/bank_root_acceptance.json`, SHA256
`84bcf5f54932eaee5622cc61b6d8061286bffe81101538b2d6cfbab00503156a`.

Historical reference encoding remains **UNVERIFIED**. A two-control fusion
profile was declared before loading reference/score arrays. Its off-default
source prototype passed 83 focused tests and independent review, but no actual
visual-field/fused-sampler trial or new accuracy result has run. The next run
can use the admitted complete baseline bank; the expanded control stays
unavailable. Development remains **14/15**.

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
  TEST4's cached URL metadata is under `data/test4/metadata_cache`, so its
  eventual offline benchmark needs `--data-dir data/test4`. Root verified
  all six literal video paths and cached URL identities without decoding,
  localization or label reads. Its method binding is still pending.
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
