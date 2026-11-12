# Project code knowledge

Refreshed 12 November 2026 from the current checkout and closed experiment
records, including masked-native DVLT source `3823286`, road-context source `67be7cb`, semantic-sidecar source `f2e8ee5`, joint-initial-proposal source `ad9cd96`, raw-query-veto source `b872862`, native multiview source `a902b8f`, 500k trial source `fcc3203`, adaptive-resampling source
`54a85e4`, earlier heading-variance
source `6d2768c` and dataset addition `fa7df5e`. Read this alongside the
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
- `graph_filter.GraphParticleFilter._block_stats` now ranks complete 3x3
  posterior neighbourhoods before retaining 64 candidates. Its shared occupied
  cell implementation includes empty neighbouring centres and avoids allocating
  the full dense grid. This corrects shared mode readout; accuracy at the revised
  source still needs evaluation.
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
- `scripts/raw_vpr_junction_probe.py` runs that fixed diagnostic from Root's
  accepted bank and fresh preparation at the same source, without selecting
  an Answer. It binds the complete original native clock and input chain.
- `reference_snapshot_evidence.py` retains distinct protected descriptor/raw
  feature versions as alternatives, with total weight one per provider image.
  Every original call passes its own per-pool checks before exact copies fold.
  `reference_snapshot_appearance.py` builds the corresponding weighted field.
  `scripts/admit_reference_snapshots.py` and the separate
  `scripts/reference_snapshot_junction_probe_v2.py` implement guarded research
  IO. V2 checks every ZIP member's declared size before reading any compressed
  NPY header. The preserved V1 numerical worker remains on hold. Neither
  representation asserts descriptor equivalence or verified encoder history.
- `junction_heading_likelihood.py` computes the complete active junction-arc
  union width and a circle-normalized predictive heading kernel. The explicit
  `junction_sequence.align_with_snapshot_normalized_heading` entry and
  `scripts/reference_snapshot_normalized_heading_probe.py` test that target
  separately. Existing entries and configuration stay unchanged. The new worker
  binds both snapshot-data and heading-model declarations before numerical
  inputs; its robust yaw-bias KF update remains approximate.
- `snapshot_future_guide.py` builds bounded backward visual messages for
  proposal guidance. The explicit legacy-heading and normalized-heading future
  entries preserve their respective target factors. Initial/turn proposals
  retain their original corrections; the new auxiliary mixture retains at least
  one quarter of current target mass before the exact importance correction.
  `scripts/reference_snapshot_future_guide_probe.py` binds the data, heading and
  proposal declarations before numerical inputs. None is a default pipeline or
  Answer-policy entry.
- `common_null_appearance.py` defines a separate visual density-ratio target.
  It normalizes the query-weighted and coverage-only Gaussian grids separately,
  mixes each with the same uniform background, and returns direct F32 log-ratios
  with neutral value zero outside coverage. Its sampler adapter verifies the
  original snapshot/native schedule. Explicit common-null entries and
  `scripts/reference_snapshot_common_null_probe.py` use EXIT-only proposals
  and preserve the original heading-variance model.
- `no_adaptive_heading_variance.py` provides independent motion copies with
  only native residual-derived heading variance replaced by zero. The explicit
  future-guided Q-ablation entries and
  `scripts/reference_snapshot_no_adaptive_q_probe.py` preserve original native
  variance separately from effective zero and composed unknown-motion variance.
  The fixed drift and observation noise remain; this is not measured covariance.
- `adaptive_auxiliary_resampling.py` supplies a fixed pre-guide target-ESS
  decision. New explicit future-guide entries and
  `scripts/reference_snapshot_adaptive_resampling_probe.py` use the half-particle
  threshold; skipped steps retain weights, state and identity ancestry. Triggered
  steps retain the original defensive guided proposal and importance correction.
  They export resampling decisions and distinct initial-label counts. No target,
  native heading variance or default pipeline policy changes.
- `high_particle_budget.py` defines a separate fixed 500k configuration and
  global ESS/defensive-guide calculations; it does not widen the original
  configuration or helper caps. Explicit high-budget entries use pointwise
  batches for future/normalized-heading helpers and retain one global auxiliary
  normalization. Only the final pure spatial summary is computed; the entire
  clock, ESS, ancestry diagnostics and edge/F64-offset genealogy remain intact.
  `scripts/reference_snapshot_high_particle_probe.py` binds its separate run13
  declaration and external 16 GiB RSS bound before inputs. Case execution is
  complete for all four controls; none passed 200 m.
- `raw_query_veto.py` defines a pure, fixed `.162` maximum-raw-score gate.
  Two explicit `junction_sequence` entries apply it only to target visual
  factors while retaining the original appearance object and all proposal
  guides. `scripts/reference_snapshot_raw_query_veto_probe.py` binds the
  separate run15 declaration, policy provenance and 33 source roles. It is
  diagnostic only and leaves production policy unchanged.
- `joint_initial_proposal.py` defines fixed guide-only speed bins with total
  tail coverage. The separate run16 integration conditions the initial road
  position proposal on these bins, preserving actual continuous speed draws
  and the original target through the exact conditional importance ratio.
- `road_segment_semantics.py` preserves loaded graph-artifact attributes,
  exact directed raw keys, keyed paths and mapped/neutral segment status.
  `road_segment_semantics_io.py` provides separate bounded storage, and
  `scripts/prepare_road_segment_semantics.py` is the opt-in producer. The old
  network remains an exact 18-array archive; no speed model or Answer policy
  consumes the sidecar yet.
- `multiview_motion.py` is an opt-in seven-frame geometric frontend promoted
  byte-for-byte from a reviewed prototype. It estimates FIT-only anchor E/PnP
  poses with held-out later-view checks and preserves unknown translation.
  `multiview_motion_inputs.py`, `scripts/prepare_multiview_motion_inputs.py` and
  `scripts/multiview_motion_probe.py` form a separate two-phase native-window
  diagnostic. The first phase verifies the full original pixel/clock stream
  and freezes all 32 packets; the numerical phase uses those closed packets
  while original video/cache assets remain opaque historical proof. Full raw
  tracking arrays, lost IDs and every outcome are retained. Each window has an
  independent unit gauge, without a global route, calibrated covariance or
  metric-scale claim. These entries do not change pipeline defaults.
- `scripts/overlay_clips.py`, `scripts/build_overlay_fleet.py`,
  `benchmarks/build_manifest.py`, and `scripts/check_overlay_gt.py` manage
  video registration and offline label acquisition/quality checks.

## Remaining development failure

Detroit's municipal search densified cells selected from already-decoded
coarse tracks. Cells needed later in the drive received no decoded support;
only the first portion was covered well. The earlier investigation tested an
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
The subsequent investigation used early raw VPR evidence as soft spatial mixtures
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

Historical reference encoding remains **UNVERIFIED**. The two-control fusion
profile was declared before loading reference/score arrays. Root ran the
complete baseline-bank control at `f699182` after fresh same-source map
preparation. All typed geometry, topology and cue records matched the earlier
graph trial. The 50,000-particle/1,110-step trial closed in 47.047 s at
1,182,334,976 bytes peak sampled aggregate RSS. Root rebuilt the entire
6,573,600-value field exactly and checked all 80 query events, 20 correlation
groups, original cue powers and native endpoints before scoring.

The unchanged lineage scored **7,152.842366478733 m**, worse than v4's
retained Answer. This is another negative diagnostic, with no policy adoption
or fleet improvement. Its score is
`E:/monocular-osm-joint-vpr-proof/run4/component_score.json`, SHA256
`02b0f30a9ac8ce5d6099cee49fe1ea84f8ea7f8a5b15e23e702f24db36044188`.
The expanded strict control stays unavailable. Development remains **14/15**.

A report-only check found that the first rejected expanded overlap differs
by Float64 L2 **1.3859061218023863e-7**, with exactly equal coordinates and raw
score spread **2.2351741790771484e-8**. This does not establish compatibility
for the remaining bank or verify historical encoding. A separately declared
research study will apply the project's earlier **L2 <= 1e-6 against the
lexical canonical vector** rule, preserving exact per-pool tensors, receipts,
query/model/clock/coordinate checks and full raw-score envelopes. It requires
a new compatibility status/profile; old strict failures and production
identity gates remain unchanged. That separate `b74f18d` study also refused
the expanded bank: another duplicate has L2 **9.086784224324718e-6** and raw
score spread **4.023313522338867e-7**, with equal coordinates. The complete
baseline's five arrays are exactly identical to the previously scored strict
control. No new sampler or accuracy score ran. The guarded study closed in
58.344 s at 324,284,416 bytes peak sampled aggregate RSS.

The later separately declared representation retained distinct protected
reference-feature snapshots as alternatives. Each provider image has total
weight one, split uniformly over its distinct descriptor/raw-column snapshots;
exact physical copies add no weight. It retains every original per-pool and
native-input gate and makes no descriptor-equivalence or encoding-history
claim. The producer at `bb299e5` completed all 42 physical calls in 112.547 s
at 172,994,560 bytes peak sampled aggregate RSS. Root independently verified
every original occurrence, raw column, coordinate, actual descriptor row and
consumed tensor, exact-key folding and per-provider weights. Baseline contains
2,606 snapshots/provider images; expanded contains 16,072 snapshots of 13,164
provider images across all 25,631 original occurrences. Its separate acceptance
is `E:/monocular-osm-joint-vpr-proof/run6/evidence_root_acceptance.json`, SHA256
`62523a86ce98b4b5fb0190784bcfe717ead1d89900ca367f585c9db1750aa90a`.

Both fixed 50,000-particle controls then ran at `e344755`, after fresh preparation
with exactly equal typed geometry and cue/topology records. Root rebuilt their
entire fields (6,573,600 and 10,568,560 F32 values), checked all native motion
arrays, 80 query events, correlation powers, original cues and reconstructed
road points, and closed both predictions before scoring either. Baseline
exactly reproduced the prior **7,152.842366478733 m** error. Expanded scored
**6,338.264241249098 m**. Both are negative against the retained v4 Answer;
neither changes policy or the **14/15** fleet result. Trial directories are
`E:/monocular-osm-joint-vpr-proof/run6/trial_baseline_v2` and
`E:/monocular-osm-joint-vpr-proof/run6/trial_expanded_v2`. These checks do not
replay every particle or prove directed transition paths, MAP optimality or
calibrated confidence.

Source/private review then found two distinct issues. `_block_stats` selected
64 individual cells before summing neighbourhood mass; a 73-particle fixture
can hide a heavier nine-cell cluster behind 64 lighter isolated cells. The
summary-only correction passed 22 new tests and 165 relevant regressions; Root
also reproduced the old failure using the previous implementation. A private
50,000-particle benchmark took 0.189 s and about 42 MB additional traced peak
allocation, without a full dense-grid allocation. This is not a certified RSS
peak. The separately declared run7 replay at `3001550` completed both controls
in 65.312 and 68.062 seconds. All saved ESS bytes, complete fields, query events
and sign-factor records match their pre-correction controls exactly. Both
selected lineages also match exactly, and their single offline scores remain
7,152.842366478733 and 6,338.264241249098 m. The bug is fixed, but this readout
correction does not repair these routes or improve the fleet. Proofs are under
`E:/monocular-osm-joint-vpr-proof/run7`. Separately,
the heading model gives zero residual throughout a junction arc while keeping
point-Gaussian normalization. Private integration gives total factor mass
1.000, 2.593 and 4.187 for point, 90-degree and 180-degree expectations with the
same variance/outlier settings. A normalized heading model was separately
prospectively declared under
`E:/monocular-osm-joint-vpr-proof/run8/heading_study_contract.json`. The explicit
source/private implementation passed 199 relevant tests, including 48 pure
kernel and 17 new integration/worker tests. Root repeated those 65 tests and
60 independent scalar quadrature checks. Both peers approved integration,
including exact old-entry outputs, final particle states and RNG equivalence.
Fresh preparation and both fixed controls at `d4266f5` then completed. Root
reconstructed their fields/native clocks/road points and verified the new
target profile; both predictions closed before either score. Baseline scored
**18,947.181863246762 m** and expanded **20,134.2580526717 m**, worse than both
the earlier snapshot controls and retained v4 Answer. The normalized target
remains off by default, with no policy admission or fleet improvement.
The mathematical normalization finding did not repair these routes. Run8
retains all immutable output and closure records. Run9 then tested separately
declared future appearance guides and defensive auxiliary sampling over both
fixed heading targets.
The proposal implementation passed 267 relevant tests, including 44 pure-guide
and 24 new integration/worker tests; Root repeated those 68 tests. Peers verified
exact old-entry outputs, particle states and RNG for both heading targets.
A CPU2 private benchmark of the maximum 16M-value field and 1,201 endpoints
took 4.01 s, with 66.65 MB additional traced memory and 216.32 MB sampled total
RSS, not an absolute peak guarantee. The guide is query-spacing dependent and
is not a physical motion posterior. Gaussian sigma above eight cells refuses
the whole control before allocation; it is never clipped or cropped. Run9's
final prospective contract is `future_guide_study_contract_v2.json`; the first
contract is preserved unlaunched. All four fixed controls at `05b06c5` completed
after fresh 31-source preparation. Root rebuilt each entire visual field and
backward guide volume, checked native/event/road-point integrity and closed
every prediction before any score. Symmetric means were:

| Heading target | Baseline references (m) | Expanded references (m) |
|---|---:|---:|
| Original junction heading | 4,910.147300513726 | 6,041.076373316711 |
| Normalized junction heading | 3,605.817422391153 | 1,058.1609295408853 |

The expanded normalized control improves on the retained 1,387.24 m diagnostic
comparison, but still fails 200 m. Its start/end errors are 5,200/5,913 m and
its predicted-to-GT mean is 91.54 m: the lower symmetric mean does not establish
a complete correct route. No per-clip control selection, Answer policy adoption
or fresh fleet gain follows. Development remains **14/15**. Proofs are under
`E:/monocular-osm-joint-vpr-proof/run9`.

Source/private review also found that the current visual compatibility
potential strongly rewards reference coverage even for a flat query; this is a
model effect, not an arithmetic bug. A separately declared common-null density
ratio makes flat queries and unsupported regions neutral in private tests.
Its implementation passed 301 relevant regressions. Root repeated 37 pure and
34 integration tests plus 60 scalar and four exact-flat checks; peers confirmed
old-entry outputs, particle states and RNG remain unchanged. The full-cap CPU2
field benchmark took 0.316 s with 83.9 MB additional traced memory, not a peak
RSS guarantee. All four fixed EXIT-only controls at `7168b55` completed under
`E:/monocular-osm-joint-vpr-proof/run10` after fresh 31-source preparation and
complete direct-field/native/event/road-point checks. Every prediction closed
before any new score. Symmetric means were **5,020.884804690638 / 10,531.389060528203 m**
for baseline/expanded with original heading and **11,178.426729963401 /
10,820.891157969181 m** with normalized heading. All fail the target; the common-null
model stays experimental, with no adoption or fleet improvement.
A separate heading-process audit found that inlier counts
do not affect the current yaw-derived variance; an abrupt 90-degree jump is
confounded with noise, whereas a smooth three-second turn receives small
variance. No measured geometric covariance has been replayed on the case.
The separate run11 ablation preserved the run9 visual target and future
proposals while removing only native residual-derived heading variance. It
retained fixed yaw drift, unknown-motion uncertainty and all original motion
means. This model change remains separate from run9/run10 results. Its explicit
implementation passed 367 relevant private tests. Root repeated 32 pure and
34 integration tests; both peers approved, including exact old future/common-null
entry outputs, particle states and RNG. Fresh 33-source case preparation and
four fixed controls completed at `6d2768c`. Original and effective variance are
never relabelled as each other. Root verified unchanged original motion and
variance, effective native zeros, preserved unknown-motion composed variance,
complete fields/backward guides, native schedules, signs and road points. All
four predictions closed before scoring. Errors were **2,210.963842246671 /
2,991.919359098665 m** for baseline/expanded with original heading and
**1,421.750511279427 / 19,686.662627628422 m** with normalized heading.
All fail 200 m; no variance-policy or fleet gain is established. Proofs are
under `E:/monocular-osm-joint-vpr-proof/run11`. Development remains **14/15**.

The offline study workers still use fixed research declarations and historical
input-auditor modules. They are not a generic pipeline integration. The
source-only design under `output/snapshot_joint_runtime_design_20261111`
describes tracked native proof, immutable OWN-call packets, a new opt-in
post-v4 fallback child, exact Answer/audit binding and a shared parent deadline.
Post-v4 work needs a prospective time reserve inside the same 14,400 seconds;
it cannot create a fresh per-phase budget. The cosine admission proof cannot
certify a PF route, and posterior concentration is not calibrated accuracy.
Any useful method still needs one globally frozen policy and a fresh original
fifteen-clip selected-Answer evaluation. No tested diagnostic is adopted here.
The source/private audit is under
`output/sequence_motion_source_audit_20261111`. Unconditional auxiliary
resampling also has a demonstrated ancestry-variance risk under changing guides,
but constant uniform weights/guides resample identically; collapse is not
inevitable. No motion or resampling change is part of the summary correction.

Run12 separately declares a fixed adaptive-resampling study over the run9
parents. All four bank/heading controls, 50,000 particles, seed zero and a strict
pre-guide ESS threshold of half the particle count are fixed before new values.
The parent runs resampled all 1,110 endpoints despite high target ESS at most
steps; that is motivation, not proof of a failure mechanism. The new diagnostic
counts unweighted initial labels, not correct routes or posterior confidence.
Root repeated 24 pure and 52 integration/worker checks. Historical-source
comparison also preserved all six existing future/common-null/Q entry outputs,
final particle arrays and RNG exactly. Fresh same-source preparation and all
four controls completed at `54a85e4`; all predictions closed before scoring.
Root verified complete fields and backward guides, original native motion and
Q, all observations, road points, strict ESS decisions and label-count invariants.
Baseline/expanded errors were **1,619.8912780797152 / 6,040.932357636948 m**
with original heading, and **3,603.439024702106 / 20,274.605749613336 m**
with normalized heading. Resampling occurred 193/64/175/111 times rather than
1,110; final distinct initial-label counts were 15/6/1/1. This does not measure
correct-route support or establish a parent ancestry comparison. All four fail
200 m, so no policy is admitted and the validated fleet stays **14/15**.
Proofs: `E:/monocular-osm-joint-vpr-proof/run12`.

A bounded private 50k/500k feasibility study under
`output/high_particle_budget_feasibility_20261111` found that full summaries
can dominate a larger sampler: diffuse 500k summaries took 5.08 s each in its
fixed synthetic fixture. That is not an actual-city runtime. A proposed larger
budget should preserve the entire native window and exact edge/offset genealogy
while computing only the final pure summary used for route readout. This needs
separate source/prospective validation. Run13
prospectively fixes four 500k/seed-zero controls, an 1,800-second inner limit,
12 GiB genealogy payload cap and separate 16 GiB external RSS cap with 8 GiB
physical/commit headroom. A private AST comparison verifies that final-only
summaries preserve full PF state/RNG and final routes. Every unavailable or
failed outcome must close before scoring complete routes; none may disappear
from the experiment inventory. The pure helper passed 52 checks and an
independent peer; Root repeated 68 integration/worker checks. Peers verified
all eight older entry outputs, final PF arrays and RNG exactly, genuine 500k
chunk cache/order, and all worker source/declaration gates. The four actual
controls at `fcc3203` then completed after fresh 33-source preparation in
20.563 s. Root verified unchanged parent filter parameters except particle
count, complete fields/backward guides, original motion/Q, observations, road
points, strict ESS decisions and sparse final-summary indices. All four full
predictions and workers closed before any score. Errors were
**2,289.2568063537246 / 13,752.969619377378 m** with original heading and
**1,136.6058158505887 / 3,529.6162976527376 m** with normalized heading,
baseline/expanded respectively. Runtimes were 283.640/230.672/272.282/237.328 s;
peak sampled aggregate RSS stayed below 9.39 billion bytes, within the separate
16 GiB cap. Resampling counts were 203/60/341/191, with 52/39/6/13 initial
labels remaining. These are unweighted genealogy counts, not correct-route
support. All fail 200 m. The lowest mean also has 5.2/5.7 km endpoint errors and
a 2.43 km route versus the 10.37 km GT. No larger-budget policy is adopted;
validated development stays **14/15**. Proofs:
`E:/monocular-osm-joint-vpr-proof/run13`.

A source/synthetic motion review under
`output/multiview_motion_audit_20261111` also distinguishes observable rotation
from observable translation. A pure-rotation fixture can fail essential-pose
cheirality while preserving accurately recoverable rotation. The current cache
loses that rotation and resets each accepted translation to unit length. A
persistent multi-view shared-scale front end is a proposed new direction; no
actual-video gain, measured metric scale or covariance is established by these
fixtures. The earlier depth, XFeat, LingBot and fixed-height controls remain
closed separately. A subsequent private estimated-motion frontend uses
FIT-only essential geometry and PnP with fixed held-out tracks. It retains
relative translation magnitudes and refuses weak geometry. One physically
consistent synthetic image fixture passes, while a negative image fixture
remains refused. The source-only next plan freezes 32 uniform seven-row native
windows with exact ordinal/PTS lineage, one window resident and no GT use.
No actual-window replay, continuous gauge connection or localization result
is established by those private checks. The separate prospective native pilot is under
`E:/monocular-osm-multiview-motion-proof/run1`. Root derived the fixed schedule
from 3,360 accepted ordinal/clock witnesses: 4,441 original READ rows, first
video ordinal 119, stride four, last 17,879. This is not a timestamp-to-frame
inference. The 32 interior centres include first/last READ rows through their
seven-row windows and include formerly pose-invalid rows. New replay must
match positive OpenCV-reported times and the original dense pixel hash; no
independent container-PTS or camera-calibration upgrade is implied. Source
validation passed 71 focused checks (16 frontend, 34 input, 21 numerical),
repeated by Root and independent peers. The actual pilot at `a902b8f` then
completed both guarded phases. Input replay verified all 4,441 original native
frames/clocks and saved all 224 selected frames in 32 packets: 157.750 s,
116,174,848 B peak sampled aggregate RSS. Root independently checked every
packet/header/frame hash and closed all input actors before geometry. Numerical
evaluation took 7.187 s at 89,292,800 B peak sampled aggregate RSS, with original
raw assets remaining opaque. Every raw tracking record and all 32 outcomes
were retained. **All 32 windows were unavailable**: 26 failed triangulation or
held-out cheirality, five failed the parallax floor and one failed later-view
held-out geometry. No thresholds, windows or source changed after pixels.
Root verified typed outputs, fixed seed splits, clock/ordinal bindings and
actor closure. These counts locate refusals; they do not identify true pose,
intrinsics, outlier cause or calibration. No new motion/gauge/route/accuracy
is adopted, and development remains **14/15**. Completion proof:
`E:/monocular-osm-multiview-motion-proof/run1/study_completion.json`.

A source-only title audit under `output/title_route_intent_audit_20261111`
confirms expanded search already uses multiple named places and undirected MST
corridors. It loses ordered origin/destination roles. A soft directed intent
model remains unimplemented; city-centre attraction or shortest corridors can
miss a cropped drive, as earlier negative coverage studies already show.

## Closed absolute-query veto and shared text fix

A bounded raw-query eligibility audit under
`E:/monocular-osm-joint-vpr-proof/run14` found that the inherited `.162` cutoff
enables 63/80 baseline and 75/80 expanded query maxima. It therefore is not a
no-op. This cutoff was designed for a POSTHOC DEV decoded-track median, not a
calibrated query inlier probability. Run15 prospectively declares one separate
heuristic target change: multiply each original visual log factor by that
query flag, retaining all 80 events and original powers without renormalizing.
Original initial/turn/ancestor/backward guides remain ungated with exact
importance corrections. Source `b872862` contains the explicit opt-in helper,
sampler entries and worker. Root repeated 96 focused and 71 existing tests;
independent reviews verified the old particle state and RNG remain exact.
Fresh preparation matched all 18 network arrays and full metadata. All four
full-window predictions, worker births and frozen bytes were closed before
any new ground-truth score. Each retained all 80 query events and native Q.

| Bank / heading | Parent run9 error (m) | Query-veto error (m) |
| --- | ---: | ---: |
| Baseline / legacy | 4,910.15 | 4,910.17 |
| Expanded / legacy | 6,041.08 | 6,041.08 |
| Baseline / normalized | 3,605.82 | 3,608.23 |
| Expanded / normalized | 1,058.16 | 1,064.98 |

All four fail 200 m at raw precision. Runtime was 70.47–76.06 seconds per
prediction under CPU2/RSS3GiB limits. The lowest-error variant still comes
from a partial route: expanded/normalized predicts 2,703 m against 10,368 m
of ground truth, with start/end errors of 5,200/5,896 m. No candidate was
adopted; the global selected-Answer result remains 14/15. Complete records:
`E:/monocular-osm-joint-vpr-proof/run15/study_completion.json`.

`scene_text.is_position_stamp` now also rejects compact DMS fragments with
minute/second marks and an explicit hemisphere, including missing degree marks
and short fragments. Normal road references, cardinal words and apostrophe
names remain available. The shared fix applies to fresh and cached extraction
and restored context; it does not rewrite cache artifacts or change GPS/GT
parsers. The author/peer focused suite passed 84 checks plus 26 independent
string variants. Root repeated 75 scene/context/clue tests. Reapplying the new
filter to the old frozen temporal OCR excludes one previously missed
coordinate-like row while preserving all four EXIT refs, times, confidences
and pairings. This is an exclusion fix, not an accuracy gain; old artifacts
and the last validated 14/15 remain historical evidence.

## Closed joint speed-position initial proposal

The current initial position guide assumes 8 m/s, then independently draws
continuous cruising scale and its deviation. That is a valid importance
proposal, but can start position and speed poorly coupled. A separate design
samples the unchanged continuous speed prior first, then road position
conditional on a fixed guide-only speed bin. The exact correction is the road
cell prior divided by that bin's actual conditional proposal; no speed or
bin-mass factor applies. Defensive mixing bounds this ratio by four.
Two independent finite-state proofs recover the original target expectation.

The design uses 16 fixed log-spaced representatives over 0.5–55 m/s, with
unbounded exterior bins; actual speed samples and tails remain unchanged.
One full road partition and one conditional guide/proposal are active at a
time. It changes only initialization, with a separate opt-in profile and
new RNG order. It does not combine the query veto, adaptive resampling or
larger particle count. Source-only design:
`output/joint_speed_position_proposal_design_20261111/design.json`.
Run16 prospectively declares this separate proposal. The fixed bin helper has
36 passing private tests and an independent review. The initializer passed
a private 200k-cell/100k-edge/50k-particle/all-16-bin benchmark in 1.60 s,
with 39,026,258 bytes of traced extra scratch against 64 MiB. G0 was computed
once and one active future raster was reused; this is an initializer-only
benchmark, not a full-filter RSS guarantee. Root repeated 106 new and 98
existing private tests; independent review preserved complete old particle
state and RNG. The separate worker pins 33 source roles and all four study
declarations before values. Source `ad9cd96` then completed all four actual
controls. Fresh preparation preserved all 18 network arrays and full metadata.
Root independently reconstructed the full 180,027-cell partition and every
one of 50,000 initial weights per control, with exactly zero difference. All
four outcomes and worker births closed before any new ground-truth score.

| Bank / heading | Parent run9 error (m) | Joint-initial error (m) |
| --- | ---: | ---: |
| Baseline / legacy | 4,910.15 | 1,998.79 |
| Expanded / legacy | 6,041.08 | 7,155.23 |
| Baseline / normalized | 3,605.82 | 5,174.74 |
| Expanded / normalized | 1,058.16 | 1,065.79 |

All four fail 200 m at raw precision. The best still predicts a partial
2,586 m route against 10,368 m of ground truth, with start/end errors of
5,250/5,896 m. No proposal was adopted and no selected-Answer fleet ran.
Prediction runtimes were 73.19–80.22 s and maximum sampled aggregate
RSS was 1,296,850,944 bytes, below the 3 GiB bound. Complete records:
`E:/monocular-osm-joint-vpr-proof/run16/study_completion.json`.

## Frozen map tag availability

A separately declared Root-only metadata inventory closed under
`E:/monocular-osm-joint-vpr-proof/run17/audit_external_closed.json`. The original
unsimplified graph has 175,570 nodes and 330,592 directed edges. Highway class,
oneway, original way ID and reversal metadata occur on every edge; road refs
on 16,222 and maxspeed tags on 134,951. Directional/conditional maxspeed and
all tested destination fields are absent. No numeric speed values, images,
OCR, ground truth or route inference were used. This supports investigating
an exact original-segment road-class sidecar; it cannot support destination
associations from these absent tags. Legal maxspeed is not observed speed or
a hard motion bound. Directional/conditional fields were not requested by the
current useful-tag list; their zero coverage does not establish absence in
OSM. A `reversed=False` value also cannot certify original way-forward
orientation for `oneway=-1`.

Source/private design found a public simplification aggregation seam that can
preserve ordered original `(u,v,key)` tokens before attribute deduplication.
Five fabricated-graph checks preserve all 18 compact arrays exactly, directed
orientation, EXIT boundaries and explicit parallel-key loss. The sidecar must
retain exact identities and missing/ambiguous values, with no nearest-road
association or averaging of speed tags across merged segments. It remains a
design, not a new runtime input.

Two independent designs support a normalized, persistent speed transition
using the previous physical segment and measured `step/dt`. The existing
`exp(logk+eps)/unit_seconds` is only a cruising-rate proxy. Unknown, stationary,
mixed and missing-tag intervals must stay neutral. The same clipped log-scale
kernel cancels, including its boundary atoms; a normalized epsilon mixture
needs an exact target/proposal ratio once. A broad target with the old narrow
proposal can have unstable weights; direct target sampling or a declared
defensive mixture needs private validation. No road-speed parameters have
been fitted and no model is implemented. Designs and private proofs are in
`output/road_conditioned_speed_design_20261111/`.

## Sealed original-segment semantics preparation

Source `f2e8ee5` implements the opt-in Run18 preparation under
`E:/monocular-osm-joint-vpr-proof/run18/segment_semantics_study_contract.json`.
Root repeated 225 new/focused and existing checks. Independent reviews
preserve old construction math, all 18 arrays and particle state/RNG. The Root
validator passed 12 private end-to-end probes, including resealed corruption
checks; it independently replays keyed simplification because geometry alone
cannot distinguish coincident parallel paths.

The actual full-map producer closed in 60.48 s, peak aggregate RSS
2,610,544,640 bytes. Its 178,423,446-byte catalog preserves all 330,592 loaded
raw edge records and every attribute value/type. The seven separate semantic
arrays occupy 16,861,000 bytes. The independent validator closed in 35.47 s,
peak 2,646,536,192 bytes. Both stayed within CPU2/RSS3GiB, headroom and deadline
limits, with all observed worker births closed and protected bytes unchanged.

Root verified all 18 network arrays and full old network/matched/topology/plan
metadata exactly against Run16. Every compact key's original path was replayed;
every raw disposition and all 330,592 segment associations were rebuilt. All
segments map exactly in this packet. A further 754,188 previous-segment lookup
checks cover segment starts, midpoints and final edge endpoints. Numeric base
maxspeed context is available on 134,951 raw records; 195,641 remain missing.
No missing state was removed or assigned an invented speed.

The packet is sealed by
`E:/monocular-osm-joint-vpr-proof/run18/semantic_root_acceptance.json`.
It proves preservation of this loaded graph artifact, with source and resource
binding. It does not upgrade original OSM coverage, way-forward direction or
acquisition history. The old network remains an exact 18-array archive, and
the sidecar remains separate. No road-speed model, new route, selected Answer
or fleet score was produced; the last validated fleet remains 14/15.

The next step is a separately declared, normalized road-context epsilon
transition using measured `step/dt`, preserving unknown/stationary/mixed and
missing-tag intervals. It must use genuine joint sampling or an exact full
proposal density, retain broad slow support, and treat base speed tags as
uncertain context. Legal limits are not measured travel speed. No new model
parameters or implementation have been adopted. A successful component still
needs generic runtime integration and a fresh original-fifteen Answer run.

## Closed road-context transition study

Run19 froze one new target before any new route:
`E:/monocular-osm-joint-vpr-proof/run19/road_speed_study_contract.json`.
The original clipped cruising-scale update and Gaussian epsilon innovation
remain in their original order. On fully reported measured-moving intervals
with exact scalar map context, the epsilon transition directly samples a
normalized 25% parent / 75% road-context mixture. Road variance adds 0.75
squared to the parent stationary diffusion variance. Unknown, stationary,
mixed and missing-context intervals retain the parent transition.

This is a weak, uncalibrated anchor: constant-context stationary epsilon mean
is 0.75 times the road equilibrium shift, and both slow and fast tails widen.
It is not a persistent traffic-regime model or an exact OU semigroup. Numeric
base tags remain uncertain map context, not measured speed or a hard legal
cap. The model uses measured `log(step)-log(dt)` and the previous physical
segment; `pf.index=t` remains unchanged while motion support uses `t-1`.

Independent mathematical, sampling-hook and context-cache reviews are GO
with these explicit limits. The implementation is held and privately tested;
382 focused and parent-regression cases pass at Root.
Five fixed transition steps exported all 50k particle pre/post states and
pre-predict PCG64 states for Root replay. The four original bank/heading
controls retained all visual/EXIT/native-Q/proposal/genealogy settings; no
closed query-veto, joint-initialization, adaptive or high-N variant is mixed
in. All four predictions closed before any new score. All twenty full-cloud
transition snapshots replayed exactly, including clipped K, actual Gaussian
innovation, uniforms and epsilon. Fresh basic preparation preserved all 18
network arrays and the complete four old metadata objects against R18.

| Bank / heading | Run9 parent error (m) | Road-context error (m) |
|---|---:|---:|
| Baseline / legacy | 4910.147300513726 | 1638.2766488560471 |
| Expanded / legacy | 6041.076373316711 | 19760.57970939646 |
| Baseline / normalized | 3605.817422391153 | 18845.335325828848 |
| Expanded / normalized | 1058.1609295408853 | 20289.24917215141 |

All four failed 200 m. One paired control improved and three worsened; the
best new diagnostic also remained worse than the retained Detroit Answer.
No parameters were retuned, no route or policy was adopted, and no new
selected-Answer fleet ran. The validated fleet remains **14/15**. Prediction
workers used 116.094–177.953 s and peaked at 1,312,571,392 bytes aggregate RSS,
within two-CPU, 3 GiB RSS and 8 GiB headroom guards. Full-window single lineages were scored
unchanged; full PF/genealogy replay or physical-speed calibration is not
claimed. Completion: `E:/monocular-osm-joint-vpr-proof/run19/study_completion.json`,
SHA256 `af1f1e036e3fd0428648b0442a8958a42dd75b27242eed6461be26b1ee394cad`.

The benchmark analysis window is separate from container duration. Overlay
fleet acquisition declares ten-minute windows and can narrow each to a
continuous annotation segment; the manifest supplies its explicit
`vo_segment` to frame extraction. Scoring uses the complete frozen GT file.
A long upload does not imply that the benchmark should localize its entire
duration. Source rules alone do not prove a particular cached clock/annotation
endpoint join; preserve the registered case window and full GT during tests.

A bounded 11 November primary-source scan also found
[Déjà View/DVLT](https://research.nvidia.com/labs/dvl/projects/dvlt/), a
NeurIPS 2026 joint multiview camera/geometry model with
[official code](https://github.com/nv-tlabs/dvlt) and published weights.
Its official commit `f59bb18259ab446a7585c4f47731a77cfdefa426` now passes two
separately registered synthetic CPU profiles in an isolated Python 3.12.16
environment with the exact core dependency pins and PyTorch 2.5.1+cpu.
The complete publisher checkpoint matches its SHA256; Root independently
matched all 368 stored tensors and 117,081,731 F32 state elements to the
actual loaded model. Upstream math is unchanged. A scoped relative-file
loader call avoids an upstream Windows drive-path bug. All parameters and
buffers stayed on CPU in eval mode, and the state/buffers remained unchanged.

| Synthetic input | Forward time (s) | Guarded total (s) | Peak sampled aggregate RSS (bytes) |
|---|---:|---:|---:|
| Two views, 224 square | 2.719 | 10.484 | 1,267,519,488 |
| Two views, 504 square | 17.422 | 24.922 | 1,383,497,728 |
| Seven views, 504 square | 83.812 | 91.500 | 2,882,199,552 |

All three separately registered K12/F32/seed0 profiles returned finite arrays within CPU2/3GiB/8GiB
headroom guards. All generated pixels, preprocessing tensors, output headers
and loaded-state witnesses were independently verified, and every actor
closed. An initial import-validator false refusal for Torch's virtual
namespaces is retained alongside its corrected successful gate.

The seven-view test retained all nine arrays (90,684,300 bytes) under its
prospective 128 MiB payload / 256 MiB archive caps. Root independently
verified all seven generated frames, exact preprocessing, finite outputs and
the same complete loaded state. This establishes only the measured synthetic
profiles. Other batch/long-video capacity, native-video geometry, calibration,
metric scale, covariance and localization accuracy remain unverified.
No actual video/GT, new selected
Answer or fleet run was used; R19 stays closed negative and TEST4 stays blind.
The separately declared masked native-window pilot below requires completed
detection-only quads, independent track/reprojection support and an explicit
unknown scale. The original isolated windows cannot prove a common global
gauge. Two-view acceptance:
`E:/monocular-osm-dvlt-feasibility/run1/synthetic_two_profile_acceptance.json`,
SHA256 `a2c3f29dfb058c605344402dd5f03b3b5cbbcc85b2ae2ec398ff80e8ab665351`.
Seven-view acceptance:
`E:/monocular-osm-dvlt-feasibility/run1/synthetic_seven_view_acceptance.json`,
SHA256 `186b1a0241d8a191b0bd65082b16bfe777c533ebc0f4b1090c95b93a1e8373a2`.

## Masked native DVLT diagnostic

Source `0f4f0ec` adds diagnostic workers; `3823286` binds their prospective
fixed32 study. Root's 147 focused tests passed after binding, and 34 private
mask/controller checks passed separately. This code does not select a route
or change an assisted policy.

- `native_text_detection.py` witnesses genuine detection-only RapidOCR
  session, postprocessing and native-output completion. It preserves all raw
  quadrilaterals and separate, explicitly bounded mask operands. Completed
  empty output is distinct from an unavailable detector. Recognition is off.
  The unchanged `native_pixel_mask.py` applies quarter-height edge bands and
  all padded text rectangles on the original pixel grid. Neither certifies
  detector recall or complete removal of location text.
- `dvlt_native_protocol.py` and `prepare_dvlt_native_masks.py` separate new NN
  study authority from the old packets' pixel/ordinal/clock DATA provenance.
  Root independently reconstructs every accepted mask and BGR byte. Numerical
  actors consume only Root-closed masked packets; original unmasked NPZs,
  video, caches and historical protected maps remain opaque to them.
- `dvlt_native_runtime.py` uses the genuine pinned public K12/F32/CPU2/seed0
  path, complete raw-to-loaded state equality and unchanged CPU/eval buffers.
  The actual preprocessing tensor and every public prediction are retained.
  Its complete live environment and official core pins are verified. The
  clean E checkout has 112 distributions; the historical 113-entry record
  also discovered C working-directory project egg-info. Both records remain
  preserved, with that context difference explicit and no package changes.
- `dvlt_native_geometry.py` extracts independent masked-support GFTT/LK
  tracks, retaining original seed IDs/parity and all invalid/rejected rows.
  It maps native indices to the official processed pixel centers exactly.
  Fixed views 0/3 condition landmarks; observed views 1/2/4/5/6 test poses
  without CHECK fitting. A FIT-only rotation alternative refuses observable
  translation when rotation explains the held CHECK tracks. CHECK is held
  out of post-fit geometry; all seven images are seen by DVLT.
- `dvlt_native_window_probe.py` preserves pose-only and model-depth/pose
  reports separately, including failed state checks and nonfinite arrays.
  Root independently verifies typed outputs and recomputes pure geometry
  from retained observations. Continuous comparisons use declared
  `atol=1e-4`, `rtol=1e-5`; decisions and seed IDs must agree exactly.

The prospective primary diagnostic uses the full extracted CHECK denominator:
invalid projections count as infinity, with at least 20 valid tracks, 90%
coverage, median at most 2 and p90 at most 4 native pixels in every later view,
plus the fixed parallax/rotation refusal. The unchanged stricter persistent
CHECK comparison is secondary. No outlier removal, CHECK pose/scale fitting,
window replacement or best-window selection is permitted.

All 32 original seven-frame windows remain in the denominator. Detector and
DVLT lifetimes are separate, with CPU2/3GiB/8GiB-headroom/1800-second inner and
1830-second outer actor guards. Root owns the 14400-second parent deadline;
the completion record reports actual elapsed time and whether it was met.
Unrun or failed actors remain operational outcomes, not geometry negatives.

The native study completed all 32 mask actors and all 32 model actors in
1978.656 seconds, within its parent deadline. Every one of the 224 detector
frames and every model state/output packet passed the independent checks.
All 32 pure geometry reports were independently recomputed. Peak sampled
aggregate actor RSS was 339,275,776 bytes for masking and 2,145,640,448 bytes
for DVLT; numerical actors took 44.875–49.171 seconds including their gates.

| Native diagnostic | Passes / fixed32 |
|---|---:|
| Primary pose-only, full extracted CHECK | 0/32 |
| Secondary model-depth/pose, full extracted CHECK | 0/32 |
| Secondary pose-only, strict persistent CHECK | 2/32 |
| Secondary model-depth/pose, strict persistent CHECK | 0/32 |

The two persistent-support positives were windows 7 and 20, with respectively
76/169 and 73/127 original CHECK tracks eligible. They fail the primary 90%
full-CHECK coverage gate. The controller records 30 `check_unavailable` and
two `complete_local_diagnostic` outcomes, with no detector, model, resource,
input, actor or deadline-unrun failure. Do not turn those two conditional
results into primary successes, select them as a subset or compare them
causally with the differently masked/seeded classical pilot.

Root's final audit closed all 130 observed actor births and rechecked all
1,956 unique current numerical consumed-file hashes. Original NoNN authority
and prior negative studies remain unchanged. This pilot establishes no metric
scale, measured native calibration, shared cross-window gauge, calibrated
covariance, route accuracy or policy adoption. The validated fleet remains
14/15; TEST4 remains six blind additions.
Study: `E:/monocular-osm-dvlt-native-proof/run1/study_contract.json`, SHA256
`7d3bd4170458d24ed3a0ed1f14b7f4a4e77e135fce9be58d099438706aa290d9`.
Completion: `E:/monocular-osm-dvlt-native-proof/run1/study_completion.json`,
SHA256 `5ed5eddf6f81dc256c3386a6aae7409a959ee323d48240759e096dd4557a9a2f`.
Independent final audit:
`E:/monocular-osm-dvlt-native-proof/run1/independent_final_audit.json`, SHA256
`09dd4a611226fc3e2d858cf0c99d79e677ea653590491a53fa3c471281f5f43f`.

Integration is still separate work. `Trajectory` and
`graph_filter.odometry_from_trajectory` currently couple heading validity to
usable translation. The staged matcher shortcut in `stages/trajectory.py`
keeps XZ only and uniformly remaps time when row counts differ. It cannot
preserve these native orientation/observability/clock contracts. A future
backend needs separate rotation/translation validity, a new cache identity,
exact native timing and independently checked connected gauges; disjoint
unit-normalized windows cannot be concatenated into a continuous trajectory.

## Pair-VPR CPU feasibility

A bounded fresh source scan adds Pi3X and Pair-VPR to the research notes.
Root froze clean Pair-VPR `fd764f9` and the prospective DINOv2 dependency
`4d83ffd`, matching all 18 reviewed source snapshots to official Git objects.
The publisher's historical unpinned DINO revision remains unknown. The runtime
source applies the reviewed CPU device/fallback fixes and a separate localHub
wrapper: both original DINO constructors use the pinned local source with
`pretrained=False`, followed by complete strict Pair checkpoint loading. The
architecture and published attention math are retained.

The publisher's 810,860,658-byte checkpoint matches SHA256
`18e7b95ba57d4d578fbf0a06a1846fc2dfb976c74da56bfd7d6a68afbcea8423`.
Static raw-tensor inventory V3 and the genuine CPU SDK independently matched
all 480 F32 tensors: 202,670,413 stored elements, including 201,857,101
parameter elements and two fixed positional buffers. V1's insufficient
post-unpickle validation and V2's refusal of a legitimate nested metadata key
remain preserved; neither historical result was upgraded. V3 passed 33
private tests plus 16 subtests; the parameterized CPU guard passed 17 checks.

A separately registered generated-image profile now completes the genuine
whole Pair-VPR model in isolated Python 3.11.18 with Torch 2.5.0+cpu,
TorchVision 0.20.0+cpu and 18 frozen distributions. The project and DVLT
environments are unchanged. All 21,470 site-package files, including pyc, are
bound. Source/private runner tests and independent peers passed all 36 cases
after fixing live-input identity checks.

The profile uses two fixed 322-square RGB patterns, full public preprocessing,
512D globals, 529x768 dense features and both batch1 pair directions, repeated
twice. All eight forwards completed in 40.453 seconds guarded total, with
2,290,393,088 bytes peak sampled aggregate RSS under CPU16/3GiB/8GiB-headroom
limits. Observed global forwards took 0.109–0.171 seconds; directional pair
forwards took 0.171–0.188 seconds. Root independently checked all 20 arrays
(12,107,504 bytes), exact generated/preprocessed pixels, F32 score sums,
bit-exact repeats, and complete raw/prepared/loaded and post-forward state.
The public positional preparation was actually byte-identical at this fixed
resolution. State, buffers, plain attributes and recorded live tensors stayed
unchanged; all observed actor births closed.

This establishes that generated-input profile only. It is not an empirical
ranking, native/gallery accuracy, calibrated probability or full-gallery
CPU16/four-hour result. No policy or selected Answer changed. The next test
must encode fresh actual query/reference pixels, retain every fixed Top100
pair in both directions and preserve physical IDs, coordinates and exact
native timing; incompatible old descriptors cannot stand in for new inputs.
Acceptance: `E:/monocular-osm-pair-vpr-feasibility/run1/synthetic322_root_acceptance.json`,
SHA256 `4719f7edca4b06f728863f92d07b43f3e96ecd6664d1f67d92bda997db1cd01c`.
See [research notes](recent_methods_2026-10.md) for public sources and limits.

## Pair-VPR matched reference data

The separate November 12 input gate now accepts the complete current reference
pixel cohort as **data only**. All 37 explicitly declared cache archives match
their protected encoded hashes (16,623,818,342 bytes). Their 21,136 stored
512-square BGR rows were streamed through member EOF/CRC and hashed individually.
All 42 physical calls and 28,330 reference occurrences join exactly by provider
ID and F64 latitude/longitude, with all 80 query times bound to the prior native
replay receipts. The separate native-query export is described below.

The baseline bank has 2,606 current images; the expanded bank has 13,164. Their
union contains 13,975 images. Every provider has one actual current pixel version
in each bank. The expanded bank's 16,072 old descriptor/score associations remain
in the alias ledger; they do not invent additional current images. Fresh image
identity is provider ID plus the actual typed cached-pixel SHA. Exact aliases
fold, and weights are recomputed as uniform 1/K over actual fresh pixel versions.
This does not reconstruct which historical pixels created the old descriptors,
or support a model-only causal comparison. Original JPEG bytes are unavailable.

V1 stopped on a legitimate cache-reuse metadata format and remains a closed
failed run. Separate V2 handles explicit reuse receipts and directly frozen
current caches, without fabricating a historical seal. Exact per-bank membership
and the complete 42-call union are checked before pixel bodies. Its independent
source review passed 17 private cases. The actual V2 gate completed in 27.531
seconds at 141,729,792 bytes peak sampled aggregate RSS. Independent Root QA
then rehashed all 21,136 rows and rejoined every original association in 31.719
seconds at 202,436,608 bytes. Both workers and their registered inputs closed.
Acceptance: `E:/monocular-osm-pair-vpr-native-proof/run1/reference_cohort_root_acceptance.json`,
SHA256 `74208b46f850d3a41a07c70542e6f5549db45f8da6eb259fdbb0ded1cce04d20`.

The 13,975 lossless reference NPY packets are now exported and independently
verified: 10,992,176,000 bytes. They contain 13,973 distinct typed pixel bodies;
identical pixels under different provider IDs still retain separate gallery
rows. Export took 112.515 seconds; independent packet QA took 15.843 seconds.
Acceptance: `E:/monocular-osm-pair-vpr-native-proof/run1/reference_packets_root_acceptance.json`,
SHA256 `3de2a1f5e97d751d04be35d26cc5b56e033f3131195ac3e87abf9e53ed67cacf`.

All 80 original native query frames are also exported, with 221,184,000 full
BGR pixel bytes and separate original-local, dense and native-ordinal bindings.
The producer executes the unchanged historical `9901` native-grid AST through
a capture wrapper that requires positive timestamps after every retained read.
It preserves the exact original query indices, including dense-index cases,
and exports three clock arrays. Its 27 private tests and independent peer pass.
The actual CPU2/512MiB worker took 57.813 seconds at 146,485,248 bytes peak
sampled aggregate RSS. Independent Root QA took 1.063 seconds at 49,098,752
bytes, matching both index-tagged pixel aggregates, all clock headers and all
42 original call bindings. Both actors and their registered inputs closed.
Acceptance: `E:/monocular-osm-pair-vpr-native-proof/run1/native_query_packets_root_acceptance.json`,
SHA256 `66fe7961639d7b79cf9be8e248d87ea2f0c6b46a6b6a6ef2b47cda534e66a4c3`.

The combined raw carrier V2 is now Root-accepted as **DATA only**, after exact
named-bank/native membership and original 80-row query-clock joins. V1 remains
unadmitted, with its pending output preserved; its missing source checks do not
establish an actual dataset mismatch.
Acceptance: `E:/monocular-osm-pair-vpr-native-proof/run1/carrier_v2_root_acceptance.json`,
SHA256 `85ccea8d9bf029a06f38f24fe678192dd801447b3f5ceb7d387d011fe06ab59e`.

The separate geometry DATA V2 completed in 1.687 seconds at 187,224,064 bytes
peak sampled aggregate RSS. Its first independent QA stopped on a
leading-start-zero assertion and remains a preserved failed attempt. Separate
QA V2 completed in 0.516 seconds at 65,695,744 bytes, and the geometry is now
accepted for input transport, without a route or accuracy claim.
Acceptance: `E:/monocular-osm-pair-vpr-native-proof/run1/geometry_root_acceptance.json`,
SHA256 `fd084ea48fe6875c9d96e4a4c70d7042cb7689903bd5f92759df55002959cc9e`.

Before any real Pair forward, the fixed `595bd674` view profile blacks the top
and bottom quarter of **every** reference and query canvas. It preserves all
13,975 reference keys and 80 query observations, with 14,053 distinct view
bodies totaling 11,211,797,120 file bytes. This is a uniform band operation,
with no detector, text-recall or all-text-exclusion claim. The producer closed
in 151.078 seconds at 158,224,384 bytes peak sampled aggregate RSS; independent
DATA QA closed in 114.391 seconds at 213,868,544 bytes.

Final metadata projection V2 adds only the already closed video path, size and
SHA to that view packet; every pixel, alias and clock field stays unchanged.
Projection closed in 0.781 seconds at 157,339,648 bytes, and independent metadata
QA closed in 1.125 seconds at 185,417,728 bytes. All four view/projection actors
and their registered inputs closed. The final input is **admitted** through
separate Root view-DATA and model-input certificates. Raw and view artifacts
retain their explicit authority denials; the projection does not self-grant
model or policy authority.
Final packet SHA256:
`cad49be3c187a6795d409b1ab3ee4060e310ab654e68db5feb5d607c061cf097`.
View DATA acceptance: `E:/monocular-osm-pair-vpr-native-proof/run1/masked_views_root_acceptance.json`,
SHA256 `2fbe0b288f73acd364540829eaf13661d17f22d82bdba637cad370352f554be0`.
Model-input acceptance: `E:/monocular-osm-pair-vpr-native-proof/run1/model_input_root_acceptance.json`,
SHA256 `4223a7dd60be39daadc6f96e83395bb8614c31d1049c452323f5088946ade63f`.

The held feature worker has 32 private cases plus two independent peer probes;
the independent saved-output feature QA passed 52 cases. The common-parent
controller V1 remains unadmitted:
private probes found PID reuse between check and kill, a lost zero-exit road
resource failure, and feature acceptance written before final input checks.
Its separate V2 repair (`ad750`) passed 26 independent private cases; actual
common-parent runtime remains a separate gate.

One separate source/private Pair-to-road proposal has passed independent review.
Full-gallery global distances select the fixed top100 only. Both ordered learned
logits define an explicitly uncalibrated compatibility energy; the geographic
factor is `log(.1 + .9*N/D)`, with a full-gallery density denominator, uniform
fresh-version weights, and the original temporal power applied once. The new
direct-log/future-guide adapter preserves the sampler's numerical tail. These
are output-only prototypes, with no production policy change. The prospective
search uses blocked F64 squared L2 on F32 globals with deterministic image-key
ties; it does not claim numerical identity to FAISS F32 search.

The first actual fixed two-bank method ran under the common
CPU16/3 GiB/14,400-second watcher. It freezes 36,094 current input files and
305 current project Python sources; all runtime templates were filled before
launch. Registration: `E:/monocular-osm-pair-vpr-native-proof/run1/real_method_registration.json`,
SHA256 `5fbb0c4cdd57d15e2c462641b9f6ffc14ce66e56955dcfe97841dbbb0027c64b`.
It closed **failed**, after 14,053 genuine global forwards and before any Pair
forward. `RealBackend.__init__` sets `self.pair` to a source `Path`, shadowing the
matched subclass's `pair(...)` method. All saved features and the failed result
are preserved; final model-state and input/output integrity checks passed, but
there is no complete feature acceptance, road outcome or GT score. The outer
actor took 2,765.390 seconds at 2,430,246,912 bytes peak sampled aggregate RSS.
Closure: `E:/monocular-osm-pair-vpr-native-proof/run1/real_method_external_closed.json`,
SHA256 `19881e73901c526470a3fb13266e9d7880bebb78cedf6a20388191ccf8824057`.

A separate source V2 renames only the matched transport method and its caller
to `ordered_pair`, preserving the literal original model backend. Its new
regression uses the genuine constructor layout with a Path-valued `pair`;
33 private cases pass independently. The dependent QA, road, controller and
scaffold copies change only eight source-SHA literals; 91 focused author checks
and an independent exact-source-delta review pass.

The fresh full rerun under `E:/monocular-osm-pair-vpr-native-proof/run2` is closed.
Its 36,097 frozen inputs include the unchanged accepted DATA, model and 305
project Python sources. Registration SHA256:
`e46409f428ba2eaf8a48b8a1c335d2db39f3668f73a52a109f17683c65108d62`.
It completed all 46,053 genuine forwards (14,053 global and 32,000 ordered Pair),
all 16,000 comparison receipts, final full-state and input/output checks, and
independent output QA. The controller issued the feature DATA acceptance after
closing NN and QA subtrees. Road filtering then stopped on the external watcher's
CPU-affinity subset assertion before either route completed. The whole attempt
remains **FAILED and unadmitted**: 10,119.750 seconds, 2,430,930,944 bytes peak
sampled aggregate RSS, all eleven logged process births closed, no GT read or
scoring. Closure SHA256:
`6e5dced58a781b8787ac6a650aa0b184aec5a8a569a6f72380b050442e207cda`.

Private source tests reproduce a possible PID-reuse race inside psutil child
enumeration that can admit an unrelated process and falsely fail affinity.
The actual offending identity was not saved, so this is not a proven cause of
run2. Generated native matrix and two-bank 50k PF probes preserve CPU0..15.
A distinct watcher (`506b5991`) now validates birth identities across direct-child
admission, records violating identities/masks, and protects cleanup against log
failures. Its 33 private cases pass; a genuine nested-process CPU16 smoke closes
all five observed births. The final normal path matches that smoke, with final
exception changes covered by fault tests. Limits remain unchanged; masks are
never silently healed. This does not establish the actual run2 failure cause.

A separate road-only engineering replay is active at
`E:/monocular-osm-pair-vpr-native-proof/road_replay1`, registration SHA256
`804c9cf5e9d9cd336a8f535cf03092b6456f3a13594c7ce720bffa6d716fd2f7`.
It keeps the original control/study/feature subjects and ORIGINAL absolute
lease unchanged, changes only the worker output directory, and performs no new
NN. It must not renew that lease or claim successful same-parent/end-to-end/
fleet execution. The failed full attempt is never promoted. Follow
`latest_checkpoint_20261112_pair_vpr_road_engineering_replay1` in
`output/goal_progress.json`; both old full-method actors have closed.
There is still no new route, selected Answer or fleet accuracy result.
The validated original fleet remains 14/15, with the
retained Detroit error 1,387.2377579105248 m; all six TEST4 additions remain blind.

A bounded November 12 paper/repository refresh added three source-only leads;
it is not an exhaustive NeurIPS survey. No new CPU16/4h replacement was verified,
and none of these methods ran on project videos:

- [BevSplat, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/e5ba3d6d93213db6b1d1931c6517fe1a-Abstract-Conference.html)
  adds satellite-based absolute geometry, but needs new imagery/calibration and
  initialization inputs; its published pipeline uses CUDA rasterizers.
- [VGGT-SLAM 1.0](https://github.com/MIT-SPARK/VGGT-SLAM/tree/1c548f7dd72e9ae418bbbd74e7f9c16a886405bc)
  adds global submap alignment and loop closure. Its reviewed CPU branch still
  invokes CUDA, and reconstruction alone does not georeference a road route.
  Do not substitute the later 2.0 system for the reviewed NeurIPS 2025 version.
- [SAFEVPR](https://arxiv.org/abs/2605.28048v2), an adjacent RA-L 2026 release,
  adds sequence-match verification. It requires coherent reference windows and
  independent labelled calibration; the present unordered gallery does not
  provide that contract, and full CPU extraction remains unmeasured.

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
