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

The native study is registered and masking has started; the complete native
results are pending. This pilot establishes no metric scale, measured native
calibration, shared cross-window gauge, calibrated covariance or localization
accuracy. The validated fleet remains 14/15; TEST4 remains six blind additions.
Study: `E:/monocular-osm-dvlt-native-proof/run1/study_contract.json`, SHA256
`7d3bd4170458d24ed3a0ed1f14b7f4a4e77e135fce9be58d099438706aa290d9`.

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
