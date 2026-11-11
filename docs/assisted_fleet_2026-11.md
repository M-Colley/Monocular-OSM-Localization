# Assisted policy v3: the complete fifteen-clip result (November 2026)

The assisted policy (`--assisted-only --assisted-profile=v3`; stages
`baseline → cues → poi_joint | metric_pf | expanded`, see
`docs/experiments_2026-10.md` section 11) ran on all fifteen dev clips at
commit `a8add7a` (run `assisted_v3_fleet_fixed`, 30-31 October, one rep,
CPU retrieval, 7,200 s budget per clip). All fifteen finished with rc 0. It
had not been scored. This note scores it.

## Scoring

The run's own aggregator refuses it ("reps ran different effective method
configurations"): it fingerprinted the absolute runtime deadline, which
differs per clip by construction. That is the bug `32c2079` fixes, and
`docs/recent_methods_2026-10.md` names the reviewed comparison adapter for
this run: branch `codex/deadline-config-fingerprint-20261030` at `5bdae79`
(`a8add7a` plus that fix only). Scoring used `5bdae79`'s aggregator and
`scripts/check_dev_goal.py`, on a byte copy of the run directory
(`E:/claude-scoring/a8add7a/`), so the original run directory is unchanged.

* Summary `summary.json` SHA256 `4bcf6b5b16b466cb41facacf7ebc42c61d773f4fc9cd93e5e9928e01325bc67b`;
  aggregator `5bdae79` (clean); no warnings.
* `check_dev_goal.py` (5bdae79): every clip passes the frozen-input audit;
  the only failures are three errors above 200 m. Log SHA256
  `292827af456a32c15c3213316cde5107d6ace11c9b259d7176d15d307a5229ae`.
* The current `HEAD` audit (`efd7c04`) rejects all fifteen because it
  requires `monocular_osm/retrieval_catalog.py`, which was added after this
  run (`1c889f3`); that is a version mismatch, not a finding about the run.

## Result

Symmetric mean route error (m) of the selected answer.

| Clip | v6 track (9/15) | **Assisted v3** | Stages (s) | Selected |
|---|---|---|---|---|
| Ulm 4K | 15 | **16** | baseline 1,354 | baseline |
| Ulm #2 | 46 | **43** | baseline 2,401 | baseline |
| London | 62 | **62** | baseline 2,377 | baseline |
| Berlin | 135 | **135** | baseline 2,626 | baseline |
| KITTI 0009 | 3,735 | **13** | baseline 2,348, cues 43, poi_joint 2,183 | poi_joint |
| KITTI 0033 | 7,004 | **5** | baseline 2,385, cues 276, metric_pf 117 | metric_pf |
| Chicago `vCSEG6KaFng` | 169 | **169** | baseline 2,332 | baseline |
| Chicago `LmIHvLMLqFk` | 133 | **133** | baseline 2,329, cues 569, expanded *timed out* | baseline |
| Phoenix `wJsEQTCAg1c` | 18,644 | **28** | baseline 1,351, cues 285, expanded 256 | expanded |
| Detroit `ZhGb8q1kliY` | 55 | **55** | baseline 2,284 | baseline |
| Detroit `1nF_7l07i-E` | 77 | **77** | baseline 2,392 | baseline |
| Detroit `g5lnpYCk1Ec` | 1,573 | 1,387 | baseline 2,508, cues 471, expanded *timed out* | baseline |
| Atlanta `y66ZkRpUh4k` | 6,590 | 6,590 | baseline 2,144, cues 342, expanded *timed out* | baseline |
| Atlanta `kxEDNj5L_yQ` | 6,024 | 6,024 | baseline 2,057, cues 517, expanded *timed out* | baseline |
| Newcastle `Wrn4_uCxRCQ` | 35 | **35** | baseline 1,974, cues 525, expanded *timed out* | baseline |
| **Within 200 m** | 9 / 15 | **12 / 15** | | |
| **Median** | 135 | **62** | | |

It beats the city-centre guess on 14 of 15 clips (all but Atlanta at night).

**This is in-sample.** The policy's stages and thresholds were developed on
these fifteen clips (disclosed in section 11), so 12/15 measures the system
on its development data, not generalization; the held-out split `test1` was
scored for the earlier frozen method and its results were public before the
policy was built, so it is no longer a clean test for the policy either.

**All three failures are runs whose `expanded` stage hit the deadline.** On
Atlanta `kxEDNj5L_yQ` it was listing six 3 km regions at up to 1,500
references each, two reused from the baseline, i.e. up to ~6,000 new images
to download and embed on CPU (~0.8 s each) in the 4,494 s left after the
baseline and cues; the policy then kept the baseline, by design. Five clips
timed out there in total; on two (Chicago `LmIHvLMLqFk`, Newcastle) the kept
baseline was already within 200 m. The same stage completed on Phoenix (256 s,
references reused) and fixed it, and an earlier single-clip run of the policy
on Atlanta `kxEDNj5L_yQ` (`300a82f`) selected `expanded` at 32.76 m. Whether
the three failures are compute-limited is measured next: the five timed-out
clips rerun at `a8add7a` with a 14,400 s budget and nothing else changed
(run `budget14400`, from copies of the run's prepared inputs, which now hold
the first run's caches, so its runtimes are warm).

## The 14,400 s rerun (`budget14400`)

The five clips whose `expanded` stage timed out reran at `a8add7a` with a
14,400 s budget and nothing else changed (scored by `5bdae79`, no warnings).
Their inputs were copies of the first run's, including its caches, so stages
that reuse references ran warm (baselines 77-1,902 s instead of ~2,000 s).

| Clip | 7,200 s | 14,400 s | `expanded` at 14,400 s | Selected |
|---|---|---|---|---|
| Atlanta `kxEDNj5L_yQ` | 6,024 | **33** | complete, 998 s (metadata regions) | expanded |
| Atlanta `y66ZkRpUh4k` (night) | 6,590 | 6,590 | complete, 4,122 s | baseline |
| Newcastle `Wrn4_uCxRCQ` | 35 | 35 | complete, 2,885 s | baseline |
| Chicago `LmIHvLMLqFk` | 133 | 133 | timed out after 13,552 s (city grid) | baseline |
| Detroit `g5lnpYCk1Ec` | 1,387 | 1,387 | timed out after 13,220 s (city grid) | baseline |

With the longer budget the assisted policy is **within 200 m on 13 of the 15
dev clips** (in-sample, with the warm-cache caveat). The two that remain are a
night drive whose completed expanded search was rightly not selected, and a
drive mostly outside the city disc whose fallback search, a grid of 27
municipal cells at 250 then 1,500 references each, does not finish in four
hours of CPU (Chicago's 47-cell grid neither). That fallback is the policy's
compute ceiling on a CPU: thousands of reference images to download and embed
per stage.

By the pre-registered `budget_rule` (one of kxED, y66, g5l crosses to within
200 m and neither LmIH nor Newcastle leaves it), the held-out assisted runs,
the test1 replay and test2, use 14,400 s.

## Reference-mean centring of the retrieval track (offline replay, not adopted)

Codex's night-clip study found that subtracting each reference's mean
similarity over the query frames before the exact sequence decode
("reference-mean", a hubness correction) helps on a local pool. The baseline
children of `assisted_v3_fleet_fixed` saved both retrieval passes in full
(`retrieval_audit/*.npz`: the 80 x ~1,500 similarity matrix, reference
positions, query times, decoded path), so `scripts/replay_track_centering.py`
replays the pipeline's own `_viterbi_decode` with and without centring on all
fifteen dev clips, without new embedding (the replay reproduces the saved path
on most passes, and the official error where it does: Berlin 135.1, Ulm #2
42.7, London 62.5 m).

* **Pass 2 (same reference pool), track route error:** better on Berlin
  135 -> 94, Detroit `1nF` 77 -> 57, `ZhGb` 55 -> 31, Ulm #2 43 -> 25 m;
  worse on London 62 -> 76, Chicago `LmIH` 133 -> 143, Newcastle 35 -> 37,
  Chicago `vCSE` 169 -> 190 m. No clip crosses 200 m either way.
* **Pass 1, distance from the robust centre to the route** (the area pass 2
  then searches): rescues some failures (Detroit `g5l` 206 -> 32 m, Atlanta
  `y66` 6.4 -> 1.9 km, KITTI 0009 3.7 -> 1.0 km) but breaks two working
  clips catastrophically (Detroit `ZhGb` 182 m -> 7.4 km, Chicago `vCSE`
  52 m -> 1.5 km).

Centring the city-wide first pass is therefore unsafe, and on the local
second pass it trades clip against clip without changing the 200 m count;
consistent with the policy's use of it only inside its `expanded` stage. Not
adopted. In-sample analysis; GT used for scoring only.

## Would a multi-area second pass help? (offline diagnostic)

`scripts/pass1_hypothesis_rank.py` pools each query's top-5 first-pass
references (the same saved matrices) into 2 km areas ranked by summed
similarity and finds the rank of the true area (an area centre within 1 km of
the route). Data: `docs/pass1_hypothesis_rank_2026-11.json`.

| True area rank | Clips |
|---|---|
| 1 | London, Ulm 4K, Ulm #2, Newcastle, Detroit `1nF` and `ZhGb`, Chicago `LmIH` |
| 2 | Berlin, Chicago `vCSE` (both already within 200 m), **Detroit `g5l`** |
| 8 / 12 | **Atlanta `y66` (night)**, **KITTI 0009 (2011 footage)** |
| not in the disc | KITTI 0033, Atlanta `kxED`, Phoenix (reached by the policy's wider searches) |

Running the second pass over the top two or three areas instead of one could
reach at most one remaining failure (Detroit `g5l`, whose route lies mostly
outside the city disc). The night and historical clips rank their true area
8th and 12th: what they need is retrieval that recognises those conditions,
not more hypotheses. Not implemented.

## ImAge on the city-wide first pass (pilot, not adopted)

`scripts/pilot_image_pass1.py` re-scores six clips' saved first-pass
reference sets with ImAge (NeurIPS 2025, the official GSV checkpoint Codex
verified; whole frame and best of three square crops) against MegaLoc's saved
matrix, through the same decoder and the area diagnostic above. Data:
`docs/image_pass1_pilot_2026-11.json`.

| Clip | MegaLoc centre to route (true-area rank) | ImAge whole | ImAge crops |
|---|---|---|---|
| London | 100 m (1) | 100 m (1) | 46 m (1) |
| Detroit `1nF` | 262 m (1) | 128 m (1) | 94 m (1) |
| Ulm #2 | 109 m (1) | 89 m (1) | 94 m (1) |
| Chicago `vCSE` | 52 m (2) | **2,227 m** (2) | **2,227 m** (2) |
| Atlanta `y66` (night) | 6.4 km (8) | 6.4 km (6) | 6.4 km (**2**) |
| KITTI 0009 (2011) | 3.7 km (12) | 6.1 km (4) | 9.0 km (10) |

ImAge sharpens the centre on three working clips but breaks Chicago `vCSE`,
and on the night clip it lifts the true area to second place without moving
the decoded centre (still 6.4 km off); KITTI is not helped. As a drop-in
first-pass encoder it is not an improvement. The night result is the one
lead: a second pass over the top two areas with ImAge crops could reach it,
which is two untested changes and only one clip. Not adopted.

## Policy v4 pilot: sign-site verification (`ee38432`)

Policy v4 is v3 plus one stage: when the cues read one to six literal
shop-sign branches but native VO is too weak for the joint POI attempt
(valid fraction below 0.80), each branch's site gets a dense local pool
(300 m, 600 images); the site whose imagery the sign-read frames match best
is kept, and the whole window is decoded over that pool plus a 3 km /
1,500-image route pool (`monocular_osm/poi_site_verify.py`). Built from the
Atlanta `y66` diagnosis above (the true site had 1 sampled image within
200 m, the false one 25). Pilot on the two remaining v3 failures, 16 CPU
threads (`--assisted-cpu-threads 16`), 14,400 s budget, from `wt-head`:

| Clip | v3 (`a8add7a`, 14,400 s) | v4 pilot | Path |
|---|---|---|---|
| Atlanta `y66ZkRpUh4k` (night) | 6,590 m | **43 m** | `poi_site` selected (2 branches, VO 0.49), 2,325 s |
| Detroit `g5lnpYCk1Ec` | 1,387 m | 1,387 m | baseline kept; expanded complete, not admitted, 7,429 s |

Atlanta is solved by the new stage as designed (in-sample: the stage was
built from this clip). Detroit is not, and the reasons are now known:

* At 16 threads the municipality grid finishes (27 coarse cells at 250
  images, 3 fine cells at 1,500), but its route is 2.5 km off. The drive's
  cells (9, 10, 13) got no frame at the coarse level, so they were never
  densified; only cell 5 (the first 150 s) was, and that part is right
  (78 m). Offline, other GT-free shortlist rules computed from the saved
  matrices reach the true cells only with raw-matrix maxima that the grid
  code does not have; with its per-frame decoded cosines, voting puts them
  4th-7th. No rule is adopted on one clip.
* The metadata regions (Roseville and the named corridor) lie 10-21 km from
  the recorded part of the drive, and the metadata union was refused anyway
  (one of 1,200 sampled images had no URL). Later runs no longer refuse it
  (commit 8ca1bcd; the v4 fleet at `409a959` predates the fix), but these
  regions do not cover the drive, so the union is not expected to change
  this clip.
* OCR failed on every run of this clip: the fixed-count sampler read the frame
  before the VO horizon (`int(118.99999...)`; fixed in `1498923`). Fixed,
  it reads generic shops (no literal branch) and the exit numbers 186B and
  188; the OSM junctions with those refs lie 159-234 m and 196 m from the
  true position at the read times. A prototype decode centred on them
  (dense pool plus 3 km route pool, image-only) reaches 909 m: the location
  is right, the freeway imagery cannot place the 8.5 km route. An exit-number
  cue would need VO or road-graph alignment, not image decoding; not built.
* **Two-anchor VO pin (offline check, 2026-11-09; not adopted).** The
  simplest such alignment does not work.
  - **Setup.** The v4 run's own OCR (409a959, stamps filtered) reads `186B` at
    t = 391.9 s and `188` at t = 469.5 s. OSM has three `motorway_junction`
    nodes with ref 186B and one with ref 188 within 30 km of the Detroit
    geocode. For each pair, a 2-D similarity transform maps the cached VO
    track (`trajectory_v2_4-599.0_s4_fauto_vo3.npz`) onto the two junctions at
    the read times, and the whole track is scored.
  - **Correct pair.** With the pair the car actually passed (159 m and 196 m
    from it at the read times), the route is 1,923 m off.
  - **Selection rule.** The ground-truth-free rule (the pinned track's median
    distance to OSM major roads near the reads) picks the wrong 186B (963 m
    from the car); that pair scores 1,756 m.
  - **Why.** Two anchors 1-2 km and 77 s apart fix scale and rotation too
    loosely: a few degrees of rotation over a 5 km lever arm, plus VO drift
    across ten minutes, outweigh the anchors' 200 m accuracy.
  - **What would be needed.** A usable exit cue would need full road-graph
    map matching of the VO between anchors. That would be developed on this
    one clip and is not built.
  - Ground truth was used only to score.

## Policy v4: the complete fifteen-clip run

`assisted_v4_fleet`: clean worktree at `409a959`, `--assisted-only
--assisted-profile=v4 --assisted-cpu-threads=16`, 14,400 s budget, one rep,
warm copies of the v3 fleet inputs (`E:/claude-scoring/v4fleet/data`), every
clip `rc=0`. Scored offline by the HEAD aggregator; the strict artifact audit
(`scripts/check_dev_goal.py`) passes every clip's evidence, and its only
failing reason is Detroit's 200 m outcome.

| Clip | Answer, symmetric (m) | Selected stage | Wall (s) |
|---|---|---|---|
| Ulm 4K (`ULl8s4qydrk`), 0-300 s | 16 | baseline | 772 |
| Ulm #2 (`LC9Sa--u5KE`) | 43 | baseline | 89 |
| Berlin (`lBlKR2ek0w4`), 0-420 s | 135 | baseline | 209 |
| London (`T4wTL3LpLqU`) | 62 | baseline | 89 |
| KITTI 2011_09_26 drive 0009 | 13 | poi_joint | 3,164 |
| KITTI 2011_09_30 drive 0033 | 5 | metric_pf | 1,835 |
| Chicago (`vCSEG6KaFng`) | 169 | baseline | 170 |
| Chicago (`LmIHvLMLqFk`) | 133 | baseline | 6,599 |
| Phoenix (`wJsEQTCAg1c`) | 28 | expanded | 576 |
| Detroit (`ZhGb8q1kliY`) | 55 | baseline | 128 |
| Detroit (`1nF_7l07i-E`) | 78 | baseline | 162 |
| Detroit (`g5lnpYCk1Ec`) | **1,387** | baseline | 7,823 |
| Atlanta (`y66ZkRpUh4k`, night) | 43 | poi_site | 2,840 |
| Atlanta (`kxEDNj5L_yQ`) | 33 | expanded | 1,516 |
| Newcastle (`Wrn4_uCxRCQ`) | 35 | baseline | 14,276 |

**14 of 15 within 200 m** (93 %, Wilson 95 % 70-99 %), median 43 m, against
13/15 for v3 at the same budget: the night clip Atlanta `y66` is the one
change, as in the pilot. Detroit `g5lnpYCk1Ec` is the remaining failure, for
the reasons above. This is in-sample: v4's new stage was built from `y66`,
and the policy's thresholds from these clips. v4 is untested on held-out
data; on test2, v3 matched the plain retrieval track (2/7,
`docs/heldout_test2_2026-11.md`).

Newcastle ran while other CPU-heavy work shared the machine (from about
11:45 to 12:40 UTC this session's YouTube screen for test3 and two
ground-truth builds, and other projects' simulations throughout), and used
14,276 s of its 14,400 s budget (2,885 s in the v3 rerun). It kept its baseline (35 m), the same
answer as v3, so the contention did not change its outcome; on a budget-bound
clip it could have.

## Raw-evidence fine-cell pilot (10 November 2026; not adopted)

The opt-in `--vpr-city-cell-selector=raw-votes` pilot at clean `62868aa`
reused warm v4 inputs, 16 CPU threads and the 14,400 s limit. Each query
voted for its top three coarse-cell raw maxima; the same three-cell fine
budget, decoder and final admission rule remained in place. The prospective
rule is `raw_city_votes_dev_2026_11`.

All 27 coarse matrices qualified and all three fine retrievals completed.
Votes selected cells 1, 9 and 13, so two previously starved areas were
densified. The expanded route's median cosine was 0.168222 against the
required 0.177707. The policy kept its baseline.

| Frozen output | Symmetric route error (m) | Role |
|---|---|---|
| Actual selected Answer | 1,387.24 | Failed the 200 m target |
| Expanded child | 1,714.67 | Offline component diagnostic; not selected |

The run completed with rc 0 in 2,514.2 s and consistent provenance. The
strict pilot audit found no input-evidence failure; Detroit failed the error
threshold. Its fourteen missing siblings are expected for a one-clip pilot,
so this is not a complete fleet result. The component diagnostic used the
already frozen child route without changing it or choosing a route using
ground truth. Rejection protected the selected result from a worse route.
No admission threshold was relaxed, no full-fleet trial was started, and the
default remains `decoded`.

Run: `E:/monocular-osm-runs/raw-city-votes-20261110/output/benchmark_runs/raw_city_votes_v4_g5_20261110`.
Summary SHA256: `b44ff5b775a621e1e1784373b1983edc3ffaf0eb157f2fc93fa446cf0841259a`.
The unrounded component diagnostic is
`E:/monocular-osm-raw-city-votes-proof/run1/component_diagnostic.json`, SHA256
`bc71f29cf01c40a245ee5fc3fdc3a7bf3c7d69f9f8eeeaad2df92f3d9fdbd847`.

The best verified full development result remains v4's **14/15**. A bounded
road-graph sequence prototype retaining all OCR junction hypotheses is the
next investigation; it has no measured accuracy or policy admission yet.

## Literal road-sign input check (10 November 2026)

The cached native OCR did not contain literal EXIT, so its bare `186B` and
`188` could not satisfy the new road-sign parser. At clean `e292224`, a
pre-registered crop probe read all four >=0.90 numeric seeds with fixed
padding. All four reads completed; none recovered EXIT. The result at
`E:/monocular-osm-exit-context-proof/run1/context_result.json` has SHA256
`1473c783c6839cac435bb272879826a6a1031b6306f712ccff8bb83ff55b71f9`.

A separate prospective temporal probe read full native frames at fixed
offsets +/-0.5, 1, 2 and 4 seconds around every original eligible seed.
Rounding and window clipping were fixed before OCR; all 29 unique planned
frames completed. The same cached RapidOCR weights and CPU 2/1 runtime
recovered these literal EXIT observations, without using labels:

| Reference | Actual native time (s) | Evidence |
|---|---|---|
| 186A | 364.03831982313926 | Inline `EXIT 186A` |
| 186B | 392.40972733971995 | EXIT and reference in nearby native boxes |
| 186B | 392.9121725731895 | Second view of the same event |
| 190A | 594.8281637301534 | EXIT and reference in nearby native boxes |

The two 186B reads must not count as independent sign events. Bare 188
remains unverified. All detections keep their actual frame times; an EXIT
from another frame never certifies an original bare number. Coordinate
stamps were filtered, and no route was inferred or scored. This establishes
usable input evidence, not a localization gain.

The temporal probe completed in 49.562 s, with 404,078,592 bytes peak sampled
aggregate RSS and both physical/commit headroom above 8 GiB. Registration
SHA256 is `cc28dbd2e35fc08ae4499db71c3dae85c03ce9fd2ea76f0bfa9de2f20b46bd9a`;
result `E:/monocular-osm-exit-temporal-proof/run1/temporal_result.json` has
SHA256 `d86782eb13974c9874838fbd83a658f9d4f574352af971eb68e412be55e4a608`.
The strict full-city map and road sequence experiment remain separate work.

## Full road-sequence diagnostic (10 November 2026; negative, not adopted)

At clean `7ddea55`, the fixed correlation-v2 sampler completed the entire
native Detroit window: 4,440 increments composed into 1,110 steps, 50,000
particles and seed 0. It retained all four literal observations (the two
186B views each have log-factor power 0.5) and every matching junction:
two 186A, three 186B and two 190A hypotheses. The complete named-city graph
retained 93,004 directed edges and all 330,592 polyline segments. Exact
matched node IDs, directed incidence and public coordinates were verified.

Separate guarded preparation completed in 21.547 s, peaking at
2,584,084,480 sampled aggregate RSS bytes. The numerical worker completed
in 776.141 s, peaking at 1,149,431,808 bytes. CPU affinity was two logical
CPUs; both physical and actual commit headroom stayed above 8 GiB. Every
registered input/source hash remained unchanged. Every observed worker
birth closed before prediction hashing and the single offline GT score.

| Frozen diagnostic | Unrounded error (m) |
|---|---:|
| Symmetric mean | 6,323.285942994211 |
| Start | 7,420.339254033987 |
| End | 7,394.620411573654 |

This is worse than v4's retained 1,387.2377579105248 m Answer. No selected
Answer, threshold or policy was changed, and no full-fleet gain follows.
The best verified full development result remains **14/15**.

The unchanged output also exposes two limitations. The first sign arrived
360.052 s into the window; at that point ESS/50,000 was about 0.0002.
ESS does not measure historical ancestor diversity, which was not exported.
Unknown motion covered 81.865 s (13.761%); 122 purely unknown composed bins
accumulated 1,410.144 m of model-driven endpoint chord displacement. That is
neither measured speed nor evidence that the vehicle actually moved there.

The current flat, unnormalized transition potentials assign weight one to
each legal branch. Their total outgoing mass grows with junction degree,
which can overwhelm a few soft signs. This is a source-level modeling
limitation, not proof of the cause of this particular failure. The next
separately registered comparison uses conditional-uniform legal turns with
total outgoing mass one, preserving all geometry, cues, motion assumptions,
proposal corrections and resource bounds. It has no measured result yet.

Proof directory: `E:/monocular-osm-junction-sequence-proof/run1`.
Prediction closure SHA256:
`201f96b8c3e5d697727168f35b1b670271640cf3a35c2a395e9276539bc4f4e4`.
`component_score.json` SHA256:
`ba9a65ba9f9a3b450d385bc6c390db7705182ba9ebb0af58ef0cdac9f99e6bd8`.
The report-only `closed_diagnostic.json` has SHA256
`8d373c766df75ded94a3ed948a27dd3ef4421d8da1d9675d64fd113d233c572c`.

## Conditional-uniform graph prior (10 November 2026; also negative)

Clean `72437d9` added an explicit opt-in `conditional-uniform-v3` graph
prior. This is a graph prototype version, separate from assisted v3.
Genuine transition probabilities sum to one across legal successors, with
all original proposal corrections retained. The default stays `legacy`.
Both profiles now record the complete effective filter configuration and
state that the base speed range is not a hard speed cap.

Map preparation was repeated truthfully under the new source. Root verified
every typed map-array hash, topology metadata and complete sign/candidate
record equal to the preceding trial before inference. Geometry, signs,
motion, correlation powers, seed, particle count and resource limits were
held fixed. Preparation closed before the numerical actor began.

The full-window trial completed in **64.156 s**, with **1,177,436,160 bytes**
peak sampled aggregate RSS. Every observed numerical worker closed and all
registered source/input bytes were verified before the one offline score.
The unmodified representative lineage scored **18,281.577297323704 m**
symmetric mean error (start **22,331.86457438476 m**, end
**16,705.675519077064 m**). It is worse than both the previous graph trial
and the retained v4 Answer. It is not adopted and triggers no full-fleet
trial or admission relaxation. Development remains **14/15**.

Proof: `E:/monocular-osm-junction-uniform-proof/run1`.
Prepared equivalence SHA256:
`f3a3f86b82cb81cccc67551e60e2525b58d2477ee832b346899b7a47ecd53342`.
Prediction closure SHA256:
`c0ee393a6cfa3af9ebdf80580515ad429cabcca5a9e1880d605e0316a858fc14`.
`component_score.json` SHA256:
`c8ac4e0ad55f6b4944810589ff16e07c2073864d85333c83b69c285fe148352f`.

These finite-particle negatives do not prove that road-constrained
localization is impossible. They do show that these road/sign-only profiles
do not solve this case. The next source investigation combines earlier
audited visual evidence with these constraints, rather than further tuning
the same three late sign events. It has no implemented combined likelihood
or actual combined-method result yet.

## Native visual-input replay (11 November 2026)

The next investigation keeps earlier raw visual alternatives in a spatial
mixture alongside native VO, roads and literal EXIT evidence. Before loading
reference positions or similarity values, Root froze all numeric retrieval
audit pairs: two baseline records and 40 expanded/control records, including
five physical aliases. No record was dropped to obtain a compatible subset.
The public reference payload inventory also froze 148 candidate files in 37
directories by opaque file hash; this did not verify historical encoding.

Source `0695745` adds a complete-bank input replay and pure streaming bank
validator. Root's 65 focused tests and independent source/private reviews
passed. The first actual replay refused a relative historical output path
before numerical-member loading; that failure is preserved. The corrected
helper resolves relative paths against the explicitly pinned original
execution checkout without rewriting saved metadata.

The corrected guarded replay closed in **55.110 s**, peaking at
**147,611,648 bytes** sampled aggregate RSS. All 42 physical calls passed the
original native80 pixel/clock check. Every query time exactly matches the
signed VO clock; applying factors at the next composed motion boundary adds
at most **0.401956186775692 s** of declared lag. All observed worker births
closed and protected input/source hashes remained unchanged before Root
accepted this input-only result. No NN, descriptor recomputation, route
inference, GT or policy selection ran.

Native facts: `E:/monocular-osm-joint-vpr-proof/run2/native_result/native_replay.json`,
SHA256 `67476f9171724727dbef7531c8bb8869ccc0da82d1f7dd1999566d73251207a9`.
Root acceptance: `E:/monocular-osm-joint-vpr-proof/run2/native_root_acceptance.json`,
SHA256 `407e1495c0c4e729d9d5d6c7c842012dcf8c22c8a382232b2e7950f3e4cec5e5`.

Whole-bank admission at `b702c81` subsequently completed in **55.625 s**,
peaking at **151,584,768 bytes** sampled aggregate RSS. The complete baseline
bank passed: 2,699 reference occurrences became 2,606 unique references,
with 93 overlaps. Root independently verified every canonical raw matrix
column/coordinate and the actual descriptor rows/consumed tensors. All
overlap coordinates and descriptors agree exactly; raw-score spread is zero.

The expanded bank remains **UNAVAILABLE**: overlapping IDs have conflicting
computed protected descriptor rows. Its failure is preserved; no cells or
aliases were dropped to obtain a usable subset. All observed worker births
closed and registered input/source bytes remained unchanged. No model, field,
route inference, GT score or policy adoption ran in this admission job.

Admission report: `E:/monocular-osm-joint-vpr-proof/run3/bank_result/bank_admission.json`,
SHA256 `0acf3c74da5143808d4f753c24fe0c6c5915ca2f5af7699085e752bb42746c44`.
Root acceptance: `E:/monocular-osm-joint-vpr-proof/run3/bank_root_acceptance.json`,
SHA256 `84bcf5f54932eaee5622cc61b6d8061286bffe81101538b2d6cfbab00503156a`.

Source `2e0f169` implements the separately invoked visual/road prototype with
the predeclared all-reference field, robust mixture, 30-second correlation
powers and corrected initial guidance. Root and the independent peer each
passed 83 focused tests, including the existing road-only/probe controls.
Historical reference encoding remains **UNVERIFIED**. At this input-check
checkpoint no actual fused trial or new localization accuracy was established.
The subsequent baseline-bank trial is recorded below.

## Raw appearance/road fusion (11 November 2026; negative)

Clean `f699182` ran the predeclared complete baseline-bank control. Genuine
map preparation was repeated at that source, closing in **20.875 s** with
**2,585,473,024 bytes** peak sampled aggregate RSS. Every typed network array,
complete network metadata, matched/correlated cue record, topology receipt
and public plan matched the earlier conditional-uniform trial. The expanded
strict control stayed unavailable; none of its cells were selected or dropped
to rescue it.

The fixed 50,000-particle, seed-0 trial consumed all 4,441 native VO samples,
1,110 composed steps, 2,606 references and all 80 visual factors. Its field
contains 6,573,600 F32 values (26,294,400 bytes). Fixed 30-second groups give
20 effective groups; original EXIT powers remain 1, 0.5, 0.5, 1. Initial p/q,
legal-turn prior/q and ancestor 1/g corrections remain explicit. The numerical
worker closed in **47.047 s**, at **1,182,334,976 bytes** peak sampled aggregate
RSS. Root rebuilt the exact field and checked every query event, native
endpoint and returned edge/offset point before GT. All observed workers closed
and protected hashes remained unchanged.

| Single frozen diagnostic | Unrounded error (m) |
|---|---:|
| Symmetric mean | 7,152.842366478733 |
| Start | 13,296.185461995614 |
| End | 4,374.377619451206 |

This improves on the conditional-uniform road/sign-only diagnostic's
18,281.58 m, but is worse than v4's retained **1,387.2377579105248 m** Answer.
It is not adopted and supports no full-fleet improvement. Development remains
**14/15**. Posterior mass is uncalibrated. Complete sampled transition-path
witnesses are not exported, so the point-on-edge check is not a production
route-admission proof.

Proof directory: `E:/monocular-osm-joint-vpr-proof/run4`.
Prediction closure SHA256:
`a0e65cb797f860d5e0bfe915bd94424867bd08e915951d895316ff595acbde23`.
Exact output-integrity check SHA256:
`e10e93d9036799b35f913babd52851fc37a73f29d8e8b11d20d236698f3ba95f`.
Component score SHA256:
`02b0f30a9ac8ce5d6099cee49fe1ea84f8ea7f8a5b15e23e702f24db36044188`.
The first setup attempt stopped before launch on Windows text decoding.
Root's verifier corrections (composition end indices exclude the initial
sample; JSON converts tuple config values to lists) are preserved separately.
No prediction, source method or parameter changed or reran for scoring.

The first expanded overlap conflict was quantified from the already consumed
frozen receipts: descriptor Float64 L2 **1.3859061218023863e-7**, maximum
component difference **7.450580596923828e-9**, exactly equal coordinates and
raw-cosine spread **2.2351741790771484e-8**. This report covers the first
conflict, not the whole bank. The strict bank remains unavailable.

The next prospectively declared research variant uses the earlier all-fifteen
study's **L2 <= 1e-6 against the lexical canonical descriptor** rule. Every
physical call, exact per-pool consumed tensor, native query/model/clock,
public receipt, coordinate and raw-score envelope gate remains required.
Numerical compatibility needs its own status/profile and does not assert
bitwise identity or verify historical encoding. Original strict failures and
production identity gates stay unchanged; any other conflict refuses the
whole bank. This post-hoc development variant has no result yet.

## Numerical-compatibility input study (11 November 2026; expanded unavailable)

Clean `b74f18d` ran the independently registered L2 study using the earlier
fixed **1e-6** bound. Root passed 31 pure-bank tests and 152 compatible/existing
regressions; independent peers approved the IO and field/math paths. Every
original native/OWN/payload, acquisition, query/model, per-pool consumed-tensor,
coordinate and raw-score gate remained required. The source uses a distinct
compatibility type and leaves strict gates/statuses unchanged.

The guarded input job closed in **58.344 s**, with **324,284,416 bytes** peak
sampled aggregate RSS. Baseline remained complete (2,606 unique images), and
Root verified all five array values/headers exactly equal to the earlier
strict baseline control. Expanded remained **UNAVAILABLE** at a later
duplicate. A report-only check of the already consumed prefix measured L2
**9.086784224324718e-6**, maximum component difference
**4.256144165992737e-7**, raw-score spread **4.023313522338867e-7**, and exactly
equal coordinates. This is not a full-bank qualification. No tolerance was
raised, no subset was rescued, and no new sampler/accuracy result followed.

Proof: `E:/monocular-osm-joint-vpr-proof/run5`.
Admission report SHA256:
`ca1b8462fb75149f4653bf7eb8d36752c60f2928c72f4b40bbf65fe02d9f225a`.
Root acceptance SHA256:
`ec79c80777a4515f9620f19a57cefda34046db0ad9b7d4106875dd1c9414544f`.
All observed workers closed; protected sources, inputs and outputs stayed
unchanged. Historical encoding remains **UNVERIFIED**; development is **14/15**.

The next distinct research representation models multiple protected feature
snapshots of one public image as alternatives. A snapshot is identified by
provider ID, computed descriptor-row hash and original raw80-column hash.
Every original physical call is validated before grouping exact copies.
Each image contributes total weight one across its K distinct snapshots, with
weight 1/K each. No descriptor or raw-score equivalence is asserted between
different snapshots; all versions remain in one spatial mixture per query.
Provider coordinates must still agree exactly, and all original per-pool and
native-input gates remain required. Strict and L2 failures are preserved.
Its prospective registration is under
`E:/monocular-osm-joint-vpr-proof/run6`. The closed results follow below.

## Protected snapshot-mixture controls (11 November 2026; negative)

Producer `bb299e5` validated all 42 physical calls in **112.547 s**, at
**172,994,560 bytes** peak sampled aggregate RSS. Root independently checked
every original occurrence, exact coordinate/raw column, actual descriptor row
and consumed tensor, lexical exact-key folding and uniform 1/K provider weights.
Baseline has **2,606 snapshots of 2,606 images** from 2,699 occurrences.
Expanded has **16,072 snapshots of 13,164 images** from all 25,631 occurrences.
Neither old strict nor numerical-compatibility status is upgraded; historical
encoding remains **UNVERIFIED**. Producer acceptance SHA256:
`62523a86ce98b4b5fb0190784bcfe717ead1d89900ca367f585c9db1750aa90a`.

The V1 numerical worker was held before execution after source review found
that ZIP member size checks followed NPY header reads. Separate V2 source
`e344755` checks every member and total declared uncompressed size before the
first header. Root's 38 focused tests passed, including the compressed-header
negative; two independent peers approved the narrow fix. V1 source and
unlaunched registrations remain preserved. Data, field, sampler and scoring
rules did not change.

Fresh same-source map preparation closed in **20.781 s** with all 18 typed
network arrays, complete geometry metadata and cue/topology/plan records equal
to the earlier preparation. Both fixed 50,000-particle, seed-zero controls
completed all 4,441 native frames and 1,110 composed steps. Root rebuilt their
complete fields and verified all 80 query events, 20 correlation groups,
original cue powers, native arrays and output road points. Both predictions
and all workers were closed before either single unmodified lineage was scored.

| Control | Symmetric mean error (m) | Trial runtime (s) | Peak sampled RSS (bytes) |
|---|---:|---:|---:|
| Baseline snapshot bank | 7,152.842366478733 | 46.203 | 1,215,549,440 |
| Expanded snapshot bank | 6,338.264241249098 | 48.750 | 1,288,372,224 |

Both are worse than v4's retained **1,387.2377579105248 m** Answer. Baseline
exactly reproduces the prior strict-bank control. No policy adoption or fleet
gain follows; development remains **14/15**. Point reconstruction does not
prove complete directed transition paths, particle importance corrections,
MAP optimality or calibrated confidence.

The immutable trial subdirectories are `trial_baseline_v2` and
`trial_expanded_v2` under the run6 proof directory. Score SHA256 values:
baseline `821f77b5ab0f977e320abc707edd56164c5129050c2c013097e624de260fe0d2`;
expanded `a48fe5f0b10ef2919ec7cb741273c0dedb1b69dca2d3028c213d64662a87698d`.
Root's first native-array equality check rejected NaN headings despite exact
typed bytes; the recorded verifier correction compares typed hashes, including
NaN payloads. Neither prediction nor numerical source was changed or rerun.

## Complete-neighbourhood mode readout (11 November 2026; correctness fix)

Private review found that selecting the 64 heaviest individual cells before
summing 3x3 neighbourhood mass can omit a heavier broad mode. Source `3001550`
now ranks complete neighbourhoods first, including empty neighbouring centres,
with signed-coordinate handling and deterministic ties. Root passed all 22 new
tests and reproduced the failure using the previous implementation. The author
passed 165 relevant regressions; both peers independently passed the new tests,
and one added 36 scalar-oracle cases. Only mode/readout functions changed.

The separate run7 control used the complete accepted snapshot evidence and
fresh same-source preparation. Baseline and expanded trials closed in
**65.312 s / 68.062 s**, at **1,215,709,184 / 1,244,946,432 bytes** peak sampled
aggregate RSS. Root verified full fields, native arrays and output road points.
Every saved target ESS byte, complete field receipt, query event and sign-factor
record matches its previous control. Both selected coordinate lineages match
exactly too. Both predictions closed before either new offline score, which
therefore remained **7,152.842366478733 / 6,338.264241249098 m**.

The readout bug is corrected, but it does not explain these two failed routes.
No selected-Answer fleet evaluation or policy adoption followed; the last valid
fleet result remains **14/15**. Proof: `E:/monocular-osm-joint-vpr-proof/run7`.

A separate source/private audit found a heading-factor normalization issue:
zero residual across a junction arc retains point-Gaussian density normalization.
Private integration gives extra state-dependent mass at wide arcs. Run8 declares
a normalized circular predictive kernel while retaining genuine signed
innovations and explicitly approximate robust KF bias updates. It is a separate
target experiment; no normalized-heading case trial or accuracy result exists
yet at that checkpoint. The completed normalized-target control follows below.

## Normalized junction-heading target (11 November 2026; negative)

The separate off-default target at `d4266f5` normalizes the circular nearest-arc
predictive kernel and recomputes inlier responsibility before the retained
approximate robust KF bias update. It does not assert exact latent-arc
convolution or exact posterior bias moments. All existing public entries,
configuration and workers remain unchanged. Private validation passed 199
relevant tests. Root repeated 48 pure and 17 integration tests plus 60 independent
scalar quadratures; peers verified the implementation and exact legacy output,
particle-state and RNG equivalence.

Both the original snapshot-data declaration and separate heading-target
declaration were checked before numerical inputs. Fresh preparation bound all
29 source roles and reproduced the prior geometry/topology/cue records. Root
rebuilt both complete appearance fields and verified the full native motion,
all 80 scheduled queries, original signs, road-point reconstruction and resolved
target profile. Both predictions and all actors closed before any new scoring.

| Control | Symmetric mean error (m) | Trial runtime (s) | Peak sampled RSS (bytes) |
|---|---:|---:|---:|
| Baseline snapshots, normalized heading | 18,947.181863246762 | 65.750 | 1,213,440,000 |
| Expanded snapshots, normalized heading | 20,134.2580526717 | 65.718 | 1,249,275,904 |

Both are worse than the corresponding earlier controls and retained v4
**1,387.2377579105248 m** Answer. The normalization fixes a predictive-kernel
inconsistency but does not establish a useful localization model. It stays
experimental and off by default. Development remains **14/15**, with no new
selected-Answer fleet evaluation or adoption.

Proof: `E:/monocular-osm-joint-vpr-proof/run8`. Score SHA256 values:
baseline `5c47c23527da67d6f7141fb4e350cb33ffcbe51030d6515dac5ea5a87f788497`;
expanded `ec19c8c8b4bf85e8ad3a391958a72971657801effc75120927f75cbc8f51f6b3`.
The next source/private study tests future appearance proposal guidance,
preserving each target and exact importance corrections. It is not a parameter
retuning or policy adoption of these negative routes.

## Future appearance proposals (11 November 2026; no goal pass)

Source `05b06c5` adds explicit future-guided entries over each existing heading
target. Full-grid backward messages retain all 80 original query factors and
their powers. Fixed Gaussian reachability is a proposal heuristic; it is
query-spacing dependent, not a physical motion posterior. Initial and turn
ratios stay unchanged. A defensive auxiliary mixture retains at least one
quarter of current target mass and uses the exact inverse effective-guide
correction. Existing entries preserve their outputs, particle state and RNG.

Source/private validation passed **267 tests**, including 44 new pure-guide and
24 integration/worker tests that Root independently repeated. Both peers
approved. The maximum private 16M-value/1,201-endpoint CPU2 benchmark took
4.01 s with 66.65 MB additional traced memory; actual inference retained the
external CPU2/RSS3GiB/headroom8GiB/1,830s guard. Sigma above eight cells refuses
the full control before allocation. The unlaunched first declaration is
preserved; the final `future_guide_study_contract_v2.json` adds that resource
bound and explicit approximation limits before any case execution.

Fresh preparation bound all **31** required source roles and reproduced prior
geometry/cues. Root rebuilt every complete appearance field and backward guide
volume, verified native clocks, query and sign powers and output road points,
then closed **all four** predictions and workers before any new offline score.

| Reference bank | Heading target | Symmetric mean (m) | Runtime (s) | Peak sampled RSS (bytes) |
|---|---|---:|---:|---:|
| Baseline | Original | 4,910.147300513726 | 70.813 | 1,240,473,600 |
| Expanded | Original | 6,041.076373316711 | 76.515 | 1,289,461,760 |
| Baseline | Normalized | 3,605.817422391153 | 71.687 | 1,241,530,368 |
| Expanded | Normalized | 1,058.1609295408853 | 75.453 | 1,289,867,264 |

The last control has lower mean error than retained v4's 1,387.24 m, but remains
well above 200 m. Its start/end errors are **5,200.04 / 5,912.70 m**; the
predicted-to-GT mean is **91.54 m** while GT-to-predicted mean is **2,024.78 m**.
It follows a limited part of the route rather than establishing correct complete
localization. No per-clip winning-control choice, new Answer policy, selected
fleet evaluation or goal pass is claimed. Development remains **14/15**.

Proof: `E:/monocular-osm-joint-vpr-proof/run9`. Four score SHA256 values in table
order: `4356bafcf3871fa40624de23892107cf928670f2dfb7f42ec8aefc0a3e6bb424`,
`725bc0c09a89d2be1755836734aab0659e080529e8ab4a4f6b3c83a826758975`,
`e0696a0fd4f9314d09e56e89c71f27a0bbb6dcf509dc19330083dcdb65f98e43`,
`1857fc5a9a225bb2faeac54efd36f7771d6b90a79e19f57ae0ffc894758fdab9`.
All historical reference-encoding and calibration limitations remain.

## Common-null visual density ratio (11 November 2026; negative)

The separate target at `7168b55` normalizes the query-weighted Gaussian grid N
and coverage-only grid D over the same frozen full grid. It directly samples
`log(.1/M+.9*N/sumN)-log(.1/M+.9*D/sumD)`, before original temporal powering.
Flat and unsupported observations are neutral in private tests. Both banks use
the declared F64 statistics/histogram math; no old F32 equivalence is claimed.
This is a new spatial compatibility model, not calibrated image likelihood.

Private validation passed 301 regressions; Root independently passed 37 pure
and 34 integration tests plus 64 scalar/flat checks. Both peers approved,
including exact preservation of prior entry outputs, final particle states and
RNG. The controls keep EXIT-only proposals, original heading-variance inputs,
all 80 native observations/powers and each parent heading target unchanged.

Fresh preparation bound 31 source roles with equal geometry/cues. Root rebuilt
each entire direct log-ratio field, verified native means/variance/clock,
scheduled factors, source/target identity and road points, then closed all four
predictions and workers before any scoring.

| References | Heading target | Symmetric mean (m) | Runtime (s) | Peak sampled RSS (bytes) |
|---|---|---:|---:|---:|
| Baseline | Original | 5,020.884804690638 | 66.500 | 1,214,914,560 |
| Expanded | Original | 10,531.389060528203 | 71.672 | 1,250,783,232 |
| Baseline | Normalized | 11,178.426729963401 | 71.125 | 1,211,928,576 |
| Expanded | Normalized | 10,820.891157969181 | 69.531 | 1,250,226,176 |

All four fail 200 m and are worse than retained v4's Answer. Neutrality under
flat evidence does not establish useful localization. No new policy or fleet
gain follows; the model remains experimental and development remains **14/15**.

Proof: `E:/monocular-osm-joint-vpr-proof/run10`. Score SHA256 values in table
order: `a7e48dd1362f26ab5d5884cada25008a303bc233554afd8271b490bf4796a034`,
`132cb2eb51c6c3d79c7f523d7d2f654b20a5d7da87792d6fc699d3aad5258834`,
`703a5dd972cd3e7da63fbd547233306caa7dc6183cd307af5dce918babbf8a98`,
`0d3059779c49f826390a8999e5e44022a758d2a326c7ac9f6b6e0faa1176d484`.
Run11 separately declares a no-adaptive-extra-Q ablation over the run9 targets
and future proposals; it is source/private work only at this checkpoint.
