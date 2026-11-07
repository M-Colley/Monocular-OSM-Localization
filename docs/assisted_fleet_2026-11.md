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
  (one of 1,200 sampled images had no URL).
* OCR failed on every run of this clip: the fixed-count sampler read the frame
  before the VO horizon (`int(118.99999...)`; fixed in `1498923`). Fixed,
  it reads generic shops (no literal branch) and the exit numbers 186B and
  188; the OSM junctions with those refs lie 159-234 m and 196 m from the
  true position at the read times. A prototype decode centred on them
  (dense pool plus 3 km route pool, image-only) reaches 909 m: the location
  is right, the freeway imagery cannot place the 8.5 km route. An exit-number
  cue would need VO or road-graph alignment, not image decoding; not built.

The complete fifteen-clip v4 run (`assisted_v4_fleet`, HEAD after `409a959`,
16 threads, 14,400 s, warm copies of the v3 fleet inputs) runs after the
held-out track run; v4 is in-sample on dev and untested on held-out data.
