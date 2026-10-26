# October 2026 experiments: where the failures come from, and three new methods

Four questions, each answered on the exact data benchmark run v2 used (its
cold-run archives: the Mapillary metadata and pixels each rep downloaded, read
only), with ground truth used for scoring only. Nothing below was tuned on the
clips it is reported on, and the one test with a decision rule was registered
before its test data were scored (`benchmarks/preregistered_rules.json`).

## 1. Most core failures happen before retrieval: the search disc misses the route

For each core clip, the first (city-wide) retrieval pass searches a disc and
keeps a capped sample of the Mapillary images in it. Measured against the true
route (`scripts/compare_vpr_backbones.py --report-only` and an offline check of
the archived candidate pools):

| Clip | Pass-1 disc | Route inside the disc | Images in the disc | Kept (cap) | Route within 50 m of a kept image | ... of any image in the disc |
|---|---|---|---|---|---|---|
| Ulm 4K | 3 km | 100 % | 212,431 | 1,500 | 64 % | 100 % |
| London | 3 km | 100 % | 354,627 | 1,500 | 57 % | 100 % |
| Ulm #2 | 800 m | **0 %** | 8,732 | 1,500 | 0 % | 0 % |
| Berlin | 3 km | **0 %** | 378,215 | 1,500 | 0 % | 0 % |
| KITTI 0033 | 3 km | **0 %** | 197,765 | 1,500 | 0 % | 0 % |
| KITTI 0009 | 8 km | 100 % | 425,171 | 1,200 | **3 %** | 92 % |

* **Three of the four failures were search discs that did not contain the
  route at all**: Ulm #2's one-word title seed ("Napoleon"), Berlin's title
  seed (junk words dragged its centre; and retrieval searched only the central
  3 km of the 6 km seed disc), KITTI 0033's seed from scene text. No backbone
  could have retrieved the right place. The seed rule fixed on 2026-10-17
  (commit `989e83f`) addresses the first two; whether it fixes them end to end
  is what benchmark run v3 measures.
* **The fourth, KITTI 0009, is reference sparsity.** The disc covers the whole
  city (8 km), Mapillary has images within 50 m of 92 % of the route, but the
  1,200 kept (0.3 % of 425,171) cover 3 % of it.
* **Picking the kept images more cleverly does not help at this cap.** One image
  per grid cell (same cap, deterministic) covers 70 % / 55 % / 0 % of the route
  on Ulm 4K / London / KITTI 0009 against 64 % / 57 % / 3 % for the current
  pick: 1,500 images over 28 km² are one per ~130 m cell, too sparse for 50 m
  coverage anywhere. The cap itself (download and embedding cost) is the
  constraint, and the current pick, which follows Mapillary's own density
  along busy streets, is as good as uniform spreading. This is why the clips
  that work drive on heavily photographed central streets.
* **Open structural issue:** the retrieval disc is capped at 3 km even when
  the seed (and road-graph) disc is 6 km; a route in the outer ring of a seed
  disc is searched by the graph but not by retrieval. Not changed here (it
  trades coverage against density); a candidate for a measured follow-up.

## 2. SAGE (ICLR 2026) against MegaLoc: no case for a swap

`scripts/compare_vpr_backbones.py`: both backbones on the same 1,500
first-pass references and the pipeline's 80 query frames per clip, CPU fp32
(MegaLoc's reference embeddings reused from the v2 archive only where their
content fingerprint matches). SAGE ViT-B/14 from `chenshunpeng/SAGE@b52f9fc`
(MIT). Only the two clips whose reference set covers the route can tell the
backbones apart:

| Clip | Model | R@1 ≤ 50 m | R@5 ≤ 50 m | R@1 ≤ 100 m | Retrieval-centre error | Confidence AUROC |
|---|---|---|---|---|---|---|
| Ulm 4K | MegaLoc | 0.07 | 0.12 | **0.24** | **60 m** | 0.84 |
| Ulm 4K | SAGE | **0.10** | 0.12 | 0.15 | 1,850 m | **0.87** |
| London | MegaLoc | 0.16 | 0.28 | 0.30 | 95 m | 0.79 |
| London | SAGE | **0.23** | **0.33** | **0.35** | 100 m | **0.82** |

SAGE retrieves the right place a little more often per frame and ranks its
own confidence slightly better, but on Ulm 4K its wrong matches pull the robust
retrieval centre — what the pipeline actually uses — 1.85 km off. Two clips
are not a basis for changing the default.

## 3. SafeVPR calibrated abstention: registered test negative

Registered as `safevpr_frame_verification` (commit `7befbc6`) before any
held-out score existed; run by `scripts/eval_safevpr.py` with SafeVPR's own
functions (`f11d5fd`, MIT): per query frame, MegaLoc's top-1 first-pass
reference, labelled correct within 50 m; SafeVPR's patch-MNN score (DINOv2
ViT-L/14, T = 1) and the MegaLoc cosine, each calibrated with Mondrian LTT at
α = 0.10 on the core clips and applied unchanged to the overlay clips.

| | Calibration (core) | Test (overlay) |
|---|---|---|
| Frames | 476 | 720 |
| Correct top-1 (≤ 50 m) | 4.0 % | 3.3 % |
| Accepted by patch-MNN / cosine | 0 / 0 | 0 / 0 |
| AUROC patch-MNN / cosine | 0.71 / **0.94** | 0.77 / **0.92** |

* **S1 (patch-MNN FDR ≤ 0.10 on overlay): untestable** — with 4 % correct
  frames no threshold can be certified at α = 0.10, so nothing is accepted.
* **S2 (cosine worse than patch-MNN): not supported** — both accept nothing.
* **S3 (patch-MNN AUROC > cosine): not supported** — the reverse: 0.77 vs 0.92.
  SafeVPR's verifier was built for aligned 10-frame windows from the same rig;
  a dashcam frame against one crowd-sourced photo is a different problem.

Exploratory, not registered: the MegaLoc cosine *ranks* frames well, and the
ranking carries over (the 10 highest-cosine frames are 60 % correct on core and
50 % on overlay, against base rates of 4 % and 3 %; the core's top-25 cosine
threshold, 0.250, accepts 14 overlay frames of which 8 are correct). All
correct overlay frames come from the two clips the pipeline localized
(Newcastle, Atlanta `y66ZkRpUh4k`). This agrees with the clip-level result of
the τ rule: confidence ranks across cities, its calibration does not travel.

## 4. LingBot-Map (arXiv 2604.14141) as odometry for the particle filter

<!-- LINGBOT:BEGIN -->
`scripts/run_lingbot_map.py` runs the released model (Apache-2.0, 4.6 GB,
repo `b9dd1b1`) through its own loader and post-processing on the CPU (SDPA
attention; 6.4 s/frame on 0009, 7.4 s/frame on 0033 on a shared CPU, keyframe
interval chosen by the model's own rule). Its poses are scored against OXTS and
fed to the road-graph particle filter by `scripts/eval_graph_filter.py`, beside
the variants already in `docs/graph_filter_kitti.md`.

**As odometry, against OXTS:**

| | LingBot-Map 0009 | VO 0009 | LingBot-Map 0033 | VO 0033 |
|---|---|---|---|---|
| Heading error, median / 90th pct | 1.3° / 4.2° | 1.9° / 5.4° | **3.2° / 9.1°** | 12.2° / 27.3° |
| Heading drift | 8.9°/min | 8.3°/min | **4.9°/min** | 14.1°/min |
| Step length vs OXTS | 0.049x, corr 0.52 | unit steps | 0.075x, corr **0.01** | unit steps |

Its heading is much better than this repo's VO on the longer drive; its step
lengths are not metric (scale normalised internally — the paper's ATE is
similarity-aligned) and on 0033 carry no speed signal at all. The road-plane
speed (`monocular_osm/plane_speed.py`) correlates 0.81-0.88 with the true step.

**In the particle filter (drive 0033, 5 seeds; 0009 fails with every
monocular odometry):**

| Odometry | Map A localized | A: steps < 10 m | Map B (city) localized | B: steps < 10 m |
|---|---|---|---|---|
| LingBot-Map, own step profile (scale unknown) | 1/5 | 15 % | 0/5 | 0 % |
| LingBot-Map steps x one road-plane scale | 0/5 (all end 53-57 m off; smoothed route 12 m) | 41 % | 1/5 | 17 % |
| Road-plane speed + VO heading (`vo_plane`, earlier) | 5/5 | 47 % | **5/5** | 26 % |
| Road-plane speed + LingBot heading | 5/5 | **64 %** | 4/5 | **45 %** |

The last variant was defined from the OXTS diagnostics above (take the speed
from the road plane and the heading from LingBot-Map) before it was run.
Better heading keeps the filter on the right street for more of the drive, but
it locks on no sooner (~155 s of 159) and lost one city-wide seed to the
familiar 8.8 km wrong attractor. One drive is not a basis for a default:
speed, not heading, remains the deciding signal, and the road-plane estimate
supplies it better than LingBot-Map does.
<!-- LINGBOT:END -->

## 5. The seed fixes, end to end

Runs v3 (core) and v4 (overlay, with the plate fix) re-ran the fleet with the
fixes from section 1. On the core clips, which the seed rule's route coverage
was checked on, v3 is better than v2 on 5 of 6 clips (median 466 m vs 4,351 m).
On the overlay clips v4 is better on 2, worse on 6 (median 6,748 m vs 4,316 m).
Removing bad seeds fixes the clips where the seed disc excluded the route; it
does not create the reference coverage a city-wide search needs (section 1),
so clips that had been rescued by a lucky or weak seed now fail. The binding
constraint is the same as in section 1: references within ~50 m of the route.

## 6. The retrieval-free particle filter on dashcam footage (registered test negative)

Registered as `graph_filter_dashcam` (commit `d5aa0e0`) before any dashcam
filter result existed; run by `scripts/eval_graph_filter_dashcam.py`: road-plane
speed (assumed camera height 1.5 m) on the frames each cached VO used, VO
heading, the `--graph-filter` defaults, 600,000 particles, 3 seeds, on the drive
network of the pipeline's own city-extent search disc (8 km here).

| Clip | Route in disc | VO poses valid | Plane speed: scale / corr vs GT | Filter route error | Pipeline (v3/v4) |
|---|---|---|---|---|---|
| Ulm 4K | 100 % | 51 % | 0.50 / 0.92 | 2,203 | **24** |
| Ulm #2 | 100 % | 64 % | 0.26 / 0.84 | 2,332 | **142** |
| Berlin | 100 % | 80 % | 0.41 / 0.56 | 5,887 | **789** |
| London | 100 % | 63 % | 0.20 / 0.82 | 7,627 | **18** |
| Chicago `vCSEG6KaFng` | 100 % | 41 % | 0.62 / 0.74 | 8,525 | **41** |
| Chicago `LmIHvLMLqFk` | 100 % | 41 % | 0.22 / 0.68 | 6,000 | **254** |
| Scottsdale `wJsEQTCAg1c` | 0 % | 56 % | 0.22 / 0.42 | 12,112 | 13,772 |
| Detroit `ZhGb8q1kliY` | 100 % | 53 % | 0.46 / 0.42 | 5,473 | 7,442 |
| Detroit `1nF_7l07i-E` | 100 % | 58 % | 0.09 / 0.19 | 6,761 | 6,471 |
| Detroit `g5lnpYCk1Ec` | 20 % | 86 % | 0.30 / 0.43 | 5,423 | 8,803 |
| Atlanta `y66ZkRpUh4k` | 100 % | 49 % | 0.14 / 0.79 | 5,091 | 6,748 |
| Atlanta `kxEDNj5L_yQ` | 30 % | 87 % | 0.03 / 0.60 | 13,079 | 6,829 |
| Newcastle `Wrn4_uCxRCQ` | 100 % | 71 % | 0.19 / 0.85 | 4,085 | **362** |

* **G1 (localizes at least one clip): not supported** — every clip ends
  2-13 km off.
* **G2 (beats the pipeline on more clips than it loses): not supported** —
  better on 4 (all where both are kilometres off), worse on 9 (p = 0.27).
* **Why: the odometry, as on KITTI, not the map matching.** On dashcam footage
  the road-plane speed follows the true speed profile (correlation 0.56-0.92
  on 8 of 13 clips) but reads 2-30x short (scale 0.03-0.62, against 0.76 on
  KITTI), far outside the filter's scale prior; only 5-49 % of moving steps get
  an estimate; and the VO keeps only 41-87 % of its poses, each lost pose read
  by the filter as a stop. A frame from Atlanta `y66ZkRpUh4k` shows one cause
  of the short scale: the dashcam is pitched up, so the road fills only the
  bottom ~15 % of the frame and the level-camera road band covers buildings,
  parked cars and the burned-in stamp.

**Post hoc (designed after the registered run, so not evidence of
generalisation):** two GT-free changes to the road band, `plane_speed(...,
horizon="foe", mask_static=True)` — place the band below the VO's own focus of
expansion (the horizon of a car moving parallel to the road) instead of below
the image centre, and drop pixels that never change over the clip (burned-in
stamps). Re-run with everything else unchanged:

| | Better | Worse |
|---|---|---|
| Speed scale vs ground truth | Ulm 4K 0.50 -> 0.92, London 0.20 -> 0.73, Berlin 0.41 -> 0.71, Atlanta `y66ZkRpUh4k` 0.14 -> 0.59 | Chicago `vCSEG6KaFng` 0.62 -> 0.12, Detroit `g5lnpYCk1Ec` 0.30 -> 0.08, Ulm #2 0.26 -> 0.15 |
| Filter | Detroit `1nF_7l07i-E` route 6.8 -> 1.4 km | no clip localized; 4 better / 9 worse than the pipeline, as registered |

The narrower band also leaves fewer road features (coverage falls on most
clips) and does nothing for the VO's lost poses. Conclusion: on this footage
the retrieval-free path fails at the odometry, as it did on KITTI before the
road-plane speed; making it work on dashcams needs a VO that keeps its poses
and a scale estimate robust to unknown mounting, not a better map matcher.

## 7. Denser first-pass references (pilot): coverage is not enough on KITTI

`scripts/dense_retrieval_pilot.py`: the same 8 km first-pass disc as v2 for
KITTI 0009, searched with one reference per 75 m (the pipeline's own fetcher
listed 498,575 images; sha1-ordered thinning kept 8,610; no ground truth),
embedded exactly as the pipeline does, against v2's 1,200-image set on the
same 80 query frames:

| | v2 set (1,200) | Dense (8,610) |
|---|---|---|
| Queries with a reference within 50 m | 3 % | **76 %** |
| Top-1 within 50 m / within 100 m | 0 % / 0 % | 0 % / 0 % |
| Top-5 within 50 m | 0 % | 1.3 % |
| Retrieval-centre error | 4,457 m | 2,649 m |

Density fixes coverage but not retrieval: for the 59 queries with a correct
reference nearby, the best correct one ranks at a median of **3,203 of 8,610**
(10th percentile 333; only 7 % within the top 100). MegaLoc does not recognise
2011 KITTI frames in present-day Mapillary photos at all, so no re-ranking or
sequence decoding over this list can find them. On KITTI the binding problem
is appearance, and only the retrieval-free particle filter has localized it
(drive 0033, section 4 and docs/graph_filter_kitti.md). A second pilot on a
daytime dashcam clip (Detroit `1nF_7l07i-E`) tests whether density helps where
the footage is modern.


## 8. The Detroit pilot found that the reference pool was incomplete everywhere

The Detroit pilot never tested density: the pipeline's fetcher listed 195,428
images in the 8 km disc, and **none within 250 m of the route**, although direct
small queries find 45-50 images near every route point. The pool was listed
with Mapillary's Graph API bbox search, which answers HTTP 500 "Please reduce
the amount of data you're asking for" for a box it cannot scan in time (probed
downtown: 130 m and 270 m boxes answer, 530 m and 1 km do not). For an 8 km disc
the fetcher coarsens its boxes to ~540 m and dropped every rejected box
silently; only 356 of 900 returned anything, and the empty ones were the dense
centre:

```
old Detroit listing, 533 m cells over the 16 km box ('#' > 500 images, '+' some, '.' none)
..#..##+##+##++#+####.##++...#
......+#########+##.#+#..#+...
......+#+...###+......+.......
............+##+......#++.+..+     <- the city centre (disc centre) is empty
............##+.........##++.#
```

**First fix (commit `0a1c771`): split rejected boxes.** Up to four levels of
quarters, one retry for other failures, lost boxes counted. Detroit's disc
then listed 2,148,149 images (22 min). Benchmark run v5 used it and was
stopped after five clips, because the Graph search turned out to be unreliable
beyond the rejections:

* In Berlin it rejects boxes down to **16 m** (532 rejections in one 2 km
  listing; 87 boxes still lost after splitting), so splitting cannot finish.
* Answers it does give are **silently truncated**: the same 100 m Berlin box
  returned 339, 34 or 22 images depending only on the page size requested.
* Karlsruhe, with no error at all, listed 498,876 images of the 836,869 its
  coverage tiles hold.

**Second fix (commit `bf2b83c`): list from the coverage tiles.** Mapillary's
zoom-14 vector tiles (`mly1_public`, layer `image`) list every image of a
~1.5 km tile, with id and position, in one response. They are decoded by
`monocular_osm/mvt.py` (on `protobuf`; GDAL's MVT driver via pyogrio decodes
the same 135,313 points of an 8 MB tile identically but refuses tiles over
10 MiB, which dense downtown tiles exceed), and thumbnail URLs are looked up by
id for the kept sample only. Pass 1 on Detroit now takes 69 s at 2.1 GB peak
memory.

| Detroit `1nF_7l07i-E`, 8 km disc | Original listing | Split boxes (v5) | Coverage tiles (v6) |
|---|---|---|---|
| Images listed | 195,428 | 2,148,149 | **7,126,059** |
| Time | minutes | 22 min | **38 s** |
| Route within 50 m of any listed image | 0 % | 80 % | **100 %** |
| Route within 50 m / 250 m of the 1,200 kept | 0 % / 0 % | 25 % / 95 % | 15 % / 85 % |
| ... of 3,000 kept | 0 % / 0 % | 40 % / 100 % | 15 % / 100 % |

The last two rows are a caveat, not a regression: the kept sample (lowest
sha1 of the id, i.e. uniform over images) follows image density, and the
complete pool includes the heavily re-photographed spots that the time-outs
had happened to flatten. Spreading the sample over space instead does not
help, as section 1 found at 3 km. Over the complete pools of ten dev discs at
the same cap, one image per grid cell (sha1-lowest in each cell, cells grown
until at most `cap` are occupied: 107-448 m) covers the route within 250 m
better on 1 disc, worse on 5, equally on 4, and within 50 m worse on 7: drives
go where the photos are, so the density-following sample stays.

**v5, the five clips it ran (split-box listing):** Ulm 4K 27 m (v3: 24),
Ulm #2 79 m (142), Berlin 692 m (789), London 19 m (18), KITTI 0009 5,346 m
(5,346, identical: its listing had no rejected box and the same sample). The
changes on Ulm #2 and Ulm 4K, whose listings had no rejected box either, come
from Mapillary's data changing between the runs, not from the fix. v6
re-measures every dev clip with the coverage-tile listing and nothing else
changed (the particle filter is opt-in; the new speed options default to the
old behaviour).

A second, separate failure shows in the archived references
(`scripts/pass1_coverage.py`): when the title or a plate gives a seed, pass 1
searches only 3 km around it, although the seed is trusted to 6 km (title) or
20 km (plate). The route lies inside that 3 km core for 0 % of Atlanta
`kxEDNj5L_yQ` (100 % inside 8 km), 0 % of KITTI 0033 (plate seed) and 38 % of
Berlin. Widening the disc would thin the ~1,500 references 4-5 times on the
seeded clips that work (London 18 m, Ulm #2 142 m). `--vpr-seed-ring` (commit
`c02334c`, off by default) instead adds a sparse ring out to the seed radius
at the city-extent mode's caps and keeps the core set unchanged; its adoption
rule (`seed_ring` in `benchmarks/preregistered_rules.json`) was fixed before
any ring run: adopt only if no seeded clip within 200 m without the ring gets
worse than 200 m with it, and at least one gets better than 200 m or more
improve than worsen. (Amended 2026-10-25, before any ring run, to compare v7
with v6 rather than v6 with v5, so both share the coverage-tile listing.)

Phoenix (`wJsEQTCAg1c`) is out of reach of both: its title names five
municipalities of the metro area (Tempe, Scottsdale, Chandler, Gilbert,
Queen Creek), the analysis window drives in Scottsdale/Tempe 15 km from the
Phoenix centroid, outside the 8 km city disc, and no single seed disc built
from the five names contains it (their median lies ~19 km from the route).
Only searching each named place separately could reach it; not attempted.

## 9. v6: the complete reference pool, end to end

Benchmark run v6 (commit `d87fb20`, all 15 dev clips, one cold rep each, CPU
retrieval) changes only the listing (section 8). Chicago `vCSEG6KaFng` failed
three times on an overloaded Overpass server (an infrastructure failure: "all
query workers are busy") and was rerun once it recovered. Symmetric route
error in metres; P is the pipeline's route (`result.position`), T the
retrieval track as a route (`baselines.vpr_track_only`, saved by runs since
`f4f914b`), C the route checked against the track (`baselines.track_checked`);
P, T and C are the candidates of the pre-registered `answer_selection` rule,
registered after the first five v6 clips (disclosed in the rule).

| Clip | v3/v4 | P | T | C | City centre |
|---|---|---|---|---|---|
| Ulm 4K | 24 | 36 | **15** | 36 | 445 |
| Ulm #2 | 142 | 106 | **46** | 106 | 350 |
| London | 18 | **22** | 62 | 22 | 1,501 |
| Berlin | 789 | 695 | **135** | 135 | 909 |
| KITTI 0009 | 5,346 | 3,608 | 3,735 | 3,608 | **2,563** |
| KITTI 0033 | 6,830 | 7,110 | 7,004 | 7,110 | **6,709** |
| Chicago `vCSEG6KaFng` | 41 | **49** | 169 | 49 | 1,924 |
| Chicago `LmIHvLMLqFk` | 254 | 177 | **133** | 177 | 1,230 |
| Phoenix `wJsEQTCAg1c` | 13,772 | 18,507 | 18,644 | 18,644 | **14,802** |
| Detroit `ZhGb8q1kliY` | 7,442 | 77 | **55** | 77 | 403 |
| Detroit `1nF_7l07i-E` | 6,471 | 395 | **77** | 77 | 491 |
| Detroit `g5lnpYCk1Ec` | 8,803 | 2,045 | **1,573** | 1,573 | 8,206 |
| Atlanta `y66ZkRpUh4k` | 6,748 | 6,870 | 6,590 | 6,870 | **2,958** |
| Atlanta `kxEDNj5L_yQ` | 6,829 | **5,485** | 6,024 | 5,485 | 7,280 |
| Newcastle `Wrn4_uCxRCQ` | 362 | 342 | **35** | 342 | 887 |
| **Within 200 m** | 4 / 15 | 6 / 15 | **9 / 15** | 8 / 15 | 0 / 15 |
| **Median** | 5,346 | 395 | **135** | 177 | 1,501 |

* **The complete pool is the largest single change in this project's
  history on the dev clips:** two Detroit clips go from 7.4 and 6.5 km to
  77 m and 395 m (77 m as a track), Chicago `LmIHvLMLqFk` from 254 to 177 m.
  Clips whose earlier pools were complete move within run-to-run noise.
* **The retrieval track is a better answer than the trajectory fit on most
  clips that retrieval locates.** Berlin's track has the same per-frame
  accuracy as London's (median 102 vs 103 m, 76 % vs 91 % of frames within
  250 m), but the fit turns it into a 695 m route; the track itself scores
  135 m. Newcastle: 342 m fit, 35 m track. The fit wins where it is precise
  (London 22 m, Chicago `vCSEG6KaFng` 49 m, a clip with only 42 % of VO
  poses valid where the track is 169 m).
* **What remains is three kinds of failure, none of them retrieval
  ranking among good references:** the route lies outside the searched disc
  (Phoenix: Scottsdale/Tempe, 15 km from the Phoenix centroid; Detroit
  `g5lnpYCk1Ec`, "Detroit to Roseville", 80 % of the window outside the
  8 km disc); a seed's 3 km core misses the route (Atlanta `kxEDNj5L_yQ`,
  KITTI 0033: the seed-ring test, v7); or footage the street imagery does not
  resemble (KITTI's 2011 drives; Atlanta `y66ZkRpUh4k` is filmed at night,
  and its pass-1 sample has a reference within 250 m of the whole route).
  Every failing clip's track is kilometres off rather than near-misses, and
  the retrieval similarity separates them only weakly (median top-1 cosine
  0.105-0.154 on failures against 0.159-0.268 on successes).

## 10. Two pre-registered decisions, and the frozen method

**Seed ring (rule `seed_ring`, run v7 = v6 + `--vpr-seed-ring`, the six
seeded dev clips).** Symmetric error of the pipeline's route:

| Clip | v6 | v7 (ring) | Ring |
|---|---|---|---|
| Berlin | 695 | 695 | to 6 km, +886 refs |
| London | 22 | 22 | to 6 km, +403 |
| Ulm #2 | 106 | 106 | to 8 km, +808 |
| Newcastle | 342 | 342 | to 5.9 km, +1,070 |
| Atlanta `kxEDNj5L_yQ` | 5,485 | **1,969** (track 6,024 -> 1,153) | to 6 km, +906 |
| KITTI 0033 | 7,110 | 13,103 | to 8 km, +1,055 |

S1 (no clip within 200 m breaks) holds; S2 fails (none crosses to within
200 m; one better, one worse, four unchanged). **Not adopted.** On the four
clips whose route lies in the core, the ring's extra references never won a
retrieval, which is the cheap half of the result; on Atlanta it found the
right part of the city for a quarter of the frames (24 % of the track within
250 m, from 0 %), not enough for a route.

**Answer (rule `answer_selection`, v6, all 15 dev clips).** Within 200 m:
the fit 6, the track 9, the checked fit 8 (medians 395 / 135 / 177 m). **The
retrieval track is selected:** from commit `ccd0bbe` the pipeline writes it
to `result["answer"]` by default (`--answer track`; the fit when retrieval
found nothing), and `result["position"]` stays the trajectory fit.

**Frozen method for the held-out clips:** commit `ccd0bbe` defaults, run once
as `test1` on the seven `split=test` clips under rule `test_split_2026_10`.
