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
Running. `scripts/run_lingbot_map.py` runs the released model (Apache-2.0,
4.6 GB, repo `b9dd1b1`) through its own loader and post-processing on the CPU
(SDPA attention, ~6 s/frame on a shared CPU). Drive 0009 (447 frames, 48 min):

* **Not metric.** Its steps are 0.049x the OXTS distance (its scale is
  normalised internally; the paper's ATE is similarity-aligned).
* **Heading slightly better than this repo's VO, same drift**: median
  absolute heading error 1.3° (VO 1.9°), 90th percentile 4.2° (VO 5.4°),
  drift 8.9°/min (VO 8.3°/min).
* **Speed profile weaker than the road-plane estimate**: correlation with the
  true step 0.52 (road plane 0.88, ground flow 0.60).
* The filter with it fails on 0009 (0/5 seeds with its own profile, 0/5 with a
  road-plane global scale) — as does every monocular odometry on this 46 s,
  one-turn drive. Drive 0033 is the test.
<!-- LINGBOT:END -->
