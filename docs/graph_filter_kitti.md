# Road-graph particle filter on KITTI: what it needs and what it does here

`monocular_osm/graph_filter.py` is a sequential Monte Carlo localiser on the
directed OSM road graph, in the style of Brubaker, Geiger & Urtasun
([CVPR 2013](https://openaccess.thecvf.com/content_cvpr_2013/html/Brubaker_Lost_Leveraging_the_2013_CVPR_paper.html)):
odometry plus a road map, no GPS, no appearance. It is available as the opt-in
`--graph-filter` channel. This page records its evaluation on the two KITTI
drives in the benchmark. The historical research evaluator is
`python scripts/eval_graph_filter.py` (hours of CPU; results land in
`output/graph_filter/`). It reads raw camera PNGs, KITTI rectified camera
calibration and OXTS sensor timestamps. Its five-seed results describe that
input regime. A separate MP4-only test is recorded immediately below.

## Strict MP4-only check (2026-10-27)

`scripts/probe_strict_video_filter.py` ran fresh inference from the registered
whole MP4 window and the cached whole-city Karlsruhe road graph. Inference
used 1,594 native 1226×370 video frames at stride 1, generic 70° horizontal-FOV
intrinsics (`fx=fy=875.4547`), road-plane camera height 1.5 m and rotation-based
VO heading. Timing came entirely from MP4 presentation times and its encoded
10 FPS timebase: the requested full window was 0:165 s and decoded video times
were 0.0–159.3 s. The process read no calibration, raw PNGs, GPS/OXTS fixes or
sensor timestamp files. No position prior, VPR observations or network access
were used.

The city-wide filter used 600,000 particles, seed 0 and 319 steps at about 2 Hz.
Road-plane coverage was 51.83%; the estimator's own spread set the metric-speed
deviation sigma to 0.09738, with metric log-scale prior sigma 0.25. The frame
cache was bounded to 128 MB and genealogy storage to 256 MB. Genealogy thinning
changes smoothed-route interpolation, while particle modes and random draws
remain unchanged. CPU inference took 439.9 s, with sampled peak private memory
1,093 MB and RSS 674 MB.

Prediction, configuration, source snapshots, video hash and map hash were frozen
before a separate scoring process opened labels. Scoring aligned synced OXTS
positions by rendered-frame index; it read no sensor timestamp files.

| Seed 0 metric | Result |
|---|---:|
| Time to localize (video time) | 78.5 s |
| Mean posterior-mode error after localization | 5.48 m |
| Final posterior-mode error | 13.30 m |
| Mean posterior-mode error over the whole drive | 4,695.75 m |
| Smoothed per-frame mean error | 8.71 m |
| Smoothed route GT→prediction mean, committed waypoints | 2.80 m |
| Smoothed route symmetric mean, committed waypoints | 4.83 m |
| Smoothed route endpoint error | 3.77 m |

The accurate smoothed route is an offline result; the early city-wide modes
are wrong. This establishes one successful seed on one drive under the stated
MP4-only assumptions. It does not establish a five-seed success rate or the
performance of the default pipeline's sampling and 200,000-particle budget.
Posterior mass remains uncalibrated.

Reproduce inference first, then run scoring separately:

```bash
python scripts/probe_strict_video_filter.py --video data/kitti/drive_0033.mp4 --city "Karlsruhe, Germany" --graph output/graph_filter/graph_B_karlsruhe_city.graphml --start 0 --end 165 --particles 600000 --seed 0 --threads 2 --frame-memory-mb 128 --path-budget-mb 256 --output output/strict_video_filter/kitti0033_seed0
```

Local evidence is under `output/strict_video_filter/kitti0033_seed0/`:
`frozen_prediction.json`, `offline_score.json` and `source_snapshot/`.
Frozen manifest SHA256:
`d4a6b085f1f62207cc81dc68a0446a3f0c4e969c2905c852aeecb761f474c4ec`.
The command refuses to reuse a nonempty output directory; choose a fresh one
for another run. Source snapshots preserve the exact code loaded for this run.

## Historical calibrated-camera and sensor-time summary

* **With real odometry it works, even on a whole city.** KITTI 0033 (165 s,
  Karlsruhe) with the OXTS speed and heading localizes all 5 seeds in about 25 s
  to 7 m on a 35 km² box, and in about 31 s to 7 m on the *entire* Karlsruhe
  road network (1,605 directed km, 600,000 particles). That matches the
  published result for this problem (about 3-4 m after 20-52 s on a comparable
  map).
* **Speed is the missing signal, not heading.** With the *true* heading but a
  constant step length (what this repo's visual odometry provides) it localizes
  only at the very end of the drive (157 s); with VO headings and the true speed
  it localizes in 29 s.
* **This repo's monocular VO does not suffice.** Every VO step is normalised to
  unit length, so there is no speed, and the yaw drifts 8-14 degrees per minute.
  The filter localized on 2 of 5 seeds on the box and 0 of 5 on the city, and the
  failures are *deterministic wrong attractors* (the same 4.3, 5.7 and 8.8 km
  errors repeat across seeds). The ground-flow speed profile correlates only
  0.44-0.60 with true step length and did not rescue it (1 of 5).
* **Road-plane speed closes most of the gap in the research setup** (added
  2026-10-16, [below](#road-plane-speed)). Giving each VO step a metric length
  from the road plane and an assumed camera height localizes 0033 on the
  whole Karlsruhe map in 5 of 5 seeds (final ~10 m, smoothed route within
  1 m), with calibrated intrinsics and sensor timing, no GPS position
  observations and no image retrieval. Success came near the end of the
  drive, with sensitivity to assumed height; 0009 still failed.
* **Its confidence is not calibrated for this input.** Only 3 of 14 monocular-VO
  runs reporting confidence above 0.8 were right. Do not present it as a
  probability of correctness.
* **The short drive (0009, 46 s, one turn) is too short** even with perfect
  odometry: it localizes (4 of 5) only on the box, with 0.07-0.10 posterior mass,
  and never on the city.

These ablations show the value of metric steps for this map-matching core.
Other trajectory sources with metric steps (VGGT-Long, Depth Anything 3,
MapAnything) were not evaluated here.

## Setup

Map A is a fixed 35 km² box around the recording area (the same box for both
drives, a Brubaker-style assumption that is **not deployable**); map B is the
full Karlsruhe drive network (a named-city spatial prior). "Localized" means
the mode error first falls under 10 m and stays under 20 m afterwards. Oracle
variants use OXTS speed/heading; VO variants use calibrated camera PNGs and
OXTS timestamp differences. GPS positions enter scoring and diagnostics, rather
than position observations. The road-plane scale-prior choice following bias
diagnostics is described below. Numbers are mean ± std over 5 seeds at a 2 Hz
filter rate.

![error versus time](graph_filter_error_vs_time.png)

Generated by `scripts/eval_graph_filter.py` at commit `be2afd0`; 5 seeds per row, mean ± std over seeds. Filter at 2 Hz. Map A = fixed box (8.425, 48.958, 8.495, 49.02) (W,S,E,N), same for both drives; map B = Karlsruhe city (osmnx drive network).

| Drive | Map | Odometry | N | Localized seeds | Time to localize (s) | Final err (m) | Mean err (m) | Steps < 10 m | Mass ≤ 25 m of GT at end | Smoothed route err (m) | README-style route err (m) | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0009 | A | oracle | 300,000 | 4/5 | 38.6 ± 1.1 | 787 ± 1566 | 2177 ± 519 | 15 ± 8 % | 0.08 ± 0.01 | 762 ± 1502 | 737 ± 1470 | 18 ± 0 |
| 0009 | A | oracle_unit | 300,000 | 0/5 | — | 3140 ± 603 | 2835 ± 133 | 0 ± 0 % | 0.01 ± 0.00 | 3197 ± 511 | 3143 ± 534 | 18 ± 0 |
| 0009 | A | vo_oracle_speed | 300,000 | 0/5 | — | 2740 ± 112 | 2728 ± 267 | 0 ± 0 % | 0.03 ± 0.01 | 2781 ± 139 | 2687 ± 161 | 19 ± 0 |
| 0009 | A | vo | 300,000 | 0/5 | — | 2896 ± 507 | 3019 ± 225 | 0 ± 0 % | 0.00 ± 0.00 | 2921 ± 474 | 2896 ± 484 | 18 ± 0 |
| 0009 | A | vo_gflow | 300,000 | 0/5 | — | 2723 ± 119 | 2769 ± 391 | 0 ± 0 % | 0.01 ± 0.00 | 2774 ± 131 | 2747 ± 123 | 17 ± 0 |
| 0009 | B | oracle | 600,000 | 0/5 | — | 5430 ± 1 | 4422 ± 748 | 0 ± 0 % | 0.01 ± 0.00 | 5360 ± 13 | 5302 ± 46 | 44 ± 1 |
| 0009 | B | vo | 600,000 | 0/5 | — | 5444 ± 10 | 4890 ± 571 | 0 ± 0 % | 0.00 ± 0.00 | 5372 ± 8 | 5348 ± 35 | 49 ± 13 |
| 0033 | A | oracle | 300,000 | 5/5 | 24.6 ± 0.3 | 7 ± 0 | 462 ± 16 | 84 ± 0 % | 1.00 ± 0.00 | 5 ± 1 | 1 ± 0 | 79 ± 17 |
| 0033 | A | oracle_unit | 300,000 | 5/5 | 156.7 ± 0.0 | 4 ± 0 | 920 ± 321 | 45 ± 3 % | 0.96 ± 0.00 | 7 ± 1 | 1 ± 0 | 96 ± 34 |
| 0033 | A | vo_oracle_speed | 300,000 | 5/5 | 29.3 ± 0.3 | 8 ± 0 | 562 ± 25 | 75 ± 1 % | 1.00 ± 0.00 | 5 ± 1 | 1 ± 0 | 149 ± 47 |
| 0033 | A | vo | 300,000 | 2/5 | 155.7 ± 1.0 | 3123 ± 2596 | 3400 ± 1285 | 14 ± 12 % | 0.37 ± 0.44 | 3284 ± 2720 | 3259 ± 2710 | 201 ± 60 |
| 0033 | A | vo_gflow | 300,000 | 1/5 | 165.3 ± 0.0 | 4086 ± 2105 | 4576 ± 522 | 4 ± 7 % | 0.18 ± 0.36 | 4321 ± 2219 | 4280 ± 2207 | 93 ± 5 |
| 0033 | B | oracle | 600,000 | 5/5 | 30.6 ± 0.3 | 7 ± 0 | 1742 ± 98 | 81 ± 0 % | 1.00 ± 0.00 | 4 ± 1 | 1 ± 0 | 256 ± 2 |
| 0033 | B | vo | 600,000 | 0/5 | — | 8801 ± 1 | 9052 ± 9 | 0 ± 0 % | 0.00 ± 0.00 | 9056 ± 2 | 9038 ± 19 | 1938 ± 102 |

README-style route err = `monocular_osm.metrics.score_route(...).gt_to_pred_mean_m` of the smoothed route vs the committed GT waypoints (the benchmark's headline metric; one-directional, see metrics.py).

Existing pipeline on the same drives (README, GPS-free, mean route error): KITTI 0009 5,568 m, 0033 7,126 m; city-centroid baseline 2,645 m / 6,843 m.

## Monocular VO vs OXTS

```json
{
 "0009": {
  "frames": 447,
  "valid_pair_frac": 0.8430493273542601,
  "moving_frac_oxts": 0.9013452914798207,
  "vo_zero_step_while_moving_frac": 0.05829596412556054,
  "heading_rotation": {
   "abs_err_deg_median": 1.9495443000512354,
   "abs_err_deg_p90": 5.441106158046802,
   "drift_deg_per_min": -8.276183267028339,
   "step_noise_deg": 0.45789117881508296,
   "mirror_check_corr": 0.996532652381788
  },
  "heading_translation": {
   "abs_err_deg_median": 5.235779607771247,
   "abs_err_deg_p90": 21.713845360951684,
   "drift_deg_per_min": -100.1349210399174,
   "step_noise_deg": 24.660278101634294,
   "mirror_check_corr": 0.3753550965508598
  },
  "distance_m": 331.30538003085826,
  "ground_flow": {
   "confidence": 1.0,
   "scale_median": 0.45659360155544926,
   "rel_err_after_global_scale_median": 0.3398133821014681,
   "corr_with_true_step": 0.5987463088401531
  }
 },
 "0033": {
  "frames": 1594,
  "valid_pair_frac": 0.9968612680477087,
  "moving_frac_oxts": 1.0,
  "vo_zero_step_while_moving_frac": 0.003138731952291274,
  "heading_rotation": {
   "abs_err_deg_median": 12.205330587753158,
   "abs_err_deg_p90": 27.344149865740018,
   "drift_deg_per_min": 14.080114267212535,
   "step_noise_deg": 0.9010844410399543,
   "mirror_check_corr": 0.992060397046046
  },
  "heading_translation": {
   "abs_err_deg_median": 13.306016815874038,
   "abs_err_deg_p90": 29.963794721594414,
   "drift_deg_per_min": 14.319799618368783,
   "step_noise_deg": 22.482988595256202,
   "mirror_check_corr": 0.9788545427219585
  },
  "distance_m": 1705.9650788914014,
  "ground_flow": {
   "confidence": 1.0,
   "scale_median": 0.5588969187373818,
   "rel_err_after_global_scale_median": 0.2888587930479804,
   "corr_with_true_step": 0.4408717579000626
  }
 }
}
```

## Ground truth vs map coverage (max distance GT -> nearest road)

```json
{
 "0009|A": {
  "max_m": 3.1887144269668366,
  "median_m": 1.775673356040382
 },
 "0009|B": {
  "max_m": 3.1887144269668366,
  "median_m": 1.775673356040382
 },
 "0033|A": {
  "max_m": 4.9299980732107365,
  "median_m": 1.6866474001975722
 },
 "0033|B": {
  "max_m": 4.929998073208023,
  "median_m": 1.6866474001953737
 }
}
```

## Road-plane speed

`monocular_osm/plane_speed.py` gives each VO step a metric length. For road
points two views are related by the plane homography `R + t m^T` (`m = n/d`);
with the VO's rotation and unit translation, `m` is linear in the
correspondences, so RANSAC per frame pair (scikit-image) gives the camera's
height above the road in step units, and an **assumed** camera height
(1.5 m; KITTI's camera is at 1.65 m) turns that into metres (Kitt et al. 2010;
Song & Chandraker 2014). Corners and Lucas-Kanade tracking are OpenCV's. The
road region and RANSAC settings were fixed before any filter run (two
alternative road-region settings were tried on 0009's speed profile and
discarded in favour of the default); the filter's global-scale prior
(log-sigma 0.25) was widened after the 0.76-0.84 scale bias below had been
seen. Regenerate with

```bash
python scripts/eval_graph_filter.py --odometry vo_plane vo_plane_rel --odometry-B vo_plane vo_plane_rel --tag plane
python scripts/eval_graph_filter.py --drives 0033 --maps B --odometry vo_plane --odometry-B vo_plane --camera-height-m 1.2 --tag plane_B_h1.2
```

**The speed estimate against OXTS** (in brackets: the ground-flow estimator
above):

| Drive | Correlation with true step | Median relative error after one global scale | Scale at 1.5 m assumed | Scale at the true 1.65 m | Pairs measured |
|---|---|---|---|---|---|
| 0009 | 0.88 (0.60) | 11 % (34 %) | 0.69 | 0.76 | 57 % |
| 0033 | 0.81 (0.44) | 11 % (29 %) | 0.76 | 0.84 | 55 % |

**The filter with it** (5 seeds, calibrated images and sensor timing as above;
`vo_plane` = metric steps at the assumed height, `vo_plane_rel` = the same
profile with the global scale left unknown, as `vo_gflow`):

| Drive | Map | Odometry | Localized seeds | Time to localize (s) | Final err (m) | Mean err (m) | Steps < 10 m | Smoothed route err (m) | README-style route err (m) |
|---|---|---|---|---|---|---|---|---|---|
| 0033 | A | vo_plane | 5/5 | 155.1 ± 0.0 | 10 ± 0 | 657 ± 130 | 47 % | 7 ± 1 | 1 ± 0 |
| 0033 | A | vo_plane_rel | 4/5 | 155.3 ± 1.1 | 1138 ± 2264 | 2591 ± 1440 | 28 % | 1196 ± 2373 | 1183 ± 2364 |
| 0033 | B | vo_plane | **5/5** | 155.1 ± 0.0 | **11 ± 0** | 5706 ± 1168 | 26 % | 8 ± 2 | **1 ± 0** |
| 0033 | B | vo_plane_rel | 0/5 | — | 8801 ± 1 | 9050 ± 16 | 0 % | 9055 ± 2 | 9016 ± 41 |
| 0009 | A | vo_plane | 0/5 | — | 2607 ± 22 | 2727 ± 219 | 0 % | 2642 ± 9 | 2616 ± 19 |
| 0009 | A | vo_plane_rel | 0/5 | — | 2662 ± 133 | 2605 ± 530 | 0 % | 2682 ± 146 | 2644 ± 138 |

For comparison (top table): 0033 on map B with unit-step VO 0/5 (8,801 m),
with OXTS odometry 5/5 (7 m, localized at 31 s); the full pipeline on this
drive is 7,094 m off and the centre of "Karlsruhe, Germany" 6,709 m
(benchmark run v2).

**Camera-height sensitivity** (0033, map B, `vo_plane`, 5 seeds):

| Assumed height | Scale vs OXTS | Localized seeds | Final errors (m) |
|---|---|---|---|
| 1.2 m | 0.61 | 1/5 | 11, 8800, 8800, 8802, 11759 |
| 1.5 m (default) | 0.76 | 5/5 | 10, 10, 11, 11, 11 |
| 1.8 m | 0.91 | 4/5 | 10, 10, 10, 10, 8803 |

What this shows, and what it does not:

* **Speed was the missing signal in this research setup.** With the whole-city
  spatial prior, calibrated images, sensor timing and no GPS position
  observations or image retrieval, metric steps turn a deterministic 8.8 km
  failure into 5 of 5 localizations of the loop drive.
* **The absolute scale matters, not only the speed profile.** The same profile
  with an unknown scale fails on the city map (0/5), and an assumed height
  20 % too low fails (1/5). The estimator reads about 0.8x short even at the
  true height, so the default sits near the edge of what works.
* **It localizes late.** The filter locks on at ~155 s of a 159 s drive (29-31 s
  with INS speed); for most of the drive its online answer is wrong (mean
  error 5.7 km on map B). The smoothed route at the end is right. It is an
  offline answer for a whole clip, not online tracking.
* **Its confidence still cannot be trusted.** The failed 1.2 m runs ended with
  confidence 1.00 on answers 8.8-11.8 km off.
* **One successful research drive.** 0009 (46 s, one turn) fails with any
  odometry on the city map. The strict generic-intrinsics MP4 result above
  supplies one additional seed on 0033. It leaves independent dashcam,
  rolling-shutter, shorter-window and multiple-seed validation outstanding.

