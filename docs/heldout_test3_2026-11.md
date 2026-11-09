# Held-out test, November 2026 (`test3`): the metro-ring search

Rule `test3_2026_11` (`benchmarks/preregistered_rules.json`) was registered on
2026-11-07, before any test3 clip was downloaded beyond the screen's frames.
It tests the metro-ring search (`--vpr-metro-ring --retrieval-only`, frozen at
`f582928`).

**How the ring works.** When the retrieval track's confidence is below 0.162,
the ring searches a 20 km disc around the city geocode, minus the disc
already searched, in 4 km cells. A merge step then chooses between that
evidence and the ordinary track.

**What it is compared with.** Each clip is scored three ways:
- the ring's answer;
- the same run's ordinary track, recorded before the ring step and scored by
  `scripts/metro_ring_paired.py` with the same aggregator;
- the city-centre guess.

**The run.** One rep per clip, CPU retrieval, `--timeout 28800`, on the
ground-truth builder's own 0-600 s downloads. The worktree is at `d1e9ea0`
(amendment of 2026-11-08: `d1e9ea0` adds only data, runner split support and
scoring on top of `f582928`; there is no diff under `monocular_osm/` or
`main.py`).

**The clips.** Ten clips from nine cities that no development step used,
chosen by the fixed screen in `benchmarks/test3_screen/` (selection table and
replacements there). Three candidates failed ground truth and were replaced
under the rule: Belfast ×2 and Aarhus, all edited uploads.

Nothing was changed between seeing a test3 result and reporting all of them.

**Result in one line:** the ring answer and the ordinary track both put
5 of 10 clips within 200 m. The ring improved one clip (Zagreb, 111 to 85 m)
and made two worse (Katowice `5ewPAw_gIeI`, 2.7 to 15.3 km; Düsseldorf
`HfK2LuXQtfA`, 4.1 to 4.9 km). T1 and T3 do not hold; T2 holds. The ring is
not adopted.

## Per clip

Symmetric mean route error (m). "Gate" is the run's own confidence (pass-2
similarity); the ring runs below 0.162. Modes follow `scripts/failure_taxonomy.py`:
- **A:** route outside the 8 km first-pass disc;
- **B:** route inside it, but the 2 km fine disc is elsewhere;
- **D:** route inside the fine disc, still above 200 m.

Wall time is per clip, on a CPU shared with the parallel development check.

| Clip | City | Gate | Ring ran | **Ordinary track** | **With ring** | City centre | Mode | Wall |
|---|---|---|---|---|---|---|---|---|
| `6GciDYJ1FUc` | Eindhoven | 0.193 | no | **21** | **21** | 857 | ok | 34 min |
| `5koGsNdaVvE` | Bratislava | 0.218 | no | **24** | **24** | 1,685 | ok | 36 min |
| `K_54XHyzRn8` | Riga | 0.172 | no | **31** | **31** | 4,116 | ok | 45 min |
| `7n4XQUBEH5Y` | Zagreb | 0.159 | yes | **111** | **85** (better) | 999 | ok | 4.1 h |
| `G6bUX4GvSbE` | Bristol | 0.140 | yes | **198** | **198** | 7,771 | ok | 3.6 h |
| `8BUtb7TKUFA` | Düsseldorf | 0.182 | no | 260 | 260 | 2,608 | D | 36 min |
| `5ewPAw_gIeI` | Katowice | 0.135 | yes | 2,690 | 15,261 (worse) | 6,178 | B | 3.8 h |
| `HfK2LuXQtfA` | Düsseldorf | 0.137 | yes | 4,127 | 4,852 (worse) | 3,287 | B | 3.8 h |
| `vp_D8mxgEkA` | Cork | 0.145 | yes | 4,679 | 4,768 | 2,757 | B | 2.7 h |
| `MMKAehhi-tA` | Katowice | 0.134 | yes | 37,762 | 37,763 | 42,105 | A | 3.4 h |
| **Within 200 m** | | | 6 of 10 | **5 / 10** | **5 / 10** | 0 / 10 | | |
| | | | | Wilson 95 % 24-76 % | 24-76 % | 0-28 % | | |
| **Median** | | | | **229** | **229** | 3,022 | | |

Bristol is 198.1 m with the ordinary track and 197.9 m with the ring. The two
values differ by 0.1 %, which counts as a tie, and both lie 2 m inside the
threshold.

## Pre-registered hypotheses

- **T1, the ring is within 200 m on more clips than the ordinary track: does
  not hold.** Both are 5/10.
- **T2, no clip within 200 m with the ordinary track is above 200 m with the
  ring: holds.** No such clip.
- **T3, the ring improves more clips than it worsens: does not hold.** One
  better, two worse, seven ties (changes under 10 %); one-sided sign test
  P(X >= 1 | n = 3) = 0.875.

The ring beats the city centre on 7 clips and the ordinary track on 8.

## Why the ring did not help here

These diagnostics use the ground truth and were computed after the result,
to describe it. Nothing was changed because of them.

- **No test3 route lies where the ring searches.** Nine of the ten routes lie
  inside the 8 km first-pass disc. The tenth, Katowice `MMKAehhi-tA` (titled
  "Podzamcze - Katowice"; disclosed at selection), is 36 km or more from the
  pipeline's Katowice geocode. That is outside the ring's 20 km outer radius,
  so the ring could not reach it, and both answers stay at about 38 km.
- **The gate does not single out coverage failures.** It fired on six clips:
  - two the ordinary track already had right (Zagreb, Bristol);
  - three mode-B misses, where the route is inside the searched disc and the
    ring's annulus cannot contain it;
  - the one mode-A miss beyond the ring.
- **Two clips got worse.** On Katowice `5ewPAw_gIeI` and Düsseldorf
  `HfK2LuXQtfA`, all 80 frames switched to ring cells on similarity gains of
  about 0.002 (0.135 to 0.137; 0.137 to 0.139). That is a tenth of the 0.02
  margin below which the pipeline itself treats two similarities as equal. The answer moved from 2.3 to 13.1 km and from 3.5
  to 4.6 km from the route.
- **On the other four triggered clips** the merge took only 2-20 frames from
  the ring. The final centre stayed where the ordinary track's was: the same
  distance from the route, to within 0.1 km. Zagreb's 111 to 85 m gain is a re-decode at the same place with a
  fresh reference set.
- **Cost.** Each triggered clip took about 3 more CPU hours: a median of 3.7 h
  against 36 min for untriggered clips, under contention.

The development check of the same method (`metro_ring_dev_2026_11`, all 29
used clips) is still running and will be reported in full. It does not change
this result.

## What test3 adds about the track itself

The ordinary track here is `f582928` without the ring. That is not the
`ccd0bbe` track scored on test1 and test2, but it is the same retrieval
design.

| Held-out set | Commit | Within 200 m | Wilson 95 % |
|---|---|---|---|
| test1 (7 clips) | `ccd0bbe` | 5/7 | 36-92 % |
| test2 (7 clips) | `ccd0bbe` | 2/7 | 8-64 % |
| test3 (10 clips) | `f582928` | 5/10 | 24-76 % |
| **Pooled** | | **12/24** | 31-69 % |

**How the misses split.** Across the three held-out sets there are 12 misses:
- mode A: 5 (Ottawa, Montreal, Paris, Munich, Katowice `MMKAehhi-tA`);
- mode B: 4 (Bern, Katowice `5ewPAw_gIeI`, Düsseldorf `HfK2LuXQtfA`, Cork);
- mode C: 2 (Budapest, Warsaw `lCm2a3kOEmE`);
- mode D: 1 (Düsseldorf `8BUtb7TKUFA`).

The test1/test2 reading that "held-out misses are coverage misses" (6/7)
does not repeat on test3, where three of the five misses are wrong-district
(mode B) failures inside the searched disc.

Düsseldorf `8BUtb7TKUFA` (260 m) is the first scored clip whose route lies
entirely inside the fine disc but misses 200 m. The fine-disc rule now stands
at 19/20 over 39 clips, not 14/14 over 29.

## Disclosures

- **First attempt.** It failed on all ten clips within seconds, before any
  localization: the pipeline found no video. The ground-truth builder stores
  `input_0-600.mp4` under a title-only folder, while the pipeline looks for
  `input.*` under a title-plus-city folder. The downloads were hard-linked in
  place and the run repeated at the same commit; this counts as an
  infrastructure rerun under the rule. The failed attempt's logs are kept
  (`E:/claude-scoring/test3/interrupted/`).
- **Second attempt.** No infrastructure retries and no Overpass or Mapillary
  errors. All ten produced an 80-point track. Provenance is consistent: one
  commit, clean tree, one environment fingerprint, no leak flags.
- **Burned-in GPS stamps.** They stay in the frames, as for every overlay
  clip. `scene_text.is_position_stamp` drops them from all OCR evidence. The
  builder's `gps_overlay_track.json` is never read by the pipeline.
- **Shared CPU.** The run shared the CPU with the development check, so wall
  times are inflated. One rep per clip: earlier repeated cold runs differed by
  tens to hundreds of metres, and Bristol (198 m) is within that of the
  threshold.
- **Riga waypoint.** The Riga ground truth has a flagged waypoint, reviewed as
  a real turn-around in six consecutive raw fixes and kept unedited
  (`benchmarks/test3_screen/README.md`).

## Reproduce

```bash
cd <worktree at d1e9ea0>
CUDA_VISIBLE_DEVICES=-1 python scripts/benchmark.py run --run-name metro_test3 --fleet all --split test3 \
  --skip-download --timeout 28800 --infra-retries 2 --infra-wait 600 --min-free-disk-gb 40 --min-free-vram-gb 0 \
  --pipeline-arg=--vpr-metro-ring --pipeline-arg=--retrieval-only --data-dir <data with input.mp4 per clip>
python scripts/metro_ring_paired.py shadow output/benchmark_runs/metro_test3 output/benchmark_runs/metro_test3_ordinary
python scripts/benchmark.py aggregate --run-name metro_test3 --offline
python scripts/benchmark.py aggregate --run-name metro_test3_ordinary --offline
python scripts/metro_ring_paired.py compare output/benchmark_runs/metro_test3 output/benchmark_runs/metro_test3_ordinary
python scripts/failure_taxonomy.py --run test3=output/benchmark_runs/metro_test3_ordinary --manifest benchmarks/manifest.json
```

The run and summaries are in `E:/claude-scoring/wt-test3/output/benchmark_runs/metro_test3{,_ordinary}`.
