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

**Result in one line:** on test3 the ring answer and the ordinary track both
put 5 of 10 clips within 200 m. The ring improved one clip (Zagreb, 111 to 85 m)
and made two worse (Katowice `5ewPAw_gIeI`, 2.7 to 15.3 km; Düsseldorf
`HfK2LuXQtfA`, 4.1 to 4.9 km). T1 and T3 do not hold; T2 holds. On the 29 clips it was developed on, the
ring reaches 20/29 against 16/29, with no losses (M1 and M2 hold, in-sample).
It stays opt-in and is not made the default.

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

The development check of the same method (below) does show gains, on the
clips whose failures shaped it.

## Development check (`metro_ring_dev_2026_11`, in-sample)

All 29 clips of the used splits were run at `f582928`, worktree
`C:/Users/markc/metro-eval/wt-f582928`, with the same arguments. The data
were copies holding the frozen runs' inputs and warm ordinary-retrieval
caches; the ring caches started cold. One rep per clip. This is in-sample by
construction: the 20 km outer radius was chosen on these clips' ground truth.

| Split | Clip | Gate | Ring ran | Kept | **Ordinary track** | **With ring** | Change | Wall |
|---|---|---|---|---|---|---|---|---|
| test2 | Warsaw (GnXsgpVPvww) | 0.193 | no | baseline | 90 | 90 | tie | 56 min |
| test2 | Amsterdam (2sS6glBBm8g) | 0.213 | no | baseline | 172 | 172 | tie | 54 min |
| test2 | Warsaw (lCm2a3kOEmE) | 0.147 | yes | ring+refine | 335 | 178 | better (gain) | 2.9 h |
| test2 | Budapest (1bGf8OD9Fqo) | 0.179 | no | baseline | 408 | 408 | tie | 33 min |
| test2 | Montreal (0jJbdTcDseI) | 0.152 | yes | ring+refine | 3,004 | 5,235 | worse | 3.3 h |
| test2 | Munich (XO6H3mGMKYg) | 0.124 | yes | ring+refine | 8,130 | 5,338 | better | 4.7 h |
| test2 | Paris (HpBfjRA60ws) | 0.141 | yes | ring+refine | 16,150 | 1,116 | better | 3.1 h |
| test | Strasbourg (4LvkBo69jZo) | 0.232 | no | baseline | 34 | 34 | tie | 41 min |
| test | Manchester (5b-tgjtwb2U) | 0.182 | no | baseline | 56 | 56 | tie | 38 min |
| test | Frankfurt am Main (iJ2JvNk4Rmk) | 0.187 | no | baseline | 84 | 84 | tie | 38 min |
| test | Bern (zt8rHvHKBUU) | 0.163 | no | baseline | 140 | 140 | tie | 36 min |
| test | Sydney (-1xwoetuoOg) | 0.150 | yes | ring+refine | 187 | 189 | tie | 3.4 h |
| test | Ottawa (bF5IfC2ZHjQ) | 0.131 | yes | baseline | 1,380 | 1,380 | tie | 4.2 h |
| test | NYC (nsONSyk9wtc) | 0.152 | yes | ring+refine | 3,029 | 141 | better (gain) | 4.1 h |
| dev | Ulm 4K (ULl8s4qydrk), 0-300 s | 0.268 | no | baseline | 16 | 16 | tie | 1 min |
| dev | Newcastle upon Tyne (Wrn4_uCxRCQ) | 0.159 | yes | ring+refine | 35 | 35 | tie | 1.8 h |
| dev | Ulm #2 (LC9Sa--u5KE) | 0.201 | no | baseline | 43 | 43 | tie | 1 min |
| dev | Detroit (ZhGb8q1kliY) | 0.167 | no | baseline | 55 | 55 | tie | 2 min |
| dev | London (T4wTL3LpLqU) | 0.193 | no | baseline | 62 | 62 | tie | 1 min |
| dev | Detroit (1nF_7l07i-E) | 0.163 | no | baseline | 78 | 78 | tie | 2 min |
| dev | Chicago (LmIHvLMLqFk) | 0.162 | yes | baseline | 133 | 133 | tie | 1.5 h |
| dev | Berlin (lBlKR2ek0w4), 0-420 s | 0.189 | no | baseline | 135 | 135 | tie | 3 min |
| dev | Chicago (vCSEG6KaFng) | 0.186 | no | baseline | 169 | 169 | tie | 3 min |
| dev | Detroit (g5lnpYCk1Ec) | 0.158 | yes | ring+refine | 1,387 | 13,882 | worse | 1.8 h |
| dev | KITTI 2011_09_26 drive 0009 | 0.117 | yes | ring+refine | 3,735 | 18,228 | worse | 2.4 h |
| dev | Atlanta (kxEDNj5L_yQ) | 0.148 | yes | ring | 6,024 | 24 | better (gain) | 2.1 h |
| dev | Atlanta (y66ZkRpUh4k) | 0.143 | yes | ring | 6,590 | 3,469 | better | 1.8 h |
| dev | KITTI 2011_09_30 drive 0033 | 0.105 | yes | ring+refine | 7,004 | 5,751 | better | 2.3 h |
| dev | Phoenix (wJsEQTCAg1c) | 0.132 | yes | ring+refine | 18,644 | 19 | better (gain) | 2.0 h |

| | Ordinary track | With ring |
|---|---|---|
| Within 200 m | 16 / 29 (Wilson 38-72 %) | **20 / 29** (51-83 %) |
| test2 / test1 / dev | 2/7, 5/7, 9/15 | 3/7, 6/7, 11/15 |
| Median | 172 m | 140 m |

- **M1, no harm: holds.** No clip within 200 m with the ordinary track is
  above it with the ring.
- **M2: holds.** Four clips cross into 200 m: Warsaw `lCm2a3kOEmE`, NYC,
  Atlanta `kxEDNj5L_yQ` (6,024 to 24 m) and Phoenix (18.6 km to 19 m). The
  ring improves 8 clips and worsens 3 (Montreal, KITTI 0009, Detroit
  `g5lnpYCk1Ec`), with 18 ties; one-sided sign test p = 0.113.

**Reading the two results together.** The ring does what it was designed
for on the clips that motivated it: routes 8-20 km from the city geocode.
It did nothing on test3, because no test3 route lay there. That is evidence
about how often new footage needs this search, not proof that the
mechanism fails.

Two things keep it opt-in:
- the only held-out estimate shows no gain, at about 3 CPU hours per clip
  where it runs;
- when it is wrong it can move a nearby answer far away. That happened here
  on KITTI 0009 (3.7 to 18.2 km) and Detroit `g5lnpYCk1Ec` (1.4 to 13.9 km),
  and on test3 Katowice (2.7 to 15.3 km).

Single reps are noisy. This run's ordinary track differs from the frozen
`ccd0bbe` track on test1:
- NYC: 3,029 m here, 53 m there;
- Bern: 140 m here, 3,437 m there.

These differences come from repeated cold retrieval and the changed Mapillary
listing, so individual gains and losses here are weak evidence.

**Run conditions.**
- The first launch stalled for about 7 h on the runner's default 8 GB VRAM
  wait and was relaunched with `--min-free-vram-gb 0`. Completed reps were
  kept.
- The disk guard (40 GB) refused Atlanta `kxEDNj5L_yQ` and Newcastle after
  13 triggered clips had filled C: with about 12 GB of ring cache each.
  With the user's approval the finished clips' ring caches were deleted and
  the two clips were run at the same commit.
- No run hit an infrastructure retry, a timeout or a traceback. In total the
  check took 46.6 h of wall-clock time.

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
