# Held-out test, November 2026 (`test4`)

Rule `test4_2026_11` (`benchmarks/preregistered_rules.json`) was registered on
2026-11-14, before any test4 localization. It runs the frozen assisted policy
v4 on the six test4 clips:
- commit `409a959`, `--assisted-only --assisted-profile=v4 --assisted-cpu-threads=16`;
- 14,400 s budget, one rep per clip, CPU only;
- clean worktree, cold caches.

Ground truth was read only after all six runs had finished. Nothing was
changed between seeing a result and reporting all of them.

test4 was selected and frozen on 2026-11-10 by its own rule
(`benchmarks/test4_screen/`), which never reads a localization result. Its
six clips come from five cities that no development step used: Bucharest
(2), Sofia, Kansas City, Albuquerque and Trondheim.

**Result in one line:** assisted v4 puts **1 of 6** clips within 200 m
(Trondheim, 187.5 m), against 14/15 on the development clips. It kept the
plain retrieval track on all six, so its answer is the track's answer on
every clip. The goal "all within 200 m" is not met on test4.

## Result

Symmetric mean route error (m), scored with the unchanged
`monocular_osm.metrics.score_route`:

| Clip | Assisted v4 | Baseline track | City centre | Selected | Route in pass-1 / pass-2 disc | Mode | Wall (s) |
|---|---|---|---|---|---|---|---|
| Trondheim `pFbUvfr8k-Q` (night) | **187.5** | 187.5 | 441.3 | baseline | 1.00 / 1.00 | ok | 13,832 |
| Bucharest `tK7bHJXAJX8` (night) | 318.8 | 318.8 | 992.0 | baseline | 1.00 / 0.70 | C | 10,227 |
| Kansas City `ZLmQWroXthY` | 511.3 | 511.3 | 8,981.5 | baseline | 0.10 / 0.40 | A | 14,276 |
| Sofia `jkkl1QgsPSo` | 945.4 | 945.4 | 1,935.7 | baseline | 1.00 / 0.55 | C | 14,274 |
| Bucharest `yd5g7jsWiEM` | 5,009.8 | 5,009.8 | 4,201.5 | baseline | 1.00 / 0.00 | B | 9,115 |
| Albuquerque `jz6o-_Y6r2E` | 9,671.3 | 9,671.3 | 6,950.5 | baseline | 1.00 / 0.00 | B | 13,842 |
| **Within 200 m** | **1/6** | 1/6 | 0/6 | | | | |
| **Median** | **728** | 728 | 3,069 | | | | |

The route-in-disc shares and modes follow `scripts/failure_taxonomy.py`:
- A: the route lies mostly outside the 8 km pass-1 city disc.
- B: the route is inside that disc, but pass 2 searched elsewhere.
- C: pass 2 covers only part of the route.

Every baseline used the city-disc pass 1; none had a title-named prior.

## Pre-registered hypotheses

- **V1 (descriptive primary):** 1/6 within 200 m (17 %, Wilson 95 % 3-56 %),
  against v4's in-sample 14/15 (Wilson 70-99 %). The two intervals do not
  overlap.
- **V2 (no harm):** holds, but trivially. The policy kept the baseline track
  on every clip, so its answer equals the track's.
- **V3 (beats the city centre on at least 4 of 6):** holds, 4/6. It loses on
  Bucharest `yd5g7jsWiEM` and Albuquerque. At n = 6 this is not
  distinguishable from chance (one-sided sign test p = 0.34). On test2 the
  same comparison was 7/7.
- **Goal check:** 1/6. The user's goal "all within 200 m" holds only at 6/6.

## Why the policy never left the baseline

The decision trace is the same on all six clips:
1. The baseline is weak (median cosine 0.117-0.160, below the 0.162 gate),
   so the policy requests cues.
2. The cues stage runs completely: VO valid on 69-97 % of frames, OCR done.
   It yields no sign branch and no metric-PF eligibility.
3. The policy requests the expanded retrieval.
4. The expanded retrieval is not admitted, so the policy keeps the baseline.

| Clip | Baseline median cosine | Expanded stage | Why not admitted |
|---|---|---|---|
| Kansas City | 0.160 | timed out at the work deadline (14,251 s) | no evidence |
| Sofia | 0.136 | timed out at the work deadline (14,259 s) | no evidence |
| Albuquerque | 0.137 | city grid of 46 cells, 12 without any Mapillary image -> "partial" | partial; median 0.153 < 0.162 |
| Trondheim | 0.117 | city grid of 47 cells, 12 without images -> "partial" | partial; median 0.126 < 0.162 |
| Bucharest `tK7b` | 0.117 | complete, identical to the baseline pool | median 0.117 < 0.162 |
| Bucharest `yd5g` | 0.143 | complete | median 0.156 < 0.163 |

On test2, three of seven baselines cleared the gate and were kept as strong.
On test4, none did.

**Post hoc, not part of the rule:** we also scored the declined expanded
candidates where a route exists:
- Trondheim: 100.4 m (already within 200 m with the baseline);
- Albuquerque: 2,426.7 m (still a miss);
- Bucharest `yd5g7jsWiEM`: 10,666.7 m (worse than the baseline);
- Bucharest `tK7b`: 318.8 m (the same route as the baseline).

Admitting every declined candidate would not have moved any clip across
200 m, so the gate's refusals did not cost the goal on test4.

## Infrastructure check (done before any score was read)

- All six runs returned rc 0, with no runner timeout, no infrastructure retry
  and no traceback.
- Every Mapillary listing completed (0 lost tiles). No baseline or expanded
  child logged a network error.
- The cues stage's city retail-inventory request to Overpass returned 502s on
  two clips but succeeded on retry: Kansas City after one failure (5,505
  POIs), Albuquerque after three (3,152 POIs). Unlike the two test2 cases, no
  cue ran with a missing inventory.
- The two expanded-stage timeouts are the policy's own 14,400 s work
  deadline. The dev fleet treats these as method behaviour, and the deadline
  is the budget the rule fixes.
- The "partial" grids are cells with no public imagery. Their listings
  completed and returned zero images.

No run was rerun.

## Pooled held-out picture

The plain retrieval track decided every test4 answer, so test4 adds to the
track-family table from `docs/heldout_test3_2026-11.md`:

| Held-out set | Commit | Within 200 m | Wilson 95 % |
|---|---|---|---|
| test1 (7 clips) | `ccd0bbe` | 5/7 | 36-92 % |
| test2 (7 clips) | `ccd0bbe` | 2/7 | 8-64 % |
| test3 (10 clips) | `f582928` | 5/10 | 24-76 % |
| test4 (6 clips) | `409a959` (assisted v4 = its baseline track) | 1/6 | 3-56 % |
| **Pooled** | | **13/30** | 27-61 % |

Across all held-out data, the assisted policies reach 3/13 (Wilson 8-50 %):
v3 got 2/7 on test2 and v4 got 1/6 on test4. In-sample they reach 13/15 and
14/15.

**The 17 held-out misses split as:**
- mode A: 6 (test4 adds Kansas City, an I-435 highway drive mostly outside
  the city disc);
- mode B: 6 (adds Bucharest `yd5g7jsWiEM` and Albuquerque);
- mode C: 4 (adds Bucharest `tK7b` and Sofia);
- mode D: 1.

Sixteen of 17 misses are search-coverage or search-placement misses, not
errors inside a correctly searched area.

## Reading

test4 confirms the generalisation gap that test2 showed. The assisted
policy's stages and thresholds were developed on the 15 dev clips. On new
cities it falls back to the plain city-wide retrieval track, which then
decides the outcome.

- **The track bounds the held-out share.** Across four held-out sets it
  pools to 13/30.
- **The misses are about where the search looks**, not about how precise it
  is once it finds the route:
  - Kansas City is a highway drive outside the 8 km disc;
  - two clips put pass 2 in the wrong place;
  - two cover only part of the route.

  These are the same failure modes the test3 diagnostics measured.
- **Two misses are near the line:** Bucharest at night (319 m) and Kansas
  City (511 m).
- **CPU compute sets a ceiling.** The expanded city-grid search, the
  policy's main fallback, again ran out of the four-hour CPU budget on two
  of six clips.

## Disclosures

- test4 has six clips (its target of seven was not met), so only large
  effects are visible. It is a single rep; retrieval reruns can differ by
  tens to hundreds of metres.
- **Runner manifest** (`E:/claude-scoring/test4/manifest_test4.json`, sha256
  `7e8960f7...`, registered in the rule):
  - the split is relabelled `test2`, because `409a959`'s runner lists splits
    only up to test2;
  - `yd5g7jsWiEM`'s city is passed as "Bucuresti, Romania", the title's
    spelling, because `409a959`'s city-fairness guard refuses "Bucharest".
    Both strings geocode to the same point. Scoring uses the frozen manifest
    (sha256 `7f4150ff...`) and its city "Bucharest, Romania".
- v4 (`409a959`) predates and uses nothing from the test3 diagnostics or the
  10-12 November experiments. No test4 information reached any method.
- The expanded-candidate scores above are post hoc diagnostics. The
  pre-registered outcome is the selected answer.

## Reproduction

```bash
python scripts/score_heldout_assisted.py --run-dir E:/claude-scoring/wt-test4-v4/output/benchmark_runs/assisted_v4_test4 --out docs/heldout_test4_2026-11.json
```

The script checks three things before scoring:
- the frozen manifest and every ground-truth digest;
- the run's runner-manifest hash;
- that the run used the clean commit `409a959`.

It writes the per-clip records, including each `result.json` sha256, to
`docs/heldout_test4_2026-11.json` (sha256 `99f0a93b...`). The run took
21.0 h of CPU wall-clock time, 2026-11-14 13:36 to 2026-11-15 10:36 UTC.
