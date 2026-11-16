# Coarse-to-fine frontier extension: in-sample development check (November 2026)

Rule `c2f_extension_dev_2026_11` in `benchmarks/preregistered_rules.json`
(registered 2026-11-15 at `16eb137`, before any extension run). Method: commit
`8581acd`, the plain retrieval-track pipeline plus the opt-in
`--vpr-c2f-extend` (at most 6 extra 2 km discs; gate and step constants fixed
in that commit). Clips: the 19 of the 45 used clips on which the pipeline's own
GT-free gate fired on their previous ordinary track. Every split is already
used, so this is **development evidence only**.

## Result

**The extension is not a candidate for a held-out test.** Both decision
hypotheses fail:

- **X1 (no harm): fails.** Budapest `1bGf8OD9Fqo` goes from 157 m with the
  run's own ordinary track to 222 m with the extension.
- **X2 (at least two gains across 200 m): fails.** One gain: Düsseldorf
  `8BUtb7TKUFA`, 260 m to 154 m.
- The count within 200 m is unchanged: **10/19** with the ordinary track and
  **10/19** with the extension.
- **X3 (descriptive):** 8 better, 3 worse, 8 tied within 10 %; one-sided sign
  test p = 0.113. Median error 168 m ordinary vs 178 m extended.

The extension stays opt-in and off by default, as the rule required in every
case.

| Split | Clip | Ordinary (m) | Extended (m) | Change | Extra discs | Frames moved |
|---|---|---:|---:|---|---:|---:|
| test4 | Kansas City `ZLmQWroXthY` | 511 | 385 | better | 1 | 43 |
| test4 | Sofia `jkkl1QgsPSo` | 945 | 940 | tie | 2 | 73 |
| test4 | Bucharest `tK7bHJXAJX8` | 319 | 319 | tie (gate did not fire) | 0 | 0 |
| test2 | Budapest `1bGf8OD9Fqo` | 157 | 222 | **worse (loss)** | 1 | 47 |
| test | Strasbourg `4LvkBo69jZo` | 43 | 11 | better | 2 | 26 |
| test | Manchester `5b-tgjtwb2U` | 55 | 27 | better | 1 | 36 |
| test | Ottawa `bF5IfC2ZHjQ` | 1,419 | 1,278 | tie | 2 | 20 |
| test | Bern `zt8rHvHKBUU` | 168 | 20 | better | 2 | 46 |
| test3 | Bratislava `5koGsNdaVvE` | 24 | 25 | tie | 1 | 1 |
| test3 | Eindhoven `6GciDYJ1FUc` | 21 | 19 | better | 1 | 17 |
| test3 | Zagreb `7n4XQUBEH5Y` | 272 | 272 | tie (gate did not fire) | 0 | 0 |
| test3 | Düsseldorf `8BUtb7TKUFA` | 260 | 154 | **better (gain)** | 1 | 42 |
| test3 | Riga `K_54XHyzRn8` | 31 | 61 | worse | 1 | 23 |
| test3 | Cork `vp_D8mxgEkA` | 4,679 | 4,787 | tie | 1 | 35 |
| dev | Berlin 0-420 s | 135 | 48 | better | 1 | 15 |
| dev | Chicago `LmIHvLMLqFk` | 133 | 178 | worse | 1 | 44 |
| dev | Newcastle `Wrn4_uCxRCQ` | 35 | 35 | tie | 1 | 1 |
| dev | Detroit `g5lnpYCk1Ec` | 1,387 | 813 | better | 2 | 54 |
| dev | Atlanta `y66ZkRpUh4k` | 6,590 | 7,056 | tie | 1 | 53 |

Errors are the symmetric mean route error of each answer (one rep), scored by
`benchmark.py aggregate --offline`. Per-clip records, including the gate counts
and track similarities, are in `docs/c2f_extension_dev_2026-11.json`.

## Reading

- The extension does what it was built for on the clips whose route runs past
  the pass-2 disc and whose pass-1 area was right: Bern 168 to 20 m, Berlin
  135 to 48 m, Strasbourg, Manchester and Düsseldorf all improve.
- It does not rescue the far misses. In the failure taxonomy
  (`docs/failure_taxonomy_2026-11.json`) Ottawa and Detroit are mode A (wrong
  pass-1 area): Detroit improves from 1,387 to 813 m but stays far off, and
  Ottawa barely moves. Atlanta is mode B (the fine retrieval misses the route)
  and stays about 7 km off. Sofia is mode C, the case the extension was built
  for, yet two extra discs and 73 moved frames leave it at 940 m. Cork (no
  taxonomy entry) stays about 4.8 km off.
- The acceptance rule does not separate good from bad extensions. It keeps the
  extended track unless the merged similarity drops by more than 0.02. In
  Budapest (the loss) and Düsseldorf (the gain) the merged similarity equals
  the ordinary one to all printed digits, yet 47 and 42 frames moved, in
  opposite directions of error. Track similarity alone carries no signal for
  this decision. This is an observation from this in-sample run, not a tuned
  rule; any new acceptance rule would need its own registration and fresh data.
- The gate fired on 17 of the 19 clips (it was selected to fire on all 19 on
  their earlier runs; retrieval reruns differ). 22 extra discs were fetched;
  mean wall-clock 2,914 s per clip, CPU only.

## Disclosures

- In-sample on every clip; the method was designed after the test4
  diagnostics. Single rep. The paired design controls for retrieval-rerun
  variance in the comparison, not for the ordinary track's own level.
- **Infrastructure:** all 19 runs exited cleanly with results. An Overpass
  outage made the geometry stage fail in 6 clips (Chicago, Detroit, Atlanta,
  Bratislava, Eindhoven, Cork). With `--answer track` the retrieval answer is
  saved before the geometry stages, every one of the 19 answers is the
  retrieval track, and the extension runs inside the retrieval stage, so
  neither side of the comparison is affected. This check and the decision not
  to rerun were committed (`6d8d4b5`) before any score was read.
- The goal "all within 200 m" cannot be certified by in-sample evidence, and
  is not met here (10/19).

## Reproduction

From a clean checkout at `8581acd` with the staged inputs of the rule's
`staging_amendment` (`/e/claude-scoring/ext/run_ext.sh` ran the five groups):

```
python scripts/metro_ring_paired.py shadow RUNS/ext_<split> RUNS/ext_<split>_ordinary --record c2f_extension
python scripts/benchmark.py aggregate --run-name ext_<split> --runs-dir RUNS --offline
python scripts/benchmark.py aggregate --run-name ext_<split>_ordinary --runs-dir RUNS --offline
python scripts/metro_ring_paired.py compare RUNS/ext_<split> RUNS/ext_<split>_ordinary --record c2f_extension
```
