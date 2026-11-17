# Dense city-wide pass 1: in-sample development check (November 2026)

Rule `dense_pass1_dev_2026_11` in `benchmarks/preregistered_rules.json`
(registered at `ee369ca` before any scored run). Method: commit `b66b9ee`, the
plain retrieval-track pipeline with `--retrieval-only`. Each clip was run
twice: once with the default pass-1 cap of 1,200 references, and once with
`--vpr-pass1-cap 24000`. The GPU was visible to both arms, and the arms
alternated clip by clip. The clips were the 30 clips of the four used test
splits. Every split is already used, so this is **development evidence
only**.

## Result

**Both decision hypotheses hold, so the dense pass 1 is a candidate for a
fresh held-out test.** That test needs new data, your approval and its own
rule. The dense pass 1 stays opt-in and off by default.

- **D1 (no harm): holds.** No clip goes from within 200 m to above 200 m.
- **D2 (gains): holds.** Five clips cross into 200 m, with no losses:

  | Clip | Ordinary | Dense |
  |---|---:|---:|
  | Sofia `jkkl1QgsPSo` | 945 m | 27 m |
  | Trondheim `pFbUvfr8k-Q` | 207 m | 22 m |
  | Warsaw `lCm2a3kOEmE` | 335 m | 15 m |
  | Katowice `5ewPAw_gIeI` | 2,687 m | 89 m |
  | Zagreb `7n4XQUBEH5Y` | 272 m | 12 m |

- **Within 200 m:** 11/29 with the ordinary arm, **16/29** with the dense
  arm.
- **D3 (descriptive):**
  - 16 better, 3 worse, 10 tied within 10 %; one-sided sign test p = 0.002.
  - Median error 325 m ordinary vs 144 m dense.
  - On the 21 clips that take the city-wide branch, the pass-1 centre lies
    within 2 km of the true route on 14 (ordinary) vs 16 (dense).
  - Mean wall-clock 225 s ordinary vs 926 s dense per clip (GPU, warm
    ordinary caches, cold dense listings and downloads).

| Split | Clip | Branch | Ordinary (m) | Dense (m) | Change | Pass-1 centre to route, ordinary / dense (m) |
|---|---|---|---:|---:|---|---|
| test4 | Kansas City `ZLmQWroXthY` | city | 511 | 393 | better | 387 / 85 |
| test4 | Sofia `jkkl1QgsPSo` | city | 945 | 27 | **gain** | 80 / 11 |
| test4 | Albuquerque `jz6o-_Y6r2E` | city | 9,671 | 11,176 | worse | 9,581 / 11,141 |
| test4 | Trondheim `pFbUvfr8k-Q` | city | 207 | 22 | **gain** | 61 / 81 |
| test4 | Bucharest `tK7bHJXAJX8` | city | 410 | 350 | better | 1,016 / 69 |
| test2 | Montreal `0jJbdTcDseI` | city | 3,458 | 12,894 | worse | 3,260 / 12,887 |
| test2 | Budapest `1bGf8OD9Fqo` | city | 157 | 24 | better | 51 / 259 |
| test2 | Amsterdam `2sS6glBBm8g` | city | 178 | 175 | tie | 153 / 32 |
| test2 | Warsaw `GnXsgpVPvww` | other | 90 | 90 | tie | 31 / 31 |
| test2 | Paris `HpBfjRA60ws` | city | 24,364 | 11,703 | better | 22,597 / 8,314 |
| test2 | Munich `XO6H3mGMKYg` | other | 8,130 | 8,130 | tie | 7,896 / 7,896 |
| test2 | Warsaw `lCm2a3kOEmE` | city | 335 | 15 | **gain** | 310 / 28 |
| test | Sydney `-1xwoetuoOg` | city | 184 | 144 | better | 16 / 78 |
| test | Strasbourg `4LvkBo69jZo` | city | 43 | 10 | better | 164 / 64 |
| test | Manchester `5b-tgjtwb2U` | other | 55 | 55 | tie | 56 / 56 |
| test | Ottawa `bF5IfC2ZHjQ` | city | 1,419 | 350 | better | 1,788 / 267 |
| test | Frankfurt `iJ2JvNk4Rmk` | other | 325 | 325 | tie | 92 / 92 |
| test | NYC `nsONSyk9wtc` | other | 51 | 51 | tie | 1,617 / 1,617 |
| test | Bern `zt8rHvHKBUU` | city | 168 | 16 | better | 930 / 62 |
| test3 | Katowice `5ewPAw_gIeI` | city | 2,687 | 89 | **gain** | 2,316 / 251 |
| test3 | Bratislava `5koGsNdaVvE` | city | 24 | 18 | better | 124 / 134 |
| test3 | Eindhoven `6GciDYJ1FUc` | other | 21 | 21 | tie | 82 / 82 |
| test3 | Zagreb `7n4XQUBEH5Y` | city | 272 | 12 | **gain** | 88 / 64 |
| test3 | Düsseldorf `8BUtb7TKUFA` | other | 260 | 260 | tie | 7 / 7 |
| test3 | Bristol `G6bUX4GvSbE` | city | 6,322 | 8,684 | worse | 6,183 / 8,636 |
| test3 | Düsseldorf `HfK2LuXQtfA` | other | 4,127 | 4,127 | tie | 3,942 / 3,942 |
| test3 | Riga `K_54XHyzRn8` | city | 31 | 15 | better | 170 / 225 |
| test3 | Katowice `MMKAehhi-tA` | city | 37,762 | 37,762 | tie | 35,360 / 35,439 |
| test3 | Cork `vp_D8mxgEkA` | city | 4,679 | 418 | better | 3,835 / 192 |

Errors are the symmetric mean route error of each answer (one rep per arm),
scored by `benchmark.py aggregate --offline`. "Branch" says whether the clip
took the city-wide pass 1 that the cap changes ("city"), or a seeded or
small-disc pass 1 that it does not ("other", identical in both arms).
Per-clip records are in `docs/dense_pass1_dev_2026-11.json`.

## Reading

- **Denser references find the right area more often, and then the track
  is good.** Bern, Ottawa, Katowice, Cork and Bucharest move their pass-1
  centre from 0.9-3.8 km off the route to under 300 m. That matches the
  coverage diagnostic: the misses had few served references near their
  route (`docs/reference_inputs_2026-11.md`).
- **The dense pass 1 often makes pass 2 unnecessary.** In 13 of the 21
  city-branch clips the coarse-to-fine step kept the pass-1 track, because
  the 2 km pass 2 (still 1,500 references) matched less well. The ordinary
  arm kept pass 2 on all 21.
- **It does not fix the far misses.** Albuquerque, Montreal and Bristol get
  worse (they were already 3-10 km off and stay so). Paris improves from
  24 km to 12 km, but stays far off. Katowice `MMKAehhi-tA` is a tie at
  38 km: its route lies outside the searched city disc.
- **Earlier evidence.** About 3,000 references moved no test3 clip across
  200 m (`docs/heldout_test3_2026-11.md`). 24,000 moves five here. This check
  does not show where between the two the effect appears.

## Disclosures

- **In-sample on every clip.** All 30 were used in earlier held-out tests, in
  the failure taxonomy and in the coverage diagnostic that motivated this
  check. Single rep per arm. 24,000 was a feasibility choice, not tuned.
- **One clip not run.** `yd5g7jsWiEM` was refused in both arms, before any
  computation, by the runner's pre-existing city-fairness guard. Its title
  names the city in Romanian ("Bucuresti"), and the guard does not recognise
  "Bucharest". The check and this exclusion were committed (`5999fbe`,
  corrected count `e490a61`) before any score was read.
- **Differences between the arms beyond the cap:**
  - Queries were embedded on the GPU (fp16) in both arms. The ordinary arm
    reused cached CPU (fp32) pass-1 reference descriptors where warm; the
    dense arm computed all of its references on the GPU.
  - The dense listing has its own cache signature, so it was fetched later
    than the ordinary arm's cached listing.
- **The ordinary arm here (11/29) is a fresh rerun of the plain
  retrieval-track pipeline.** It is not the frozen assisted v4 numbers
  (pooled held-out 13/30).
- **The goal "all within 200 m" is not met** (16/29), and in-sample
  evidence cannot certify it.

## Reproduction

Worktree at `b66b9ee` with the staging described in the rule;
`/e/claude-scoring/dense/run_dense.sh` ran both arms. Then, per split:

```
python scripts/benchmark.py aggregate --run-name dense_ord_<split> --runs-dir RUNS --offline
python scripts/benchmark.py aggregate --run-name dense_24k_<split> --runs-dir RUNS --offline
```

```
python scripts/compare_paired_arms.py RUNS dense_ord dense_24k test4 test2 test test3 --out docs/dense_pass1_dev_2026-11.json
```

That script pairs the two arms, reads each result's pass-1 centre, and
measures its distance to `ground_truth/<clip>.json`.
