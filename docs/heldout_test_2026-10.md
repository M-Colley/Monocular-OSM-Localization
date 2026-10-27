# Held-out test, October 2026 (`test1`)

The method was frozen at commit `ccd0bbe` (coverage-tile Mapillary listing,
answer = the retrieval track, seed ring off) after two pre-registered
decisions on the dev clips (`docs/experiments_2026-10.md`, sections 8-10), and
run once on the seven `split=test` clips under rule `test_split_2026_10`
(`benchmarks/preregistered_rules.json`, registered before any test run). One
cold rep per clip, CPU retrieval, from a clean worktree at `ccd0bbe`; scored
by that commit's own aggregator (`dirty: false`). Nothing was changed between
seeing a test result and reporting all of them.

Ottawa and Strasbourg first failed on the Overpass server ("at capacity",
5 attempts each, before the pipeline produced a result) and were rerun with
`--retry-failed` at the same commit; no test result had been looked at
before the rerun.

## Result

Symmetric mean route error (m). "Answer" is `result["answer"]`, the frozen
method's output; the fit (`result["position"]`) and the checked fit are
reported alongside, as the `answer_selection` rule requires.

| Clip | City (as titled) | **Answer (track)** | Fit | Checked fit | City centre | Track frames within 250 m |
|---|---|---|---|---|---|---|
| `nsONSyk9wtc` | NYC | **53** | 660 | 53 | 3,329 | 78 % |
| `4LvkBo69jZo` | Strasbourg | **34** | 490 | 34 | 1,757 | 91 % |
| `5b-tgjtwb2U` | Manchester | **56** | 617 | 56 | 1,162 | 81 % |
| `iJ2JvNk4Rmk` | Frankfurt am Main | **84** | 271 | 271 | 606 | 75 % |
| `-1xwoetuoOg` | Sydney | **192** | 501 | 192 | 5,486 | 55 % |
| `bF5IfC2ZHjQ` | Ottawa | 1,380 | 2,201 | 1,380 | 6,476 | 16 % |
| `zt8rHvHKBUU` | Bern | 3,437 | 3,886 | 3,437 | **1,608** | 0 % |
| **Within 200 m** | | **5 / 7** | 0 / 7 | 4 / 7 | 0 / 7 | |
| **Median** | | **84** | 617 | 192 | 1,757 | |

**T1** (the answer beats the city-centre guess on more clips than it loses):
6 wins, 1 loss (Bern); one-sided sign test p = 0.0625. Direction as
hypothesised; with seven clips this is not significant at 0.05, and no
significance level was registered.

**T2** (share within 200 m, Wilson 95 % interval): test 5/7 = 71 %
[36 %, 92 %]; dev (v6, the same answer) 9/15 = 60 % [36 %, 80 %]. The
held-out share is in line with dev. The trajectory fit alone, which the dev
clips had ranked lower, gets 0/7 on test (dev 6/15): choosing the answer by a
pre-registered rule on dev, rather than keeping the fit, is what carried over.

## The two failures

* **Ottawa** (`bF5IfC2ZHjQ`): only 30 % of the route lies inside the 8 km
  city disc (median 10.8 km from the centroid); the title names no place.
  The same failure as Phoenix and Detroit `g5lnpYCk1Ec` on dev: the search
  never covers the drive.
* **Bern** (`zt8rHvHKBUU`): the route lies entirely inside the disc, but
  retrieval picked the wrong district (track median 5 km off, no frame
  within 250 m). The same failure as Atlanta `y66ZkRpUh4k` and KITTI on dev:
  references cover the route, the ranking does not find it. Bern is the one
  clip where the city-centre guess is better.

## Reading

On footage from seven cities none of the method's development used, the
frozen method localizes five drives to within 200 m (34-192 m) from the
video and a city name alone, against 0 of 7 for the city-centre guess and
for the method's own trajectory fit. The remaining errors are kilometre-scale
misses of two kinds that the dev clips already showed, not near-misses: a
drive outside the searched area, and retrieval choosing the wrong district.
Seven clips bound the success rate only loosely (36-92 %).
