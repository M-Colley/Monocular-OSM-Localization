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
