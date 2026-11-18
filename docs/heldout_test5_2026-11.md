# Held-out test5: dense pass 1 on new footage (November 2026)

Rule `test5_2026_11` in `benchmarks/preregistered_rules.json`. The rule and
the data selection (`benchmarks/test5_screen/`) were registered together
(`6c49870`) before any search, download or localization. The selection was
frozen at `3f8f8f1` and the run started from it. The pipeline code
(`monocular_osm/` tree `00624b05`) is identical to the version that passed
the in-sample check (`dense_pass1_dev_2026_11`).

The method is the plain retrieval-track pipeline with `--retrieval-only`.
Each clip ran twice, on the GPU, with the arms alternating clip by clip: an
ordinary arm (default pass-1 cap 1,200) and a dense arm
(`--vpr-pass1-cap 24000`).

## Result

**Both registered hypotheses hold, but the support rests on one clip, and
absolute accuracy on new footage is poor.**

- **T1 (no harm): holds.** No clip goes from within 200 m to above 200 m.
- **T2 (benefit): holds.** The dense arm has 2/8 within 200 m, the ordinary
  arm 1/8. The difference is a single gain: Calgary `VaPcyHM-Wtc`, 6,881 m to
  171 m.
- **T3 (descriptive):**
  - Within 200 m: ordinary 1/8 (Wilson 95 % interval 2-47 %), dense 2/8
    (7-59 %).
  - 1 better, 1 worse, 6 tied; one-sided sign test p = 0.75.
  - Median error: 7,042 m ordinary vs 5,548 m dense.
- **Goal check: not met.** 2 of 8 clips are within 200 m.

Under the rule's decision, the dense pass 1 "may be proposed as the
recommended GPU configuration". On this evidence I would not change the
default: the benefit is one clip in eight, and the method fails on most new
footage either way.

| Clip | Branch | Ordinary (m) | Dense (m) | Change | Pass-1 centre to route, ordinary / dense (m) |
|---|---|---:|---:|---|---|
| Melbourne `YJmgh_0SoGs` | other (OCR prior) | 1,346 | 1,346 | tie | 210 / 210 |
| Rotterdam `FPQyfJsbjWM` | city | 26 | 25 | tie | 68 / 16 |
| Glasgow `ml4PPnpc-Kw` | other (small extent) | 11,085 | 11,085 | tie | 9,907 / 9,907 |
| Houston `kMQOOXaUBh0` | city | 7,203 | 16,274 | worse | 5,842 / 16,126 |
| Calgary `VaPcyHM-Wtc` | city | 6,881 | 171 | **gain** | 6,372 / 179 |
| Belgrade `qgaSRWY8Aio` | other (title prior) | 678 | 678 | tie | 6 / 6 |
| Portland `3cT07o9aXyQ` | city | 9,762 | 9,750 | tie | 9,974 / 9,140 |
| Portland `zsqxTSNLw5A` | city | 10,052 | 10,500 | tie | 9,059 / 9,300 |

Errors are the symmetric mean route error (one rep per arm), scored by
`benchmark.py aggregate --offline`. Per-clip records are in
`docs/test5_dense_2026-11.json` (`scripts/compare_paired_arms.py`).

## Reading

- **Only 5 of the 8 clips could show an effect.** Melbourne took an OCR
  prior, Belgrade a title prior, and Glasgow's city extent fits in the
  default disc. Those three use the same pass 1 in both arms.
- **Of those 5:**
  - Calgary is the in-sample pattern: the dense pass finds the right area.
  - Rotterdam was already right.
  - Houston and both Portland clips are wrong-area misses (mode A) in both
    arms, 6-16 km off.
  - Dense references did not move those three onto the route. Houston moved
    further off.
- **The in-sample gain did not reproduce at the same rate.** In-sample it
  was 11/29 to 16/29; here it is 1/8 to 2/8. With 8 clips this is weak
  evidence either way.
- **The binding failure on new footage is still a wrong pass-1 area.**
  Denser references fix it sometimes, not reliably.

## Data selection

- **Pool:** the fixed 40-city pool was screened to exhaustion (120 queries).
  - 861 candidates were ineligible on title or duration.
  - 387 failed the stamp screen, almost all for want of a coordinate stamp.
  - 2 were excluded as unreadable streams.
  - 8 passed, and all 8 built and passed ground-truth QC.
- **Target:** 10 was not met, and the rule reports 8.
- **Five operational amendments to the screen**
  (`benchmarks/test5_screen/operational_amendment_0{1..5}.json`), each
  committed before the next screen outcome, with no localization in
  existence:
  1. seekable HTTPS streams for frame reads;
  2. "unverified" only when frames are unread;
  3. a JavaScript runtime for yt-dlp;
  4. up to three stream opens per attempt;
  5. a three-attempt cap for persistently unreadable streams.

  None changed eligibility, the stamp reader or the pass criterion.

## Disclosures

- **Single rep per arm.** Both arms ran on the GPU. Every listing and
  download was fresh, because these cities had no caches.
- **Not pooled with earlier held-out results.** The method differs (plain
  pipeline vs frozen assisted v4).
- **Infrastructure:** 16/16 runs completed. The check was committed
  (`8e96a91`) before any score was read.
