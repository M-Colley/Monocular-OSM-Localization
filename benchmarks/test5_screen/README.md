# Test5: selection and method registered before any search (17 November 2026)

New held-out data for rule `test5_2026_11` (dense pass 1 vs ordinary). Selection never reads a localization result,
descriptor, predicted route or error.

- `selection_preregistration.json` fixes the target (10), a 40-city pool in a fixed order, the three test4 queries,
  the unchanged test4 V3 title eligibility, the unchanged test3 stamp screen and the unchanged GT builder and QC.
  `preregistration_receipt.json` pins its hash.
- `screen.py` screens in that order and stops at each pass; `build_one.py` builds and QCs one pass
  (first 0-600 s, <= 1080p, data dir `data/test5`); `drive.py` alternates them until 10 clips pass QC or the pool ends.
- Records: `screen_results.json` (append-only rows, no coordinates), `queries/` (frozen search results),
  `build_attempts/`. Stamp readings and QC logs stay private under `output/test5_data_acquisition_20261117/`.
- The method is bound in `benchmarks/preregistered_rules.json` (`test5_2026_11`), registered in the same commit.
