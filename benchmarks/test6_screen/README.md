# Test6 Worldwide Stratified Screen Protocol

Globally stratified held-out evaluation split (`test6`, target 12 clips across all 6 inhabited continents: Asia 3, Latin America 2, Middle East & Africa 2, Europe 2, North America 2, Oceania 1).

Unlike `test1`–`test5` (which covered Europe, North America, and 2 Australian clips with English-only queries), `test6` enforces:
1. **Six-continent stratification** (`continent_quotas` in `selection_preregistration.json`).
2. **Multilingual & native-script queries and title eligibility** (`ドラレコ` / `東京`, `行車記錄器` / `台北`, `블랙박스` / `서울`, `câmera veicular` / `São Paulo`, `conduciendo` / `manejando`, Arabic/Hebrew/Greek/Extended-Latin endonyms).
3. **Strict non-peeking separation**: clip selection and GPS-overlay ground-truth extraction (`build_overlay_fleet.py` + `check_overlay_gt.py`) never inspect any localization output, VPR descriptor, or predicted trajectory.
4. **Receipt pinning**: before running `screen_worldwide.py --execute`, freeze `preregistration_receipt.json` (`python benchmarks/test6_screen/screen_worldwide.py --freeze-receipt`) and bind the method commit in `benchmarks/preregistered_rules.json`.
