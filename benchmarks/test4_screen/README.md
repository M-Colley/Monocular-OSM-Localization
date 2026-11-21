# Test4: selection registered before acquisition (10 November 2026)

This is a new, separate held-out data batch. No test4 localization may run
until its method binding is frozen separately. Selection never reads a
localization result, descriptor, predicted route or error.

The target is seven road/dashcam videos. Start with these unused passes of
the fixed test3 screen, in this exact order: `tK7bHJXAJX8`, `yd5g7jsWiEM`
(Bucharest), `jkkl1QgsPSo` (Sofia), `oPbnvEIRm3A`, `ZLmQWroXthY`
(Kansas City), `jz6o-_Y6r2E` (Albuquerque).

`hrbPCnxiQeQ` is excluded **before acquisition/localization**: its title says
"Flying the New Orleans City tour with ATC audio ... Passenger's 1st time".
It is aircraft footage, outside the road-vehicle task. Its earlier stamp
screen pass is preserved; it is not relabelled a stamp failure.

For replacements, screen cities in this fixed order: Trondheim, Tampere,
Rennes, Grenoble, Dresden. Within each city use the previous three queries
in order (`{city} dashcam GPS`, `{city} dashcam VIOFO`, `{city} dashcam drive`),
12 results each, preserving search order and deduplicating video IDs.
Require a city-naming title, duration at least 420 seconds, and affirmative
road/motor-vehicle wording. Exclude aircraft/rail/walking/cycling,
timelapse/sped-up and rear-facing titles. As in the prior screen, require
a coordinate stamp in a YouTube frame thumbnail and at least two of three
full-resolution frames at 25/50/75% to parse within 40 km of the named city.
No localization output is consulted. Unread or blocked frames are
unverified, not negative evidence. Bot checks pause acquisition and are
recorded; they do not advance to another candidate as an accuracy decision.

Download only the first 0–600 seconds (or the shorter whole upload) for
full ground-truth construction, at the prior 1080p cap. Use the unchanged
`build_overlay_fleet.py` CPU OCR, five-second sampling, longest continuous
run, at least 12 fixes, and `check_overlay_gt.py` plausibility checks. Require
at least 500 m in the resulting analysis window. Keep acquisition/GT
failures in the ledger and move through the fixed replacement order until
seven eligible builds exist or the pool ends. Never remove a candidate
because its route is far from the named city or localization would be hard.

New label coordinates, raw fixes and route shapes stay within data QC;
algorithm work receives only ID, city, availability, source path and
duration/plausibility flags. Stamps remain in the input under the existing
policy and are filtered by `scene_text.is_position_stamp` from OCR cues.
The builder's GT, raw stamp cache and GT-derived search disc are evaluation
inputs only. Preserve source URL/title and the exact local download path.

`selection_preregistration.json` and `preregistration_receipt.json` freeze
this selection rule before new network/data work. `method_binding` is
explicitly pending; adding data is not permission to evaluate an unfrozen
method. All qualifying test4 outcomes must eventually be reported together;
inference failures remain failures and cannot trigger data replacement.

## Acquisition operations (not localization results)

The first replacement-screen attempt used the previous helper's default
ONNX thread pools. On a CPU2 process this caused severe oversubscription.
It was stopped with its finished screening verdict preserved; the same
frozen query result/order resumed with RapidOCR's supported ORT intra-op 2,
inter-op 1 settings. Weights, preprocessing and thresholds are unchanged.
The first prefix download was retained unchanged when its label-reader
process received the same operational correction. Pause records and failed
exclusive-slot attempts are preserved; none is a GT or localization verdict.

The first replacement pass is `pFbUvfr8k-Q`, Trondheim (426 s), rank 4 of
the first fixed query. Its prefix/GT build must still pass the same criteria;
this record alone does not close selection. New raw stamp details and QC
logs are kept outside the algorithm-visible screening records.

Independent CPU2 label builders may run in parallel. They write unique
video/label files and serialize the unchanged overlay-index merge with a
Windows file lock. Original label files are never rebuilt by this batch.

The first Kansas City candidate `oPbnvEIRm3A` subsequently failed the fixed
GT builder; its bytes and negative record are retained, and the replacement
search continues. No localization result was produced or used.

An English-only motor-word implementation missed French bus/truck titles.
Before reading those candidates' stamps, V3 registered generic language
aliases and the existing no-timelapse spelling aliases. Its exact regexes,
source hashes and two ordered metadata-only corrections are in
`metadata_eligibility_v3_registration.json`. The original records remain in
`screen_results_before_v3.json` and in the append-only row history, linked
by supersession hashes. Completed stamp and GT-QC verdicts are untouched.
The two corrected metadata entries did not pass the stamp screen.

## Closed data-selection outcome

The five-city replacement pool finished all 15 registered queries. Its
only passing replacement was Trondheim, which passed the unchanged GT QC.
Together with five usable initial candidates, this freezes **six usable
TEST4 videos**. The target was seven and was **not met**. The Kansas City
`oPbnvEIRm3A` GT-builder decline and the aircraft exclusion remain recorded.
No search extension or localization-based replacement was made.

`selection_outcome.json` fixes the eligible order and opaque evidence
hashes. `fixed_pool_closure.json` records the exhausted query pool and
closed observed process births. `availability.json` gives only paths,
durations and QC flags. New coordinates and route shapes remain restricted
to data QC. The separate global method binding is still pending, and no
TEST4 localization or accuracy scoring has run.
