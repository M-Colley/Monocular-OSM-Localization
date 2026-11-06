# How the test2 clips were chosen

The second held-out split (`split="test2"`, 2026-11-05) was selected by these
scripts, before any pipeline run on any candidate. Each screen searches
YouTube (titles and durations only), keeps clips whose title names a city
not used in dev or test, at least 7 minutes long, and (rounds 2-3) not
titled as timelapse/hyperlapse/sped-up or rear-facing footage. A candidate
passes when the coordinate stamp read from three frames (round 1: YouTube's
own 480x360 frame thumbnails; rounds 2-3: three full-resolution frames at
25/50/75 % of the video, seeked directly from the stream) parses within 40 km
of the named city in at least two of them. The first two passing clips per
city in search order are taken.

* `screen1.py` / `screen1.json`: 27 cities, 448 videos, thumbnails only (too
  small to read most stamps: Amsterdam's 52.38 read as 62.38). Its two
  timelapse passes and one rear-facing pass are excluded by the round-2 rule.
* `screen2.py` / `screen2.json`: 37 cities with targeted queries, plus the
  round-1 near-misses, verified at full resolution. 7 pass; the first two per
  city give Amsterdam, Warsaw x2, Montreal x2, Paris.
* `screen3.py` / `screen3.json`: 34 more cities; stopped after 28 when YouTube
  began answering with a bot check. Budapest and Munich pass. Candidates with
  a thumbnail stamp but no verified frame after the block may be false
  negatives and are re-checked once YouTube answers again, together with the
  six cities not reached.

Ground truth is built by `scripts/build_overlay_fleet.py` from the stamps; a
clip whose build fails (Montreal `y2om3vbdcKc`: too few readable fixes) is
dropped and replaced by the next passing clip of the same city
(`jur2mWIr9IY`), a rule that uses no localization result.
