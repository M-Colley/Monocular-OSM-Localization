"""Round 3 of the test3 screen: verify the still-unread candidates from
downloaded one-second sections instead of direct stream seeks.

recheck5.py found that every remaining 0-of-3 candidate fails the same way:
OpenCV's direct read of the YouTube stream times out ("Stream timeout
triggered after 30 s", "Unable to read from socket"), so these candidates were
never verified, not rejected. Here ``yt-dlp --download-sections`` fetches one
second at each of the same three instants (25/50/75 % of the video, <=1080p)
and the stamp is read from the first frame of each piece with the same OCR and
the same pass rule (>= 2 of 3 within 40 km of the named city). Run before any
test3 ground truth beyond the two Düsseldorf clips and before any run.

    python recheck5b.py recheck5.json recheck5b.json WORKDIR
"""
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("screen5", HERE / "screen5.py")
s5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s5)


def downloaded_fixes(vid, dur, workdir):
    out = []
    for frac in (0.25, 0.5, 0.75):
        t = int(dur * frac)
        target = Path(workdir) / f"{vid}_{t}.mp4"
        if not target.exists():
            time.sleep(s5.THROTTLE_S)
            subprocess.run(["yt-dlp", "-q", "--no-warnings", "-f", "bv*[height<=1080][ext=mp4]/bv*[height<=1080]",
                            "--download-sections", f"*{t}-{t + 1}", "-o", str(target),
                            f"https://www.youtube.com/watch?v={vid}"], capture_output=True, timeout=300)
        cap = cv2.VideoCapture(str(target))
        ok, img = cap.read()
        cap.release()
        got = s5.read_stamp(img) if ok else None
        if got:
            out.append({"t": t, "region": got[0], "lat": got[1][0], "lon": got[1][1]})
    return out


def main(recheck_path, out_path, workdir):
    Path(workdir).mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(recheck_path).read_text(encoding="utf-8"))
    out = Path(out_path)
    done = {r["id"]: r for r in json.loads(out.read_text(encoding="utf-8"))} if out.exists() else {}
    for r in rows:
        if r["ok"] or r["section_fixes"] or r["id"] in done:
            continue
        centre = s5.geocode(r["city"])
        sf = downloaded_fixes(r["id"], r["duration_s"], workdir)
        near = [f for f in sf if centre and
                s5.geodesic_distance_m(centre[0], centre[1], f["lat"], f["lon"]) < 40_000]
        done[r["id"]] = {**{k: r[k] for k in ("city", "id", "title", "duration_s", "query", "rank")},
                         "section_fixes": sf, "n_near": len(near), "ok": len(near) >= 2,
                         "round": "downloaded 1 s sections"}
        print(("OK  " if len(near) >= 2 else "    ") + f"{r['city'][:28]:28s} {r['id']} sections "
              f"{len(sf)}/{len(near)} near  {r['title'][:60]}", flush=True)
        out.write_text(json.dumps(list(done.values()), indent=1, ensure_ascii=False), encoding="utf-8")
    print("rechecked", len(done), "passing", sum(d["ok"] for d in done.values()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
