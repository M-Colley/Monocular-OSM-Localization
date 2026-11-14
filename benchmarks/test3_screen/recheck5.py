"""Round 2 of the test3 screen: re-read the frames of thumbnail-stamped
candidates whose three full-resolution frames all came back unread.

screen5.py treats a stream that cannot be opened (an expired or refused
``yt-dlp -g`` answer, several of them during YouTube's bot checks) like a
frame without a stamp. As in test2's round 4, every candidate with a thumbnail
stamp and 0 of 3 frames read is verified once more with the same frames and
the same pass rule. Candidates whose frames were read but put the stamp
elsewhere keep their verdict. Run before any test3 ground truth beyond the two
Düsseldorf clips (first in city order, so unaffected) and before any run.

    python recheck5.py screen5.json recheck5.json
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("screen5", HERE / "screen5.py")
s5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s5)


def main(screen_path, out_path):
    rows = json.loads(Path(screen_path).read_text(encoding="utf-8"))
    out = Path(out_path)
    done = {r["id"]: r for r in json.loads(out.read_text(encoding="utf-8"))} if out.exists() else {}
    for r in rows:
        if not r.get("id") or not r["thumb_fixes"] or r["section_fixes"] or r["id"] in done:
            continue
        centre = s5.geocode(r["city"])
        sf = s5.section_fixes(r["id"], r["duration_s"])
        near = [f for f in sf if centre and
                s5.geodesic_distance_m(centre[0], centre[1], f["lat"], f["lon"]) < 40_000]
        done[r["id"]] = {"city": r["city"], "id": r["id"], "title": r["title"],
                         "duration_s": r["duration_s"], "query": r["query"], "rank": r["rank"],
                         "section_fixes": sf, "n_near": len(near), "ok": len(near) >= 2,
                         "round": "recheck of 0-of-3 frame reads"}
        print(("OK  " if len(near) >= 2 else "    ") + f"{r['city'][:28]:28s} {r['id']} sections "
              f"{len(sf)}/{len(near)} near  {r['title'][:60]}", flush=True)
        out.write_text(json.dumps(list(done.values()), indent=1, ensure_ascii=False), encoding="utf-8")
    print("rechecked", len(done), "passing", sum(d["ok"] for d in done.values()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
