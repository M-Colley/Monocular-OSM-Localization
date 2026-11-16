"""Camera types of the served pass-1 references (input diagnostic).

    python scripts/pass1_camera_types.py OUT.json DATA_DIR [DATA_DIR ...]

GT-free input diagnostic: share of panoramas among each clip's served pass-1
Mapillary references (camera_type from the Graph API), set against the frozen
failure taxonomy. Nothing is tuned on it."""
import glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import benchmark
env, rec = benchmark.child_env()
token = env.get("MLY_TOKEN"); assert token, "token not set"
OUT = Path(sys.argv[1]); cache = json.loads(OUT.read_text()) if OUT.exists() else {}
tax = json.load(open(ROOT / "docs/failure_taxonomy_2026-11.json", encoding="utf-8"))
rows = tax if isinstance(tax, list) else tax.get("clips") or tax.get("rows")
rows = list(rows.values()) if isinstance(rows, dict) else rows
taxo = {str(r.get("clip") or r.get("id")): r for r in rows}
man = json.load(open(ROOT / "benchmarks/manifest.json", encoding="utf-8"))
clips = man["clips"] if isinstance(man, dict) else man
clips = list(clips.values()) if isinstance(clips, dict) else clips
DIRS = sys.argv[2:]
sess = requests.Session()
def lookup(ids):
    def one(chunk):
        for attempt in range(3):
            try:
                r = sess.get("https://graph.mapillary.com/images", params={"access_token": token, "image_ids": ",".join(chunk), "fields": "id,camera_type"}, timeout=60)
                if r.status_code == 200:
                    return {str(d["id"]): d.get("camera_type") for d in r.json().get("data", [])}
            except Exception:
                pass
            time.sleep(2.0 * (attempt + 1))
        return {}
    out = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for got in ex.map(one, [ids[k:k + 100] for k in range(0, len(ids), 100)]):
            out.update(got)
    return out
for c in clips:
    cid = c["id"]; vid = (c.get("video") or {}).get("id") if isinstance(c.get("video"), dict) else None
    key = (vid or cid.replace("overlay_", "").split("_0-")[0]).lower()
    metas = [m for d in DIRS for m in glob.glob(f"{d}/*{key}*/mapillary/mly_ref_meta_*.json")]
    if not metas or cid not in taxo: continue
    meta = json.load(open(metas[0]))
    ids = [r["id"] for r in meta["refs"]]
    if cid not in cache:
        ct = lookup(ids)
        cache[cid] = {"n": len(ids), "resolved": len(ct), "types": {t: sum(1 for v in ct.values() if v == t) for t in set(ct.values())}}
        OUT.write_text(json.dumps(cache, indent=1))
    t = taxo[cid]; s = cache[cid]
    pano = sum(v for k, v in s["types"].items() if k in ("spherical", "equirectangular"))
    print(f'{cid:24s} {c.get("split", ""):5s} mode={t.get("mode")!s:3s} err={t.get("error_m")!s:>8s} pano={pano / max(1, s["resolved"]):.2f} ({s["resolved"]}/{s["n"]}) {s["types"]}', flush=True)
