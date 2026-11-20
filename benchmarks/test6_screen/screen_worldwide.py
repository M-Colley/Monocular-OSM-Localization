"""Globally stratified test6 road-video screen across 6 continents (no localization or scoring).

Supports multilingual dashcam queries and native-script city names while preserving
the exact non-peeking thumbnail + 3-frame GPS-stamp verification and GT QC of test5.

Usage:
    python benchmarks/test6_screen/screen_worldwide.py --check-config
    python benchmarks/test6_screen/screen_worldwide.py --freeze-receipt
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREREG_PATH = HERE / "selection_preregistration.json"
RECEIPT_PATH = HERE / "preregistration_receipt.json"
RESULTS_PATH = HERE / "screen_results.json"

EXCLUDE_RE = re.compile(
    r"time[\s-]?lapse|timelaps|zeitraffer|acc[ée]l[ée]r[ée]|"
    r"hyper[\s-]?lapse|sped[\s-]?up|speed\s*up|\b[2-9]x\b|\b10x\b|"
    r"rear[\s-]?cam|rear[\s-]?view|rear\s+camera|backwards|"
    r"\bflying\b|\bflight\b|aircraft|drone|helicopter|"
    r"\btrain\b|\btram\b|railway|gautrain|metro\s+line|subway|"
    r"\bwalk(?:ing)?\b|cycling|bicycle|"
    r"gta\s*5|assetto\s*corsa|euro\s*truck|simulator|beamng|forza",
    re.I,
)

# Restores test4 V3 generic motor-vehicle aliases (title_eligibility_v3.py) plus
# global road & GPS-dashcam model aliases (operational_amendment_03.json).
GLOBAL_V3_ROAD_RE = re.compile(
    r"dash\s*cam|dashboard\s*cam(?:era)?|driv(?:e|ing)|driveassist|\bcar\b|road|"
    r"motorway|highway|autobahn|expressway|freeway|parkway|tollway|turnpike|ring\s*road|bypass|"
    r"motorcycle|motorbike|\bbus\b|truck|taxi|\bi-\d|"
    r"\b(?:flixbus|coaches?|camions?|camionnettes?|autocars?|voitures?|poids[\s-]+lourds?|"
    r"auto|lkw|pkw|autofahrt|busfahrt|motorrad|motorradtour|vrachtwagen|motorfiets|"
    r"bil|biltur|bilkjøring|bilkjoring|buss|autolla|autoilu|automatka|"
    r"ddpai|vantrue|thinkware|rexing|mio\s*mivue|transcend|drivepro|gs9000|qvia|lukas|papago|dvr|"
    r"autopista|autov[ií]a|rodovia|estrada|ruta|ruas|avenida|marginal|"
    r"recorrido|viaje|trajeto|passeio|c[aâ]mara|c[aâ]mera|conduzindo|conduzir)\b|"
    r"قيادة|شارع|طريق|كاميرا|נהיגה|כביש|מצלמת\s*דרך|οδήγηση|δρόμος|ขับรถ|กล้องติดรถยนต์",
    re.I,
)


def load_preregistration() -> dict:
    raw = PREREG_PATH.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if RECEIPT_PATH.is_file():
        receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        actual = hashlib.sha256(raw).hexdigest()
        if actual != receipt.get("selection_sha256"):
            raise ValueError("selection_preregistration.json modified after receipt freeze")
    return data


def is_multilingual_road_title(title: str, road_terms: list[str]) -> bool:
    if not title or EXCLUDE_RE.search(title):
        return False
    folded = title.casefold()
    return any(term.casefold() in folded for term in road_terms) or bool(GLOBAL_V3_ROAD_RE.search(title))


def match_worldwide_city_key(title: str, keys: list[list[str]]) -> list[str] | None:
    if not title:
        return None
    folded = title.casefold()
    for key, display in keys:
        k_fold = key.casefold()
        has_cjk = any("\u3040" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af" for ch in k_fold)
        if has_cjk:
            if k_fold in folded:
                return [key, display]
        elif re.search(rf"(?<!\w){re.escape(k_fold)}(?!\w)", folded):
            return [key, display]
    return None


def freeze_receipt() -> dict:
    raw = PREREG_PATH.read_bytes()
    payload = {
        "schema": "test6-preregistration-receipt-v1",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "selection_sha256": hashlib.sha256(raw).hexdigest(),
    }
    RECEIPT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def validate_preregistration(data: dict) -> dict:
    quotas = data["continent_quotas"]
    assert sum(quotas.values()) == data["target"], "continent_quotas must sum to target"
    pool_counts = Counter(c["region"] for c in data["city_order"])
    for region, q in quotas.items():
        assert pool_counts[region] >= q, f"region {region} has fewer cities ({pool_counts[region]}) than quota ({q})"
    return {"target": data["target"], "quotas": quotas, "pool_by_region": dict(pool_counts)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-config", action="store_true", help="Validate stratified pool & quotas")
    ap.add_argument("--freeze-receipt", action="store_true", help="Write SHA256 receipt before screening")
    args = ap.parse_args(argv)

    data = load_preregistration()
    summary = validate_preregistration(data)
    if args.freeze_receipt:
        rec = freeze_receipt()
        print(json.dumps({"status": "frozen", "receipt": rec, **summary}, indent=2))
        return 0
    print(json.dumps({"status": "valid", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
