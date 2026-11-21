"""Generic motor-vehicle title aliases; no images, labels or model outputs."""
import re

VERSION = "ground-motor-title-aliases-v3"
ORIGINAL_ROAD_PATTERN = r"dash\s*cam|driv(?:e|ing)|driveassist|\bcar\b|road|motorway|highway|autobahn|motorcycle|motorbike|\bbus\b|truck|taxi|\bi-\d"
MOTOR_ALIASES_PATTERN = r"\b(?:flixbus|coaches?|camions?|camionnettes?|autocars?|voitures?|poids[\s-]+lourds?|auto|lkw|pkw|autofahrt|busfahrt|motorrad|motorradtour|vrachtwagen|motorfiets|bil|biltur|bilkjøring|bilkjoring|buss|autolla|autoilu|automatka)\b"
ROAD_PATTERN = f"(?:{ORIGINAL_ROAD_PATTERN})|(?:{MOTOR_ALIASES_PATTERN})"
ROAD = re.compile(ROAD_PATTERN, re.I)
# Spelling/localization aliases of the existing no-timelapse criterion.
# These are metadata exclusions, never a response to stamp/GT outcomes.
ORIGINAL_TIME_EXCLUSION_PATTERN = r"time\s*-?lapse|hyper\s*-?lapse|\b\d+x\b|sped\s*up|speed\s*up|rear\s*-?view|rear camera|backwards"
TIME_EXCLUSION_ALIASES_PATTERN = r"\btimelaps\b|\bzeitraffer\b|\bacc[ée]l[ée]r[ée]\b"
TIME_EXCLUSION_PATTERN = f"(?:{ORIGINAL_TIME_EXCLUSION_PATTERN})|(?:{TIME_EXCLUSION_ALIASES_PATTERN})"
EXCLUDE = re.compile(TIME_EXCLUSION_PATTERN, re.I)


def road_word(title: str) -> bool:
    return bool(ROAD.search(title))
