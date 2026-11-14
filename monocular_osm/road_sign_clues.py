"""Pure experimental road-sign clues from literal OCR and public map metadata.

An EXIT reference identifies a signposted target, not a camera position or a
claim that the camera takes that exit. Every spatially feasible reading and
every exact OSM ref match is retained. No IO, model, route or GT input exists.
"""
from __future__ import annotations
from collections import defaultdict
from bisect import bisect_left, bisect_right
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

TAG_DOCUMENTATION = "https://wiki.openstreetmap.org/wiki/Tag:highway=motorway_junction"
JUNCTION_ENDPOINT = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
MAX_MATCH_CANDIDATES = 100_000
_EXIT = re.compile(r"(?<![A-Z])EXIT(?![A-Z])", re.I)
_REF = re.compile(r"([0-9]{1,4})(?:\s*([A-Z]))?", re.I)
_ROUTE = re.compile(r"(?:I|US|SR|A|B|M)\s*[- ]?\s*[0-9]{1,4}[A-Z]?(?:\(M\))?", re.I)
_DIRECTIONS = {"NORTH","SOUTH","EAST","WEST","N","S","E","W","NE","NW","SE","SW"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RoadSignConfig:
    exit_confidence: float = .90
    context_confidence: float = .80
    same_frame_tolerance_s: float = 1e-6
    max_bbox_gap_heights: float = 2.
    max_center_distance_heights: float = 6.
    max_detections: int = 10_000
    max_pairs: int = 50_000
    max_text_chars: int = 512

    def __post_init__(self):
        for key in ("exit_confidence","context_confidence"):
            value = getattr(self,key)
            if isinstance(value,bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("road-sign confidence limits must be finite in [0,1]")
        for key in ("same_frame_tolerance_s","max_bbox_gap_heights","max_center_distance_heights"):
            value = getattr(self,key)
            if isinstance(value,bool) or not math.isfinite(value) or value < 0:
                raise ValueError("road-sign geometric limits must be finite and nonnegative")
        for key in ("max_detections","max_pairs","max_text_chars"):
            value = getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or value < 1:
                raise ValueError("road-sign budgets must be positive integers")


def _ref(value):
    match = _REF.fullmatch(str(value).strip().upper())
    return match.group(1) + (match.group(2) or "") if match else None


def _refs(value):
    # OSM uses semicolons for multiple refs. Never invent an omitted number or
    # turn a road-number list such as '75 65' into the new number '7565'.
    parts = re.split(r"\s*[;,]\s*", str(value).strip())
    parsed = [_ref(part) for part in parts]
    return sorted(set(parsed)) if parsed and all(parsed) else []


def _bbox(value, shape):
    if not isinstance(value,(list,tuple)) or len(value) != 4:
        return None
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in value):
        return None
    x0,y0,x1,y1 = map(float,value)
    if not 0 <= x0 < x1 <= shape[1] or not 0 <= y0 < y1 <= shape[0]: return None
    return [x0,y0,x1,y1]


def _near(a,b,config):
    scale = max(a[3]-a[1],b[3]-b[1])
    gap = math.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0)) / scale
    center = math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2) / scale
    return gap <= config.max_bbox_gap_heights and center <= config.max_center_distance_heights


def extract_road_sign_clues(detections, frame_shape_hw, *, config=None):
    """Literal English EXIT + adjacent reference; bare numbers never localize.

    Auxiliary directions/routes/text remain unlocalized diagnostic context.
    No OCR substitution, nearest-ref choice or per-city parsing occurs.
    """
    config = config or RoadSignConfig()
    shape = list(frame_shape_hw)
    if len(shape) != 2 or any(isinstance(v,bool) or not isinstance(v,int) or v < 1 for v in shape):
        raise ValueError("road signs need the actual native image height/width")
    raw = []
    for row in detections:
        if len(raw) >= config.max_detections: raise ValueError("road-sign detection budget exceeded")
        raw.append(deepcopy(dict(row)))
    anchors, references, context, rejected = [], defaultdict(list), [], []
    for index,row in enumerate(raw):
        confidence, timestamp = row.get("confidence"),row.get("t_sec")
        if (isinstance(confidence,bool) or not isinstance(confidence,(float,int)) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1 or isinstance(timestamp,bool) or not isinstance(timestamp,(float,int))
                or not math.isfinite(timestamp) or timestamp < 0 or not isinstance(row.get("text"),str)
                or len(row["text"]) > config.max_text_chars):
            raise ValueError("OCR text/times/confidence must be finite and correctly typed")
        box = _bbox(row.get("bbox"),shape)
        normalized = row["text"].strip().upper()
        record = {"source_index":index,"text":row["text"],"confidence":float(confidence),
                  "t_sec":float(timestamp),"bbox":box}
        if box is None:
            rejected.append({**record,"reason":"native_bbox_missing_or_invalid"})
            continue
        if confidence >= config.context_confidence:
            kind = ("direction" if normalized in _DIRECTIONS else "explicit_route" if _ROUTE.fullmatch(normalized)
                    else "untyped_numeric" if re.fullmatch(r"[0-9]{1,4}(?:\s+[0-9]{1,4})*",normalized) else "other_text")
            context.append({**record,"kind":kind,"usable_as_camera_heading":False,
                            "usable_as_camera_position":False})
        if confidence < config.exit_confidence: continue
        if _EXIT.search(normalized):
            tail = _EXIT.sub("",normalized,count=1).strip(" :#")
            anchors.append({**record,"inline_refs":_refs(tail)})
        ref = _ref(normalized)
        if ref: references[float(timestamp)].append({**record,"ref":ref})
    reference_times = sorted(references)
    windows, pairs = [], sum(len(anchor["inline_refs"]) for anchor in anchors)
    if pairs > config.max_pairs: raise ValueError("road-sign inline pair budget exceeded")
    for anchor in anchors:
        lo = bisect_left(reference_times,anchor["t_sec"]-config.same_frame_tolerance_s)
        hi = bisect_right(reference_times,anchor["t_sec"]+config.same_frame_tolerance_s)
        times = reference_times[lo:hi]
        pairs += sum(len(references[timestamp]) for timestamp in times)
        if pairs > config.max_pairs: raise ValueError("road-sign pair budget exceeded before pairing")
        windows.append(times)
    observations = []
    for anchor,times in zip(anchors,windows):
        for ref in anchor["inline_refs"]:
            observations.append({"ref":ref,"t_sec":anchor["t_sec"],"confidence":anchor["confidence"],
                "anchor":deepcopy(anchor),"reference":deepcopy(anchor),"pairing":"inline_EXIT_reference"})
        for timestamp in times:
            for reference in references[timestamp]:
                if _near(anchor["bbox"],reference["bbox"],config):
                    observations.append({"ref":reference["ref"],"t_sec":anchor["t_sec"],
                        "confidence":min(anchor["confidence"],reference["confidence"]),
                        "anchor":deepcopy(anchor),"reference":deepcopy(reference),"pairing":"same_frame_nearby_native_boxes"})
    settings = asdict(config)
    return {"experimental":True,"validation_status":"posthoc_dev_unvalidated","selected_answer":None,
        "confidence_calibrated":False,"camera_fix":None,"interpretation":"signposted exit target, not camera position or exit taken",
        "config":settings,"config_sha256":_digest(settings),"detections_sha256":_digest(raw),
        "frame_shape_hw":shape,"n_detections":len(raw),"exit_anchors":anchors,"observations":observations,
        "observations_sha256":_digest(observations),
        "context":context,"rejected":rejected,"tag_documentation":TAG_DOCUMENTATION}


def plan_motorway_junction_inventory(city, bbox_wsen, *, bbox_source):
    """Full public city-box query: every junction/ref, no observed-ref filter."""
    box = list(bbox_wsen)
    if (not isinstance(city,str) or not city or len(box) != 4
            or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box)
            or not -180 <= box[0] < box[2] <= 180 or not -90 <= box[1] < box[3] <= 90
            or max(box[2]-box[0],box[3]-box[1]) > 1.):
        raise ValueError("numbered junction inventory needs the bounded full named-city box")
    if (not isinstance(bbox_source,dict) or bbox_source.get("city") != city
            or bbox_source.get("scope") != "named_city_bbox"
            or bbox_source.get("source_kind") not in {"public_geocode","public_city_inventory"}
            or not re.fullmatch(r"[0-9a-f]{64}",bbox_source.get("source_sha256", ""))):
        raise ValueError("named-city public bbox provenance is required")
    west,south,east,north = map(float,box)
    bbox = ",".join(f"{v:.17f}".rstrip("0").rstrip(".") for v in (south,west,north,east))
    query = f'[out:json][timeout:45];nwr["highway"="motorway_junction"]["ref"]({bbox});out center tags;'
    plan = {"schema":1,"city":city,"scope":"named_city_bbox","bbox_wsen":[west,south,east,north],
        "bbox_source":deepcopy(bbox_source),"endpoint":JUNCTION_ENDPOINT,"transport":"get","query":query,
        "query_sha256":hashlib.sha256(query.encode()).hexdigest(),"max_elements":10_000,"max_response_bytes":8*1024**2,
        "coverage":"all motorway_junction refs in the full public city bbox; no OCR-ref filtering"}
    plan["plan_sha256"] = _digest(plan)
    return plan


def build_motorway_junction_inventory(response_body, plan):
    """Parse one complete public response in memory, preserving all map refs."""
    expected = plan_motorway_junction_inventory(plan["city"],plan["bbox_wsen"],bbox_source=plan["bbox_source"])
    if expected != plan: raise ValueError("motorway junction inventory plan changed")
    if not isinstance(response_body,bytes) or len(response_body) > plan["max_response_bytes"]:
        raise ValueError("motorway junction response exceeds its byte budget")
    def reject(value): raise ValueError("non-finite junction JSON constant")
    payload = json.loads(response_body.decode("utf-8-sig"),parse_constant=reject)
    if (not isinstance(payload,dict) or payload.get("remark") or not isinstance(payload.get("elements"),list)
            or len(payload["elements"]) > plan["max_elements"]):
        raise ValueError("numbered junction response is incomplete or over budget")
    by_id = {}
    for element in payload["elements"]:
        if (not isinstance(element,dict) or element.get("type") not in {"node","way","relation"}
                or isinstance(element.get("id"),bool) or not isinstance(element.get("id"),int) or element["id"] <= 0):
            raise ValueError("junction response has invalid OSM source IDs")
        tags = element.get("tags") or {}
        location = element if element["type"] == "node" else element.get("center") or {}
        if (not isinstance(tags,dict) or tags.get("highway") != "motorway_junction" or not isinstance(tags.get("ref"),str)
                or not isinstance(location,dict) or any(isinstance(location.get(k),bool)
                    or not isinstance(location.get(k),(int,float)) or not math.isfinite(location[k]) for k in ("lat","lon"))
                or abs(location["lat"]) > 90 or abs(location["lon"]) > 180):
            raise ValueError("junction response lacks expected tags/finite coordinates")
        key = f"{element['type']}/{element['id']}"
        record = {"id":key,"lat":float(location["lat"]),"lon":float(location["lon"]),
                  "ref":tags["ref"],"normalized_refs":_refs(tags["ref"]),"tags":deepcopy(tags)}
        if key in by_id and by_id[key] != record: raise ValueError("conflicting duplicate junction ID")
        by_id[key] = record
    junctions = [by_id[key] for key in sorted(by_id)]
    return {"schema":1,"status":"complete","city":plan["city"],"scope":"named_city_bbox",
        "plan":deepcopy(plan),"response_sha256":hashlib.sha256(response_body).hexdigest(),
        "junctions":junctions,"snapshot_sha256":_digest(junctions)}


def match_motorway_junctions(clues, inventory):
    """Keep every exact semicolon-ref match; never select a camera fix.

    Route numbers, signposted destinations and directions are context only.
    They are not interchangeable with an exit's OSM ref/name or camera yaw.
    """
    if (clues.get("config_sha256") != _digest(clues.get("config"))
            or clues.get("observations_sha256") != _digest(clues.get("observations"))
            or clues.get("selected_answer") is not None):
        raise ValueError("frozen road-sign observations/config changed")
    if not isinstance(inventory,dict) or inventory.get("status") != "complete":
        return {"status":"unavailable","selected_answer":None,"camera_fix":None,
                "reason":"complete_public_junction_inventory_required","observations":deepcopy(clues["observations"])}
    junctions = inventory.get("junctions")
    if (not isinstance(junctions,list) or inventory.get("scope") != "named_city_bbox"
            or inventory.get("snapshot_sha256") != _digest(junctions)
            or len(junctions) > 10_000 or len({row["id"] for row in junctions}) != len(junctions)):
        raise ValueError("public junction inventory is invalid or changed")
    by_ref = defaultdict(list)
    for junction in junctions:
        for ref in junction["normalized_refs"]: by_ref[ref].append(junction)
    if sum(len(by_ref.get(observation["ref"],[])) for observation in clues["observations"]) > MAX_MATCH_CANDIDATES:
        raise ValueError("road-sign map-candidate budget exceeded before candidate copies")
    matches = []
    for observation in clues["observations"]:
        if _ref(observation["ref"]) != observation["ref"]:
            raise ValueError("observed junction reference is not a literal normalized value")
        candidates = deepcopy(by_ref.get(observation["ref"],[]))
        matches.append({"observation":deepcopy(observation),"candidates":candidates,
                        "camera_fix":None,"constraint":"signposted exit target only; no camera-at-exit assumption"})
    return {"status":"complete","experimental":True,"selected_answer":None,"camera_fix":None,
        "city":inventory["city"],"inventory_sha256":inventory["snapshot_sha256"],
        "observations_sha256":clues["observations_sha256"],"matches":matches,
        "selection":"none; all independent per-observation candidate sets retained",
        "context":deepcopy(clues["context"]),"tag_documentation":TAG_DOCUMENTATION}
