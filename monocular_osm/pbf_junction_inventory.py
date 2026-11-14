"""Offline, bounded OSM node-only junction inventories with explicit scope.

No download, OCR, observed-ref filter, route or GT input. The caller supplies
hash-verified public extract and city-boundary provenance. This is not a
retail inventory and never claims completeness for ways or relations.
"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import warnings
from types import MappingProxyType

from .road_sign_clues import _refs, MAX_MATCH_CANDIDATES

MAX_NODES = 10_000
GDAL_OPTIONS = MappingProxyType({"OSM_MAX_TMPFILE_SIZE":"32","OGR_INTERLEAVED_READING":"NO",
    "OSM_INDEX_POINTS":"NO","OSM_USE_POINTS_INDEX":"NO","OSM_INDEX_WAYS":"NO","OSM_USE_WAYS_INDEX":"NO"})
SQL_WHERE = "highway = 'motorway_junction' AND ref IS NOT NULL"
OSM_CONFIG = """[general]
tags_format=json
[points]
osm_id=yes
attributes=highway,ref,name
all_tags=yes
other_tags=no
"""
_LOCK = threading.RLock()


def _hash_string(value):return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block:=handle.read(1024**2):digest.update(block)
    return digest.hexdigest()


def _write(path,value):
    temporary=path.with_suffix(".part.json")
    try:
        temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def parse_extract_poly(body):
    """Parse the public Geofabrik .poly selection footprint, including holes."""
    from shapely.geometry import Polygon, mapping
    from shapely.ops import unary_union
    if not isinstance(body,bytes) or len(body)>2*1024**2:raise ValueError("extract polygon body is missing or too large")
    lines=body.decode("utf-8-sig").splitlines()
    outers,holes,points,is_hole=[],[],None,False
    count=0;terminated=False
    for line in lines[1:]:
        line=line.strip()
        if not line:continue
        if points is None:
            if line=="END":terminated=True;break
            is_hole=line.startswith("!");points=[];continue
        if line=="END":
            if len(points)<4 or points[0]!=points[-1]:raise ValueError("extract polygon ring is incomplete")
            ring=Polygon(points)
            if not ring.is_valid:raise ValueError("extract polygon ring is invalid")
            (holes if is_hole else outers).append(ring);points=None;continue
        xy=list(map(float,line.split()))
        if len(xy)!=2 or not all(math.isfinite(v) for v in xy) or abs(xy[0])>180 or abs(xy[1])>90:
            raise ValueError("extract polygon has invalid WGS84 vertices")
        points.append(tuple(xy));count+=1
        if count>100_000:raise ValueError("extract polygon vertex budget exceeded")
    if points is not None or not outers or not terminated:raise ValueError("extract polygon is incomplete")
    geometry=unary_union(outers).difference(unary_union(holes))
    if geometry.is_empty or not geometry.is_valid:raise ValueError("extract selection polygon is empty or invalid")
    return {"geometry":json.loads(json.dumps(mapping(geometry))),"source_sha256":hashlib.sha256(body).hexdigest()}


def plan_pbf_junction_inventory(city,bbox_wsen,*,bbox_source,extract_source,extract_polygon,city_geometry=None):
    """Freeze complete public scope, all-ref node predicate and parser bounds."""
    from shapely.geometry import box,shape,mapping
    bbox=list(bbox_wsen)
    if (len(bbox)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in bbox)
            or not -180<=bbox[0]<bbox[2]<=180 or not -90<=bbox[1]<bbox[3]<=90
            or max(bbox[2]-bbox[0],bbox[3]-bbox[1])>1.):raise ValueError("public city bbox is invalid or unbounded")
    if (not city or bbox_source.get("city")!=city or bbox_source.get("scope")!="named_city_bbox"
            or bbox_source.get("source_kind") not in {"public_geocode","public_city_inventory"}
            or not _hash_string(bbox_source.get("source_sha256"))):raise ValueError("public city bbox identity is required")
    if (extract_source.get("verified") is not True or not _hash_string(extract_source.get("sha256"))
            or isinstance(extract_source.get("bytes"),bool) or not isinstance(extract_source.get("bytes"),int)
            or extract_source["bytes"]<1 or any(not isinstance(extract_source.get(key),str) or not extract_source[key]
                for key in ("provider","source_url","snapshot_date","license"))):raise ValueError("verified public extract identity is required")
    if not _hash_string(extract_polygon.get("source_sha256")):raise ValueError("public extract footprint identity is required")
    region=shape(extract_polygon["geometry"]);bounding=box(*bbox)
    scope=shape(city_geometry) if city_geometry is not None else bounding
    if (region.is_empty or not region.is_valid or region.geom_type not in {"Polygon","MultiPolygon"}
            or scope.is_empty or not scope.is_valid
            or scope.geom_type not in {"Polygon","MultiPolygon"} or not bounding.covers(scope)):
        raise ValueError("public city geometry/extract coverage is invalid")
    covered=region.covers(scope)
    plan={"schema":1,"city":city,"bbox_wsen":bbox,"bbox_source":deepcopy(bbox_source),
        "scope":"named_city_admin_polygon_junction_nodes" if city_geometry is not None else "named_city_bbox_junction_nodes",
        "scope_geometry":mapping(scope),"extract_source":deepcopy(extract_source),"extract_polygon":deepcopy(extract_polygon),
        "complete_public_scope_coverage":bool(covered),"complete_bbox_coverage":bool(region.covers(bounding)),
        "scope_area_fraction_in_extract":min(1.,float(scope.intersection(region).area/scope.area)),
        "node_only":True,"ways_relations_included":False,"retail_inventory":False,
        "layer":"points","where":SQL_WHERE,"max_nodes":MAX_NODES,"read_limit":MAX_NODES+1,
        "driver_options":{"TAGS_FORMAT":"JSON","MAX_TMPFILE_SIZE":32,"INTERLEAVED_READING":"NO"},
        "gdal_config_options":dict(GDAL_OPTIONS),
        "osm_config_sha256":hashlib.sha256(OSM_CONFIG.encode()).hexdigest(),
        "reading_strategy":"pyogrio OSM SET interest_layers=points; other layers ignored; no way geometry completeness claim"}
    plan=json.loads(json.dumps(plan))
    plan["plan_sha256"]=_digest(plan)
    return plan


@contextmanager
def _gdal_options(temporary):
    import pyogrio
    options={"CPL_TMPDIR":str(Path(temporary).resolve()),**GDAL_OPTIONS}
    with _LOCK:
        previous={key:pyogrio.get_gdal_config_option(key) for key in options}
        pyogrio.set_gdal_config_options(options)
        try:yield
        finally:pyogrio.set_gdal_config_options(previous)


def _validate_plan(plan):
    kind=plan.get("scope")
    if kind not in {"named_city_admin_polygon_junction_nodes","named_city_bbox_junction_nodes"}:
        raise ValueError("unknown public node inventory scope")
    expected=plan_pbf_junction_inventory(plan["city"],plan["bbox_wsen"],bbox_source=plan["bbox_source"],
        extract_source=plan["extract_source"],extract_polygon=plan["extract_polygon"],
        city_geometry=plan["scope_geometry"] if kind=="named_city_admin_polygon_junction_nodes" else None)
    if expected!=plan:raise ValueError("fixed offline junction inventory plan changed")


def _node_identifier(value):
    import operator
    import numpy as np
    if isinstance(value,(bool,np.bool_)):raise ValueError("boolean is not an OSM node identifier")
    if isinstance(value,str):
        if not re.fullmatch(r"[1-9][0-9]*",value):raise ValueError("OSM node ID must retain its exact integer/string type")
        result=int(value)
    else:
        try:result=operator.index(value)
        except TypeError:raise ValueError("floating OSM IDs would lose source identity") from None
    if result<=0 or result>2**63-1:raise ValueError("OSM node ID is outside its signed64-bit domain")
    return result


def _normalize_rows(meta,geometries,fields,plan):
    from shapely import from_wkb
    from shapely.geometry import shape
    names=list(meta["fields"])
    if (meta.get("crs")!="EPSG:4326" or meta.get("geometry_type")!="Point"
            or len(names)!=4 or set(names)!={"osm_id","highway","ref","all_tags"} or len(fields)!=4):
        raise ValueError("OSM point reader returned an incompatible typed schema/CRS")
    columns=dict(zip(names,fields));n=len(columns["osm_id"])
    if n>plan["max_nodes"]:raise ValueError("matching node budget exceeded; truncated reads are not inventories")
    if any(len(column)!=n for column in fields) or geometries is None or len(geometries)!=n:
        raise ValueError("OSM point geometry/field rows disagree")
    city=shape(plan["scope_geometry"]);west,south,east,north=plan["bbox_wsen"]
    nodes={};seen={};excluded=0
    for index in range(n):
        identifier=_node_identifier(columns["osm_id"][index])
        raw=columns["all_tags"][index]
        if not isinstance(raw,str):raise ValueError("OSM all_tags must be a JSON string, not HSTORE or missing")
        tags=json.loads(raw)
        if (not isinstance(tags,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in tags.items())
                or tags.get("highway")!="motorway_junction" or not isinstance(tags.get("ref"),str)
                or tags["highway"]!=columns["highway"][index] or tags["ref"]!=columns["ref"][index]):
            raise ValueError("JSON all_tags and promoted highway/ref fields disagree")
        point=from_wkb(geometries[index])
        if point is None or point.is_empty or point.geom_type!="Point" or not math.isfinite(point.x) or not math.isfinite(point.y):
            raise ValueError("OSM node geometry is not a finite WGS84 point")
        if not west-1e-9<=point.x<=east+1e-9 or not south-1e-9<=point.y<=north+1e-9:
            raise ValueError("OSM point reader returned a node outside the full query bbox")
        record={"id":f"node/{identifier}","lat":point.y,"lon":point.x,"ref":tags["ref"],
                "normalized_refs":_refs(tags["ref"]),"tags":tags}
        if identifier in seen and seen[identifier]!=record:raise ValueError("conflicting duplicate source node ID")
        seen[identifier]=record
        if not city.covers(point):excluded+=1;continue
        nodes[identifier]=record
    return [nodes[key] for key in sorted(nodes)],{"bbox_node_rows":n,"outside_public_scope_rows":excluded,
                                                "unique_scope_nodes":len(nodes)}


def read_city_junction_nodes(path,plan,output_dir,temp_dir):
    """Read every matching node once; cap+1 detects truncation, warnings fail.

    pyogrio's OSM layer selector declares ONLY points as an interest layer.
    Other four layers and node/way geometry indexes are explicitly unused;
    this is the documented ignore-other-layers alternative to interleaving.
    """
    import pyogrio
    from pyogrio.raw import read
    _validate_plan(plan)
    path,output,temp=Path(path).resolve(strict=True),Path(output_dir),Path(temp_dir)
    if output.exists() and any(output.iterdir()):raise FileExistsError("offline node output must be new; no re-blessing")
    if path.stat().st_size!=plan["extract_source"]["bytes"] or _sha(path)!=plan["extract_source"]["sha256"]:
        raise ValueError("public extract bytes/hash changed before parse")
    if pyogrio.list_drivers().get("OSM") not in {"r","rw"}:raise RuntimeError("installed GDAL has no OSM read driver")
    output.mkdir(parents=True,exist_ok=True);temp.mkdir(parents=True,exist_ok=True)
    config=output/"node_osmconf.ini";config.write_bytes(OSM_CONFIG.encode("utf-8"))
    _write(output/"parser_plan.json",plan)  # Full city scope and all-ref predicate before reading.
    try:
        with _gdal_options(temp),warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            meta,_,geometry,fields=read(path,layer="points",columns=["osm_id","highway","ref","all_tags"],
                read_geometry=True,force_2d=True,skip_features=0,max_features=plan["read_limit"],
                where=SQL_WHERE,bbox=tuple(plan["bbox_wsen"]),CONFIG_FILE=str(config.resolve()),**plan["driver_options"])
            if captured:raise RuntimeError("GDAL node reader warning: "+"; ".join(str(item.message) for item in captured))
        nodes,counts=_normalize_rows(meta,geometry,fields,plan)
        if _sha(path)!=plan["extract_source"]["sha256"]:raise ValueError("public extract changed during node read")
        inventory={"schema":1,"status":"complete" if plan["complete_public_scope_coverage"] else "partial",
            "city":plan["city"],"scope":plan["scope"],"complete_within_extract":True,
            "complete_for_public_city_scope":plan["complete_public_scope_coverage"],
            "complete_parent_bbox_coverage":plan["complete_bbox_coverage"],"node_only":True,
            "ways_relations_included":False,"retail_inventory":False,"junctions":nodes,"counts":counts,
            "snapshot_sha256":_digest(nodes),"plan":deepcopy(plan),"plan_sha256":plan["plan_sha256"],
            "runtime":{"pyogrio":pyogrio.__version__,"gdal":list(pyogrio.__gdal_version__),
                "OSM_driver":pyogrio.list_drivers()["OSM"],"strategy":"points-only declared-interest; other layers ignored",
                "tag_format":"JSON all_tags","sqlite_memory_limit_mb":32,"temp_dir":str(temp.resolve())}}
        _write(output/"junction_inventory.json",inventory)
        return inventory
    except Exception as error:
        _write(output/"failed.json",{"inventory_available":False,"error":f"{type(error).__name__}: {error}"})
        raise


def match_junction_targets(clues,inventory):
    """All exact target nodes in the declared scope; never a camera fix."""
    if (clues.get("observations_sha256")!=_digest(clues.get("observations"))
            or inventory.get("snapshot_sha256")!=_digest(inventory.get("junctions"))
            or inventory.get("complete_within_extract") is not True):raise ValueError("frozen node/sign evidence changed")
    by_ref={}
    for node in inventory["junctions"]:
        for ref in node["normalized_refs"]:by_ref.setdefault(ref,[]).append(node)
    total=sum(len(by_ref.get(row["ref"],[])) for row in clues["observations"])
    if total>MAX_MATCH_CANDIDATES:raise ValueError("node target candidate budget exceeded before copies")
    return {"status":inventory["status"],"city":inventory["city"],"scope":inventory["scope"],
        "complete_for_public_city_scope":inventory["complete_for_public_city_scope"],
        "complete_parent_bbox_coverage":inventory["complete_parent_bbox_coverage"],
        "selected_answer":None,"camera_fix":None,"inventory_sha256":inventory["snapshot_sha256"],
        "observations_sha256":clues["observations_sha256"],
        "matches":[{"observation":deepcopy(row),"candidates":deepcopy(by_ref.get(row["ref"],[])),
                    "constraint":"signposted target, not camera position or exit taken"} for row in clues["observations"]]}
