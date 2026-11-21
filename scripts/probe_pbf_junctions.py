"""One bounded dated public extract, offline city-node inventory, frozen clues.

No Overpass, new OCR/VO/model/reference work or GT inputs are accepted.
Scope is the public named-city ADMIN polygon, distinct from its enclosing box.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse

os.environ.update(CUDA_VISIBLE_DEVICES="-1",PROJ_NETWORK="OFF",PYTHONDONTWRITEBYTECODE="1")
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="2"
sys.dont_write_bytecode=True


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block:=handle.read(1024**2):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    temporary=path.with_suffix(".part.json")
    try:
        temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def validate_download_spec(spec,city,md5_file):
    url=urlparse(spec["source_url"])
    dated=dt.date.fromisoformat(spec["snapshot_date"]).strftime("%y%m%d")
    if (url.scheme!="https" or url.hostname!="download.geofabrik.de" or url.username or url.query or url.fragment
            or not url.path.endswith(f"-{dated}.osm.pbf") or spec["region"].casefold() not in city.casefold()
            or isinstance(spec["expected_bytes"],bool) or not isinstance(spec["expected_bytes"],int)
            or not 1<=spec["expected_bytes"]<=2*1024**3 or not re.fullmatch(r"[0-9a-f]{32}",spec["expected_md5"])):
        raise ValueError("download must be one bounded official dated extract from the user city's region")
    parts=Path(md5_file).read_text(encoding="ascii").split()
    if (sha(md5_file)!=spec["md5_response_sha256"] or len(parts)!=2 or parts[0]!=spec["expected_md5"]
            or parts[1].lstrip("*")!=Path(url.path).name):raise ValueError("official dated MD5 ledger changed")


def download_extract(spec,output):
    """One attempt, exact bytes+MD5, SHA ledger; failures keep the partial file."""
    import requests
    filename=Path(urlparse(spec["source_url"]).path).name
    path=output/filename;partial=path.with_suffix(path.suffix+".part")
    if path.exists() or partial.exists():raise FileExistsError("public extract output is never overwritten/resumed")
    started=time.monotonic();count=0;md5=hashlib.md5();digest=hashlib.sha256()
    with requests.get(spec["source_url"],timeout=(8,30),stream=True,allow_redirects=False) as response:
        if response.status_code!=200:raise RuntimeError(f"public extract HTTP {response.status_code}")
        with partial.open("xb") as handle:
            for block in response.iter_content(chunk_size=1024**2):
                if time.monotonic()-started>600:raise TimeoutError("public extract exceeded its declared download duration")
                if count+len(block)>spec["expected_bytes"]:raise ValueError("public extract body exceeded its declared bytes")
                handle.write(block);count+=len(block);md5.update(block);digest.update(block)
                write(output/"download_progress.json",{"bytes":count,"expected_bytes":spec["expected_bytes"],"elapsed_s":time.monotonic()-started})
        headers={key:response.headers.get(key) for key in ("Last-Modified","ETag","Content-Length")}
    if count!=spec["expected_bytes"] or md5.hexdigest()!=spec["expected_md5"]:
        raise ValueError("public extract is incomplete or its official MD5 does not match")
    partial.replace(path)
    ledger={**spec,"verified":True,"bytes":count,"md5":md5.hexdigest(),"sha256":digest.hexdigest(),
            "downloaded_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"elapsed_s":time.monotonic()-started,"HTTP_headers":headers}
    write(output/"extract_ledger.json",ledger)
    return path,ledger


def _freeze(output):
    write(output/"frozen_pbf_probe.json",{"scoring_has_not_run":True,"ground_truth_used":False,
        "torch_imported":"torch" in sys.modules,
        "artifacts":{str(p.relative_to(output)):sha(p) for p in output.rglob("*") if p.is_file() and p.name!="frozen_pbf_probe.json"}})


def infer(args):
    source=args.source_root.resolve(strict=True)
    if (subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()!=args.source_commit
            or subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True).strip()):
        raise ValueError("PBF probe needs its declared clean source pin")
    sys.path.insert(0,str(source))
    from monocular_osm import pbf_junction_inventory as pbf,road_sign_clues as signs
    if not Path(pbf.__file__).resolve().is_relative_to(source):raise ValueError("offline adapter escaped pinned source")
    spec_loader=importlib.util.spec_from_file_location("pbf_frozen_metadata_inputs",Path(__file__).with_name("probe_tiled_retail_inventory.py"))
    prior=importlib.util.module_from_spec(spec_loader);spec_loader.loader.exec_module(prior)
    manifest=read(args.inputs);ocr,bbox=prior.verified_inputs(manifest)
    download=read(args.extract_spec);validate_download_spec(download,manifest["city"],args.md5_file)
    if sha(args.extract_poly)!=args.extract_poly_sha256:raise ValueError("public extract polygon changed")
    footprint=pbf.parse_extract_poly(args.extract_poly.read_bytes())
    geocodes=read(manifest["bbox_response"]["path"])
    boundary=next(row for row in geocodes if row["osm_type"]==manifest["bbox_response"]["osm_type"] and row["osm_id"]==manifest["bbox_response"]["osm_id"])
    geometry=boundary["geojson"]
    from shapely.geometry import shape
    if not shape(footprint["geometry"]).covers(shape(geometry)):
        raise ValueError("selected extract does not cover the full public named-city ADMIN polygon")
    parser=args.parser_family.resolve(strict=True)
    if sha(parser/"frozen_parser.json")!=args.parser_freeze_sha256:raise ValueError("frozen all-clip parser family changed")
    for relative,expected in read(parser/"frozen_parser.json")["artifacts"].items():
        path=(parser/relative).resolve(strict=True)
        if not path.is_relative_to(parser) or sha(path)!=expected:raise ValueError("frozen road-sign parser artifact changed")
    clue=signs.extract_road_sign_clues(ocr["detections"],ocr["sampled_frames"][0]["shape"][:2])
    if clue!=read(parser/(Path(manifest["ocr"]["path"]).parent.name+".json")):
        raise ValueError("new parser differs from the frozen observations; do not choose rules from map results")
    output=args.output.resolve()
    if output.drive.upper()!="E:" or output.is_relative_to(source):raise ValueError("use a fresh isolated E data family")
    if shutil.disk_usage(output.parent).free<download["expected_bytes"]+1024**3:raise OSError("insufficient bounded extract/temp disk headroom")
    output.mkdir(parents=True,exist_ok=False)
    sources={}
    for name in ("pbf_junction_inventory.py","road_sign_clues.py"):
        path=output/("source_"+name);shutil.copyfile(source/"monocular_osm"/name,path);sources[name]=sha(path)
    shutil.copyfile(__file__,output/"probe_source.py")
    for name,path in (("extract.poly",args.extract_poly),("official.md5",args.md5_file),
        ("original_ocr.json",Path(manifest["ocr"]["path"])),("public_city_boundary.json",Path(manifest["bbox_response"]["path"]))):
        shutil.copyfile(path,output/name)
    write(output/"road_sign_observations.json",clue)
    write(output/"acquisition_plan.json",{"source_commit":args.source_commit,"source_files":sources,"runner_sha256":sha(__file__),
        "public_city":manifest["city"],"full_bbox_wsen":bbox,"admin_geometry":geometry,
        "bbox_source":manifest["bbox_response"],"extract_footprint_sha256":sha(args.extract_poly),
        "full_admin_scope_covered":True,"whole_bbox_coverage_claimed":False,
        "download_spec":download,"download_attempts":1,"download_HTTP_timeout_s":[8,30],"download_duration_limit_s":600,
        "input_manifest":manifest,"input_manifest_sha256":sha(args.inputs),"parser_freeze_sha256":args.parser_freeze_sha256,
        "node_query":{"layer":"points","where":pbf.SQL_WHERE,"observed_ref_filter":False,"max_rows":10001},
        "ground_truth_used":False,"new_ocr":False,"new_vo":False,"new_models":False,"new_references":False})
    import psutil
    process=psutil.Process();started=time.monotonic()
    result={"status":"unknown","selected_answer":None,"camera_fix":None}
    try:
        extract,ledger=download_extract(download,output)
        plan=pbf.plan_pbf_junction_inventory(manifest["city"],bbox,bbox_source={
            "city":manifest["city"],"scope":"named_city_bbox","source_kind":"public_geocode",
            "source_sha256":manifest["bbox_response"]["sha256"]},extract_source=ledger,extract_polygon=footprint,city_geometry=geometry)
        inventory=pbf.read_city_junction_nodes(extract,plan,output/"parser",output/"gdal_tmp")
        targets=pbf.match_junction_targets(clue,inventory)
        write(output/"all_junction_targets.json",targets)
        result.update(status=inventory["status"],scope=inventory["scope"],counts=inventory["counts"],
            complete_for_public_city_scope=inventory["complete_for_public_city_scope"],
            complete_parent_bbox_coverage=inventory["complete_parent_bbox_coverage"],
            targets=[{"ref":row["observation"]["ref"],"t_sec":row["observation"]["t_sec"],
                      "candidates":row["candidates"]} for row in targets["matches"]])
    except Exception as error:result["error"]=f"{type(error).__name__}: {error}"
    memory=process.memory_info()
    result["resources"]={"elapsed_s":time.monotonic()-started,"peak_rss_bytes":getattr(memory,"peak_wset",memory.rss),
                         "peak_private_bytes":getattr(memory,"peak_pagefile",memory.vms)}
    write(output/"summary.json",result)
    _freeze(output)
    print(json.dumps(result,indent=2,ensure_ascii=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True);parser.add_argument("--source-commit",required=True)
    parser.add_argument("--inputs",type=Path,required=True);parser.add_argument("--extract-spec",type=Path,required=True)
    parser.add_argument("--md5-file",type=Path,required=True);parser.add_argument("--extract-poly",type=Path,required=True)
    parser.add_argument("--extract-poly-sha256",required=True);parser.add_argument("--parser-family",type=Path,required=True)
    parser.add_argument("--parser-freeze-sha256",required=True);parser.add_argument("--output",type=Path,required=True)
    infer(parser.parse_args())
