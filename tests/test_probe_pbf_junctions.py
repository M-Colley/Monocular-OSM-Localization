"""Download ledger/body bounds only; no actual network or large extract."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/"scripts/probe_pbf_junctions.py"
module_spec=importlib.util.spec_from_file_location("pbf_probe_test",SCRIPT)
probe=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(probe)


def _spec(tmp_path,body=b"public-map-bytes"):
    value={"provider":"Geofabrik","region":"Region","source_url":"https://download.geofabrik.de/region-260101.osm.pbf",
        "expected_bytes":len(body),"expected_md5":hashlib.md5(body).hexdigest(),"snapshot_date":"2026-01-01",
        "license":"https://opendatacommons.org/licenses/odbl/1-0/"}
    file=tmp_path/"official.md5"
    file.write_text(value["expected_md5"]+"  region-260101.osm.pbf\n",encoding="ascii")
    value["md5_response_sha256"]=probe.sha(file)
    return value,file


def _response(monkeypatch,body,status=200):
    import requests
    calls=[]
    class Response:
        status_code=status
        headers={"Content-Length":str(len(body))}
        def __enter__(self):return self
        def __exit__(self,*args):calls.append("closed")
        def iter_content(self,chunk_size):yield body[:5];yield body[5:]
    def get(url,**kwargs):
        assert kwargs=={"timeout":(8,30),"stream":True,"allow_redirects":False}
        calls.append(url);return Response()
    monkeypatch.setattr(requests,"get",get)
    return calls


def test_one_official_dated_body_is_size_md5_and_sha_verified_before_publish(tmp_path,monkeypatch):
    body=b"public-map-bytes";spec,md5=_spec(tmp_path,body)
    probe.validate_download_spec(spec,"Synthetic City, Region",md5)
    calls=_response(monkeypatch,body)
    path,ledger=probe.download_extract(spec,tmp_path)
    assert path.read_bytes()==body and ledger["verified"]
    assert ledger["sha256"]==hashlib.sha256(body).hexdigest() and ledger["md5"]==spec["expected_md5"]
    assert len(calls)==2 and calls[-1]=="closed"
    with pytest.raises(FileExistsError):probe.download_extract(spec,tmp_path)


@pytest.mark.parametrize("fault",["short","large","bad_md5","http"])
def test_unverified_or_partial_extract_never_gets_a_final_pbf_or_ledger(tmp_path,monkeypatch,fault):
    spec,_=_spec(tmp_path)
    body=b"public-map-bytes"
    if fault=="short":body=body[:-1]
    elif fault=="large":body+=b"extra"
    elif fault=="bad_md5":spec["expected_md5"]="a"*32
    calls=_response(monkeypatch,body,status=504 if fault=="http" else 200)
    with pytest.raises((ValueError,RuntimeError)):probe.download_extract(spec,tmp_path)
    assert not(tmp_path/"region-260101.osm.pbf").exists() and not(tmp_path/"extract_ledger.json").exists()
    assert len([call for call in calls if call.startswith("https://")])==1


@pytest.mark.parametrize("fault",["wrong_region","moving_latest","other_host","changed_md5"])
def test_region_source_date_and_official_digest_are_frozen_before_data_request(tmp_path,fault):
    spec,md5=_spec(tmp_path)
    if fault=="wrong_region":spec["region"]="Different State"
    elif fault=="moving_latest":spec["source_url"]="https://download.geofabrik.de/region-latest.osm.pbf"
    elif fault=="other_host":spec["source_url"]="https://example.test/region-260101.osm.pbf"
    else:md5.write_text("changed",encoding="ascii")
    with pytest.raises(ValueError):probe.validate_download_spec(spec,"Synthetic City, Region",md5)
