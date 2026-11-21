"""The ordinary benchmark transports explicit profiles without label inputs."""
from copy import deepcopy
import pytest
from tests.test_benchmark import M
from monocular_osm.cli import build_arg_parser
from monocular_osm.assisted_comparison import SCHEMA

ENTRY={"id":"synthetic-dev","video":{"path":"example.mp4"},"city":"Example",
       "vo_segment":"10:20","gt":"ground_truth/never_opened.json"}

@pytest.mark.parametrize("profile",["v1","v2","v3"])
def test_standard_benchmark_explicit_profile_preserves_registered_inputs(tmp_path,profile):
    entry=deepcopy(ENTRY)
    extra=M.check_passthrough(["--assisted-only",f"--assisted-profile={profile}"])
    argv=M.build_pipeline_argv(entry,city=entry["city"],output_dir=tmp_path/"out",data_dir=tmp_path/"data",extra=extra)
    args=build_arg_parser().parse_args(argv[3:])
    assert args.answer=="assisted" and args.assisted_profile==profile
    assert argv[argv.index("--city")+1]==entry["city"]
    assert argv[argv.index("--vo-segment")+1]==entry["vo_segment"]
    assert entry["gt"] not in argv and entry==ENTRY
    M.assert_leak_free(argv,entry)

@pytest.mark.parametrize("override",["--city=Other","--vo-segment=0:99","--ground-truth-waypoints=labels.json"])
def test_new_profile_cannot_override_registered_or_label_boundaries(override):
    with pytest.raises((SystemExit,M.LeakError)):
        M.check_passthrough(["--assisted-only","--assisted-profile=v3",override])

def test_benchmark_preserves_truthful_full_decoder_proof_and_selected_route():
    comparison={"schema":SCHEMA,
        "actual_decoder":{"family":"full_sequence","variant":"reference_mean","candidate_pruning":False},
        "raw_is_diagnostic_only":True}
    value={"provenance":{"config":{"answer":"assisted","assisted_profile":"v3"},
        "assisted_inputs":{"policy":{"version":"assisted-policy-posthoc-dev-v3"}},
        "assisted_audit":{"selected_candidate":"expanded","reached_stages":[{"kind":"expanded","comparison":comparison}]}},
        "answer":{"mode":"assisted","route_latlon":[[42.,-83.],[42.001,-83.]]}}
    result=M._provenance_of(value)
    assert result["config"]["assisted_profile"]=="v3"
    assert result["assisted_audit"]==value["provenance"]["assisted_audit"]
    assert result["assisted_selected_answer"]["route_sha256"]==M._fingerprint(value["answer"]["route_latlon"])
