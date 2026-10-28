from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from monocular_osm.assisted_policy import decide_assisted_policy
from monocular_osm.kartaview_vpr import HUB_MEGALOC
from monocular_osm.stages import assisted as stage
from monocular_osm.stages.context import RunContext
from monocular_osm.provenance import file_sha256


def candidate(score=.17):
    route = [[49.001 + i*.001, 8.4 + i*.001] for i in range(4)]
    return {"status": "complete", "route_latlon": route, "route_timestamps": [0., 1., 2., 3.],
        "comparison": {"statistic": stage.STATISTIC,
            "model": {"name": "megaloc", "hub": HUB_MEGALOC, "weights_sha256": "1"*64, "device": "cpu"},
            "query": {"indices": [0, 1, 2, 3], "times": [0., 1., 2., 3.], "mode": "warp",
                      "preprocessing": {"size": [512, 512]}, "pixels_sha256": "2"*64, "descriptors_sha256": "3"*64},
            "inner_decoder": {"actual": "viterbi", "fallback": False, "settings": {"free_speed": 40}},
            "decoded_reference_cosines": [score]*4, "decoded_reference_latlon": route}}


def run(tmp_path, responses):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    cfg = SimpleNamespace(output_dir=tmp_path / "output", city="Example", vo_start_sec=0., vo_end_sec=3.)
    cfg.output_dir.mkdir()
    ctx = RunContext(cfg=cfg, fusion_w=None, video_path=video)
    calls = []
    def child(kind, ctx):
        calls.append(kind)
        response = responses[kind]
        if isinstance(response, Exception):
            raise response
        payload = deepcopy(response)
        payload.update(video_sha256=ctx.result["assisted_inputs"]["video"]["sha256"],
                       window=ctx.result["assisted_inputs"]["registered_window"],
                       source_sha256=ctx.result["assisted_inputs"]["source_sha256"])
        directory = cfg.output_dir / "fake_children" / kind
        directory.mkdir(parents=True)
        candidate_path, result_path, frozen_path = [directory / name for name in
            ("candidate.json", "result.json", "child_frozen.json")]
        candidate_path.write_text(json.dumps(payload))
        result_path.write_text(json.dumps({"result": "mocked acquired inputs"}))
        frozen_path.write_text(json.dumps({"scoring_has_not_run": True,
            "artifacts": {p.name: file_sha256(p) for p in (candidate_path, result_path)}}))
        return {**payload, "worker_returncode": 0, **{role: {"path": str(p), "sha256": file_sha256(p)}
            for role, p in (("candidate_artifact", candidate_path), ("result_artifact", result_path),
                            ("frozen_artifact", frozen_path))}}
    def finalize(ctx):
        if "assisted_position" in ctx.result:
            ctx.result["answer"] = deepcopy(ctx.result["assisted_position"])
        (cfg.output_dir / "result.json").write_text(json.dumps(ctx.result))
    return stage.run_assisted(ctx, finalize=finalize, run_child=child), calls


def test_strong_baseline_exact_route_without_future_work(tmp_path):
    baseline = candidate()
    result, calls = run(tmp_path, {"baseline": baseline})
    assert calls == ["baseline"]
    assert result["answer"]["route_latlon"] == baseline["route_latlon"]
    assert result["answer"]["route_t_sec"] == baseline["route_timestamps"]


def test_partial_joint_never_replaces_baseline_or_runs_unrequested_pf(tmp_path):
    baseline, partial = candidate(.12), candidate(.99)
    partial.update(status="partial", retained_branch_count=2, successful_branch_count=1)
    cues = {"status": "complete", "cues": {"duration_s": 50., "vo": {"status": "complete", "valid_fraction": .9},
                                            "ocr": {"status": "complete", "clue_branch_count": 2}}}
    result, calls = run(tmp_path, {"baseline": baseline, "cues": cues, "poi_joint": partial,
                                  "expanded": RuntimeError("reference outage")})
    assert calls == ["baseline", "cues", "poi_joint", "expanded"]
    assert result["answer"]["route_latlon"] == baseline["route_latlon"]
    assert result["assisted_audit"]["evidence"]["poi_joint"]["route_valid"] is False
    audit = result["assisted_audit"]
    assert [decide_assisted_policy(e) for e in audit["decision_inputs"]] == audit["decision_trace"]


@pytest.mark.parametrize("fault", ["decoder", "model", "crops"])
def test_mismatched_expansion_is_declined(tmp_path, fault):
    expanded = candidate(.9)
    if fault == "decoder":
        expanded["comparison"]["inner_decoder"].update(actual="argmax", fallback=True)
    elif fault == "model":
        expanded["comparison"]["model"]["weights_sha256"] = "4"*64
    else:
        expanded["comparison"]["query"]["mode"] = "crops"
    result, calls = run(tmp_path, {"baseline": candidate(.12), "cues": {"status": "failed"}, "expanded": expanded})
    assert calls == ["baseline", "cues", "expanded"]
    assert result["assisted_audit"]["selected_candidate"] == "baseline"


def test_unrounded_decoded_cosine_drives_cutoff(tmp_path):
    result, calls = run(tmp_path, {"baseline": candidate(.16196), "cues": {"status": "failed"},
                                  "expanded": {"status": "unavailable"}})
    assert calls == ["baseline", "cues", "expanded"]
    assert result["assisted_audit"]["evidence"]["baseline"]["similarity_median"] == .16196


def test_actual_worker_module_entrypoint_defines_helpers_before_main():
    import os
    import subprocess
    import sys
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1"}
    completed = subprocess.run([sys.executable, "-m", "monocular_osm.assisted_worker", "--check-runtime"],
                               capture_output=True, text=True, env=env, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["helpers_ready"] is True
