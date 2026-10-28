"""Explicit opt-in registration; legacy versions/defaults remain independent."""
import pytest
from monocular_osm import assisted_policy as v1, assisted_policy_v2 as v2, assisted_policy_v3 as v3
from monocular_osm.assisted_profiles import resolve_policy
from monocular_osm.cli import build_arg_parser
from monocular_osm.pipeline import PipelineConfig


@pytest.mark.parametrize("profile,policy", [("v1", v1), ("v2", v2), ("v3", v3)])
def test_exact_profile_resolution_and_cli_configuration(tmp_path, profile, policy):
    args = build_arg_parser().parse_args(["--video", "video.mp4", "--city", "Example",
                                          "--assisted-only", "--assisted-profile", profile])
    assert args.answer == "assisted" and args.assisted_profile == profile
    cfg = PipelineConfig(url="", city="Example", data_dir=tmp_path / "data",
                         output_dir=tmp_path / "output", answer="assisted", assisted_profile=profile)
    assert cfg.assisted_profile == profile and cfg.frame_stride is None
    assert resolve_policy(profile) is policy


def test_old_default_and_constants_stay_v1(tmp_path):
    args = build_arg_parser().parse_args(["--video", "video.mp4", "--city", "Example", "--assisted-only"])
    cfg = PipelineConfig(url="", city="Example", data_dir=tmp_path / "data", output_dir=tmp_path / "output")
    assert args.assisted_profile == cfg.assisted_profile == "v1"
    assert v1.policy_config()["baseline_median_min"] == v2.policy_config()["baseline_median_min"] == v3.policy_config()["baseline_median_min"] == .162
    assert len({v1.CONFIG_SHA256, v2.CONFIG_SHA256, v3.CONFIG_SHA256}) == 3
    assert v3.policy_config()["metadata_union_primary"] == "reference_mean"


@pytest.mark.parametrize("profile", ["latest", "V3", "v4", "", None])
def test_unknown_profiles_refuse_without_fallback(tmp_path, profile):
    with pytest.raises(ValueError):
        resolve_policy(profile)
    with pytest.raises(ValueError):
        PipelineConfig(url="", city="Example", data_dir=tmp_path / "data",
                       output_dir=tmp_path / "output", assisted_profile=profile)


@pytest.mark.parametrize("kwargs", [{"max_frames": 80}, {"frame_max_width": 1280},
    {"osm_around": (49., 8., 1000.)}, {"vpr_query_mode": "crops"}, {"ocr_engine": "easyocr"},
    {"graph_filter": True}, {"ground_truth_streets": ("label",)}])
def test_v3_keeps_registered_input_restrictions(tmp_path, kwargs):
    with pytest.raises(ValueError):
        PipelineConfig(url="", city="Example", data_dir=tmp_path / "data", output_dir=tmp_path / "output",
                       answer="assisted", assisted_profile="v3", **kwargs)
