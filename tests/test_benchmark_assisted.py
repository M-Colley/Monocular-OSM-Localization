"""Assisted diagnostics and cold-reference ownership on isolated fixtures."""
from pathlib import Path

import pytest

from tests.test_benchmark import M, _fake_result, _ll, _setup_root, _write_rep, GT_XY


@pytest.mark.parametrize("keep_arrays", [False, True])
def test_cold_repeats_cover_nested_assisted_reference_leaves(tmp_path, keep_arrays):
    slug = tmp_path / "data/submission"
    leaves = [slug / "assisted" / scope / slug.name for scope in ("native", "expanded")]
    external = tmp_path / "external"
    shared = external / slug.name / "mapillary_region_poi_123"
    for leaf in [slug, *leaves]:
        (leaf / "mapillary_regions/poi_123").mkdir(parents=True)
        (leaf / "mapillary_regions/poi_123/metadata.json").write_bytes(b"warm metadata")
        (leaf / "mapillary_regions/poi_123/pixels.npz").write_bytes(b"warm pixels")
        (leaf / "trajectory_v3.npz").write_bytes(b"VO stays warm")
    shared.mkdir(parents=True)
    (shared / "pixels.npz").write_bytes(b"external warm")
    unrelated = slug / "assisted/native/different_submission/mapillary"
    unrelated.mkdir(parents=True)
    (unrelated / "pixels.npz").write_bytes(b"other clip")
    for rep in range(2):
        M.set_vpr_caches_aside(slug, region_cache_root=external)
        for leaf in [slug, *leaves]:
            assert not (leaf / "mapillary_regions").exists()
            assert (leaf / "trajectory_v3.npz").read_bytes() == b"VO stays warm"
            (leaf / "mapillary_regions/poi_123").mkdir(parents=True)
            (leaf / "mapillary_regions/poi_123/metadata.json").write_bytes(b"fresh metadata")
            (leaf / "mapillary_regions/poi_123/pixels.npz").write_bytes(b"fresh pixels")
        assert not shared.exists()
        shared.mkdir()
        (shared / "pixels.npz").write_bytes(b"external fresh")
        archive = slug / M.COLD_ARCHIVE_DIR / "run/clip" / f"rep{rep}"
        M.restore_vpr_caches(slug, archive, keep_arrays=keep_arrays, region_cache_root=external)
        for leaf in [slug, *leaves]:
            assert (leaf / "mapillary_regions/poi_123/pixels.npz").read_bytes() == b"warm pixels"
            saved = leaf / archive.relative_to(slug) / "mapillary_regions/poi_123"
            assert (saved / "metadata.json").read_bytes() == b"fresh metadata"
            assert (saved / "pixels.npz").exists() is keep_arrays
        assert (shared / "pixels.npz").read_bytes() == b"external warm"
        assert (unrelated / "pixels.npz").read_bytes() == b"other clip"


def test_restore_finds_child_reference_leaves_created_by_first_cold_run(tmp_path):
    slug = tmp_path / "data/submission"
    slug.mkdir(parents=True)
    M.set_vpr_caches_aside(slug)
    leaf = slug / "assisted/expanded" / slug.name
    (leaf / "mapillary_city/coarse").mkdir(parents=True)
    (leaf / "mapillary_city/coarse/refs.json").write_bytes(b"newly acquired")
    archive = slug / M.COLD_ARCHIVE_DIR / "run/clip/rep0"
    M.restore_vpr_caches(slug, archive)
    assert not (leaf / "mapillary_city").exists()
    assert (leaf / archive.relative_to(slug) / "mapillary_city/coarse/refs.json").read_bytes() == b"newly acquired"


def test_nested_escape_is_refused_before_any_parent_cache_move(tmp_path):
    slug, outside = tmp_path / "data/submission", tmp_path / "outside"
    (slug / "mapillary").mkdir(parents=True)
    (slug / "mapillary/keep").write_bytes(b"warm")
    outside.mkdir()
    (slug / "assisted").mkdir()
    try:
        (slug / "assisted/native").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory links unavailable: {error}")
    with pytest.raises(ValueError, match="escapes its root"):
        M.set_vpr_caches_aside(slug)
    assert (slug / "mapillary/keep").read_bytes() == b"warm"


def result():
    value = _fake_result(0., 10.)
    for key in ("position", "matcher_position", "vpr_prior"):
        value.pop(key)
    route = _ll(GT_XY + [0., 10.])
    value["assisted_position"] = {"route_latlon": route, "source": "assisted_child:baseline"}
    value["answer"] = {"mode": "assisted", "method": "assisted_policy", "source": "result.assisted_position",
                       "route_latlon": route}
    value["channels"] = {"vpr": {"status": "disabled", "reason": "isolated_assisted_children"},
                         "assisted": {"status": "ok"}}
    value["provenance"]["config"].update(answer="assisted", use_vpr_prior=True)
    value["provenance"]["assisted_inputs"] = {"execution_id": "a" * 32, "experimental": True}
    value["provenance"]["assisted_audit"] = {"selected_candidate": "baseline",
        "reached_stages": [{"kind": "baseline", "status": "complete", "worker_returncode": 0,
                            "comparison": {"model": {"name": "megaloc", "hub": "pinned",
                                            "weights_sha256": "b" * 64, "device": "cpu"}}},
                           {"kind": "cues", "status": "failed", "worker_returncode": 1}]}
    return value


def test_assisted_provenance_keeps_policy_children_and_actual_selected_route_hash():
    value = result()
    prov = M._provenance_of(value)
    assert prov["assisted_inputs"] == value["provenance"]["assisted_inputs"]
    assert prov["assisted_audit"] == value["provenance"]["assisted_audit"]
    assert prov["assisted_selected_answer"]["route_sha256"] == M._fingerprint(value["answer"]["route_latlon"])
    assert prov["vpr_model"]["weights_sha256"] == "b" * 64 and "device" not in prov["vpr_model"]


def test_malformed_assisted_child_record_degrades_without_crashing_aggregate():
    value = result()
    value["provenance"]["assisted_audit"]["reached_stages"] = [None, "incomplete record"]
    health = M._health_of(value)
    assert not health["assisted_only"] and health["assisted_child_outcomes"] == {}
    assert "assisted policy did not select a complete successful child" in M.degraded_reasons(health, token_set=True)


def test_valid_assisted_fallback_exposes_failures_without_fake_ordinary_vpr_degradation():
    value = result()
    health = M._health_of(value)
    assert health["assisted_only"] and health["assisted_child_outcomes"]["cues"] == "failed"
    assert M.degraded_reasons(health, token_set=True) == []
    value["provenance"]["assisted_audit"]["reached_stages"][0]["worker_returncode"] = 1
    assert any("assisted policy" in reason for reason in M.degraded_reasons(M._health_of(value), token_set=True))


def test_all_assisted_fleet_keeps_primary_method_and_hides_unexecuted_fit(tmp_path):
    root, manifest = _setup_root(tmp_path)
    run = tmp_path / "runs/assisted"
    _write_rep(run, "clipA", 0, result())
    summary = M.aggregate(run, manifest, root=root, allow_network=False, n_resamples=99)
    methods = summary["fleets"]["core"]["methods"]
    assert "answer" in methods and "assisted_policy" in methods
    assert "pipeline" not in methods and "matcher_only" not in methods
    assert "Experimental assisted policy" in M.render_markdown(summary)
