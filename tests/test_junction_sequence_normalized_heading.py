"""Private heading-state integration; no real motion, maps or labels."""
from copy import deepcopy
import math
from types import SimpleNamespace
import numpy as np
import pytest
from monocular_osm import junction_sequence as j, junction_heading_likelihood as h
from monocular_osm.graph_filter import FilterConfig, GraphParticleFilter, wrap_angle


def state():
    net = SimpleNamespace(edge_len=np.full(3, 100.),
        succ_table=np.array([[1, 2], [-1, -1], [-1, -1]]),
        succ_head=np.array([[0., math.pi/2], [0., 0.], [0., 0.]]), seg_heading=np.array([0., 0., math.pi/2]))
    net.locate = lambda edge, offset: (np.zeros_like(offset), np.zeros_like(offset), net.seg_heading[edge], np.zeros_like(edge))
    pf = j._NormalizedHeadingOfflineFilter.__new__(j._NormalizedHeadingOfflineFilter)
    pf.net = net; pf.cfg = FilterConfig(n_particles=3, heading_sigma_deg=15., heading_outlier=.2)
    pf.edge = np.zeros(3, np.int64); pf.offset = np.array([95., 5., 50.])
    pf.prev_head = np.array([np.nan, -math.pi/2, np.nan]); pf.road_heading = np.zeros(3)
    pf.delta = np.full(3, .2); pf.delta_var = np.full(3, .1); pf.delta_ready = True
    pf.logw = np.log([.2, .3, .5])
    return pf


@pytest.mark.parametrize("outlier", [0., .2])
def test_normalized_responsibility_precedes_approximate_bias_update(outlier):
    pf = state(); pf.cfg.heading_outlier = outlier
    before = {k: getattr(pf, k).copy() for k in ("delta", "delta_var", "logw")}
    nu = GraphParticleFilter._heading_innovation(pf, .7 + before["delta"])
    width = h.arc_union_width(pf.net, pf.edge, pf.offset, pf.prev_head, pf.cfg.junction_zone_m)
    S = before["delta_var"] + math.radians(15.)**2 * 2.
    lin = h.normalized_loglike(nu, S, width, 0.); ll = h.normalized_loglike(nu, S, width, outlier)
    rho = np.exp(math.log1p(-outlier) + lin - ll) if outlier else np.ones(3)
    gain = rho * before["delta_var"] / S
    if outlier:
        a = math.log1p(-outlier) - .5*nu**2/S - .5*np.log(2*math.pi*S)
        legacy_rho = np.exp(a - np.logaddexp(a, math.log(outlier/(2*math.pi))))
        assert np.max(abs(rho-legacy_rho)) > .01
    pf.update_heading(.7, .5)
    np.testing.assert_allclose(pf.logw, before["logw"] + ll, rtol=0, atol=2e-15)
    np.testing.assert_allclose(pf.delta, wrap_angle(before["delta"] + gain*nu), rtol=0, atol=2e-15)
    np.testing.assert_allclose(pf.delta_var, (1-gain)*before["delta_var"], rtol=0, atol=2e-15)
    assert j._NormalizedHeadingOfflineFilter._heading_innovation is GraphParticleFilter._heading_innovation


def test_first_heading_and_skipped_measurements_never_use_normalizer(monkeypatch):
    pf = state(); pf.delta_ready = False; old_logw = pf.logw.copy()
    monkeypatch.setattr(h, "arc_union_width", lambda *x: pytest.fail("normalizer before initialization"))
    pf.update_heading(.7, .5)
    np.testing.assert_allclose(pf.delta, [-.7]*3)
    np.testing.assert_array_equal(pf.logw, old_logw)
    before = deepcopy((pf.delta, pf.delta_var, pf.logw))
    pf.update_heading(np.nan, .5); pf.update_heading(.2, 0.)
    for a, b in zip((pf.delta, pf.delta_var, pf.logw), before): np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("change", [{"particles": 100}, {"seed": 1}, {"junction_prior": "legacy"}])
def test_explicit_entry_refuses_nonfixed_sampler_before_construction(monkeypatch, change):
    monkeypatch.setattr(j, "_align", lambda *x, **k: pytest.fail("nonfixed sampler entered"))
    cfg = j.Config(**{"junction_prior": "conditional-uniform-v3", **change})
    with pytest.raises(ValueError): j.align_with_snapshot_normalized_heading(None, None, None, None, None, None, None, config=cfg)


def test_old_entry_does_not_call_new_heading_target(monkeypatch):
    from tests.test_reference_snapshot_junction_probe_v2 import case
    from tests.test_junction_sequence import matches
    net, times, odom, status, _, bank, facts = case()
    monkeypatch.setattr(h, "arc_union_width", lambda *x: pytest.fail("old entry changed target"))
    result = j.align_with_snapshot_appearance(net, odom, status, times, matches([(times[-1], [])]), bank, facts)
    assert result["status"] == "complete_diagnostic_only"
    assert "heading_mode" not in result["profile"]
    assert result["config"]["junction_prior"] == "conditional-uniform-v3"
