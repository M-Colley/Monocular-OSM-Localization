"""Private fixed500k/chunk/readout fixtures; no cases, labels or models."""
import copy
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
from monocular_osm import junction_sequence as j, high_particle_budget as b
from monocular_osm import junction_heading_likelihood as h, snapshot_future_guide as f
from monocular_osm.graph_filter import GraphParticleFilter
from tests.test_junction_sequence_normalized_heading import state
from tests.test_reference_snapshot_junction_probe_v2 import case
from tests.test_junction_sequence import matches

ENTRIES=(j.align_with_snapshot_future_guide_high_particle_budget,
         j.align_with_snapshot_normalized_heading_future_guide_high_particle_budget)


@pytest.mark.parametrize("shape",[(1,),(512,32),(500_000,)])
def test_raw_pointwise_guide_keeps_order_and_parent_math_in_bounded_calls(shape):
    pf=j._HighParticleFutureGuideFilter.__new__(j._HighParticleFutureGuideFilter)
    pf.index=2;pf.net=object();pf.cues=[];seen=[]
    def sample(index,net,edge,offset):
        assert index==2 and net is pf.net and edge.size<=200_000
        seen.append(edge.size);return 1e-6+.2*edge+.3*offset
    pf.future_guide=SimpleNamespace(sample=sample)
    edge=(np.arange(np.prod(shape))%3).reshape(shape);offset=np.linspace(0.,1.,edge.size).reshape(shape)
    result=pf.guide(edge,offset,np.zeros(shape))
    np.testing.assert_array_equal(result,1e-6+.2*edge+.3*offset)
    assert result.shape==shape and sum(seen)==edge.size
    assert len(seen)==(edge.size+199_999)//200_000


def test_guide_bound_refuses_before_any_sample():
    pf=j._HighParticleFutureGuideFilter.__new__(j._HighParticleFutureGuideFilter)
    pf.future_guide=SimpleNamespace(sample=lambda *a:pytest.fail("sample beforecap"))
    with pytest.raises(ValueError):pf.guide(np.zeros(500_001,np.int32),0.,0.)


@pytest.mark.parametrize("count",[3,50_001])
def test_normalized_chunking_matches_repeated_three_particle_oracle(monkeypatch,count):
    ref=state();ref.update_heading(.7,.5);original=state()
    pf=j._HighParticleNormalizedHeadingFilter.__new__(j._HighParticleNormalizedHeadingFilter)
    pf.__dict__.update(copy.deepcopy(vars(original)))
    for key in ("edge","offset","prev_head","road_heading","delta","delta_var","logw"):
        setattr(pf,key,np.resize(getattr(original,key),count))
    pf.cfg.n_particles=count
    real_width,real_like=h.arc_union_width,h.normalized_loglike;calls=[]
    def width(net,edge,offset,previous,zone):
        assert len(edge)<=50_000;calls.append(len(edge));return real_width(net,edge,offset,previous,zone)
    def like(residual,variance,extent,outlier):
        assert len(residual)<=50_000;return real_like(residual,variance,extent,outlier)
    monkeypatch.setattr(h,"arc_union_width",width);monkeypatch.setattr(h,"normalized_loglike",like)
    pf.update_heading(.7,.5)
    for key in ("delta","delta_var","logw"):
        np.testing.assert_array_equal(getattr(pf,key),np.resize(getattr(ref,key),count))
    assert calls==([3] if count==3 else [50_000,1])


def test_heading_bound_refuses_before_innovation():
    pf=j._HighParticleNormalizedHeadingFilter.__new__(j._HighParticleNormalizedHeadingFilter)
    pf.cfg=SimpleNamespace();pf.edge=np.zeros(500_001,np.int32)
    pf._heading_innovation=lambda *a:pytest.fail("innovation beforecap")
    with pytest.raises(ValueError):pf.update_heading(.2,.5)


@pytest.mark.parametrize("entry",ENTRIES)
@pytest.mark.parametrize("config",[j.Config(),b.HighParticleConfig(particles=499_999),
    b.HighParticleConfig(seed=1),b.HighParticleConfig(inner_s=3600.)])
def test_exact_dedicated_configuration_before_dispatch(monkeypatch,entry,config):
    monkeypatch.setattr(j,"_align",lambda *a,**k:pytest.fail("invalid budget entered"))
    with pytest.raises(ValueError):entry(None,None,None,None,None,None,None,config=config)


def test_high_config_cannot_widen_old_generic_entry(monkeypatch):
    monkeypatch.setattr(j,"coarsen_motion",lambda *a:pytest.fail("oldbudget widened"))
    with pytest.raises(ValueError):j.align(None,None,None,None,None,config=b.HighParticleConfig())


@pytest.mark.parametrize("entry",ENTRIES)
def test_explicit_entries_set_only_declared_high_flags(monkeypatch,entry):
    seen=[];monkeypatch.setattr(j,"_align",lambda *a,**k:seen.append(k))
    entry(None,None,None,None,None,None,None)
    assert type(seen[0]["config"]) is b.HighParticleConfig
    assert all(seen[0][k] is True for k in ("high_particle_budget","adaptive_resampling","future_proposal","snapshot_appearance"))
    assert not seen[0].get("no_adaptive_heading_q",False) and not seen[0].get("common_null",False)


@pytest.mark.parametrize("entry",[j.align_with_snapshot_future_guide_adaptive_resampling,
    j.align_with_snapshot_normalized_heading_future_guide_adaptive_resampling,
    j.align_with_snapshot_common_null_appearance,j.align_with_snapshot_common_null_normalized_heading])
def test_old_entry_dispatch_cannot_enable_high_budget(monkeypatch,entry):
    seen=[];monkeypatch.setattr(j,"_align",lambda *a,**k:seen.append(k))
    entry(None,None,None,None,None,None,None)
    assert not seen[0].get("high_particle_budget",False)
