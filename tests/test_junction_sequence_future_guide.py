"""Private proposal identities; no actual data or labels."""
from types import SimpleNamespace
import math
import numpy as np
import pytest
from monocular_osm import junction_sequence as j
from monocular_osm.graph_filter import GraphParticleFilter


def test_raw_product_not_defended_inside_guide():
    pf=j._SnapshotFutureGuideFilter.__new__(j._SnapshotFutureGuideFilter)
    pf.net=object();pf.index=0;pf.unit_seconds=1.;pf.units=np.array([0.,1.])
    target=SimpleNamespace(remaining=lambda net,edge,offset:900.+100*np.asarray(edge)-offset)
    pf.cues=[SimpleNamespace(step=1,targets=(target,))];calls=[]
    def sample(index,net,edge,offset):
        calls.append((index,np.asarray(edge).copy(),np.asarray(offset).copy()))
        return .2+.1*np.asarray(edge)
    pf.future_guide=SimpleNamespace(sample=sample)
    edge=np.array([[0,1],[1,0]]);offset=np.zeros_like(edge,float);logk=np.full(edge.shape,math.log(8.))
    old=j._OfflineFilter.guide(pf,edge,offset,logk)
    np.testing.assert_array_equal(pf.guide(edge,offset,logk),old*(.2+.1*edge))
    assert calls[0][0]==0 and calls[0][1].shape==edge.shape
    pf.guide(edge,offset+3.,logk)
    assert len(calls)==2 and np.all(calls[1][2]==3.)


def test_aux_defense_and_inverse_effective_guide_recover_target():
    from monocular_osm.snapshot_future_guide import defensive_auxiliary_guide
    w=np.array([.1,.3,.6]);raw=np.array([1e-6,.1,1.]);effective=defensive_auxiliary_guide(w,raw)
    class Fake:
        def resample(self,q):self.q=q.copy();return np.array([0,2,1])
    pf=Fake();ids=j.auxiliary_resample(pf,w,effective)
    np.testing.assert_allclose(pf.q,.25*w+.75*w*raw/(w@raw),rtol=1e-15)
    assert np.all(pf.q>=.25*w-1e-16) and np.max(w/pf.q)<=4.+1e-14
    inv=1/effective[ids]
    np.testing.assert_allclose(np.exp(pf.logw),inv/inv.sum(),rtol=1e-15)
    target=pf.q/effective
    np.testing.assert_allclose(target/target.sum(),w,rtol=1e-15)


def test_future_variants_keep_declared_heading_targets():
    assert j._SnapshotFutureGuideFilter.update_heading is GraphParticleFilter.update_heading
    assert j._NormalizedHeadingSnapshotFutureGuideFilter.update_heading is j._NormalizedHeadingOfflineFilter.update_heading


@pytest.mark.parametrize('entry',[j.align_with_snapshot_appearance,j.align_with_snapshot_normalized_heading])
def test_old_entries_never_construct_future_guide(monkeypatch,entry):
    from monocular_osm import snapshot_future_guide as core
    from tests.test_reference_snapshot_junction_probe_v2 import case
    from tests.test_junction_sequence import matches
    net,times,odom,status,_,bank,facts=case()
    monkeypatch.setattr(core,'build',lambda *x,**k:pytest.fail('old target enabled future proposal'))
    result=entry(net,odom,status,times,matches([(times[-1],[])]),bank,facts)
    assert result['status']=='complete_diagnostic_only' and 'future_guide_profile' not in result['profile']


@pytest.mark.parametrize("entry",[j.align_with_snapshot_future_guide,j.align_with_snapshot_normalized_heading_future_guide])
@pytest.mark.parametrize("change",[{"particles":100},{"seed":1},{"junction_prior":"legacy"}])
def test_new_entries_refuse_nonfixed_sampler(monkeypatch,entry,change):
    monkeypatch.setattr(j,"_align",lambda *x,**k:pytest.fail("nonfixed sampler entered"))
    cfg=j.Config(**{"junction_prior":"conditional-uniform-v3",**change})
    with pytest.raises(ValueError):entry(None,None,None,None,None,None,None,config=cfg)
