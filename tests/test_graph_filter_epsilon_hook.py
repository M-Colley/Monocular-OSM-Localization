"""Synthetic portable historical-expression oracle; no case data."""
import ast
from copy import deepcopy
import inspect
import math
import textwrap
import numpy as np
import pytest
from monocular_osm import graph_filter as gf
from tests.test_junction_sequence import roads


def parent_predict():
    text=textwrap.dedent(inspect.getsource(gf.GraphParticleFilter.predict))
    old="""        self.eps = a * self.eps + self.rng.normal(
            0.0, self._dev_sigma * math.sqrt(1.0 - a * a), n)"""
    new="""        _eps_mean0 = a * self.eps
        _eps_innovation = self.rng.normal(
            0.0, self._dev_sigma * math.sqrt(1.0 - a * a), n)
        self.eps = self._epsilon_transition(_eps_mean0, _eps_innovation, a)"""
    assert text.count(new)==1
    ns=dict(gf.__dict__);exec(compile(text.replace(new,old),'PRIVATE_PARENT_EXPRESSION','exec'),ns)
    return ns['predict']


class Parent(gf.GraphParticleFilter):predict=parent_predict()


def equal(a,b):
    for key in ('edge','offset','logk','eps','delta','delta_var','logw','x','y','road_heading','prev_head'):
        assert getattr(a,key).dtype==getattr(b,key).dtype
        assert getattr(a,key).tobytes()==getattr(b,key).tobytes(),key
    assert a.rng.bit_generator.state==b.rng.bit_generator.state
    assert a.delta_ready==b.delta_ready and a.n_reinit==b.n_reinit


@pytest.mark.parametrize('metric',[False,True])
@pytest.mark.parametrize('step,dt',[(1.,.1),(0.,.1),(0.,0.),(1.,0.),(0.,np.nextafter(0.,1.))])
def test_default_state_bytes_and_rng_exact_parent(metric,step,dt):
    net=roads(two=True);cfg=gf.FilterConfig(n_particles=128,seed=19,path_storage=False)
    a=Parent(net,cfg,metric=metric,unit_seconds=.1);b=gf.GraphParticleFilter(net,cfg,metric=metric,unit_seconds=.1)
    a.initialise();b.initialise()
    for pf in (a,b):
        lo,hi=pf._k_bounds;pf.logk[:2]=[lo-cfg.logk_bounds_margin,hi+cfg.logk_bounds_margin]
    a.predict(step,.1,dt);b.predict(step,.1,dt);equal(a,b)


def test_hook_receives_retained_innovation_after_K_before_return():
    class Spy(gf.GraphParticleFilter):
        def _epsilon_transition(self,mean0,innovation,a):
            self.observed=(mean0.copy(),innovation.copy(),a,self.logk.copy())
            return super()._epsilon_transition(mean0,innovation,a)
    net=roads();cfg=gf.FilterConfig(n_particles=32,seed=8,path_storage=False)
    pf=Spy(net,cfg,metric=False,unit_seconds=.1);pf.initialise();pf.eps[:]=1e16
    oldk=pf.logk.copy();oldeps=pf.eps.copy();rng=np.random.default_rng();rng.bit_generator.state=deepcopy(pf.rng.bit_generator.state)
    dt=.1;lo,hi=pf._k_bounds
    k=np.clip(oldk+rng.normal(0.,cfg.logk_rw*math.sqrt(dt),32),lo-cfg.logk_bounds_margin,hi+cfg.logk_bounds_margin)
    a=math.exp(-dt/cfg.speed_dev_tau_s);noise=rng.normal(0.,pf._dev_sigma*math.sqrt(1-a*a),32)
    pf.predict(0.,0.,dt);mean,actual,coefficient,newk=pf.observed
    np.testing.assert_array_equal(mean,a*oldeps);np.testing.assert_array_equal(actual,noise)
    np.testing.assert_array_equal(newk,k);assert coefficient==a and np.var(actual)>0
    assert np.var(pf.eps-mean)==0
    assert pf.rng.bit_generator.state==rng.bit_generator.state


def test_hook_result_drives_same_step_distance_without_extra_draws():
    class Fixed(gf.GraphParticleFilter):
        def _epsilon_transition(self,mean0,innovation,a):return np.full_like(mean0,math.log(2.))
    cfg=gf.FilterConfig(n_particles=32,seed=8,path_storage=False,logk_rw=0.,step_noise=0.)
    pf=Fixed(roads(),cfg,metric=False,unit_seconds=.1);pf.initialise()
    pf.edge[:]=0;pf.offset[:]=0.;pf.logk[:]=0.
    rng=np.random.default_rng();rng.bit_generator.state=deepcopy(pf.rng.bit_generator.state)
    rng.normal(0.,0.,32);a=math.exp(-.1/cfg.speed_dev_tau_s)
    rng.normal(0.,pf._dev_sigma*math.sqrt(1-a*a),32);rng.normal(0.,0.,32)
    pf.predict(1.,float('nan'),.1)
    np.testing.assert_allclose(pf.offset,2.,rtol=0,atol=1e-15)
    assert pf.rng.bit_generator.state==rng.bit_generator.state
