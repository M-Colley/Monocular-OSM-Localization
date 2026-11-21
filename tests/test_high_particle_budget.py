"""Private fixed-budget/global-importance math only; no full genealogy or cases."""
import ast
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pytest

from monocular_osm import high_particle_budget as h
from monocular_osm import adaptive_auxiliary_resampling as old
from monocular_osm import snapshot_future_guide as parent


def test_exact_dedicated_frozen_config_and_payload_without_allocating_genealogy():
    cfg=h.HighParticleConfig();assert cfg.validate() is None
    assert asdict(cfg)==dict(particles=500000,seed=0,genealogy_bytes=12<<30,
        inner_s=1800.,junction_prior="conditional-uniform-v3")
    assert 16*cfg.particles*1201==9_608_000_000 < cfg.genealogy_bytes < h.PROFILE["external_rss_bytes"]
    assert h.BUDGET_MODE=="fixed-500k-final-summary-v1"
    assert h.RESAMPLING_MODE=="target-ess-half-v1"
    with pytest.raises(FrozenInstanceError):cfg.particles=50000


@pytest.mark.parametrize("field,value", [
    ("particles",50000),("particles",500001),("particles",500000.),("particles",True),
    ("seed",1),("seed",0.),("seed",False),
    ("genealogy_bytes",1<<30),("genealogy_bytes",(12<<30)+1),("genealogy_bytes",float(12<<30)),
    ("inner_s",1799.),("inner_s",1801.),("inner_s",1800),("inner_s",float("nan")),
    ("inner_s",float("inf")),("inner_s",True),
    ("junction_prior","legacy"),("junction_prior",None),
])
def test_every_fixed_config_field_rejects_other_values_or_types(field,value):
    with pytest.raises(ValueError):replace(h.HighParticleConfig(),**{field:value}).validate()


@pytest.mark.parametrize("w,expected", [([.25]*4,False),([.5,.5,0.,0.],False),
    ([.6,.4,0.,0.],True),([1.],False),([1.,0.,0.,0.],True)])
def test_exact_old_formula_strict_half_and_no_clamping(w,expected):
    w=np.asarray(w,dtype=np.float64)
    assert h.decision(w)==old.decision(w)==dict(ess=float(1/np.sum(w*w)),
        resample=expected,particle_count=len(w))


def test_fixed_full500k_capacity_and_global_one_dot_before_defense(monkeypatch):
    w=np.linspace(1.,2.,500000);w/=w.sum();g=np.linspace(1e-6,1.,500000)
    decision=h.decision(w)
    assert decision["particle_count"]==500000 and decision["ess"]==float(1/np.sum(w*w))
    actual_dot=np.dot;calls=[]
    def dot(left,right):
        calls.append((left.shape,right.shape))
        return actual_dot(left,right)
    monkeypatch.setattr(h.np,"dot",dot)
    e=h.defensive_auxiliary_guide(w,g)
    assert calls==[((500000,),(500000,))]
    oracle=math.fsum(float(a)*float(b) for a,b in zip(w,g))
    np.testing.assert_allclose(e,.25+.75*g/oracle,rtol=1e-15,atol=1e-15)
    assert e.dtype==np.float64 and e.shape==(500000,)
    with pytest.raises(ValueError):old.decision(w)
    with pytest.raises(ValueError):parent.defensive_auxiliary_guide(w,g)


@pytest.mark.parametrize("count",[1,2,31,50000])
def test_literal_parent_effective_guide_exact_for_original_capacity(count):
    rng=np.random.default_rng(91);w=rng.uniform(.1,1.,count);w/=w.sum()
    g=rng.uniform(1e-6,1.,count)
    np.testing.assert_array_equal(h.defensive_auxiliary_guide(w,g),parent.defensive_auxiliary_guide(w,g))


def test_global_q_inverse_guide_recovers_carried_weight_not_weight_squared():
    w=np.array([.97,.01,.01,.01]);g=np.array([.01,.2,.7,1.])
    assert h.decision(w)["resample"]
    e=h.defensive_auxiliary_guide(w,g);q=w*e;q/=q.sum()
    recovered=q/e;recovered/=recovered.sum()
    np.testing.assert_allclose(recovered,w,rtol=1e-15)
    assert np.all(q>=.25*w-1e-16) and np.max(1/e)<=4
    wrong=recovered*w;wrong/=wrong.sum()
    assert not np.allclose(wrong,w)


def test_per_chunk_normalization_would_change_registered_global_proposal():
    w=np.array([.45,.05,.05,.45]);g=np.array([1.,2.,20.,30.])
    global_e=h.defensive_auxiliary_guide(w,g)
    local_e=np.concatenate([h.defensive_auxiliary_guide(w[:2]/w[:2].sum(),g[:2]),
                            h.defensive_auxiliary_guide(w[2:]/w[2:].sum(),g[2:])])
    assert not np.allclose(global_e,local_e)
    global_q=w*global_e;global_q/=global_q.sum()
    local_q=w*local_e;local_q/=local_q.sum()
    assert not np.allclose(global_q,local_q)  # a different proposal, not the declared one


def test_readonly_strided_inputs_and_carried_weights_rng_unchanged():
    w=np.full(40,.05)[::2];g=np.linspace(.1,1.,40)[::2]
    for v in (w,g):v.flags.writeable=False
    logw=np.log(w);state_before=(w.tobytes(),g.tobytes(),logw.tobytes(),w.flags.writeable,g.flags.writeable)
    rng=np.random.get_state();assert not h.decision(w)["resample"]
    e=h.defensive_auxiliary_guide(w,g)
    assert not np.shares_memory(e,w) and not np.shares_memory(e,g)
    assert (w.tobytes(),g.tobytes(),logw.tobytes(),w.flags.writeable,g.flags.writeable)==state_before
    after=np.random.get_state();assert rng[0]==after[0] and rng[2:]==after[2:]
    np.testing.assert_array_equal(rng[1],after[1])


@pytest.mark.parametrize("bad",[[1.],np.ones(1,dtype=np.float32),np.array([True]),
    np.empty(0),np.ones((1,1)),np.array([np.nan]),np.array([np.inf]),
    np.array([-.1,1.1]),np.zeros(2),np.array([.4,.4])])
def test_invalid_weights_refuse_in_both_operations(bad):
    with pytest.raises(ValueError):h.decision(bad)
    with pytest.raises(ValueError):h.defensive_auxiliary_guide(bad,np.ones(1))


@pytest.mark.parametrize("bad",[np.ones(1,dtype=np.float32),np.ones((1,2)),
    np.array([0.,1.]),np.array([-1.,1.]),np.array([np.inf,1.]),np.array([np.nan,1.])])
def test_invalid_raw_guide_refused(bad):
    with pytest.raises(ValueError):h.defensive_auxiliary_guide(np.full(2,.5),bad)


def test_oversize_or_bad_shape_refused_before_any_numerical_scratch(monkeypatch):
    too_many=np.zeros(500001);valid=np.full(500000,1/500000)
    monkeypatch.setattr(h.np,"isfinite",lambda *a:pytest.fail("preflight admitted unbounded numerics"))
    with pytest.raises(ValueError):h.decision(too_many)
    with pytest.raises(ValueError):h.defensive_auxiliary_guide(too_many,too_many)
    with pytest.raises(ValueError):h.defensive_auxiliary_guide(valid,too_many)


def test_overflow_is_refused_without_nan_or_infinite_effective_output():
    with pytest.raises(ValueError,match="overflow"):
        h.defensive_auxiliary_guide(np.array([0.,1.]),np.array([1e308,1e-308]))


def test_profile_digest_immutable_and_original_helper_bytes_unchanged():
    expected=hashlib.sha256(json.dumps(dict(h.PROFILE),sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()
    assert h.PROFILE_SHA256==expected
    with pytest.raises(TypeError):h.PROFILE["particles"]=1
    assert hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest()=="b45ebbe290cd6cb73b421ab8a1411ec732262f8f5baf932a14a784825f19ba6f"
    assert hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest()=="2ed9a79910acf62dc7e287209b8312367f8ba6a7f58cc34383f4e698e6280587"


def test_original_fifty_thousand_Config_AST_unchanged_by_new_configuration():
    original=subprocess.check_output(["git","show",
        "099fbbd2f779eda9ccf36fcdc4f9a7ba74f4d4b2:monocular_osm/junction_sequence.py"]).decode("utf-8")
    current=(Path(h.__file__).parent/"junction_sequence.py").read_text(encoding="utf-8")
    def config(text):
        return next(node for node in ast.parse(text).body if isinstance(node,ast.ClassDef) and node.name=="Config")
    assert ast.dump(config(original),include_attributes=False)==ast.dump(config(current),include_attributes=False)
