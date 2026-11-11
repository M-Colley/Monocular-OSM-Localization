"""Private fixed-bin/importance tests; no actual speed or road inputs."""
import hashlib
import json
import math
import numpy as np
import pytest
from monocular_osm import joint_initial_proposal as h


def test_fixed_representatives_midpoints_and_all_sixteen_bins():
    reps=np.asarray(h.SPEED_REPRESENTATIVES)
    logs=np.asarray(h.LOG_SPEED_REPRESENTATIVES)
    bounds=np.asarray(h.LOG_SPEED_BOUNDARIES)
    assert reps.dtype==logs.dtype==bounds.dtype==np.float64
    assert reps.shape==(16,) and bounds.shape==(15,)
    assert reps[0]==.5 and reps[-1]==55.
    np.testing.assert_array_equal(reps,np.geomspace(.5,55.,16))
    np.testing.assert_array_equal(logs,np.log(reps))
    np.testing.assert_array_equal(bounds,.5*(logs[:-1]+logs[1:]))
    np.testing.assert_array_equal(h.bin_indices(logs,np.zeros(16),1.),np.arange(16))


def test_each_exact_midpoint_ties_higher_and_adjacent_floats():
    bounds=np.asarray(h.LOG_SPEED_BOUNDARIES)
    below=np.nextafter(bounds,-np.inf)
    above=np.nextafter(bounds,np.inf)
    np.testing.assert_array_equal(h.bin_indices(below,np.zeros(15),1.),np.arange(15))
    np.testing.assert_array_equal(h.bin_indices(bounds,np.zeros(15),1.),np.arange(1,16))
    np.testing.assert_array_equal(h.bin_indices(above,np.zeros(15),1.),np.arange(1,16))


def test_total_tails_no_representative_replacement_and_units():
    logk=np.array([-1000.,-20.,math.log(.5),math.log(55.),20.,1000.])
    eps=np.array([-.3,-.1,0.,0.,.1,.3])
    before=logk.copy(),eps.copy()
    actual=h.bin_indices(logk,eps,1.)
    assert actual.tolist()==[0,0,0,15,15,15]
    for unit in (.01,.5,20.):
        np.testing.assert_array_equal(h.bin_indices(logk+math.log(unit),eps,unit),actual)
    np.testing.assert_array_equal(logk,before[0]);np.testing.assert_array_equal(eps,before[1])


def test_clipped_prior_atoms_and_eps_tails_remain_distinct_latents():
    logk=np.clip(np.array([-100.,-90.,0.,90.,100.]),math.log(.5),math.log(55.))
    eps=np.array([-10.,10.,0.,-10.,10.])
    before=logk.tobytes(),eps.tobytes()
    bins=h.bin_indices(logk,eps,1.)
    assert logk[0]==logk[1] and bins[0]==0 and bins[1]==15
    assert logk[-1]==logk[-2] and bins[-1]==15 and bins[-2]==0
    assert (logk.tobytes(),eps.tobytes())==before


def test_readonly_strided_inputs_and_rng_unchanged_independent_output():
    logk=np.linspace(-2.,5.,24)[::2];eps=np.zeros(24)[::2]
    logk.setflags(write=False);eps.setflags(write=False)
    before=(logk.tobytes(),eps.tobytes(),logk.flags.writeable,eps.flags.writeable)
    state=np.random.get_state()
    result=h.bin_indices(logk,eps,np.float64(.4))
    result[:]=0
    assert (logk.tobytes(),eps.tobytes(),logk.flags.writeable,eps.flags.writeable)==before
    after=np.random.get_state()
    assert state[0]==after[0] and state[2:]==after[2:]
    np.testing.assert_array_equal(state[1],after[1])


@pytest.mark.parametrize("logk,eps",[
    ([1.],np.zeros(1)),(np.zeros(1),[0.]),
    (np.zeros(1,np.float32),np.zeros(1)),(np.zeros(1),np.zeros(1,np.float32)),
    (np.zeros(1,bool),np.zeros(1)),(np.zeros((1,1)),np.zeros((1,1))),
    (np.empty(0),np.empty(0)),(np.zeros(2),np.zeros(1)),
    (np.zeros(1,dtype=">f8"),np.zeros(1)),
])
def test_invalid_latent_abi_before_numeric_validation(logk,eps,monkeypatch):
    monkeypatch.setattr(h.np,"isfinite",lambda _:pytest.fail("invalid ABI read"))
    with pytest.raises(ValueError,match="Float64"):
        h.bin_indices(logk,eps,1.)


@pytest.mark.parametrize("unit",[0.,-1.,np.nan,np.inf,-np.inf,True,np.bool_(False),"1",None,complex(1,0)])
def test_invalid_unit_before_latent_values(unit,monkeypatch):
    logk=np.zeros(1);eps=np.zeros(1)
    monkeypatch.setattr(h.np,"isfinite",lambda _:pytest.fail("invalid unit latent read"))
    with pytest.raises(ValueError,match="unit_seconds"):
        h.bin_indices(logk,eps,unit)


@pytest.mark.parametrize("value",[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize("which",[0,1])
def test_nonfinite_latent_draws_refused(value,which):
    arrays=[np.zeros(3),np.zeros(3)];arrays[which][2]=value
    with pytest.raises(ValueError,match="finite"):
        h.bin_indices(*arrays,1.)


def test_particle_cap_before_numeric_work_and_full_cap():
    logk=np.linspace(-1000.,1000.,50_000);eps=np.zeros(50_000)
    actual=h.bin_indices(logk,eps,1.)
    assert actual.dtype==np.int32 and actual.shape==(50_000,)
    assert actual[0]==0 and actual[-1]==15
    with pytest.raises(ValueError,match="50000"):
        h.bin_indices(np.zeros(50_001),np.zeros(50_001),1.)


def test_arithmetic_overflow_refuses_without_latent_clipping():
    logk=np.array([1e308]);eps=np.array([1e308]);before=logk.tobytes(),eps.tobytes()
    with pytest.raises(ValueError,match="overflow"):
        h.bin_indices(logk,eps,1.)
    assert (logk.tobytes(),eps.tobytes())==before


def test_three_cells_two_bins_genuine_conditional_importance_identity():
    from monocular_osm.junction_sequence import initial_proposal
    lengths=np.array([1.,2.,3.]);p=lengths/lengths.sum()
    speed=np.array([.7,.3]);guides=np.array([[9.,.4,.05],[.05,.7,10.]])
    pairs=[initial_proposal(lengths,g) for g in guides]
    q=np.array([x[0] for x in pairs]);weights=np.exp([x[1] for x in pairs])
    target=speed[:,None]*p[None,:];proposal=speed[:,None]*q
    np.testing.assert_allclose(proposal*weights,target,atol=1e-16,rtol=1e-15)
    assert np.all(q>=.25*p) and weights.max()<=4.
    f=np.array([[2.,-1.,3.],[4.,0.,-2.]])
    expected=np.sum(target*f)
    assert np.sum(proposal*weights*f)==pytest.approx(expected,abs=1e-15)
    wrong=proposal*(p/(speed@q))[None,:];wrong/=wrong.sum()
    assert abs(np.sum(wrong*f)-expected)>.1  # marginal q is the wrong joint denominator


def test_full_declared_cells_flat_guide_preserves_road_length_prior():
    from monocular_osm.junction_sequence import initial_proposal
    lengths=np.linspace(.5,100.,h.MAX_CELLS)
    q,logw=initial_proposal(lengths,np.ones(h.MAX_CELLS))
    np.testing.assert_allclose(q,lengths/lengths.sum(),atol=0.,rtol=1e-15)
    np.testing.assert_allclose(logw,0.,atol=4e-15,rtol=0.)


def test_shape_cap_precedes_validation_scratch(monkeypatch):
    logk=np.empty(50_001);eps=np.empty(50_001)
    monkeypatch.setattr(h.np,"isfinite",lambda _:pytest.fail("overcap scratch"))
    with pytest.raises(ValueError,match="50000"):
        h.bin_indices(logk,eps,1.)


def test_fixed_immutable_profile_and_scope():
    assert h.MODE=="speed-conditioned-initial-v1"
    assert h.MAX_PARTICLES==50_000 and h.MAX_CELLS==200_000 and h.SCRATCH_BYTES==64<<20
    assert isinstance(h.SPEED_REPRESENTATIVES,tuple) and isinstance(h.LOG_SPEED_BOUNDARIES,tuple)
    digest=hashlib.sha256(json.dumps(dict(h.PROFILE),sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()
    assert digest==h.PROFILE_SHA256
    assert h.PROFILE["proposal_only"] and not h.PROFILE["target_changed"]
    assert not h.PROFILE["latent_draws_replaced_or_clipped"]
    with pytest.raises(TypeError):h.PROFILE["bins"]=32
