"""Private fixed-kernel proofs; no actual map/motion/GT/model payloads."""
from copy import deepcopy
import hashlib,json,math
import numpy as np
import pytest
from monocular_osm import road_speed_kernel as k

class Draws:
    def __init__(self,values):self.values=np.array(values,np.float64);self.calls=[]
    def random(self,n):
        self.calls.append(n);assert n==len(self.values)
        return self.values.copy()

class NoDraw:
    def random(self,n):pytest.fail('unexpected mixture draw')

def inputs(n=4,dt=.5):
    a=math.exp(-dt/10.)
    return [np.full(n,a*.2),np.linspace(-.1,.1,n),a,np.full(n,math.log(4.)),
            1.,dt,np.full(n,12.),np.ones(n,np.bool_)]

def test_exact_mixture_threshold_actual_uniforms_and_conditional_formula():
    args=inputs(6);args[-1][[1,4]]=False;args[-2][[1,4]]=0.
    draw=Draws([0.,np.nextafter(.25,0.),.25,np.nextafter(1.,0.)])
    actual,d=k.sample_epsilon(*args,draw)
    expected=args[0]+args[1]
    ids=np.array([3,5]);a=args[2]
    c=np.log(args[6][ids])-args[3][ids]-(math.log(args[4])-math.log(args[5]))
    expected[ids]=args[0][ids]+(1-a)*c+args[1][ids]*k.NOISE_GAIN
    np.testing.assert_array_equal(actual,expected)
    assert draw.calls==[4]
    np.testing.assert_array_equal(d['uniform'],[0.,-1.,np.nextafter(.25,0.),.25,-1.,np.nextafter(1.,0.)])
    assert np.flatnonzero(d['road_selected']).tolist()==[3,5]
    assert (d['context_count'],d['applied_count'],d['road_component_count'])==(4,4,2)

def test_retained_noise_not_subtraction_recovery_and_no_epsilon_clipping():
    args=inputs(2);args[0][:]=1e8;args[1][:]=[5e-9,-5e-9]
    args[3][:]=0.;args[4]=args[5];args[6][:]=1.
    actual,_=k.sample_epsilon(*args,Draws([.8,.8]))
    np.testing.assert_array_equal(actual,args[0]+args[1]*k.NOISE_GAIN)
    recovered=(args[0]+args[1])-args[0]
    assert np.all(recovered==0.) and np.all(actual!=args[0])

@pytest.mark.parametrize('u',[1.,0.,-1.])
def test_all_neutral_exact_original_values_and_rng(u):
    args=inputs();args[4]=u;args[-1][:]=False;args[-2][:]=0.
    rng=np.random.default_rng(11);before=deepcopy(rng.bit_generator.state)
    actual,d=k.sample_epsilon(*args,rng)
    assert actual.tobytes()==(args[0]+args[1]).tobytes()
    assert rng.bit_generator.state==before and np.all(d['uniform']==-1.)
    assert not d['road_selected'].any() and not d['effective_eligible'].any()
    assert d['context_count']==d['applied_count']==d['road_component_count']==0

@pytest.mark.parametrize('dt',[0.,1e-320])
def test_exact_a1_degeneracy_context_count_and_no_added_draw(dt):
    args=inputs(dt=dt);args[1][:]=0.;args[4]=0. if dt==0 else 1.
    actual,d=k.sample_epsilon(*args,NoDraw())
    np.testing.assert_array_equal(actual,args[0])
    assert d['context_count']==4 and d['applied_count']==0
    assert np.all(d['uniform']==-1.) and not d['road_selected'].any()

def test_stable_lograte_extreme_tails_no_ratio_underflow_or_clip():
    args=inputs(1,dt=1e300);args[4]=1e-300;args[6][:]=1.;args[3][:]=0.;args[1][:]=0.
    actual,_=k.sample_epsilon(*args,Draws([.9]))
    assert actual[0]==-(math.log(1e-300)-math.log(1e300))
    assert actual[0]>1000. and np.isfinite(actual[0])

@pytest.mark.parametrize('slot',[0,1,3,6])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_nonfinite_input_before_rng(slot,bad):
    args=inputs();args[slot][2]=bad
    with pytest.raises(ValueError,match='finite'):k.sample_epsilon(*args,NoDraw())

@pytest.mark.parametrize('slot',[0,1,3,6])
def test_wrong_float_dtype_before_rng(slot):
    args=inputs();args[slot]=args[slot].astype(np.float32)
    with pytest.raises(ValueError,match='Float64'):k.sample_epsilon(*args,NoDraw())

@pytest.mark.parametrize('change',['zero','over','shape','maskdtype','maskshape','matrix','list','endian'])
def test_abi_and_particle_cap_before_scratch(change,monkeypatch):
    args=inputs(0 if change=='zero' else 50001 if change=='over' else 4)
    if change=='shape':args[1]=np.zeros(3)
    if change=='maskdtype':args[-1]=args[-1].astype(np.int32)
    if change=='maskshape':args[-1]=np.ones(3,np.bool_)
    if change=='matrix':args[0]=args[0].reshape(2,2)
    if change=='list':args[0]=args[0].tolist()
    if change=='endian':args[0]=args[0].astype('>f8')
    monkeypatch.setattr(k.np,'isfinite',lambda *a:pytest.fail('invalid ABI value read'))
    with pytest.raises(ValueError,match='bounded'):k.sample_epsilon(*args,NoDraw())

@pytest.mark.parametrize('slot,value',[(2,True),(2,-.1),(2,1.1),(2,np.nan),(2,np.inf),
    (4,np.nan),(4,np.inf),(5,-1.),(5,np.inf),(5,False)])
def test_invalid_scalar_inputs(slot,value):
    args=inputs();args[slot]=value
    with pytest.raises(ValueError):k.sample_epsilon(*args,NoDraw())

def test_wrong_parent_tau_no_unfrozen_parameter_knob():
    args=inputs();args[2]=np.nextafter(args[2],0.)
    with pytest.raises(ValueError,match='fixed parent'):k.sample_epsilon(*args,NoDraw())

@pytest.mark.parametrize('change',['missing','negative','step','original_overflow','road_overflow'])
def test_contradictory_context_and_overflow_refuse_before_draw(change):
    args=inputs()
    if change=='missing':args[6][0]=0.
    if change=='negative':args[6][0]=-1.
    if change=='step':args[4]=0.
    if change=='original_overflow':args[0][:]=1e308;args[1][:]=1e308
    if change=='road_overflow':args[0][:]=1e308;args[1][:]=7e307
    with pytest.raises(ValueError):k.sample_epsilon(*args,NoDraw())

def test_readonly_strided_inputs_and_outputs_do_not_alias():
    original=inputs(8);args=[]
    for value in original:
        if isinstance(value,np.ndarray):
            view=value[::2];view.flags.writeable=False;args.append(view)
        else:args.append(value)
    before=[(v.tobytes(),v.flags.writeable) for v in args if isinstance(v,np.ndarray)]
    result,d=k.sample_epsilon(*args,Draws([.1,.2,.8,.9]))
    result[:]=0.;d['uniform'][:]=0.;d['effective_eligible'][:]=False
    after=[(v.tobytes(),v.flags.writeable) for v in args if isinstance(v,np.ndarray)]
    assert after==before

@pytest.mark.parametrize('dt',[.5,1.,10.])
def test_normalized_quadrature_and_direct_sample_moments(dt):
    from scipy.integrate import quad
    args=inputs(50_000,dt);a=args[2];c=math.log(4.)
    args[3][:]=0.;args[4]=dt;args[6][:]=4.
    rng=np.random.default_rng(19)
    args[1][:]=rng.normal(0.,.3*math.sqrt(1-a*a),50_000)
    out,d=k.sample_epsilon(*args,rng)
    m0=a*.2;m1=m0+(1-a)*c;v0=.3**2*(1-a*a);v1=(.3**2+.75**2)*(1-a*a)
    mean=.25*m0+.75*m1;var=.25*v0+.75*v1+.25*.75*(m1-m0)**2
    def normal(x,m,v):return math.exp(-.5*(x-m)**2/v)/math.sqrt(2*math.pi*v)
    def density(x):return .25*normal(x,m0,v0)+.75*normal(x,m1,v1)
    mass=quad(density,-math.inf,math.inf,epsabs=1e-10)[0]
    qmean=quad(lambda x:x*density(x),-math.inf,math.inf,epsabs=1e-10)[0]
    assert mass==pytest.approx(1.,abs=1e-9) and qmean==pytest.approx(mean,abs=1e-9)
    fourth=.25*(3*v0*v0+6*v0*(m0-mean)**2+(m0-mean)**4)+.75*(3*v1*v1+6*v1*(m1-mean)**2+(m1-mean)**4)
    assert abs(out.mean()-mean)<6*math.sqrt(var/50_000)
    assert abs(out.var()-var)<6*math.sqrt((fourth-var*var)/50_000)
    assert d['context_count']==d['applied_count']==50_000
    assert d['road_component_count']==int((d['uniform']>=.25).sum())

@pytest.mark.parametrize('bad',[-.1,1.,np.nan])
def test_invalid_actual_uniforms_not_silently_clipped(bad):
    with pytest.raises(ValueError,match='uniforms'):k.sample_epsilon(*inputs(1),Draws([bad]))

def test_fixed_witness_schedule_all_steps_no_outcome_selection():
    for total in range(1,1201):
        actual=k.witness_steps(total)
        expected=sorted({1+((total-1)*j+2)//4 for j in range(5)})
        assert actual.dtype==np.int32 and actual.tolist()==expected
        assert actual[0]==1 and actual[-1]==total and len(actual)<=5
    assert k.witness_steps(1200).tolist()==[1,301,601,900,1200]
    assert 5*50_000*(7*8+4)<64<<20

@pytest.mark.parametrize('bad',[0,-1,1201,1.,True,np.int64(1)])
def test_invalid_witness_schedule_refused(bad):
    with pytest.raises(ValueError):k.witness_steps(bad)

def test_immutable_fixed_profile_source_contract():
    assert k.MODE=='base-speed-context-ou-mixture-v1'
    assert (k.BASE_WEIGHT,k.ROAD_WEIGHT,k.PARENT_SIGMA,k.PARENT_TAU,k.ROAD_EXTRA_SD)==(.25,.75,.3,10.,.75)
    assert k.NOISE_GAIN==math.sqrt((.3**2+.75**2)/.3**2)
    digest=hashlib.sha256(json.dumps(dict(k.PROFILE),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert digest==k.PROFILE_SHA256
    assert k.PROFILE['target_changed'] and not k.PROFILE['proposal_only']
    assert not k.PROFILE['measured_speed_or_covariance']
    with pytest.raises(TypeError):k.PROFILE['road_weight']=.8
