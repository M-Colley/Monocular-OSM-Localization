"""Fixed normalized road-context epsilon TARGET kernel; no IO or admission.

Caller supplies the genuine retained mean0=a*Eold and ACTUAL parent Gaussian
innovation after the unchanged clipped-K draw. Never recover that innovation
by subtracting rounded states. Direct mixture sampling has q=p: no PDF weight.
"""
import hashlib
import json
import math
from types import MappingProxyType
import numpy as np

MODE='base-speed-context-ou-mixture-v1'
MAX_PARTICLES=50_000
MAX_STEPS=1200
BASE_WEIGHT=.25
ROAD_WEIGHT=.75
PARENT_SIGMA=.3
PARENT_TAU=10.
ROAD_EXTRA_SD=.75
NOISE_GAIN=math.sqrt((PARENT_SIGMA**2+ROAD_EXTRA_SD**2)/PARENT_SIGMA**2)
PROFILE=MappingProxyType({
    'method':MODE,'parent_weight':BASE_WEIGHT,'road_weight':ROAD_WEIGHT,
    'parent_speed_dev_sigma':PARENT_SIGMA,'parent_speed_dev_tau_s':PARENT_TAU,
    'extra_road_diffusion_log_sd':ROAD_EXTRA_SD,'maximum_particles':MAX_PARTICLES,
    'maximum_composed_steps':MAX_STEPS,'witness_steps':5,'witness_array_bytes':64<<20,
    'mean':'road_mean=retained mean0+(1-a)*(logM-newK-(logu-logdt))',
    'variance':'Vroad=(.3^2+.75^2)*(1-a^2), stationary-diffusion broadening, not (1-a)^2 equilibrium uncertainty',
    'sampling':'Original innovation first; independent U for effective eligible indices only; U>=.25 road; reuse actual innovation times fixed gain',
    'uniform_sentinel':-1.,'default':'mean0+innovation, exact original operation; no new U if allneutral or a==1',
    'correction':'direct q=p, no extra target/PDF/importance factor',
    'witness_schedule':'unique sorted int32 1+((T-1)*j+2)//4 for j=0..4',
    'stationary_mean':'For fixedcontext/K/rate .75*c, notc',
    'interpretation':'NEW uncalibrated discrete target; broader slow AND fast tails, no persistent traffic regimes or exact semigroup',
    'target_changed':True,'appearance_target_unchanged':True,'proposal_only':False,
    'measured_speed_or_covariance':False,'legal_applicability_verified':False,
    'native_data_context_or_policy_admission':False})
PROFILE_SHA256=hashlib.sha256(json.dumps(dict(PROFILE),sort_keys=True,
    separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _real(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)):
        raise ValueError('finite real '+name+' required')
    try:result=float(value)
    except (ValueError,OverflowError) as error:raise ValueError('finite real '+name+' required') from error
    if not math.isfinite(result):raise ValueError('finite real '+name+' required')
    return result

def witness_steps(total_steps):
    """Predetermined full-particle snapshots, never selected from outcomes."""
    if type(total_steps)is not int or not 1<=total_steps<=MAX_STEPS:
        raise ValueError('fixed full-window composed steps 1..1200 required')
    return np.unique(np.array([1+((total_steps-1)*j+2)//4 for j in range(5)],np.int32))

def sample_epsilon(mean0,innovation,a,new_logk,u,dt,base_mps,eligible,rng):
    """Return ``(epsilon, diagnostics)`` without modifying supplied vectors.

    Eligible means the caller's full native-moving AND accepted-context mask.
    Missing MPS is zero/ineligible. Effective eligibility additionally excludes
    the exact a==1 degeneracy; context_count records the pre-degeneracy mask.
    Diagnostics contain ACTUAL U, with -1 for undrawn values (not invented0).
    """
    vectors=(mean0,innovation,new_logk,base_mps)
    if not (all(isinstance(v,np.ndarray) and v.dtype==np.float64 and v.ndim==1 for v in vectors)
            and isinstance(eligible,np.ndarray) and eligible.dtype==np.bool_ and eligible.ndim==1
            and 1<=mean0.size<=MAX_PARTICLES and all(v.shape==mean0.shape for v in vectors[1:])
            and eligible.shape==mean0.shape):
        raise ValueError('same-shape bounded Float64 vectors/bool mask, 1<=P<=50000 required')
    a=_real(a,'a');u=_real(u,'u');dt=_real(dt,'dt')
    if dt<0 or not 0<=a<=1 or a!=math.exp(-dt/PARENT_TAU):
        raise ValueError('fixed parent a=exp(-dt/10), nonnegative dt required')
    if any(not np.isfinite(v).all() for v in vectors) or np.any(base_mps<0):
        raise ValueError('finite latent/innovation and nonnegative context vectors required')
    if np.any(eligible & (base_mps<=0)):
        raise ValueError('positive MPS required for every eligible context')
    with np.errstate(over='ignore',invalid='ignore'):
        epsilon=mean0+innovation
    if not np.isfinite(epsilon).all():raise ValueError('original epsilon arithmetic overflow')
    uniform=np.full(mean0.shape,-1.,np.float64)
    selected=np.zeros(mean0.shape,np.bool_)
    effective=eligible.copy() if dt>0 and a<1 else np.zeros(mean0.shape,np.bool_)
    count=int(np.count_nonzero(eligible));ids=np.flatnonzero(effective)
    if len(ids):
        if u<=0:raise ValueError('positive measured step required for effective context')
        lograte=math.log(u)-math.log(dt)
        with np.errstate(over='ignore',invalid='ignore'):
            center=np.log(base_mps[ids])-new_logk[ids]-lograte
            road_mean=mean0[ids]+(1.-a)*center
            road_value=road_mean+innovation[ids]*NOISE_GAIN
        if not np.isfinite(road_value).all():raise ValueError('road epsilon arithmetic overflow; no clipping')
        draw=getattr(rng,'random',None)
        if not callable(draw):raise ValueError('caller RNG with random required')
        values=draw(len(ids))
        if not (isinstance(values,np.ndarray) and values.dtype==np.float64 and values.shape==ids.shape
                and np.isfinite(values).all() and np.all(values>=0) and np.all(values<1)):
            raise ValueError('actual finite Float64 mixture uniforms in [0,1) required')
        uniform[ids]=values;road=values>=BASE_WEIGHT
        selected[ids[road]]=True;epsilon[ids[road]]=road_value[road]
    return epsilon,{'uniform':uniform,'road_selected':selected,'effective_eligible':effective,
        'context_count':count,'applied_count':int(len(ids)),
        'road_component_count':int(np.count_nonzero(selected))}
