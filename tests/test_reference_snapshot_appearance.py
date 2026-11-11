"""Pure synthetic snapshot alternatives, roads and native clocks only."""
from copy import deepcopy
import math
import numpy as np
import pytest
from monocular_osm import reference_snapshot_appearance as a, junction_sequence as j
from tests import test_raw_vpr_appearance as old
from tests.test_reference_snapshot_junction_probe import case


def test_all_one_weights_call_genuine_builder_and_preserve_old_field_bits(monkeypatch):
    net, times, _, _, motion, evidence, facts = case()
    original = old.case()[5]
    genuine, calls = a.strict.poi.build_appearance_field, []
    def wrapped(*args, **kwargs):
        calls.append(kwargs)
        return genuine(*args, **kwargs)
    monkeypatch.setattr(a.strict.poi, "build_appearance_field", wrapped)
    result = a.prepare_snapshot(evidence, net, facts, times, motion.times)
    assert len(calls) == 1 and calls[0]["reference_mean"] is False
    control = a.strict.prepare(original, net, facts, times, motion.times)
    for key in ("values", "query_times"):
        np.testing.assert_array_equal(getattr(result.field, key), getattr(control.field, key))
    for key in ("maxima", "groups", "powers", "steps"):
        np.testing.assert_array_equal(getattr(result, key), getattr(control, key))
    assert result.receipt["all_one_weights_used_genuine_original_builder"] is True
    assert result.receipt["descriptor_equivalence_assumed"] is False
    assert result.receipt["snapshot_count"] == result.receipt["unique_provider_count"] == 3


def test_weighted_field_matches_hand_population_stats_and_gaussian_mixture_oracle():
    refs = np.array([[100.,100.], [100.,100.], [700.,100.]])
    weights = np.array([.5,.5,1.], np.float64)
    scores = np.repeat(np.array([[.25,.35,.1]], np.float32), 80, axis=0)
    values = a._weighted_values(scores, refs, weights, np.zeros(2), (9,15), lambda: None)
    raw = scores[0].astype(np.float64)
    mean = sum(float(w*x) for w,x in zip(weights,raw)) / sum(weights)
    std = math.sqrt(sum(float(w*(x-mean)**2) for w,x in zip(weights,raw)) / sum(weights))
    z = np.clip((raw-mean)/max(std,.005), -6., 6.)
    kernels = [math.exp(-((2-x)**2+(2-y)**2)/(2*1.5**2)) for x,y in refs/100.]
    denominator = sum(w*k for w,k in zip(weights,kernels))
    numerator = sum(w*k*math.exp(float(v)) for w,k,v in zip(weights,kernels,z))
    assert values.dtype == np.float32 and values.shape == (80,9,15)
    assert float(values[0,2,2]) == pytest.approx(math.log(numerator/denominator), abs=2e-7)
    np.testing.assert_array_equal(values, np.repeat(values[:1],80,axis=0))
    assert float(values[0,-1,-1]) == -6.


def test_snapshot_copy_with_split_weight_is_mathematically_invariant():
    refs = np.array([[100.,100.], [100.,100.], [700.,100.]])
    scores = np.repeat(np.array([[.25,.35,.1]], np.float32),80,axis=0)
    control = a._weighted_values(scores,refs,np.array([.5,.5,1.]),np.zeros(2),(9,15),lambda:None)
    duplicate = a._weighted_values(scores[:,[0,0,1,2]],refs[[0,0,1,2]],
        np.array([.25,.25,.5,1.]),np.zeros(2),(9,15),lambda:None)
    np.testing.assert_allclose(duplicate,control,rtol=0,atol=2e-7)


def test_weighted_field_full_crs_floor_native_groups_and_every_snapshot():
    net,times,_,_,motion,evidence,facts = case(versions=True)
    app = a.prepare_snapshot(evidence,net,facts,times,motion.times)
    assert app.receipt["snapshot_count"] == 4 and app.receipt["unique_provider_count"] == 3
    assert app.receipt["all_one_weights_used_genuine_original_builder"] is False
    assert app.receipt["all_queries_once"] is True
    assert np.linalg.norm(net.origin)>1_000_000
    local = np.array([[500.,0.]])
    t = float(evidence["query_times"][0])
    value = app.field.sample(net.local_to_crs(local)[:,None,:],np.array([t]))[0,0]
    oracle = math.log(.1+.9*math.exp(float(value)-app.maxima[0]))
    assert app.log_factor(0,net,local,time=t)[0] == pytest.approx(oracle,abs=1e-15)
    assert np.all(app.groups==0) and np.all(app.powers==1/80)
    assert app.field.values.dtype==np.float32
    with pytest.raises(ValueError,match="exact original"):
        app.log_factor(0,net,local,time=t+1e-10)


@pytest.mark.parametrize("fault",["status","profile","equivalence","weight","weight_dtype","coords",
    "raw_hash","duplicate_key","missing_occurrence","count","clock","lag","model"])
def test_snapshot_type_full_mapping_weights_and_native_identity_fail_closed(fault):
    net,times,_,_,motion,evidence,facts = case(versions=True)
    if fault=="status": evidence["status"]="complete_compatible_research_bank"
    if fault=="profile": evidence["profile_sha256"]=a.strict.PROFILE_SHA256
    if fault=="equivalence": evidence["descriptor_equivalence_assumed"]=True
    if fault=="weight": evidence["snapshot_weights"][0]+=.01
    if fault=="weight_dtype": evidence["snapshot_weights"]=evidence["snapshot_weights"].astype(np.float32)
    if fault=="coords": evidence["ref_latlon"][1,0]+=.001
    if fault=="raw_hash": evidence["raw_similarities"][0,0]+=.001
    if fault=="duplicate_key": evidence["mapping"][1]=deepcopy(evidence["mapping"][0])
    if fault=="missing_occurrence": evidence["mapping"][0]["occurrences"].pop()
    if fault=="count": evidence["snapshot_count"]-=1
    if fault=="clock": facts["signed_vo_clock_sha256"]="e"*64
    if fault=="lag": facts["all_physical_calls_and_aliases"][0]["query_to_vo"][0]["composed_lag_s"]+=.001
    if fault=="model": evidence["appearance_identity"]["model"]["weights_sha256"]="e"*64
    with pytest.raises(ValueError): a.prepare_snapshot(evidence,net,facts,times,motion.times)


def test_whole_geometry_extent_refuses_before_any_field_allocation(monkeypatch):
    net,times,_,_,motion,evidence,facts=case(versions=True)
    net.seg_xy0[-1,0]=1e8
    monkeypatch.setattr(a,"_weighted_values",lambda *x:pytest.fail("field before full extent budget"))
    monkeypatch.setattr(a.strict.poi,"build_appearance_field",lambda *x,**y:pytest.fail("field before full extent budget"))
    with pytest.raises(ValueError,match="16M.*no crop"):
        a.prepare_snapshot(evidence,net,facts,times,motion.times)


def test_flat_weighted_rows_use_std_floor_without_nan_or_added_signal():
    scores=np.full((80,3),np.float32(.2))
    values=a._weighted_values(scores,np.array([[100.,100.],[100.,100.],[700.,100.]]),
        np.array([.5,.5,1.]),np.zeros(2),(9,15),lambda:None)
    assert np.isfinite(values).all() and np.max(abs(values[values!=-6.]))==0.


@pytest.mark.parametrize("versions",[False,True])
def test_fixed_fused_tiny_retains_all80_queries_and_no_policy_upgrade(versions):
    net,times,odom,status,_,evidence,facts=case(versions)
    matched=old.matches([(times[-1],[])])
    result=j.align_with_snapshot_appearance(net,odom,status,times,matched,evidence,facts)
    assert result["status"]=="complete_diagnostic_only"
    assert [r["query_row"] for r in result["appearance_events"]]==list(range(80))
    assert result["profile"]["method"]=="offline-all-junction-reference-snapshot-appearance-v1"
    assert result["profile"]["descriptor_equivalence_assumed"] is False
    assert result["accuracy_admitted"] is False and result["selection"] is None
    assert result["config"]["particles"]==50000 and result["config"]["seed"]==0
    if not versions:
        control=j.align_with_appearance(net,odom,status,times,matched,old.case()[5],facts)
        for key,value in control.items():
            if isinstance(value,np.ndarray):np.testing.assert_array_equal(result[key],value)
        assert result["appearance_events"]==control["appearance_events"]


@pytest.mark.parametrize("config",[j.Config(junction_prior="legacy"),
    j.Config(particles=100,junction_prior="conditional-uniform-v3"),
    j.Config(seed=1,junction_prior="conditional-uniform-v3")])
def test_explicit_snapshot_entry_refuses_nonregistered_sampler_config(config):
    with pytest.raises(ValueError,match="conditional-uniform-v3 / 50k / seed0"):
        j.align_with_snapshot_appearance(None,None,None,None,None,None,None,config=config)
