"""Synthetic numerical checks only; no real galleries, models, files or GT."""
from itertools import product
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from monocular_osm import heading_sequence as h


def case(n=3, q=3):
    rng = np.random.default_rng(33)
    return {"view_cosines": rng.uniform(.12, .29, (q, 3, n)),
        "ref_ids": np.array(["ref_"+str(i) for i in range(n)]),
        "ref_latlon": np.column_stack([np.full(n, 52.), np.arange(n)*.001]),
        "query_times": np.arange(q)*7.5,
        "camera_types": ["perspective"]*n,
        "reference_headings_deg": (np.arange(n)*71.+13.) % 360.,
        "crop_offsets_deg": np.array([-17., 0., 17.])}


def arguments(c, **kwargs):
    return h._inputs(**c, vo_yaw_delta_deg=kwargs.get("vo_yaw_delta_deg"),
        vo_quality=kwargs.get("vo_quality"), max_temp_bytes=kwargs.get("max_temp_bytes", h.PROFILE["default_temp_bytes"]),
        max_backpointer_bytes=kwargs.get("max_backpointer_bytes", h.PROFILE["max_joint_backpointer_bytes"]))


def shortest(value):
    return (value+math.pi) % (2*math.pi)-math.pi


def angular(value, sigma, epsilon):
    z = sigma*math.sqrt(2*math.pi)*math.erf(math.pi/(math.sqrt(2)*sigma))
    pdf = (1-epsilon)*math.exp(-.5*(shortest(value)/sigma)**2)/z + epsilon/(2*math.pi)
    peak = (1-epsilon)/z + epsilon/(2*math.pi)
    return math.log(pdf/peak)


def independent_emissions(data, variant, mode):
    scores = data["scores"].astype(float)
    pooled = scores.max(axis=1)
    means = pooled.mean(axis=0) if variant == "reference_mean" else np.zeros(scores.shape[2])
    result = np.empty((len(scores), scores.shape[2], 16))
    for t, ref, heading in product(range(len(scores)), range(scores.shape[2]), range(16)):
        if mode == "erased" or not data["active"][ref]:
            result[t,ref,heading] = (pooled[t,ref]-means[ref])/.02
            continue
        augmented = []
        for angle in h.HEADINGS_DEG:
            per_view = []
            for view in range(scores.shape[1]):
                weights = [math.exp(-.5*(shortest(math.radians(a)+data["offsets"][view]-math.radians(data["angles"][ref]))/math.radians(45))**2)
                    for a in h.HEADINGS_DEG]
                k = int(round(angle/22.5))
                probability = .8*weights[k]/sum(weights)+.2/16
                per_view.append((scores[t,view,ref]-means[ref])/.02+math.log(probability))
            augmented.append(max(per_view))
        base = (pooled[t,ref]-means[ref])/.02
        correction = math.log(sum(math.exp(a-base) for a in augmented)/16)
        result[t,ref,heading] = augmented[heading]-correction
    return result


def independent_transition(xy, dt, mode, yaw, quality):
    n = len(xy); result = np.full((n*16,n*16), -np.inf)
    for old, new in product(range(n*16), repeat=2):
        i, a = divmod(old,16); j, b = divmod(new,16)
        dx, dy = xy[j]-xy[i]; d = math.hypot(dx,dy)
        free = 30+40*dt
        if d > np.nextafter(free+300,np.inf): continue
        first, second = math.radians(a*22.5), math.radians(b*22.5)
        delta = shortest(second-first)
        value = -.5*(max(0,d-free)/150)**2
        value += angular(delta,math.radians(45)*math.sqrt(dt/7.5),.1)
        east, north = math.sin(first)+math.sin(second), math.cos(first)+math.cos(second)
        if d > 20 and math.hypot(east,north) >= 1e-12:
            value += min(1,d/150)*angular(math.atan2(east,north)-math.atan2(dx,dy),math.radians(60),.2)
        if mode == "reference_heading_vo" and math.isfinite(yaw) and quality:
            value += .5*quality*angular(delta-yaw,math.radians(30),.25)
        result[old,new] = value
    return result


def exhaustive(data, variant, mode):
    emissions = independent_emissions(data,variant,mode).reshape(len(data["times"]),-1)
    paths = np.array(list(product(range(emissions.shape[1]), repeat=len(emissions))),np.int64)
    values = sum(emissions[t,paths[:,t]] for t in range(len(emissions)))
    for t,dt in enumerate(np.diff(data["times"])):
        transition = independent_transition(data["xy"],float(dt),mode,float(data["yaw"][t]),float(data["quality"][t]))
        values += transition[paths[:,t],paths[:,t+1]]
    best = values.max()
    # Perdecessor/terminal ties are lexicographic in reverse time, not a
    # promise of lexicographically smallest entire forward path.
    tied = np.flatnonzero(np.abs(values-best) < 1e-12)
    selected = min(tied,key=lambda i:tuple(reversed(paths[i].tolist())))
    return paths[selected],float(best)


def test_profile_declares_fixed_hypotheses_and_null_confidence():
    assert h.PROFILE["appearance_temperature"] == .02
    assert h.PROFILE["reference_heading_sigma_deg"] == 45
    assert h.PROFILE["reference_position_sigma_m"] == 150
    assert h.PROFILE["modes"] == ("erased","reference_heading","reference_heading_vo")
    assert h.PROFILE["candidate_pruning"] is False and h.PROFILE["confidence"] is None
    assert h.PROFILE["primary_route"] == "unsmoothed joint MAP reference/heading path"
    assert h._json_sha(dict(h.PROFILE)) == h.PROFILE_SHA256


def test_heading_mass_is_neutral_after_max_views_and_unknowns_exactly_equal_pool():
    rng=np.random.default_rng(2);scores=rng.uniform(-.3,.4,(3,7))
    active=np.array([True,True,False,False,False,False,False])
    angles=np.array([13.,359.,np.nan,77.,10.,0.,np.nan])
    pooled,headed=h._emission_row(scores,active,angles,np.radians([-17.,0.,17.]),h.PROFILE["default_temp_bytes"])
    assert np.allclose(np.exp(headed-pooled[:,None]).mean(axis=1),1.,rtol=0,atol=3e-15)
    assert np.array_equal(headed[~active],np.broadcast_to(pooled[~active,None],(5,16)))
    assert np.ptp(headed[:2],axis=1).min() > .1


def test_reference_mean_collapses_views_before_time_mean():
    scores=np.array([[[.9],[.1]],[[.1],[.8]]])
    means=h._reference_means(scores,h.PROFILE["default_temp_bytes"])
    assert means.tolist() == [(.9+.8)/2]
    assert means[0] != scores.mean(axis=0).max()


def test_angular_density_ratio_wrap_peak_and_normalization():
    x=np.linspace(-np.pi,np.pi,100001);sigma=np.radians(60);eps=.2
    p=h._angular_penalty(x,sigma,eps)
    peak=(1-eps)/(sigma*np.sqrt(2*np.pi)*math.erf(np.pi/(np.sqrt(2)*sigma)))+eps/(2*np.pi)
    assert np.isclose(np.trapezoid(np.exp(p)*peak,x),1.,atol=1e-10)
    assert abs(float(h._angular_penalty(0.,sigma,eps))) < 1e-15
    assert np.isfinite(h._angular_penalty(np.pi,sigma,eps))
    assert np.allclose(p,h._angular_penalty(x+8*np.pi,sigma,eps),atol=3e-15)


def test_all_six_decoders_match_independent_exhaustive_small_lattice():
    c=case(n=2,q=3);c["query_times"]=np.array([0.,7.5,15.3])
    extra={"vo_yaw_delta_deg":np.array([35.,-10.]),"vo_quality":np.array([.8,.3])}
    data=arguments(c,**extra);result=h.decode_heading_tracks(**c,**extra)
    assert len(result["outputs"]) == 6
    for name, output in result["outputs"].items():
        variant,mode=name.split("/");path,value=exhaustive(data,variant,mode)
        actual=np.array(output["chosen_ref_indices"])*16+np.array(output["heading_bin_indices"])
        assert actual.tolist() == path.tolist(),name
        assert output["objective"] == pytest.approx(value,abs=3e-12)
        assert output["confidence"] is None and not output["smoothed"] and not output["candidate_pruning"]


def test_id_ties_and_input_permutation_are_deterministic():
    c=case();c["view_cosines"][:]=.2;c["ref_ids"]=np.array(["z","a","é"])
    c["ref_latlon"][:]=[52.,0.];c["camera_types"]=[None,"spherical","fisheye"]
    first=h.decode_heading_tracks(**c)
    order=np.array([2,0,1]);changed={k:v for k,v in c.items()}
    for key in ("ref_ids","ref_latlon","reference_headings_deg"):changed[key]=np.asarray(c[key])[order]
    changed["view_cosines"]=c["view_cosines"][:,:,order]
    changed["camera_types"]=[c["camera_types"][i] for i in order]
    second=h.decode_heading_tracks(**changed)
    for name,o in first["outputs"].items():
        assert o["chosen_ref_ids"] == ["a"]*3 and o["heading_bin_indices"] == [0]*3
        assert o["selected_view_indices"] == [0]*3
        assert all(o[k]==second["outputs"][name][k] for k in ("chosen_ref_ids","camera_heading_deg","objective","route_latlon"))


def test_C_F_and_tiny_edge_blocks_have_identical_paths_and_values():
    c=case(n=4);a=h.decode_heading_tracks(**c)
    changed=dict(c,view_cosines=np.asfortranarray(c["view_cosines"]))
    b=h.decode_heading_tracks(**changed,max_temp_bytes=76*1024)
    assert a["outputs"] == b["outputs"] and a["inputs"] == b["inputs"]
    assert max(v["temporary_bound_bytes"] for v in b["allocation"]["steps"]) <= 76*1024
    assert a["allocation"]["joint_backpointer_bytes"] == 6*3*4*16*4


@pytest.mark.parametrize("cameras",[[None]*3,["spherical"]*3,["equirectangular"]*3,["fisheye"]*3,["brown"]*3,["unrecognized"]*3])
def test_unknown_and_panorama_angles_never_activate_heading(cameras):
    c=case();c["camera_types"]=cameras;r=h.decode_heading_tracks(**c)
    for variant in ("raw","reference_mean"):
        o=r["outputs"][variant+"/erased"]
        for mode in ("reference_heading","reference_heading_vo"):
            b=r["outputs"][variant+"/"+mode]
            assert all(o[k]==b[k] for k in ("chosen_ref_ids","heading_bin_indices","objective"))
            assert not any(b["reference_heading_active"])


def test_missing_or_zero_quality_VO_is_neutral():
    c=case();a=h.decode_heading_tracks(**c)
    b=h.decode_heading_tracks(**c,vo_yaw_delta_deg=[90.,-180.],vo_quality=[0.,0.])
    d=h.decode_heading_tracks(**c,vo_yaw_delta_deg=[np.nan,np.nan],vo_quality=[1.,1.])
    assert a["outputs"] == b["outputs"] == d["outputs"]
    for variant in ("raw","reference_mean"):
        no=a["outputs"][variant+"/reference_heading"];yes=a["outputs"][variant+"/reference_heading_vo"]
        assert no["objective"] == yes["objective"] and no["camera_heading_deg"] == yes["camera_heading_deg"]


def test_clockwise_VO_sign_and_wrapping_are_explicit_not_inferred():
    xy=np.zeros((1,2));tree=cKDTree(xy);previous=np.full((6,1,16),-100.);previous[:,:,0]=0.
    plus,_,_=h._advance(previous,tree,xy,7.5,np.pi/2,1.,h.PROFILE["default_temp_bytes"])
    minus,_,_=h._advance(previous,tree,xy,7.5,-np.pi/2,1.,h.PROFILE["default_temp_bytes"])
    wrapped,_,_=h._advance(previous,tree,xy,7.5,np.pi/2+2*np.pi,1.,h.PROFILE["default_temp_bytes"])
    assert 0 < plus[2,0].argmax() < 8
    assert minus[2,0].argmax() == (-plus[2,0].argmax())%16
    assert np.allclose(plus,wrapped,rtol=0,atol=2e-15)
    assert np.array_equal(plus[:2],minus[:2])


def test_actual_dt_changes_physical_edges_without_appearance_pruning():
    xy=np.array([[0.,0.],[700.,0.]]);tree=cKDTree(xy);p=np.full((6,2,16),-100.);p[:,0]=0.
    short,_,a=h._advance(p,tree,xy,7.5,np.nan,0,h.PROFILE["default_temp_bytes"])
    long,_,b=h._advance(p,tree,xy,10.,np.nan,0,h.PROFILE["default_temp_bytes"])
    assert a["radius_m"] == 630. and a["edge_count"] == 2
    assert b["radius_m"] == 730. and b["edge_count"] == 4
    assert short[0,1].max() == pytest.approx(-100.) and long[0,1].max() > -2.


def test_stationary_and_undefined_reversal_bearings_are_neutral_but_turn_is_not():
    previous=np.full((6,1,16),-1000.);previous[:,:,0]=0.
    stationary,_,_=h._advance(previous,cKDTree([[0.,0.]]),np.array([[0.,0.]]),7.5,np.nan,0,h.PROFILE["default_temp_bytes"])
    expected=np.array([angular(math.radians(i*22.5),math.radians(45),.1) for i in range(16)])
    assert np.allclose(stationary[0,0],expected,atol=1e-14)
    xy=np.array([[0.,0.],[100.,0.]]);previous=np.full((6,2,16),-1000.);previous[:,0,0]=0.
    moved,_,_=h._advance(previous,cKDTree(xy),xy,7.5,np.nan,0,h.PROFILE["default_temp_bytes"])
    assert moved[0,1,8] == pytest.approx(expected[8],abs=1e-14)  # No undefined atan2(0,0) heading.
    assert moved[0,1,8] < moved[0,1,0]


def test_radius_boundary_and_neighbor_chunks_preserve_self_and_exact_endpoint():
    xy=np.array([[0.,0.],[630.,0.],[630.01,0.]])
    pairs=set()
    for src,dst,*_ in h._neighbor_blocks(cKDTree(xy),xy,630.,76*1024):pairs.update(zip(src.tolist(),dst.tolist()))
    assert (0,0) in pairs and (0,1) in pairs and (1,0) in pairs
    assert (0,2) not in pairs and (2,0) not in pairs


def test_spatial_cardinality_refuses_before_any_neighbor_lists():
    class Tree:
        def query_ball_point(self,points,*a,**kw):
            assert kw.get("return_length"),"neighbor list allocated before cap"
            return np.array([2_000_001])
    with pytest.raises(ValueError,match="edge budget"):
        next(h._neighbor_blocks(Tree(),np.zeros((1,2)),630.,h.PROFILE["default_temp_bytes"]))


def test_neighbor_list_space_refuses_before_list_materialization():
    class Tree:
        def query_ball_point(self,points,*a,**kw):
            assert kw.get("return_length"),"oversize list materialized"
            return np.full(4,4)
    with pytest.raises(ValueError,match="neighbor-list"):
        next(h._neighbor_blocks(Tree(),np.zeros((4,2)),630.,h.PROFILE["fixed_scratch_bytes"]+h.PROFILE["scratch_bytes_per_edge"]+100))


def test_backpointer_and_temporary_limits_fail_before_explicit_large_allocation(monkeypatch):
    c=case()
    monkeypatch.setattr(h.np,"empty",lambda *a,**k:pytest.fail("allocation before preflight"))
    with pytest.raises(ValueError,match="backpointer allocation"):
        h.decode_heading_tracks(**c,max_backpointer_bytes=1)
    with pytest.raises(ValueError,match="temporary memory"):
        h.decode_heading_tracks(**c,max_temp_bytes=1)
    with pytest.raises(ValueError,match="temporary memory"):
        h.decode_heading_tracks(**c,max_temp_bytes=h.PROFILE["hard_max_temp_bytes"]+1)


@pytest.mark.parametrize("change,match",[
    ({"ref_ids":np.array(["a","a","b"])},"unique"),
    ({"ref_ids":np.array(["a","b","c"],dtype="U1000")},"bounded string"),
    ({"camera_types":["perspective",2,None]},"camera-type"),
    ({"reference_headings_deg":[0.,-1.,20.]},"reference headings"),
    ({"reference_headings_deg":[0.,np.inf,20.]},"reference headings"),
    ({"crop_offsets_deg":[0.,1.]},"crop offsets"),
    ({"query_times":[0.,7.5,7.5]},"increasing"),
    ({"vo_yaw_delta_deg":[np.inf,0.]},"VO increments"),
    ({"vo_quality":[1.,1.1]},"quality"),
    ({"vo_quality":[np.nan,0.]},"quality"),
])
def test_invalid_inputs_decline(change,match):
    c=case();c.update(change)
    with pytest.raises(ValueError,match=match):h.decode_heading_tracks(**c)


def test_inputs_remain_unchanged_and_every_row_reference_is_bound():
    c=case();before={k:(v.copy() if isinstance(v,np.ndarray) else list(v)) for k,v in c.items()}
    result=h.decode_heading_tracks(**c)
    assert result["inputs"]["view_cosines"] == h._array_sha(c["view_cosines"])
    assert result["allocation"]["reference_count"] == 3 and result["allocation"]["heading_states"] == 48
    for key,value in before.items():assert np.array_equal(value,c[key])
    changed=dict(c,view_cosines=c["view_cosines"].copy());changed["view_cosines"][1,2,2]+=.001
    assert h.decode_heading_tracks(**changed)["inputs"]["view_cosines"] != result["inputs"]["view_cosines"]


def test_camera_iterator_is_bound_before_it_is_consumed():
    c=case();expected=h.decode_heading_tracks(**c)
    changed=dict(c,camera_types=iter(c["camera_types"]))
    assert h.decode_heading_tracks(**changed) == expected


@pytest.mark.parametrize("camera_types",[None,"perspective",1])
def test_scalar_camera_metadata_declines(camera_types):
    c=case();c["camera_types"]=camera_types
    with pytest.raises(ValueError,match="camera-type"):h.decode_heading_tracks(**c)


def test_subnormal_positive_actual_interval_has_finite_potentials():
    c=case(n=1,q=2);c["query_times"]=np.array([0.,np.nextafter(0.,1.)])
    with np.errstate(over="raise",invalid="raise",divide="raise"):
        result=h.decode_heading_tracks(**c)
    assert all(np.isfinite(v["objective"]) for v in result["outputs"].values())


def test_erased_ablation_reports_available_headings_as_unused():
    result=h.decode_heading_tracks(**case())
    for name,output in result["outputs"].items():
        assert all(output["reference_heading_available"])
        assert output["reference_heading_active"] == [not name.endswith("/erased")]*3
