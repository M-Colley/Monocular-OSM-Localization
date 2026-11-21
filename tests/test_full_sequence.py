"""Independent exhaustive/dense checks, pruning counterexample and allocation limits."""
import itertools
import math
import numpy as np
import pytest
from monocular_osm import full_sequence as f, sparse_sequence as sparse
from monocular_osm.geo import LocalFrame


def evidence(n=4,q=3,seed=0):
    rng=np.random.default_rng(seed)
    scores=rng.uniform(-.4,.8,(q,n)).astype(np.float32)
    ids=np.array([f'id_{i:05d}' for i in range(n)])
    refs=np.column_stack([np.full(n,42.),-83.+np.arange(n)*.001])
    times=np.cumsum(np.r_[0.,rng.uniform(.1,8.,q-1)])
    return scores,ids,refs,times


def brute(scores,ids,refs,times,origin,center):
    frame=LocalFrame(*origin);xy=np.column_stack(frame.to_xy(refs[:,0],refs[:,1]))
    emissions=scores.astype(float)-(scores.mean(axis=0,dtype=np.float64) if center else 0.)
    best=-float('inf');paths=[]
    for path in itertools.product(range(len(ids)),repeat=len(times)):
        value=float(emissions[0,path[0]])
        for i in range(1,len(times)):
            dx,dy=xy[path[i-1]]-xy[path[i]]
            free=30.+40.*max(float(times[i]-times[i-1]),.5)
            value=value-.02*max(math.hypot(dx,dy)-free,0.)+emissions[i,path[i]]
        if value>best+1e-12:best,paths=value,[path]
        elif abs(value-best)<=1e-12:paths.append(path)
    return best,paths


@pytest.mark.parametrize('seed',range(8))
def test_tiny_matches_independent_all_path_enumeration(seed):
    scores,ids,refs,times=evidence(seed=seed)
    got=f.decode_full_tracks(scores,ids,refs,times)
    for variant in ['raw','reference_mean']:
        best,paths=brute(scores,ids,refs,times,got['inputs']['metric_origin_latlon'],variant=='reference_mean')
        assert got['outputs'][variant]['objective']==pytest.approx(best,abs=1e-12)
        assert tuple(got['outputs'][variant]['chosen_ref_indices']) in paths


@pytest.mark.parametrize('n',[1,63,64,65,127,128,129,257])
def test_block_edges_match_independent_dense_all_state_dp(n):
    scores,ids,refs,times=evidence(n=n,q=4)
    got=f.decode_full_tracks(scores,ids,refs,times)
    frame=LocalFrame(*got['inputs']['metric_origin_latlon']);xy=np.column_stack(frame.to_xy(refs[:,0],refs[:,1]))
    distance=np.hypot(xy[:,None,0]-xy[None,:,0],xy[:,None,1]-xy[None,:,1])
    for variant in ['raw','reference_mean']:
        emissions=scores.astype(float)-(scores.mean(axis=0,dtype=np.float64) if variant=='reference_mean' else 0.)
        value=emissions[0].copy();back=[]
        for row,dt in enumerate(np.diff(times),1):
            objective=value[:,None]-.02*np.maximum(distance-(30+40*max(float(dt),.5)),0)
            pred=objective.argmax(axis=0);value=objective[pred,np.arange(n)]+emissions[row];back.append(pred)
        state=int(value.argmax());path=[state]
        for b in reversed(back):state=int(b[state]);path.append(state)
        assert got['outputs'][variant]['chosen_ref_indices']==path[::-1]
        assert got['outputs'][variant]['objective']==float(value.max())


def test_recovers_globally_good_state_pruned_from_every_sparse_query():
    scores,ids,refs,times=evidence(n=129,q=2)
    refs[:,0]=0.;refs[:64,1]=0.;refs[64:128,1]=.01;refs[128,1]=.005;times[:]=[0.,.1]
    scores[0,:64]=.5;scores[1,:64]=-.5;scores[0,64:128]=-.5;scores[1,64:128]=.5;scores[:,128]=.4
    exact=f.decode_full_tracks(scores,ids,refs,times)['outputs']['raw']
    pruned=sparse.decode_sparse_tracks(scores,ids,refs,times)['outputs']['raw']
    assert exact['chosen_ref_indices']==[128,128] and exact['objective']==pytest.approx(.8)
    assert exact['objective']>pruned['objective'] and exact['candidate_pruning'] is False


def test_actual_intervals_and_half_second_floor_change_transitions():
    scores=np.array([[1.,0.],[1.,0.],[0.,1.]])
    ids=np.array(['a','b']);refs=np.array([[0.,0.],[0.,.003]])
    short=f.decode_full_tracks(scores,ids,refs,[0.,.1,.2])['outputs']['raw']
    long=f.decode_full_tracks(scores,ids,refs,[0.,.1,20.])['outputs']['raw']
    assert short['chosen_ref_indices']==[0,0,0] and long['chosen_ref_indices']==[0,0,1]


def test_id_ties_and_permutation_stable_routes_and_objectives():
    scores,ids,refs,times=evidence(129,3);scores[:]=0.;ids=ids[::-1]
    first=f.decode_full_tracks(scores,ids,refs,times)
    perm=np.random.default_rng(3).permutation(len(ids));second=f.decode_full_tracks(scores[:,perm],ids[perm],refs[perm],times)
    for variant in ['raw','reference_mean']:
        assert first['outputs'][variant]['chosen_ref_ids']==[min(ids)]*3
        assert second['outputs'][variant]['chosen_ref_ids']==first['outputs'][variant]['chosen_ref_ids']
        assert second['outputs'][variant]['route_latlon']==first['outputs'][variant]['route_latlon']


def test_unpruned_sparse_is_bitwise_identical_for_shared_objective():
    scores,ids,refs,times=evidence(31,8,4)
    exact=f.decode_full_tracks(scores,ids,refs,times);small=sparse.decode_sparse_tracks(scores,ids,refs,times)
    for variant in ['raw','reference_mean']:
        for key in ['objective','route_latlon','raw_route_latlon','chosen_ref_ids','selected_emissions']:
            assert exact['outputs'][variant][key]==small['outputs'][variant][key]


def test_maximum_reference_count_never_allocates_global_square(monkeypatch):
    scores,ids,refs,times=evidence(10000,2)
    scores[:]=0.
    original=np.hypot;shapes=[]
    def bounded(x,y,*args,**kwargs):
        shape=np.broadcast_shapes(np.shape(x),np.shape(y))
        if len(shape)==2:
            assert shape[0]==10000 and shape[1]<=64
            shapes.append(shape)
        return original(x,y,*args,**kwargs)
    monkeypatch.setattr(f.np,'hypot',bounded)
    got=f.decode_full_tracks(scores,ids,refs,times)
    assert shapes and max(a*b for a,b in shapes)==640000
    assert got['allocation']['distance_objective_temporary_values']==1920000
    assert got['allocation']['backpointer_shape_per_variant']==[2,10000]
    assert got['allocation']['backpointer_tables']==2
    assert got['outputs']['raw']['chosen_ref_ids']==[ids[0]]*2


def test_nontrivial_permutation_and_c_f_layout_preserve_semantics():
    scores,ids,refs,times=evidence(259,7,12)
    first=f.decode_full_tracks(scores,ids,refs,times)
    assert first==f.decode_full_tracks(np.asfortranarray(scores),ids,refs,times)
    perm=np.random.default_rng(9).permutation(len(ids))
    second=f.decode_full_tracks(scores[:,perm],ids[perm],refs[perm],times)
    for variant in ['raw','reference_mean']:
        for key in ['objective','chosen_ref_ids','route_latlon','selected_raw_cosines','selected_emissions']:
            assert second['outputs'][variant][key]==first['outputs'][variant][key]


@pytest.mark.parametrize('fault',['one_query','too_many_queries','too_many_refs','nan','infinite','out_of_cosine',
    'float16','missing_id','duplicate_id','bad_coordinate','backwards_time','overflow_interval','antipodal'])
def test_invalid_or_unbounded_input_refused_before_transition_work(fault,monkeypatch):
    scores,ids,refs,times=evidence()
    if fault=='one_query':scores,times=scores[:1],times[:1]
    if fault=='too_many_queries':scores,times=np.zeros((81,4)),np.arange(81.)
    if fault=='too_many_refs':scores,ids,refs,times=evidence(10001,2)
    if fault=='nan':scores[0,0]=np.nan
    if fault=='infinite':scores[0,0]=np.inf
    if fault=='out_of_cosine':scores[0,0]=1.1
    if fault=='float16':scores=scores.astype(np.float16)
    if fault=='missing_id':ids=ids[:-1]
    if fault=='duplicate_id':ids[1]=ids[0]
    if fault=='bad_coordinate':refs[0,0]=91.
    if fault=='backwards_time':times=times[::-1]
    if fault=='overflow_interval':times=np.array([0.,1.,1e308])
    if fault=='antipodal':refs[:,1]=[0.,180.,0.,180.]
    monkeypatch.setattr(f,'_decode_all',lambda *args:pytest.fail('must refuse before transitions'))
    with pytest.raises(ValueError):f.decode_full_tracks(scores,ids,refs,times)
