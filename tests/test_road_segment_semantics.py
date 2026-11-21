"""Invented graphs only; no actual maps, cases, models or GT."""
from copy import deepcopy
import hashlib,json
import networkx as nx
import numpy as np
import osmnx as ox
import pytest
from monocular_osm import road_segment_semantics as s
from monocular_osm.graph_filter import RoadNetwork

def chain(points=((0.,0.),(10.,0.),(20.,0.))):
    g=nx.MultiDiGraph(crs='EPSG:3857')
    for i,p in enumerate(points):g.add_node(i+1,x=p[0],y=p[1])
    for i in range(len(points)-1):
        g.add_edge(i+1,i+2,key=7+i,length=10.,osmid=100+i,highway='primary',
                   maxspeed=('30','50 mph')[i%2],ref='R'+str(i),oneway=True,reversed=False)
    return g

def prepared(g,*,simplify=True):
    plain=g.copy();rows=[];cat=s.attach(g,emit_record=rows.append)
    if simplify:
        plain=ox.simplification.simplify_graph(plain,remove_rings=False)
        g=ox.simplification.simplify_graph(g,remove_rings=False,track_merged=True,
            edge_attr_aggs={'length':sum,'travel_time':sum,s.TOKEN:s.concat_paths})
    before=RoadNetwork.from_graph(plain);trace=[];net=RoadNetwork.from_graph(g,construction_trace=trace)
    for row in trace:row['source_keyed_path']=g.edges[row['edge_key']][s.TOKEN]
    packet=s.bind_mapping(cat,trace,net)
    for name in s.NET_ARRAYS:
        np.testing.assert_array_equal(np.asarray(getattr(before,name)),np.asarray(getattr(net,name)))
    return cat,rows,trace,net,packet

def test_catalog_order_loaded_types_all18_geometry_and_two_sources():
    g=chain();expected=list(g.edges(keys=True));cat,rows,trace,net,packet=prepared(g)
    assert [r['key'] for r in rows]==[list(k) for k in expected]
    assert rows[0]['attributes']['maxspeed']=={'type':'builtins.str','value':'30'}
    assert rows[0]['attributes']['oneway']=={'type':'builtins.bool','value':True}
    b=b''.join(s.record_bytes(r) for r in rows)
    assert cat.encoded_sha256==hashlib.sha256(b).hexdigest() and cat.encoded_bytes==len(b)
    assert set(packet['arrays'])==set(s.ARRAY_NAMES)
    assert packet['arrays']['compact_path_raw_edge_index'].tolist()==[0,1]
    assert packet['arrays']['segment_raw_edge_index'].tolist()==[0,1]
    assert packet['arrays']['segment_semantic_status'].tolist()==[1,1]

def test_exact_boundary_endpoint_and_wrong_network_binding():
    _,_,_,net,packet=prepared(chain())
    edge=np.zeros(4,np.int64);offset=np.array([0.,np.nextafter(10.,0.),10.,20.])
    actual=s.lookup_previous(net,packet,edge,offset)
    np.testing.assert_array_equal(actual['segment_index'],net.locate(edge,offset)[3])
    assert actual['raw_edge_index'].tolist()==[0,0,1,1]
    net.edge_len[0]+=1.
    with pytest.raises(ValueError,match='binding'):s.lookup_previous(net,packet,edge,offset)

def test_parallel_omissions_retained_and_whole_path_neutral():
    g=chain();g.add_edge(1,2,key=9,length=10.,maxspeed='90');g.add_edge(2,3,key=10,length=10.,maxspeed='100')
    cat,rows,trace,net,packet=prepared(g);a=packet['arrays']
    assert len(rows)==4 and len(cat.keys)==4
    omitted=np.flatnonzero(a['raw_edge_disposition']&s.RAW_STATUS['omitted_parallel'])
    assert set(map(tuple,cat.keys[omitted]))=={(1,2,9),(2,3,10)}
    assert np.all(a['segment_semantic_status']==s.SEG_STATUS['ambiguous_parallel_neutral'])
    assert np.all(a['segment_raw_edge_index']==-1)

@pytest.mark.parametrize('points,status,flag',[
    (((0.,0.),(0.,0.),(10.,0.)),1,'zero_chord'),
    (((0.,0.),(1e-7,0.),(10.,0.)),3,'tiny_chord'),
    (((0.,0.),(0.,0.)),4,'zero_chord')])
def test_zero_tiny_and_synthetic_statuses(points,status,flag):
    _,_,_,_,packet=prepared(chain(points))
    assert np.all(packet['arrays']['segment_semantic_status']==status)
    assert np.any(packet['arrays']['raw_edge_disposition']&s.RAW_STATUS[flag])

def test_raw_geometry_preserved_but_neutral_without_subvertex_authority():
    from shapely.geometry import LineString
    g=chain(((0.,0.),(10.,0.)))
    g.edges[1,2,7]['geometry']=LineString([(0.,0.),(5.,1.),(10.,0.)])
    _,rows,_,_,packet=prepared(g,simplify=False)
    assert rows[0]['attributes']['geometry']['value']['coordinates']==[[0.,0.],[5.,1.],[10.,0.]]
    assert rows[0]['attributes']['geometry']['value']['wkb_hex']==g.edges[1,2,7]['geometry'].wkb_hex
    assert np.all(packet['arrays']['segment_semantic_status']==5)
    assert np.all(packet['arrays']['segment_raw_edge_index']==-1)

def test_directed_reverse_paths_preserve_distinct_raw_keys_and_all18():
    g=chain()
    for u,v,k,d in list(g.edges(keys=True,data=True)):g.add_edge(v,u,key=k+100,**{**d,'reversed':True})
    cat,rows,trace,net,packet=prepared(g)
    assert len(net.edge_keys)==2 and len(cat.keys)==4
    assert np.all(packet['arrays']['segment_semantic_status']==1)
    assert sorted(packet['arrays']['segment_raw_edge_index'].tolist())==[0,1,2,3]

@pytest.mark.parametrize('value,expected', [('36',10.),('36 km/h',10.),(' 10 MPH ',4.4704),
    (36,10.),(36.,10.),(np.int64(36),10.),('.36 km/h',.1)])
def test_supported_scalar_units_fullmatch_context_only(value,expected):
    parsed=s.parse_maxspeed({'maxspeed':value})
    assert parsed['status']=='base_numeric_context' and parsed['mps']==pytest.approx(expected)
    assert parsed['legal_applicability_verified'] is False
    assert parsed['source_way_orientation']=='unknown'

@pytest.mark.parametrize('value',[None,True,0,-1,np.nan,np.inf,10**1000,'0','-30','+30','30;50',
    '30|50','30-50','DE:urban','walk','none','50 kph','50 kmh','text 50','50 mph text',[30],(30,),'NaN'])
def test_unsupported_or_overflow_speed_neutral(value):
    result=s.parse_maxspeed({'maxspeed':value})
    assert result['status']=='neutral' and result['mps'] is None

@pytest.mark.parametrize('name',['maxspeed:forward','maxspeed:backward','maxspeed:conditional','maxspeed:lanes','maxspeed:advisory'])
def test_unsupported_overrides_never_borrow_base(name):
    result=s.parse_maxspeed({'maxspeed':'50',name:'30'})
    assert result['status']=='neutral' and result['reason']=='unsupported_override'

def test_loaded_list_numpy_scalar_types_and_depth_preserved():
    g=chain();g.edges[1,2,7]['custom']=[np.int64(4),('a',False)]
    rows=[];s.attach(g,emit_record=rows.append)
    value=rows[0]['attributes']['custom']
    assert value['type']=='builtins.list'
    assert value['value'][0]=={'type':'numpy.int64','value':4}
    assert value['value'][1]['type']=='builtins.tuple'
    assert s.PROFILE['original_OSM_coverage']=='unknown; graph-artifact fields only'

def test_namespace_collision_refuses_before_emit_or_mutation():
    g=chain();g.edges[2,3,8][s.TOKEN]=('foreign',)
    with pytest.raises(ValueError,match='collision'):s.attach(g,emit_record=lambda _:pytest.fail('partial emit'))
    assert s.TOKEN not in g.edges[1,2,7]

@pytest.mark.parametrize('kind',['string','depth','nonfinite'])
def test_record_caps_before_emit_tokens(kind):
    g=chain()
    if kind=='string':value='x'*(s.MAX_RECORD_BYTES+1)
    elif kind=='depth':value=[[[[['x']]]]]
    else:value=np.nan
    g.edges[1,2,7]['bad']=value
    with pytest.raises(ValueError):s.attach(g,emit_record=lambda _:pytest.fail('oversize emit'))
    assert all(s.TOKEN not in d for _,_,d in g.edges(data=True))

def test_callback_late_raw_drift_refused_and_no_tokens():
    g=chain();rows=[]
    def emit(row):
        rows.append(row)
        if row['index']==1:g.edges[1,2,7]['maxspeed']='99'
    with pytest.raises(ValueError,match='drift'):s.attach(g,emit_record=emit)
    assert all(s.TOKEN not in d for _,_,d in g.edges(data=True))

def test_aggregate_catalog_cap_preserves_no_partial_authority(monkeypatch):
    g=chain();rows=[];monkeypatch.setattr(s,'MAX_CATALOG_BYTES',1)
    with pytest.raises(ValueError,match='aggregate'):s.attach(g,emit_record=rows.append)
    assert not rows and all(s.TOKEN not in d for _,_,d in g.edges(data=True))

@pytest.mark.parametrize('kind',['key','chain','trace','keep'])
def test_wrong_membership_and_construction_trace_refused(kind):
    cat,_,trace,net,_=prepared(chain());bad=deepcopy(trace)
    if kind=='key':bad[0]['source_keyed_path']=((1,2,999),(2,3,8))
    if kind=='chain':bad[0]['source_keyed_path']=((2,3,8),(1,2,7))
    if kind=='trace':bad[0]['compact_vertices'][1,0]+=.1
    if kind=='keep':bad[0]['keep_indices'][1]=0
    with pytest.raises(ValueError):s.bind_mapping(cat,bad,net)

def test_semantic_array_drift_refused_by_lookup():
    _,_,_,net,p=prepared(chain());p['arrays']['segment_raw_edge_index'][0]=1
    with pytest.raises(ValueError,match='drift'):s.lookup_previous(net,p,np.array([0]),np.array([0.]))

def test_typed_budget_and_membership_cap_before_allocation(monkeypatch):
    cat,_,trace,net,_=prepared(chain())
    with monkeypatch.context() as patch:
        patch.setattr(s,'MAX_ARRAY_BYTES',1)
        patch.setattr(s.np,'empty',lambda *a,**k:pytest.fail('overbudget allocation'))
        with pytest.raises(ValueError,match='budget'):s.bind_mapping(cat,trace,net)
    with monkeypatch.context() as patch:
        patch.setattr(s,'MAX_MEMBERS',1)
        with pytest.raises(ValueError,match='membership cap'):s.bind_mapping(cat,trace,net)

def test_raw_cap_before_index_arrays(monkeypatch):
    from types import SimpleNamespace as NS
    graph=NS(is_directed=lambda:True,is_multigraph=lambda:True,
        number_of_edges=lambda:s.MAX_RAW+1,number_of_nodes=lambda:2)
    monkeypatch.setattr(s.np,'empty',lambda *a,**k:pytest.fail('overcap index allocation'))
    with pytest.raises(ValueError,match='raw graph cap'):s.attach(graph,emit_record=lambda _:None)

def test_raw_geometry_cap_before_ndarray_copy(monkeypatch):
    class TooLong:
        geom_type='LineString'
        coords=range(129)
    monkeypatch.setattr(s.np,'asarray',lambda *a,**k:pytest.fail('oversize raw geometry copy'))
    with pytest.raises(ValueError,match='before copy'):s._value(TooLong(),[8192])

def test_concat_exact_order_nonzero_keys_and_overcap(monkeypatch):
    assert s.concat_paths([((2,3,9),),((3,4,7),)])==((2,3,9),(3,4,7))
    with pytest.raises(ValueError):s.concat_paths([((True,3,4),)])
    monkeypatch.setattr(s,'MAX_MEMBERS',1)
    with pytest.raises(ValueError,match='cap'):s.concat_paths([((2,3,9),),((3,4,7),)])

def test_original_way_orientation_never_promoted():
    g=nx.MultiDiGraph(crs='EPSG:3857');g.add_node(1,x=0.,y=0.);g.add_node(2,x=10.,y=0.)
    ox.graph._add_paths(g,[{'nodes':[1,2],'osmid':9,'oneway':'-1'}],False)
    rows=[];s.attach(g,emit_record=rows.append)
    assert rows[0]['key'][:2]==[2,1]
    assert rows[0]['attributes']['reversed']['value'] is False
    assert s.PROFILE['source_way_orientation'].startswith('unknown')

def test_profile_and_all_fixed_limits_immutable():
    digest=hashlib.sha256(json.dumps(dict(s.PROFILE),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert digest==s.PROFILE_SHA256
    assert (s.MAX_RAW,s.MAX_COMPACT,s.MAX_SEGMENTS,s.MAX_MEMBERS)==(500000,100000,500000,1000000)
    assert s.MAX_CATALOG_BYTES==192<<20 and s.MAX_RECORD_BYTES==8192 and s.MAX_ARRAY_BYTES==64<<20
    assert s.MAX_DEPTH==4 and not s.PROFILE['motion_model_or_policy_changed']
    with pytest.raises(TypeError):s.PROFILE['raw_edges']=2

def test_all_records_emitted_before_any_token_added():
    g=chain();records=[]
    def emit(row):
        assert all(s.TOKEN not in d for _,_,d in g.edges(data=True))
        records.append(row)
    s.attach(g,emit_record=emit)
    assert len(records)==2 and all(s.TOKEN in d for _,_,d in g.edges(data=True))

def test_depth_four_allowed_and_record_mutation_refused():
    g=chain();g.edges[1,2,7]['depth_four']=[[[['value']]]]
    rows=[];s.attach(g,emit_record=rows.append);assert len(rows)==2
    g=chain()
    def mutate(row):row['index']=999
    with pytest.raises(ValueError,match='mutated record'):s.attach(g,emit_record=mutate)
    assert all(s.TOKEN not in d for _,_,d in g.edges(data=True))

def test_raw_key_int64_bounds_and_bool_rejected():
    assert s._key((-(1<<63),(1<<63)-1,0))==(-(1<<63),(1<<63)-1,0)
    for value in [(1<<63,2,0),(1,2,-(1<<63)-1),(True,2,0),(1.,2,0)]:
        with pytest.raises(ValueError,match='int64'):s._key(value)

def test_signed_negative_coordinates_repeat_loop_and_untouched_parallel():
    g=chain(((-30.,-20.),(-20.,-20.),(-20.,-10.)))
    g.add_edge(3,1,key=99,length=14.,maxspeed='20')
    _,_,_,_,packet=prepared(g)
    assert np.all(packet['arrays']['segment_semantic_status']==1)
    g=chain(((0.,0.),(10.,0.)));g.add_edge(1,2,key=99,length=10.,maxspeed='60')
    _,_,_,_,packet=prepared(g,simplify=False)
    assert np.all(packet['arrays']['segment_semantic_status']==1)
    assert not np.any(packet['arrays']['raw_edge_disposition']&s.RAW_STATUS['omitted_parallel'])

def test_unprojected_endpoints_match_genuine_projected_trace():
    g=chain(((8.,48.),(8.0001,48.0001),(8.0002,48.)))
    g.graph['crs']='EPSG:4326'
    _,_,_,_,packet=prepared(g)
    assert np.all(packet['arrays']['segment_semantic_status']==1)
