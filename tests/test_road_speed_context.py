"""Tiny fabricated sealed packets; no Root certificates, maps or GT."""
from copy import deepcopy
import hashlib
import json
import networkx as nx
import numpy as np
import osmnx as ox
import pytest

from monocular_osm import road_speed_context as context
from monocular_osm import road_segment_semantics as sem
from monocular_osm.graph_filter import RoadNetwork, Odometry
from monocular_osm.junction_sequence import coarsen_motion


def case():
    graph = nx.MultiDiGraph(crs="EPSG:32632")
    for n in (1, 2, 3): graph.add_node(n, x=455000. + n * 50., y=5428000.)
    graph.add_edge(1, 2, key=7, osmid=100, highway="primary", maxspeed="50", length=50.)
    graph.add_edge(2, 3, key=8, osmid=101, highway="primary", maxspeed="100", length=50.)
    records = []; catalog = sem.attach(graph, emit_record=records.append)
    graph = ox.simplification.simplify_graph(graph, remove_rings=False,
        edge_attr_aggs={"length": sum, "travel_time": sum, sem.TOKEN: sem.concat_paths})
    trace = []; net = RoadNetwork.from_graph(graph, construction_trace=trace)
    for row in trace: row["source_keyed_path"] = graph.edges[row["edge_key"]][sem.TOKEN]
    return net, sem.bind_mapping(catalog, trace, net), records


def reseal_catalog(packet, records):
    data = b"".join(sem.record_bytes(r) for r in records)
    packet["metadata"]["catalog_sha256"] = hashlib.sha256(data).hexdigest()
    packet["metadata"]["catalog_bytes"] = len(data)


def make(net=None, packet=None, records=None):
    if net is None: net, packet, records = case()
    return context.prepare(net, packet, iter(records),
        catalog_sha256=packet["metadata"]["catalog_sha256"],
        catalog_bytes=packet["metadata"]["catalog_bytes"])


def test_full_base_copy_known_turn_binding_and_exact_boundary_lookup():
    original, packet, records = case()
    base, ctx = make(original, packet, records)
    assert sem.net_binding(base) == sem.net_binding(original)
    runtime = ctx.bind_runtime(base.with_turn_prior(0., normalise=True))
    out = ctx.lookup_previous(runtime, np.array([0, 0, 0]), np.array([0., 50., 100.]))
    np.testing.assert_array_equal(out["segment"], [0, 1, 1])
    np.testing.assert_array_equal(out["mps"], [50. / 3.6, 100. / 3.6, 100. / 3.6])
    assert out["eligible"].all()
    assert ctx.receipt["known_turn_transform_verified"] is True
    assert ctx.receipt["Root_authority_admitted"] is False
    assert ctx.receipt["segment_mps_sha256"] == sem.typed_sha(ctx.segment_mps)
    assert ctx.receipt["segment_status_sha256"] == sem.typed_sha(ctx.segment_status)


def test_OWN_bytes_prevents_alias_or_write_flag_drift_and_metadata_is_copied():
    old, packet, records = case(); base, ctx = make(old, packet, records)
    old.edge_base[:] = 999.
    runtime = ctx.bind_runtime(base.with_turn_prior(0., normalise=True))
    for array in (base.edge_base, base.seg_gstart, runtime.succ_logprior, ctx.segment_mps, ctx.segment_status):
        with pytest.raises(ValueError): array.setflags(write=True)
    with pytest.raises(AttributeError): ctx._mps = np.zeros(2)
    receipt = ctx.receipt; receipt["catalog_bytes"] = -1
    assert ctx.receipt["catalog_bytes"] > 0
    packet["arrays"]["segment_raw_edge_index"][:] = -1
    assert ctx.lookup_previous(runtime, np.array([0]), np.array([0.]))["eligible"][0]


def test_runtime_lookup_does_not_hash_or_parse_any_map(monkeypatch):
    base, ctx = make(); runtime = ctx.bind_runtime(base.with_turn_prior(0., normalise=True))
    monkeypatch.setattr(sem, "net_binding", lambda *a: pytest.fail("map rehash at runtime"))
    monkeypatch.setattr(sem, "typed_sha", lambda *a: pytest.fail("map rehash at runtime"))
    monkeypatch.setattr(sem, "parse_maxspeed", lambda *a: pytest.fail("late parsing"))
    assert ctx.lookup_previous(runtime, np.array([0]), np.array([49.]))["mps"][0] > 0
    runtime.edge_base = runtime.edge_base.copy()
    with pytest.raises(ValueError, match="replaced"):
        ctx.lookup_previous(runtime, np.array([0]), np.array([0.]))


@pytest.mark.parametrize("value,kind,reason", [
    (None, "builtins.NoneType", "unsupported_type"), (True, "builtins.bool", "unsupported_type"),
    ([50, 100], "builtins.list", "unsupported_type"), ("signals", "builtins.str", "unsupported_base"),
    ("50;70", "builtins.str", "unsupported_base"), (0, "builtins.int", "nonpositive_or_nonfinite"),
    ("NaN", "builtins.str", "unsupported_base")])
def test_only_scalar_c476_base_context_else_neutral(value, kind, reason):
    net, packet, records = case()
    records[0]["attributes"]["maxspeed"] = {"type": kind, "value": value}
    reseal_catalog(packet, records)
    base, ctx = make(net, packet, records)
    assert ctx.segment_mps[0] == 0 and ctx.segment_status[0] == context.STATUS[reason]


def test_present_override_and_negative_mapping_never_borrow_base_or_final_record():
    net, packet, records = case()
    records[0]["attributes"]["maxspeed:forward"] = {"type": "builtins.str", "value": "120"}
    reseal_catalog(packet, records)
    arrays = packet["arrays"]
    arrays["segment_raw_edge_index"][1] = -1
    arrays["segment_semantic_status"][1] = sem.SEG_STATUS["ambiguous_parallel_neutral"]
    for name in ("segment_raw_edge_index", "segment_semantic_status"):
        a = arrays[name]; packet["metadata"]["array_headers"][name]["sha256"] = sem.typed_sha(a)
    _, ctx = make(net, packet, records)
    np.testing.assert_array_equal(ctx.segment_mps, [0., 0.])
    np.testing.assert_array_equal(ctx.segment_status, [context.STATUS["unsupported_override"], context.STATUS["parallel_mapping"]])


@pytest.mark.parametrize("fault", ["short", "extra", "index", "key", "hash", "source_drift", "packet_drift"])
def test_entire_catalog_and_final_drift_checks(fault):
    net, packet, records = case()
    rows = deepcopy(records)
    if fault == "short": rows = rows[:1]
    if fault == "extra": rows.append(rows[-1])
    if fault == "index": rows[-1]["index"] = 0
    if fault == "key": rows[-1]["key"][-1] += 1
    if fault == "hash": rows[-1]["attributes"]["maxspeed"]["value"] = "90"
    def generate():
        yield from rows
        if fault == "source_drift": net.edge_len[0] += 1.
        if fault == "packet_drift": packet["arrays"]["raw_edge_keys"][0, 0] += 1
    with pytest.raises(ValueError):
        context.prepare(net, packet, generate(), catalog_sha256=packet["metadata"]["catalog_sha256"],
                        catalog_bytes=packet["metadata"]["catalog_bytes"])


def test_neutral_only_still_exhausts_every_catalog_record():
    net, packet, records = case()
    for row in records: row["attributes"].pop("maxspeed")
    reseal_catalog(packet, records)
    observed = []
    def generate():
        for row in records:
            observed.append(row["index"])
            yield row
    _, ctx = context.prepare(net, packet, generate(), catalog_sha256=packet["metadata"]["catalog_sha256"],
                             catalog_bytes=packet["metadata"]["catalog_bytes"])
    assert observed == [0, 1] and not ctx.segment_mps.any()


@pytest.mark.parametrize("fault", ["normalised", "profile", "rawcap", "bytecap", "dtype", "header", "minus_one"])
def test_bad_binding_cap_or_packet_refused_before_iterator(fault, monkeypatch):
    net, packet, records = case()
    if fault == "normalised": net = net.with_turn_prior(0., normalise=True)
    if fault == "profile": packet["metadata"]["profile_sha256"] = "f" * 64
    if fault == "rawcap": monkeypatch.setattr(context, "MAX_RAW", 1)
    if fault == "bytecap": monkeypatch.setattr(context, "MAX_BASE_BYTES", 0)
    if fault == "dtype": packet["arrays"]["raw_edge_keys"] = packet["arrays"]["raw_edge_keys"].astype(np.int32)
    if fault == "header": packet["metadata"]["array_headers"]["raw_edge_keys"]["sha256"] = "0" * 64
    if fault == "minus_one":
        packet["arrays"]["segment_raw_edge_index"][0] = -1
        a = packet["arrays"]["segment_raw_edge_index"]
        packet["metadata"]["array_headers"]["segment_raw_edge_index"]["sha256"] = sem.typed_sha(a)
    def rows():
        pytest.fail("catalog body reached before invalid binding/cap")
        yield
    with pytest.raises(ValueError):
        context.prepare(net, packet, rows(), catalog_sha256=packet["metadata"]["catalog_sha256"],
                        catalog_bytes=packet["metadata"]["catalog_bytes"])


def test_record_content_bound_precedes_canonical_encoding(monkeypatch):
    monkeypatch.setattr(sem, "record_bytes", lambda *a: pytest.fail("encoding oversized body"))
    with pytest.raises(ValueError, match="content cap"):
        context._record_bytes({"private": "x" * 9000})


@pytest.mark.parametrize("statuses,steps,eligible", [
    (["measured", "measured"], [1., 1.], True),
    (["measured", "unknown"], [1., 0.], False),
    (["measured", "stationary"], [1., 0.], False),
    (["unknown", "unknown"], [0., 0.], False)])
def test_exact_native_moving_support_not_compensated_predict_step(statuses, steps, eligible):
    odom = Odometry(steps, [0., 0.], [.25, .25], metric=False, unit_seconds=.25)
    motion = coarsen_motion(odom, statuses, np.array([0., .25, .5]))
    schedule = context.moving_schedule(motion)
    assert schedule["eligible"].tolist() == [eligible]
    if statuses == ["unknown", "unknown"]: assert schedule["parent_step"][0] > 0
    for array in schedule.values():
        with pytest.raises(ValueError): array.setflags(write=True)


def test_terminal_partial_is_eligible_but_forged_counts_fail():
    odom = Odometry([1., 1., 1.], [0., 0., 0.], [.25] * 3, metric=False, unit_seconds=.25)
    motion = coarsen_motion(odom, ["measured"] * 3, np.array([0., .25, .5, .75]))
    assert context.moving_schedule(motion)["eligible"].tolist() == [True, True]
    motion.counts[1, 1] = 1
    with pytest.raises(ValueError, match="inconsistent"):
        context.moving_schedule(motion)
