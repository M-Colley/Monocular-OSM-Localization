"""Pure single-city planning/topology adapter and correlated OCR event policy.

No HTTP, files, video, models or registrations. Acquisition history remains the
caller's responsibility: this module plans requests and checks supplied objects.
It never converts bare numeric detections into EXIT evidence or snaps to a road.
"""
from copy import deepcopy
import math

import networkx as nx
import numpy as np
from pyproj import Transformer

from .road_sign_clues import _digest, _ref, plan_motorway_junction_inventory


CORRELATION_PROFILE = "same-ref-first-observation-maxspan8s-geometric-mean-v2"
KEEP_NODE = "junction_sequence_keep_inventory_node"
MAX_SEGMENTS = 500_000
MAX_EDGES = 100_000
MAX_DEGREE = 32
MAX_SUCCESSOR_ENTRIES = 1_000_000


def _groups(matched):
    rows = matched.get("matches")
    if matched.get("status") != "complete" or not isinstance(rows, list):
        raise ValueError("complete matched road-sign inventory required")
    by_ref = {}
    for index, row in enumerate(rows):
        obs = row["observation"]
        ref, stamp = obs.get("ref"), obs.get("t_sec")
        if (not isinstance(ref, str) or _ref(ref) != ref or isinstance(stamp, bool)
                or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp < 0):
            raise ValueError("normalized literal refs and finite native times required")
        by_ref.setdefault(ref, []).append((float(stamp), index))
    groups = []
    for ref in sorted(by_ref):
        current, first = [], None
        for stamp, index in sorted(by_ref[ref]):
            if current and stamp - first > 8.:
                groups.append((ref, first, current))
                current = []
            if not current:
                first = stamp
            current.append(index)
        if current:
            groups.append((ref, first, current))
    annotations = [None] * len(rows)
    for rank, (ref, first, indices) in enumerate(groups):
        annotation = {"group_id": f"event{rank:04d}", "count": len(indices), "power": 1. / len(indices)}
        for index in indices:
            annotations[index] = annotation.copy()
    return annotations


def annotate_correlations(matched):
    """Retain every row/time/candidate; same-ref events have max-span <=8s.

    Repeats contribute a geometric mean of already mixed/outlier sign factors,
    not independent landmark detections. This is a conservative heuristic, not
    calibrated effective sample size. 0/7/14 seconds forms TWO events.
    """
    annotations = _groups(matched)
    result = deepcopy(matched)
    result["correlation_profile"] = CORRELATION_PROFILE
    for row, annotation in zip(result["matches"], annotations):
        # Derived metadata lives outside the original observation, whose frozen
        # observations_sha256 continues to describe exactly its original body.
        row["correlation"] = annotation
    return result


def validate_correlations(matched):
    """Recompute from ALL matches before map/motion filtering; legacy =>1."""
    rows = matched["matches"]
    supplied = [row.get("correlation") for row in rows]
    if "correlation_profile" not in matched and all(x is None for x in supplied):
        return [{"group_id": None, "count": 1, "power": 1.} for _ in rows]
    if matched.get("correlation_profile") != CORRELATION_PROFILE:
        raise ValueError("unrecognized correlated-event profile")
    expected = _groups(matched)
    for actual, wanted in zip(supplied, expected):
        if (not isinstance(actual, dict) or set(actual) != set(wanted)
                or type(actual.get("count")) is not int or type(actual.get("power")) is not float
                or actual != wanted):
            raise ValueError("correlation annotations do not match complete fixed event grouping")
    return expected


def plan_inputs(city, bbox_wsen, *, bbox_source):
    """Delegate unchanged full single-city all-ref inventory plan."""
    inventory = plan_motorway_junction_inventory(city, bbox_wsen, bbox_source=bbox_source)
    plan = {"schema": "junction-sequence-single-public-city-inputs-v1",
            "scope": "single_registered_full_named_city_bbox", "inventory_plan": inventory,
            "graph_request": {"bbox": inventory["bbox_wsen"], "network_type": "drive",
                              "simplify": False, "retain_all": True, "truncate_by_edge": True},
            "post_download_simplification": {"node_attrs_include": [KEEP_NODE], "remove_rings": False},
            "observed_ref_filter": False, "around": None,
            "boundary_semantics": "retain incident road endpoints outside the full bbox; no cue-centered crop",
            "no_acquisition_history_proof": True}
    plan["plan_sha256"] = _digest(plan)
    return plan


def preflight_network(graph):
    """Bound geometry and dense successor allocation BEFORE flattening.

    Out-degree is a conservative upper bound: RoadNetwork can remove reverse
    twins, but never add successors. A bound failure retains no road subset.
    """
    from .graph_filter import _edge_coords
    edges = graph.number_of_edges()
    if not 0 < edges <= MAX_EDGES:
        raise ValueError("complete prepared graph exceeds prototype edge cap")
    degree = max((graph.out_degree(node) for node in graph), default=0)
    if degree > MAX_DEGREE:
        raise ValueError("dense successor degree upper bound exceeds prototype cap")
    successor_upper = sum(graph.out_degree(v) for _, v, _ in graph.edges(keys=True))
    if successor_upper > MAX_SUCCESSOR_ENTRIES:
        raise ValueError("successor-entry upper bound exceeds prototype cap")
    segments = 0
    for u, v, _, data in graph.edges(keys=True, data=True):
        coords = _edge_coords(graph, u, v, data)
        if len(coords) > MAX_SEGMENTS + 1:
            raise ValueError("one complete geometry exceeds bounded segment validation work")
        count = int(np.count_nonzero(np.hypot(*np.diff(coords, axis=0).T) > 1e-6))
        segments += max(1, count)  # exactly the builder's degenerate-edge fallback
        if segments > MAX_SEGMENTS:
            raise ValueError("complete prepared graph exceeds prototype segment cap")
    return {"segments": segments, "edges": edges, "successor_degree_upper_bound": max(1, degree),
            "successor_entries_upper_bound": successor_upper,
            "dense_successor_payload_upper_bytes": 24 * edges * max(1, degree),
            "segment_array_payload_bytes": 48 * segments}


def prepare_graph(raw_graph, inventory, matched, plan, *, copy_graph=True, simplify_fn=None, project_fn=None):
    """Mark ALL present inventory nodes; REQUIRE all matched exact hypotheses.

    Unmatched missing inventory nodes are reported, not silently erased and not
    a reason to reject an otherwise complete driving graph. Matched missing,
    non-node, isolated or coordinate-conflicting hypotheses refuse the whole
    method. An already simplified cache cannot re-establish lost node identity.

    Ownership/memory: default copy_graph=True leaves raw_graph unmodified and
    makes a shallow NetworkX copy. Explicit copy_graph=False consumes a private
    disposable graph, marking it in place and dropping our raw reference after
    simplification; caller must hand off ownership (e.g. container.pop()). The
    genuine simplifier still copies, and projection creates graph/GDF temporaries.
    Prepare under a SEPARATE 3GiB guarded worker; export compact arrays and exit
    before allocating sequence genealogy. Finite-array counts certify no RSS.
    """
    from .graph_filter import RoadNetwork
    if type(copy_graph) is not bool:
        raise ValueError("copy_graph requires explicit boolean ownership mode")
    if not isinstance(raw_graph, nx.MultiDiGraph) or raw_graph.graph.get("simplified"):
        raise ValueError("original directed simplify=False/retain_all driving graph required")
    expected = plan_inputs(plan["inventory_plan"]["city"], plan["inventory_plan"]["bbox_wsen"],
                           bbox_source=plan["inventory_plan"]["bbox_source"])
    if plan != expected or inventory.get("status") != "complete" or inventory.get("plan") != plan["inventory_plan"]:
        raise ValueError("single-city inventory/graph plan changed or incomplete")
    junctions = inventory["junctions"]
    if (len(junctions) > 10_000 or inventory.get("snapshot_sha256") != _digest(junctions)
            or matched.get("status") != "complete" or matched.get("inventory_sha256") != inventory["snapshot_sha256"]):
        raise ValueError("complete original public inventory and matched snapshot required")
    if len(raw_graph) > 500_000 or raw_graph.number_of_edges() > 500_000:
        raise ValueError("raw driving graph exceeds fixed source-input bounds")
    records = {r["id"]: r for r in junctions}
    if len(records) != len(junctions):
        raise ValueError("duplicate public junction identity")
    required = set()
    for row in matched["matches"]:
        ref = row["observation"]["ref"]
        expected_candidates = [r for r in junctions if ref in r["normalized_refs"]]
        if row["candidates"] != expected_candidates:
            raise ValueError("all exact public candidate hypotheses must be retained")
        required.update(r["id"] for r in expected_candidates)
    present, missing, unsupported = {}, [], []
    for identity in sorted(records):
        try:
            node = int(identity.removeprefix("node/"))
        except ValueError:
            unsupported.append(identity)
            continue
        if identity != f"node/{node}" or node <= 0:
            unsupported.append(identity)
        elif node in raw_graph:
            present[identity] = node
        else:
            missing.append(identity)
    absent_required = sorted(required - present.keys())
    if absent_required:
        raise ValueError(f"matched topology hypotheses unavailable: {absent_required}")
    if simplify_fn is None or project_fn is None:
        import osmnx as ox
        simplify_fn = simplify_fn or ox.simplification.simplify_graph
        project_fn = project_fn or ox.project_graph
    raw_nodes, raw_edges = len(raw_graph), raw_graph.number_of_edges()
    original_crs = raw_graph.graph["crs"]
    graph = raw_graph.copy() if copy_graph else raw_graph
    original = {}
    for identity, node in present.items():
        graph.nodes[node][KEEP_NODE] = True
        original[node] = (float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]))
    graph = simplify_fn(graph, node_attrs_include=[KEEP_NODE], remove_rings=False)
    del raw_graph
    if not isinstance(graph, nx.MultiDiGraph) or not set(present.values()) <= set(graph):
        raise ValueError("simplification removed retained inventory topology IDs")
    graph = project_fn(graph)
    to_projected = Transformer.from_crs(original_crs, graph.graph["crs"], always_xy=True)
    public_to_projected = Transformer.from_crs("EPSG:4326", graph.graph["crs"], always_xy=True)
    for identity, node in present.items():
        x, y = to_projected.transform(*original[node])
        nx_, ny = float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])
        if not all(math.isfinite(v) for v in (x, y, nx_, ny)) or math.hypot(x - nx_, y - ny) > 1e-5:
            raise ValueError("retained topology node moved during preparation")
        if identity in required:
            record = records[identity]
            px, py = public_to_projected.transform(record["lon"], record["lat"])
            if (graph.degree(node) == 0 or not math.isfinite(px) or not math.isfinite(py)
                    or math.hypot(px - nx_, py - ny) > 5.):
                raise ValueError("matched topology incidence/public coordinate proof failed")
    allocation = preflight_network(graph)
    net = RoadNetwork.from_graph(graph)
    if len(net.seg_gstart) != allocation["segments"]:
        raise ValueError("preflight/builder segment counts disagree")
    receipt = {"schema": "junction-sequence-preserved-city-topology-v1", "status": "complete",
               "plan_sha256": plan["plan_sha256"], "inventory_sha256": inventory["snapshot_sha256"],
               "required_hypotheses": sorted(required), "preserved_inventory_ids": sorted(present),
               "missing_unmatched_inventory_ids": missing, "unsupported_unmatched_inventory_ids": unsupported,
               "raw_node_count": raw_nodes, "raw_edge_count": raw_edges,
               "prepared_node_count": len(graph), "prepared_edge_count": graph.number_of_edges(),
               "all_matched_exact_nodes_and_incidence": True, "nearest_edge_projection_used": False,
               "unmatched_inventory_retained": True, "acquisition_history_verified": False,
               "preparation_requires_separate_guarded_worker": True,
               "copy_graph": copy_graph, "raw_graph_caller_owned_and_not_modified": copy_graph,
               "private_raw_ownership_consumed": not copy_graph,
               "overlapping_graph_copies_and_projection_temporaries_possible": True,
               "preflight_allocation": allocation,
               "graph_stats": net.stats()}
    return net, receipt
