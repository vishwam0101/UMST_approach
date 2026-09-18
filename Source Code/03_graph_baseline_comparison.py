#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_graph_baseline_comparison.py
===============================

Compare three graph structures under one identical delivery workload:

    UMST    union of MSTs (our contribution)      umst_graph.graphml
    MST     a single minimum spanning tree        mst_graph.graphml
    CLIQUE  the dense reference graph             gh_hotspot_graph.graphml

Each graph is run under two arms:

    bundling   packages share a vehicle when co-located and co-directed
    baseline   every package gets its own dedicated vehicle, door to door

Both arms follow the *same* routes; only sharing differs.  The order pool and
the delivery deadlines are frozen once per city and reused byte-for-byte by
every graph and every arm, so a graph that routes badly is penalised instead of
being handed a proportionally looser deadline.

This file is standalone.  It reads the .graphml files as final inputs, never
regenerates them, never calls GraphHopper, and never imports from the notebooks.

Usage
-----
    python 03_graph_baseline_comparison.py                       # UMST, defaults
    python 03_graph_baseline_comparison.py --graph-type ALL
    python 03_graph_baseline_comparison.py --load-per-hotspot 2 --n-runs 1
    python 03_graph_baseline_comparison.py --graph-type ALL --aggregate
    %run 03_graph_baseline_comparison.py --graph-type CLIQUE     # from a notebook

Every constant in the CONFIG block below is also a CLI flag: the flag name is
the constant lowercased with underscores turned into dashes
(CITY_NAME -> --city-name).  The CLI overrides the constant, and the fully
resolved config is printed at startup.

Known limitations (properties of the model, deliberately not fixed here)
-----------------------------------------------------------------------
* Bundling is myopic single-hop co-direction merging with no explicit savings
  test.  The savings are implicit: k packages share one edge traversal.
* Bundles dissolve and re-form at each hop, so a "bundle" is a single-edge
  convoy and bundles_formed counts re-formations, not distinct groupings.
* Vehicle repositioning between trips is free.
* The per-hop relay handoff penalty (VEHICLE_CHANGE_TIME) structurally
  disadvantages sparse graphs.  That is a genuine property of relay delivery,
  not an artifact of the implementation.
* mst_graph.graphml was built with weight="distance" (road km) while the MSTs
  inside UMST were built with weight="weight" (geodesic km).  On Columbus_mini
  those two trees differ by 7 of 25 edges, so MST is not the tree UMST was
  constructed around and is not in general a subgraph of it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

# ============================================================================
# CONFIG
# ============================================================================

# ---------- city / data selection ----------
CITY_NAME              = "Columbus"     # "Columbus" | "Chicago" | "City of New York"
MINI                   = True           # mini city subset -> "<City>_mini - RL Delivery Data"
WITH_GEODESIC_DISTANCE = False          # True -> "GeoDist<City>..." folder variant

# ---------- the toggle ----------
GRAPH_TYPE             = "UMST"         # "UMST" | "MST" | "CLIQUE" | "ALL"

# ---------- order pool ----------
ORDER_POOL_GRAPH       = "UMST"         # which graph defines the feasible order set
MAX_DELIVERY_TIME      = 900            # seconds; an OD pair is feasible if its
                                        # shortest-path travel time on ORDER_POOL_GRAPH
                                        # is <= this
LOAD_PER_HOTSPOT       = 100            # total orders = num_hotspots * this
HOURS                  = 1              # order arrival window, hours
PEAKS                  = [0.25, 0.75]   # gaussian peak positions as fraction of the hour
SIGMA                  = 10             # gaussian spread, minutes

# ---------- deadline (tunable) ----------
TIME_LIMIT_REF_GRAPH   = "UMST"         # which graph's routing the deadline is frozen on
BUFFER_TIME_TYPE       = "multiply"     # "multiply"|"fixed_buffer"|"rangewise"|"randomized"
TIME_TOLERANCE_FACTOR  = 2.5            # used by "multiply"
FIXED_BUFFER_SECONDS   = 300            # used by "fixed_buffer"
RANDOM_BUFFER_RANGE    = (120, 480)     # used by "randomized", seconds
RANGEWISE_BANDS        = [              # used by "rangewise": (upper_bound_s, buffer_s)
    (600, 300), (1200, 600), (1500, 720), (float("inf"), 900),
]

# ---------- simulation ----------
TIME_STEP              = 1              # seconds per tick
GRACE_TIME             = 1800           # seconds past the arrival window before cutoff
WAIT_TIME              = 120            # max seconds an order holds at a hotspot
                                        # hoping for a bundling partner
MAX_BUNDLE_SIZE        = 5
VEHICLE_COOLDOWN       = 5              # seconds before a released vehicle is reusable
VEHICLE_CHANGE_TIME    = 1              # seconds charged per relay handoff
SOLO_DEPARTURE         = True           # False reproduces the old hold-forever behaviour

# ---------- experiment ----------
N_RUNS                 = 5
DROP_RATES             = [0.0]          # e.g. [0.0, 0.1, 0.25, 0.5] for the resilience sweep
GLOBAL_SEED            = 42
SAVE_PER_ORDER         = True           # write orders_run0.csv
MAKE_PLOTS             = True           # only affects --aggregate


# ============================================================================
# 1. CONFIG PLUMBING, PATHS, ARGPARSE
# ============================================================================

GRAPH_TYPES = ("UMST", "MST", "CLIQUE")

GRAPH_FILES = {
    "UMST":   "umst_graph.graphml",
    "MST":    "mst_graph.graphml",
    "CLIQUE": "gh_hotspot_graph.graphml",
}

MODES = ("bundling", "baseline")

ROOT = Path(__file__).resolve().parents[1]          # UMST_approach/
DATA_ROOT = ROOT / "Delivery_Data"
RESULTS_ROOT = ROOT / "Results"


def _as_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "on"):
        return True
    if text in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("expected a boolean, got %r" % (value,))


def _as_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(part) for part in str(value).replace(" ", "").split(",") if part]


def _as_float_pair(value):
    pair = _as_float_list(value)
    if len(pair) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated numbers")
    return (pair[0], pair[1])


def _as_bands(value):
    """Parse the rangewise bands.

    Accepts the module-level list directly, or a JSON string such as
    '[[600,300],[1200,600],["inf",900]]'.  JSON has no Infinity literal, so
    "inf" (or null) in the upper-bound slot means "everything above".
    """
    raw = value if isinstance(value, (list, tuple)) else json.loads(value)
    bands = []
    for upper, buffer_s in raw:
        if upper is None or (isinstance(upper, str) and upper.lower() in ("inf", "infinity")):
            upper = float("inf")
        bands.append((float(upper), float(buffer_s)))
    if not bands:
        raise argparse.ArgumentTypeError("RANGEWISE_BANDS must not be empty")
    return bands


def _as_graph_choice(value):
    text = str(value).strip().upper()
    if text not in GRAPH_TYPES:
        raise argparse.ArgumentTypeError("expected one of %s" % (", ".join(GRAPH_TYPES),))
    return text


def _as_graph_or_all(value):
    text = str(value).strip().upper()
    if text not in GRAPH_TYPES + ("ALL",):
        raise argparse.ArgumentTypeError("expected one of %s, ALL" % (", ".join(GRAPH_TYPES),))
    return text


def _as_buffer_type(value):
    text = str(value).strip().lower()
    allowed = ("multiply", "fixed_buffer", "rangewise", "randomized")
    if text not in allowed:
        raise argparse.ArgumentTypeError("expected one of %s" % (", ".join(allowed),))
    return text


# (constant name, converter, help).  This is also the banner print order.
CONFIG_SPEC = [
    ("CITY_NAME",              str,               "city folder stem, e.g. Columbus"),
    ("MINI",                   _as_bool,          "use the '<City>_mini' subset"),
    ("WITH_GEODESIC_DISTANCE", _as_bool,          "use the 'GeoDist<City>' folder variant"),
    ("GRAPH_TYPE",             _as_graph_or_all,  "UMST | MST | CLIQUE | ALL"),
    ("ORDER_POOL_GRAPH",       _as_graph_choice,  "graph whose reachability defines the order pool"),
    ("MAX_DELIVERY_TIME",      float,             "seconds; OD feasibility threshold"),
    ("LOAD_PER_HOTSPOT",       int,               "orders per hotspot (total = hotspots * this)"),
    ("HOURS",                  int,               "order arrival window in hours"),
    ("PEAKS",                  _as_float_list,    "comma-separated peak positions, fraction of an hour"),
    ("SIGMA",                  float,             "gaussian spread in minutes"),
    ("TIME_LIMIT_REF_GRAPH",   _as_graph_choice,  "graph the frozen deadline is computed on"),
    ("BUFFER_TIME_TYPE",       _as_buffer_type,   "multiply | fixed_buffer | rangewise | randomized"),
    ("TIME_TOLERANCE_FACTOR",  float,             "multiplier used by BUFFER_TIME_TYPE=multiply"),
    ("FIXED_BUFFER_SECONDS",   float,             "seconds added by BUFFER_TIME_TYPE=fixed_buffer"),
    ("RANDOM_BUFFER_RANGE",    _as_float_pair,    "lo,hi seconds for BUFFER_TIME_TYPE=randomized"),
    ("RANGEWISE_BANDS",        _as_bands,         "JSON [[upper_s, buffer_s], ...]; use inf for the last"),
    ("TIME_STEP",              int,               "seconds per simulation tick"),
    ("GRACE_TIME",             int,               "seconds past the arrival window before cutoff"),
    ("WAIT_TIME",              int,               "max seconds an order holds at a hotspot"),
    ("MAX_BUNDLE_SIZE",        int,               "max orders sharing one vehicle on one edge"),
    ("VEHICLE_COOLDOWN",       int,               "seconds before a released vehicle is reusable"),
    ("VEHICLE_CHANGE_TIME",    float,             "seconds charged per relay handoff"),
    ("SOLO_DEPARTURE",         _as_bool,          "False reproduces the old hold-forever behaviour"),
    ("N_RUNS",                 int,               "runs per (graph, drop rate)"),
    ("DROP_RATES",             _as_float_list,    "comma-separated edge failure rates"),
    ("GLOBAL_SEED",            int,               "master seed"),
    ("SAVE_PER_ORDER",         _as_bool,          "write orders_run0.csv"),
    ("MAKE_PLOTS",             _as_bool,          "draw comparison plots during --aggregate"),
]


class Config(object):
    """Resolved configuration: attribute access plus the derived paths."""

    def __init__(self, values):
        for name, _conv, _help in CONFIG_SPEC:
            setattr(self, name, values[name])

    def as_dict(self):
        """JSON-safe view of the config (infinities become the string "inf")."""
        def clean(value):
            if isinstance(value, float) and math.isinf(value):
                return "inf"
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            return value
        return {name: clean(getattr(self, name)) for name, _c, _h in CONFIG_SPEC}

    def clone_for_city(self, city_name, mini, geodesic):
        """Same settings, different city. Used by the --cities sweep."""
        values = {name: getattr(self, name) for name, _c, _h in CONFIG_SPEC}
        values["CITY_NAME"] = city_name
        values["MINI"] = mini
        values["WITH_GEODESIC_DISTANCE"] = geodesic
        return Config(values)

    # -- derived ----------------------------------------------------------
    @property
    def city_folder(self):
        # mirrors notebook 02 cell 19 exactly
        prefix = "GeoDist" if self.WITH_GEODESIC_DISTANCE else ""
        suffix = "_mini" if self.MINI else ""
        return "%s%s%s - RL Delivery Data" % (prefix, self.CITY_NAME, suffix)

    @property
    def city_key(self):
        prefix = "GeoDist_" if self.WITH_GEODESIC_DISTANCE else ""
        suffix = "_mini" if self.MINI else ""
        return "%s%s%s" % (prefix, self.CITY_NAME, suffix)

    @property
    def data_dir(self):
        return DATA_ROOT / self.city_folder

    @property
    def graph_dir(self):
        return self.data_dir / "UMST Graph" / "graphs"

    @property
    def city_results_dir(self):
        return RESULTS_ROOT / self.city_key

    @property
    def shared_dir(self):
        return self.city_results_dir / "_shared"

    def out_dir(self, graph_type, drop_rate):
        return self.city_results_dir / graph_type / drop_folder(drop_rate)

    @property
    def comparison_dir(self):
        return self.city_results_dir / "comparison"

    @property
    def graph_types_to_run(self):
        return list(GRAPH_TYPES) if self.GRAPH_TYPE == "ALL" else [self.GRAPH_TYPE]

    @property
    def max_sim_time(self):
        return self.HOURS * 3600 + self.GRACE_TIME


def drop_folder(drop_rate):
    return "drop_%02d" % int(round(drop_rate * 100))


CITY_SUFFIX = " - RL Delivery Data"
NEWLINE = chr(10)


def parse_city_stem(stem):
    """Exact inverse of Config.city_folder: stem -> (city_name, mini, geodesic).

    'Columbus_mini'           -> ('Columbus', True,  False)
    'GeoDistChicago'          -> ('Chicago',  False, True)
    'GeoDistCity of New York' -> ('City of New York', False, True)
    """
    stem = str(stem).strip()
    if stem.endswith(CITY_SUFFIX):
        stem = stem[:-len(CITY_SUFFIX)]
    geodesic = stem.startswith("GeoDist")
    rest = stem[len("GeoDist"):] if geodesic else stem
    mini = rest.endswith("_mini")
    city_name = rest[:-len("_mini")] if mini else rest
    return city_name, mini, geodesic


def discover_city_stems():
    """Every Delivery_Data city folder that actually carries all three graphs."""
    stems = []
    if not DATA_ROOT.is_dir():
        return stems
    for folder in sorted(DATA_ROOT.iterdir()):
        if not folder.is_dir() or not folder.name.endswith(CITY_SUFFIX):
            continue
        graph_dir = folder / "UMST Graph" / "graphs"
        if not graph_dir.is_dir():
            continue
        try:
            for graph_type in GRAPH_TYPES:
                resolve_graph_path(graph_dir, graph_type)
        except FileNotFoundError:
            continue
        stems.append(folder.name[:-len(CITY_SUFFIX)])
    return stems


def city_health(stem):
    """Report whether a city's graphs actually carry usable road distance/time.

    A graph saved before the GraphHopper enrichment step has 'distance' but no
    'time'.  build_adjacency refuses it (assertion 2) rather than treating every
    trip as instantaneous, so flag it here where the message can be useful.
    """
    city_name, mini, geodesic = parse_city_stem(stem)
    folder = DATA_ROOT / ("%s%s%s%s" % ("GeoDist" if geodesic else "", city_name,
                                        "_mini" if mini else "", CITY_SUFFIX))
    graph_dir = folder / "UMST Graph" / "graphs"
    problems = []
    nodes = 0
    for graph_type in GRAPH_TYPES:
        try:
            graph = nx.read_graphml(str(resolve_graph_path(graph_dir, graph_type)))
        except Exception as exc:
            problems.append("%s unreadable (%s)" % (graph_type, exc))
            continue
        nodes = max(nodes, graph.number_of_nodes())
        bad = sum(1 for _u, _v, d in graph.edges(data=True)
                  if "distance" not in d or "time" not in d
                  or float(d["distance"]) <= 0 or float(d["time"]) <= 0)
        if bad:
            problems.append("%s: %d/%d edges lack usable distance/time"
                            % (graph_type, bad, graph.number_of_edges()))
    return nodes, problems


def _as_city_list(value):
    """--cities ALL, or a comma-separated list of city folder stems."""
    if str(value).strip().upper() == "ALL":
        return "ALL"
    return [part.strip() for part in str(value).split(",") if part.strip()]


def resolve_cities(cfg, cities_arg):
    """[(stem, Config), ...] for this invocation."""
    if cities_arg is None:
        return [(cfg.city_key, cfg)]

    available = discover_city_stems()
    stems = available if cities_arg == "ALL" else list(cities_arg)

    unknown = [s for s in stems if s not in available]
    if unknown:
        message = ["Unknown city folder stem(s): " + ", ".join(unknown),
                   "Available (folders under %s that carry all three graphs):" % DATA_ROOT]
        message.extend("  " + stem for stem in (available or ["(none found)"]))
        raise SystemExit(NEWLINE.join(message))
    if not stems:
        raise SystemExit("No usable city folders found under %s" % DATA_ROOT)

    return [(stem, cfg.clone_for_city(*parse_city_stem(stem))) for stem in stems]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="UMST / MST / Clique baseline comparison harness",
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="do not simulate; read the existing summary.json files and write "
             "comparison_table.csv, comparison_summary.json and the plots",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress the per-arm progress lines",
    )
    parser.add_argument(
        "--cities", type=_as_city_list, default=None,
        help="sweep several cities: ALL for every usable Delivery_Data folder, or a "
             "comma-separated list of folder stems (e.g. Columbus_mini,GeoDistChicago). "
             "Overrides --city-name/--mini/--with-geodesic-distance.",
    )
    parser.add_argument(
        "--list-cities", action="store_true",
        help="print the city folder stems that carry all three graphs, then exit",
    )
    for name, converter, help_text in CONFIG_SPEC:
        flag = "--" + name.lower().replace("_", "-")
        parser.add_argument(flag, dest=name, type=converter, default=None, help=help_text)
    return parser


def resolve_config(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    values = {}
    for name, converter, _help in CONFIG_SPEC:
        override = getattr(args, name)
        values[name] = converter(globals()[name]) if override is None else override
    return Config(values), args


def resolve_graph_path(graph_dir, graph_type):
    """The graphml files ship as either <name>.graphml or <name>.graphml.xml."""
    stem = GRAPH_FILES[graph_type]
    for candidate in (graph_dir / stem, graph_dir / (stem + ".xml")):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No graph file for %s. Looked for:\n  %s\n  %s"
        % (graph_type, graph_dir / stem, graph_dir / (stem + ".xml"))
    )


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and math.isinf(obj):
        return "inf"
    raise TypeError("not JSON serialisable: %r" % (type(obj),))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def read_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rule(char="="):
    return char * 78


# ============================================================================
# 2. GRAPH LOADING, CANONICAL INDEX, ADJACENCY, STATS
# ============================================================================

def load_graphs(graph_dir):
    """Load all three graphs.  Returns {graph_type: nx.Graph}."""
    graphs = {}
    for graph_type in GRAPH_TYPES:
        path = resolve_graph_path(graph_dir, graph_type)
        graphs[graph_type] = nx.read_graphml(str(path))
    return graphs


def build_canonical_index(graphs):
    """One node<->index mapping shared by every graph.

    Notebook 02 built a separate mapping per graph from sorted(graph.nodes()),
    so an integer index could mean different tracts on different graphs.  Here
    the node sets must be identical and the mapping is built exactly once.
    """
    reference_type = GRAPH_TYPES[0]
    reference_nodes = set(graphs[reference_type].nodes())
    for graph_type in GRAPH_TYPES[1:]:
        other = set(graphs[graph_type].nodes())
        if other != reference_nodes:
            missing = sorted(reference_nodes - other)
            extra = sorted(other - reference_nodes)
            raise AssertionError(
                "ASSERTION 1 FAILED: node sets differ between %s and %s.\n"
                "  in %s only (%d): %s\n"
                "  in %s only (%d): %s"
                % (reference_type, graph_type,
                   reference_type, len(missing), missing[:20],
                   graph_type, len(extra), extra[:20])
            )

    nodes = sorted(reference_nodes)
    tract_to_index = {tract: idx for idx, tract in enumerate(nodes)}
    index_to_tract = {idx: tract for idx, tract in enumerate(nodes)}

    node_positions = {}
    for tract, data in graphs[reference_type].nodes(data=True):
        idx = tract_to_index[tract]
        node_positions[idx] = (float(data.get("lat", 0.0)), float(data.get("lon", 0.0)))

    return tract_to_index, index_to_tract, node_positions


def build_adjacency(graph, tract_to_index, graph_type="?"):
    """adjacency[u] = [(v, distance_km, time_seconds), ...], undirected.

    The graphml stores travel time in minutes; everything downstream works in
    seconds.  A graph missing 'time' would make every trip instantaneous, so
    fail loudly rather than run.
    """
    adjacency = {idx: [] for idx in tract_to_index.values()}

    for u, v, data in graph.edges(data=True):
        if "distance" not in data or "time" not in data:
            raise AssertionError(
                "ASSERTION 2 FAILED: edge %s--%s on %s is missing distance and/or time"
                % (u, v, graph_type)
            )
        distance_km = float(data["distance"])
        time_seconds = float(data["time"]) * 60.0
        if not (distance_km > 0.0):
            raise AssertionError(
                "ASSERTION 2 FAILED: edge %s--%s on %s has distance=%r (expected > 0)"
                % (u, v, graph_type, distance_km)
            )
        if not (time_seconds > 0.0):
            raise AssertionError(
                "ASSERTION 2 FAILED: edge %s--%s on %s has time=%r minutes (expected > 0)"
                % (u, v, graph_type, data["time"])
            )
        u_idx = tract_to_index[u]
        v_idx = tract_to_index[v]
        adjacency[u_idx].append((v_idx, distance_km, time_seconds))
        adjacency[v_idx].append((u_idx, distance_km, time_seconds))

    return adjacency


def build_edge_lookup(adjacency):
    """edge_lookup[(u, v)] = (distance_km, time_seconds).

    Notebook 02's get_edge_info linear-scanned the neighbour list on every
    call, which dominated runtime and was worst on the dense graph.
    """
    lookup = {}
    for u, neighbours in adjacency.items():
        for v, distance_km, time_seconds in neighbours:
            lookup[(u, v)] = (distance_km, time_seconds)
    return lookup


def dijkstra_all(adjacency, source):
    """Single-source Dijkstra on time_seconds.

    Returns (dist, prev, hops): dist[v] seconds, prev[v] predecessor index,
    hops[v] number of edges on the minimum-time path.
    """
    dist = {source: 0.0}
    prev = {source: None}
    hops = {source: 0}
    heap = [(0.0, source)]

    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist.get(node, float("inf")):
            continue
        for neighbour, _distance_km, time_seconds in adjacency.get(node, ()):
            new_cost = cost + time_seconds
            if new_cost < dist.get(neighbour, float("inf")):
                dist[neighbour] = new_cost
                prev[neighbour] = node
                hops[neighbour] = hops[node] + 1
                heapq.heappush(heap, (new_cost, neighbour))

    return dist, prev, hops


def reconstruct_path(prev, target):
    """Path as a list of indices, or [] when target was never reached."""
    if target not in prev:
        return []
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path


def compute_graph_stats(graph_type, graph, adjacency, max_delivery_time):
    """Structural stats plus the all-pairs coverage headline."""
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    total_edge_length_km = sum(float(d["distance"]) for _u, _v, d in graph.edges(data=True))

    reachable_times = []
    reachable_hops = []
    within_limit = 0
    total_pairs = 0

    for source in sorted(adjacency.keys()):
        dist, _prev, hops = dijkstra_all(adjacency, source)
        for target in adjacency.keys():
            if target == source:
                continue
            total_pairs += 1
            travel_time = dist.get(target)
            if travel_time is None:
                continue
            reachable_times.append(travel_time)
            reachable_hops.append(hops[target])
            if travel_time <= max_delivery_time:
                within_limit += 1

    return {
        "graph_type": graph_type,
        "nodes": n_nodes,
        "edges": n_edges,
        "avg_degree": (2.0 * n_edges / n_nodes) if n_nodes else 0.0,
        "density": nx.density(graph),
        "total_edge_length_km": total_edge_length_km,
        "total_od_pairs": total_pairs,
        "pairs_within_max_delivery_time": within_limit,
        "coverage_within_max_delivery_time": (within_limit / total_pairs) if total_pairs else 0.0,
        "connected_pairs": len(reachable_times),
        "mean_path_time_all_pairs": float(np.mean(reachable_times)) if reachable_times else 0.0,
        "mean_hops_all_pairs": float(np.mean(reachable_hops)) if reachable_hops else 0.0,
    }


# ============================================================================
# 3. THE FROZEN ORDER POOL
# ============================================================================

class Order(object):
    """An order before any graph has routed it: identity, endpoints, arrival."""

    __slots__ = ("id", "start_node", "end_node", "start_time")

    def __init__(self, order_id, start_node, end_node, start_time):
        self.id = order_id
        self.start_node = start_node
        self.end_node = end_node
        self.start_time = start_time

    def as_row(self):
        return [self.id, self.start_node, self.end_node, self.start_time]


def order_pool_hash(cfg):
    payload = {
        "city_key": cfg.city_key,
        "order_pool_graph": cfg.ORDER_POOL_GRAPH,
        "max_delivery_time": cfg.MAX_DELIVERY_TIME,
        "load_per_hotspot": cfg.LOAD_PER_HOTSPOT,
        "hours": cfg.HOURS,
        "peaks": list(cfg.PEAKS),
        "sigma": cfg.SIGMA,
        "global_seed": cfg.GLOBAL_SEED,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def feasible_pairs_on(adjacency, max_delivery_time):
    """Ordered (start, end) pairs routable within max_delivery_time."""
    pairs = []
    for source in sorted(adjacency.keys()):
        dist, _prev, _hops = dijkstra_all(adjacency, source)
        for target, travel_time in dist.items():
            if target != source and travel_time <= max_delivery_time:
                pairs.append((source, target))
    pairs.sort()
    return pairs


def generate_order_pool(adjacency, cfg, num_hotspots):
    """Build the pool from scratch.  Seeded entirely by GLOBAL_SEED."""
    rng = np.random.RandomState(cfg.GLOBAL_SEED)
    py_rng = random.Random(cfg.GLOBAL_SEED)

    pairs = feasible_pairs_on(adjacency, cfg.MAX_DELIVERY_TIME)
    if not pairs:
        raise RuntimeError(
            "No OD pair on %s is routable within MAX_DELIVERY_TIME=%.0fs."
            % (cfg.ORDER_POOL_GRAPH, cfg.MAX_DELIVERY_TIME)
        )

    total_orders = num_hotspots * cfg.LOAD_PER_HOTSPOT

    # Two-gaussian arrival mixture over the minute bins of the window.
    minutes = 60 * cfg.HOURS
    x = np.arange(minutes, dtype=float)
    density = np.zeros(minutes, dtype=float)
    for peak in cfg.PEAKS:
        mu = peak * minutes
        density += (1.0 / (cfg.SIGMA * math.sqrt(2.0 * math.pi))) * \
                   np.exp(-0.5 * ((x - mu) / cfg.SIGMA) ** 2)
    probabilities = density / density.sum()
    per_minute = rng.multinomial(total_orders, probabilities)

    orders = []
    order_id = 0
    for minute in range(minutes):
        for _ in range(int(per_minute[minute])):
            start_node, end_node = pairs[py_rng.randrange(len(pairs))]
            start_time = py_rng.randint(minute * 60, (minute + 1) * 60 - 1)
            orders.append(Order(order_id, start_node, end_node, start_time))
            order_id += 1

    return orders, pairs


def build_order_pool(adjacencies, cfg, num_hotspots, verbose=True):
    """The frozen workload, shared byte-for-byte across all three graphs.

    Cached under Results/<CITY_KEY>/_shared/order_pool_<hash>.json so that a
    UMST run today and a CLIQUE run tomorrow still compare like with like.
    """
    pool_hash = order_pool_hash(cfg)
    cache_path = cfg.shared_dir / ("order_pool_%s.json" % pool_hash)

    if cache_path.is_file():
        payload = read_json(cache_path)
        orders = [Order(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in payload["orders"]]
        if verbose:
            print("  order pool  : loaded from cache %s" % cache_path.name)
        return orders, pool_hash, payload["feasible_pairs"]

    adjacency = adjacencies[cfg.ORDER_POOL_GRAPH]
    orders, pairs = generate_order_pool(adjacency, cfg, num_hotspots)

    write_json(cache_path, {
        "pool_hash": pool_hash,
        "city_key": cfg.city_key,
        "order_pool_graph": cfg.ORDER_POOL_GRAPH,
        "max_delivery_time": cfg.MAX_DELIVERY_TIME,
        "load_per_hotspot": cfg.LOAD_PER_HOTSPOT,
        "hours": cfg.HOURS,
        "peaks": list(cfg.PEAKS),
        "sigma": cfg.SIGMA,
        "global_seed": cfg.GLOBAL_SEED,
        "num_hotspots": num_hotspots,
        "feasible_pairs": len(pairs),
        "n_orders": len(orders),
        "orders": [o.as_row() for o in orders],
    })
    if verbose:
        print("  order pool  : generated and cached to %s" % cache_path.name)
    return orders, pool_hash, len(pairs)


# ============================================================================
# 4. DEADLINES, FROZEN ONCE
# ============================================================================

def time_limits_hash(cfg, pool_hash):
    payload = {
        "pool_hash": pool_hash,
        "ref_graph": cfg.TIME_LIMIT_REF_GRAPH,
        "buffer_time_type": cfg.BUFFER_TIME_TYPE,
        "time_tolerance_factor": cfg.TIME_TOLERANCE_FACTOR,
        "fixed_buffer_seconds": cfg.FIXED_BUFFER_SECONDS,
        "random_buffer_range": list(cfg.RANDOM_BUFFER_RANGE),
        "rangewise_bands": [[("inf" if math.isinf(u) else u), b] for u, b in cfg.RANGEWISE_BANDS],
        "global_seed": cfg.GLOBAL_SEED,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _apply_buffer(travel_time, cfg, rng):
    if cfg.BUFFER_TIME_TYPE == "multiply":
        return travel_time * cfg.TIME_TOLERANCE_FACTOR
    if cfg.BUFFER_TIME_TYPE == "fixed_buffer":
        return travel_time + cfg.FIXED_BUFFER_SECONDS
    if cfg.BUFFER_TIME_TYPE == "randomized":
        lo, hi = cfg.RANDOM_BUFFER_RANGE
        return travel_time + rng.uniform(lo, hi)
    if cfg.BUFFER_TIME_TYPE == "rangewise":
        for upper, buffer_s in cfg.RANGEWISE_BANDS:
            if travel_time <= upper:
                return travel_time + buffer_s
        return travel_time + cfg.RANGEWISE_BANDS[-1][1]
    raise ValueError("unknown BUFFER_TIME_TYPE %r" % (cfg.BUFFER_TIME_TYPE,))


def compute_time_limits(order_pool, ref_adjacency, cfg, pool_hash, verbose=True):
    """Deadlines, computed once on TIME_LIMIT_REF_GRAPH and never recomputed.

    Notebook 02 recomputed the deadline per graph from that graph's own
    routing, so a graph that routed badly got a proportionally looser deadline
    and success_rate could not detect the difference.  Rerouting after edge
    failures must not recompute it either.
    """
    limits_hash = time_limits_hash(cfg, pool_hash)
    cache_path = cfg.shared_dir / ("time_limits_%s.json" % limits_hash)

    if cache_path.is_file():
        payload = read_json(cache_path)
        time_limits = {int(k): float(v) for k, v in payload["time_limits"].items()}
        ref_times = {int(k): float(v) for k, v in payload["ref_travel_times"].items()}
        if len(time_limits) == len(order_pool):
            if verbose:
                print("  deadlines   : loaded from cache %s" % cache_path.name)
            return time_limits, ref_times, limits_hash

    rng = random.Random(cfg.GLOBAL_SEED + 977)

    # Route each order once on the reference graph; cache Dijkstra per source.
    prev_cache = {}
    dist_cache = {}
    for order in order_pool:
        if order.start_node not in dist_cache:
            dist, prev, _hops = dijkstra_all(ref_adjacency, order.start_node)
            dist_cache[order.start_node] = dist
            prev_cache[order.start_node] = prev

    ref_times = {}
    time_limits = {}
    for order in order_pool:
        travel_time = dist_cache[order.start_node].get(order.end_node)
        if travel_time is None:
            raise RuntimeError(
                "Order %d (%d -> %d) has no path on the deadline reference graph %s."
                % (order.id, order.start_node, order.end_node, cfg.TIME_LIMIT_REF_GRAPH)
            )
        ref_times[order.id] = travel_time
        time_limits[order.id] = _apply_buffer(travel_time, cfg, rng)

    write_json(cache_path, {
        "time_limits_hash": limits_hash,
        "pool_hash": pool_hash,
        "ref_graph": cfg.TIME_LIMIT_REF_GRAPH,
        "buffer_time_type": cfg.BUFFER_TIME_TYPE,
        "time_limits": {str(k): v for k, v in time_limits.items()},
        "ref_travel_times": {str(k): v for k, v in ref_times.items()},
    })
    if verbose:
        print("  deadlines   : computed on %s and cached to %s"
              % (cfg.TIME_LIMIT_REF_GRAPH, cache_path.name))
    return time_limits, ref_times, limits_hash


# ============================================================================
# 5. DELIVERY / VEHICLE / BUNDLE
# ============================================================================

class Delivery(object):
    """One order as routed on one graph, with its full per-run state."""

    id_counter = 0

    def __init__(self, order, shortest_path, expected_travel_time, time_limit):
        self.id = order.id
        Delivery.id_counter += 1

        # route on this graph (restored verbatim by reset())
        self.start_node = order.start_node
        self.end_node = order.end_node
        self.start_time = order.start_time
        self.base_path = list(shortest_path)
        self.base_expected_travel_time = expected_travel_time
        self.shortest_path = list(shortest_path)
        self.expected_travel_time = expected_travel_time

        # the frozen deadline; never recomputed, not even on reroute
        self.time_limit = time_limit

        self.reset()

    def reset(self):
        self.shortest_path = list(self.base_path)
        self.expected_travel_time = self.base_expected_travel_time

        self.current_node = self.start_node
        self.path_index = 0
        self.in_transition = False
        self.time_till_next_node = 0.0

        # plain countdown plus a boolean; no 0 / >0 / -1 tri-state, no snapping
        self.hold_remaining = 0
        self.activated = False

        self.completed = False
        self.successful = False
        self.failed = False
        self.end_time = None

        self.distance_traveled = 0.0
        self.actual_path = [self.start_node]
        self.num_vehicle_changes = 0
        self.times_bundled = 0
        self.total_hold_time = 0

        self.actual_delivery_time = 0.0
        self.delay = 0.0
        self.within_buffer = False

        self.current_bundle_id = None
        self.held_vehicle = None

    def next_node(self):
        if self.path_index < len(self.shortest_path) - 1:
            return self.shortest_path[self.path_index + 1]
        return None

    def arrive_at_node(self, node):
        self.in_transition = False
        self.time_till_next_node = 0.0
        self.current_node = node
        self.path_index += 1
        self.actual_path.append(node)

    def hops(self):
        return max(0, len(self.actual_path) - 1)

    def finalize_metrics(self, vehicle_change_time):
        if self.completed and self.end_time is not None:
            self.actual_delivery_time = (self.end_time - self.start_time) + \
                                        (self.num_vehicle_changes * vehicle_change_time)
            self.delay = self.actual_delivery_time - self.time_limit
            self.within_buffer = self.actual_delivery_time <= self.time_limit
            self.successful = self.within_buffer

    def __repr__(self):
        state = "done" if self.completed else ("moving" if self.in_transition else "waiting")
        return "D%d:%d->%d|%s@N%d" % (self.id, self.start_node, self.end_node,
                                      state, self.current_node)


class Vehicle(object):
    """An edge-level relay vehicle, not a persistent courier.

    total_trips counts edge traversals (vehicle-trips), which is what should be
    read as workload; len(simulation.vehicles) counts distinct vehicle objects
    and must not be read as fleet size.
    """

    id_counter = 0

    def __init__(self, start_node, allow_reuse=True):
        self.id = Vehicle.id_counter
        Vehicle.id_counter += 1

        self.current_node = start_node
        self.allow_reuse = allow_reuse

        self.total_distance = 0.0
        self.total_trips = 0
        self.all_deliveries_carried = set()

        self.is_in_use = False
        self.available_at_time = 0
        self.cooldown_period = 0

    def move_to(self, next_node, distance_km):
        self.current_node = next_node
        self.total_distance += distance_km
        self.total_trips += 1

    def start_new_trip(self, start_node, current_time):
        if not self.allow_reuse:
            raise ValueError("vehicle reuse is not allowed in baseline mode")
        self.current_node = start_node          # repositioning is free
        self.is_in_use = True

    def release(self, current_time):
        self.is_in_use = False
        self.available_at_time = current_time + self.cooldown_period

    def __repr__(self):
        return "V%d@N%d|%.1fkm" % (self.id, self.current_node, self.total_distance)


class Bundle(object):
    """A group of deliveries sharing one vehicle over one or more hops."""

    id_counter = 0

    def __init__(self, delivery_ids, vehicle, current_node, next_node, step):
        self.id = Bundle.id_counter
        Bundle.id_counter += 1

        self.delivery_ids = set(delivery_ids)
        self.vehicle = vehicle
        self.current_node = current_node
        self.next_node = next_node
        self.created_at_step = step

        self.is_active = True
        self.dissolved_at_step = None
        self.dissolution_reason = None

    def add_delivery(self, delivery_id):
        self.delivery_ids.add(delivery_id)

    def remove_delivery(self, delivery_id):
        self.delivery_ids.discard(delivery_id)

    def dissolve(self, step, reason):
        self.is_active = False
        self.dissolved_at_step = step
        self.dissolution_reason = reason

    def size(self):
        return len(self.delivery_ids)

    def __repr__(self):
        return "Bundle%d:[%s]@V%d" % (
            self.id, ",".join("D%d" % d for d in sorted(self.delivery_ids)), self.vehicle.id)


# ============================================================================
# 6. THE SIMULATION
# ============================================================================

class Simulation(object):
    """Tick-level relay delivery simulation.

    Ordering within a tick:
      phase 0  activation
      phase 1  movement and arrival
      phase 2  unbundling of diverging bundles     (bundling arm only)
      phase 3  bundling and dispatch
      phase 4  metrics tick
    """

    def __init__(self, deliveries, adjacency, edge_lookup, cfg,
                 use_bundling=True, verbose=False):
        self.cfg = cfg
        self.adjacency = adjacency
        self.edge_lookup = edge_lookup
        self.use_bundling = use_bundling
        self.verbose = verbose

        self.time_step = cfg.TIME_STEP
        self.clock = 0
        self.step_count = 0
        self.max_time = cfg.max_sim_time

        self.all_deliveries = {d.id: d for d in deliveries}
        # activation is driven by a sorted pointer, not an O(n) scan per tick
        self._arrival_order = sorted(self.all_deliveries.values(),
                                     key=lambda d: (d.start_time, d.id))
        self._arrival_ptr = 0
        self.n_activated = 0

        self.waiting = set()        # standing at a node
        self.in_transit = set()     # traversing an edge

        self.bundles = {}
        self.active_bundles = set()

        self.vehicles = {}
        self._vehicle_pool = []     # min-heap of (available_at_time, seq, vehicle)
        self._pool_seq = 0
        self._holders = defaultdict(int)   # vehicle id -> orders currently holding it
        self.n_in_use = 0

        self.bundles_formed = 0
        self.bundles_dissolved = 0
        self.bundle_sizes_at_departure = []
        self.solo_departures = 0

        self.timeline = {
            "timestamps": [],
            "active_deliveries": [],
            "bundled_deliveries": [],
            "num_active_bundles": [],
            "avg_bundle_size": [],
            "vehicles_in_use": [],
        }
        self.peak_concurrent_vehicles = 0

    # -- vehicle bookkeeping ---------------------------------------------
    def acquire_vehicle(self, node):
        """A cooled-down vehicle from the pool, or a new one. Repositioning is free."""
        if self.use_bundling and self._vehicle_pool:
            available_at, _seq, vehicle = self._vehicle_pool[0]
            if available_at <= self.clock:
                heapq.heappop(self._vehicle_pool)
                vehicle.start_new_trip(node, self.clock)
                return vehicle

        vehicle = Vehicle(node, allow_reuse=self.use_bundling)
        vehicle.cooldown_period = self.cfg.VEHICLE_COOLDOWN
        vehicle.is_in_use = True
        self.vehicles[vehicle.id] = vehicle
        return vehicle

    def _hold_vehicle(self, delivery, vehicle):
        if delivery.held_vehicle is vehicle:
            return
        if delivery.held_vehicle is not None:
            self._drop_vehicle(delivery)
        delivery.held_vehicle = vehicle
        if self._holders[vehicle.id] == 0:
            self.n_in_use += 1
            vehicle.is_in_use = True
        self._holders[vehicle.id] += 1
        vehicle.all_deliveries_carried.add(delivery.id)

    def _drop_vehicle(self, delivery):
        vehicle = delivery.held_vehicle
        if vehicle is None:
            return
        delivery.held_vehicle = None
        self._holders[vehicle.id] -= 1
        if self._holders[vehicle.id] <= 0:
            self._holders[vehicle.id] = 0
            self.n_in_use -= 1
            vehicle.release(self.clock)
            if vehicle.allow_reuse:
                self._pool_seq += 1
                heapq.heappush(self._vehicle_pool,
                               (vehicle.available_at_time, self._pool_seq, vehicle))

    # -- bundle bookkeeping ----------------------------------------------
    def dissolve_bundle(self, bundle_id, reason):
        bundle = self.bundles.get(bundle_id)
        if bundle is None or not bundle.is_active:
            return
        for d_id in list(bundle.delivery_ids):
            self.all_deliveries[d_id].current_bundle_id = None
        bundle.dissolve(self.step_count, reason)
        self.active_bundles.discard(bundle_id)
        self.bundles_dissolved += 1

    def leave_bundle(self, delivery, reason):
        bundle_id = delivery.current_bundle_id
        if bundle_id is None:
            return
        bundle = self.bundles[bundle_id]
        bundle.remove_delivery(delivery.id)
        delivery.current_bundle_id = None
        if bundle.size() < 2:
            self.dissolve_bundle(bundle_id, reason)

    # -- the tick ---------------------------------------------------------
    def run(self):
        while self.clock < self.max_time and (
                self.waiting or self.in_transit or self._arrival_ptr < len(self._arrival_order)):
            self.step()
        self.finalize()

    def step(self):
        self.step_count += 1
        self.phase0_activate()
        self.phase1_move()
        if self.use_bundling:
            self.phase2_unbundle()
            self.phase3_dispatch_bundling()
        else:
            self.phase3_dispatch_baseline()
        self.phase4_metrics()
        self.clock += self.time_step

    # phase 0 -------------------------------------------------------------
    def phase0_activate(self):
        orders = self._arrival_order
        while self._arrival_ptr < len(orders) and orders[self._arrival_ptr].start_time <= self.clock:
            delivery = orders[self._arrival_ptr]
            self._arrival_ptr += 1
            if delivery.failed:
                continue                       # unroutable after edge failures
            if delivery.start_node == delivery.end_node:
                # degenerate order: the feasible-pair list excludes these, so
                # this is defensive only. It never gets a vehicle, so it must
                # not count towards n_activated (see assertion 8).
                delivery.completed = True
                delivery.end_time = self.clock
                continue
            delivery.activated = True
            self.n_activated += 1
            if self.use_bundling:
                delivery.hold_remaining = self.cfg.WAIT_TIME
            else:
                # baseline: one dedicated vehicle at activation, no holding
                delivery.hold_remaining = 0
                vehicle = Vehicle(delivery.current_node, allow_reuse=False)
                self.vehicles[vehicle.id] = vehicle
                self._hold_vehicle(delivery, vehicle)
            self.waiting.add(delivery.id)

    # phase 1 -------------------------------------------------------------
    def phase1_move(self):
        arrived = []
        for d_id in self.in_transit:
            delivery = self.all_deliveries[d_id]
            delivery.time_till_next_node -= self.time_step
            if delivery.time_till_next_node <= 0:
                arrived.append(d_id)

        for d_id in arrived:
            delivery = self.all_deliveries[d_id]
            self.in_transit.discard(d_id)

            next_node = delivery.next_node()
            if next_node is None:                       # defensive; should not fire
                delivery.in_transition = False
                self.waiting.add(d_id)
                continue

            delivery.arrive_at_node(next_node)

            if delivery.current_node == delivery.end_node:
                delivery.completed = True
                delivery.end_time = self.clock
                self.leave_bundle(delivery, "completed")
                self._drop_vehicle(delivery)
                continue

            # standing at an intermediate hotspot: hold timer restarts
            delivery.hold_remaining = self.cfg.WAIT_TIME
            self.waiting.add(d_id)

            bundle_id = delivery.current_bundle_id
            if bundle_id is not None:
                self.bundles[bundle_id].current_node = delivery.current_node
            elif self.use_bundling:
                # a solo relay drops its vehicle at every hop; the baseline arm
                # keeps its dedicated vehicle all the way to the destination
                self._drop_vehicle(delivery)

    # phase 2 -------------------------------------------------------------
    def phase2_unbundle(self):
        """Dissolve bundles whose standing members no longer share a next hop.

        The largest group keeps the mother vehicle (it stays held); every other
        member drops it and returns to the waiting pool with its hold timer
        running.  Phase 3 re-forms the surviving groups on dispatch.
        """
        for bundle_id in list(self.active_bundles):
            bundle = self.bundles[bundle_id]
            standing = [d_id for d_id in bundle.delivery_ids
                        if not self.all_deliveries[d_id].in_transition
                        and not self.all_deliveries[d_id].completed]
            if len(standing) < 2 or len(standing) != bundle.size():
                continue

            groups = defaultdict(list)
            for d_id in standing:
                groups[self.all_deliveries[d_id].next_node()].append(d_id)
            if len(groups) <= 1:
                continue

            ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), sorted(kv[1])[0]))
            self.dissolve_bundle(bundle_id, "path_divergence")
            for position, (_next_node, delivery_ids) in enumerate(ordered):
                if position == 0:
                    continue                    # largest group keeps the vehicle
                for d_id in delivery_ids:
                    self._drop_vehicle(self.all_deliveries[d_id])

    # phase 3 -------------------------------------------------------------
    def phase3_dispatch_bundling(self):
        buckets = defaultdict(list)
        for d_id in self.waiting:
            delivery = self.all_deliveries[d_id]
            next_node = delivery.next_node()
            if next_node is None:               # defensive; should not fire
                continue
            buckets[(delivery.current_node, next_node)].append(d_id)

        departed = set()
        vehicles_used_this_tick = set()

        for (current_node, next_node), delivery_ids in buckets.items():
            delivery_ids.sort()
            if len(delivery_ids) >= 2:
                for start in range(0, len(delivery_ids), self.cfg.MAX_BUNDLE_SIZE):
                    group = delivery_ids[start:start + self.cfg.MAX_BUNDLE_SIZE]
                    if len(group) >= 2:
                        self.depart_group(group, current_node, next_node, vehicles_used_this_tick)
                        departed.update(group)
                    elif self.try_depart_solo(group[0], current_node, next_node,
                                              vehicles_used_this_tick):
                        departed.add(group[0])
            elif self.try_depart_solo(delivery_ids[0], current_node, next_node,
                                      vehicles_used_this_tick):
                departed.add(delivery_ids[0])

        self.waiting -= departed

        # anything that did not depart burns hold time
        for d_id in self.waiting:
            delivery = self.all_deliveries[d_id]
            delivery.hold_remaining -= self.time_step
            delivery.total_hold_time += self.time_step

    def phase3_dispatch_baseline(self):
        """Every order moves on its own dedicated vehicle, no holding, no sharing."""
        for d_id in list(self.waiting):
            delivery = self.all_deliveries[d_id]
            next_node = delivery.next_node()
            if next_node is None:
                continue
            vehicle = delivery.held_vehicle
            distance_km = self.depart_order(delivery, delivery.current_node, next_node)
            vehicle.move_to(next_node, distance_km)
            self.waiting.discard(d_id)
            self.in_transit.add(d_id)

    def depart_order(self, delivery, current_node, next_node):
        """Order-side departure.  Returns the edge distance in km.

        num_vehicle_changes is charged here and only here: exactly one relay
        handoff per edge traversal per order, in both arms, so the per-hop
        penalty is a property of the graph rather than of which code path fired.
        """
        edge = self.edge_lookup.get((current_node, next_node))
        if edge is None:
            raise AssertionError(
                "no edge N%d -> N%d in the current adjacency (delivery %d)"
                % (current_node, next_node, delivery.id))
        distance_km, travel_time = edge

        delivery.in_transition = True
        delivery.time_till_next_node = travel_time
        delivery.distance_traveled += distance_km
        delivery.num_vehicle_changes += 1
        delivery.hold_remaining = 0
        return distance_km

    def pick_group_vehicle(self, group, current_node, vehicles_used_this_tick):
        """Prefer the vehicle already held by most of the group (relay continuity)."""
        held = Counter()
        for d_id in group:
            vehicle = self.all_deliveries[d_id].held_vehicle
            if vehicle is not None and vehicle.id not in vehicles_used_this_tick:
                held[vehicle.id] += 1
        if held:
            best_id = max(sorted(held), key=lambda vid: held[vid])
            return self.vehicles[best_id]
        return self.acquire_vehicle(current_node)

    def depart_group(self, group, current_node, next_node, vehicles_used_this_tick):
        # ASSERTION 6: every member of a departing bundle shares the same next hop
        for d_id in group:
            delivery = self.all_deliveries[d_id]
            if delivery.current_node != current_node or delivery.next_node() != next_node:
                raise AssertionError(
                    "ASSERTION 6 FAILED: delivery %d at N%d heading to N%s was placed in "
                    "the bundle N%d -> N%d"
                    % (d_id, delivery.current_node, delivery.next_node(),
                       current_node, next_node))

        vehicle = self.pick_group_vehicle(group, current_node, vehicles_used_this_tick)
        vehicles_used_this_tick.add(vehicle.id)

        for d_id in group:
            self.leave_bundle(self.all_deliveries[d_id], "reformed")

        bundle = Bundle(group, vehicle, current_node, next_node, self.step_count)
        self.bundles[bundle.id] = bundle
        self.active_bundles.add(bundle.id)
        self.bundles_formed += 1
        self.bundle_sizes_at_departure.append(len(group))

        distance_km = self.edge_lookup[(current_node, next_node)][0]
        for d_id in group:
            delivery = self.all_deliveries[d_id]
            delivery.current_bundle_id = bundle.id
            delivery.times_bundled += 1
            self._hold_vehicle(delivery, vehicle)
            self.depart_order(delivery, current_node, next_node)
            self.in_transit.add(d_id)

        vehicle.move_to(next_node, distance_km)     # one edge, one odometer tick

    def try_depart_solo(self, d_id, current_node, next_node, vehicles_used_this_tick):
        """A lone order departs once its hold has expired.

        Notebook 02 had no solo-departure path at all: the len(delivery_ids) < 2
        branch set a timer and continued, and initiate_delivery_movement was
        only reachable from inside the >= 2 branch, so a lone order waited
        forever.  SOLO_DEPARTURE=False reproduces that behaviour on purpose.
        """
        delivery = self.all_deliveries[d_id]
        if delivery.hold_remaining > 0 or not self.cfg.SOLO_DEPARTURE:
            return False

        self.leave_bundle(delivery, "solo_departure")

        vehicle = delivery.held_vehicle
        if vehicle is None or vehicle.id in vehicles_used_this_tick:
            vehicle = self.acquire_vehicle(current_node)
        vehicles_used_this_tick.add(vehicle.id)

        self._hold_vehicle(delivery, vehicle)
        distance_km = self.depart_order(delivery, current_node, next_node)
        vehicle.move_to(next_node, distance_km)
        self.in_transit.add(d_id)
        self.solo_departures += 1
        return True

    # phase 4 -------------------------------------------------------------
    def phase4_metrics(self):
        bundled_now = 0
        sizes = []
        for bundle_id in self.active_bundles:
            size = self.bundles[bundle_id].size()
            bundled_now += size
            sizes.append(size)

        self.timeline["timestamps"].append(self.clock)
        self.timeline["active_deliveries"].append(len(self.waiting) + len(self.in_transit))
        self.timeline["bundled_deliveries"].append(bundled_now)
        self.timeline["num_active_bundles"].append(len(self.active_bundles))
        self.timeline["avg_bundle_size"].append(float(np.mean(sizes)) if sizes else 0.0)
        self.timeline["vehicles_in_use"].append(self.n_in_use)
        if self.n_in_use > self.peak_concurrent_vehicles:
            self.peak_concurrent_vehicles = self.n_in_use

    # -- results ----------------------------------------------------------
    def finalize(self):
        for delivery in self.all_deliveries.values():
            delivery.finalize_metrics(self.cfg.VEHICLE_CHANGE_TIME)

    def get_results(self):
        deliveries = list(self.all_deliveries.values())
        n_orders = len(deliveries)

        completed = [d for d in deliveries if d.completed]
        successful = [d for d in completed if d.successful]
        failed = [d for d in deliveries if d.failed]

        # Distance over all traffic, completed or not: every vehicle-km was
        # earned by at least one package-km, which is what makes the saving
        # well defined and assertion 7 exact.
        package_distance_km = sum(d.distance_traveled for d in deliveries)
        vehicle_distance_km = sum(v.total_distance for v in self.vehicles.values())
        distance_saved_km = package_distance_km - vehicle_distance_km

        times = [d.actual_delivery_time for d in completed]
        delays = [d.delay for d in completed]
        holds = [d.total_hold_time for d in deliveries if d.activated]
        hops = [d.hops() for d in completed]

        tolerance = 1e-6 * max(1.0, package_distance_km)
        if self.use_bundling and vehicle_distance_km > package_distance_km + tolerance:
            raise AssertionError(
                "ASSERTION 7 FAILED: vehicle_distance_km (%.3f) > package_distance_km (%.3f)"
                % (vehicle_distance_km, package_distance_km))

        if not self.use_bundling:
            distinct = len(self.vehicles)
            if distinct != self.n_activated:
                raise AssertionError(
                    "ASSERTION 8 FAILED: baseline distinct_vehicles=%d but %d orders were "
                    "activated (n_orders=%d, failed=%d)"
                    % (distinct, self.n_activated, n_orders, len(failed)))

        ever_bundled = sum(1 for d in deliveries if d.times_bundled > 0)

        return {
            # volume
            "n_orders": n_orders,
            "completed": len(completed),
            "completion_rate": len(completed) / n_orders if n_orders else 0.0,
            "successful": len(successful),
            "success_rate": len(successful) / n_orders if n_orders else 0.0,
            "failed": len(failed),
            "failed_rate": len(failed) / n_orders if n_orders else 0.0,

            # distance
            "vehicle_distance_km": vehicle_distance_km,
            "package_distance_km": package_distance_km,
            "distance_saved_km": distance_saved_km,
            "distance_saved_pct": (100.0 * distance_saved_km / package_distance_km)
                                  if package_distance_km > 0 else 0.0,

            # time
            "avg_delivery_time": float(np.mean(times)) if times else 0.0,
            "median_delivery_time": float(np.median(times)) if times else 0.0,
            "p90": float(np.percentile(times, 90)) if times else 0.0,
            "p95": float(np.percentile(times, 95)) if times else 0.0,
            "avg_delay": float(np.mean(delays)) if delays else 0.0,
            "median_delay": float(np.median(delays)) if delays else 0.0,
            "max_delay": float(np.max(delays)) if delays else 0.0,
            "avg_hold_time": float(np.mean(holds)) if holds else 0.0,

            # vehicles
            "distinct_vehicles": len(self.vehicles),
            "vehicle_trips": sum(v.total_trips for v in self.vehicles.values()),
            "peak_concurrent_vehicles": self.peak_concurrent_vehicles,
            "avg_concurrent_vehicles": float(np.mean(self.timeline["vehicles_in_use"]))
                                       if self.timeline["vehicles_in_use"] else 0.0,

            # bundling (zero by construction in the baseline arm)
            "orders_ever_bundled": ever_bundled,
            "bundle_participation_rate": ever_bundled / n_orders if n_orders else 0.0,
            "bundles_formed": self.bundles_formed,
            "avg_bundle_size": float(np.mean(self.bundle_sizes_at_departure))
                               if self.bundle_sizes_at_departure else 0.0,
            "max_bundle_size_observed": int(np.max(self.bundle_sizes_at_departure))
                                        if self.bundle_sizes_at_departure else 0,
            "avg_hops": float(np.mean(hops)) if hops else 0.0,

            # housekeeping
            "solo_departures": self.solo_departures,
            "sim_end_clock": self.clock,
            "in_flight_at_cutoff": len(self.waiting) + len(self.in_transit),
        }


METRIC_COLUMNS = [
    "n_orders", "completed", "completion_rate", "successful", "success_rate",
    "failed", "failed_rate",
    "vehicle_distance_km", "package_distance_km", "distance_saved_km", "distance_saved_pct",
    "avg_delivery_time", "median_delivery_time", "p90", "p95",
    "avg_delay", "median_delay", "max_delay", "avg_hold_time",
    "distinct_vehicles", "vehicle_trips", "peak_concurrent_vehicles", "avg_concurrent_vehicles",
    "orders_ever_bundled", "bundle_participation_rate", "bundles_formed",
    "avg_bundle_size", "max_bundle_size_observed", "avg_hops",
    "solo_departures", "sim_end_clock", "in_flight_at_cutoff",
]


# ============================================================================
# 7. PER-GRAPH ROUTING AND EDGE FAILURES
# ============================================================================

def route_orders(order_pool, time_limits, adjacency, graph_type, verbose=True):
    """Route every order on this graph, carrying the frozen deadline over untouched."""
    Delivery.id_counter = 0

    prev_cache = {}
    dist_cache = {}
    deliveries = []
    dropped = []

    for order in order_pool:
        if order.start_node not in dist_cache:
            dist, prev, _hops = dijkstra_all(adjacency, order.start_node)
            dist_cache[order.start_node] = dist
            prev_cache[order.start_node] = prev

        travel_time = dist_cache[order.start_node].get(order.end_node)
        if travel_time is None:
            dropped.append(order.id)
            continue
        path = reconstruct_path(prev_cache[order.start_node], order.end_node)
        deliveries.append(Delivery(order, path, travel_time, time_limits[order.id]))

    # ASSERTION 5: all three graphs are connected, so nothing may be dropped.
    if dropped:
        raise AssertionError(
            "ASSERTION 5 FAILED: %d orders have no path on %s (first few: %s). "
            "The denominator would differ between arms."
            % (len(dropped), graph_type, dropped[:10]))

    avg_hops = float(np.mean([len(d.base_path) - 1 for d in deliveries])) if deliveries else 0.0
    avg_time = float(np.mean([d.base_expected_travel_time for d in deliveries])) if deliveries else 0.0
    if verbose:
        print("  routed %d orders on %-6s : avg_hops=%.2f  avg_expected_travel_time=%.1fs"
              % (len(deliveries), graph_type, avg_hops, avg_time))
    return deliveries, {"avg_hops": avg_hops, "avg_expected_travel_time": avg_time}


def drop_edges(adjacency, drop_rate=0.0, seed=42):
    """Degraded adjacency with drop_rate of the undirected edges removed."""
    if drop_rate <= 0.0:
        return adjacency, build_edge_lookup(adjacency), 0, 0

    all_edges = set()
    for u, neighbours in adjacency.items():
        for v, _distance_km, _time_s in neighbours:
            all_edges.add(frozenset((u, v)))

    n_drop = max(1, int(len(all_edges) * drop_rate))
    rng = random.Random(seed)
    failed = set(rng.sample(sorted(all_edges, key=lambda e: tuple(sorted(e))), n_drop))

    degraded = {
        u: [(v, distance_km, time_s) for v, distance_km, time_s in neighbours
            if frozenset((u, v)) not in failed]
        for u, neighbours in adjacency.items()
    }
    return degraded, build_edge_lookup(degraded), n_drop, len(all_edges)


def apply_edge_failures(deliveries, degraded_adjacency, degraded_lookup):
    """Revalidate each baked-in path; reroute if broken, mark failed if impossible.

    The frozen time_limit is never recomputed here.  Notebook 02's version
    overwrote it with the 'multiply' rule regardless of BUFFER_TIME_TYPE, which
    handed a rerouted order a fresh, looser deadline.
    """
    rerouted = 0
    failed = 0
    dist_cache = {}
    prev_cache = {}

    for delivery in deliveries:
        path = delivery.shortest_path
        if all((path[i], path[i + 1]) in degraded_lookup for i in range(len(path) - 1)):
            continue

        if delivery.start_node not in dist_cache:
            dist, prev, _hops = dijkstra_all(degraded_adjacency, delivery.start_node)
            dist_cache[delivery.start_node] = dist
            prev_cache[delivery.start_node] = prev

        travel_time = dist_cache[delivery.start_node].get(delivery.end_node)
        if travel_time is None:
            delivery.failed = True
            failed += 1
            continue

        delivery.shortest_path = reconstruct_path(prev_cache[delivery.start_node],
                                                  delivery.end_node)
        delivery.expected_travel_time = travel_time      # time_limit untouched
        rerouted += 1

    return rerouted, failed


# ============================================================================
# 8. RUNNER AND OUTPUT
# ============================================================================

def summarise(values):
    array = np.asarray(values, dtype=float)
    n = len(array)
    std = float(np.std(array, ddof=1)) if n > 1 else 0.0
    return {
        "mean": float(np.mean(array)) if n else 0.0,
        "std": std,
        "min": float(np.min(array)) if n else 0.0,
        "max": float(np.max(array)) if n else 0.0,
        "median": float(np.median(array)) if n else 0.0,
        "ci_95": (1.96 * std / math.sqrt(n)) if n else 0.0,
    }


def per_order_rows(deliveries, mode):
    rows = []
    for delivery in sorted(deliveries, key=lambda d: d.id):
        rows.append({
            "mode": mode,
            "id": delivery.id,
            "start_node": delivery.start_node,
            "end_node": delivery.end_node,
            "start_time": delivery.start_time,
            "end_time": delivery.end_time if delivery.end_time is not None else "",
            "expected_travel_time": round(delivery.expected_travel_time, 3),
            "time_limit": round(delivery.time_limit, 3),
            "actual_delivery_time": round(delivery.actual_delivery_time, 3),
            "delay": round(delivery.delay, 3),
            "within_buffer": int(delivery.within_buffer),
            "hops": delivery.hops(),
            "distance_traveled": round(delivery.distance_traveled, 4),
            "num_vehicle_changes": delivery.num_vehicle_changes,
            "times_bundled": delivery.times_bundled,
            "total_hold_time": delivery.total_hold_time,
            "completed": int(delivery.completed),
            "failed": int(delivery.failed),
        })
    return rows


PER_ORDER_COLUMNS = [
    "mode", "id", "start_node", "end_node", "start_time", "end_time",
    "expected_travel_time", "time_limit", "actual_delivery_time", "delay",
    "within_buffer", "hops", "distance_traveled", "num_vehicle_changes",
    "times_bundled", "total_hold_time", "completed", "failed",
]


def run_experiment(graph_type, drop_rate, cfg, shared, quiet=False):
    """N_RUNS x 2 arms on one graph at one drop rate.  Writes its own folder."""
    adjacency = shared["adjacencies"][graph_type]
    order_pool = shared["order_pool"]
    time_limits = shared["time_limits"]

    deliveries, routing_stats = route_orders(order_pool, time_limits, adjacency, graph_type,
                                             verbose=not quiet)

    rows = []
    order_rows = []
    drop_info = {}

    for run_idx in range(cfg.N_RUNS):
        degraded_adjacency, degraded_lookup, n_dropped, n_edges = drop_edges(
            adjacency, drop_rate, seed=cfg.GLOBAL_SEED + run_idx)
        drop_info = {"edges_dropped": n_dropped, "edges_total": n_edges}

        for mode in MODES:
            for delivery in deliveries:
                delivery.reset()
            Delivery.id_counter = 0
            Vehicle.id_counter = 0
            Bundle.id_counter = 0

            rerouted = failed = 0
            if drop_rate > 0.0:
                rerouted, failed = apply_edge_failures(deliveries, degraded_adjacency,
                                                       degraded_lookup)

            simulation = Simulation(deliveries, degraded_adjacency, degraded_lookup, cfg,
                                    use_bundling=(mode == "bundling"))
            simulation.run()
            results = simulation.get_results()
            results["rerouted"] = rerouted

            row = {"run_idx": run_idx, "mode": mode}
            row.update({key: results[key] for key in METRIC_COLUMNS})
            rows.append(row)

            if run_idx == 0 and cfg.SAVE_PER_ORDER:
                order_rows.extend(per_order_rows(deliveries, mode))

            if not quiet:
                print("    run %d %-8s : completed %5d/%-5d  success %6.2f%%  "
                      "veh_km %9.1f  saved %5.1f%%  vehicles %5d"
                      % (run_idx, mode, results["completed"], results["n_orders"],
                         100.0 * results["success_rate"], results["vehicle_distance_km"],
                         results["distance_saved_pct"], results["distinct_vehicles"]))

    out_dir = cfg.out_dir(graph_type, drop_rate)
    os.makedirs(str(out_dir), exist_ok=True)

    write_csv(out_dir / "runs.csv", ["run_idx", "mode"] + METRIC_COLUMNS, rows)

    modes_summary = {}
    for mode in MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        modes_summary[mode] = {
            metric: summarise([r[metric] for r in mode_rows]) for metric in METRIC_COLUMNS
        }

    summary = {
        "city_key": cfg.city_key,
        "graph_type": graph_type,
        "drop_rate": drop_rate,
        "n_runs": cfg.N_RUNS,
        "order_pool_hash": shared["pool_hash"],
        "time_limits_hash": shared["limits_hash"],
        "n_orders": len(deliveries),
        "routing": routing_stats,
        "edge_failures": drop_info,
        "config": cfg.as_dict(),
        "modes": modes_summary,
    }
    write_json(out_dir / "summary.json", summary)

    if cfg.SAVE_PER_ORDER and order_rows:
        write_csv(out_dir / "orders_run0.csv", PER_ORDER_COLUMNS, order_rows)

    if not quiet:
        print("    -> %s" % out_dir)
    return summary


# ============================================================================
# 9. AGGREGATION AND PLOTS
# ============================================================================

# Categorical slots 1 and 2 of the validated default palette (blue / orange),
# plus slot 3 (aqua) for the graph-level panel that belongs to neither arm.
ARM_COLORS = {"bundling": "#2a78d6", "baseline": "#eb6834"}
GRAPH_LEVEL_COLOR = "#1baf7a"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#dcdbd6"

# (metric, title, y-axis unit, scale, arms shown).  distance_saved_pct is zero
# for the baseline by construction, so that panel shows the bundling arm alone
# rather than a constant-zero second series.
PLOT_PANELS = [
    ("success_rate",        "Success rate",      "% of orders on time", 100.0, MODES),
    ("vehicle_distance_km", "Vehicle distance",  "km",                  1.0,   MODES),
    ("distance_saved_pct",  "Distance saved",    "% of package-km",     1.0,   ("bundling",)),
    ("distinct_vehicles",   "Distinct vehicles", "count",               1.0,   MODES),
    ("avg_delivery_time",   "Avg delivery time", "seconds",             1.0,   MODES),
]

COMPARISON_COLUMNS = [
    "success_rate", "completion_rate", "failed_rate",
    "vehicle_distance_km", "package_distance_km", "distance_saved_km", "distance_saved_pct",
    "avg_delivery_time", "median_delivery_time", "p90", "p95",
    "avg_delay", "max_delay", "avg_hold_time",
    "distinct_vehicles", "vehicle_trips", "peak_concurrent_vehicles",
    "bundle_participation_rate", "bundles_formed", "avg_bundle_size", "avg_hops",
]


def load_summaries(cfg):
    """{(graph_type, drop_rate): summary} for whatever has already been run."""
    summaries = {}
    for graph_type in GRAPH_TYPES:
        for drop_rate in cfg.DROP_RATES:
            path = cfg.out_dir(graph_type, drop_rate) / "summary.json"
            if path.is_file():
                summaries[(graph_type, drop_rate)] = read_json(path)
    return summaries


def check_shared_inputs(summaries):
    """ASSERTIONS 3 and 4 across separate invocations.

    Every graph must have run against the same order pool and the same frozen
    deadlines; the hashes are what make that checkable a day later.
    """
    pool_hashes = set(s.get("order_pool_hash") for s in summaries.values())
    limit_hashes = set(s.get("time_limits_hash") for s in summaries.values())
    problems = []
    if len(pool_hashes) > 1:
        problems.append("ASSERTION 3 FAILED: order pool hashes differ across graphs: %s"
                        % sorted(pool_hashes))
    if len(limit_hashes) > 1:
        problems.append("ASSERTION 4 FAILED: time limit hashes differ across graphs: %s"
                        % sorted(limit_hashes))
    if problems:
        raise AssertionError("\n".join(problems))
    return (sorted(pool_hashes)[0] if pool_hashes else None,
            sorted(limit_hashes)[0] if limit_hashes else None)


def aggregate(cfg, quiet=False):
    summaries = load_summaries(cfg)
    if not summaries:
        print("No summary.json found under %s -- run the experiment first."
              % cfg.city_results_dir)
        return None

    pool_hash, limits_hash = check_shared_inputs(summaries)

    graph_stats = {}
    stats_path = cfg.shared_dir / "graph_stats.json"
    if stats_path.is_file():
        graph_stats = read_json(stats_path)

    rows = []
    for (graph_type, drop_rate) in sorted(summaries, key=lambda k: (k[1], GRAPH_TYPES.index(k[0]))):
        summary = summaries[(graph_type, drop_rate)]
        stats = graph_stats.get(graph_type, {})
        for mode in MODES:
            metrics = summary["modes"].get(mode, {})
            row = {
                "city_key": summary["city_key"],
                "graph_type": graph_type,
                "drop_rate": drop_rate,
                "mode": mode,
                "n_runs": summary["n_runs"],
                "n_orders": summary["n_orders"],
                "avg_hops_routed": round(summary["routing"]["avg_hops"], 4),
                "coverage_within_max_delivery_time":
                    round(stats.get("coverage_within_max_delivery_time", float("nan")), 6),
                "graph_edges": stats.get("edges", ""),
                "graph_avg_degree": round(stats.get("avg_degree", float("nan")), 4),
            }
            for metric in COMPARISON_COLUMNS:
                stat = metrics.get(metric, {})
                row[metric] = round(stat.get("mean", float("nan")), 6)
                row[metric + "_ci95"] = round(stat.get("ci_95", float("nan")), 6)
            rows.append(row)

    fieldnames = ["city_key", "graph_type", "drop_rate", "mode", "n_runs", "n_orders",
                  "avg_hops_routed", "coverage_within_max_delivery_time",
                  "graph_edges", "graph_avg_degree"]
    for metric in COMPARISON_COLUMNS:
        fieldnames.extend([metric, metric + "_ci95"])

    os.makedirs(str(cfg.comparison_dir), exist_ok=True)
    table_path = cfg.comparison_dir / "comparison_table.csv"
    write_csv(table_path, fieldnames, rows)

    nested = defaultdict(dict)
    for (graph_type, drop_rate), summary in summaries.items():
        nested["drop_%02d" % int(round(drop_rate * 100))][graph_type] = {
            "n_orders": summary["n_orders"],
            "n_runs": summary["n_runs"],
            "routing": summary["routing"],
            "graph_stats": graph_stats.get(graph_type, {}),
            "modes": summary["modes"],
        }
    comparison_summary = {
        "city_key": cfg.city_key,
        "order_pool_hash": pool_hash,
        "time_limits_hash": limits_hash,
        "graphs_present": sorted(set(g for g, _d in summaries)),
        "drop_rates_present": sorted(set(d for _g, d in summaries)),
        "config": cfg.as_dict(),
        "by_drop_rate": dict(nested),
    }
    write_json(cfg.comparison_dir / "comparison_summary.json", comparison_summary)

    print("\n" + rule())
    print("AGGREGATE")
    print(rule())
    print("  order pool hash : %s" % pool_hash)
    print("  deadline hash   : %s   (identical across graphs -> assertions 3 and 4 hold)"
          % limits_hash)
    print("  wrote %s" % table_path)
    print("  wrote %s" % (cfg.comparison_dir / "comparison_summary.json"))
    print_comparison_table(summaries, graph_stats, cfg)

    if cfg.MAKE_PLOTS:
        make_plots(summaries, graph_stats, cfg)

    return comparison_summary


def print_comparison_table(summaries, graph_stats, cfg):
    for drop_rate in sorted(set(d for _g, d in summaries)):
        print("\n  drop_rate = %.2f" % drop_rate)
        header = ("    %-7s %-9s %8s %8s %10s %9s %9s %9s"
                  % ("graph", "mode", "success%", "compl%", "veh_km", "saved%",
                     "vehicles", "avg_t_s"))
        print(header)
        print("    " + "-" * (len(header) - 4))
        for graph_type in GRAPH_TYPES:
            summary = summaries.get((graph_type, drop_rate))
            if summary is None:
                continue
            for mode in MODES:
                metrics = summary["modes"][mode]
                print("    %-7s %-9s %8.2f %8.2f %10.1f %9.2f %9.0f %9.1f"
                      % (graph_type, mode,
                         100.0 * metrics["success_rate"]["mean"],
                         100.0 * metrics["completion_rate"]["mean"],
                         metrics["vehicle_distance_km"]["mean"],
                         metrics["distance_saved_pct"]["mean"],
                         metrics["distinct_vehicles"]["mean"],
                         metrics["avg_delivery_time"]["mean"]))
            stats = graph_stats.get(graph_type)
            if stats:
                print("    %-7s %-9s coverage within %.0fs = %5.1f%%   (%d edges, avg deg %.2f)"
                      % (graph_type, "[graph]", cfg.MAX_DELIVERY_TIME,
                         100.0 * stats["coverage_within_max_delivery_time"],
                         stats["edges"], stats["avg_degree"]))


def make_plots(summaries, graph_stats, cfg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                              # pragma: no cover
        print("  plots skipped: matplotlib unavailable (%s)" % exc)
        return

    drop_rates = sorted(set(d for _g, d in summaries))
    single = len(drop_rates) == 1

    for drop_rate in drop_rates:
        present = [g for g in GRAPH_TYPES if (g, drop_rate) in summaries]
        if not present:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        fig.patch.set_facecolor("#fcfcfb")
        axes = axes.ravel()
        x = np.arange(len(present), dtype=float)
        width = 0.36

        for panel_idx, (metric, title, unit, scale, arms) in enumerate(PLOT_PANELS):
            ax = axes[panel_idx]
            for arm_idx, mode in enumerate(arms):
                means = [summaries[(g, drop_rate)]["modes"][mode][metric]["mean"] * scale
                         for g in present]
                errors = [summaries[(g, drop_rate)]["modes"][mode][metric]["ci_95"] * scale
                          for g in present]
                if len(arms) == 1:
                    offset, bar_width = 0.0, width * 1.3
                else:
                    offset, bar_width = (arm_idx - 0.5) * (width + 0.02), width
                ax.bar(x + offset, means, bar_width, label=mode,
                       color=ARM_COLORS[mode], edgecolor="#fcfcfb", linewidth=2.0)
                if any(e > 0 for e in errors):
                    ax.errorbar(x + offset, means, yerr=errors, fmt="none",
                                ecolor=INK_SECONDARY, elinewidth=1.4, capsize=4)
                label_bars(ax, x + offset, means, errors)
            style_axis(ax, title, unit, x, present)

        # graph-level panel: one bar per graph, belongs to neither arm
        ax = axes[len(PLOT_PANELS)]
        coverage = [100.0 * graph_stats.get(g, {}).get("coverage_within_max_delivery_time", 0.0)
                    for g in present]
        ax.bar(x, coverage, width * 1.3, color=GRAPH_LEVEL_COLOR,
               edgecolor="#fcfcfb", linewidth=2.0)
        label_bars(ax, x, coverage, [0.0] * len(coverage))
        style_axis(ax, "OD coverage within %.0fs (graph-level)" % cfg.MAX_DELIVERY_TIME,
                   "% of all OD pairs", x, present)
        ax.set_ylim(0, 112)

        handles = [plt.Rectangle((0, 0), 1, 1, color=ARM_COLORS[m]) for m in MODES]
        fig.legend(handles, list(MODES), loc="upper right", frameon=False,
                   ncol=2, fontsize=10, bbox_to_anchor=(0.995, 0.985))
        fig.suptitle("%s  -  %d runs, drop rate %.0f%%  (error bars: 95%% CI)"
                     % (cfg.city_key, cfg.N_RUNS, 100.0 * drop_rate),
                     fontsize=13, color=INK_PRIMARY, x=0.01, ha="left", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        name = "comparison_plots.png" if single else \
               ("comparison_plots_%s.png" % drop_folder(drop_rate))
        path = cfg.comparison_dir / name
        fig.savefig(str(path), dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        print("  wrote %s" % path)


def label_bars(ax, positions, values, errors):
    """Direct-label each bar clear of its error-bar cap, never on top of it."""
    span = max(values + [v + e for v, e in zip(values, errors)] + [1e-9])
    for position, value, error in zip(positions, values, errors):
        ax.annotate("%.4g" % value,
                    xy=(position, value + error + 0.03 * span),
                    ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)


def style_axis(ax, title, unit, x, labels):
    ax.set_title(title, fontsize=11, color=INK_PRIMARY, loc="left", pad=8)
    ax.set_ylabel(unit, fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=INK_PRIMARY)
    ax.tick_params(axis="y", labelsize=8, colors=INK_SECONDARY, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_facecolor("#fcfcfb")
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.margins(y=0.16)


# ============================================================================
# 10. STARTUP BANNER AND MAIN
# ============================================================================

def print_banner(cfg, graphs, stats, pool_hash, n_orders, feasible_pairs, num_hotspots):
    print(rule())
    print("UMST / MST / CLIQUE BASELINE COMPARISON")
    print(rule())
    print("resolved config")
    for name, _conv, _help in CONFIG_SPEC:
        print("  %-24s = %s" % (name, getattr(cfg, name)))
    print()
    print("paths")
    print("  city folder : %s" % cfg.data_dir)
    print("  graph dir   : %s" % cfg.graph_dir)
    print("  results     : %s" % cfg.city_results_dir)
    print()
    print("graphs (canonical node index shared by all three)")
    print("  %-7s %6s %6s %9s %9s %10s %12s"
          % ("graph", "nodes", "edges", "avg_deg", "density", "edge_km", "coverage"))
    for graph_type in GRAPH_TYPES:
        s = stats[graph_type]
        print("  %-7s %6d %6d %9.2f %9.4f %10.1f %11.1f%%"
              % (graph_type, s["nodes"], s["edges"], s["avg_degree"], s["density"],
                 s["total_edge_length_km"],
                 100.0 * s["coverage_within_max_delivery_time"]))
    print("  coverage = share of the %d ordered OD pairs routable within %.0fs"
          % (stats[GRAPH_TYPES[0]]["total_od_pairs"], cfg.MAX_DELIVERY_TIME))
    print()
    print("workload")
    print("  hotspots            : %d" % num_hotspots)
    print("  LOAD_PER_HOTSPOT    : %d  (per hotspot, NOT a total)" % cfg.LOAD_PER_HOTSPOT)
    print("  TOTAL ORDERS        : %d" % n_orders)
    print("  feasible OD pairs   : %d of %d on %s"
          % (feasible_pairs, stats[cfg.ORDER_POOL_GRAPH]["total_od_pairs"], cfg.ORDER_POOL_GRAPH))
    print("  order pool hash     : %s" % pool_hash)
    print()

    pool_coverage = stats[cfg.ORDER_POOL_GRAPH]["coverage_within_max_delivery_time"]
    print("WARNING")
    print("  The order pool is defined by %s feasibility, so an order that is feasible"
          % cfg.ORDER_POOL_GRAPH)
    print("  there may exceed MAX_DELIVERY_TIME on a sparser graph. Every arm attempts")
    print("  the identical order set and a graph's failures are its own -- that is a real")
    print("  result about sparsity, not a bug.")
    for graph_type in GRAPH_TYPES:
        if graph_type == cfg.ORDER_POOL_GRAPH:
            continue
        gap = pool_coverage - stats[graph_type]["coverage_within_max_delivery_time"]
        print("    %-6s covers %5.1f%% of OD pairs vs %5.1f%% on %s  (gap %.1f pts)"
              % (graph_type,
                 100.0 * stats[graph_type]["coverage_within_max_delivery_time"],
                 100.0 * pool_coverage, cfg.ORDER_POOL_GRAPH, 100.0 * gap))
    print()
    print("NOTE")
    print("  mst_graph.graphml was built with weight='distance' (road km) while the MSTs")
    print("  inside UMST were built with weight='weight' (geodesic km), so MST is not the")
    print("  tree UMST was constructed around and is not in general a subgraph of it.")
    if cfg.DROP_RATES == [0.0] and cfg.N_RUNS > 1:
        print()
        print("NOTE")
        print("  The simulation is deterministic and DROP_RATES=[0.0], so the %d runs are"
              % cfg.N_RUNS)
        print("  identical and every CI will be zero. Variance comes from edge failures.")
    print(rule())


def prepare_shared(cfg, quiet=False):
    """Load graphs, build the canonical index, freeze the pool and the deadlines."""
    if not cfg.graph_dir.is_dir():
        raise FileNotFoundError("Graph directory does not exist: %s" % cfg.graph_dir)

    graphs = load_graphs(cfg.graph_dir)
    tract_to_index, index_to_tract, node_positions = build_canonical_index(graphs)
    num_hotspots = len(tract_to_index)

    adjacencies = {}
    edge_lookups = {}
    for graph_type in GRAPH_TYPES:
        adjacencies[graph_type] = build_adjacency(graphs[graph_type], tract_to_index, graph_type)
        edge_lookups[graph_type] = build_edge_lookup(adjacencies[graph_type])

    # coverage is computed for all three graphs regardless of the toggle
    stats = {g: compute_graph_stats(g, graphs[g], adjacencies[g], cfg.MAX_DELIVERY_TIME)
             for g in GRAPH_TYPES}
    write_json(cfg.shared_dir / "graph_stats.json", stats)

    order_pool, pool_hash, feasible_pairs = build_order_pool(
        adjacencies, cfg, num_hotspots, verbose=not quiet)
    time_limits, ref_times, limits_hash = compute_time_limits(
        order_pool, adjacencies[cfg.TIME_LIMIT_REF_GRAPH], cfg, pool_hash, verbose=not quiet)

    return {
        "graphs": graphs,
        "adjacencies": adjacencies,
        "edge_lookups": edge_lookups,
        "stats": stats,
        "tract_to_index": tract_to_index,
        "index_to_tract": index_to_tract,
        "node_positions": node_positions,
        "num_hotspots": num_hotspots,
        "order_pool": order_pool,
        "pool_hash": pool_hash,
        "feasible_pairs": feasible_pairs,
        "time_limits": time_limits,
        "ref_travel_times": ref_times,
        "limits_hash": limits_hash,
    }


def refresh_graph_stats(cfg):
    """graph_stats.json without running any simulation (used by --aggregate)."""
    graphs = load_graphs(cfg.graph_dir)
    tract_to_index, _i2t, _pos = build_canonical_index(graphs)
    stats = {g: compute_graph_stats(g, graphs[g],
                                    build_adjacency(graphs[g], tract_to_index, g),
                                    cfg.MAX_DELIVERY_TIME)
             for g in GRAPH_TYPES}
    write_json(cfg.shared_dir / "graph_stats.json", stats)
    return stats


def write_cross_city_summary(city_configs):
    """Roll every city's comparison_table.csv into one file at the Results root."""
    rows = []
    fieldnames = None
    for stem, cfg in city_configs:
        table = cfg.comparison_dir / "comparison_table.csv"
        if not table.is_file():
            continue
        with open(str(table), "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["city_folder"] = stem
                rows.append(row)
                if fieldnames is None:
                    fieldnames = ["city_folder"] + [f for f in reader.fieldnames]
    if not rows:
        return None
    path = RESULTS_ROOT / "cross_city_comparison.csv"
    write_csv(path, fieldnames, rows)
    print("\nwrote %s  (%d rows across %d cities)"
          % (path, len(rows), len(set(r["city_folder"] for r in rows))))
    return path


def run_one_city(cfg, args):
    """Everything for a single city: prepare, sweep the graphs, aggregate."""
    if args.aggregate:
        if not (cfg.shared_dir / "graph_stats.json").is_file() and cfg.graph_dir.is_dir():
            refresh_graph_stats(cfg)
        return aggregate(cfg, quiet=args.quiet)

    shared = prepare_shared(cfg, quiet=args.quiet)
    print_banner(cfg, shared["graphs"], shared["stats"], shared["pool_hash"],
                 len(shared["order_pool"]), shared["feasible_pairs"], shared["num_hotspots"])

    for graph_type in cfg.graph_types_to_run:
        for drop_rate in cfg.DROP_RATES:
            print("\n%s\n%s  drop_rate=%.2f\n%s"
                  % (rule("-"), graph_type, drop_rate, rule("-")))
            run_experiment(graph_type, drop_rate, cfg, shared, quiet=args.quiet)

    if cfg.GRAPH_TYPE == "ALL":
        return aggregate(cfg, quiet=args.quiet)

    print("\nRun --graph-type ALL (or the other two graphs) then --aggregate "
          "to build the comparison table.")
    return None


def main(argv=None):
    cfg, args = resolve_config(argv)

    if args.list_cities:
        stems = discover_city_stems()
        print("City folder stems under %s that carry all three graphs:" % DATA_ROOT)
        print()
        usable = []
        for stem in stems:
            city_name, mini, geodesic = parse_city_stem(stem)
            nodes, problems = city_health(stem)
            status = "USABLE" if not problems else "UNUSABLE"
            if not problems:
                usable.append(stem)
            print("  [%-8s] %-26s %5d nodes" % (status, stem, nodes))
            print("             --city-name %-18s --mini %-5s --with-geodesic-distance %s"
                  % (repr(city_name), str(mini).lower(), str(geodesic).lower()))
            for problem in problems:
                print("             ! %s" % problem)
        if not stems:
            print("  (none found)")
        print()
        print("Sweep them all with:  --cities %s" % ",".join(usable) if usable
              else "No usable cities found.")
        return 0

    city_configs = resolve_cities(cfg, args.cities)

    if len(city_configs) > 1:
        print(rule())
        print("CITY SWEEP: %d cities x %s graph(s) x %d drop rate(s)"
              % (len(city_configs), len(cfg.graph_types_to_run), len(cfg.DROP_RATES)))
        for stem, _c in city_configs:
            print("  %s" % stem)
        print(rule())

    for stem, city_cfg in city_configs:
        if len(city_configs) > 1:
            print("\n\n" + rule("#"))
            print("### CITY: %s" % stem)
            print(rule("#"))
        try:
            run_one_city(city_cfg, args)
        except Exception as exc:                       # keep the sweep going
            if len(city_configs) == 1:
                raise
            print("\n  !! %s FAILED: %s: %s" % (stem, type(exc).__name__, exc))

    if len(city_configs) > 1:
        write_cross_city_summary(city_configs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
