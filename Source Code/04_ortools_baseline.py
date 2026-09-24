#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_ortools_baseline.py
======================

An OR-Tools PDPTW arm for the UMST / MST / CLIQUE comparison.

    UMST    union of MSTs (our contribution)      umst_graph.graphml
    MST     a single minimum spanning tree        mst_graph.graphml
    CLIQUE  the dense reference graph             gh_hotspot_graph.graphml

03_graph_baseline_comparison.py runs two arms -- a myopic co-direction merge
heuristic (`bundling`) and one-vehicle-per-package (`baseline`).  Neither says
whether the heuristic is any *good*.  This script adds a third arm, `ortools`:
given the identical frozen order pool and the identical frozen deadlines, what
does a classical vehicle-routing solver achieve on each network?

The arc cost matrix is the all-pairs shortest path on *the graph being
evaluated*, so a sparse network forces detours and inflates the solver's own
optimum.  That is the entire mechanism by which this compares graphs.

This file never modifies 03; it imports it and reuses its graph loading, its
canonical node index, its Dijkstra, and -- critically -- its hash-keyed order
pool and deadline caches, which is what keeps the arms comparable.

Usage
-----
    python 04_ortools_baseline.py --calibrate
    python 04_ortools_baseline.py --cities Columbus_mini --graph-type ALL
    python 04_ortools_baseline.py --cities ALL --graph-type ALL
    python 04_ortools_baseline.py --aggregate --cities ALL

Model
-----
Classical pickup-and-delivery VRP with time windows, decomposed twice:

  1. DISPATCH EPOCHS.  The arrival window is cut into EPOCH_SECONDS batches and
     each is solved at its close.  A monolithic solve would be clairvoyant --
     it would know at t=0 every order arriving all hour -- while the heuristic
     it is compared against holds each order at most WAIT_TIME.  Setting
     EPOCH_SECONDS == WAIT_TIME gives both arms identical lookahead, so the
     remaining difference is decision quality rather than information.
     The batch wait is charged to the order: delivery time is measured from the
     order's own arrival, not from the epoch close.

  2. SPATIAL CLUSTERS.  Each epoch is clustered by hotspot lat/lon to roughly
     TARGET_CLUSTER_SIZE orders per subproblem.  This is not only a speed hack:
     measured at n=2000 on a 240s budget, 8 clusters beat one monolithic solve
     by 7.8%, because guided local search gets traction at ~300 orders and
     barely improves the first solution at ~3700.  It is also physically sound
     -- two orders 40 km apart can never share a vehicle without blowing both
     deadlines.  Clustering is on lat/lon, which is graph-independent, and the
     assignment is computed once per (city, epoch) and reused by all three
     graphs; clustering on graph distance would hand each graph a different
     decomposition and destroy the comparison.

Known model differences from the relay simulation in 03
-------------------------------------------------------
* No relay handoffs.  A package stays on one vehicle, so num_vehicle_changes
  is 0 where the relay arms charge VEHICLE_CHANGE_TIME per hop.  That is a
  genuine structural advantage of tours over relays.  --stop-service-time
  charges per-stop dwell if you want to neutralise it.
* avg_hops counts STOPS, not graph edges.  One leg may span many edges.
* Vehicles deadhead: a tour may drive empty between a dropoff and the next
  pickup.  The relay sim's vehicles never move empty.  Reported as deadhead_km.
* Batch-optimal at an EPOCH_SECONDS dispatch window, not globally optimal.
* Decomposition inflates distinct_vehicles, since clusters do not share
  vehicles.  Read vehicle_distance_km and vehicle_trips as the primary
  outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ============================================================================
# 0. LOAD 03 AS A MODULE
# ============================================================================
# The leading digit makes "03_graph_baseline_comparison" an invalid module
# name, so a normal import is impossible.  Its body is definitions only and
# main() is __name__-guarded, so exec_module is safe.  Registering it in
# sys.modules is required for ProcessPoolExecutor to unpickle anything that
# refers to it.

_HERE = Path(__file__).resolve().parent
_GBC_PATH = _HERE / "03_graph_baseline_comparison.py"

if not _GBC_PATH.is_file():                                  # pragma: no cover
    raise SystemExit("Cannot find %s -- 04 must live beside 03." % _GBC_PATH)

_spec = importlib.util.spec_from_file_location("gbc", _GBC_PATH)
gbc = importlib.util.module_from_spec(_spec)
sys.modules["gbc"] = gbc
_spec.loader.exec_module(gbc)

GRAPH_TYPES = gbc.GRAPH_TYPES
MODE = "ortools"


# ============================================================================
# 1. CONFIG
# ============================================================================
# Every constant here is also a CLI flag: lowercase it and turn underscores
# into dashes (EPOCH_SECONDS -> --epoch-seconds).  03's own constants are
# inherited unchanged and get their flags too.

# ---------- decomposition ----------
EPOCH_SECONDS          = 120        # dispatch epoch; keep equal to WAIT_TIME
TARGET_CLUSTER_SIZE    = 300        # orders per subproblem
MAX_SUBPROBLEM_ORDERS  = 400        # hard cap; clusters are split until under it.
                                    # Lower than it looks: one vehicle per order means
                                    # a 475-order cluster is a 475-vehicle model, which
                                    # can stall the first-solution heuristic.

# ---------- solver ----------
SOLVER_TIME_LIMIT      = 15         # seconds per subproblem
FIRST_SOLUTION         = "PARALLEL_CHEAPEST_INSERTION"
METAHEURISTIC          = "GUIDED_LOCAL_SEARCH"

# ---------- fleet ----------
VEHICLE_CAPACITY       = 5          # = MAX_BUNDLE_SIZE; orders onboard at once
VEHICLE_FIXED_COST     = 100        # metres; tiebreak against idle extra routes
LATENESS_PENALTY       = 17         # cost per second late, in the objective's
                                    # metre units: 17 m/s ~= "one minute late is
                                    # worth one kilometre of driving". At 1000 a
                                    # single second outweighs a kilometre and the
                                    # solver stops optimising distance at all.
STOP_SERVICE_TIME      = 0          # seconds charged per stop; 0 = tours are free

# ---------- delivery radius policy ----------
MDT_MODE               = "coverage"  # "coverage" (per city) | "fixed" (uniform)
COVERAGE_TARGET        = 0.60        # coverage mode: target on the anchor graph
COVERAGE_ANCHOR        = "UMST"      # coverage mode: graph to anchor on
                                     # fixed mode reuses 03's MAX_DELIVERY_TIME

# ---------- execution ----------
N_WORKERS              = 0          # 0 -> cpu_count() - 2
SAVE_ROUTES            = True       # write routes.json (large on big cities)
MAKE_PLOTS             = True


def _as_mdt_mode(value):
    text = str(value).strip().lower()
    if text not in ("coverage", "fixed"):
        raise argparse.ArgumentTypeError("expected 'coverage' or 'fixed'")
    return text


def _as_first_solution(value):
    text = str(value).strip().upper()
    if not hasattr(routing_enums_pb2.FirstSolutionStrategy, text):
        raise argparse.ArgumentTypeError("unknown first solution strategy %r" % (value,))
    return text


def _as_metaheuristic(value):
    text = str(value).strip().upper()
    if not hasattr(routing_enums_pb2.LocalSearchMetaheuristic, text):
        raise argparse.ArgumentTypeError("unknown metaheuristic %r" % (value,))
    return text


# (constant, converter, help).  Appended to 03's spec, so 04 accepts every
# flag 03 does plus these.  This is also the banner print order.
ORTOOLS_SPEC = [
    ("EPOCH_SECONDS",         int,                 "dispatch epoch in seconds; keep equal to WAIT_TIME"),
    ("TARGET_CLUSTER_SIZE",   int,                 "target orders per spatial subproblem"),
    ("MAX_SUBPROBLEM_ORDERS", int,                 "hard cap on orders in one subproblem"),
    ("SOLVER_TIME_LIMIT",     int,                 "seconds of solver time per subproblem"),
    ("FIRST_SOLUTION",        _as_first_solution,  "OR-Tools FirstSolutionStrategy name"),
    ("METAHEURISTIC",         _as_metaheuristic,   "OR-Tools LocalSearchMetaheuristic name"),
    ("VEHICLE_CAPACITY",      int,                 "orders a vehicle may carry at once"),
    ("VEHICLE_FIXED_COST",    int,                 "metres charged per vehicle used"),
    ("LATENESS_PENALTY",      int,                 "cost per second of deadline violation"),
    ("STOP_SERVICE_TIME",     int,                 "seconds charged per stop"),
    ("MDT_MODE",              _as_mdt_mode,        "coverage (per city) | fixed (uniform)"),
    ("COVERAGE_TARGET",       float,               "coverage mode: target OD coverage, 0-1"),
    ("COVERAGE_ANCHOR",       gbc._as_graph_choice, "coverage mode: graph to calibrate on"),
    ("N_WORKERS",             int,                 "parallel solver processes; 0 = cpu_count-2"),
    ("SAVE_ROUTES",           gbc._as_bool,        "write routes.json"),
    ("MAKE_PLOTS",            gbc._as_bool,        "draw comparison plots during aggregation"),
]

# 04 may redeclare one of 03's constants (MAKE_PLOTS does, since the two scripts
# draw different plots).  Dedupe by name with 04's entry winning, or argparse sees
# the same flag twice and refuses to build the parser at all.
_OVERRIDDEN = set(name for name, _c, _h in ORTOOLS_SPEC)
FULL_SPEC = [e for e in gbc.CONFIG_SPEC if e[0] not in _OVERRIDDEN] + ORTOOLS_SPEC


def _default_for(name):
    """Defaults come from this module when defined here, else from 03."""
    if name in globals():
        return globals()[name]
    return getattr(gbc, name)


class Config(gbc.Config):
    """03's Config widened to cover the OR-Tools constants.

    The inherited derived paths (city_folder, city_key, shared_dir, ...) are
    reused verbatim -- in particular shared_dir still points at
    Results/<CITY_KEY>/_shared, which is the pool and deadline cache that 03
    and 04 deliberately share.
    """

    def __init__(self, values):
        for name, _conv, _help in FULL_SPEC:
            setattr(self, name, values[name])

    def as_dict(self):
        def clean(value):
            if isinstance(value, float) and math.isinf(value):
                return "inf"
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            return value
        return {name: clean(getattr(self, name)) for name, _c, _h in FULL_SPEC}

    def clone_for_city(self, city_name, mini, geodesic):
        values = {name: getattr(self, name) for name, _c, _h in FULL_SPEC}
        values["CITY_NAME"] = city_name
        values["MINI"] = mini
        values["WITH_GEODESIC_DISTANCE"] = geodesic
        return Config(values)

    # -- OR-Tools output tree (kept out of 03's Results/<CITY_KEY>/) --------
    @property
    def ortools_root(self):
        return gbc.RESULTS_ROOT / "OR tools"

    @property
    def or_city_dir(self):
        return self.ortools_root / self.city_key

    @property
    def or_shared_dir(self):
        return self.or_city_dir / "_shared"

    @property
    def or_comparison_dir(self):
        return self.or_city_dir / "comparison"

    def or_graph_dir(self, graph_type):
        return self.or_city_dir / graph_type

    @property
    def n_workers(self):
        if self.N_WORKERS > 0:
            return self.N_WORKERS
        return max(1, (os.cpu_count() or 2) - 2)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="OR-Tools PDPTW baseline over UMST / MST / CLIQUE",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="resolve MAX_DELIVERY_TIME for every city under the active --mdt-mode, "
             "write Results/'OR tools'/calibrated_mdt.json, print the matching 03 "
             "commands, then exit",
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="do not solve; read the existing summary.json files and write the "
             "comparison table, the heuristic join and the plots",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress per-subproblem progress lines",
    )
    parser.add_argument(
        "--cities", type=gbc._as_city_list, default=None,
        help="ALL, or a comma-separated list of city folder stems. Overrides "
             "--city-name/--mini/--with-geodesic-distance.",
    )
    parser.add_argument(
        "--list-cities", action="store_true",
        help="print the city folder stems that carry all three graphs, then exit",
    )
    for name, converter, help_text in FULL_SPEC:
        flag = "--" + name.lower().replace("_", "-")
        parser.add_argument(flag, dest=name, type=converter, default=None, help=help_text)
    return parser


def resolve_config(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    values = {}
    for name, converter, _help in FULL_SPEC:
        override = getattr(args, name)
        values[name] = converter(_default_for(name)) if override is None else override
    return Config(values), args


# ============================================================================
# 2. DELIVERY RADIUS: THE ONE PLACE MAX_DELIVERY_TIME IS DECIDED
# ============================================================================

def resolve_mdt(cfg, adjacencies, verbose=True):
    """Resolve MAX_DELIVERY_TIME for this city under the active policy.

    The result feeds order_pool_hash, so switching --mdt-mode automatically
    produces a different pool hash, a fresh cached pool and fresh deadlines.
    Nothing is invalidated and nothing is deleted: results from both policies
    coexist on disk under their own hashes, and 03's assertions 3 and 4 refuse
    to mix them.

    'fixed'    -- whatever --max-delivery-time says, identical in every city.
    'coverage' -- the COVERAGE_TARGET quantile of the anchor graph's all-pairs
                  travel time distribution, so demand is structurally identical
                  across cities rather than parameter-identical: "orders are
                  drawn from the nearest 60% of OD pairs reachable on UMST".
                  A single fixed threshold cannot do this -- it saturates on
                  the mini cities (everything reachable) and collapses on the
                  full-size ones (almost nothing reachable).
    """
    if cfg.MDT_MODE == "fixed":
        return float(cfg.MAX_DELIVERY_TIME)

    anchor = adjacencies[cfg.COVERAGE_ANCHOR]
    times = []
    for source in sorted(anchor):
        dist, _prev, _hops = gbc.dijkstra_all(anchor, source)
        times.extend(value for target, value in dist.items() if target != source)
    if not times:                                            # pragma: no cover
        raise RuntimeError("anchor graph %s has no connected pairs" % cfg.COVERAGE_ANCHOR)

    # rounded to whole seconds so the pool hash is stable across runs
    mdt = float(round(float(np.percentile(np.array(times), 100.0 * cfg.COVERAGE_TARGET))))
    if verbose:
        print("  delivery radius: %.0fs (%.1f min) -- %s coverage %.0f%%"
              % (mdt, mdt / 60.0, cfg.COVERAGE_ANCHOR, 100.0 * cfg.COVERAGE_TARGET))
    return mdt


def calibrate_all(cfg, stems, quiet=False):
    """Resolve the radius for every city and print the matching 03 commands.

    Printing the commands is not a convenience: 03 and 04 must run at the same
    MAX_DELIVERY_TIME or their pool hashes diverge and the arms stop being
    comparable (assertion 11).
    """
    print(gbc.rule())
    print("DELIVERY RADIUS CALIBRATION   mode=%s" % cfg.MDT_MODE)
    if cfg.MDT_MODE == "coverage":
        print("  target: %s covers %.0f%% of OD pairs in every city"
              % (cfg.COVERAGE_ANCHOR, 100.0 * cfg.COVERAGE_TARGET))
    else:
        print("  uniform %.0fs (%.1f min) in every city"
              % (cfg.MAX_DELIVERY_TIME, cfg.MAX_DELIVERY_TIME / 60.0))
    print(gbc.rule())
    print("  %-20s %8s %9s %9s %9s %9s" % ("city", "mdt_s", "mdt_min", "UMST", "MST", "CLIQUE"))
    print("  " + "-" * 70)

    entries = {}
    warnings = []
    for stem in stems:
        city_cfg = cfg.clone_for_city(*gbc.parse_city_stem(stem))
        graphs = gbc.load_graphs(city_cfg.graph_dir)
        tract_to_index, _i2t, _pos = gbc.build_canonical_index(graphs)
        adjacencies = {g: gbc.build_adjacency(graphs[g], tract_to_index, g)
                       for g in GRAPH_TYPES}
        mdt = resolve_mdt(city_cfg, adjacencies, verbose=False)
        coverage = {
            g: gbc.compute_graph_stats(g, graphs[g], adjacencies[g],
                                       mdt)["coverage_within_max_delivery_time"]
            for g in GRAPH_TYPES
        }
        entries[stem] = {
            "city_key": city_cfg.city_key,
            "max_delivery_time": mdt,
            "coverage": coverage,
            "nodes": len(tract_to_index),
        }
        print("  %-20s %8.0f %9.1f %8.1f%% %8.1f%% %8.1f%%"
              % (stem, mdt, mdt / 60.0, 100.0 * coverage["UMST"],
                 100.0 * coverage["MST"], 100.0 * coverage["CLIQUE"]))
        if coverage[cfg.ORDER_POOL_GRAPH] < 0.20:
            warnings.append((stem, coverage["UMST"]))

    if warnings:
        print()
        print("WARNING")
        print("  These cities cover under 20%% of OD pairs at this radius, so their order")
        print("  pools collapse to short local trips. That is a legitimate choice, but the")
        print("  coverage statistic stops discriminating between graphs there:")
        for stem, cov in warnings:
            print("    %-20s UMST coverage %.1f%%" % (stem, 100.0 * cov))

    print()
    print("Run 03 at these thresholds so both arms share an order pool:")
    print()
    for stem in stems:
        print("  python 03_graph_baseline_comparison.py --cities %s --graph-type ALL \\"
              % stem)
        print("      --load-per-hotspot %d --n-runs 1 --max-delivery-time %.0f"
              % (cfg.LOAD_PER_HOTSPOT, entries[stem]["max_delivery_time"]))
    print()

    payload = {
        "mdt_mode": cfg.MDT_MODE,
        "coverage_target": cfg.COVERAGE_TARGET,
        "coverage_anchor": cfg.COVERAGE_ANCHOR,
        "fixed_max_delivery_time": cfg.MAX_DELIVERY_TIME,
        "load_per_hotspot": cfg.LOAD_PER_HOTSPOT,
        "cities": entries,
    }
    path = cfg.ortools_root / "calibrated_mdt.json"
    gbc.write_json(path, payload)
    print("wrote %s" % path)
    return payload


# ============================================================================
# 3. COST MATRICES
# ============================================================================

def build_matrices(cfg, graph_type, adjacency, num_nodes, verbose=True):
    """All-pairs hotspot-level distance (km) and time (s) on ONE graph.

    These are hotspot-sized (863x863 at worst, 6 MB), never task-sized: a
    subproblem slices them with np.ix_.  Slicing from the *evaluated* graph is
    the mechanism that makes this a comparison of networks -- a sparse graph
    forces detours and inflates the solver's own optimum.

    Unreachable pairs cannot occur (all three graphs are connected, checked by
    03's assertion 5) but are filled with a large sentinel rather than left at
    zero, so a connectivity bug shows up as an absurd objective instead of as
    free teleportation.
    """
    cache_km = cfg.or_shared_dir / ("matrix_%s_dist.npy" % graph_type)
    cache_s = cfg.or_shared_dir / ("matrix_%s_time.npy" % graph_type)
    if cache_km.is_file() and cache_s.is_file():
        dist_km = np.load(str(cache_km))
        time_s = np.load(str(cache_s))
        if dist_km.shape == (num_nodes, num_nodes):
            if verbose:
                print("  matrices    : loaded %s cache" % graph_type)
            return dist_km, time_s

    started = time.time()
    big_km = 1.0e7
    big_s = 1.0e9
    dist_km = np.full((num_nodes, num_nodes), big_km, dtype=np.float64)
    time_s = np.full((num_nodes, num_nodes), big_s, dtype=np.float64)

    # Dijkstra on time gives the routing policy; the distance of that same
    # minimum-time path is what the vehicle actually drives, so both must come
    # from one traversal rather than from two independent shortest-path runs.
    for source in range(num_nodes):
        dist, prev, _hops = gbc.dijkstra_all(adjacency, source)
        km_to = {source: 0.0}
        order = sorted(dist, key=lambda node: dist[node])
        for node in order:
            if node == source:
                continue
            parent = prev[node]
            edge_km = None
            for neighbour, neighbour_km, _neighbour_s in adjacency[parent]:
                if neighbour == node:
                    edge_km = neighbour_km
                    break
            km_to[node] = km_to.get(parent, 0.0) + (edge_km or 0.0)
        for node, seconds in dist.items():
            time_s[source][node] = seconds
            dist_km[source][node] = km_to.get(node, big_km)

    np.fill_diagonal(dist_km, 0.0)
    np.fill_diagonal(time_s, 0.0)

    gbc.write_json(cfg.or_shared_dir / ".keep.json", {"purpose": "matrix cache"})
    np.save(str(cache_km), dist_km)
    np.save(str(cache_s), time_s)
    if verbose:
        print("  matrices    : built %s in %.1fs (%.1f MB each)"
              % (graph_type, time.time() - started, dist_km.nbytes / 1e6))
    return dist_km, time_s


# ============================================================================
# 4. EPOCHS AND SPATIAL CLUSTERS
# ============================================================================

def build_epochs(order_pool, epoch_seconds):
    """Partition the pool by arrival time.  epochs[k] = [order, ...].

    Every order lands in exactly one epoch (assertion 5); the caller checks it.
    """
    epochs = defaultdict(list)
    for order in order_pool:
        epochs[int(order.start_time) // int(epoch_seconds)].append(order)
    for key in epochs:
        epochs[key].sort(key=lambda o: (o.start_time, o.id))
    return dict(epochs)


def _kmeans(points, k, seed, max_iters=30):
    """Deterministic k-means++ on lat/lon.  Returns integer labels.

    Determinism matters twice over: the assignment is cached and reused by all
    three graphs, and it must be reproducible across runs so a rerun does not
    silently change the decomposition.
    """
    n = len(points)
    if k <= 1:
        return np.zeros(n, dtype=int)
    if n <= k:
        return np.arange(n, dtype=int)

    rng = np.random.RandomState(seed)
    first = int(rng.randint(n))
    centers = [points[first]]
    closest = ((points - points[first]) ** 2).sum(1)
    for _ in range(1, k):
        total = closest.sum()
        probs = (closest / total) if total > 0 else np.full(n, 1.0 / n)
        pick = int(rng.choice(n, p=probs))
        centers.append(points[pick])
        closest = np.minimum(closest, ((points - points[pick]) ** 2).sum(1))

    centroids = np.array(centers, dtype=float)
    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iters):
        d2 = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(2)
        new_labels = d2.argmin(1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            member = labels == j
            if member.any():
                centroids[j] = points[member].mean(0)
    return labels


def cluster_epoch(orders, node_positions, cfg, seed):
    """Split one epoch into spatially coherent subproblems.

    k is chosen to hit TARGET_CLUSTER_SIZE, then raised until no cluster
    exceeds MAX_SUBPROBLEM_ORDERS -- k-means gives no size guarantee, and one
    oversized cluster would blow both the memory budget and the solve time.
    """
    n = len(orders)
    if n <= cfg.MAX_SUBPROBLEM_ORDERS:
        return [list(range(n))]

    points = np.array([node_positions[o.start_node] for o in orders], dtype=float)
    k = max(1, int(math.ceil(n / float(cfg.TARGET_CLUSTER_SIZE))))

    for _attempt in range(12):
        labels = _kmeans(points, k, seed)
        groups = defaultdict(list)
        for position, label in enumerate(labels):
            groups[int(label)].append(position)
        if max(len(g) for g in groups.values()) <= cfg.MAX_SUBPROBLEM_ORDERS:
            return [groups[key] for key in sorted(groups)]
        k = int(math.ceil(k * 1.5))

    # Fall back to slicing the largest clusters; never return an oversized one.
    out = []
    for key in sorted(groups):
        members = groups[key]
        for start in range(0, len(members), cfg.MAX_SUBPROBLEM_ORDERS):
            out.append(members[start:start + cfg.MAX_SUBPROBLEM_ORDERS])
    return out


def build_clusters(epochs, node_positions, cfg, verbose=True):
    """Cluster assignment for every epoch, cached and shared by all 3 graphs.

    Cached because it MUST be identical across UMST, MST and CLIQUE
    (assertion 6): clustering on graph distance, or re-clustering per graph,
    would hand each graph a different decomposition and destroy the comparison.
    lat/lon is graph-independent, which is why it is the clustering key.
    """
    cache = cfg.or_shared_dir / "clusters.json"
    signature = {
        "epoch_seconds": cfg.EPOCH_SECONDS,
        "target_cluster_size": cfg.TARGET_CLUSTER_SIZE,
        "max_subproblem_orders": cfg.MAX_SUBPROBLEM_ORDERS,
        "global_seed": cfg.GLOBAL_SEED,
        "n_orders": sum(len(v) for v in epochs.values()),
    }
    if cache.is_file():
        payload = gbc.read_json(cache)
        if payload.get("signature") == signature:
            if verbose:
                print("  clusters    : loaded from cache")
            return {int(k): [list(map(int, c)) for c in v]
                    for k, v in payload["clusters"].items()}

    clusters = {}
    for epoch_idx in sorted(epochs):
        clusters[epoch_idx] = cluster_epoch(
            epochs[epoch_idx], node_positions, cfg, cfg.GLOBAL_SEED + epoch_idx)

    gbc.write_json(cache, {"signature": signature,
                           "clusters": {str(k): v for k, v in clusters.items()}})
    if verbose:
        sizes = [len(c) for v in clusters.values() for c in v]
        print("  clusters    : %d subproblems, sizes min %d / mean %.0f / max %d"
              % (len(sizes), min(sizes), float(np.mean(sizes)), max(sizes)))
    return clusters


# ============================================================================
# 5. THE PDPTW SUBPROBLEM
# ============================================================================
# Solved in worker processes.  _WORKER holds the two matrices and the solver
# parameters so they are pickled once per worker rather than once per task.

_WORKER = {}


def _init_worker(dist_km, time_s, params):
    _WORKER["dist_km"] = dist_km
    _WORKER["time_s"] = time_s
    _WORKER["params"] = params


def solve_subproblem(task):
    """One (epoch, cluster) PDPTW.  Returns routes plus per-order outcomes.

    Node layout: 0 is a virtual depot with zero-cost arcs both ways, so
    vehicles start and end wherever is cheapest -- repositioning is free,
    matching 03's Vehicle.start_new_trip which sets current_node on acquisition.
    Order i occupies pickup 2i+1 and dropoff 2i+2.
    """
    dist_km = _WORKER["dist_km"]
    time_s = _WORKER["time_s"]
    params = _WORKER["params"]

    epoch_idx = task["epoch"]
    cluster_idx = task["cluster"]
    epoch_close = task["epoch_close"]
    rows = task["orders"]                 # (id, start_node, end_node, start_time, deadline)
    n = len(rows)
    started = time.time()

    if n == 0:                                               # pragma: no cover
        return {"epoch": epoch_idx, "cluster": cluster_idx, "n": 0, "status": "EMPTY",
                "objective": 0, "seconds": 0.0, "routes": [], "orders": []}

    n_nodes = 2 * n + 1
    loc = np.zeros(n_nodes, dtype=np.int64)
    for i, row in enumerate(rows):
        loc[2 * i + 1] = row[1]
        loc[2 * i + 2] = row[2]

    # Depot arcs are free in both dimensions: a vehicle appears at its first
    # pickup and vanishes after its last dropoff at no cost.  That mirrors 03,
    # whose Vehicle.start_new_trip repositions for free.
    #
    # The zeroing happens HERE, on sub_km itself, before the solver matrices are
    # derived from it -- not afterwards on the derived copies.  _walk_solution
    # accumulates reported kilometres from sub_km, so zeroing only the derived
    # matrices would charge every route a phantom leg from hotspot 0 to its
    # first pickup while the solver saw that leg as free.
    sub_km = dist_km[np.ix_(loc, loc)].copy()
    sub_s = time_s[np.ix_(loc, loc)].copy()
    sub_km[0, :] = 0.0
    sub_km[:, 0] = 0.0
    sub_s[0, :] = 0.0
    sub_s[:, 0] = 0.0

    cost_m = np.rint(sub_km * 1000.0).astype(np.int64)
    travel_s = np.rint(sub_s).astype(np.int64)
    if params["stop_service_time"]:
        travel_s = travel_s + int(params["stop_service_time"])
        travel_s[0, :] = 0
        travel_s[:, 0] = 0

    # One vehicle per order.  Fewer would FORCE chaining, and a chained vehicle
    # deadheads from a dropoff to the next pickup -- distance that 03's baseline
    # arm never pays, since it conjures a fresh vehicle at every origin.  With n
    # vehicles the solver can always fall back to one-per-order, which makes
    # vehicle_distance_km <= solo_distance_km an invariant rather than a hope.
    n_vehicles = n
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # RegisterTransitMatrix keeps every arc evaluation in C++.  With a Python
    # callback the insertion heuristics make millions of interpreter round
    # trips and fail to build any first solution past ~500 orders.
    cost_idx = routing.RegisterTransitMatrix(cost_m.tolist())
    routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)
    if params["vehicle_fixed_cost"]:
        routing.SetFixedCostOfAllVehicles(int(params["vehicle_fixed_cost"]))

    time_idx = routing.RegisterTransitMatrix(travel_s.tolist())
    horizon = int(epoch_close) + 24 * 3600
    routing.AddDimension(time_idx, 0, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for vehicle in range(n_vehicles):
        time_dim.CumulVar(routing.Start(vehicle)).SetValue(int(epoch_close))

    demand = [0] * n_nodes
    for i in range(n):
        demand[2 * i + 1] = 1
        demand[2 * i + 2] = -1
    demand_idx = routing.RegisterUnaryTransitVector(demand)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [int(params["capacity"])] * n_vehicles, True, "Load")

    for i, row in enumerate(rows):
        pickup = manager.NodeToIndex(2 * i + 1)
        dropoff = manager.NodeToIndex(2 * i + 2)
        routing.AddPickupAndDelivery(pickup, dropoff)
        routing.solver().Add(routing.VehicleVar(pickup) == routing.VehicleVar(dropoff))
        routing.solver().Add(time_dim.CumulVar(pickup) <= time_dim.CumulVar(dropoff))
        # SOFT, not hard.  03's simulation never refuses an order -- it delivers
        # late and records within_buffer=False.  A hard window would make the
        # solver drop orders and break the shared denominator that 03's
        # assertions 3-5 protect.
        deadline = max(0, int(row[4]))
        time_dim.SetCumulVarSoftUpperBound(dropoff, deadline, int(params["lateness_penalty"]))

    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, params["first_solution"])
    search.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, params["metaheuristic"])
    search.time_limit.FromSeconds(int(params["time_limit"]))

    solution = routing.SolveWithParameters(search)
    seconds = time.time() - started

    if solution is None:
        return {"epoch": epoch_idx, "cluster": cluster_idx, "n": n, "status": "NO_SOLUTION",
                "objective": -1, "seconds": seconds, "routes": [], "orders": []}

    routes, order_rows = _walk_solution(
        routing, manager, solution, time_dim, rows, sub_km, n)

    return {"epoch": epoch_idx, "cluster": cluster_idx, "n": n, "status": "OK",
            "objective": int(solution.ObjectiveValue()), "seconds": seconds,
            "routes": routes, "orders": order_rows}


def _walk_solution(routing, manager, solution, time_dim, rows, sub_km, n):
    """Turn a solution into vehicle routes and per-order outcomes.

    Distance is accumulated from sub_km (float km) rather than from the integer
    metre objective, so the reported kilometres are not the rounding the solver
    optimised against.

    An arc is credited to every order currently onboard BEFORE the arriving
    node's own pickup/dropoff is applied -- an order pays for the leg it rides,
    not for the leg that fetched it.
    """
    routes = []
    order_rows = []

    for vehicle in range(routing.vehicles()):
        index = routing.Start(vehicle)
        if not routing.IsVehicleUsed(solution, vehicle):
            continue

        previous_node = manager.IndexToNode(index)
        route_km = 0.0
        loaded_km = 0.0
        trips = 0
        onboard = set()
        peak_load = 0
        carried = []
        per_order_km = {}
        per_order_stops = {}

        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            if routing.IsEnd(index):
                break                       # the closing arc to the depot is free
            node = manager.IndexToNode(index)

            arc_km = float(sub_km[previous_node][node])
            route_km += arc_km
            trips += 1
            if onboard:
                loaded_km += arc_km
                for local in onboard:
                    per_order_km[local] += arc_km
                    per_order_stops[local] += 1

            local = (node - 1) // 2
            if node % 2 == 1:                               # pickup
                onboard.add(local)
                carried.append(local)
                per_order_km[local] = 0.0
                per_order_stops[local] = 0
            else:                                           # dropoff
                onboard.discard(local)
                arrival = solution.Min(time_dim.CumulVar(index))
                row = rows[local]
                order_rows.append({
                    "id": int(row[0]),
                    "start_node": int(row[1]),
                    "end_node": int(row[2]),
                    "start_time": int(row[3]),
                    "time_limit_abs": int(row[4]),
                    "dropoff_clock": int(arrival),
                    "loaded_km": round(per_order_km.get(local, 0.0), 6),
                    "stops": int(per_order_stops.get(local, 0)),
                    "vehicle": int(vehicle),
                })
            peak_load = max(peak_load, len(onboard))
            previous_node = node

        if onboard:                                          # pragma: no cover
            raise AssertionError(
                "ASSERTION 9 FAILED: vehicle %d ended its route still carrying %s"
                % (vehicle, sorted(onboard)))

        routes.append({
            "vehicle": int(vehicle),
            "dist_km": round(route_km, 6),
            "loaded_km": round(loaded_km, 6),
            "trips": int(trips),
            "n_orders": len(carried),
            "peak_load": int(peak_load),
            "orders": [int(rows[local][0]) for local in carried],
        })

    return routes, order_rows


# ============================================================================
# 6. METRICS
# ============================================================================

OR_METRIC_COLUMNS = [
    # volume
    "n_orders", "completed", "completion_rate", "successful", "success_rate",
    "failed", "failed_rate",
    # distance
    "vehicle_distance_km", "package_distance_km", "loaded_distance_km",
    "solo_distance_km", "deadhead_km",
    "distance_saved_km", "distance_saved_pct", "saved_vs_solo_pct",
    # time
    "avg_delivery_time", "median_delivery_time", "p90", "p95",
    "avg_delay", "median_delay", "max_delay", "avg_hold_time",
    # vehicles
    "distinct_vehicles", "vehicle_trips",
    # consolidation
    "orders_ever_bundled", "bundle_participation_rate", "bundles_formed",
    "avg_bundle_size", "max_bundle_size_observed", "avg_hops",
    # solver housekeeping
    "n_subproblems", "total_solve_seconds", "objective_total",
]


def compute_metrics(results, order_pool, time_limits, dist_km, cfg):
    """Fold every subproblem into one metric row, and check the invariants.

    Column names follow 03's METRIC_COLUMNS wherever the quantity means the
    same thing, so the tables join on `id`.  Where tours and relays genuinely
    differ the extra columns are explicit rather than smuggled into an existing
    name: loaded_distance_km, solo_distance_km, deadhead_km, saved_vs_solo_pct.
    """
    routes = [r for res in results for r in res["routes"]]
    rows = [o for res in results for o in res["orders"]]
    n_orders = len(order_pool)

    # ASSERTION 12: a subproblem that found nothing must never pass silently
    bad = [(r["epoch"], r["cluster"], r["status"]) for r in results if r["status"] != "OK"]
    if bad:
        raise AssertionError(
            "ASSERTION 12 FAILED: %d subproblem(s) returned no solution, e.g. %s. "
            "Raise --solver-time-limit or lower --target-cluster-size."
            % (len(bad), bad[:5]))

    # ASSERTION 7: every order routed exactly once
    seen = [row["id"] for row in rows]
    if len(seen) != n_orders or len(set(seen)) != n_orders:
        missing = sorted(set(o.id for o in order_pool) - set(seen))
        raise AssertionError(
            "ASSERTION 7 FAILED: %d order rows for a pool of %d (%d distinct). "
            "First missing: %s"
            % (len(seen), n_orders, len(set(seen)), missing[:10]))

    # ASSERTION 10: capacity respected
    over = [r for r in routes if r["peak_load"] > cfg.VEHICLE_CAPACITY]
    if over:
        raise AssertionError(
            "ASSERTION 10 FAILED: %d route(s) exceeded capacity %d, worst %d"
            % (len(over), cfg.VEHICLE_CAPACITY, max(r["peak_load"] for r in over)))

    vehicle_distance_km = sum(r["dist_km"] for r in routes)
    loaded_distance_km = sum(r["loaded_km"] for r in routes)
    deadhead_km = vehicle_distance_km - loaded_distance_km

    # ASSERTION 8: a vehicle cannot travel less than the loaded part of its own
    # route.  Unlike 03 this is NOT vehicle <= package: a tour deadheads
    # between a dropoff and the next pickup, which relay vehicles never do.
    if deadhead_km < -1e-6 * max(1.0, vehicle_distance_km):
        raise AssertionError(
            "ASSERTION 8 FAILED: loaded distance %.3f exceeds vehicle distance %.3f"
            % (loaded_distance_km, vehicle_distance_km))

    package_distance_km = sum(row["loaded_km"] for row in rows)
    solo_distance_km = sum(float(dist_km[o.start_node][o.end_node]) for o in order_pool)

    epoch = int(cfg.EPOCH_SECONDS)
    actual, delay, hold, hops = [], [], [], []
    successful = 0
    for row in rows:
        limit = float(time_limits[row["id"]])
        taken = float(row["dropoff_clock"] - row["start_time"])
        actual.append(taken)
        delay.append(taken - limit)
        hold.append(float((row["start_time"] // epoch + 1) * epoch - row["start_time"]))
        hops.append(int(row["stops"]))
        if taken <= limit:
            successful += 1

    shared = [r for r in routes if r["n_orders"] >= 2]
    ever_bundled = sum(r["n_orders"] for r in shared)
    peak_loads = [r["peak_load"] for r in routes]

    return {
        "n_orders": n_orders,
        # soft deadlines mean nothing is ever dropped
        "completed": len(rows),
        "completion_rate": len(rows) / n_orders if n_orders else 0.0,
        "successful": successful,
        "success_rate": successful / n_orders if n_orders else 0.0,
        "failed": 0,
        "failed_rate": 0.0,

        "vehicle_distance_km": vehicle_distance_km,
        "package_distance_km": package_distance_km,
        "loaded_distance_km": loaded_distance_km,
        "solo_distance_km": solo_distance_km,
        "deadhead_km": deadhead_km,
        "distance_saved_km": package_distance_km - vehicle_distance_km,
        "distance_saved_pct": (100.0 * (package_distance_km - vehicle_distance_km)
                               / package_distance_km) if package_distance_km > 0 else 0.0,
        "saved_vs_solo_pct": (100.0 * (solo_distance_km - vehicle_distance_km)
                              / solo_distance_km) if solo_distance_km > 0 else 0.0,

        "avg_delivery_time": float(np.mean(actual)) if actual else 0.0,
        "median_delivery_time": float(np.median(actual)) if actual else 0.0,
        "p90": float(np.percentile(actual, 90)) if actual else 0.0,
        "p95": float(np.percentile(actual, 95)) if actual else 0.0,
        "avg_delay": float(np.mean(delay)) if delay else 0.0,
        "median_delay": float(np.median(delay)) if delay else 0.0,
        "max_delay": float(np.max(delay)) if delay else 0.0,
        "avg_hold_time": float(np.mean(hold)) if hold else 0.0,

        "distinct_vehicles": len(routes),
        "vehicle_trips": sum(r["trips"] for r in routes),

        "orders_ever_bundled": ever_bundled,
        "bundle_participation_rate": ever_bundled / n_orders if n_orders else 0.0,
        "bundles_formed": len(shared),
        "avg_bundle_size": float(np.mean(peak_loads)) if peak_loads else 0.0,
        "max_bundle_size_observed": int(np.max(peak_loads)) if peak_loads else 0,
        "avg_hops": float(np.mean(hops)) if hops else 0.0,

        "n_subproblems": len(results),
        "total_solve_seconds": sum(r["seconds"] for r in results),
        "objective_total": sum(r["objective"] for r in results),
    }


PER_ORDER_COLUMNS = [
    "id", "start_node", "end_node", "start_time", "epoch", "dropoff_clock",
    "actual_delivery_time", "time_limit", "delay", "within_buffer",
    "loaded_km", "solo_km", "stops", "vehicle", "hold_time",
]


def per_order_rows(rows, order_pool, time_limits, dist_km, cfg):
    by_id = {o.id: o for o in order_pool}
    epoch = int(cfg.EPOCH_SECONDS)
    out = []
    for row in sorted(rows, key=lambda r: r["id"]):
        order = by_id[row["id"]]
        limit = float(time_limits[row["id"]])
        taken = float(row["dropoff_clock"] - row["start_time"])
        out.append({
            "id": row["id"],
            "start_node": row["start_node"],
            "end_node": row["end_node"],
            "start_time": row["start_time"],
            "epoch": row["start_time"] // epoch,
            "dropoff_clock": row["dropoff_clock"],
            "actual_delivery_time": round(taken, 3),
            "time_limit": round(limit, 3),
            "delay": round(taken - limit, 3),
            "within_buffer": int(taken <= limit),
            "loaded_km": round(row["loaded_km"], 4),
            "solo_km": round(float(dist_km[order.start_node][order.end_node]), 4),
            "stops": row["stops"],
            "vehicle": row["vehicle"],
            "hold_time": int((row["start_time"] // epoch + 1) * epoch - row["start_time"]),
        })
    return out


# ============================================================================
# 7. RUNNER
# ============================================================================

def run_graph(graph_type, cfg, shared, quiet=False):
    """Solve every (epoch, cluster) on one graph and write its folder."""
    adjacency = shared["adjacencies"][graph_type]
    order_pool = shared["order_pool"]
    time_limits = shared["time_limits"]
    epochs = shared["epochs"]
    clusters = shared["clusters"]

    dist_km, time_s = build_matrices(cfg, graph_type, adjacency,
                                     shared["num_hotspots"], verbose=not quiet)

    tasks = []
    for epoch_idx in sorted(epochs):
        orders = epochs[epoch_idx]
        epoch_close = (epoch_idx + 1) * int(cfg.EPOCH_SECONDS)
        for cluster_idx, members in enumerate(clusters[epoch_idx]):
            rows = []
            for position in members:
                order = orders[position]
                rows.append((order.id, order.start_node, order.end_node,
                             order.start_time,
                             int(order.start_time + time_limits[order.id])))
            tasks.append({"epoch": epoch_idx, "cluster": cluster_idx,
                          "epoch_close": epoch_close, "orders": rows})

    params = {
        "capacity": cfg.VEHICLE_CAPACITY,
        "vehicle_fixed_cost": cfg.VEHICLE_FIXED_COST,
        "lateness_penalty": cfg.LATENESS_PENALTY,
        "stop_service_time": cfg.STOP_SERVICE_TIME,
        "first_solution": cfg.FIRST_SOLUTION,
        "metaheuristic": cfg.METAHEURISTIC,
        "time_limit": cfg.SOLVER_TIME_LIMIT,
    }

    workers = min(cfg.n_workers, len(tasks))
    started = time.time()
    if not quiet:
        print("  solving %-6s: %d subproblems on %d worker(s)"
              % (graph_type, len(tasks), max(1, workers)))

    results = []
    if workers <= 1:
        _init_worker(dist_km, time_s, params)
        for done, task in enumerate(tasks, 1):
            results.append(solve_subproblem(task))
            if not quiet:
                _progress(done, len(tasks), started)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(dist_km, time_s, params)) as pool:
            for done, result in enumerate(pool.map(solve_subproblem, tasks), 1):
                results.append(result)
                if not quiet:
                    _progress(done, len(tasks), started)
    if not quiet:
        sys.stdout.write("\n")

    # A first solution is not guaranteed inside SOLVER_TIME_LIMIT: with one
    # vehicle per order a large cluster builds a large model, and the insertion
    # heuristic can exhaust its budget before placing every pickup/dropoff pair.
    # Retry those few with a longer budget rather than weakening the model or
    # letting assertion 12 kill an otherwise good run.
    stragglers = [i for i, r in enumerate(results) if r["status"] != "OK"]
    if stragglers:
        retry_params = dict(params, time_limit=params["time_limit"] * 4)
        if not quiet:
            print("    %d subproblem(s) found no solution in %ds; retrying at %ds"
                  % (len(stragglers), params["time_limit"], retry_params["time_limit"]))
        _init_worker(dist_km, time_s, retry_params)
        for position in stragglers:
            results[position] = solve_subproblem(tasks[position])
        still = [i for i in stragglers if results[i]["status"] != "OK"]
        if not quiet:
            print("    retry recovered %d of %d" % (len(stragglers) - len(still),
                                                    len(stragglers)))

    metrics = compute_metrics(results, order_pool, time_limits, dist_km, cfg)
    order_rows = per_order_rows([o for r in results for o in r["orders"]],
                                order_pool, time_limits, dist_km, cfg)

    out_dir = cfg.or_graph_dir(graph_type)
    os.makedirs(str(out_dir), exist_ok=True)

    gbc.write_csv(out_dir / "runs.csv", ["run_idx", "mode"] + OR_METRIC_COLUMNS,
                  [dict({"run_idx": 0, "mode": MODE}, **metrics)])
    gbc.write_csv(out_dir / "orders.csv", PER_ORDER_COLUMNS, order_rows)
    gbc.write_csv(out_dir / "epochs.csv",
                  ["epoch", "cluster", "n", "status", "objective", "seconds"],
                  [{k: r[k] for k in ("epoch", "cluster", "n", "status",
                                      "objective", "seconds")} for r in results])

    summary = {
        "city_key": cfg.city_key,
        "graph_type": graph_type,
        "mode": MODE,
        "drop_rate": 0.0,
        "n_runs": 1,
        "order_pool_hash": shared["pool_hash"],
        "time_limits_hash": shared["limits_hash"],
        "max_delivery_time": cfg.MAX_DELIVERY_TIME,
        "n_orders": len(order_pool),
        "routing": {"avg_hops": metrics["avg_hops"],
                    "avg_expected_travel_time": metrics["avg_delivery_time"]},
        "graph_stats": shared["stats"][graph_type],
        "config": cfg.as_dict(),
        # one deterministic run, so every std and ci_95 is zero by construction
        "modes": {MODE: {metric: gbc.summarise([metrics[metric]])
                         for metric in OR_METRIC_COLUMNS}},
    }
    gbc.write_json(out_dir / "summary.json", summary)

    if cfg.SAVE_ROUTES:
        gbc.write_json(out_dir / "routes.json",
                       {"city_key": cfg.city_key, "graph_type": graph_type,
                        "epochs": [{"epoch": r["epoch"], "cluster": r["cluster"],
                                    "routes": r["routes"]} for r in results]})

    if not quiet:
        print("    veh_km %10.1f  solo_km %10.1f  saved %5.1f%%  success %6.2f%%  "
              "vehicles %6d  (%.0fs)"
              % (metrics["vehicle_distance_km"], metrics["solo_distance_km"],
                 metrics["saved_vs_solo_pct"], 100.0 * metrics["success_rate"],
                 metrics["distinct_vehicles"], time.time() - started))
        print("    -> %s" % out_dir)
    return summary


def _progress(done, total, started):
    elapsed = time.time() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    sys.stdout.write("\r    %d/%d subproblems  %.0fs elapsed  ~%.0fs left   "
                     % (done, total, elapsed, remaining))
    sys.stdout.flush()


def prepare_shared(cfg, quiet=False):
    """Graphs, canonical index, resolved radius, frozen pool, epochs, clusters.

    The pool and the deadlines come from 03's hash-keyed cache under
    Results/<CITY_KEY>/_shared, so this generates exactly the files 03 would
    when they are absent and loads 03's when they are present.  That shared
    cache is the whole basis of comparability between the arms.
    """
    if not cfg.graph_dir.is_dir():
        raise FileNotFoundError("Graph directory does not exist: %s" % cfg.graph_dir)

    graphs = gbc.load_graphs(cfg.graph_dir)
    tract_to_index, index_to_tract, node_positions = gbc.build_canonical_index(graphs)
    num_hotspots = len(tract_to_index)

    adjacencies = {g: gbc.build_adjacency(graphs[g], tract_to_index, g)
                   for g in GRAPH_TYPES}

    # must happen before the pool is built: it feeds order_pool_hash
    cfg.MAX_DELIVERY_TIME = resolve_mdt(cfg, adjacencies, verbose=not quiet)

    stats = {g: gbc.compute_graph_stats(g, graphs[g], adjacencies[g],
                                        cfg.MAX_DELIVERY_TIME)
             for g in GRAPH_TYPES}
    gbc.write_json(cfg.or_shared_dir / "graph_stats.json", stats)

    order_pool, pool_hash, feasible_pairs = gbc.build_order_pool(
        adjacencies, cfg, num_hotspots, verbose=not quiet)
    time_limits, ref_times, limits_hash = gbc.compute_time_limits(
        order_pool, adjacencies[cfg.TIME_LIMIT_REF_GRAPH], cfg, pool_hash,
        verbose=not quiet)

    epochs = build_epochs(order_pool, cfg.EPOCH_SECONDS)

    # ASSERTION 5: epochs partition the pool
    total = sum(len(v) for v in epochs.values())
    if total != len(order_pool):
        raise AssertionError(
            "ASSERTION 5 FAILED: epochs hold %d orders for a pool of %d"
            % (total, len(order_pool)))

    clusters = build_clusters(epochs, node_positions, cfg, verbose=not quiet)

    # ASSERTION 6: clusters partition each epoch
    for epoch_idx, members in clusters.items():
        flat = [p for group in members for p in group]
        if sorted(flat) != list(range(len(epochs[epoch_idx]))):
            raise AssertionError(
                "ASSERTION 6 FAILED: clusters of epoch %d do not partition its %d orders"
                % (epoch_idx, len(epochs[epoch_idx])))

    gbc.write_json(cfg.or_shared_dir / "ortools_inputs.json", {
        "city_key": cfg.city_key,
        "order_pool_hash": pool_hash,
        "time_limits_hash": limits_hash,
        "max_delivery_time": cfg.MAX_DELIVERY_TIME,
        "mdt_mode": cfg.MDT_MODE,
        "n_orders": len(order_pool),
        "num_hotspots": num_hotspots,
        "feasible_pairs": feasible_pairs,
        "n_epochs": len(epochs),
        "n_subproblems": sum(len(v) for v in clusters.values()),
        "config": cfg.as_dict(),
    })

    return {
        "graphs": graphs, "adjacencies": adjacencies, "stats": stats,
        "tract_to_index": tract_to_index, "node_positions": node_positions,
        "num_hotspots": num_hotspots,
        "order_pool": order_pool, "pool_hash": pool_hash,
        "feasible_pairs": feasible_pairs,
        "time_limits": time_limits, "limits_hash": limits_hash,
        "epochs": epochs, "clusters": clusters,
    }


def print_banner(cfg, shared):
    stats = shared["stats"]
    epochs = shared["epochs"]
    sizes = [len(c) for v in shared["clusters"].values() for c in v]
    print(gbc.rule())
    print("OR-TOOLS PDPTW BASELINE   city=%s   graphs=%s"
          % (cfg.city_key, ",".join(cfg.graph_types_to_run)))
    print(gbc.rule())
    print("graphs")
    for graph_type in GRAPH_TYPES:
        s = stats[graph_type]
        print("  %-6s %5d nodes %7d edges  avg_deg %6.2f  coverage %5.1f%%"
              % (graph_type, s["nodes"], s["edges"], s["avg_degree"],
                 100.0 * s["coverage_within_max_delivery_time"]))
    print()
    print("workload")
    print("  hotspots            : %d" % shared["num_hotspots"])
    print("  LOAD_PER_HOTSPOT    : %d  (per hotspot, NOT a total)" % cfg.LOAD_PER_HOTSPOT)
    print("  TOTAL ORDERS        : %d" % len(shared["order_pool"]))
    print("  delivery radius     : %.0fs (%.1f min), mode=%s"
          % (cfg.MAX_DELIVERY_TIME, cfg.MAX_DELIVERY_TIME / 60.0, cfg.MDT_MODE))
    print("  order pool hash     : %s" % shared["pool_hash"])
    print("  deadline hash       : %s" % shared["limits_hash"])
    print()
    print("decomposition")
    print("  epochs              : %d x %ds" % (len(epochs), cfg.EPOCH_SECONDS))
    print("  subproblems         : %d  (orders min %d / mean %.0f / max %d)"
          % (len(sizes), min(sizes), float(np.mean(sizes)), max(sizes)))
    print("  capacity            : %d onboard, fleet unlimited" % cfg.VEHICLE_CAPACITY)
    print("  solver              : %s + %s, %ds per subproblem"
          % (cfg.FIRST_SOLUTION, cfg.METAHEURISTIC, cfg.SOLVER_TIME_LIMIT))
    print("  workers             : %d" % cfg.n_workers)
    print()
    print("MODEL NOTES  (properties of tours vs relays, not bugs)")
    if cfg.EPOCH_SECONDS != cfg.WAIT_TIME:
        print("  ! EPOCH_SECONDS=%d differs from WAIT_TIME=%d, so this arm and 03's"
              % (cfg.EPOCH_SECONDS, cfg.WAIT_TIME))
        print("    bundling arm do NOT have the same lookahead. Part of any gap is")
        print("    information rather than decision quality.")
    else:
        print("  EPOCH_SECONDS == WAIT_TIME, so both arms have identical lookahead")
        print("    and any gap is decision quality.")
    print("  Batch-optimal at a %ds dispatch window, not globally optimal." % cfg.EPOCH_SECONDS)
    print("  No relay handoffs: a package stays on one vehicle (STOP_SERVICE_TIME=%d)."
          % cfg.STOP_SERVICE_TIME)
    print("  avg_hops counts STOPS, not graph edges.")
    print("  Clusters do not share vehicles, so distinct_vehicles is inflated;")
    print("    read vehicle_distance_km and vehicle_trips as the primary outputs.")
    if stats["CLIQUE"]["density"] < 0.999:
        print("  ! CLIQUE is NOT complete here (density %.3f, %d edges) -- it is a k-NN"
              % (stats["CLIQUE"]["density"], stats["CLIQUE"]["edges"]))
        print("    graph, so the toggle name is misleading on this city.")
    print(gbc.rule())


# ============================================================================
# 8. AGGREGATION, THE HEURISTIC JOIN, PLOTS
# ============================================================================

HEADLINE_METRICS = [
    "vehicle_distance_km", "success_rate", "avg_delivery_time",
    "distinct_vehicles", "vehicle_trips", "avg_hops",
]


def load_summaries(cfg):
    summaries = {}
    for graph_type in GRAPH_TYPES:
        path = cfg.or_graph_dir(graph_type) / "summary.json"
        if path.is_file():
            summaries[graph_type] = gbc.read_json(path)
    return summaries


def check_shared_inputs(summaries):
    """03's assertions 3 and 4, applied across separate 04 invocations.

    The hashes are what make "UMST today, CLIQUE next week" checkable: if two
    graphs ran against different pools or different deadlines the comparison is
    void, and that has to fail loudly rather than produce a plausible table.
    """
    pool_hashes = set(s.get("order_pool_hash") for s in summaries.values())
    limit_hashes = set(s.get("time_limits_hash") for s in summaries.values())
    radii = set(round(float(s.get("max_delivery_time", 0)), 3) for s in summaries.values())
    problems = []
    if len(pool_hashes) > 1:
        problems.append("ASSERTION 3 FAILED: order pool hashes differ across graphs: %s"
                        % sorted(pool_hashes))
    if len(limit_hashes) > 1:
        problems.append("ASSERTION 4 FAILED: deadline hashes differ across graphs: %s"
                        % sorted(limit_hashes))
    if len(radii) > 1:
        problems.append("ASSERTION 11 FAILED: MAX_DELIVERY_TIME differs across graphs: %s"
                        % sorted(radii))
    if problems:
        raise AssertionError("\n".join(problems))
    return (sorted(pool_hashes)[0] if pool_hashes else None,
            sorted(limit_hashes)[0] if limit_hashes else None)


def read_heuristic_table(cfg):
    """03's comparison_table.csv for this city at drop rate 0, by graph and mode.

    Returns {} when 03 has not been run, or when it ran against a different
    pool -- in which case joining would silently compare two different
    workloads, which is exactly the failure the hashes exist to prevent.
    """
    import csv

    table = cfg.comparison_dir / "comparison_table.csv"
    if not table.is_file():
        return {}, "03 has not been run for this city"

    summary_path = cfg.comparison_dir / "comparison_summary.json"
    if summary_path.is_file():
        payload = gbc.read_json(summary_path)
        ours = load_summaries(cfg)
        if ours:
            theirs = payload.get("order_pool_hash")
            mine = sorted(set(s["order_pool_hash"] for s in ours.values()))[0]
            if theirs and theirs != mine:
                return {}, ("03 ran against pool %s but 04 used %s -- rerun 03 with "
                            "--max-delivery-time %.0f --load-per-hotspot %d"
                            % (theirs, mine, cfg.MAX_DELIVERY_TIME, cfg.LOAD_PER_HOTSPOT))

    out = {}
    with open(str(table), "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                if abs(float(row["drop_rate"])) > 1e-9:
                    continue
            except (TypeError, ValueError):
                continue
            out[(row["graph_type"], row["mode"])] = row
    return out, None


def _num(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_comparison(cfg, summaries, quiet=False):
    rows = []
    for graph_type in GRAPH_TYPES:
        summary = summaries.get(graph_type)
        if summary is None:
            continue
        metrics = summary["modes"][MODE]
        stats = summary.get("graph_stats", {})
        row = {
            "city_key": summary["city_key"],
            "graph_type": graph_type,
            "mode": MODE,
            "n_orders": summary["n_orders"],
            "max_delivery_time": summary.get("max_delivery_time", ""),
            "coverage_within_max_delivery_time":
                round(stats.get("coverage_within_max_delivery_time", float("nan")), 6),
            "graph_edges": stats.get("edges", ""),
            "graph_avg_degree": round(stats.get("avg_degree", float("nan")), 4),
        }
        for metric in OR_METRIC_COLUMNS:
            row[metric] = round(metrics.get(metric, {}).get("mean", float("nan")), 6)
        rows.append(row)

    fieldnames = ["city_key", "graph_type", "mode", "n_orders", "max_delivery_time",
                  "coverage_within_max_delivery_time", "graph_edges",
                  "graph_avg_degree"] + OR_METRIC_COLUMNS
    path = cfg.or_comparison_dir / "ortools_comparison_table.csv"
    gbc.write_csv(path, fieldnames, rows)
    if not quiet:
        print("  wrote %s" % path)
    return rows


VS_COLUMNS = [
    "city_key", "graph_type", "n_orders",
    # completion first: everything below is meaningless without it
    "baseline_completion_pct", "bundling_completion_pct", "ortools_completion_pct",
    "recovery_reliable",
    "baseline_veh_km", "bundling_veh_km", "ortools_veh_km", "ortools_solo_km",
    "baseline_km_per_delivered", "bundling_km_per_delivered", "ortools_km_per_delivered",
    "heuristic_saving_km", "optimizer_saving_km", "recovery_pct",
    "baseline_success_pct", "bundling_success_pct", "ortools_success_pct",
    "bundling_avg_time_s", "ortools_avg_time_s",
    "bundling_vehicle_trips", "ortools_vehicle_trips",
]

# 03's simulation stops at HOURS*3600 + GRACE_TIME and counts only the distance
# actually driven by then, so on a slow graph its vehicle_distance_km covers only
# the orders that finished.  04 has no cutoff -- soft deadlines mean every order
# is delivered, late if need be.  Below this completion rate the two arms are not
# doing the same amount of work and their kilometres must not be subtracted.
RELIABLE_COMPLETION = 0.99


def write_heuristic_join(cfg, summaries, quiet=False):
    """The payoff table: how much of the achievable consolidation the heuristic got.

    recovery_pct = (baseline - bundling) / (baseline - ortools).  The baseline arm
    is one vehicle per package, so baseline - ortools is the consolidation a solver
    can find on that graph and baseline - bundling is what the myopic heuristic
    found.

    That ratio is only meaningful when both arms delivered the same orders.  03
    cuts its simulation off and can leave half the pool undelivered on a slow
    graph (measured: MST completes 52.7% on GeoDistColumbus, 48.6% on
    GeoDistChicago), which truncates its distance and inflates recovery_pct
    without limit.  Such rows are kept -- deleting evidence is worse -- but
    flagged `recovery_reliable = NO` and reported with a per-delivered-order
    figure that is comparable even under truncation.

    ortools_solo_km is an independent check on the whole distance pipeline: it is
    computed by 04 from its own matrices and must equal 03's baseline_veh_km
    wherever completion is 100%.
    """
    heuristic, problem = read_heuristic_table(cfg)
    if problem:
        if not quiet:
            print("  heuristic join skipped: %s" % problem)
        return None

    rows = []
    for graph_type in GRAPH_TYPES:
        summary = summaries.get(graph_type)
        bundling = heuristic.get((graph_type, "bundling"))
        baseline = heuristic.get((graph_type, "baseline"))
        if summary is None or bundling is None or baseline is None:
            continue
        metrics = summary["modes"][MODE]
        n_orders = summary["n_orders"]

        base_done = _num(baseline["completion_rate"], 0.0)
        bund_done = _num(bundling["completion_rate"], 0.0)
        or_done = metrics["completion_rate"]["mean"]

        base_km = _num(baseline["vehicle_distance_km"])
        bund_km = _num(bundling["vehicle_distance_km"])
        or_km = metrics["vehicle_distance_km"]["mean"]

        optimizer_saving = base_km - or_km
        heuristic_saving = base_km - bund_km
        recovery = (100.0 * heuristic_saving / optimizer_saving) \
            if abs(optimizer_saving) > 1e-9 else float("nan")
        reliable = min(base_done, bund_done, or_done) >= RELIABLE_COMPLETION

        def per_delivered(km, rate):
            delivered = rate * n_orders
            return (km / delivered) if delivered > 0 else float("nan")

        rows.append({
            "city_key": summary["city_key"],
            "graph_type": graph_type,
            "n_orders": n_orders,
            "baseline_completion_pct": round(100.0 * base_done, 2),
            "bundling_completion_pct": round(100.0 * bund_done, 2),
            "ortools_completion_pct": round(100.0 * or_done, 2),
            "recovery_reliable": "yes" if reliable else "NO",
            "baseline_veh_km": round(base_km, 3),
            "bundling_veh_km": round(bund_km, 3),
            "ortools_veh_km": round(or_km, 3),
            "ortools_solo_km": round(metrics["solo_distance_km"]["mean"], 3),
            "baseline_km_per_delivered": round(per_delivered(base_km, base_done), 5),
            "bundling_km_per_delivered": round(per_delivered(bund_km, bund_done), 5),
            "ortools_km_per_delivered": round(per_delivered(or_km, or_done), 5),
            "heuristic_saving_km": round(heuristic_saving, 3),
            "optimizer_saving_km": round(optimizer_saving, 3),
            "recovery_pct": round(recovery, 2),
            "baseline_success_pct": round(100.0 * _num(baseline["success_rate"]), 2),
            "bundling_success_pct": round(100.0 * _num(bundling["success_rate"]), 2),
            "ortools_success_pct": round(100.0 * metrics["success_rate"]["mean"], 2),
            "bundling_avg_time_s": round(_num(bundling["avg_delivery_time"]), 1),
            "ortools_avg_time_s": round(metrics["avg_delivery_time"]["mean"], 1),
            "bundling_vehicle_trips": round(_num(bundling["vehicle_trips"]), 0),
            "ortools_vehicle_trips": round(metrics["vehicle_trips"]["mean"], 0),
        })

    if not rows:
        if not quiet:
            print("  heuristic join skipped: no matching graph/mode rows in 03's table")
        return None

    path = cfg.or_comparison_dir / "ortools_vs_heuristic.csv"
    gbc.write_csv(path, VS_COLUMNS, rows)
    if not quiet:
        print("  wrote %s" % path)
        print()
        print("  %-7s %11s %11s %11s %9s %9s  %s"
              % ("graph", "baseline_km", "bundling_km", "ortools_km",
                 "recovery", "03_compl", "km per delivered order (base/bund/or)"))
        print("  " + "-" * 104)
        for row in rows:
            print("  %-7s %11.1f %11.1f %11.1f %8.1f%% %8.1f%%  %.4f / %.4f / %.4f%s"
                  % (row["graph_type"], row["baseline_veh_km"], row["bundling_veh_km"],
                     row["ortools_veh_km"], row["recovery_pct"],
                     row["baseline_completion_pct"],
                     row["baseline_km_per_delivered"], row["bundling_km_per_delivered"],
                     row["ortools_km_per_delivered"],
                     "" if row["recovery_reliable"] == "yes" else "   <-- UNRELIABLE"))

        unreliable = [r for r in rows if r["recovery_reliable"] != "yes"]
        if unreliable:
            print()
            print("  WARNING: %d row(s) flagged UNRELIABLE." % len(unreliable))
            print("    03 stops simulating at HOURS*3600 + GRACE_TIME and counts only the")
            print("    distance driven by then, so on these graphs it left part of the pool")
            print("    undelivered while 04 delivered all of it. Subtracting those kilometres")
            print("    inflates recovery_pct without limit. Compare km_per_delivered instead,")
            print("    or raise 03's --grace-time until completion reaches 100%.")

        mismatched = [r for r in rows
                      if r["baseline_completion_pct"] >= 99.995
                      and abs(r["ortools_solo_km"] - r["baseline_veh_km"])
                      > 1e-3 * max(1.0, r["baseline_veh_km"])]
        if mismatched:
            print()
            print("  WARNING: solo_km != 03 baseline_veh_km at full completion on %s."
                  % ", ".join(r["graph_type"] for r in mismatched))
            print("    These are computed independently and must agree. Investigate before")
            print("    trusting any distance in this table.")
    return rows


def print_table(summaries, cfg):
    print()
    print("  %-7s %10s %10s %9s %9s %9s %9s"
          % ("graph", "veh_km", "solo_km", "saved%", "success%", "vehicles", "avg_t_s"))
    print("  " + "-" * 70)
    for graph_type in GRAPH_TYPES:
        summary = summaries.get(graph_type)
        if summary is None:
            continue
        m = summary["modes"][MODE]
        print("  %-7s %10.1f %10.1f %8.2f%% %8.2f%% %9.0f %9.1f"
              % (graph_type, m["vehicle_distance_km"]["mean"],
                 m["solo_distance_km"]["mean"], m["saved_vs_solo_pct"]["mean"],
                 100.0 * m["success_rate"]["mean"], m["distinct_vehicles"]["mean"],
                 m["avg_delivery_time"]["mean"]))
        stats = summary.get("graph_stats", {})
        if stats:
            print("  %-7s %-10s coverage within %.0fs = %5.1f%%   (%d edges, avg deg %.2f)"
                  % (graph_type, "[graph]", cfg.MAX_DELIVERY_TIME,
                     100.0 * stats["coverage_within_max_delivery_time"],
                     stats["edges"], stats["avg_degree"]))


def make_plots(cfg, summaries, quiet=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                 # pragma: no cover
        print("  plots skipped: matplotlib unavailable (%s)" % exc)
        return

    present = [g for g in GRAPH_TYPES if g in summaries]
    if not present:
        return
    heuristic, _problem = read_heuristic_table(cfg)

    # 03's palette, so the figures sit beside its own without clashing
    colors = {"baseline": gbc.ARM_COLORS["baseline"],
              "bundling": gbc.ARM_COLORS["bundling"],
              MODE: gbc.GRAPH_LEVEL_COLOR}
    panels = [
        ("vehicle_distance_km", "Vehicle distance", "km", 1.0),
        ("success_rate", "Success rate", "% of orders on time", 100.0),
        ("avg_delivery_time", "Avg delivery time", "seconds", 1.0),
        ("vehicle_trips", "Vehicle trips", "edge traversals", 1.0),
        ("distinct_vehicles", "Distinct vehicles", "count", 1.0),
        ("avg_hops", "Avg hops / stops", "count", 1.0),
    ]

    arms = [m for m in ("baseline", "bundling") if heuristic] + [MODE]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.patch.set_facecolor("#fcfcfb")
    axes = axes.ravel()
    x = np.arange(len(present), dtype=float)
    width = 0.8 / max(1, len(arms))

    for axis, (metric, title, unit, scale) in zip(axes, panels):
        for slot, arm in enumerate(arms):
            values = []
            for graph_type in present:
                if arm == MODE:
                    values.append(scale * summaries[graph_type]["modes"][MODE]
                                  [metric]["mean"])
                else:
                    row = heuristic.get((graph_type, arm), {})
                    values.append(scale * _num(row.get(metric), 0.0))
            axis.bar(x + (slot - (len(arms) - 1) / 2.0) * width, values, width,
                     label=arm, color=colors[arm], edgecolor="none")
        axis.set_title(title, color=gbc.INK_PRIMARY, fontsize=11)
        axis.set_ylabel(unit, color=gbc.INK_SECONDARY, fontsize=9)
        axis.set_xticks(x)
        axis.set_xticklabels(present, color=gbc.INK_SECONDARY)
        axis.grid(axis="y", color=gbc.GRID_COLOR, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(arms), frameon=False)
    fig.suptitle("%s  --  OR-Tools PDPTW vs 03 heuristic arms" % cfg.city_key,
                 color=gbc.INK_PRIMARY, fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    os.makedirs(str(cfg.or_comparison_dir), exist_ok=True)
    path = cfg.or_comparison_dir / "ortools_plots.png"
    fig.savefig(str(path), dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    if not quiet:
        print("  wrote %s" % path)


def aggregate(cfg, quiet=False):
    summaries = load_summaries(cfg)
    if not summaries:
        print("No summary.json under %s -- solve first." % cfg.or_city_dir)
        return None

    pool_hash, limits_hash = check_shared_inputs(summaries)
    any_summary = list(summaries.values())[0]
    cfg.MAX_DELIVERY_TIME = float(any_summary.get("max_delivery_time",
                                                  cfg.MAX_DELIVERY_TIME))

    print("\n" + gbc.rule())
    print("AGGREGATE  %s" % cfg.city_key)
    print(gbc.rule())
    print("  order pool hash : %s" % pool_hash)
    print("  deadline hash   : %s   (identical across graphs -> assertions 3, 4, 11 hold)"
          % limits_hash)

    write_comparison(cfg, summaries, quiet=quiet)
    print_table(summaries, cfg)
    print()
    write_heuristic_join(cfg, summaries, quiet=quiet)
    if cfg.MAKE_PLOTS:
        make_plots(cfg, summaries, quiet=quiet)
    return summaries


def write_cross_city(city_configs):
    import csv

    rows = []
    fieldnames = None
    for _stem, cfg in city_configs:
        table = cfg.or_comparison_dir / "ortools_comparison_table.csv"
        if not table.is_file():
            continue
        with open(str(table), "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames)
    if not rows:
        return None
    path = gbc.RESULTS_ROOT / "OR tools" / "cross_city_ortools.csv"
    gbc.write_csv(path, fieldnames, rows)
    print("\nwrote %s  (%d rows across %d cities)"
          % (path, len(rows), len(set(r["city_key"] for r in rows))))
    return path


# ============================================================================
# 9. MAIN
# ============================================================================

def run_one_city(cfg, args):
    if args.aggregate:
        return aggregate(cfg, quiet=args.quiet)

    shared = prepare_shared(cfg, quiet=args.quiet)
    print_banner(cfg, shared)

    for graph_type in cfg.graph_types_to_run:
        print("\n%s\n%s\n%s" % (gbc.rule("-"), graph_type, gbc.rule("-")))
        run_graph(graph_type, cfg, shared, quiet=args.quiet)

    if cfg.GRAPH_TYPE == "ALL":
        return aggregate(cfg, quiet=args.quiet)

    print("\nRun --graph-type ALL (or the other graphs) then --aggregate "
          "to build the comparison table.")
    return None


def main(argv=None):
    cfg, args = resolve_config(argv)

    if args.list_cities:
        return gbc.main(["--list-cities"])

    if args.calibrate:
        stems = args.cities
        if stems is None:
            stems = [cfg.city_key]
        elif stems == "ALL":
            stems = gbc.discover_city_stems()
        calibrate_all(cfg, stems, quiet=args.quiet)
        return 0

    city_configs = gbc.resolve_cities(cfg, args.cities)

    if len(city_configs) > 1:
        print(gbc.rule())
        print("CITY SWEEP: %d cities x %d graph(s)"
              % (len(city_configs), len(cfg.graph_types_to_run)))
        for stem, _c in city_configs:
            print("  %s" % stem)
        print(gbc.rule())

    for stem, city_cfg in city_configs:
        if len(city_configs) > 1:
            print("\n\n" + gbc.rule("#"))
            print("### CITY: %s" % stem)
            print(gbc.rule("#"))
        try:
            run_one_city(city_cfg, args)
        except Exception as exc:                             # keep the sweep going
            if len(city_configs) == 1:
                raise
            print("\n  !! %s FAILED: %s: %s" % (stem, type(exc).__name__, exc))

    if len(city_configs) > 1:
        write_cross_city(city_configs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
