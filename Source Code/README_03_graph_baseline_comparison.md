# `03_graph_baseline_comparison.py`

UMST / MST / Clique delivery-simulation comparison harness.

Runs one identical delivery workload over three graph structures and two dispatch
strategies, and reports how distance, time, and vehicle count change.

---

## Table of contents

1. [What it does](#1-what-it-does)
2. [Requirements](#2-requirements)
3. [Quick start](#3-quick-start)
4. [Input data it expects](#4-input-data-it-expects)
5. [Cities — listing, switching, sweeping](#5-cities--listing-switching-sweeping)
6. [Graphs — the three structures](#6-graphs--the-three-structures)
7. [Full hyperparameter reference](#7-full-hyperparameter-reference)
8. [Example commands](#8-example-commands)
9. [Where results go and what is in them](#9-where-results-go-and-what-is-in-them)
10. [Metric definitions](#10-metric-definitions)
11. [How the comparison is kept fair](#11-how-the-comparison-is-kept-fair)
12. [Caching and hashes](#12-caching-and-hashes)
13. [Built-in assertions](#13-built-in-assertions)
14. [Scale guidance and known data issues](#14-scale-guidance-and-known-data-issues)
15. [Known model limitations](#15-known-model-limitations)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. What it does

Three graph structures, all built over the same census-tract hotspots:

| Toggle | Graph file | What it is |
|---|---|---|
| `UMST` | `umst_graph.graphml` | union of MSTs — the contribution being evaluated |
| `MST` | `mst_graph.graphml` | a single minimum spanning tree |
| `CLIQUE` | `gh_hotspot_graph.graphml` | the dense reference graph (see [§14](#14-scale-guidance-and-known-data-issues) — **not actually complete on full-size cities**) |

Each graph is run under two arms:

- **`bundling`** — packages share a vehicle when they are at the same hotspot heading
  to the same next hotspot. Vehicles are edge-level relays: acquired at departure,
  handed off at each hop.
- **`baseline`** — every package gets its own dedicated vehicle, door to door, no
  waiting and no sharing.

**Both arms follow the identical route.** Only sharing differs, so any gap between
them is attributable to bundling and nothing else.

The output is 3 graphs × 2 arms of metrics, plus a graph-level coverage statistic,
plus an optional edge-failure resilience sweep.

The script is standalone. It reads the `.graphml` files as final inputs, never
regenerates them, never calls GraphHopper, and never imports from the notebooks.

---

## 2. Requirements

```
python >= 3.9
networkx
numpy
matplotlib     # only needed for --aggregate plots; everything else runs without it
```

All three are already in the project's `requirements.txt`. No GPU, no network access.

---

## 3. Quick start

```bash
cd "Source Code"

# 1. See which cities are usable
python 03_graph_baseline_comparison.py --list-cities

# 2. Smoke test — tiny workload, one run, finishes in a couple of seconds
python 03_graph_baseline_comparison.py --load-per-hotspot 2 --n-runs 1

# 3. A real single-city comparison across all three graphs
python 03_graph_baseline_comparison.py --graph-type ALL

# 4. Everything, every usable city
python 03_graph_baseline_comparison.py --cities ALL --graph-type ALL
```

From inside a notebook:

```python
%run 03_graph_baseline_comparison.py --graph-type ALL --load-per-hotspot 20
```

Every run prints a startup banner with the fully resolved config, node/edge counts
for all three graphs, the order-pool size and hash, and the sparsity warning.

---

## 4. Input data it expects

```
UMST_approach/
├── Delivery_Data/
│   └── <prefix><City><suffix> - RL Delivery Data/
│       └── UMST Graph/
│           └── graphs/
│               ├── umst_graph.graphml          (or .graphml.xml)
│               ├── mst_graph.graphml           (or .graphml.xml)
│               └── gh_hotspot_graph.graphml    (or .graphml.xml)
├── Results/                                    <- everything is written here
└── Source Code/
    ├── 03_graph_baseline_comparison.py
    └── README_03_graph_baseline_comparison.md  (this file)
```

Both the `.graphml` and `.graphml.xml` suffixes are accepted — the files in this repo
actually use `.graphml.xml`, and the script falls back automatically.

**Required graph attributes.** Nodes need `lat` / `lon`. Edges need `distance`
(kilometres) and `time` (**minutes** — converted to seconds internally). A graph
missing `time`, or carrying `distance <= 0` or `time <= 0`, is rejected at load time
rather than silently treated as instantaneous travel.

---

## 5. Cities — listing, switching, sweeping

A city is identified by its folder stem — the part before `" - RL Delivery Data"`.
The stem encodes three settings:

```
[GeoDist]<CityName>[_mini]
   │          │        └── MINI = true
   │          └─────────── CITY_NAME
   └────────────────────── WITH_GEODESIC_DISTANCE = true
```

### List what is available

```bash
python 03_graph_baseline_comparison.py --list-cities
```

```
  [USABLE  ] Chicago_mini                  31 nodes
             --city-name 'Chicago'          --mini true  --with-geodesic-distance false
  [USABLE  ] Columbus_mini                 26 nodes
             --city-name 'Columbus'         --mini true  --with-geodesic-distance false
  [USABLE  ] GeoDistChicago               863 nodes
             --city-name 'Chicago'          --mini false --with-geodesic-distance true
  [UNUSABLE] GeoDistCity of New York     2190 nodes
             ! UMST: 3568/3568 edges lack usable distance/time
  [USABLE  ] GeoDistColumbus              278 nodes
             --city-name 'Columbus'         --mini false --with-geodesic-distance true
```

`--list-cities` opens each graph and checks the edge attributes, so `UNUSABLE` means
the data is genuinely unfit, not that a file is missing.

### Switch to one city

Two equivalent ways:

```bash
# by stem (shorter, recommended)
python 03_graph_baseline_comparison.py --cities GeoDistChicago

# by the three underlying flags
python 03_graph_baseline_comparison.py \
    --city-name Chicago --mini false --with-geodesic-distance true
```

### Sweep several cities

```bash
python 03_graph_baseline_comparison.py --cities ALL --graph-type ALL
python 03_graph_baseline_comparison.py --cities Columbus_mini,GeoDistChicago --graph-type ALL
```

`--cities` overrides `--city-name` / `--mini` / `--with-geodesic-distance`. Each city
gets its own results folder and its own comparison table; a combined
`Results/cross_city_comparison.csv` is written at the end. **If one city fails the
sweep continues** and prints which one failed and why.

---

## 6. Graphs — the three structures

```bash
--graph-type UMST      # just one
--graph-type MST
--graph-type CLIQUE
--graph-type ALL       # all three, then aggregate automatically
```

`ALL` runs them sequentially into their own folders and then builds the comparison
table and plots without a second command.

Running one graph at a time also works and is still a valid comparison — the order
pool and the deadlines are cached by hash (see [§12](#12-caching-and-hashes)), so you
can run UMST today and CLIQUE next week against the identical workload, then:

```bash
python 03_graph_baseline_comparison.py --aggregate
```

`--aggregate` never simulates. It reads the `summary.json` files already on disk,
verifies every graph used the same pool and the same deadlines, and writes the
comparison outputs.

---

## 7. Full hyperparameter reference

Every constant at the top of the script is also a CLI flag: **lowercase the constant
and turn underscores into dashes** (`MAX_BUNDLE_SIZE` → `--max-bundle-size`). The CLI
wins over the file. Edit the file to change a default permanently.

### City / data selection

| Constant | Flag | Default | Meaning |
|---|---|---|---|
| `CITY_NAME` | `--city-name` | `"Columbus"` | City folder stem, without prefix/suffix |
| `MINI` | `--mini` | `True` | Use the `<City>_mini` subset |
| `WITH_GEODESIC_DISTANCE` | `--with-geodesic-distance` | `False` | Use the `GeoDist<City>` folder variant |

Booleans accept `true/false`, `1/0`, `yes/no`, `on/off`.

### The toggle

| Constant | Flag | Default | Meaning |
|---|---|---|---|
| `GRAPH_TYPE` | `--graph-type` | `"UMST"` | `UMST` \| `MST` \| `CLIQUE` \| `ALL` |

### Order pool — defines the workload

| Constant | Flag | Default | Meaning |
|---|---|---|---|
| `ORDER_POOL_GRAPH` | `--order-pool-graph` | `"UMST"` | Which graph's reachability defines the feasible order set |
| `MAX_DELIVERY_TIME` | `--max-delivery-time` | `900` | Seconds. An OD pair is feasible if its shortest-path travel time on `ORDER_POOL_GRAPH` is ≤ this |
| `LOAD_PER_HOTSPOT` | `--load-per-hotspot` | `100` | **Orders per hotspot, not a total.** Total orders = hotspots × this |
| `HOURS` | `--hours` | `1` | Order arrival window, hours |
| `PEAKS` | `--peaks` | `[0.25, 0.75]` | Gaussian peak positions as a fraction of the window. Comma-separated |
| `SIGMA` | `--sigma` | `10` | Gaussian spread, minutes |

Arrivals are a Gaussian mixture over minute bins, drawn with `np.random.multinomial`,
then scattered uniformly inside each minute. OD pairs are sampled uniformly from the
feasible list.

### Deadline — defines what "on time" means

| Constant | Flag | Default | Meaning |
|---|---|---|---|
| `TIME_LIMIT_REF_GRAPH` | `--time-limit-ref-graph` | `"UMST"` | Which graph's routing the deadline is frozen on |
| `BUFFER_TIME_TYPE` | `--buffer-time-type` | `"multiply"` | `multiply` \| `fixed_buffer` \| `rangewise` \| `randomized` |
| `TIME_TOLERANCE_FACTOR` | `--time-tolerance-factor` | `2.5` | Used by `multiply` |
| `FIXED_BUFFER_SECONDS` | `--fixed-buffer-seconds` | `300` | Used by `fixed_buffer` |
| `RANDOM_BUFFER_RANGE` | `--random-buffer-range` | `(120, 480)` | Used by `randomized`, seconds. Pass as `lo,hi` |
| `RANGEWISE_BANDS` | `--rangewise-bands` | see below | Used by `rangewise`. Pass as JSON |

| `BUFFER_TIME_TYPE` | Deadline |
|---|---|
| `multiply` | `travel_time × TIME_TOLERANCE_FACTOR` |
| `fixed_buffer` | `travel_time + FIXED_BUFFER_SECONDS` |
| `randomized` | `travel_time + uniform(*RANDOM_BUFFER_RANGE)`, seeded |
| `rangewise` | `travel_time + buffer` from the first matching band |

Default bands — `(upper_bound_seconds, buffer_seconds)`:

```python
[(600, 300), (1200, 600), (1500, 720), (inf, 900)]
```

From the CLI, pass JSON and use the string `"inf"` for the last upper bound:

```bash
--buffer-time-type rangewise --rangewise-bands '[[300,120],["inf",600]]'
```

The deadline changes **only** which deliveries count as successful. It does not change
routing, distance, or vehicle counts — verified: all four buffer types produce
identical `veh_km` and differ only in `success%`.

### Simulation

| Constant | Flag | Default | Meaning |
|---|---|---|---|
| `TIME_STEP` | `--time-step` | `1` | Seconds per tick |
| `GRACE_TIME` | `--grace-time` | `1800` | Seconds past the arrival window before cutoff |
| `WAIT_TIME` | `--wait-time` | `120` | Max seconds an order holds at a hotspot hoping for a bundling partner |
| `MAX_BUNDLE_SIZE` | `--max-bundle-size` | `5` | Max orders sharing one vehicle on one edge |
| `VEHICLE_COOLDOWN` | `--vehicle-cooldown` | `5` | Seconds before a released vehicle is reusable |
| `VEHICLE_CHANGE_TIME` | `--vehicle-change-time` | `1` | Seconds charged per relay handoff |
| `SOLO_DEPARTURE` | `--solo-departure` | `True` | `False` reproduces the old hold-forever behaviour |

**`WAIT_TIME` is the single most influential knob.** It is the whole bundling
mechanism. On Columbus_mini at `--load-per-hotspot 20`:

| `WAIT_TIME` | participation | distance saved | success | avg hold |
|---|---|---|---|---|
| `0` | 1.5% | 0.3% | 100% | 0s |
| `120` (default) | 75.6% | 32.7% | 100% | 140s |
| `600` | 95.2% | 48.1% | 87.3% | 267s |

More waiting buys more sharing and costs punctuality. Use this pair of runs as a
sanity check after any change to the dispatch logic — if they do not move in opposite
directions, something is wrong.

`SOLO_DEPARTURE=False` exists to reproduce the original notebook behaviour, where a
lone order waited forever for a partner that might never come. It is kept so the
before/after can be shown on demand, not because it is a sensible setting.

### Experiment

| Constant | Flag | Default | Meaning |
|---|---|---|---|
| `N_RUNS` | `--n-runs` | `5` | Runs per (graph, drop rate) |
| `DROP_RATES` | `--drop-rates` | `[0.0]` | Comma-separated edge-failure rates, e.g. `0.0,0.1,0.25,0.5` |
| `GLOBAL_SEED` | `--global-seed` | `42` | Master seed for the pool, the deadlines, and edge drops |
| `SAVE_PER_ORDER` | `--save-per-order` | `True` | Write `orders_run0.csv` |
| `MAKE_PLOTS` | `--make-plots` | `True` | Draw the comparison plots during aggregation |

> **At `--drop-rates 0.0` the simulation is fully deterministic**, so all `N_RUNS`
> runs are byte-identical and every confidence interval is zero. Variance comes only
> from edge failures. The banner warns about this. Use `--n-runs 1` when sweeping
> only drop rate 0, and raise it when you actually drop edges.

### Mode flags (not hyperparameters)

| Flag | Meaning |
|---|---|
| `--aggregate` | Do not simulate; read existing `summary.json` files and write the comparison outputs |
| `--cities` | `ALL`, or a comma-separated list of city folder stems |
| `--list-cities` | Print usable cities with a health check, then exit |
| `--quiet` | Suppress per-arm progress lines |
| `-h`, `--help` | Full flag list with defaults |

---

## 8. Example commands

```bash
# --- getting oriented -------------------------------------------------------
python 03_graph_baseline_comparison.py --list-cities
python 03_graph_baseline_comparison.py --help

# --- smoke test (seconds) ---------------------------------------------------
python 03_graph_baseline_comparison.py --load-per-hotspot 2 --n-runs 1

# --- the headline single-city comparison ------------------------------------
python 03_graph_baseline_comparison.py --graph-type ALL

# --- resilience sweep across edge failure rates -----------------------------
python 03_graph_baseline_comparison.py --graph-type ALL --drop-rates 0.0,0.1,0.25,0.5 --n-runs 5

# --- one graph at a time, aggregate later -----------------------------------
python 03_graph_baseline_comparison.py --graph-type UMST
python 03_graph_baseline_comparison.py --graph-type MST
python 03_graph_baseline_comparison.py --graph-type CLIQUE
python 03_graph_baseline_comparison.py --aggregate

# --- every usable city ------------------------------------------------------
python 03_graph_baseline_comparison.py --cities ALL --graph-type ALL --load-per-hotspot 20 --n-runs 2

# --- a full-size city, with a delivery-time budget that suits its scale ------
python 03_graph_baseline_comparison.py --cities GeoDistChicago --graph-type ALL \
    --load-per-hotspot 10 --max-delivery-time 2700

# --- WAIT_TIME sensitivity (the bundling mechanism) -------------------------
python 03_graph_baseline_comparison.py --load-per-hotspot 20 --n-runs 1 --wait-time 0
python 03_graph_baseline_comparison.py --load-per-hotspot 20 --n-runs 1 --wait-time 600

# --- reproduce the original hold-forever behaviour --------------------------
python 03_graph_baseline_comparison.py --load-per-hotspot 10 --n-runs 1 --solo-departure false

# --- alternative deadline rules ---------------------------------------------
python 03_graph_baseline_comparison.py --buffer-time-type fixed_buffer --fixed-buffer-seconds 600
python 03_graph_baseline_comparison.py --buffer-time-type randomized --random-buffer-range 60,300
python 03_graph_baseline_comparison.py --buffer-time-type rangewise \
    --rangewise-bands '[[300,120],[900,400],["inf",900]]'

# --- change the workload shape ----------------------------------------------
python 03_graph_baseline_comparison.py --peaks 0.5 --sigma 5          # one sharp rush
python 03_graph_baseline_comparison.py --peaks 0.2,0.5,0.8 --hours 2  # three peaks, 2h

# --- define the order pool by the dense graph instead of UMST ---------------
python 03_graph_baseline_comparison.py --graph-type ALL --order-pool-graph CLIQUE

# --- larger convoys, costlier handoffs --------------------------------------
python 03_graph_baseline_comparison.py --max-bundle-size 10 --vehicle-change-time 30
```

---

## 9. Where results go and what is in them

Everything is written under **`UMST_approach/Results/`**. Directories are created with
`exist_ok=True` and **nothing is ever deleted** — reruns overwrite matching files only.

```
UMST_approach/Results/
├── cross_city_comparison.csv          <- only when sweeping >1 city
└── <CITY_KEY>/                        <- e.g. Columbus_mini, GeoDist_Chicago
    ├── _shared/
    │   ├── graph_stats.json           structural + coverage stats for ALL THREE graphs
    │   ├── order_pool_<hash>.json     the frozen workload
    │   └── time_limits_<hash>.json    the frozen deadlines
    ├── UMST/
    │   ├── drop_00/
    │   │   ├── runs.csv               one row per (run_idx, mode)
    │   │   ├── summary.json           mean/std/min/max/median/ci_95 per metric per mode
    │   │   └── orders_run0.csv        per-order detail, run 0 only, both arms
    │   ├── drop_10/ ...
    │   └── drop_25/ ...
    ├── MST/    (same shape)
    ├── CLIQUE/ (same shape)
    └── comparison/
        ├── comparison_table.csv       52 columns, one row per (graph, drop rate, mode)
        ├── comparison_summary.json    nested by drop rate, then graph, then mode
        └── comparison_plots*.png
```

`CITY_KEY` is `[GeoDist_]<City>[_mini]` — note the underscore after `GeoDist`, which
distinguishes the results key from the data folder stem.

`drop_XX` is `int(drop_rate * 100)`, zero-padded to two digits.

Plot filenames: `comparison_plots.png` when there is exactly one drop rate, otherwise
one `comparison_plots_drop_XX.png` per rate.

### `runs.csv` — 34 columns

```
run_idx, mode, n_orders, completed, completion_rate, successful, success_rate,
failed, failed_rate, vehicle_distance_km, package_distance_km, distance_saved_km,
distance_saved_pct, avg_delivery_time, median_delivery_time, p90, p95, avg_delay,
median_delay, max_delay, avg_hold_time, distinct_vehicles, vehicle_trips,
peak_concurrent_vehicles, avg_concurrent_vehicles, orders_ever_bundled,
bundle_participation_rate, bundles_formed, avg_bundle_size,
max_bundle_size_observed, avg_hops, solo_departures, sim_end_clock,
in_flight_at_cutoff
```

### `orders_run0.csv` — 18 columns

```
mode, id, start_node, end_node, start_time, end_time, expected_travel_time,
time_limit, actual_delivery_time, delay, within_buffer, hops, distance_traveled,
num_vehicle_changes, times_bundled, total_hold_time, completed, failed
```

Both arms are in the same file, distinguished by `mode`. `id` is the order id, stable
across every graph and every city run from the same pool — join on it to compare an
individual order's fate on UMST versus MST.

### `summary.json`

```jsonc
{
  "city_key": "Columbus_mini",
  "graph_type": "UMST",
  "drop_rate": 0.0,
  "n_runs": 2,
  "order_pool_hash": "d34b5b595bb3a7ab",
  "time_limits_hash": "02f9eaeac16a4e9a",
  "n_orders": 260,
  "routing":       { "avg_hops": 2.62, "avg_expected_travel_time": 496.15 },
  "edge_failures": { "edges_dropped": 0, "edges_total": 0 },
  "config":        { /* all 28 resolved settings */ },
  "modes": {
    "bundling": { "success_rate": { "mean": …, "std": …, "min": …,
                                    "max": …, "median": …, "ci_95": … }, … },
    "baseline": { … }
  }
}
```

`ci_95 = 1.96 × std / sqrt(n)`. All numpy types are converted to native Python and
infinities are written as the string `"inf"`, so the file is strict-valid JSON.

### `_shared/graph_stats.json`

Written once per city for **all three graphs**, regardless of which one the toggle
selected:

```
graph_type, nodes, edges, avg_degree, density, total_edge_length_km,
total_od_pairs, pairs_within_max_delivery_time,
coverage_within_max_delivery_time, connected_pairs,
mean_path_time_all_pairs, mean_hops_all_pairs
```

### `comparison_table.csv` — 52 columns

Identifiers, then graph-level context, then every comparison metric paired with its
95% CI:

```
city_key, graph_type, drop_rate, mode, n_runs, n_orders, avg_hops_routed,
coverage_within_max_delivery_time, graph_edges, graph_avg_degree,
success_rate, success_rate_ci95, completion_rate, completion_rate_ci95, …
```

### `cross_city_comparison.csv`

Every city's `comparison_table.csv` concatenated with a leading `city_folder` column.
Written only when more than one city is swept.

### The plots

Six panels — success rate, vehicle distance, distance saved, distinct vehicles,
average delivery time, and OD coverage — grouped by graph, with 95% CI error bars.
Blue is `bundling`, orange is `baseline`. The distance-saved panel shows the bundling
arm alone (baseline is zero by construction), and the coverage panel is graph-level,
belonging to neither arm.

---

## 10. Metric definitions

### Volume

| Metric | Definition |
|---|---|
| `n_orders` | Size of the frozen pool. Identical for every graph and arm |
| `completed` | Reached the destination before the cutoff |
| `successful` | Completed **within the frozen `time_limit`** |
| `failed` | No path exists after edge failures. Stays in the denominator |
| `*_rate` | The above divided by `n_orders` |

Orders still in flight at the cutoff are **incomplete, not failed**.

### Distance

| Metric | Definition |
|---|---|
| `package_distance_km` | Sum of every order's own travelled distance. k packages on one edge count k times |
| `vehicle_distance_km` | Sum of vehicle odometers. That same edge counts once |
| `distance_saved_km` | `package − vehicle` |
| `distance_saved_pct` | `100 × saved / package` |

Both sums cover all traffic, completed or not, which makes
`vehicle_distance_km <= package_distance_km` exact. In the baseline arm the two are
equal, because every package rides alone.

> **Read `distance_saved_pct` together with absolute `vehicle_distance_km`.** MST
> often posts the *highest* saving percentage while having the *worst* absolute
> vehicle-km — it has the most hops, so there is more to share. High saving on a bad
> graph is not a win.

### Time

| Metric | Definition |
|---|---|
| `avg_delivery_time` | Mean of `(end_time − start_time) + num_vehicle_changes × VEHICLE_CHANGE_TIME` over completed orders |
| `median_delivery_time`, `p90`, `p95` | Percentiles of the same |
| `avg_delay` / `median_delay` / `max_delay` | `actual_delivery_time − time_limit`, **signed** — negative means early |
| `avg_hold_time` | Mean seconds spent waiting at hotspots, over activated orders |

### Vehicles

| Metric | Definition |
|---|---|
| `distinct_vehicles` | Number of distinct vehicle objects. **Not fleet size** — vehicles are relays and get reused |
| `vehicle_trips` | Edge traversals by a vehicle. This is the honest workload number |
| `peak_concurrent_vehicles` | Maximum simultaneously in use |
| `avg_concurrent_vehicles` | Mean over all ticks |

### Bundling (zero by construction in the baseline arm)

| Metric | Definition |
|---|---|
| `orders_ever_bundled` | Orders that shared a vehicle at least once |
| `bundle_participation_rate` | That, divided by `n_orders` |
| `bundles_formed` | Departing convoys of ≥2. Bundles re-form at each hop, so this counts re-formations |
| `avg_bundle_size` / `max_bundle_size_observed` | Sizes at departure |
| `avg_hops` | Mean edges traversed per completed order |
| `solo_departures` | Departures by a lone order after its hold expired |

### Graph-level (in `_shared/graph_stats.json`)

`coverage_within_max_delivery_time` is the headline: the fraction of all ordered OD
pairs routable within `MAX_DELIVERY_TIME` on that graph. It is where the cost of
sparsity shows up, and it is computed for all three graphs on every run.

---

## 11. How the comparison is kept fair

Three design decisions do the work:

**One canonical node index.** A single tract↔index mapping is built once from the
shared node set and used for every graph. The node sets must be identical or the run
aborts.

**One frozen order pool.** The workload is generated once per city and cached. Every
graph and both arms attempt the identical set of orders.

**One frozen deadline.** Deadlines are computed once on `TIME_LIMIT_REF_GRAPH` and
reused unchanged everywhere — including after rerouting caused by edge failures. If
each graph got a deadline derived from its own routing, a graph that routed badly
would receive a proportionally looser deadline and `success_rate` could not detect the
difference.

A consequence, printed as a warning at startup: the pool is defined by
`ORDER_POOL_GRAPH` feasibility, so some orders will exceed `MAX_DELIVERY_TIME` on a
sparser graph. That is a real result about sparsity, not a bug.

Relay handoffs are charged exactly once per edge traversal per order in **both** arms,
so the per-hop penalty is a property of the graph rather than of which code path ran.

---

## 12. Caching and hashes

Two artefacts are cached in `_shared/` and keyed by a hash of the settings that
define them:

| File | Hash covers |
|---|---|
| `order_pool_<hash>.json` | `city_key`, `ORDER_POOL_GRAPH`, `MAX_DELIVERY_TIME`, `LOAD_PER_HOTSPOT`, `HOURS`, `PEAKS`, `SIGMA`, `GLOBAL_SEED` |
| `time_limits_<hash>.json` | the pool hash, plus `TIME_LIMIT_REF_GRAPH`, `BUFFER_TIME_TYPE` and all buffer parameters |

Change any of those and you get a new hash and a fresh file; the old one stays. This
is what makes "run UMST today, CLIQUE next week" valid — both load the same cached
pool. Both hashes are recorded in every `summary.json`, and `--aggregate` refuses to
build a comparison table if they disagree across graphs.

To force regeneration, delete the relevant `_shared/*.json` file.

---

## 13. Built-in assertions

The script fails loudly rather than producing a plausible wrong number. All of these
have been verified to fire on injected violations:

1. All three graphs have identical node sets
2. Every edge has `distance > 0` and `time > 0`
3. The order pool is identical across graph types (hash comparison)
4. `time_limit` values are identical across graph types for the same order id
5. Zero orders dropped during per-graph routing
6. Every member of a departing bundle shares the same next hop
7. `vehicle_distance_km <= package_distance_km` in the bundling arm
8. Baseline `distinct_vehicles` equals the number of activated orders

---

## 14. Scale guidance and known data issues

### `LOAD_PER_HOTSPOT` scales with city size

It is **per hotspot**. At the default `100`:

| City | Hotspots | Total orders at load 100 |
|---|---|---|
| Columbus_mini | 26 | 2,600 |
| Chicago_mini | 31 | 3,100 |
| GeoDistColumbus | 278 | 27,800 |
| GeoDistChicago | 863 | 86,300 |

Start full-size cities at `--load-per-hotspot 10` or `20`. For reference, a four-city
sweep (3 graphs × 2 drop rates × 2 runs × 2 arms) at load 20 took **3m36s** total.

### `MAX_DELIVERY_TIME=900` is a mini-city setting

On GeoDistChicago, the densest available graph reaches only **12.9%** of OD pairs
within 15 minutes, and UMST reaches 11.4%. The pool collapses to short local trips and
the coverage statistic stops discriminating between graphs. Raise it for full-size
cities — try `--max-delivery-time 2700` and read the coverage line in the banner.

### `gh_hotspot_graph` is not a clique on full-size cities

| City | Nodes | Edges | Complete would be | Reality |
|---|---|---|---|---|
| Columbus_mini | 26 | 325 | 325 | genuinely complete |
| Chicago_mini | 31 | 465 | 465 | genuinely complete |
| GeoDistColumbus | 278 | 1,676 | 38,503 | 4.4% — k-NN, avg degree 12 |
| GeoDistChicago | 863 | 4,961 | 371,953 | 1.3% — k-NN, avg degree 11.5 |

It is built with `k_neighbors=10`, so only the mini cities are true cliques. On
GeoDistChicago the `CLIQUE` arm averages 3.27 hops, not ~1. The toggle name is
accurate for the mini cities and misleading elsewhere — worth a footnote in any write-up.

### New York cannot be run

`GeoDistCity of New York` has `distance == weight` (raw geodesic km) and **no `time`
attribute on any of its 104,275 clique edges**. The graphs were saved before the
GraphHopper enrichment step ran. Assertion 2 rejects them; without that guard every
New York trip would be instantaneous and the results would look plausible but be
meaningless. Fixing it requires re-running the GraphHopper step in the graph-building
notebook — this script deliberately never regenerates graph files.

### MST is not the tree UMST was built around

`mst_graph.graphml` was built with `weight="distance"` (road km) while the MSTs inside
UMST were built with `weight="weight"` (geodesic km). On Columbus_mini those two trees
differ by 7 of 25 edges, and 2 of MST's 25 edges are not in UMST at all. MST is
therefore not in general a subgraph of UMST. The banner prints this on every run.

---

## 15. Known model limitations

Carry these into any write-up; they are properties of the model, not bugs to fix here.

- Bundling is myopic single-hop co-direction merging with no explicit savings test.
  The savings are implicit: k packages share one edge traversal.
- Bundles dissolve and re-form at each hop, so a "bundle" is a single-edge convoy and
  `bundles_formed` counts re-formations rather than distinct groupings.
- Vehicle repositioning between trips is free.
- The per-hop relay handoff penalty structurally disadvantages sparse graphs. This is
  a genuine property of relay delivery, not an implementation artifact.
- Vehicles are edge-level relays, not persistent couriers, so `distinct_vehicles`
  must not be read as fleet size. Use `vehicle_trips` for workload.

---

## 16. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ASSERTION 2 FAILED: ... missing distance and/or time` | Graphs lack GraphHopper road data. Run `--list-cities` to confirm; the city needs its graphs rebuilt |
| `ASSERTION 1 FAILED: node sets differ` | The three graph files came from different builds of the city. Rebuild them together |
| `ASSERTION 3/4 FAILED` during `--aggregate` | Graphs were run under different pool or deadline settings. Rerun them with matching flags, or delete the stale `summary.json` |
| `Graph directory does not exist` | Wrong city flags. Use `--list-cities` and copy a stem into `--cities` |
| `Unknown city folder stem(s)` | Typo — the error lists every valid stem |
| Every CI is zero | Expected at `--drop-rates 0.0`; the simulation is deterministic. Add edge drops or use `--n-runs 1` |
| Near-zero bundling | `WAIT_TIME` too low for the mean edge traversal time, or the workload is too sparse. Raise `--wait-time` or `--load-per-hotspot` |
| Coverage near zero on a big city | `MAX_DELIVERY_TIME` too small for the city's scale. Raise it |
| Many orders incomplete at cutoff | Raise `--grace-time`, or lower `--wait-time`. Check `in_flight_at_cutoff` in `runs.csv` |
| Plots not written | matplotlib missing, or `--make-plots false`. Plots only render during aggregation |
| Run is slow | `LOAD_PER_HOTSPOT` is per hotspot. Lower it, or lower `--n-runs` at drop rate 0 |
