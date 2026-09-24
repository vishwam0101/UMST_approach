# `04_ortools_baseline.py`

An OR-Tools PDPTW arm for the UMST / MST / CLIQUE comparison.

Given the **identical** frozen order pool and the **identical** frozen deadlines that
`03_graph_baseline_comparison.py` uses, what does a classical vehicle-routing solver
achieve on each network?

---

## Table of contents

1. [What it does and why](#1-what-it-does-and-why)
2. [Requirements](#2-requirements)
3. [Quick start](#3-quick-start)
4. [The delivery radius policy](#4-the-delivery-radius-policy)
5. [The two decompositions](#5-the-two-decompositions)
6. [The model](#6-the-model)
7. [Full hyperparameter reference](#7-full-hyperparameter-reference)
8. [Example commands](#8-example-commands)
9. [Where results go](#9-where-results-go)
10. [Metric definitions](#10-metric-definitions)
11. [How the comparison is kept fair](#11-how-the-comparison-is-kept-fair)
12. [Built-in assertions](#12-built-in-assertions)
13. [Differences from the relay simulation](#13-differences-from-the-relay-simulation)
14. [Benchmarks behind the defaults](#14-benchmarks-behind-the-defaults)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. What it does and why

03 runs two arms:

| Arm | What it is |
|---|---|
| `bundling` | myopic single-hop co-direction merging |
| `baseline` | one dedicated vehicle per package, door to door |

`baseline` is deliberately trivial. It proves bundling saves *something*, but it cannot
say whether the heuristic is any **good**, nor whether a denser graph earns its cost for
an operator who plans well.

This script adds a third arm, `ortools`, that answers exactly that. The payoff is the
**recovery ratio** in `ortools_vs_heuristic.csv`:

```
recovery_pct = (baseline_veh_km - bundling_veh_km) / (baseline_veh_km - ortools_veh_km)
```

`baseline - ortools` is the consolidation a solver can actually find on that graph;
`baseline - bundling` is what the myopic heuristic found. Their ratio turns
"UMST beats MST" into "UMST recovers X% of achievable consolidation where MST recovers Y%".

### Read `recovery_reliable` before you read `recovery_pct`

03 stops simulating at `HOURS*3600 + GRACE_TIME` (5400s by default) and counts only the
distance **actually driven by then**. On a slow graph it therefore leaves part of the pool
undelivered and reports a truncated distance. 04 has no cutoff -- soft deadlines mean every
order is delivered, late if need be.

Measured 03 completion at load 100:

| City | UMST | MST | CLIQUE |
|---|---|---|---|
| Chicago_mini | 100% | 100% | 100% |
| Columbus_mini | 100% | 100% | 100% |
| GeoDistColumbus | 94.8% | **52.7%** | 96.1% |
| GeoDistChicago | 95.6% | **48.6%** | 97.1% |

Where completion is below 100% the two arms are **not doing the same amount of work**, and
subtracting their kilometres inflates `recovery_pct` without limit -- the MST row on
GeoDistColumbus reads 157% purely for this reason. Those rows are kept, not deleted, but
flagged `recovery_reliable = NO` and printed with a `<-- UNRELIABLE` marker.

Two ways to get a valid comparison on those cities:

- **Compare `*_km_per_delivered`**, which normalises by orders actually delivered and is
  meaningful under truncation.
- **Rerun 03 with a larger `--grace-time`** until completion reaches 100%. `GRACE_TIME` is
  not part of `order_pool_hash` or `time_limits_hash`, so raising it keeps the same frozen
  pool and deadlines and the arms stay comparable.

`ortools_solo_km` is an independent check on the entire distance pipeline: 04 computes it
from its own matrices, and it must equal 03's `baseline_veh_km` wherever completion is
100%. It does, exactly, on both mini cities. The join prints a warning if it ever does not.

**The mechanism that makes this a comparison of graphs** is the cost matrix: arc cost
`(i, j)` is the shortest path on *the graph being evaluated*. A sparse network forces
detours and inflates the solver's own optimum.

This file never modifies 03. It imports it and reuses its graph loading, canonical node
index, Dijkstra, and -- critically -- its hash-keyed order pool and deadline caches.

---

## 2. Requirements

```
python >= 3.9
networkx, numpy, ortools
matplotlib     # only for the plots
```

All are in the project `requirements.txt`. Verified against **ortools 9.15.6755**.
No GPU, no network access.

---

## 3. Quick start

```bash
cd "Source Code"

# 1. Resolve the delivery radius per city and print the 03 commands that match it
python 04_ortools_baseline.py --calibrate --cities ALL

# 2. Run 03 at those thresholds, so both arms share an order pool  (see step 1 output)
python 03_graph_baseline_comparison.py --cities Columbus_mini --graph-type ALL \
    --load-per-hotspot 100 --n-runs 1 --max-delivery-time 801

# 3. Smoke test -- seconds
python 04_ortools_baseline.py --cities Columbus_mini --graph-type UMST \
    --load-per-hotspot 2 --epoch-seconds 3600 --solver-time-limit 5

# 4. A real single-city comparison
python 04_ortools_baseline.py --cities Columbus_mini --graph-type ALL

# 5. Everything
python 04_ortools_baseline.py --graph-type ALL \
    --cities Chicago_mini,Columbus_mini,GeoDistColumbus,GeoDistChicago
```

**Step 2 is not optional if you want the heuristic join.** 04 refuses to join against a
03 run with a different pool hash, and says so rather than silently comparing two
different workloads.

---

## 4. The delivery radius policy

`MAX_DELIVERY_TIME` decides which OD pairs may become orders. It is inherently a **scale**
parameter, so a single fixed value cannot serve every city -- measured coverage
(UMST / MST / CLIQUE):

| City | @900s | @1800s | @2700s |
|---|---|---|---|
| Chicago_mini | 77.6 / 70.8 / 100 | **100 / 100 / 100** saturated | 100 / 100 / 100 |
| GeoDistChicago | **11.4 / 3.8 / 12.9** collapsed | 35.5 / 10.7 / 39.2 | 58.6 / 19.7 / 63.0 |

Two policies, one flag:

```bash
--mdt-mode coverage                          # DEFAULT: per city, UMST coverage = 60%
--mdt-mode coverage --coverage-target 0.75   # same policy, looser radius
--mdt-mode fixed --max-delivery-time 900     # uniform 15 minutes everywhere
```

`coverage` sets the threshold to the `COVERAGE_TARGET` quantile of the anchor graph's
all-pairs travel-time distribution, so demand is **structurally** identical across cities
rather than parameter-identical:

> Orders are drawn uniformly from the nearest 60% of OD pairs reachable on UMST.

Resolved values at the default:

| City | MDT | UMST | MST | CLIQUE |
|---|---|---|---|---|
| Chicago_mini | 695s (11.6 min) | 60.0% | 52.5% | 100.0% |
| Columbus_mini | 801s (13.3 min) | 60.0% | 40.0% | 96.9% |
| GeoDistColumbus | 2742s (45.7 min) | 60.0% | 21.8% | 64.0% |
| GeoDistChicago | 2764s (46.1 min) | 60.0% | 20.4% | 64.5% |

UMST is the anchor rather than CLIQUE deliberately: CLIQUE means different things across
the two city classes (a true clique on the mini cities, a k-NN graph on the GeoDist ones),
and anchoring on it would bake that inconsistency into the demand model.

UMST coverage becomes 60% by construction and stops discriminating -- acceptable, since
`ORDER_POOL_GRAPH = UMST` already defines the pool by UMST feasibility. **MST's collapse
from 52.5% to 20.4% as cities grow stays a real measured result.**

`MAX_DELIVERY_TIME` feeds `order_pool_hash`, so switching policy yields a different hash,
a fresh pool and fresh deadlines. **Nothing is deleted** -- both policies coexist on disk
and the assertions refuse to mix them.

---

## 5. The two decompositions

### Dispatch epochs

The arrival window is cut into `EPOCH_SECONDS` batches, each solved **at its close**.

```
t=0 ---------------------------------------------------- t=3600
    | epoch 1 | epoch 2 | epoch 3 | ... | epoch 30 |
         v          v         v             v
       VRP_1      VRP_2     VRP_3        VRP_30
```

A monolithic solve would be **clairvoyant** -- it would know at t=0 every order arriving
all hour -- while the heuristic it is compared against holds each order at most
`WAIT_TIME`. Most of the resulting gap would be information, not graph structure.

**Keep `EPOCH_SECONDS == WAIT_TIME` (both 120 by default).** Then both arms have identical
lookahead and any gap is decision quality. The banner warns when they differ.

The batch wait is **charged to the order**: an order arriving at 245s in epoch
`[240, 360)` waits 115s, and its delivery time is measured from its own arrival, not from
the epoch close. Without this OR-Tools would get free time and `success_rate` would not be
comparable.

Report results as **batch-optimal at a 120s dispatch window**, not globally optimal.

### Spatial clusters

Each epoch is clustered by hotspot lat/lon to roughly `TARGET_CLUSTER_SIZE` orders per
subproblem. This is **not only** a speed measure -- see
[section 14](#14-benchmarks-behind-the-defaults), where 8 clusters beat one monolithic
solve by 7.8% at equal wall clock. It is also physically sound: two orders 40 km apart
cannot share a vehicle without blowing both deadlines, so the cross-cluster merges being
forbidden were never going to be chosen.

**Clustering is on lat/lon, which is graph-independent**, computed once per (city, epoch)
and cached in `_shared/clusters.json` so all three graphs use the identical decomposition.
Clustering on graph distance would hand each graph a different decomposition and destroy
the comparison. Assertion 6 enforces the partition.

---

## 6. The model

Per `(city, graph, epoch, cluster)`: a capacitated pickup-and-delivery VRP with soft time
windows. `n` orders become `2n + 1` nodes -- node 0 is a virtual depot with zero-cost arcs
both ways, order `i` occupies pickup `2i+1` and dropoff `2i+2`.

| Element | Choice |
|---|---|
| Objective | total vehicle distance, in metres |
| Capacity | `VEHICLE_CAPACITY` orders onboard at once |
| Fleet | `n` vehicles -- one per order, so chaining is never forced |
| Deadlines | **soft** upper bound on dropoff, `LATENESS_PENALTY` per second |
| Repositioning | free (the zero-cost depot), mirroring 03's `Vehicle.start_new_trip` |

**Why soft deadlines.** 03's simulation never refuses an order -- it delivers late and
sets `within_buffer = False`. A hard window would make the solver drop orders and break
the shared denominator the assertions exist to protect.

**Why one vehicle per order.** Fewer would force chaining, and a chained vehicle deadheads
from a dropoff to the next pickup -- distance 03's `baseline` arm never pays, since it
conjures a fresh vehicle at every origin. With `n` vehicles the solver can always fall
back to one-per-order, which makes `vehicle_distance_km <= solo_distance_km` an invariant.

**Why `LATENESS_PENALTY = 17`.** The objective is in metres, so the penalty is an exchange
rate between seconds and metres. 17 m/s means "one minute late is worth one kilometre of
driving". At the naive 1000, a single second outweighs a kilometre and the solver stops
optimising distance altogether. Measured on Columbus_mini at load 2: penalty 0 gives
153.36 km at 94.2% success; penalty 17 gives 154.28 km at 100% success -- punctuality
costs 0.6% distance.

### The penalty is a trade-off dial, and the trade-off is the result

At realistic density the penalty does not simply "buy punctuality". Measured on
Chicago_mini/UMST at load 100:

| `--lateness-penalty` | veh_km | saved vs solo | success | avg time |
|---|---|---|---|---|
| 0 | **2753.1** | **59.7%** | 23.5% | 2405s |
| 17 *(default)* | 3514.0 | 48.6% | 41.8% | 1497s |
| 100 | 3631.5 | 46.9% | 41.8% | 1500s |
| 500 | 3687.5 | 46.1% | 41.7% | 1508s |
| 2000 | 3683.7 | 46.1% | 41.2% | 1506s |
| 03 `bundling` (relay) | 3249.8 | 52.5% | **99.65%** | **455s** |
| 03 `baseline` | 6838.4 | 0% | 100% | 410s |

**Success saturates around 42% and will not rise however hard you push**, while distance
degrades steadily. So a single OR-Tools row is a point on a curve, not "the answer".
Sweeping `--lateness-penalty` and reporting the curve is the honest presentation.

The structural reason matters more than the numbers, and belongs in any write-up:

> In the relay model a package **never detours** -- it follows its own shortest path
> (3.03 hops on Chicago_mini/UMST) and merely shares each edge with whoever happens to be
> going the same way. In a tour model a package **must** detour to consolidate, because
> the vehicle carrying it also has to serve everyone else onboard.

Relay buys consolidation for free and pays nothing in distance; VRP buys it with detours
and pays in time. That is why at penalty 0 OR-Tools beats the heuristic on distance
(2753 vs 3250 km, 15% better) while delivering at 2405s against 455s. The two arms are not
better and worse versions of one thing -- they sit at different points on a
distance-versus-time frontier, and the relay model's hotspot-handoff infrastructure is
what buys it that position.

---

## 7. Full hyperparameter reference

04 accepts **every flag 03 does**, plus the following. Lowercase the constant and turn
underscores into dashes (`EPOCH_SECONDS` -> `--epoch-seconds`). The CLI wins over the file.

### Decomposition

| Constant | Flag | Default | Meaning |
|---|---|---|---|
| `EPOCH_SECONDS` | `--epoch-seconds` | `120` | Dispatch epoch. **Keep equal to `WAIT_TIME`** |
| `TARGET_CLUSTER_SIZE` | `--target-cluster-size` | `300` | Target orders per subproblem |
| `MAX_SUBPROBLEM_ORDERS` | `--max-subproblem-orders` | `500` | Hard cap; clusters split until under it |

### Solver

| Constant | Flag | Default | Meaning |
|---|---|---|---|
| `SOLVER_TIME_LIMIT` | `--solver-time-limit` | `15` | Seconds per subproblem |
| `FIRST_SOLUTION` | `--first-solution` | `PARALLEL_CHEAPEST_INSERTION` | OR-Tools strategy name |
| `METAHEURISTIC` | `--metaheuristic` | `GUIDED_LOCAL_SEARCH` | OR-Tools metaheuristic name |

### Fleet and objective

| Constant | Flag | Default | Meaning |
|---|---|---|---|
| `VEHICLE_CAPACITY` | `--vehicle-capacity` | `5` | Orders onboard at once (= `MAX_BUNDLE_SIZE`) |
| `VEHICLE_FIXED_COST` | `--vehicle-fixed-cost` | `100` | Metres charged per vehicle used |
| `LATENESS_PENALTY` | `--lateness-penalty` | `17` | Cost per second late, in metres |
| `STOP_SERVICE_TIME` | `--stop-service-time` | `0` | Seconds charged per stop |

### Delivery radius

| Constant | Flag | Default | Meaning |
|---|---|---|---|
| `MDT_MODE` | `--mdt-mode` | `coverage` | `coverage` (per city) or `fixed` (uniform) |
| `COVERAGE_TARGET` | `--coverage-target` | `0.60` | Coverage mode: target coverage, 0-1 |
| `COVERAGE_ANCHOR` | `--coverage-anchor` | `UMST` | Coverage mode: graph to calibrate on |

### Execution

| Constant | Flag | Default | Meaning |
|---|---|---|---|
| `N_WORKERS` | `--n-workers` | `0` | Parallel processes; 0 = `cpu_count - 2`. **1 runs serially in-process**, which is what you want when debugging |
| `SAVE_ROUTES` | `--save-routes` | `True` | Write `routes.json` (large on big cities) |
| `MAKE_PLOTS` | `--make-plots` | `True` | Draw plots during aggregation |

### Mode flags

| Flag | Meaning |
|---|---|
| `--calibrate` | Resolve the radius per city, write `calibrated_mdt.json`, print the matching 03 commands, exit |
| `--aggregate` | Do not solve; read existing `summary.json` files and write the comparison outputs |
| `--cities` | `ALL`, or a comma-separated list of city folder stems |
| `--list-cities` | Delegates to 03's city health check |
| `--quiet` | Suppress per-subproblem progress |

---

## 8. Example commands

```bash
# --- getting oriented -------------------------------------------------------
python 04_ortools_baseline.py --calibrate --cities ALL
python 04_ortools_baseline.py --help

# --- smoke test (seconds) ---------------------------------------------------
python 04_ortools_baseline.py --cities Columbus_mini --graph-type UMST \
    --load-per-hotspot 2 --epoch-seconds 3600 --solver-time-limit 5

# --- the correctness check that matters -------------------------------------
# capacity 1 forbids consolidation, so this MUST reproduce 03's baseline arm
python 04_ortools_baseline.py --cities Columbus_mini --graph-type UMST \
    --vehicle-capacity 1 --vehicle-fixed-cost 0 --lateness-penalty 0

# --- the headline single-city comparison ------------------------------------
python 04_ortools_baseline.py --cities Columbus_mini --graph-type ALL

# --- all four cities --------------------------------------------------------
python 04_ortools_baseline.py --graph-type ALL \
    --cities Chicago_mini,Columbus_mini,GeoDistColumbus,GeoDistChicago

# --- uniform 15-minute radius instead of per-city calibration ---------------
python 04_ortools_baseline.py --calibrate --mdt-mode fixed --max-delivery-time 900
python 04_ortools_baseline.py --cities ALL --graph-type ALL \
    --mdt-mode fixed --max-delivery-time 900

# --- how much does solver effort buy? ---------------------------------------
python 04_ortools_baseline.py --cities Columbus_mini --graph-type UMST --solver-time-limit 5
python 04_ortools_baseline.py --cities Columbus_mini --graph-type UMST --solver-time-limit 60

# --- how much does batch size buy? ------------------------------------------
python 04_ortools_baseline.py --cities Columbus_mini --graph-type ALL --epoch-seconds 60
python 04_ortools_baseline.py --cities Columbus_mini --graph-type ALL --epoch-seconds 300

# --- charge tours for their stops, neutralising the no-handoff advantage ----
python 04_ortools_baseline.py --cities Columbus_mini --graph-type ALL --stop-service-time 30

# --- debugging: serial, one process, clean traceback ------------------------
python 04_ortools_baseline.py --cities Columbus_mini --graph-type UMST --n-workers 1

# --- rebuild tables and plots without solving -------------------------------
python 04_ortools_baseline.py --cities ALL --aggregate
```

---

## 9. Where results go

Everything is under **`UMST_approach/Results/OR tools/`**. Nothing under
`Results/<CITY_KEY>/` is touched except the `_shared/` pool and deadline cache, which 03
and 04 deliberately share.

```
Results/OR tools/
├── calibrated_mdt.json                 mode, per-city radius, coverage, the 03 commands
├── cross_city_ortools.csv              rolled up across cities
└── <CITY_KEY>/
    ├── _shared/
    │   ├── ortools_inputs.json         pool hash, deadline hash, epoch, load, solver cfg
    │   ├── graph_stats.json            structural + coverage stats for all three graphs
    │   ├── clusters.json               per-epoch cluster assignment, shared by 3 graphs
    │   └── matrix_<GRAPH>_{dist,time}.npy   cached all-pairs hotspot matrices
    ├── UMST/
    │   ├── summary.json                mode = "ortools"
    │   ├── runs.csv                    one row
    │   ├── orders.csv                  per-order detail, joins to 03's on `id`
    │   ├── epochs.csv                  per (epoch, cluster): n, status, objective, seconds
    │   └── routes.json                 the solution itself, for audit
    ├── MST/      (same shape)
    ├── CLIQUE/   (same shape)
    └── comparison/
        ├── ortools_comparison_table.csv
        ├── ortools_vs_heuristic.csv    the recovery ratio -- written only when 03's
        │                               table exists AND its pool hash matches
        └── ortools_plots.png
```

`CITY_KEY` is `[GeoDist_]<City>[_mini]` -- the underscore after `GeoDist` distinguishes
the results key from the data folder stem, exactly as in 03.

---

## 10. Metric definitions

Column names follow 03's wherever the quantity means the same thing. Where tours and
relays genuinely differ, the extra columns are explicit rather than smuggled into an
existing name.

### Distance

| Metric | Definition |
|---|---|
| `vehicle_distance_km` | Total vehicle odometer. **The headline number** |
| `loaded_distance_km` | Vehicle distance while carrying at least one order |
| `deadhead_km` | `vehicle - loaded`. Empty running, which relay vehicles never do |
| `package_distance_km` | Sum over orders of distance ridden. k packages on one leg count k times |
| `solo_distance_km` | Sum over orders of their own shortest path -- what 03's `baseline` arm costs |
| `distance_saved_km` / `_pct` | `package - vehicle`, 03-compatible |
| `saved_vs_solo_pct` | `100 * (solo - vehicle) / solo`. **Use this one** -- it is the honest "how much did consolidation save" |

### Time

| Metric | Definition |
|---|---|
| `avg_delivery_time` | Mean `dropoff_clock - order.start_time`, **including the batch wait** |
| `median` / `p90` / `p95` | Percentiles of the same |
| `avg_delay` etc. | `actual - time_limit`, signed; negative means early |
| `avg_hold_time` | Mean `epoch_close - start_time` -- the batching cost |

### Vehicles and consolidation

| Metric | Definition |
|---|---|
| `distinct_vehicles` | Non-empty routes summed over epochs and clusters. **Inflated** -- see section 13 |
| `vehicle_trips` | Arc traversals, depot arcs excluded |
| `orders_ever_bundled` | Orders on a route that carried 2 or more |
| `bundles_formed` | Routes with 2 or more orders |
| `avg_bundle_size` | Mean peak onboard count per route |
| `avg_hops` | Mean **stops** between pickup and dropoff, not graph edges |

### Solver housekeeping

`n_subproblems`, `total_solve_seconds`, `objective_total`.

---

## 11. How the comparison is kept fair

**One canonical node index.** Inherited from 03's `build_canonical_index`; the node sets
must be identical across graphs or the run aborts.

**One frozen order pool.** Loaded from 03's hash-keyed cache under
`Results/<CITY_KEY>/_shared/`. 04 generates exactly the file 03 would when it is absent.

**One frozen deadline.** Computed once on `TIME_LIMIT_REF_GRAPH` and reused everywhere, so
a graph that routes badly is penalised instead of handed a looser deadline.

**One decomposition.** Cluster assignment is computed from lat/lon and cached, so all
three graphs solve the identical set of subproblems.

**Equal solver effort per order.** Every subproblem targets `TARGET_CLUSTER_SIZE` orders
and gets `SOLVER_TIME_LIMIT` seconds, so solution quality per order is constant across
cities and graphs.

**The batch wait is charged to the order**, so OR-Tools pays for batching exactly as the
heuristic pays for holding.

---

## 12. Built-in assertions

| # | Check |
|---|---|
| 1 | Node sets identical across graphs *(inherited)* |
| 2 | Every edge has `distance > 0` and `time > 0` *(inherited)* |
| 3 | `order_pool_hash` identical across graphs |
| 4 | `time_limits_hash` identical across graphs |
| 5 | Epochs partition the pool |
| 6 | Clusters partition each epoch |
| 7 | Every order appears exactly once in the solution |
| 8 | `loaded_distance_km <= vehicle_distance_km` (deadhead is non-negative) |
| 9 | No vehicle ends a route still carrying |
| 10 | Onboard count never exceeds `VEHICLE_CAPACITY` |
| 11 | `MAX_DELIVERY_TIME` identical across graphs |
| 12 | Every subproblem returned a solution -- no silent `NONE` |

Note that assertion 8 is **not** 03's `vehicle <= package`. A tour deadheads between a
dropoff and the next pickup; a relay vehicle never moves empty.

---

## 13. Differences from the relay simulation

Carry these into any write-up. They are properties of the model, not bugs.

- **No relay handoffs.** A package stays on one vehicle, so `num_vehicle_changes` is 0
  where the relay arms charge `VEHICLE_CHANGE_TIME` per hop. A genuine structural
  advantage of tours over relays. `--stop-service-time` charges per-stop dwell if you want
  to neutralise it.
- **`avg_hops` counts stops, not graph edges.** One leg may span many edges.
- **Vehicles deadhead.** Relay vehicles never move empty.
- **Batch-optimal, not globally optimal** -- per epoch, per cluster.
- **`distinct_vehicles` is inflated**, because clusters do not share vehicles. Read
  `vehicle_distance_km` and `vehicle_trips` as the primary outputs.
- **`CLIQUE` is a true clique only on the mini cities.** GeoDistChicago has 4,961 edges
  against 371,953 for a real 863-node clique (1.3%); GeoDistColumbus 1,676 against 38,503
  (4.4%). The banner warns when density is below 0.999.

---

## 14. Benchmarks behind the defaults

Measured on GeoDistChicago/UMST. These are why the defaults are what they are.

**Python transit callbacks were the bottleneck, not node count.** Insertion heuristics
make millions of arc evaluations, each crossing into Python:

| n orders | Python callback | `RegisterTransitMatrix` |
|---|---|---|
| 500 @ 10s | obj 376,948 | **355,746** |
| 1,000 @ 30s | **no solution** | **693,506** |
| 2,000 @ 60s | no solution | no solution |

**First-solution strategy matters enormously** (n=500, 10s):

| Strategy | Objective |
|---|---|
| `PARALLEL_CHEAPEST_INSERTION` | **376,948** |
| `LOCAL_CHEAPEST_INSERTION` | 544,192 |
| `PATH_CHEAPEST_ARC` | 946,187 |
| `BEST_INSERTION` | no solution |

**Spatial decomposition beats monolithic at equal wall clock** (n=2,000, 240s):

| Approach | Objective | vs monolithic |
|---|---|---|
| Monolithic | 1,274,514 | -- |
| 4 clusters | 1,214,974 | **4.7% better** |
| 8 clusters | 1,175,620 | **7.8% better** |

More decomposition is *better* here, because guided local search gets traction at ~300
orders and barely improves the first solution at ~3,700.

**Epoch sizing.** Arrivals are a Gaussian mixture (`PEAKS=[0.25,0.75]`, `SIGMA=10`) which
over 60 minutes is nearly flat -- the busiest epoch carries only **1.30x** the average.
Measured peak-epoch sizes at load 100 and 120s epochs: Columbus_mini 112, Chicago_mini 154,
GeoDistColumbus 1,269, GeoDistChicago 3,744.

---

## 15. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `heuristic join skipped: 03 ran against pool X but 04 used Y` | Working as designed. Rerun 03 with the `--max-delivery-time` and `--load-per-hotspot` the message prints |
| `ASSERTION 12 FAILED: N subproblem(s) returned no solution` | These already got one automatic retry at 4x the time limit and still failed. Raise `--solver-time-limit` or lower `--target-cluster-size` / `--max-subproblem-orders` |
| `N subproblem(s) found no solution in 15s; retrying at 60s` | Normal and self-healing. With one vehicle per order a large cluster is a large model and the insertion heuristic can run out of budget. Persistent retries mean `--max-subproblem-orders` is too high |
| `recovery_pct` above 100%, rows marked `<-- UNRELIABLE` | 03 left part of the pool undelivered, so its distance is truncated. Compare `*_km_per_delivered`, or rerun 03 with a larger `--grace-time`. See section 1 |
| `WARNING: solo_km != 03 baseline_veh_km at full completion` | Two independent distance computations disagree. This should never fire. Do not trust any distance in the table until it is explained |
| `ASSERTION 3/4/11 FAILED` | Graphs were run under different pool, deadline or radius settings. Rerun them with matching flags |
| `vehicle_distance_km > solo_distance_km` | Should be impossible with `n` vehicles. Check that `--vehicle-fixed-cost` is not large enough to force chaining |
| Capacity-1 run does not match 03's baseline | Pass `--vehicle-fixed-cost 0 --lateness-penalty 0`. A nonzero fixed cost makes the solver chain orders and deadhead between them |
| Success rate near zero | `EPOCH_SECONDS` far larger than the deadlines, so the batch wait alone blows every deadline. Keep it equal to `WAIT_TIME` |
| Solver ignores distance entirely | `LATENESS_PENALTY` too high for a metre-denominated objective. The default 17 means "a minute late costs a kilometre" |
| Runs are slow | Lower `--solver-time-limit`, or raise `--n-workers`. Subproblems are embarrassingly parallel |
| Multiprocessing errors on Windows | Use `--n-workers 1` to run serially in-process and get a clean traceback |
| `routes.json` is enormous | `--save-routes false`. `orders.csv` keeps the per-order detail |
