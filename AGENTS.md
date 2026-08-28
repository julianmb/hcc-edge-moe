# AGENTS.md — hcc-edge-moe

## What this repo is
Heterogeneous Compute Cascade: distributed ~400B MoE inference across TWO
AMD Ryzen AI Max+ 395 "Strix Halo" workstations over USB4.
github.com/julianmb/hcc-edge-moe.

## Gotchas
1. This is a **dual-box** project — many scripts assume a peer host and
   configured USB4 networking; single-box runs will fail at discovery.
2. Model shards/cascade configs reference workshop paths
   (`~/source/halofpx-research/models/...`) on BOTH machines — keep
   both sides in sync before benchmarking.
3. Registry entry `hcc-edge-moe` exists in the workshop's superset
   `config/models.json`, not yet in halofpx's public registry.
4. Commit style: conventional commits; push straight to `main`.
