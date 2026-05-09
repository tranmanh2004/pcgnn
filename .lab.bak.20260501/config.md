# Lab Config — Barrier Archive Research

**Created:** 2026-04-30
**Branch:** `research/barrier-archive`
**Prior best (tortuosity-maze T8):** compound_v3=0.1059, archive=8/12, high_tort=18%

## Problem

T8 archive stuck at 8/12 because:
1. `me_genome_cell` uses MEAN tortuosity — genome needs ALL 24 maps to be high-tort to reach high bins. Even 2 high-tort maps out of 24 average to ~1.04 → stays in bin 0.
2. No fitness signal for barrier creation — network has no gradient toward placing continuous wall rows/cols that force A* to detour.

## Objective

Break past compound_v3=0.1059 ceiling by:
1. Fix archive descriptor: MEAN → 75th percentile tortuosity
2. Add barrier_fitness term to directly reward wall barriers

## Metrics

### Primary
- **name:** `compound_v3`
- **formula:** `solvability × high_tortuosity_rate × dir_balance`
- **direction:** higher is better
- **measure:** `conda run -n pcgnn python .lab/workspace/run_barrier.py <VARIANT> | grep compound_v3 | awk '{print $NF}' | tr -d '[:space:]'`

### Secondary
- `solvability`, `high_tortuosity_rate`, `mean_tortuosity`, `archive_cells`

## Run

- **command:** `conda run -n pcgnn python .lab/workspace/run_barrier.py <VARIANT> [out_dir]`
- **output:** `.lab/workspace/expB-<variant>/result.json`
- **wall-clock budget:** 15 phút

## Train Params

- MAX_GEN=50, POP_SIZE=50, MAPS_PER_GENOME=8, SEED=0, 14×14 maze

## Baseline

- **B0 (T8 re-run):** compound_v3=0.1059 (from tortuosity-maze research exp #4)

## Variants Planned

| ID | Description |
|---|---|
| B0 | T8 baseline re-validation |
| B1 | 75th percentile archive descriptor (instead of mean) |
| B2 | Barrier fitness (w=0.05, intra 0.10→0.07) |
| B3 | B1 + B2 combined |
| B4 | MAX tortuosity archive descriptor |

## Termination

compound_v3 ≥ 0.15 HOẶC user dừng.
