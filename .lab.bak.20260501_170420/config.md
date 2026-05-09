# Lab Config — Sol-HT Balance Research

**Created:** 2026-05-01
**Branch:** `research/sol-ht-balance`
**Prior best (B17):** compound_v3=0.1642, sol=92%, ht_rate=22%, dir_bal=0.811, archive=7/12

## Problem

B17 evolution collapses to solvability:
- f_ht weight = 0.03 (too small)
- Archive descriptor only provides passive isolation, NOT active reproductive advantage
- Genome với ht_rate cao không được chọn làm parent nhiều hơn genome sol cao
- Result: evolution converges đến L-path generators, ht_rate không được maintain

## Objective

Cân bằng sol và ht_rate — cả 2 đều được duy trì trong evolution, không để sol dominate.
Target: beat B17 compound_v3=0.1642, với cả sol≥85% VÀ ht_rate≥25%

## Metrics

### Primary
- **name:** `compound_v3`
- **formula:** `solvability × high_tortuosity_rate × dir_balance`
- **direction:** higher is better
- **measure:** `conda run -n pcgnn python .lab/workspace/run_barrier.py <VARIANT> .lab/workspace/exp-<N> | tail -20`

### Secondary
- `solvability`, `high_tortuosity_rate`, `dir_balance`, `archive_cells`
- Balance check: BOTH sol≥85% AND ht_rate≥25%

## Run

- **command:** `conda run -n pcgnn python .lab/workspace/run_barrier.py <VARIANT> .lab/workspace/exp-<N>`
- **wall-clock budget:** 15 phút per experiment

## Train Params (same as B17 research)

- MAX_GEN=50, POP_SIZE=50, MAPS_PER_GENOME=8, SEED=0, 14×14 maze

## Baseline

- **B0 (= B17 from prev research):** compound_v3=0.1642, sol=92%, ht=22%, dir=0.811

## Termination

User dừng hoặc target compound_v3 ≥ 0.20 với sol≥85% VÀ ht≥25%
