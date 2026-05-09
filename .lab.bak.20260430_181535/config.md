# Lab Config — True Non-L Maze Generation (Tortuosity)

**Created:** 2026-04-30
**Branch:** `research/tortuosity-maze`
**Prior best (escape-stripe):** S4 compound_v2=0.7794 — maps vẫn visually L-shaped

## Problem

`non_l_rate` bị fool bởi noise tiles:
- Network tạo L-corridor (top ngang + right dọc) + random floor tiles ở giữa
- A* đi qua noise tiles → đếm được ≥2 turns → metric gọi là "non-L"
- Dominant structure vẫn là L

## Objective

Generate mazes có path thật sự phức tạp:
- tortuosity(map) = path_length(P→E) / manhattan_distance(P, E) > 1.5
- Path dài hơn ít nhất 50% so với đường thẳng → maze thật, không phải L+noise

## Metrics

### Primary
- **name:** `compound_v3`
- **formula:** `solvability × high_tortuosity_rate × dir_balance`
  - `tortuosity = path_length(P→E) / manhattan_distance(P, E)`
  - `high_tortuosity_rate = fraction maps where tortuosity > 1.5`
  - L-path tortuosity ≈ 1.0 (right+down = manhattan). Maze path > 1.5.
- **direction:** higher is better
- **measure:** embedded in `run_tortuosity.py`, output in `result.json["compound_v3"]`

### Secondary (tracked)
- `compound_v2 = solvability × non_l_rate × dir_balance` (old metric)
- `mean_tortuosity`, `high_tortuosity_rate`, `dir_balance`, `non_l_rate`

## Run

- **command:** `python .lab/workspace/run_tortuosity.py <VARIANT> [out_dir]`
- **output:** `.lab/workspace/expT-<variant>/result.json`
- **wall-clock budget:** 15 phút

## Train Params

- MAX_GEN = 50, POP_SIZE = 50, MAPS_PER_GENOME = 8, SEED = 0, 14×14 maze

## Variants Planned

| ID | Description |
|---|---|
| T0 | S4 config (ME11 + dir_balance fitness), đo compound_v3 baseline |
| T1 | S4 + tortuosity_reward fitness (w=0.10, intra 0.10→0.05) |
| T2 | S4 + tortuosity_reward (w=0.20, intra 0.10→0.0) |
| T3 | T-winner + 48-cell archive |
| T4 | tortuosity threshold=2.0 thay vì 1.5 |
| T5 | tortuosity replaces dir_balance fitness |

## Scope

**In:**
- `run_tortuosity.py` (new runner)
- fitness function, archive config

**Out:** NEAT genome config params, external/, MAP size, MAX_GEN/POP_SIZE

## Termination

Infinite — chạy đến khi `compound_v3 ≥ 0.60` HOẶC user dừng.
