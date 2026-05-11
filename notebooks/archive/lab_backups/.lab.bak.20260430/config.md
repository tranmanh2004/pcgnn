# Lab Config — BC Vector L-Pattern Reduction

**Created:** 2026-04-26
**Branch:** `research/bc-vector-l-pattern`
**Source notebook:** `improve2.ipynb` (cell 4 = scope)

## Objective

Giảm tỉ lệ L-pattern trong maze sinh ra bởi PCGNN bằng cách tinh chỉnh
Behavior Characterization (BC) vector — vector dùng làm distance function
trong novelty search.

## Hypothesis

BC vector hiện tại (6 chiều: wall_dens, path_norm, dead_norm, branch_norm,
regions_norm, astar_diff) chưa capture được sự khác biệt giữa L-shaped path
và path đa-rẽ. Thêm/bỏ/đổi feature trong BC vector có thể tăng áp lực chọn
lọc khỏi L-pattern.

## Metrics

### Primary

- **name:** `non_l_pattern_rate`
- **measure:** `python .lab/workspace/measure.py <maps_dir>` → 1 float
- **direction:** higher is better (0 → toàn L-pattern, 1 → không có L-pattern)
- **definition:** Cho 50 maze cuối, chạy A* PLAYER→ENEMY, đếm số lần đổi hướng (turns) trên path. Maze có ≥2 turns = non-L. Rate = #non-L / 50.

### Secondary (tracked, không drive decision)

- `solvability` — % maze solvable
- `astar_div` — mean pairwise A* trajectory diversity (sampled Manhattan)
- `astar_diff` — mean A* difficulty (node expansion ratio)

## Run

- **command:** `python .lab/workspace/run_experiment.py <exp_id>`
- **output:** `.lab/workspace/exp-<id>/{maps.pkl, result.json}`
- **wall-clock budget:** 10 phút

## Train Params (RÚT GỌN cho exploration)

- MAX_GEN = 30 (full paper: 200)
- POP_SIZE = 50 (giữ nguyên)
- MAPS_PER_GENOME = 8 (full paper: 24)
- SEEDS = [0] (full paper: [0,1,2,3,4])
- 14×14 maze

## Scope

**In:** `behavior_characterization()` và `bc_distance()` trong `.lab/workspace/run_experiment.py`

**Out:** generator code, NEAT config, fitness weights, run command, training params

## Constraints

- Không sửa `external/`
- Không sửa NEAT genome config
- Không đổi MAX_GEN/POP_SIZE/MAPS_PER_GENOME (giữ comparable giữa exp)
- Không thay đổi định nghĩa L-pattern (turns ≤ 1)

## Termination

Infinite — chạy tới khi `non_l_pattern_rate ≥ 0.85` HOẶC user dừng.

## Baseline (exp #0, 2026-04-26)

- **non_l_pattern_rate: 0.20** (10/50 maze có ≥2 turns)
- solvability: 0.78
- astar_div: 0.2932
- astar_diff: 0.4405
- turns histogram: 0:0, 1:29, 2:0, 3:2, 4:2, 5:1, 6:0, 7:1, ≥8:4, unsolvable:11
- train time: 966.7s (~16 phút)

**Key insight:** 29/50 (58%) maze là pure L-pattern (đúng 1 turn). Đây là attractor mạnh cần phá.

## Best so far — TARGET MET ✅

- experiment: **#5** (weighted BC, turns 2.8x)
- non_l_pattern_rate: **0.88** (target 0.85 — exceeded)
- solvability: 0.90
- compound (non_L × solv): **0.792**
- Status: BC phase ready to close, sync to improve2.ipynb

### Compound metric tracking

- baseline: 0.20 × 0.78 ≈ 0.156
- exp #1 (1x): 0.34 × 0.96 ≈ 0.326
- exp #2 (3x): 0.56 × 0.56 ≈ 0.314
- exp #3 (2x): 0.62 × 0.80 ≈ 0.496
- exp #4 (2.5x): **0.82 × 0.96 ≈ 0.787** ← BEST

Surface không U-shape như giả định ban đầu. Peak shift về 2.5x. Exp #2 (3x) có thể outlier.

## Time budget note

Mỗi experiment ~16 phút (lâu hơn dự kiến 10 phút). Sẽ giữ params để các exp comparable.


## Phase 2 winner (overnight run, 2026-04-27)

- Variant: **V3**
- Description: BC + combined (diagonal + density) penalty in fitness
- non_l_pattern_rate: 0.4500
- solvability: 0.8700
- diagonal stripes: 4/100
- compound: 0.3119
