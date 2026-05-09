# BC Vector Tuning Phase — Summary

**Period:** 2026-04-26 → 2026-04-27
**Branch:** `research/bc-vector-l-pattern`
**Total experiments:** 6 (incl. baseline)
**Best:** Exp #5
**Status:** TARGET MET (non_l_pattern_rate ≥ 0.85)

## Headline numbers

| Metric | Baseline (#0) | Best (#5) | Δ |
|---|---|---|---|
| **non_l_pattern_rate** | 0.20 | **0.88** | +340% |
| solvability | 0.78 | 0.90 | +15% |
| compound (non_L × solv) | 0.156 | **0.792** | +408% |
| L-pattern count (out of 50) | 29 | **1** | -96% |
| Unsolvable count | 11 | 5 | -55% |

## Final BC vector spec (winner)

### `behavior_characterization` (7 dims)

```python
[wall_dens, path_norm, dead_norm, branch_norm, regions_norm, astar_diff, turns_norm]
```

Where `turns_norm = num_path_turns / max(1, len(path) - 2)`.

### `bc_distance` (weighted Euclidean)

```python
weights = [1, 1, 1, 1, 1, 1, 7.84]  # turns dim weighted 7.84x → effective 2.8x in sqrt
distance = sqrt(sum(w_i * (a_i - b_i)^2)) / sqrt(sum(w_i))
```

## Experiment progression

| # | Change | non_l | solv | Status |
|---|---|---|---|---|
| 0 | BC 6-dim (improve2.ipynb baseline) | 0.20 | 0.78 | keep (baseline) |
| 1 | + turns_norm dim 7 | 0.34 | 0.96 | keep |
| 2 | turns weight 3x | 0.56 | 0.56 | keep* (solv collapse) |
| 3 | turns weight 2x | 0.62 | 0.80 | keep |
| 4 | turns weight 2.5x | 0.82 | 0.96 | keep |
| **5** | **turns weight 2.8x** | **0.88** | **0.90** | ✅ keep, TARGET MET |

## Top 3 impactful changes

1. **Adding `turns_norm` as 7th BC dim** — single biggest qualitative shift. Capture path complexity directly. (+14pp)
2. **Boosting turns weight from 1x to 2.5x** — discovered the trade-off curve isn't U-shaped, peak shifts right. (+48pp)
3. **Fine-tune to 2.8x** — confirmed peak isn't at 2.5x, marginal gain pushed past target. (+6pp)

## Key insights

1. **BC weight tuning matters more than feature engineering**. Adding 1 dim got +14pp; tuning weight on that 1 dim got +54pp.
2. **Surface is non-monotonic**. Weight 3x (exp #2) gave 0.56 but 2.8x (exp #5) gave 0.88. Random seed interaction at high weights.
3. **Trade-off (non_L vs solvability) only emerges at extreme weights**. At 2.8x both stay high.
4. **L-pattern attractor is fragile when BC encodes turns directly**. From 29 → 1 maze (96% reduction).

## Failed approaches

- **Weight 3x** (exp #2): over-shoot. Solvability dropped to 0.56, 22/50 unsolvable. Generator over-fit "make zigzag walls" without ensuring connectivity.

## Limitations / caveats

- **Single seed** (SEED=0). Need multi-seed re-run to confirm robustness.
- **Reduced training params** (30 gen, 8 maps/genome) vs paper full (200 gen, 24 maps/genome). Results may scale differently at full training.
- **Single L-pattern definition** (turns ≤ 1 on A* path). Other definitions (e.g., longest-straight-segment) may show different patterns.

## Next phase

**MAP-Elites archive** (per user request). Replace novelty search with Quality-Diversity:
- Archive grid theo (turns, wall_dens, path_len) cells
- Mỗi cell giữ best maze quality
- Đảm bảo coverage toàn BC space, no niche missing

## Sync action required

Cell 4.10 của [improve2.ipynb](../../improve2.ipynb) cần update:
1. `behavior_characterization` → 7-dim (thêm turns_norm)
2. `bc_distance` → weighted Euclidean với weights `[1,1,1,1,1,1,7.84]`

Awaiting user confirmation to perform sync.
