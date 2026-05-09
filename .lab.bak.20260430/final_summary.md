# Final Research Summary — Mode Collapse Fix

**Period:** 2026-04-26 → 2026-04-27 (overnight)
**Branch:** `research/bc-vector-l-pattern`
**Total experiments:** 18 (#0-5 BC tuning, #6-13 Phase 1, #14-17 Phase 2)
**Winner:** **V10** (Phase 2, exp #16)
**Status:** TARGET MET — autonomous loop terminated successfully

---

## Headline

|                    | Original (improve2 baseline) | BC tuning winner (exp #5) | **V10 final winner** |
| ------------------ | ---------------------------- | ------------------------- | -------------------- |
| non_l_pattern_rate | 0.20                         | 0.88                      | **0.70**             |
| solvability        | 0.78                         | 0.90                      | **0.98**             |
| diagonal stripes   | unknown                      | ~44/100                   | **0/100**            |
| near-empty         | unknown                      | ~34/100                   | **1/100**            |
| near-full          | unknown                      | ~9/100                    | **0/100**            |
| **compound**       | **0.156**                    | **~0.105**                | **0.679** ✅         |

V10 cải tiến **compound +335%** so với BC winner; **+335%** so với original baseline.

---

## V10 spec (winner)

### Behavior Characterization (7-dim, unchanged from BC winner)

```python
[wall_dens, path_norm, dead_norm, branch_norm, regions_norm, astar_diff, turns_norm]
```

### bc_distance (weighted Euclidean, unchanged)

```python
weights = [1, 1, 1, 1, 1, 1, 7.84]  # turns 2.8x effective
```

### **NEW: Fitness penalty** (this is the key change vs BC winner)

```python
# In eval_genomes, after computing base_fitness:
penalty = 0.5 * (diag_score + density_extreme) / 2.0  # combined, weight 0.5
genome.fitness = max(0, base_fitness - penalty)

# Helpers:
def diag_score(level): # fraction of walls with NW-SE diag neighbor wall
def density_extreme(level): # 1.0 at <0.15 or >0.6 wall density, 0 in middle
```

---

## Phase 1 (overnight, V0-V7) — exploration

| Rank | Variant                       | Compound |
| ---- | ----------------------------- | -------- |
| 1    | V3 BC + combined penalty 0.3  | 0.312    |
| 2    | V7 BC 10-dim + light penalty  | 0.142    |
| 3    | V4 entropy dim only           | 0.137    |
| 4    | V6 low turns weight + penalty | 0.136    |
| 5    | V0 control (BC 7-dim winner)  | 0.105    |
| 6    | V1 diagonal penalty alone     | 0.082    |
| 7    | V5 corridor variance          | 0.011    |
| 8    | V2 density penalty alone      | 0.000    |

**Key finding:** Combined penalty (V3) is the only approach that reduces BOTH diagonal stripes AND near-empty maze. Penalty alone (V1, V2) or BC-only (V4, V5) don't work.

## Phase 2 (V8-V11) — refine

| Rank  | Variant                           | Compound     |
| ----- | --------------------------------- | ------------ |
| **1** | **V10 V3 + stronger penalty 0.5** | **0.679** ✅ |
| 2     | V9 V6 + entropy + combined        | 0.302        |
| 3     | V11 inverted (low BC + penalty)   | 0.222        |
| 4     | V8 entropy + corridor             | 0.045        |

**Key finding:** Penalty strength 0.5 vs 0.3 was the winning lever. With strong enough penalty, the L-niche shrinks (28 vs 42 in V3) AND degenerate patterns vanish entirely (0 diagonal, 1 empty, 0 full).

## Top 5 impactful changes

1. **Adding `turns_norm` to BC** (#0→#1, +14pp non_l)
2. **Boosting turns weight to 2.8x** (#1→#5, +54pp non_l)
3. **Combined penalty (diagonal + density)** at 0.3 strength (V0→V3, +207% compound)
4. **Stronger penalty 0.5** (V3→V10, +118% compound)
5. **Penalty target = combined**, not single (V1, V2 alone failed)

## Failed approaches

- **V1 (diagonal penalty alone)**: pushed generator to make near-empty maze instead → empty 55/100
- **V2 (density penalty alone)**: 100% diagonal stripes → diagonal hack to satisfy density
- **V5 (corridor_width_var)**: 97% diagonal → metric correlated wrong with stripes
- **V8 (entropy + corridor combined)**: combining 2 weak signals → 33% diagonal + 38% empty + 31% full
- **V11 (inverted: low BC + penalty)**: penalty alone insufficient with weak BC pressure

## Ranked Compound Across All Phases

```
V10  ████████████████████████████████  0.679  ★ winner
V0_pre BC winner (estimated)            0.156
V3   ████████████████                   0.312
V9   ███████████████                    0.302
V11  ███████████                        0.222
V7   ███████                            0.142
V4   ███████                            0.137
V6   ███████                            0.136
V0   █████                              0.105
V1   ████                               0.082
V8   ██                                 0.045
V5   █                                  0.011
V2   █                                  0.000
```

## Insights & lessons

1. **Reward hacking confirmed**: BC pressure alone (no constraint) caused mode collapse to "easiest path satisfying turns objective" (= diagonal stripes). Validated literature finding.
2. **Combined constraint > single**: Penalizing 1 thing pushes generator to gaming another. Penalize multiple at once for robustness.
3. **Penalty strength matters**: 0.3 → 0.5 single bump gave +118% compound. Should sweep.
4. **BC tuning hit ceiling**: Pure BC modifications (V4-V8) all gave compound < 0.15. Need fitness-level constraint to break ceiling.
5. **Quality-Diversity (MAP-Elites) still recommended for next phase**: Even V10 has 28% L-pattern remaining. QD archive would force coverage explicitly.

## Files

- Per-phase results: `.lab/{overnight,phase2}_summary.{md,json}`
- Per-variant: `.lab/workspace/expv-{V0..V11}-{ovn,phase2}/{maps.pkl, winner.pkl, sample.png, result.json}`
- Logs: `.lab/{overnight,phase2}.log`
- Comparison PNG (all 8 variants Phase 1): `.lab/overnight_comparison.png`
- Source-of-truth notebook: [improve2.ipynb](../improve2.ipynb) (current = exp #5 BC winner, NOT V10 yet)

## Sync action required (awaiting user approval)

To apply V10 to [improve2.ipynb](../improve2.ipynb):

1. Cell 4 — keep current BC 7-dim + weighted bc_distance (unchanged from sync earlier)
2. Cell 4 — ADD helpers `detect_diagonal_stripes()` and `density_extreme_penalty()`
3. Cell 5 — MODIFY `eval_genomes()` to subtract `0.5 × (diag + density_extreme) / 2` from fitness

This is a fitness function change, not just BC. Bigger impact on training. Recommend re-running 5 seeds × full 200 gen before declaring production-ready.

## Next phase candidates

- **MAP-Elites (QD)**: Replace novelty + penalty with QD grid archive
- **Multi-seed validation of V10**: confirm robustness
- **Weight sweep around penalty 0.5**: test 0.4, 0.6, 0.7 to find peak
- **Scale up training**: 200 gen × 24 maps/genome (paper full) instead of rút gọn
