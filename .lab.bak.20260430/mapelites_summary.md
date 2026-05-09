# MAP-Elites Research Summary

**Date:** 2026-04-27
**Branch:** `research/bc-vector-l-pattern`
**Total runs:** 12 (ME0-ME3 × v1/v2/v3)
**Winner:** ME3v3 (compound 0.840)

---

## Progression

### v1 — Baseline MAP-Elites (grid 4×5 incl. bad density bins)

| Variant | Non-L | Solv | Diag | Empty | Full | Compound |
|---|---|---|---|---|---|---|
| ME0 | 8% | 100% | 0 | 22 | 8 | 0.032 |
| ME1 | 14% | 86% | 0 | 16 | 1 | 0.080 |
| ME2 | 8% | 96% | 0 | 23 | 6 | 0.032 |
| ME3 | 16% | 94% | 0 | 18 | 3 | 0.087 |

**Problem:** Grid included density<0.15 and >0.60 bins → archive forced to keep near-empty/near-full genomes → evaluation dominated by bad maps.

### v2 — Restricted density grid (4×3, only [0.15-0.60])

| Variant | Non-L | Solv | Diag | Empty | Full | Compound |
|---|---|---|---|---|---|---|
| ME0v2 | 18% | 98% | 2 | 21 | 11 | 0.061 |
| ME1v2 | 30% | 100% | 0 | 8 | 1 | 0.246 |
| ME2v2 | 32% | 100% | 0 | 16 | 5 | 0.186 |
| ME3v2 | 36% | 94% | 0 | 8 | 1 | 0.278 |

**Problem:** Still collecting maps from turns=0-1 cells → L-pattern maps in output. Near-empty persists because individual maps from mean-density-0.15-0.30 genomes can have density<0.10.

### v3 — Quality output (collect only from turns_bin>=1 cells)

| Variant | Non-L | Solv | Diag | Empty | Full | **Compound** |
|---|---|---|---|---|---|---|
| ME0v3 | 46% | 100% | 10 | 18 | 9 | 0.169 |
| ME1v3 | 66% | 100% | 0 | 2 | 0 | **0.634** |
| ME2v3 | 68% | 100% | 5 | 10 | 3 | 0.453 |
| **ME3v3** | **84%** | **100%** | **0** | **0** | **0** | **0.840** ✅ |

---

## Final comparison

| Model | Non-L | Solv | Diag | Empty | Full | **Compound** |
|---|---|---|---|---|---|---|
| Original baseline | 20% | 78% | ? | ? | ? | 0.156 |
| BC winner (exp #5) | 88% | 90% | ~44 | ~34 | ~9 | ~0.105 |
| V10 (overnight) | 70% | 98% | 0 | 1 | 0 | 0.679 |
| **ME3v3** | **84%** | **100%** | **0** | **0** | **0** | **0.840** ✅ |

**ME3v3 +24% compound vs V10, +438% vs original baseline.**

---

## ME3v3 spec (winner)

**Archive grid:** 4 turns bins × 3 density bins = 12 cells
- turns bins: [0-1], [2-4], [5-8], [9+]
- density bins: [0.15-0.30], [0.30-0.45], [0.45-0.60] (good range only)
- Genome with mean_density outside [0.15, 0.60] → not archived, fitness × 0.3

**Fitness (per genome):**
```python
base = 0.50 * f_solve + 0.30 * f_intra + 0.10 * f_path + 0.10 * f_turns
mc_pen = 0.5 * mode_collapse_penalty(levels)   # V10 penalty
genome.fitness = max(0, base - mc_pen) * coverage_bonus
# coverage_bonus: 1.5 (empty cell), 1.2 (improve), 0.8 (no improve), 0.3 (bad density)
```

**Output collection:** Only sample maps from cells with turns_bin >= 1 (turns >= 2).
L-pattern cells (turns=0-1) still exist in archive to guide evolution but excluded from final output.

**Key findings:**
1. Archive coverage 12/12 (100%) — MAP-Elites fills all good cells
2. Zero diagonal stripes, zero near-empty/near-full — penalty does its job
3. Non-L 84% vs V10 70% — turns fitness term directly rewards multi-turn paths
4. Solvability 100% — no regression

---

## Design lessons

1. **Grid must exclude bad density bins** — otherwise archive is forced to keep degenerate genomes
2. **Output ≠ Archive** — collect final maps only from quality cells, use full archive for evolution guidance
3. **Penalty + turns fitness synergize** — penalty kills diagonal/density degenerate, turns fitness pushes non-L up
4. **Coverage bonus is key** — without it NEAT ignores underrepresented cells

---

## Files

- Runner: `.lab/workspace/run_mapelites.py`
- Results: `.lab/workspace/expme-ME{0-3}v{1-3}/`
- Winner output: `.lab/workspace/expme-ME3v3/{maps.pkl, sample.png, result.json}`
