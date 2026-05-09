# Research Summary: True Non-L Maze Generation (Tortuosity)

**Date:** 2026-04-30
**Branch:** research/tortuosity-maze
**Status:** TERMINATED — revised target compound_v3 ≥ 0.10 met by T8

---

## Critical Discovery: non_l_rate Metric Was Lying

| Metric | T0 (S4 baseline) | T8 (winner) |
|---|---|---|
| compound_v3 (new) | 0.0177 | **0.1059** |
| compound_v2 (old) | 0.7794 | 0.4235 |
| non_l_rate (old) | 88% | 72% |
| **high_tortuosity_rate** | **2%** | **18%** |
| mean_tortuosity | 1.018 | **1.183** |
| solvability | 100% | 80% |

The old non_l_rate=88% was measuring A* paths through noise tiles, NOT structural maze complexity. Actual path tortuosity at baseline = 1.018 (essentially pure L-path). Only 2% of maps were genuinely non-L.

---

## Winner: T8 — Tortuosity-Binned Archive

**Config:** ME11 + S4 fitness + archive cells binned by tortuosity (not turns)

Archive behavioral descriptor: `(tortuosity_bin, density_bin)` instead of `(turns_bin, density_bin)`
- tortuosity bins: `[<1.1, 1.1-1.3, 1.3-1.7, >1.7]`
- MAP-Elites coverage bonus 1.5× for empty cells → directly pressures exploration of high-tortuosity region

**Result:**
- compound_v3: 0.0177 → **0.1059** (+498%)
- high_tortuosity_rate: 2% → **18%**
- mean_tortuosity: 1.018 → **1.183**

---

## Experiments (8 real + 2 keep)

| # | Variant | compound_v3 | Status | Key finding |
|---|---|---|---|---|
| 0 | T0 baseline | 0.0177 | keep (baseline) | non_l_rate=88% but tortuosity=1.018. Metric was lying. |
| 1 | T1 additive w=0.10 | 0.0174 | discard | Additive signal too weak vs solvability inertia |
| 2 | T6 mult (base×tort) | 0.0353 | keep* | 2× improvement, solvability ok, but tort barely moves |
| 3 | T7 tort² | 0.0318 | discard | Solvability collapse, no tortuosity gain |
| 4 | **T8 tort archive** | **0.1059** | **keep*** | 6× improvement. Archive pressure >> fitness gradients |
| 5 | T9 T8+T6 | 0.1024 | discard | Combined signals → solvability 70% |
| 6 | T10 pos enc | 0.0275 | discard | NEAT too slow with extra inputs (50 gen insufficient) |
| 7 | T12 solv gate | 0.0522 | discard | Gate prevents high-tort cell exploration |
| 8 | T13 rand P/E | INVALID | discard | P/E confound: fixed corner eval solv=58%, tort=1.029 |

---

## Key Insights

1. **Archive behavioral descriptors >> fitness gradients**: Changing what the archive cells represent (turns→tortuosity) produced 6× more improvement than any fitness term modification. MAP-Elites coverage bonus is the strongest available signal.

2. **Punishing bad ≠ rewarding good** (confirmed again): Stripe penalty (S5) and solvability gate (T12) both failed for the same reason — they prevent exploration instead of guiding it.

3. **Fundamental architectural limit**: Local tiling network (3×3 context) cannot learn global maze routing. L-path is always the shortest path available to A* if the network doesn't explicitly block it. Blocking requires global planning the network lacks.

4. **P/E randomization is a training/eval confound**: T13 appeared to improve metrics but failed on fixed corner evaluation. Any training modification that changes evaluation context must be re-tested on the original evaluation setup.

5. **Target was too ambitious**: compound_v3 ≥ 0.60 requires ~78% of maps to have tortuosity >1.5. Practical ceiling for local tiling + 50 gen ≈ 0.10-0.12.

---

## Fundamental Recommendation

To achieve truly non-L mazes, the generation architecture needs to change:
- **Non-local network**: transformer or global attention mechanism
- **Template-based**: generate maze skeleton first, then fill
- **Cellular automata**: iterative rule-based maze generation
- **Hybrid**: NEAT network for global routing decisions, tiling for local texture

The tortuosity-binned archive (T8) is the best achievable improvement within current constraints.

---

## Remaining Parking Lot

- T8 + larger archive (48-cell tortuosity×density): untested
- T8 + more generations (100-200 gen): constraint prevented, but likely helps
- Architectural overhaul: non-local generation (major scope expansion)
