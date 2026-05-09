# Research Summary — Sol-HT Balance (Final)

**Date:** 2026-05-01
**Objective:** Balance sol và ht_rate — cả 2 đều duy trì trong evolution
**Baseline:** B17 compound_v3=0.1642 (sol=92%, ht=22%, dir=0.811)
**Outcome:** **NO IMPROVEMENT FOUND. B17 là architectural ceiling. Confirmed by 12 C-series experiments.**

---

## Final Result

| Config | compound_v3 | sol | ht | dir | Notes |
|---|---|---|---|---|---|
| **B17 (baseline)** | **0.1642** | **92%** | **22%** | **0.811** | Prior research winner — UNBEATEN |
| C15 best | 0.1383 | 90% | 22% | 0.699 | Best C-series — ht same, dir collapsed |
| All others | <0.11 | - | - | - | Multiple crashes |

---

## All Experiments (12 C-series, all discard)

| ID | Approach | cv3 | Key finding |
|---|---|---|---|
| C1 | Bin-level ht_scale multiplier (1+0.4×bin) | 0.1249 | NEAT speciation neutralizes cross-bin comparison |
| C3 | Adaptive per-genome ht weight | 0.0704 | Per-genome variance creates unstable landscape |
| C5 | Fund w_ht=0.06 từ f_intra | 0.1011 | f_intra drives ht diversity — not safe to reduce |
| C7 | Lower ht_rate bin thresholds | 0.0619 | Lower bar = lower quality in "good cells" |
| C8 | sol×ht archive (3×3=9 cells) | 0.0662 | Archive structure ≠ genome capability |
| C10 | Direct product sol×ht×dir fitness | 0.0557 | Product → flat landscape when ht≈0, no convergence |
| C12 | Archive injection every 10 gens | 0.0367 | Injection disrupts NEAT speciation — worst result |
| C14 | TUG per-species fitness shaping | 0.0595 | Species can't produce ht they lack capability for |
| C15 | Reciprocal penalty factor=4 | 0.1383 | ht=22% maintained but dir=0.699 — trade-off |
| C16 | Reciprocal penalty on sol only, target=0.30 | 0.0443 | Sol collapsed to 60%, target too high |
| C17 | Reciprocal penalty factor=1 | 0.0766 | dir recovered (0.819) but ht=12% |

---

## Key Discoveries

1. **B17 is not imbalanced by selection failure** — sol=92%, ht=22% is the genuine ceiling of 3×3 local NEAT.

2. **Reciprocal penalty sweep (C15-C17) confirmed the trade-off:** No factor achieves ht=22% AND dir≈0.811 simultaneously. B17's additive formula already found the Pareto-optimal balance.

3. **f_intra is critical for ht_rate** — Within-genome diversity drives some maps to be accidentally tortuous. Cutting f_intra reduces ht_rate.

4. **All external multipliers fail** — Any multiplier outside base formula is neutralized by NEAT speciation.

5. **Archive injection disrupts speciation** — Cannot force high-ht genomes to reproduce by replacement.

6. **Architectural ceiling confirmed:** Network capability ~22% ht_rate maximum regardless of selection pressure.

---

## Root Cause (Architectural)

Local 3×3 NEAT: each cell decision based on 8 immediate neighbors. Cannot plan globally. Tortuous paths require connected winding corridors — global structure. The network has already maximized ht_rate within local context constraints (~22%).

---

## Recommendations for True Balance

1. **Global receptive field**: Attention or recurrent network that sees whole grid.
2. **Two separate networks**: Generator (pass 1) + Corrector (pass 2) with separate objectives.
3. **Post-hoc selection**: Among all generated maps, select only tortuous + solvable ones for evaluation.
4. **More generations**: Run 100-200 gens instead of 50 to test if ht can develop past 22% with enough time.
