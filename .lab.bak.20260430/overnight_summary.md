# Overnight Run Summary

Generated: 2026-04-27 06:28:14

Variants run: 8


## Comparison Table (sorted by compound metric)

Compound = non_l × solv × (1 − diag%) × (1 − extreme%)


| Rank | Variant | Description | non_l | solv | diag/100 | empty | full | div | diff | Compound |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **V3** | BC + combined (diagonal + density) penalty in fitn | 0.45 | 0.87 | 4 | 15 | 2 | 0.239 | 0.271 | **0.312** |
| 2 | **V7** | BC 10-dim combined + small combined penalty | 0.30 | 1.00 | 11 | 36 | 11 | 0.228 | 0.524 | **0.142** |
| 3 | **V4** | BC 8-dim with spatial_entropy | 0.42 | 0.98 | 10 | 58 | 5 | 0.193 | 0.725 | **0.137** |
| 4 | **V6** | Best-of-both: reduce turns weight to 4 (2x), add c | 0.25 | 0.97 | 21 | 3 | 26 | 0.160 | 0.285 | **0.136** |
| 5 | **V0** | Control: BC 7-dim winner from exp #5 (re-run for s | 0.88 | 0.89 | 52 | 67 | 5 | 0.145 | 0.755 | **0.105** |
| 6 | **V1** | BC + diagonal penalty in fitness | 0.21 | 0.99 | 4 | 55 | 4 | 0.162 | 0.751 | **0.082** |
| 7 | **V5** | BC 8-dim with corridor_width_var | 0.67 | 0.67 | 97 | 10 | 9 | 0.240 | 0.373 | **0.011** |
| 8 | **V2** | BC + density-extreme penalty in fitness | 0.69 | 0.69 | 100 | 9 | 0 | 0.157 | 0.358 | **0.000** |

## Winner: **V3** (compound=0.312)

- Description: BC + combined (diagonal + density) penalty in fitness

- non_l_pattern_rate: 0.4500

- solvability: 0.8700

- diagonal stripes: 4/100 (4%)

- near-empty: 15/100

- near-full: 2/100

- A* difficulty: 0.2709

- pairwise A* diversity: 0.2392


## Turns Histograms

| Variant | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | ≥8 | unsolv |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 0 | 1 | 21 | 9 | 14 | 0 | 4 | 0 | 40 | 11 |
| V1 | 0 | 78 | 6 | 4 | 7 | 3 | 1 | 0 | 0 | 1 |
| V2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 69 | 31 |
| V3 | 0 | 42 | 0 | 15 | 0 | 4 | 0 | 2 | 24 | 13 |
| V4 | 0 | 56 | 1 | 12 | 0 | 8 | 0 | 4 | 17 | 2 |
| V5 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 64 | 33 |
| V6 | 0 | 72 | 6 | 2 | 3 | 3 | 8 | 0 | 3 | 3 |
| V7 | 0 | 70 | 1 | 6 | 2 | 1 | 3 | 7 | 10 | 0 |

## Files

- Raw results: `.lab/overnight_summary.json`

- Comparison image: `.lab/overnight_comparison.png`

- Per-variant: `.lab/workspace/expv-V*-ovn/{maps.pkl, winner.pkl, result.json, sample.png}`

- Progress log: `.lab/overnight.log`
