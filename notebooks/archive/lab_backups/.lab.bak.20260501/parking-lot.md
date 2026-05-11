# Parking Lot

## Tested (cleared)
- 2-pass generation: TESTED (P0) — catastrophic solvability (38%). Input distribution mismatch.
- MAX tortuosity archive descriptor: predicted discard (B4 thought) — same flaw as p75 (B1).
- CONTEXT_SIZE=2 (5×5): TESTED (B5) — improves structure quality but hurts solvability.
- Barrier fitness weight sweep: TESTED (B2 w=0.05) — some archive improvement but solv drops.

## Remaining ideas (for future research sessions)
- Two separate NEAT networks: generator + corrector trained separately with connectivity reward
- Recurrent/attention network with global receptive field (not local 3×3)
- Path-first generation: generate winding path sequence P→E, then fill walls around it
- SEED sweep: test B0 with SEED=1,2,3 to understand variance of 0.1059 baseline
- POP_SIZE=100 with MAX_GEN=100: more training budget (doubles runtime)
