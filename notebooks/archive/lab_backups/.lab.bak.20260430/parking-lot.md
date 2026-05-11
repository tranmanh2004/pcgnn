# Parking Lot — Deferred Ideas

Ideas to revisit later if main branch stagnates.

## Next phase (sau khi BC tuning xong)

- **MAP-Elites archive** — thay novelty search bằng QD. Archive grid theo (turns, wall_dens, path_len) → mỗi cell giữ 1 best maze. Đảm bảo coverage toàn BC space, không bị trap ở L-niche. Vassiliades & Mouret 2017. Effort: high. Impact: very high.

## BC-related (revisit nếu hiện tại stagnate)

- A* trajectory distance làm fitness chính (hướng B gốc)
- Combined distance: BC + A* trajectory + visual (hướng C gốc)
- Hybrid distance: 0.5×bc + 0.5×astar_trajectory
- Auto-encoder embedding: train CNN trên 1000 random maze, dùng latent làm BC

## Fitness-level (out of current scope)

- Solvability HARD constraint: unsolvable → fitness=0 dù novelty cao
- Direct L-pattern penalty: `fitness -= 0.1 × L_rate`
- Adaptive weights: curriculum (early: solv, late: diversity)
- Multi-objective NSGA-II thay weighted sum

## Training-level

- Larger context window: CONTEXT_SIZE 1→2 (8 tiles → 24 tiles)
- Recurrent generator (LSTM/GRU)
- 2-stage generation: sketch path → fill walls
- Co-evolution với "solver" population
- Curriculum: 6×6 → 10×10 → 14×14

## Other

- Test trên maze 20×20 sau khi 14×14 ổn
- Ablation: MAX_GEN 30 → 50 nếu signal yếu
- Larger population: 50 → 150 (paper gốc)
