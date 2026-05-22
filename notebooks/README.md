# Notebooks

2 notebook training quan trọng — nguồn gốc của các winner trong `checkpoints/`. Chạy được trên Kaggle/Colab (có sẵn `!pip install neat-python`).

## `01_baseline_training.ipynb` → checkpoints/baseline/

PCGNN baseline reproduction theo paper Beukman 2022 (Table 9, Maze column).

| Hyperparam | Giá trị |
|---|---|
| Map size | 14×14 |
| MAPS_PER_GENOME | 12 |
| MAX_GEN | 100 |
| POP_SIZE | 50 |
| NOVELTY_W (solv/intra/inter) | 0.202 / 0.399 / 0.399 |
| SEEDS | [0, 1, 2, 3, 4] |
| LAMBDA_ARCHIVE | 1 (mỗi gen lưu 1 map vào archive) |
| Inputs | 12 (8 ctx + 4 noise), 1 output, sigmoid |

**Output:** 5 file `neat_winner_seed{0,1,2,3,4}.pkl` → đã rename và move sang `checkpoints/baseline/`.

## `02_improved_training.ipynb` → checkpoints/improved/

Bản cải tiến cho thesis. Khác baseline ở 3 điểm chính:

1. **MAP-Elites bonus** — chia không gian (path_norm × wall_ratio) thành 4×4=16 bins, thưởng generator phủ nhiều ô (chống mode collapse).
2. **Branching fitness** — phạt map dạng sọc đơn điệu, thưởng nhánh chữ T/+.
3. **Trọng số rebalanced** — nhẹ solvability (0.20), nặng inter-novelty (0.32) + intra-novelty (0.33) + MAP-Elites (0.05) + branch (0.10).

| Hyperparam | Giá trị |
|---|---|
| Map size | 14×14 |
| MAPS_PER_GENOME | 24 (×2 so với baseline) |
| MAX_GEN | 200 (×2 so với baseline) |
| POP_SIZE | 50 |
| NOVELTY_W | solv=0.20, intra=0.33, inter=0.32, me=0.05, branch=0.10 |
| SEEDS | [0, 1, 2, 3, 4] |
| MAP-Elites bins | `path_norm: 0.5, 1.0, 1.5` × `wall_ratio: 0.2, 0.4, 0.6` |
| Network type | `RecurrentNetwork` (feed_forward=False) — KHÁC baseline |
| Inputs | 12 (cùng baseline) |

**Output:** `inctyseed0.pkl` → đã move sang `checkpoints/improved/`.

> Đây là checkpoint chính của thesis Chương 4. Generator inference được port sang Python script: `pcgnn_genmap_metrics.py::generate_level` (giữ EXACT cùng pipeline).

## Notebook cũ — `archive/notebooks_old/`

Các iteration trung gian/thất bại đã chuyển sang `archive/notebooks_old/` để giữ history:

- `improve.ipynb` — improved v1 (MAPS=6, GEN=200, weights paper) — early
- `improve2.ipynb` — improved v2 (MAPS=24, GEN=200, weights ×0.9, single seed)
- `notebook31c58b84de.ipynb` + `(2)` — Kaggle export gốc trước khi rename
- `wrong-prob.ipynb` — thử map 30×50 (vượt training size), sinh không hợp lệ

Có thể xóa nếu cần dọn dung lượng.
