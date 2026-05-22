# Checkpoints

NEAT genome winners dùng để sinh map maze 14×14 cho thesis. Mọi `.pkl` ở đây là output của `pickle.dump(best_genome, ...)` sau quá trình tiến hóa NEAT.

## Cấu trúc

```
checkpoints/
├── baseline/   # PCGNN gốc Beukman 2022, FeedForwardNetwork
└── improved/   # Bản cải tiến improve-v2, RecurrentNetwork
```

## Improved — `improved/inctyseed0.pkl`

**Sinh từ:** notebook `notebooks/02_improved_training.ipynb` (improve-v2 generator + MAP-Elites + branching).

**Đặc trưng:**
- Trained với `feed_forward = False` → load bằng `neat.nn.RecurrentNetwork.create`.
- Config: `_render_improved_config.txt` (12 inputs = 8 context + 4 noise, 1 output, sigmoid/sin/gauss).
- `padded` init = random 0/1 (KHÔNG dùng sentinel -1 như baseline).
- Auto-place PLAYER ở `(0,0)` và ENEMY ở `(h-1,w-1)` nếu là FLOOR.
- Solvability trung bình ~75-100% với seed mặc định.
- Tile encoding output: `WALL=0, FLOOR=1, PLAYER=2, ENEMY=3`.

**Dùng cho:** Web tool (tab improved), `pcgnn_genmap_metrics.py`, `render_model_map_samples.py`. Đây là checkpoint chính của thesis Chapter 4.

## Baseline — `baseline/neat_winner_seed*.pkl`

5 checkpoint cùng training pipeline với 5 seed khác nhau (0-4). `seed0` là cái được webapp + render script sử dụng mặc định.

**Sinh từ:** notebook `notebooks/01_baseline_training.ipynb` — reproduce paper Beukman 2022 (arxiv 2204.06934) với hyperparams Table 9.

**Đặc trưng:**
- Load bằng `neat.nn.FeedForwardNetwork.create` (có **probe-fallback** sang `RecurrentNetwork` nếu output toàn wall — xem `render_model_map_samples.load_baseline_net`).
- Config: `_render_improved_config.txt` (cùng 12 inputs với improved, vì topology giống nhau).
- `padded` init = `-1.0` (sentinel "outside") — KHÁC improved.
- Cùng auto-place P/E ở 2 góc.
- Solvability trung bình ~100% với seed mặc định.

**Dùng cho:** Web tool (tab baseline), so sánh với improved trong thesis Chapter 4.

## Lưu ý quan trọng

- **KHÔNG xóa hoặc rename** các file này — `webapp/backend/generators.py`, `render_model_map_samples.py`, `pcgnn_genmap_metrics.py` hard-code đường dẫn.
- Nếu retrain → save vào folder mới (vd `checkpoints/improved_v3/`) và update path, đừng overwrite winner cũ.
- Duplicate cũ (`Baseline.pkl`, `first.pkl`, `neat_winner_seed0 (5).pkl`) đã chuyển sang `archive/duplicate_pkls/`.
