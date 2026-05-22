# PCGNN Web Tool

FastAPI + React (Vite, TS) cho 3 chức năng:
- **Sinh map** — gen N map từ baseline (`neat_winner_seed0.pkl`) hoặc improved (`inctyseed0.pkl`).
- **So sánh baseline vs improved** — bảng metric trung bình + map side-by-side.
- **Chia map theo độ khó** — Easy / Medium / Hard theo 3 cách: `percentile_tier` (thesis, mặc định 5/5/90), `score_tier` (ngưỡng cứng), `range_tier` (theo dải metric).

Generator import trực tiếp từ `pcgnn_genmap_metrics.py` (improved) và `render_model_map_samples.py::generate_baseline_level` (baseline) — không copy code để giữ đúng pipeline thesis.

## Chạy

**Backend (port 8765):**
```powershell
conda activate pcgnn
pip install -r webapp\backend\requirements.txt   # lần đầu
uvicorn main:app --host 127.0.0.1 --port 8765 --app-dir webapp\backend
```

**Frontend (port 5173):**
```powershell
cd webapp\frontend
npm install      # lần đầu
npm run dev
```

Mở http://localhost:5173 — Vite proxy `/api/*` về backend tự động.

## Endpoint

| Method | Path | Body |
|---|---|---|
| GET | `/api/health` | — |
| GET | `/api/models` | — |
| POST | `/api/generate` | `{model, count, seed, perturb}` |
| POST | `/api/compare` | `{count, seed, perturb}` |
| POST | `/api/classify` | `{model, count, seed, perturb, easy_ratio, medium_ratio}` |
