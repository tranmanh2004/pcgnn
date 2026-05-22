# PCGNN — Thesis Fork

> Fork phục vụ khóa luận **"The Knight" (VNU-UET)** — pipeline PCGNN sinh map maze cho Unity Roguelike CombatAgent + PPO + Curriculum Learning.
>
> Phần upstream gốc: [PCGNN paper Beukman 2022](https://arxiv.org/abs/2204.06934). Xem [README upstream chi tiết phía dưới](#upstream--pcgnn-original-readme).

---

## 1. Cấu trúc dự án sau khi tổ chức lại

```
pcgnn/
├── README.md                       # File này
├── CLAUDE.md                       # Hướng dẫn cho Claude Code
├── checkpoints/                    # ⭐ Winner genomes (xem checkpoints/README.md)
│   ├── README.md
│   ├── baseline/                   # PCGNN gốc, FeedForwardNetwork
│   │   ├── neat_winner_seed0.pkl   # ← webapp dùng mặc định
│   │   ├── neat_winner_seed1.pkl
│   │   ├── neat_winner_seed2.pkl
│   │   ├── neat_winner_seed3.pkl
│   │   └── neat_winner_seed4.pkl
│   └── improved/                   # Bản cải tiến improve-v2, RecurrentNetwork
│       └── inctyseed0.pkl          # ← webapp dùng mặc định, checkpoint chính của thesis
│
├── webapp/                         # ⭐ Web tool FastAPI + React (xem webapp/README.md)
│   ├── README.md
│   ├── backend/                    # FastAPI port 8765
│   └── frontend/                   # React + Vite port 5173
│
├── pcgnn_genmap_metrics.py         # Generator improved + metrics + percentile_tier
├── pcgnn_maze_paper.py             # Reproduce Beukman 2022 (Bảng 2-4)
├── render_model_map_samples.py     # Render PNG so sánh baseline vs improved
│
├── config-pcgnn.txt                # Config mặc định cho pcgnn_genmap_metrics CLI
├── config_pcgnn_maze.txt           # Config cho pcgnn_maze_paper.py
├── _render_baseline_config.txt     # Config cho baseline (feed_forward=True)
├── _render_improved_config.txt     # Config cho improved (feed_forward=False)
│
├── generated_maps/                 # Output của pcgnn_genmap_metrics.py
├── rendered_map_samples/           # Output của render_model_map_samples.py
├── notebooks/                      # ⭐ 2 notebook training quan trọng (xem notebooks/README.md)
│   ├── 01_baseline_training.ipynb  #   → checkpoints/baseline/neat_winner_seed*.pkl
│   └── 02_improved_training.ipynb  #   → checkpoints/improved/inctyseed0.pkl
├── archive/                        # File backup, duplicate cũ (an toàn để xóa nếu cần dọn)
│   ├── duplicate_pkls/             #   .pkl trùng
│   ├── notebooks_bak/              #   .ipynb.bak*
│   └── notebooks_old/              #   notebook intermediate/thất bại
│
├── src/                            # Codebase PCGNN upstream (NEAT, novelty search, games)
├── doc/                            # Báo cáo + paper gốc
├── results/                        # Pickle results từ training
│
├── pcgnn_requirements.txt          # Deps cho conda env "pcgnn"
├── env.yml, env_pcgrl.yml          # Conda env spec
└── run.sh                          # Wrapper PYTHONPATH cho src/ scripts
```

---

## 2. Setup môi trường (lần đầu)

```powershell
# 1. Tạo & activate conda env
conda create -n pcgnn python=3.9 -y
conda activate pcgnn

# 2. Cài deps Python (pcgnn core + neat-python + numpy + PIL …)
pip install -r pcgnn_requirements.txt

# 3. Cài deps cho webapp backend (FastAPI + uvicorn + pydantic)
pip install -r webapp\backend\requirements.txt

# 4. Cài deps cho webapp frontend (React + Vite)
npm install --prefix webapp\frontend
```

> Yêu cầu: **Python 3.9** (qua conda), **Node 18+** (test với v24.11), **Java 16** (chỉ cần nếu chạy Mario simulation, không cần cho thesis maze).

---

## 3. Chạy web tool (3 chức năng cho thesis)

Cần **2 terminal** chạy song song:

### Terminal 1 — Backend (FastAPI, port 8765)

```powershell
conda activate pcgnn
uvicorn main:app --host 127.0.0.1 --port 8765 --app-dir "C:\Riot Games\pcgnn\webapp\backend"
```

### Terminal 2 — Frontend (Vite dev, port 5173)

```powershell
npm run dev --prefix "C:\Riot Games\pcgnn\webapp\frontend"
```

Mở browser: **http://localhost:5173**

3 tab:
- **Sinh map** — gen N map từ baseline hoặc improved.
- **So sánh baseline vs improved** — bảng metric trung bình + grid SVG side-by-side.
- **Chia map theo độ khó** — Easy/Medium/Hard theo `percentile_tier` (5/5/90 theo thesis), `score_tier`, `range_tier`.

> Chi tiết API + payload xem `webapp/README.md`.

---

## 4. Sinh batch map ra file .txt (đưa vào Unity)

```powershell
conda activate pcgnn

# Sinh 1000 map từ improved winner, phân tier 5/5/90 (config thesis)
python pcgnn_genmap_metrics.py `
    --checkpoint checkpoints/improved/inctyseed0.pkl `
    --config config-pcgnn.txt `
    --out generated_maps/thesis_1000 `
    --count 1000 --seed 0 `
    --percentile-tiers --easy-ratio 0.05 --medium-ratio 0.05
```

Output:
- `generated_maps/thesis_1000/pcgnn_NNN.txt` — map text (`#` wall, `.` floor, `P` player, `E` enemy).
- `generated_maps/thesis_1000/metrics.csv` — toàn bộ chỉ số + 3 cách phân tier.

Copy các map vào Unity `TilemapGenerator` (xem thesis Chương 2.6).

---

## 5. Render ảnh so sánh baseline vs improved

```powershell
conda activate pcgnn
python render_model_map_samples.py
```

Output: `rendered_map_samples/map_baseline_samples.png`, `map_improved_samples.png`, `map_baseline_vs_improved.png` — phục vụ figure cho báo cáo.

---

## 6. Reproduce paper Beukman 2022 (Bảng 2-4)

```powershell
conda activate pcgnn
python pcgnn_maze_paper.py
```

Train từ đầu 150 generation, 50 pop, in metric so sánh với Bảng 3-4 của paper.

---

## 7. Notes quan trọng

- ⚠️ **Đừng đổi tên/xóa file trong `checkpoints/`** — `webapp/backend/generators.py`, `render_model_map_samples.py`, `pcgnn_genmap_metrics.py` hard-code path. Nếu retrain → save vào folder mới (vd `checkpoints/improved_v3/`) và update path.
- ⚠️ **`config-pcgnn.txt` bị rewrite mỗi lần chạy `pcgnn_genmap_metrics.py`** — webapp dùng `_render_improved_config.txt` riêng để tránh phụ thuộc trạng thái này.
- ⚠️ Pipeline maze 14×14 dùng tile encoding: `WALL=0, FLOOR=1, PLAYER=2, ENEMY=3` (cả improved lẫn baseline). Text: `# . P E`.
- Mọi file backup/duplicate cũ đã chuyển sang `archive/` (an toàn để xóa toàn bộ nếu muốn dọn).
- Notebook nghiên cứu giờ ở `notebooks/`. `improve-v222.ipynb` là nguồn gốc của improved generator hiện tại.

---

## 8. Liên kết nhanh

- 📁 [Checkpoint reference](checkpoints/README.md)
- 📁 [Notebook training reference](notebooks/README.md)
- 📁 [Webapp setup & API](webapp/README.md)
- 📄 Báo cáo thesis: `C:\Riot Games\The-Knight\Dự_án_công_nghệ (10)\Graduate_Thesis.pdf`
- 📄 Paper Beukman 2022: `doc/2204.06934v1.pdf`

---
---

# Upstream — PCGNN original README

<details>
<summary>(Click để mở README gốc của Beukman et al.)</summary>

<p align="center">
<a href="https://arxiv.org/abs/2204.06934">Paper</a> &mdash; <a href="https://github.com/Michael-Beukman/PCGNN/blob/main/doc/poster.pdf">Poster</a>
</p>

## About
This repository stores the code for two different projects. Firstly, a procedural content generation approach that combines novelty search and NeuroEvolution of Augmenting Topologies (NEAT). We also investigate two new metrics for evaluating the diversity and difficulty of levels.

## General structure
To run any python file in here, use `./run.sh path/to/python/file` instead of using `python` directly, because otherwise modules are not recognised.

Most code in `src/` can be categorised into 3 main archetypes:
1. **General / Method code** — `novelty_neat/`, `baselines/`, `games/`, `common/`, `metrics/`
2. **Runs / Experiment code** — `experiments/`, `pipelines/`, `runs/`, `slurms/`
3. **Analysis Code** — `analysis/`, `external/`

## Explanation
1. Evolve a neural network using NEAT (with [neat-python](https://github.com/CodeReclaimers/neat-python))
2. The fitness function for each neural network:
   1. Generate N levels per network
   2. Calculate the average **solvability** of these N levels
   3. Calculate **intra-novelty** (how different these N levels are from each other)
   4. Calculate **inter-novelty** (how different these N levels are from other networks' levels)
   5. `Fitness = w1*Solvability + w2*IntraNovelty + w3*Novelty`
3. Update networks using the above fitness & repeat for X generations.

## Entry Points (upstream)

```bash
cd src
./run.sh main/main.py --method noveltyneat --game mario --command generate --width 114 --height 14
./run.sh main/main.py --game mario --command play-human --filename test_level.txt
./run.sh main/main.py --game mario --command play-agent --filename test_level.txt
```

## Reproducing upstream results
3-step pipeline assuming SLURM:
```bash
cd src/pipelines
./reproduce_full.sh
./analyse_all.sh
./finalise_analysis.sh
```

## Citation

```bibtex
@inproceedings{PCGNN,
  title={Procedural Content Generation using Neuroevolution and Novelty Search for Diverse Video Game Levels},
  author={Beukman, Michael and Christopher Cleghorn and James, Steven},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  year={2022}, month={July}
}

@article{a_star_metrics,
  author={Beukman, Michael and James, Steven and Christopher Cleghorn},
  title={Towards Objective Metrics for Procedurally Generated Video Game Levels},
  journal={CoRR}, year={2022},
  url={https://arxiv.org/abs/2201.10334}
}
```

## Acknowledgements
This work is based on the research supported wholly by the National Research Foundation of South Africa (Grant UID 133358).

</details>
