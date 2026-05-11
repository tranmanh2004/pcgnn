# PCGNN folder structure

This project keeps the original paper/source tree in `src/` and separates local experiment artifacts at the repository root.

## Main folders

- `src/`: original source code, baselines, metrics, external paper dependencies.
- `scripts/`: local utility scripts for map generation, paper-style metrics, and model inspection.
- `notebooks/`: active experiment notebooks.
- `notebooks/archive/`: old notebook backups.
- `notebooks/archive/lab_backups/`: old `.lab.bak.*` folders.
- `models/`: trained NEAT/PCGNN genomes (`.pkl`).
- `configs/`: NEAT configuration files.
- `generated_maps/`: generated map batches and metric CSV files.
- `Map/`: selected exported maps used outside PCGNN, kept at the old path for compatibility.
- `outputs/`: generated figures and quick visual outputs.
- `papers/`: referenced papers.
- `doc/`, `results/`: original project documentation/results.

## Current useful commands

Generate maps from the thesis model:

```powershell
conda activate pcgnn
python scripts/pcgnn_genmap_metrics.py --count 1000 --percentile-tiers --auto-spawn --out generated_maps/inctyseed0_improve_v2
```

Export paper-style metrics for `inctyseed0.pkl`:

```powershell
conda activate pcgnn
python scripts/paper_metrics_inctyseed0.py --spawn auto --out generated_maps/inctyseed0_paper_metrics_auto_14x14_100
```

Inspect stored winner models:

```powershell
conda activate pcgnn
python scripts/inspect_pkls.py
```
