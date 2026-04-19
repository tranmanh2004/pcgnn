# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PCGNN** (Procedural Content Generation with NEAT and Novelty) is a Python 3.9 research project that combines NeuroEvolution of Augmenting Topologies (NEAT) with Novelty Search to procedurally generate game levels (Maze and Mario). The core contribution is a PCG method using NEAT where novelty search replaces traditional fitness, along with new A*-based diversity/difficulty metrics.

## Setup

```bash
conda create -n pcgnn python=3.9
conda activate pcgnn
pip install -r pcgnn_requirements.txt
```

## Running Code

**Always use `./run.sh` from the `src/` directory** — never call `python` directly. The wrapper sets `PYTHONPATH` to include `external/gym-pcgrl`.

```bash
cd src

# Generate levels (Mario)
./run.sh main/main.py --method noveltyneat --game mario --command generate --width 114 --height 14

# Play a level (human)
./run.sh main/main.py --game mario --command play-human --filename test_level.txt

# Play a level (AI agent)
./run.sh main/main.py --game mario --command play-agent --filename test_level.txt

# Run a specific experiment file
./run.sh runs/proper_experiments/v100_maze/v105_run_best_neat.py
```

## Reproducing Experiments (3-step pipeline)

```bash
cd src/pipelines
bash reproduce_full.sh       # Step 1: Run all experiments (DirectGA + NoveltyNEAT)
bash analyse_all.sh          # Step 2: Recalculate metrics
bash finalise_analysis.sh    # Step 3: Generate figures and tables
```

Experiments use SLURM on a cluster. Results are pickled to `src/results/`.

## Architecture

### Core Training Pipeline

```
Experiment(config)
  └─ NoveltyNeatPCG(game, level_generator, fitness_fn, neat_config)
       └─ NEAT population evolution:
            For each generation:
              - Each genome → NeatLevelGenerator → N levels
              - fitness = w1*Solvability + w2*NoveltyIntraGenerator + w3*NoveltyMetric
            Best network saved to results/
```

### Key Module Relationships

- **`src/novelty_neat/novelty_neat.py`** — `NoveltyNeatPCG`, the main PCG class. Orchestrates NEAT evolution using `neat-python`.
- **`src/novelty_neat/generation.py`** — `NeatLevelGenerator` (abstract). Implementations: `GenerateMazeLevelsUsingTiling`, `GenerateGeneralLevelUsingTiling` (Mario), `GenerateMazeLevelUsingOnePass`. The tiling approach iterates each grid cell: network takes 8 surrounding tiles + random inputs → predicts center tile.
- **`src/novelty_neat/fitness/fitness.py`** — `NeatFitnessFunction` base. `CombinedFitness` sums weighted components.
- **`src/novelty_neat/novelty/novelty_metric.py`** — `NoveltyMetric` (between-generator diversity via archive) and `NoveltyIntraGenerator` (within-generator diversity). Uses K-nearest neighbor distance.
- **`src/games/`** — `Game`, `Level` base classes. `MarioLevel` (14×114 grid, 7 tile types), `MazeLevel` (binary grid). `MarioGame` delegates solvability to a Java runner (`java_runner.py` → `external/Mario-AI-Framework/`).
- **`src/metrics/`** — Evaluation metrics post-training. `AStarDiversityAndDifficultyMetric` compares A* trajectories. `SolvabilityMetric` checks path existence.
- **`src/experiments/experiment.py`** — Orchestrates train → evaluate → serialize. `Config` holds seeds, game, method, hyperparameters. Results pickled with compression.
- **`src/baselines/ga/`** — `GeneticAlgorithmPCG` direct GA baseline for comparison.
- **`src/runs/proper_experiments/`** — Experiment definitions: `v100_maze/`, `v200_mario/`, `v300_metrics/`. Each version contains config files (NEAT topology, mutation rates, speciation params) and run scripts.

### Fitness Computation Detail

```
levels[i][j] = j-th level from i-th network in the population

Novelty(i)        = avg distance from levels[i] to K nearest neighbors in archive
IntraNovelty(i)   = avg pairwise distance between levels[i][j] and levels[i][k]
Solvability(i)    = fraction of levels[i] that are solvable

Fitness(i) = w1*Novelty + w2*Solvability + w3*IntraNovelty
```

Distance metrics supported: Hamming, image hashing (ImageHash), Jensen-Shannon divergence, visual diversity.

### External Dependencies

- **`external/gym-pcgrl/`** — PCG RL environment (must be on `PYTHONPATH` via `run.sh`)
- **`external/Mario-AI-Framework/`** — Java Mario simulator, called via subprocess from `src/games/mario/java_runner.py`
- **`external/horn_metrics/`** — HORN framework metrics for level evaluation

### Experiment Tracking

WandB is used for online tracking via `src/experiments/logger.py`. Results also saved locally as compressed pickles in `src/results/`.
