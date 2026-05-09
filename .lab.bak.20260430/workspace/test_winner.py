"""
Test trained NEAT winner — load pkl, generate maps, compute metrics, visualize.
Inference-only — no training.
"""
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import neat
import matplotlib.pyplot as plt

# Reuse all logic from run_experiment.py (same dir)
sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import (
    generate_level, evaluate_final_maps,
    WALL, FLOOR, PLAYER, ENEMY,
)
import random

# ─────────── CONFIG ───────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PKL_PATH = PROJECT_ROOT / "neat_winner_seed0 (1).pkl"
NEAT_CONFIG_PATH = PROJECT_ROOT / ".lab" / "workspace" / "exp-5" / "config.txt"
OUT_DIR = PROJECT_ROOT / ".lab" / "workspace" / "test_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "maps_txt").mkdir(exist_ok=True)

N_MAPS = 50
N_SHOW = 10
SEED = 42  # different from training seed for novelty in inference

TILE_COLORS = {
    WALL:   [0.15, 0.15, 0.15],
    FLOOR:  [0.95, 0.95, 0.95],
    PLAYER: [0.20, 0.80, 0.20],
    ENEMY:  [0.90, 0.20, 0.20],
}

print(f"[test] Loading pkl: {PKL_PATH.name}")
print(f"[test] Using NEAT config: {NEAT_CONFIG_PATH.relative_to(PROJECT_ROOT)}")

# ─────────── LOAD ───────────
with open(PKL_PATH, "rb") as f:
    winner_genome = pickle.load(f)

config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    str(NEAT_CONFIG_PATH),
)
net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

# ─────────── GENERATE ───────────
random.seed(SEED)
np.random.seed(SEED)
print(f"[test] Generating {N_MAPS} maps (seed={SEED})...")
maps = [generate_level(net) for _ in range(N_MAPS)]

# Save raw
with open(OUT_DIR / "maps.pkl", "wb") as f:
    pickle.dump(maps, f)

# Save text format
sym = {WALL: "#", FLOOR: ".", PLAYER: "P", ENEMY: "E"}
for i, lvl in enumerate(maps, 1):
    h, w = lvl.shape
    txt = "\n".join("".join(sym.get(int(t), "?") for t in row) for row in lvl)
    (OUT_DIR / "maps_txt" / f"map_{i:02d}_{h}x{w}.txt").write_text(txt, encoding="utf-8")

# ─────────── METRICS ───────────
print("[test] Computing metrics...")
metrics = evaluate_final_maps(maps)
metrics["pkl_file"] = PKL_PATH.name
metrics["seed"] = SEED
metrics["n_maps"] = N_MAPS

with open(OUT_DIR / "result.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ─────────── VISUALIZE ───────────
print(f"[test] Saving sample.png ({N_SHOW} maps)...")
fig, axs = plt.subplots(1, N_SHOW, figsize=(20, 2))
for j in range(N_SHOW):
    lvl = maps[j]
    h, w = lvl.shape
    img = np.zeros((h, w, 3))
    for tile, color in TILE_COLORS.items():
        img[lvl == tile] = color
    axs[j].imshow(img, interpolation="nearest")
    axs[j].axis("off")
plt.suptitle(
    f"Trained winner — non_l={metrics['non_l_pattern_rate']:.2f}  "
    f"solv={metrics['solvability']:.2f}  "
    f"div={metrics['astar_div']:.3f}  diff={metrics['astar_diff']:.3f}",
    fontsize=10,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "sample.png", dpi=120, bbox_inches="tight")
plt.close()

# ─────────── REPORT ───────────
print()
print(f"non_l_pattern_rate = {metrics['non_l_pattern_rate']:.4f}")
print(f"solvability        = {metrics['solvability']:.4f}")
print(f"astar_div          = {metrics['astar_div']:.4f}")
print(f"astar_diff         = {metrics['astar_diff']:.4f}")
hist = metrics["turns_distribution"]["histogram"]
labels = metrics["turns_distribution"]["histogram_labels"]
print()
print("Turns histogram:")
for l, c in zip(labels, hist):
    bar = "#" * c
    print(f"  {l:>12}: {c:3d} {bar}")
print()
print(f"Output dir: {OUT_DIR}")
