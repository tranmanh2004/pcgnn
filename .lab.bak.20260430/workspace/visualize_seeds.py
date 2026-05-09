"""
Visualize maps from 5 seeds, 10 maps each → .lab/workspace/maps_seeds.png
"""
import pickle, random, os, tempfile, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import neat

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import generate_level, WALL, FLOOR, PLAYER, ENEMY

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NEAT_CONFIG  = PROJECT_ROOT / ".lab" / "workspace" / "exp-5" / "config.txt"
OUT_PATH     = PROJECT_ROOT / ".lab" / "workspace" / "maps_seeds.png"

PKL_FILES = [
    (0, PROJECT_ROOT / "neat_winner_seed0 (5).pkl"),
    (1, PROJECT_ROOT / "neat_winner_seed1 (1).pkl"),
    (2, PROJECT_ROOT / "neat_winner_seed2 (1).pkl"),
    (3, PROJECT_ROOT / "neat_winner_seed3 (1).pkl"),
    (4, PROJECT_ROOT / "neat_winner_seed4 (1).pkl"),
]

N_SHOW = 20   # maps per seed

TILE_COLORS = {
    WALL:   [0.15, 0.15, 0.15],
    FLOOR:  [0.95, 0.92, 0.85],
    PLAYER: [0.20, 0.75, 0.20],
    ENEMY:  [0.85, 0.20, 0.20],
}

config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    str(NEAT_CONFIG),
)

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import count_path_turns

fig, axes = plt.subplots(5, N_SHOW, figsize=(N_SHOW * 1.5, 5 * 1.6))
fig.patch.set_facecolor("#1a1a2e")

for row, (seed_idx, pkl_path) in enumerate(PKL_FILES):
    with open(pkl_path, "rb") as f:
        winner = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    random.seed(seed_idx * 100)
    maps = [generate_level(net) for _ in range(N_SHOW)]

    for col, lvl in enumerate(maps):
        ax = axes[row, col]
        h, w = lvl.shape
        img = np.zeros((h, w, 3))
        for tile, color in TILE_COLORS.items():
            img[lvl == tile] = color
        ax.imshow(img, interpolation="nearest")
        ax.axis("off")

        turns = count_path_turns(lvl)
        is_l  = 0 <= turns <= 1
        label = f"L" if is_l else f"t={turns}"
        color = "#ff4444" if is_l else "#44ff88"
        ax.set_title(label, fontsize=5.5, color=color, pad=1.5)

        if col == 0:
            ax.set_ylabel(f"Seed {seed_idx}", color="white", fontsize=8,
                          rotation=0, labelpad=35, va="center")

    n_l = sum(1 for m in maps if 0 <= count_path_turns(m) <= 1)
    print(f"Seed {seed_idx}: done  L-pattern={n_l}/{N_SHOW}")

plt.suptitle("PCGNN ME11 — 10 maps per seed  (green=P, red=E, dark=wall)",
             color="white", fontsize=11, y=1.01)
plt.tight_layout(pad=0.3)
plt.savefig(OUT_PATH, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved: {OUT_PATH}")
