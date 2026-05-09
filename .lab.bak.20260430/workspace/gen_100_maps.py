"""Generate 100 maps from trained winner pkl and save as 10x10 grid PNG."""
import sys
import pickle
import random
from pathlib import Path

import numpy as np
import neat
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import generate_level, WALL, FLOOR, PLAYER, ENEMY

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PKL_PATH = PROJECT_ROOT / "neat_winner_seed0 (1).pkl"
NEAT_CONFIG_PATH = PROJECT_ROOT / ".lab" / "workspace" / "exp-5" / "config.txt"
OUT_PATH = PROJECT_ROOT / ".lab" / "workspace" / "test_output" / "100_maps.png"

N_MAPS = 100
COLS = 10
ROWS = 10
SEED = 42

TILE_COLORS = {
    WALL:   [0.15, 0.15, 0.15],
    FLOOR:  [0.95, 0.95, 0.95],
    PLAYER: [0.20, 0.80, 0.20],
    ENEMY:  [0.90, 0.20, 0.20],
}

print(f"Loading pkl: {PKL_PATH.name}")
with open(PKL_PATH, "rb") as f:
    winner_genome = pickle.load(f)

config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    str(NEAT_CONFIG_PATH),
)
net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

random.seed(SEED)
np.random.seed(SEED)
print(f"Generating {N_MAPS} maps (seed={SEED})...")
maps = [generate_level(net) for _ in range(N_MAPS)]

print(f"Saving 10x10 grid to {OUT_PATH.name}...")
fig, axs = plt.subplots(ROWS, COLS, figsize=(COLS * 1.6, ROWS * 1.6))
for idx in range(N_MAPS):
    r, c = divmod(idx, COLS)
    lvl = maps[idx]
    h, w = lvl.shape
    img = np.zeros((h, w, 3))
    for tile, color in TILE_COLORS.items():
        img[lvl == tile] = color
    axs[r, c].imshow(img, interpolation="nearest")
    axs[r, c].axis("off")
    axs[r, c].set_title(f"#{idx + 1}", fontsize=6, pad=2)

plt.suptitle(f"Trained winner — 100 maps (seed={SEED})", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
plt.close()

print(f"Done: {OUT_PATH}")
