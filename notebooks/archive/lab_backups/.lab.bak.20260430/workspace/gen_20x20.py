"""
Gen maps từ neat_winner_seed0 (4).pkl với kích thước tùy chọn.
Chạy: python .lab/workspace/gen_20x20.py
"""
import pickle, random, io, sys
import numpy as np
import neat

# ── Constants (copy từ improve2.ipynb) ──────────────────────
WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3
CONTEXT_SIZE      = 1
CTX_TILES         = (2*CONTEXT_SIZE+1)**2 - 1   # 8
NUM_RANDOM_INPUTS = 4
NUM_INPUTS        = CTX_TILES + NUM_RANDOM_INPUTS  # 12
NUM_OUTPUTS       = 1
POP_SIZE          = 50
PERTURB_SIZE      = 0.1565

# ── Build NEAT config in-memory ──────────────────────────────
CONFIG_STR = f"""
[NEAT]
fitness_criterion      = max
fitness_threshold      = 999999
pop_size               = {POP_SIZE}
reset_on_extinction    = False
no_fitness_termination = True

[DefaultGenome]
num_inputs             = {NUM_INPUTS}
num_outputs            = {NUM_OUTPUTS}
num_hidden             = 0
feed_forward           = True
initial_connection     = full_direct
node_add_prob          = 0.6
node_delete_prob       = 0.2
conn_add_prob          = 0.5
conn_delete_prob       = 0.3
activation_default     = sigmoid
activation_mutate_rate = 0.15
activation_options     = sigmoid sin gauss
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_init_type          = gaussian
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_init_type      = gaussian
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_init_type        = gaussian
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.15
enabled_default         = True
enabled_mutate_rate     = 0.02
enabled_rate_to_true_add  = 0.0
enabled_rate_to_false_add = 0.0
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
single_structural_mutation = False
structural_mutation_surer  = default

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 3
survival_threshold = 0.2
min_species_size   = 2
"""

import tempfile, os
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write(CONFIG_STR)
    cfg_path = f.name

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path)
os.unlink(cfg_path)

# ── Load genome ──────────────────────────────────────────────
PKL_PATH = "first.pkl"
with open(PKL_PATH, "rb") as f:
    winner = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(winner, config)
print(f"Loaded: {PKL_PATH}")

# ── Generate level ───────────────────────────────────────────
def generate_level(net, map_h=14, map_w=14, perturb=True):
    half = CONTEXT_SIZE
    padded = np.full((map_h + 2*half, map_w + 2*half), -1.0)
    noise = [random.gauss(0, 1) for _ in range(NUM_RANDOM_INPUTS)]
    for r in range(half, map_h + half):
        for c in range(half, map_w + half):
            ctx = [padded[r+dr, c+dc]
                   for dr in range(-half, half+1)
                   for dc in range(-half, half+1)
                   if not (dr == 0 and dc == 0)]
            inputs = ctx + noise
            if perturb:
                inputs = [x + random.gauss(0, PERTURB_SIZE) for x in inputs]
            out = net.activate(inputs)[0]
            padded[r, c] = 1.0 if out > 0.5 else 0.0
    level = padded[half:half+map_h, half:half+map_w].astype(int)
    if level[0, 0] == FLOOR:
        level[0, 0] = PLAYER
    if level[map_h-1, map_w-1] == FLOOR:
        level[map_h-1, map_w-1] = ENEMY
    return level

def map_to_text(level):
    sym = {WALL: "#", FLOOR: ".", PLAYER: "P", ENEMY: "E"}
    return "\n".join("".join(sym.get(int(t), "?") for t in row) for row in level)

# ── Run ──────────────────────────────────────────────────────
H, W, N = 14, 14, 5
print(f"\nGenerating {N} maps at {H}x{W}:\n")
for i in range(N):
    lvl = generate_level(net, map_h=H, map_w=W)
    walls = int((lvl == WALL).sum())
    floor = int((lvl == FLOOR).sum())
    density = walls / (H * W)
    print(f"--- Map {i+1} (density={density:.2f}) ---")
    print(map_to_text(lvl))
    print()
