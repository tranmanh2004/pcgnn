"""
Gen maps từ 5 seeds ME11 → The Knight Txtmap format.

Post-processing:
  - Force P tại (0,0): ép floor, đặt PLAYER (giống paper do_empty_start_goal)
  - Force E tại (H-1,W-1): ép floor, đặt ENEMY
  - Thêm N_ENEMY_EXTRA E ngẫu nhiên trên floor cells nội thất
  - Tính A* difficulty, lọc unsolvable và density ngoài range

Chạy: conda run -n pcgnn python ".lab/workspace/gen_knight_maps.py"
"""
import pickle, random, os, tempfile, sys
from collections import deque
from pathlib import Path
import numpy as np
import neat

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import astar_difficulty, WALL, FLOOR, PLAYER, ENEMY

# ── Config ───────────────────────────────────────────────────
MAP_H         = 14
MAP_W         = 14
N_PER_SEED    = 15       # maps per seed → tổng 75 maps (skip seed thất bại)
N_ENEMY_EXTRA = 3        # E bổ sung → tổng 4 E/map
DENSITY_MIN   = 0.15
DENSITY_MAX   = 0.60
OUT_DIR       = Path("d:/Project/The Knight/Assets/Scripts/Agent/Txtmap")

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
PKL_FILES = [
    (0, PROJECT_ROOT / "neat_winner_seed0 (5).pkl"),
    (1, PROJECT_ROOT / "neat_winner_seed1 (1).pkl"),
    (2, PROJECT_ROOT / "neat_winner_seed2 (1).pkl"),
    (3, PROJECT_ROOT / "neat_winner_seed3 (1).pkl"),
    (4, PROJECT_ROOT / "neat_winner_seed4 (1).pkl"),
]

# ── NEAT constants ────────────────────────────────────────────
CONTEXT_SIZE      = 1
CTX_TILES         = (2*CONTEXT_SIZE+1)**2 - 1
NUM_RANDOM_INPUTS = 4
NUM_INPUTS        = CTX_TILES + NUM_RANDOM_INPUTS
NUM_OUTPUTS       = 1
POP_SIZE          = 50
PERTURB_SIZE      = 0.1565

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
aggregation_default    = sum
aggregation_mutate_rate= 0.0
aggregation_options    = sum
bias_init_mean         = 0.0
bias_init_stdev        = 1.0
bias_init_type         = gaussian
bias_max_value         = 30.0
bias_min_value         = -30.0
bias_mutate_power      = 0.5
bias_mutate_rate       = 0.7
bias_replace_rate      = 0.1
response_init_mean     = 1.0
response_init_stdev    = 0.0
response_init_type     = gaussian
response_max_value     = 30.0
response_min_value     = -30.0
response_mutate_power  = 0.0
response_mutate_rate   = 0.0
response_replace_rate  = 0.0
weight_init_mean       = 0.0
weight_init_stdev      = 1.0
weight_init_type       = gaussian
weight_max_value       = 30.0
weight_min_value       = -30.0
weight_mutate_power    = 0.5
weight_mutate_rate     = 0.8
weight_replace_rate    = 0.15
enabled_default        = True
enabled_mutate_rate    = 0.02
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

with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write(CONFIG_STR)
    cfg_path = f.name
neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path)
os.unlink(cfg_path)

# ── Generate level ────────────────────────────────────────────
def generate_level(net, map_h, map_w):
    half = CONTEXT_SIZE
    padded = np.full((map_h + 2*half, map_w + 2*half), -1.0)
    noise = [random.gauss(0, 1) for _ in range(NUM_RANDOM_INPUTS)]
    for r in range(half, map_h + half):
        for c in range(half, map_w + half):
            ctx = [padded[r+dr, c+dc]
                   for dr in range(-half, half+1)
                   for dc in range(-half, half+1)
                   if not (dr == 0 and dc == 0)]
            inputs = [x + random.gauss(0, PERTURB_SIZE) for x in ctx + noise]
            padded[r, c] = 1.0 if net.activate(inputs)[0] > 0.5 else 0.0
    return padded[half:half+map_h, half:half+map_w].astype(int)

# ── Post-process cho Knight ───────────────────────────────────
def postprocess(level, map_h, map_w):
    # Force P tại (0,0) — giống paper do_empty_start_goal
    level[0, 0] = PLAYER
    # Force E tại góc đối diện
    level[map_h-1, map_w-1] = ENEMY
    # Thêm E ngẫu nhiên trên floor nội thất (tránh 3×3 quanh P)
    excluded = {(r,c) for r in range(min(3,map_h)) for c in range(min(3,map_w))}
    excluded.add((map_h-1, map_w-1))
    candidates = [(r,c) for r in range(map_h) for c in range(map_w)
                  if level[r,c] == FLOOR and (r,c) not in excluded]
    random.shuffle(candidates)
    for r, c in candidates[:N_ENEMY_EXTRA]:
        level[r, c] = ENEMY
    return level

# ── Filter ────────────────────────────────────────────────────
def passes_filter(level, map_h, map_w):
    density = float((level == WALL).sum()) / (map_h * map_w)
    if not (DENSITY_MIN <= density <= DENSITY_MAX):
        return False, 0.0
    diff = astar_difficulty(level)
    if diff <= 0:   # unsolvable
        return False, 0.0
    return True, diff

# ── Map to text ───────────────────────────────────────────────
def map_to_text(level):
    sym = {WALL:"#", FLOOR:".", PLAYER:"P", ENEMY:"E"}
    return "\n".join("".join(sym.get(int(t),".") for t in row) for row in level)

# ── Main ──────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
total_saved = 0
map_counter = 0

for seed_idx, pkl_path in PKL_FILES:
    print(f"\nSeed {seed_idx}: {pkl_path.name}")
    with open(pkl_path, "rb") as f:
        winner = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winner, neat_config)

    saved = attempts = 0
    difficulties = []

    while saved < N_PER_SEED:
        attempts += 1
        if attempts > N_PER_SEED * 20:
            print(f"  [skip] seed {seed_idx} — quá nhiều attempts ({attempts}), có thể model collapse")
            break

        level = generate_level(net, MAP_H, MAP_W)
        level = postprocess(level, MAP_H, MAP_W)
        ok, diff = passes_filter(level, MAP_H, MAP_W)
        if not ok:
            continue

        difficulties.append(diff)
        map_counter += 1
        fname = f"pcgnn_s{seed_idx}_{map_counter:03d}_{MAP_H}x{MAP_W}_d{diff:.2f}.txt"
        (OUT_DIR / fname).write_bytes(map_to_text(level).encode("utf-8"))
        saved += 1

    if difficulties:
        print(f"  Saved {saved}/{N_PER_SEED}  |  attempts={attempts}  |  diff avg={np.mean(difficulties):.2f}  range=[{min(difficulties):.2f},{max(difficulties):.2f}]")
    total_saved += saved

print(f"\n{'='*50}")
print(f"Total maps saved : {total_saved}")
print(f"Output folder    : {OUT_DIR}")
print(f"\nNext: open Unity → assign pcgnn_*.txt vào TilemapGenerator Inspector")
