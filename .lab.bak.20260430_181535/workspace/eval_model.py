"""
eval_model.py — evaluate a saved NEAT genome pickle.
Usage: python .lab/workspace/eval_model.py <path_to_pkl> [n_maps]
"""
import sys, pickle, heapq, random
from collections import deque
from pathlib import Path
import numpy as np
import neat

# ── constants ────────────────────────────────────────────
WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3
MAP_H = MAP_W = 14
PERTURB_SIZE = 0.1565
CTX = 1
N_RAND = 4
NUM_INPUTS  = (2*CTX+1)**2 - 1 + N_RAND  # 12
NUM_OUTPUTS = 1
TORTUOSITY_THRESHOLD = 1.5

# ── NEAT config (matches improve2.ipynb) ─────────────────
CONFIG_STR = f"""
[NEAT]
fitness_criterion      = max
fitness_threshold      = 999999
pop_size               = 50
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

# ── generation ───────────────────────────────────────────
def generate_level(net, seed=None):
    if seed is not None:
        random.seed(seed)
    half = CTX
    padded = np.full((MAP_H + 2*half, MAP_W + 2*half), -1.0)
    noise = [random.gauss(0, 1) for _ in range(N_RAND)]
    for r in range(half, MAP_H+half):
        for c in range(half, MAP_W+half):
            ctx_vals = []
            for dr in range(-half, half+1):
                for dc in range(-half, half+1):
                    if dr == 0 and dc == 0: continue
                    ctx_vals.append(padded[r+dr, c+dc])
            inp = [x + random.gauss(0, PERTURB_SIZE) for x in ctx_vals + noise]
            padded[r, c] = 1.0 if net.activate(inp)[0] > 0.5 else 0.0
    level = padded[half:half+MAP_H, half:half+MAP_W].astype(int)
    if level[0, 0] != WALL:   level[0, 0] = PLAYER
    if level[MAP_H-1, MAP_W-1] != WALL: level[MAP_H-1, MAP_W-1] = ENEMY
    return level

# ── A* ───────────────────────────────────────────────────
def _astar(level, start, end):
    h, w = level.shape
    counter = 0
    open_set = [(abs(start[0]-end[0])+abs(start[1]-end[1]), counter, start)]
    came_from = {start: None}
    g = {start: 0}
    vis = set()
    while open_set:
        _, _, cur = heapq.heappop(open_set)
        if cur in vis: continue
        vis.add(cur)
        if cur == end:
            path = []
            node = end
            while node: path.append(node); node = came_from[node]
            return list(reversed(path))
        r, c = cur
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and (nr,nc) not in vis and level[nr,nc] != WALL:
                ng = g[cur] + 1
                if ng < g.get((nr,nc), 1e9):
                    came_from[(nr,nc)] = cur; g[(nr,nc)] = ng; counter += 1
                    heapq.heappush(open_set, (ng+abs(nr-end[0])+abs(nc-end[1]), counter, (nr,nc)))
    return None

def get_path(level):
    ps = list(zip(*np.where(level == PLAYER)))
    es = list(zip(*np.where(level == ENEMY)))
    if not ps or not es: return None
    s = ps[0]; e = min(es, key=lambda x: abs(x[0]-s[0])+abs(x[1]-s[1]))
    return _astar(level, s, e)

# ── metrics ──────────────────────────────────────────────
def is_solvable(level):
    return get_path(level) is not None

def tortuosity(level):
    ps = list(zip(*np.where(level == PLAYER)))
    es = list(zip(*np.where(level == ENEMY)))
    if not ps or not es: return None
    s = ps[0]; e = min(es, key=lambda x: abs(x[0]-s[0])+abs(x[1]-s[1]))
    manhattan = abs(e[0]-s[0]) + abs(e[1]-s[1])
    if manhattan == 0: return None
    path = _astar(level, s, e)
    if path is None or len(path) < 2: return None
    return (len(path)-1) / manhattan

def count_turns(level):
    path = get_path(level)
    if path is None or len(path) < 3: return 0
    turns = 0
    for i in range(1, len(path)-1):
        d1 = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        d2 = (path[i+1][0]-path[i][0],  path[i+1][1]-path[i][1])
        if d1 != d2: turns += 1
    return turns

def dir_balance(level):
    floor = {FLOOR, PLAYER, ENEMY}
    h, w = level.shape
    ht = vt = 0
    for r in range(h):
        for c in range(w):
            if level[r,c] in floor:
                if c+1 < w and level[r,c+1] in floor: ht += 1
                if r+1 < h and level[r+1,c] in floor: vt += 1
    return min(ht,vt)/max(ht,vt) if max(ht,vt) > 0 else 0.0

def map_to_ascii(level):
    sym = {WALL:"█", FLOOR:".", PLAYER:"P", ENEMY:"E"}
    return "\n".join("".join(sym.get(int(t),"?") for t in row) for row in level)

# ── main ─────────────────────────────────────────────────
def main():
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else "neat_winner_seed0.pkl"
    n_maps   = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    print(f"Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        genome = pickle.load(f)

    # write temp config
    cfg_path = Path(".lab/workspace/_eval_config.txt")
    cfg_path.write_text(CONFIG_STR)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, str(cfg_path))

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    print(f"Network: {len(genome.connections)} connections, {len(genome.nodes)} nodes")

    random.seed(42)
    maps = [generate_level(net) for _ in range(n_maps)]

    # ── compute metrics ──────────────────────────────────
    solv_list  = [is_solvable(m) for m in maps]
    tort_list  = [tortuosity(m) for m in maps]
    turns_list = [count_turns(m) for m in maps if is_solvable(m)]
    dir_list   = [dir_balance(m) for m in maps]

    tort_valid = [t for t in tort_list if t is not None]
    solv_rate  = float(np.mean(solv_list))
    mean_tort  = float(np.mean(tort_valid)) if tort_valid else 0.0
    high_tort  = float(np.mean([t > TORTUOSITY_THRESHOLD for t in tort_valid])) if tort_valid else 0.0
    mean_turns = float(np.mean(turns_list)) if turns_list else 0.0
    mean_dir   = float(np.mean(dir_list))
    compound   = solv_rate * high_tort * mean_dir

    # non_l_rate (old metric: turns >= 2)
    non_l = float(np.mean([count_turns(m) >= 2 for m in maps if is_solvable(m)])) if any(solv_list) else 0.0

    print(f"\n{'='*50}")
    print(f"  Model: {Path(pkl_path).name}")
    print(f"  Maps evaluated: {n_maps}")
    print(f"{'='*50}")
    print(f"  solvability      : {solv_rate*100:.1f}%")
    print(f"  mean_tortuosity  : {mean_tort:.3f}  (L-path=1.0, target>1.5)")
    print(f"  high_tort_rate   : {high_tort*100:.1f}%  (tortuosity>{TORTUOSITY_THRESHOLD})")
    print(f"  mean_turns       : {mean_turns:.1f}")
    print(f"  non_l_rate(old)  : {non_l*100:.1f}%  (turns>=2, misleading)")
    print(f"  dir_balance      : {mean_dir:.3f}")
    print(f"  compound_v3      : {compound:.4f}")
    print(f"{'='*50}")

    # tortuosity distribution
    bins = [0, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0, 999]
    labels = ["unsolvable","[1.0,1.1)","[1.1,1.3)","[1.3,1.5)","[1.5,1.7)","[1.7,2.0)",">=2.0"]
    counts = [0] * len(labels)
    counts[0] = sum(1 for t in tort_list if t is None)
    for t in tort_valid:
        for i in range(len(bins)-1):
            if bins[i] <= t < bins[i+1]:
                counts[i+1] += 1; break
    print("\n  Tortuosity distribution:")
    for label, cnt in zip(labels, counts):
        bar = "█" * int(cnt * 30 / n_maps)
        print(f"    {label:12s} {cnt:3d} {bar}")

    # show 5 sample maps
    print("\n  Sample maps (first 5 solvable):")
    shown = 0
    for m in maps:
        if is_solvable(m) and shown < 5:
            t = tortuosity(m)
            print(f"\n  --- tortuosity={t:.3f}, turns={count_turns(m)} ---")
            for line in map_to_ascii(m).split("\n"):
                print(f"  {line}")
            shown += 1
        if shown >= 5: break

if __name__ == "__main__":
    main()
