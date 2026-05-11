"""
Evaluate a saved winner genome using RecurrentNetwork + ME11 metrics.
Usage: python eval_winner.py <pkl_path> [n_maps]
"""
import sys, os, pickle, random
import numpy as np

PKL_PATH = sys.argv[1] if len(sys.argv) > 1 else "neat_winner_seed0.pkl"
N_MAPS   = int(sys.argv[2]) if len(sys.argv) > 2 else 100

# ── load winner ────────────────────────────────────────────
with open(PKL_PATH, "rb") as f:
    winner = pickle.load(f)

print(f"Loaded: {PKL_PATH}")
print(f"Genome nodes: {len(winner.nodes)}  connections: {len(winner.connections)}")

# ── inline constants (match improve2.ipynb) ─────────────────
WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3
MIN_H = MAX_H = MIN_W = MAX_W = 14
CONTEXT_SIZE = 1
CTX_TILES = (2*CONTEXT_SIZE+1)**2 - 1   # 8
NUM_RANDOM_INPUTS = 4
NUM_INPUTS  = CTX_TILES + NUM_RANDOM_INPUTS  # 12
NUM_OUTPUTS = 1
PERTURB_SIZE = 0.1565
POP_SIZE = 50

# ── build NEAT config (recurrent) ──────────────────────────
import neat, tempfile

config_str = f"""
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
feed_forward           = False
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

with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write(config_str)
    cfg_path = f.name

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path)
os.unlink(cfg_path)

net = neat.nn.RecurrentNetwork.create(winner, config)

# ── generate_level (matches notebook) ──────────────────────
def generate_level():
    net.reset()
    half = CONTEXT_SIZE
    map_h = map_w = 14
    padded = np.full((map_h + 2*half, map_w + 2*half), -1.0)
    noise  = [random.gauss(0, 1) for _ in range(NUM_RANDOM_INPUTS)]
    for r in range(half, map_h + half):
        for c in range(half, map_w + half):
            ctx = []
            for dr in range(-half, half+1):
                for dc in range(-half, half+1):
                    if dr == 0 and dc == 0: continue
                    ctx.append(padded[r+dr, c+dc])
            inputs = [x + random.gauss(0, PERTURB_SIZE) for x in ctx + noise]
            out = net.activate(inputs)[0]
            padded[r, c] = 1.0 if out > 0.5 else 0.0
    level = padded[half:half+map_h, half:half+map_w].astype(int)
    if level[0, 0] == FLOOR:  level[0, 0] = PLAYER
    if level[-1,-1] == FLOOR: level[-1,-1] = ENEMY
    return level

# ── metrics (from run_barrier.py logic) ────────────────────
from collections import deque
import heapq, gzip

def is_solvable(level):
    ps = list(zip(*np.where(level == PLAYER)))
    es = set(zip(*np.where(level == ENEMY)))
    if not ps or not es: return False
    q, vis = deque([ps[0]]), {ps[0]}
    while q:
        r, c = q.popleft()
        if (r,c) in es: return True
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr,c+dc
            h,w = level.shape
            if 0<=nr<h and 0<=nc<w and (nr,nc) not in vis and level[nr,nc] != WALL:
                vis.add((nr,nc)); q.append((nr,nc))
    return False

def astar(level):
    h,w = level.shape
    ps = list(zip(*np.where(level==PLAYER)))
    es = list(zip(*np.where(level==ENEMY)))
    if not ps or not es: return None, 0
    start = ps[0]; goal = es[0]
    open_q = [(abs(start[0]-goal[0])+abs(start[1]-goal[1]), 0, start)]
    came = {}; g = {start:0}; vis = set(); cnt = 0
    while open_q:
        _,cost,cur = heapq.heappop(open_q)
        if cur in vis: continue
        vis.add(cur); cnt += 1
        if cur == goal:
            path = []
            while cur in came: path.append(cur); cur = came[cur]
            path.append(start)
            return list(reversed(path)), cnt
        r,c = cur
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr,c+dc
            if 0<=nr<h and 0<=nc<w and (nr,nc) not in vis and level[nr,nc]!=WALL:
                ng = cost+1
                if ng < g.get((nr,nc), 1e9):
                    came[(nr,nc)] = cur; g[(nr,nc)] = ng; cnt += 1
                    heapq.heappush(open_q,(ng+abs(nr-goal[0])+abs(nc-goal[1]),ng,(nr,nc)))
    return None, cnt

def count_turns(path):
    if path is None or len(path) < 3: return 0
    turns = 0
    for i in range(1, len(path)-1):
        d1 = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        d2 = (path[i+1][0]-path[i][0],  path[i+1][1]-path[i][1])
        if d1 != d2: turns += 1
    return turns

def dir_balance(level):
    h, w = level.shape
    floor = {FLOOR, PLAYER, ENEMY}
    h_t = v_t = 0
    for r in range(h):
        for c in range(w):
            if level[r,c] in floor:
                if c+1 < w and level[r,c+1] in floor: h_t += 1
                if r+1 < h and level[r+1,c] in floor: v_t += 1
    total = h_t + v_t
    if total == 0: return 0.0
    return 1.0 - abs(h_t - v_t) / total

def astar_traj_dist(p1, p2):
    if p1 is None and p2 is None: return 0.0
    if p1 is None or p2 is None: return 1.0
    s1, s2 = set(p1), set(p2)
    return len(s1.symmetric_difference(s2)) / max(len(s1|s2), 1)

def visual_div(a, b):
    def strip(l):
        l = l.copy(); l[l==PLAYER]=FLOOR; l[l==ENEMY]=FLOOR; return l
    return float(np.mean(strip(a).flatten() != strip(b).flatten()))

# ── generate maps ───────────────────────────────────────────
random.seed(42); np.random.seed(42)
print(f"\nGenerating {N_MAPS} maps...", flush=True)
maps = [generate_level() for _ in range(N_MAPS)]

# ── compute metrics ─────────────────────────────────────────
solved   = [is_solvable(m) for m in maps]
paths    = [astar(m)[0] for m in maps]
turns    = [count_turns(p) for p in paths]
dirs     = [dir_balance(m) for m in maps]

sol_rate = np.mean(solved)
# high tortuosity: solved maps with turns >= 1.5 * manhattan
def tortuosity(level, path):
    if path is None: return 0.0
    ps = list(zip(*np.where(level==PLAYER)))
    es = list(zip(*np.where(level==ENEMY)))
    if not ps or not es: return 0.0
    manhattan = abs(ps[0][0]-es[0][0]) + abs(ps[0][1]-es[0][1])
    if manhattan == 0: return 0.0
    return len(path) / manhattan

tortuosities = [tortuosity(maps[i], paths[i]) for i in range(N_MAPS)]
ht_rate = np.mean([t >= 1.5 for t in tortuosities if tortuosities[maps.index(maps[i])] > 0
                   ] if False else [t >= 1.5 for t in tortuosities])
# simpler:
ht_rate = np.mean([tortuosity(maps[i], paths[i]) >= 1.5 for i in range(N_MAPS) if paths[i] is not None])

mean_turns   = np.mean([t for t in turns if t > 0] or [0])
mean_dir     = np.mean(dirs)
compound_v3  = sol_rate * ht_rate * mean_dir

# pairwise diversity (sample 200 pairs)
rng = np.random.default_rng(42)
all_pairs = [(i,j) for i in range(N_MAPS) for j in range(i+1, N_MAPS)]
pairs = [all_pairs[k] for k in rng.choice(len(all_pairs), min(200,len(all_pairs)), replace=False)]
astar_divs = [astar_traj_dist(paths[i], paths[j]) for i,j in pairs]
vis_divs   = [visual_div(maps[i], maps[j]) for i,j in pairs]

print(f"\n{'='*52}")
print(f"  Model: {os.path.basename(PKL_PATH)}")
print(f"  Maps : {N_MAPS}  |  Nodes: {len(winner.nodes)}  Conns: {len(winner.connections)}")
print(f"{'='*52}")
print(f"  Solvability      : {sol_rate:.4f}  ({int(sol_rate*N_MAPS)}/{N_MAPS})")
print(f"  HT rate (tort>=1.5): {ht_rate:.4f}")
print(f"  Dir balance      : {mean_dir:.4f}")
print(f"  compound_v3      : {compound_v3:.4f}   [= sol × ht × dir]")
print(f"  Mean turns       : {mean_turns:.2f}")
print(f"  A* Diversity     : {np.mean(astar_divs):.4f}")
print(f"  Visual Diversity : {np.mean(vis_divs):.4f}")
print(f"{'='*52}")
print(f"\n  Baseline B17     : compound_v3=0.1642  sol=0.92  ht=0.22  dir=0.811")
print(f"  D3 recurrent 200g: compound_v3=0.1393  sol=0.92  ht=0.20  dir=0.757")
