"""
E1 — NEAT + Recursive Backtracker maze generation.
NEAT network biases direction selection during DFS maze carving.
Maps are always solvable by construction (perfect maze).
Fitness = mean path tortuosity + intra-generator diversity.
No MAP-Elites, no stripe penalty, no complex weights.
"""
import sys, os, random, time
sys.path.insert(0, r"D:\Project\PCGNN\src")

import numpy as np
import neat

WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3
HEIGHT, WIDTH = 14, 14
MAPS_PER_GENOME = 8
MAX_GEN = 20
POP_SIZE = 50
SEED = 0

# Rooms at odd interior positions: [1,3,5,7,9,11] x [1,3,5,7,9,11] = 6x6 = 36 rooms
ROOM_ROWS = list(range(1, HEIGHT - 1, 2))
ROOM_COLS = list(range(1, WIDTH - 1, 2))
ROOM_SET  = {(r, c) for r in ROOM_ROWS for c in ROOM_COLS}

# NEAT: 6 noise inputs + 2 position inputs = 8 inputs, 4 direction outputs (U/D/L/R)
NUM_INPUTS  = 8
NUM_OUTPUTS = 4
DIR_IDX = {(-2,0):0, (2,0):1, (0,-2):2, (0,2):3}

random.seed(SEED); np.random.seed(SEED)


def generate_level(net):
    """Recursive backtracker maze with NEAT-biased direction selection."""
    maze = np.zeros((HEIGHT, WIDTH), dtype=int)

    visited = set()
    noise = [random.gauss(0, 1) for _ in range(6)]

    def get_neighbors(r, c):
        ns = []
        for dr, dc in [(-2,0),(2,0),(0,-2),(0,2)]:
            nr, nc = r+dr, c+dc
            if (nr, nc) in ROOM_SET and (nr, nc) not in visited:
                ns.append((nr, nc, dr, dc))
        return ns

    sr, sc = ROOM_ROWS[0], ROOM_COLS[0]
    stack = [(sr, sc)]
    visited.add((sr, sc))
    maze[sr, sc] = FLOOR

    while stack:
        r, c = stack[-1]
        ns = get_neighbors(r, c)
        if ns:
            inp = [r / (HEIGHT-1), c / (WIDTH-1)] + noise
            out = net.activate(inp)
            weights = [max(1e-3, out[DIR_IDX[(dr, dc)]]) for _, _, dr, dc in ns]
            nr, nc, dr, dc = random.choices(ns, weights=weights)[0]
            maze[r + dr//2, c + dc//2] = FLOOR   # carve wall between rooms
            maze[nr, nc] = FLOOR
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

    maze[ROOM_ROWS[0],  ROOM_COLS[0]]  = PLAYER
    maze[ROOM_ROWS[-1], ROOM_COLS[-1]] = ENEMY
    return maze


# ── Metrics ────────────────────────────────────────────────────────────────
import heapq

def astar_path_length(level):
    h, w = level.shape
    ps = list(zip(*np.where(level == PLAYER)))
    es = list(zip(*np.where(level == ENEMY)))
    if not ps or not es: return None
    start, goal = ps[0], es[0]
    open_q = [(0, start)]
    g = {start: 0}
    while open_q:
        _, cur = heapq.heappop(open_q)
        if cur == goal: return g[goal]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (cur[0]+dr, cur[1]+dc)
            if 0<=nb[0]<h and 0<=nb[1]<w and level[nb]!=WALL:
                ng = g[cur] + 1
                if ng < g.get(nb, 1e9):
                    g[nb] = ng
                    heapq.heappush(open_q, (ng, nb))
    return None

def tortuosity(level):
    plen = astar_path_length(level)
    if plen is None: return 0.0
    ps = list(zip(*np.where(level == PLAYER)))
    es = list(zip(*np.where(level == ENEMY)))
    if not ps or not es: return 0.0
    manhattan = abs(ps[0][0]-es[0][0]) + abs(ps[0][1]-es[0][1])
    return plen / manhattan if manhattan > 0 else 0.0

def visual_div(a, b):
    def strip(l):
        l = l.copy(); l[l==PLAYER]=FLOOR; l[l==ENEMY]=FLOOR; return l
    return float(np.mean(strip(a) != strip(b)))

def intra_diversity(levels, k=3):
    if len(levels) < 2: return 0.0
    scores = []
    for i, a in enumerate(levels):
        dists = sorted(visual_div(a, levels[j]) for j in range(len(levels)) if j != i)
        scores.append(np.mean(dists[:k]))
    return float(np.mean(scores))


# ── Fitness ─────────────────────────────────────────────────────────────────
def eval_genomes(genomes, config):
    for gid, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        levels = [generate_level(net) for _ in range(MAPS_PER_GENOME)]
        mean_tort = float(np.mean([tortuosity(l) for l in levels]))
        div       = intra_diversity(levels)
        # Normalize tortuosity: target ~2.0, cap at 3.0
        tort_score = min(1.0, max(0.0, (mean_tort - 1.0) / 2.0))
        genome.fitness = 0.6 * tort_score + 0.4 * div


# ── NEAT config ─────────────────────────────────────────────────────────────
import tempfile
cfg_text = f"""
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
node_add_prob          = 0.4
node_delete_prob       = 0.2
conn_add_prob          = 0.5
conn_delete_prob       = 0.2
activation_default     = tanh
activation_mutate_rate = 0.1
activation_options     = tanh sigmoid relu
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
weight_replace_rate     = 0.1
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
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
"""

with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write(cfg_text); cfg_path = f.name

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path)
os.unlink(cfg_path)

pop = neat.Population(config)
t0  = time.time()

class Reporter(neat.reporting.BaseReporter):
    def post_evaluate(self, config, population, species, best):
        g = len(gen_log)
        fits = [gn.fitness for gn in population.values() if gn.fitness is not None]
        net = neat.nn.FeedForwardNetwork.create(best, config)
        sample = [generate_level(net) for _ in range(16)]
        mean_t = float(np.mean([tortuosity(l) for l in sample]))
        div    = intra_diversity(sample)
        gen_log.append({"g":g,"best":max(fits),"tort":mean_t,"div":div})
        print(f"  gen {g:2d}/{MAX_GEN}  best={max(fits):.3f}  "
              f"tort={mean_t:.3f}  div={div:.3f}", flush=True)

gen_log = []
pop.add_reporter(Reporter())
winner = pop.run(eval_genomes, MAX_GEN)
elapsed = time.time() - t0

net = neat.nn.FeedForwardNetwork.create(winner, config)
sample = [generate_level(net) for _ in range(50)]
mean_t = float(np.mean([tortuosity(l) for l in sample]))
div    = intra_diversity(sample, k=5)

print(f"\nDone in {elapsed:.0f}s")
print(f"  Winner: nodes={len(winner.nodes)}, conns={len(winner.connections)}")
print(f"  Tortuosity : {mean_t:.4f}  (1.0=straight, 2.0=twice manhattan)")
print(f"  Diversity  : {div:.4f}")

with open(".lab/workspace/test_winner_e1.pkl","wb") as f:
    import pickle; pickle.dump(winner, f)
print("  Saved: .lab/workspace/test_winner_e1.pkl")

# Show a sample map
print("\nSample map (winner, 1 map):")
lvl = generate_level(net)
chars = {WALL:"#", FLOOR:".", PLAYER:"P", ENEMY:"E"}
for row in lvl:
    print("".join(chars[t] for t in row))
