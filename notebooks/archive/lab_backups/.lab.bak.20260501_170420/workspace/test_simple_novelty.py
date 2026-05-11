"""Smoke test: simplified PCGNN (solvability + intra + inter novelty)"""
import sys, os, random, time
sys.path.insert(0, r"D:\Project\PCGNN\src")
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import neat
from collections import deque
import heapq, tempfile

WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3
MAX_HEIGHT = MAX_WIDTH = 14
MAPS_PER_GENOME = 8
MAX_GEN = 10
POP_SIZE = 30
CONTEXT_SIZE = 1
CTX_TILES = (2*CONTEXT_SIZE+1)**2 - 1
NUM_RANDOM_INPUTS = 4
NUM_INPUTS = CTX_TILES + NUM_RANDOM_INPUTS
NUM_OUTPUTS = 1
NOVELTY_K = 5
NOVELTY_W_SOLVABLE = 0.202
NOVELTY_W_INTER    = 0.399
NOVELTY_W_INTRA    = 0.399
ARCHIVE_MAX_SIZE   = 100
PERTURB_SIZE = 0.1565
SEED = 0

random.seed(SEED); np.random.seed(SEED)


def generate_level(net, map_h=14, map_w=14, perturb=True):
    if hasattr(net, 'reset'): net.reset()
    half = CONTEXT_SIZE
    padded = np.full((map_h + 2*half, map_w + 2*half), -1.0, dtype=float)
    noise = [random.gauss(0, 1) for _ in range(NUM_RANDOM_INPUTS)]
    for r in range(half, map_h + half):
        for c in range(half, map_w + half):
            ctx = []
            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    if dr == 0 and dc == 0: continue
                    ctx.append(padded[r+dr, c+dc])
            inputs = ctx + noise
            if perturb:
                inputs = [x + random.gauss(0, PERTURB_SIZE) for x in inputs]
            out = net.activate(inputs)[0]
            padded[r, c] = 1.0 if out > 0.5 else 0.0
    level = padded[half:half+map_h, half:half+map_w].astype(int)
    if level[0, 0] == FLOOR: level[0, 0] = PLAYER
    if level[map_h-1, map_w-1] == FLOOR: level[map_h-1, map_w-1] = ENEMY
    return level


_level_cache = {}
_bc_cache = {}

def _astar_internal(level, start, end):
    h, w = level.shape
    counter = 0
    open_set = [(abs(start[0]-end[0])+abs(start[1]-end[1]), counter, start)]
    came_from = {start: None}; g_score = {start: 0}; visited = set()
    while open_set:
        _, _, cur = heapq.heappop(open_set)
        if cur in visited: continue
        visited.add(cur)
        if cur == end:
            path, node = [], end
            while node is not None: path.append(node); node = came_from[node]
            return list(reversed(path)), len(visited)
        r, c = cur
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and (nr,nc) not in visited and level[nr,nc] != WALL:
                ng = g_score[cur] + 1
                if ng < g_score.get((nr,nc), 1e9):
                    came_from[(nr,nc)] = cur; g_score[(nr,nc)] = ng; counter += 1
                    heapq.heappush(open_set, (ng+abs(nr-end[0])+abs(nc-end[1]), counter, (nr,nc)))
    return None, len(visited)

def _compute(level):
    key = level.tobytes()
    if key in _level_cache: return _level_cache[key]
    ps = list(zip(*np.where(level == PLAYER)))
    es = list(zip(*np.where(level == ENEMY)))
    if not ps or not es:
        result = (None, 0, False, None)
    else:
        s = ps[0]
        e = min(es, key=lambda x: abs(x[0]-s[0])+abs(x[1]-s[1]))
        path, con = _astar_internal(level, s, e)
        solv = path is not None and len(path) > 1
        result = (path, con, solv, (len(path)-1) if solv else None)
    _level_cache[key] = result
    return result

def is_solvable(l): return _compute(l)[2]
def shortest_path_bfs(l): return _compute(l)[3]
def _get_astar_result(l): p,c,_,_ = _compute(l); return p,c

def astar_difficulty(level):
    path, num_considered = _get_astar_result(level)
    if path is None: return 0.0
    return max(num_considered - len(path), 0) / max(1, int((level != WALL).sum()) - len(path))

def behavior_characterization(level):
    h, w = level.shape
    walkable = {FLOOR, PLAYER, ENEMY}
    wall_dens = float(np.sum(level[1:-1,1:-1]==WALL) / max(1, level[1:-1,1:-1].size))
    sp = shortest_path_bfs(level)
    path_norm = min(1.0, (sp/(h+w)) if sp else 0.0)
    dead, branch, total = 0, 0, 0
    for r in range(h):
        for c in range(w):
            if level[r,c] not in walkable: continue
            total += 1
            nb = sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                     if 0<=r+dr<h and 0<=c+dc<w and level[r+dr,c+dc] in walkable)
            if nb == 1: dead += 1
            elif nb >= 3: branch += 1
    total = max(1, total)
    vis = np.zeros((h,w), dtype=bool); regions = 0
    for r0 in range(h):
        for c0 in range(w):
            if level[r0,c0] in walkable and not vis[r0,c0]:
                regions += 1; q = deque([(r0,c0)]); vis[r0,c0] = True
                while q:
                    r,c = q.popleft()
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = r+dr,c+dc
                        if 0<=nr<h and 0<=nc<w and not vis[nr,nc] and level[nr,nc] in walkable:
                            vis[nr,nc]=True; q.append((nr,nc))
    path, _ = _get_astar_result(level)
    turns_norm = 0.0
    if path and len(path) >= 3:
        t = sum(1 for i in range(1, len(path)-1)
                if (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]) !=
                   (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]))
        turns_norm = min(1.0, t / max(1, len(path)-2))
    return np.array([wall_dens, path_norm, dead/total, branch/total,
                     min(1.0, regions/10.0), min(1.0, astar_difficulty(level)), turns_norm])

def _bc_cached(level):
    key = level.tobytes()
    if key not in _bc_cache: _bc_cache[key] = behavior_characterization(level)
    return _bc_cache[key]

def bc_distance(a, b):
    w = np.array([1,1,1,1,1,1,7.84])
    d = _bc_cached(a) - _bc_cached(b)
    return float(np.sqrt(np.sum(w*d*d)) / np.sqrt(w.sum()))

def intra_novelty_score(levels, k=2):
    if len(levels) < 2: return 0.0
    scores = []
    for i, li in enumerate(levels):
        dists = sorted(bc_distance(li, lj) for j,lj in enumerate(levels) if j != i)
        scores.append(np.mean(dists[:min(k, len(dists))]))
    return float(np.mean(scores))

def _gen_dist(li, lj):
    n = min(len(li), len(lj))
    return float(np.mean([bc_distance(li[k], lj[k]) for k in range(n)])) if n else 0.0

def inter_novelty_score(levels, archive, k=NOVELTY_K):
    if not archive: return 0.0
    dists = sorted(_gen_dist(levels, ref) for ref in archive)
    return float(np.mean(dists[:min(k, len(dists))])) if dists else 0.0

def solvability_fitness(levels):
    return float(np.mean([float(is_solvable(l)) for l in levels]))

novelty_archive = []

def eval_genomes(genomes, config):
    global novelty_archive
    genome_levels = {}
    for gid, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        genome_levels[gid] = [generate_level(net) for _ in range(MAPS_PER_GENOME)]
    for gid, genome in genomes:
        levels = genome_levels[gid]
        f_solve = solvability_fitness(levels)
        f_intra = intra_novelty_score(levels, k=min(NOVELTY_K, len(levels)-1))
        f_inter = inter_novelty_score(levels, novelty_archive) if novelty_archive else 0.0
        genome.fitness = (NOVELTY_W_SOLVABLE*f_solve + NOVELTY_W_INTRA*f_intra + NOVELTY_W_INTER*f_inter)
    sorted_g = sorted(genomes, key=lambda x: x[1].fitness or 0, reverse=True)
    for gid, genome in sorted_g[:5]:
        novelty_archive.append(genome_levels[gid])
    if len(novelty_archive) > ARCHIVE_MAX_SIZE:
        novelty_archive = novelty_archive[-ARCHIVE_MAX_SIZE:]


cfg_text = f"""
[NEAT]
fitness_criterion = max
fitness_threshold = 999999
pop_size = {POP_SIZE}
reset_on_extinction = False
no_fitness_termination = True
[DefaultGenome]
num_inputs = {NUM_INPUTS}
num_outputs = {NUM_OUTPUTS}
num_hidden = 0
feed_forward = False
initial_connection = full_direct
node_add_prob = 0.3
node_delete_prob = 0.1
conn_add_prob = 0.3
conn_delete_prob = 0.1
activation_default = sigmoid
activation_mutate_rate = 0.1
activation_options = sigmoid
aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_init_type = gaussian
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1
response_init_mean = 1.0
response_init_stdev = 0.0
response_init_type = gaussian
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_init_type = gaussian
weight_max_value = 30.0
weight_min_value = -30.0
weight_mutate_power = 0.5
weight_mutate_rate = 0.8
weight_replace_rate = 0.1
enabled_default = True
enabled_mutate_rate = 0.02
enabled_rate_to_true_add = 0.0
enabled_rate_to_false_add = 0.0
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
single_structural_mutation = False
structural_mutation_surer = default
[DefaultSpeciesSet]
compatibility_threshold = 3.0
[DefaultStagnation]
species_fitness_func = max
max_stagnation = 15
species_elitism = 2
[DefaultReproduction]
elitism = 2
survival_threshold = 0.2
min_species_size = 2
"""

with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write(cfg_text); cfg_path = f.name

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path)
os.unlink(cfg_path)

gen_log = []

class Reporter(neat.reporting.BaseReporter):
    def post_evaluate(self, config, population, species, best):
        fits = [g.fitness for g in population.values() if g.fitness is not None]
        net = neat.nn.RecurrentNetwork.create(best, config)
        sample = [generate_level(net) for _ in range(16)]
        sol   = solvability_fitness(sample)
        intra = intra_novelty_score(sample, k=5)
        inter = inter_novelty_score(sample, novelty_archive) if novelty_archive else 0.0
        g = len(gen_log)
        gen_log.append(dict(g=g, best=max(fits), sol=sol, intra=intra, inter=inter))
        print(f"  gen {g:2d}/{MAX_GEN}  best={max(fits):.3f}  sol={sol:.2f}"
              f"  intra={intra:.3f}  inter={inter:.3f}  archive={len(novelty_archive)}", flush=True)

pop = neat.Population(config)
pop.add_reporter(Reporter())
t0 = time.time()
winner = pop.run(eval_genomes, MAX_GEN)
elapsed = time.time() - t0

print(f"\nSmoke test done in {elapsed:.0f}s")
net = neat.nn.RecurrentNetwork.create(winner, config)
sample = [generate_level(net) for _ in range(20)]
print(f"  solve={solvability_fitness(sample):.2f}  intra={intra_novelty_score(sample,k=5):.3f}")
print(f"  Winner: nodes={len(winner.nodes)}, conns={len(winner.connections)}")

chars = {WALL:"#", FLOOR:".", PLAYER:"P", ENEMY:"E"}
print("\nSample map:")
for row in sample[0]:
    print("".join(chars.get(t, "?") for t in row))
