"""
Experiment runner — extracted from improve2.ipynb with reduced training params.

Each experiment swaps the `behavior_characterization` function (and optionally
`bc_distance`) inside this script. Run params (MAX_GEN, POP_SIZE, etc) are
fixed across experiments to keep results comparable.

Usage:
    python run_experiment.py <exp_id>

Output:
    .lab/workspace/exp-<id>/maps.pkl  (50 final maps)
    .lab/workspace/exp-<id>/result.json  (metrics)
"""
import sys
import os
import json
import time
import pickle
import gzip
import heapq
import random
import math
from pathlib import Path
from collections import deque

import numpy as np
import neat


# =====================================================================
# CONFIG (fixed across experiments)
# =====================================================================
MIN_WIDTH = MAX_WIDTH = 14
MIN_HEIGHT = MAX_HEIGHT = 14

MAPS_PER_GENOME = 8        # rút gọn (full: 24)
MAX_GEN = 30               # rút gọn (full: 200)
POP_SIZE = 50
LAMBDA_ARCHIVE = 1
SEED = 0
N_FINAL_MAPS = 50          # số map sinh ra để evaluate cuối

WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3

NOVELTY_K = 15
NOVELTY_W_SOLVABLE = 0.1818
NOVELTY_W_INTER = 0.3591
NOVELTY_W_INTRA = 0.3591
NOVELTY_W_PATH_DIV = 0.1
ARCHIVE_MAX_SIZE = 500

CONTEXT_SIZE = 1
CTX_TILES = (2 * CONTEXT_SIZE + 1) ** 2 - 1   # 8
NUM_RANDOM_INPUTS = 4
NUM_INPUTS = CTX_TILES + NUM_RANDOM_INPUTS    # 12
NUM_OUTPUTS = 1
PERTURB_SIZE = 0.1565


# =====================================================================
# MAP GENERATION
# =====================================================================
def generate_level(net, map_h=None, map_w=None, perturb=True):
    if map_h is None: map_h = MAX_HEIGHT
    if map_w is None: map_w = MAX_WIDTH
    half = CONTEXT_SIZE
    padded = np.full((map_h + 2 * half, map_w + 2 * half), -1.0, dtype=float)
    noise = [random.gauss(0, 1) for _ in range(NUM_RANDOM_INPUTS)]
    for r in range(half, map_h + half):
        for c in range(half, map_w + half):
            ctx = []
            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    if dr == 0 and dc == 0: continue
                    ctx.append(padded[r + dr, c + dc])
            inputs = ctx + noise
            if perturb:
                inputs = [x + random.gauss(0, PERTURB_SIZE) for x in inputs]
            out = net.activate(inputs)[0]
            padded[r, c] = 1.0 if out > 0.5 else 0.0
    level = padded[half:half + map_h, half:half + map_w].astype(int)
    if level[0, 0] == FLOOR: level[0, 0] = PLAYER
    if level[map_h - 1, map_w - 1] == FLOOR: level[map_h - 1, map_w - 1] = ENEMY
    return level


# =====================================================================
# CORE METRICS (helpers used by BC and measure)
# =====================================================================
def is_solvable(level):
    h, w = level.shape
    ps = list(zip(*np.where(level == PLAYER)))
    es = set(zip(*np.where(level == ENEMY)))
    if not ps or not es: return False
    q, vis = deque([ps[0]]), {ps[0]}
    while q:
        r, c = q.popleft()
        if (r, c) in es: return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and level[nr, nc] != WALL:
                vis.add((nr, nc)); q.append((nr, nc))
    return False


def shortest_path_bfs(level):
    h, w = level.shape
    ps = list(zip(*np.where(level == PLAYER)))
    es = set(zip(*np.where(level == ENEMY)))
    if not ps or not es: return None
    q, vis = deque([(ps[0], 0)]), {ps[0]}
    while q:
        (r, c), d = q.popleft()
        if (r, c) in es: return d
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and level[nr, nc] != WALL:
                vis.add((nr, nc)); q.append(((nr, nc), d + 1))
    return None


def _astar(level, start, end):
    h, w = level.shape
    def heuristic(pos): return abs(pos[0] - end[0]) + abs(pos[1] - end[1])
    counter = 0
    open_set = [(heuristic(start), counter, start)]
    came_from = {start: None}
    g_score = {start: 0}
    visited = set()
    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in visited: continue
        visited.add(current)
        if current == end:
            path = []
            node = end
            while node is not None:
                path.append(node); node = came_from[node]
            return list(reversed(path)), len(visited)
        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and level[nr, nc] != WALL:
                new_g = g_score[current] + 1
                if new_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = new_g
                    counter += 1
                    heapq.heappush(open_set, (new_g + heuristic((nr, nc)), counter, (nr, nc)))
    return None, len(visited)


def _get_astar_result(level):
    ps = list(zip(*np.where(level == PLAYER)))
    es = list(zip(*np.where(level == ENEMY)))
    if not ps or not es: return None, 0
    start = ps[0]
    end = min(es, key=lambda e: abs(e[0] - start[0]) + abs(e[1] - start[1]))
    return _astar(level, start, end)


def astar_difficulty(level):
    path, num_considered = _get_astar_result(level)
    if path is None: return 0.0
    num_passable = max(1, int((level != WALL).sum()) - len(path))
    return max(num_considered - len(path), 0) / num_passable


def _sample_trajectory(traj, N=30):
    if not traj: return [(0, 0)] * N
    return [traj[int(i * len(traj) / N)] for i in range(N)]


def astar_diversity(level_a, level_b, N=30):
    path_a, _ = _get_astar_result(level_a)
    path_b, _ = _get_astar_result(level_b)
    if not path_a: path_a = [(0, 0)]
    if not path_b: path_b = [(0, 0)]
    sa = _sample_trajectory(path_a, N); sb = _sample_trajectory(path_b, N)
    if len(sa) < len(sb): sa += [sa[-1]] * (len(sb) - len(sa))
    elif len(sb) < len(sa): sb += [sb[-1]] * (len(sa) - len(sb))
    h, w = level_a.shape
    norm = w + h
    return float(np.mean([(abs(a[0] - b[0]) + abs(a[1] - b[1])) / norm for a, b in zip(sa, sb)]))


def path_diversity_fitness(levels):
    lengths = [shortest_path_bfs(lvl) for lvl in levels]
    lengths = [l for l in lengths if l is not None]
    if len(lengths) < 2: return 0.0
    max_path = MAX_HEIGHT + MAX_WIDTH
    return float(min(1.0, np.std(lengths) / max_path))


def solvability_fitness(levels):
    return float(np.mean([float(is_solvable(lvl)) for lvl in levels]))


# =====================================================================
# BC VECTOR — *** MODIFY THIS FOR EACH EXPERIMENT ***
# =====================================================================
def behavior_characterization(level):
    """6-dim BC vector — BASELINE from improve2.ipynb cell 4.10"""
    h, w = level.shape
    walkable = {FLOOR, PLAYER, ENEMY}

    interior = level[1:-1, 1:-1]
    wall_dens = float(np.sum(interior == WALL) / max(1, interior.size))

    sp = shortest_path_bfs(level)
    path_norm = (sp / (h + w)) if sp else 0.0
    path_norm = min(1.0, path_norm)

    dead_ends = 0
    branches = 0
    for r in range(h):
        for c in range(w):
            if level[r, c] not in walkable: continue
            nb = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and level[nr, nc] in walkable:
                    nb += 1
            if nb == 1: dead_ends += 1
            elif nb >= 3: branches += 1
    total_walk = max(1, int((level != WALL).sum()))
    dead_norm = dead_ends / total_walk
    branch_norm = branches / total_walk

    visited = np.zeros((h, w), dtype=bool)
    regions = 0
    for r0 in range(h):
        for c0 in range(w):
            if level[r0, c0] in walkable and not visited[r0, c0]:
                regions += 1
                q = deque([(r0, c0)]); visited[r0, c0] = True
                while q:
                    r, c = q.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and level[nr, nc] in walkable:
                            visited[nr, nc] = True; q.append((nr, nc))
    regions_norm = min(1.0, regions / 10.0)

    diff = min(1.0, astar_difficulty(level))

    # H1: add num_turns_norm as 7th dim
    path, _ = _get_astar_result(level)
    if path is None or len(path) < 3:
        turns_norm = 0.0
    else:
        turns = 0
        for i in range(1, len(path) - 1):
            d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            if d1 != d2: turns += 1
        turns_norm = min(1.0, turns / max(1, len(path) - 2))

    return np.array([wall_dens, path_norm, dead_norm, branch_norm, regions_norm, diff, turns_norm], dtype=float)


def bc_distance(level_a, level_b):
    """Weighted Euclidean BC distance. Turns dim (last) weighted 9x → effective 3x in sqrt."""
    bc_a = behavior_characterization(level_a)
    bc_b = behavior_characterization(level_b)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.84])
    diff = bc_a - bc_b
    return float(np.sqrt(np.sum(weights * diff * diff)) / np.sqrt(weights.sum()))


# =====================================================================
# NOVELTY SEARCH
# =====================================================================
novelty_archive = []


def _generator_distance(levels_i, levels_j):
    n = min(len(levels_i), len(levels_j))
    if n == 0: return 0.0
    return float(np.mean([bc_distance(levels_i[k], levels_j[k]) for k in range(n)]))


def intra_novelty_score(levels, k=2):
    if len(levels) < 2: return 0.0
    scores = []
    for i, lvl_i in enumerate(levels):
        dists = sorted(bc_distance(lvl_i, lvl_j) for j, lvl_j in enumerate(levels) if i != j)
        k_eff = min(k, len(dists))
        scores.append(np.mean(dists[:k_eff]))
    return float(np.mean(scores))


def eval_genomes(genomes, config):
    global novelty_archive
    genome_levels = {}
    for gid, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome_levels[gid] = [generate_level(net) for _ in range(MAPS_PER_GENOME)]

    gids = [gid for gid, _ in genomes]
    for gid, genome in genomes:
        levels = genome_levels[gid]
        f_solve = solvability_fitness(levels)
        f_intra = intra_novelty_score(levels, k=10)
        f_path_div = path_diversity_fitness(levels)

        pop_ref = [genome_levels[other] for other in gids if other != gid]
        all_ref = pop_ref + novelty_archive
        if all_ref:
            dists = sorted(_generator_distance(levels, ref) for ref in all_ref)
            k_eff = min(NOVELTY_K, len(dists))
            f_inter = float(np.mean(dists[:k_eff]))
        else:
            f_inter = 0.0
        genome.fitness = (NOVELTY_W_SOLVABLE * f_solve
                          + NOVELTY_W_INTER * f_inter
                          + NOVELTY_W_INTRA * f_intra
                          + NOVELTY_W_PATH_DIV * f_path_div)

    if LAMBDA_ARCHIVE > 0 and len(gids) >= LAMBDA_ARCHIVE:
        chosen = random.sample(gids, LAMBDA_ARCHIVE)
        for cid in chosen:
            novelty_archive.append(genome_levels[cid])
    if len(novelty_archive) > ARCHIVE_MAX_SIZE:
        novelty_archive = novelty_archive[-ARCHIVE_MAX_SIZE:]


# =====================================================================
# NEAT CONFIG
# =====================================================================
NEAT_CONFIG_TEXT = f"""
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


# =====================================================================
# EVALUATION (post-train)
# =====================================================================
def count_path_turns(level):
    """Count direction changes on A* path. Returns -1 if unsolvable."""
    path, _ = _get_astar_result(level)
    if path is None or len(path) < 3: return -1
    turns = 0
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d1 != d2: turns += 1
    return turns


def evaluate_final_maps(maps):
    """Return dict of metrics."""
    n = len(maps)
    turns_per_map = [count_path_turns(m) for m in maps]
    solvable = [t >= 0 for t in turns_per_map]
    # non-L = ≥2 turns AND solvable
    non_l = [t >= 2 for t in turns_per_map]
    non_l_rate = sum(non_l) / n if n > 0 else 0.0

    # secondary: pairwise A* div + difficulty + solvability
    solv_rate = sum(solvable) / n if n > 0 else 0.0
    diff_per = [astar_difficulty(m) for m in maps]
    div_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            div_pairs.append(astar_diversity(maps[i], maps[j]))
    return {
        "non_l_pattern_rate": float(non_l_rate),
        "solvability": float(solv_rate),
        "astar_div": float(np.mean(div_pairs)) if div_pairs else 0.0,
        "astar_diff": float(np.mean(diff_per)),
        "turns_distribution": {
            "mean": float(np.mean([t for t in turns_per_map if t >= 0])) if any(solvable) else 0.0,
            "histogram": [int(sum(1 for t in turns_per_map if t == k)) for k in range(8)] + [int(sum(1 for t in turns_per_map if t >= 8)), int(sum(1 for t in turns_per_map if t == -1))],
            "histogram_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "≥8", "unsolvable"],
        },
    }


# =====================================================================
# MAIN
# =====================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <exp_id>")
        sys.exit(1)
    exp_id = sys.argv[1]
    out_dir = Path(__file__).parent / f"exp-{exp_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    global novelty_archive
    novelty_archive = []

    # Write neat config to temp
    config_path = out_dir / "config.txt"
    config_path.write_text(NEAT_CONFIG_TEXT)
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path)
    )
    pop = neat.Population(config)

    t0 = time.time()
    print(f"[exp-{exp_id}] training {MAX_GEN} gen × {POP_SIZE} pop × {MAPS_PER_GENOME} maps/genome ...", flush=True)
    winner = pop.run(eval_genomes, MAX_GEN)
    train_secs = time.time() - t0
    print(f"[exp-{exp_id}] training done in {train_secs:.1f}s", flush=True)

    # Generate final maps
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    final_maps = [generate_level(net) for _ in range(N_FINAL_MAPS)]
    with open(out_dir / "maps.pkl", "wb") as f:
        pickle.dump(final_maps, f)

    # Evaluate
    metrics = evaluate_final_maps(final_maps)
    metrics["train_seconds"] = train_secs
    metrics["exp_id"] = exp_id
    with open(out_dir / "result.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print primary
    print(f"[exp-{exp_id}] non_l_pattern_rate = {metrics['non_l_pattern_rate']:.4f}")
    print(f"[exp-{exp_id}] solvability        = {metrics['solvability']:.4f}")
    print(f"[exp-{exp_id}] astar_div          = {metrics['astar_div']:.4f}")
    print(f"[exp-{exp_id}] astar_diff         = {metrics['astar_diff']:.4f}")


if __name__ == "__main__":
    main()
