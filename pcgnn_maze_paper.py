"""
PCGNN Maze - Paper Reproduction (Beukman et al. 2022, arxiv 2204.06934)

Reproduces the Maze results from Tables 2, 3, 4 using the hyperparameters
specified in Table 9 of the paper.

Run:
    python pcgnn_maze_paper.py

Target (paper PCGNN Maze row):
    Solvability         = 1.00
    Compression Dist    = 0.488 (0.002)
    A* Diversity        = 0.13  (0.17)
    Leniency            = 0.70  (0.08)
    A* Difficulty       = 0.06  (0.08)
"""

from __future__ import annotations

import gzip
import math
import pickle
import random
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import neat
import numpy as np

# ═════════════════════════════════════════════════════════════
# Paper Maze setup (binary tiles, fixed 14x14, fixed start/goal)
# ═════════════════════════════════════════════════════════════
WALL, FLOOR = 1, 0                 # paper convention: 1 = wall, 0 = passable
N_TILE_TYPES = 2
MAP_H, MAP_W = 14, 14
START = (0, 0)
GOAL  = (MAP_H - 1, MAP_W - 1)

# ── Paper Table 9 (Maze column) ──────────────────────────────
CONTEXT_SIZE      = 1     # 3x3 window around the current cell
NUM_RANDOM_INPUTS = 2     # random variables appended to network input
NUM_INPUTS        = (2 * CONTEXT_SIZE + 1) ** 2 - 1 + NUM_RANDOM_INPUTS  # 8 + 2 = 10
NUM_OUTPUTS       = 1     # single scalar thresholded at 0.5

POP_SIZE          = 50
MAX_GEN           = 150
MAPS_PER_GENOME   = 5     # N levels per network per generation

# Fitness weights (paper Table 9): Novelty 0.399, Solvability 0.202, IntraNovelty 0.399
W_NOVELTY     = 0.399
W_SOLVABILITY = 0.202
W_INTRA       = 0.399

NOVELTY_K = 15    # inter-novelty neighbours (paper Table 9)
INTRA_K   = 10    # intra-novelty neighbours (paper Table 9)
LAMBDA    = 0     # archive growth rate (paper Table 9: 0 = never add)

CONFIG_PATH  = "config_pcgnn_maze.txt"
MODEL_PATH   = "pcgnn_maze_winner.pkl"
MAP_TXT_DIR  = "saved_maps_maze"


# ═════════════════════════════════════════════════════════════
# Level generation (tiling, paper's approach)
# ═════════════════════════════════════════════════════════════
def generate_level(net, h: int = MAP_H, w: int = MAP_W) -> np.ndarray:
    """Paper's tiling generator: scan left→right, top→bottom; for each cell
    feed the 8 surrounding tiles + random inputs; threshold the 1 output."""
    half = CONTEXT_SIZE
    # Pad with -1 so the network sees a consistent signal at the border
    padded = np.full((h + 2 * half, w + 2 * half), -1.0, dtype=float)
    padded[half:-half, half:-half] = (np.random.rand(h, w) > 0.5).astype(float)

    rand_input = list(np.random.randn(NUM_RANDOM_INPUTS))

    for r in range(half, h + half):
        for c in range(half, w + half):
            window = padded[r - half:r + half + 1, c - half:c + half + 1].flatten().tolist()
            window.pop(len(window) // 2)                        # drop the center
            out = net.activate(window + rand_input)[0]
            padded[r, c] = 1.0 if out > 0.5 else 0.0

    level = padded[half:-half, half:-half].astype(int)
    # Paper: start and goal cells are always passable
    level[START] = FLOOR
    level[GOAL]  = FLOOR
    return level


# ═════════════════════════════════════════════════════════════
# Maze solvability / trajectory (BFS = A* on unit grid)
# ═════════════════════════════════════════════════════════════
def _bfs(level: np.ndarray, start, goal):
    """Return (path, visited_set). path is None if unsolvable."""
    h, w = level.shape
    if level[start] == WALL or level[goal] == WALL:
        return None, set()
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return list(reversed(path)), set(prev.keys())
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in prev and level[nr, nc] != WALL:
                prev[(nr, nc)] = cur
                q.append((nr, nc))
    return None, set(prev.keys())


def is_solvable(level: np.ndarray) -> bool:
    path, _ = _bfs(level, START, GOAL)
    return path is not None


def reachable_from_start(level: np.ndarray) -> set:
    _, visited = _bfs(level, START, GOAL)
    return visited


# ═════════════════════════════════════════════════════════════
# Distance functions (novelty during training)
# ═════════════════════════════════════════════════════════════
def visual_diversity_reachable(a: np.ndarray, b: np.ndarray) -> float:
    """Paper's "Visual Diversity Reachable": tiles unreachable from the
    start are clamped to WALL, then normalized Hamming distance is taken."""
    aa = a.copy()
    bb = b.copy()
    ra = reachable_from_start(aa)
    rb = reachable_from_start(bb)
    mask_a = np.ones_like(aa, dtype=bool)
    for (r, c) in ra:
        mask_a[r, c] = False
    aa[mask_a] = WALL
    mask_b = np.ones_like(bb, dtype=bool)
    for (r, c) in rb:
        mask_b[r, c] = False
    bb[mask_b] = WALL
    return float(np.mean(aa != bb))


# ═════════════════════════════════════════════════════════════
# Novelty search (intra + inter) — paper fitness
# ═════════════════════════════════════════════════════════════
novelty_archive: list = []   # List[np.ndarray] — populated with λ levels per gen


def intra_novelty(levels) -> float:
    """Mean pairwise distance over a genome's own levels."""
    if len(levels) < 2:
        return 0.0
    ds = []
    for i in range(len(levels)):
        per_level = []
        for j in range(len(levels)):
            if i == j:
                continue
            per_level.append(visual_diversity_reachable(levels[i], levels[j]))
        per_level.sort()
        ds.append(float(np.mean(per_level[:INTRA_K])))
    return float(np.mean(ds))


def inter_novelty(my_levels, reference_levels) -> float:
    """Mean-of-kNN-distance from my_levels to (population ∪ archive) references."""
    if not reference_levels:
        return 0.0
    scores = []
    for lvl in my_levels:
        dists = [visual_diversity_reachable(lvl, r) for r in reference_levels]
        dists.sort()
        scores.append(float(np.mean(dists[:NOVELTY_K])))
    return float(np.mean(scores))


def solvability_fitness(levels) -> float:
    return float(np.mean([1.0 if is_solvable(l) else 0.0 for l in levels]))


def eval_genomes(genomes, config):
    """Fitness = W_SOLV * solvability + W_NOV * inter + W_INTRA * intra."""
    global novelty_archive

    # 1) Generate levels for every genome
    genome_levels = {}
    for gid, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome_levels[gid] = [generate_level(net) for _ in range(MAPS_PER_GENOME)]

    # 2) Fitness
    for gid, genome in genomes:
        mine = genome_levels[gid]
        solv  = solvability_fitness(mine)
        intra = intra_novelty(mine)
        others = [lvl for ogid, lvls in genome_levels.items() if ogid != gid for lvl in lvls]
        inter  = inter_novelty(mine, others + novelty_archive)
        genome.fitness = (
            W_SOLVABILITY * solv +
            W_INTRA       * intra +
            W_NOVELTY     * inter
        )

    # 3) Archive update: paper uses λ random levels per gen (λ=0 → never)
    if LAMBDA > 0:
        all_levels = [lvl for lvls in genome_levels.values() for lvl in lvls]
        random.shuffle(all_levels)
        novelty_archive.extend(all_levels[:LAMBDA])


# ═════════════════════════════════════════════════════════════
# Paper evaluation metrics (Tables 3 & 4)
# ═════════════════════════════════════════════════════════════
def level_to_bytes(level: np.ndarray) -> bytes:
    return level.astype(np.uint8).tobytes()


def ncd(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Compression Distance via gzip (paper's Compression Distance)."""
    ca = len(gzip.compress(level_to_bytes(a)))
    cb = len(gzip.compress(level_to_bytes(b)))
    cab = len(gzip.compress(level_to_bytes(a) + level_to_bytes(b)))
    return (cab - min(ca, cb)) / max(ca, cb)


def mean_pairwise_ncd(levels) -> float:
    if len(levels) < 2:
        return 0.0
    vals = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            vals.append(ncd(levels[i], levels[j]))
    return float(np.mean(vals))


def mean_pairwise_visual_diversity(levels) -> float:
    if len(levels) < 2:
        return 0.0
    vals = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            vals.append(float(np.mean(levels[i] != levels[j])))
    return float(np.mean(vals))


def astar_diversity(levels) -> float:
    """Pairwise trajectory dissimilarity between shortest paths.
    Uses symmetric set difference over trajectory cells, normalized."""
    paths = []
    for lvl in levels:
        p, _ = _bfs(lvl, START, GOAL)
        paths.append(set(p) if p else None)
    if sum(p is not None for p in paths) < 2:
        return 0.0
    vals = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if paths[i] is None or paths[j] is None:
                continue
            union = paths[i] | paths[j]
            if not union:
                continue
            sym_diff = paths[i] ^ paths[j]
            vals.append(len(sym_diff) / len(union))
    return float(np.mean(vals)) if vals else 0.0


def leniency(level: np.ndarray) -> float:
    """Paper's leniency for Maze: fraction of floor cells along the shortest
    path that have 2+ non-wall neighbours (easy = many ways forward)."""
    path, _ = _bfs(level, START, GOAL)
    if path is None:
        return 0.0
    h, w = level.shape
    scores = []
    for (r, c) in path:
        n_open = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and level[nr, nc] != WALL:
                n_open += 1
        scores.append(1.0 if n_open >= 2 else 0.0)   # branching step = lenient
    return float(np.mean(scores))


def astar_difficulty(level: np.ndarray) -> float:
    """Normalised dead-end exploration: (cells_explored - path_len) / cells_possible."""
    path, visited = _bfs(level, START, GOAL)
    if path is None:
        return 0.0
    possible = int(np.sum(level != WALL))
    if possible == 0:
        return 0.0
    return max(0.0, (len(visited) - len(path)) / possible)


# ═════════════════════════════════════════════════════════════
# NEAT config (paper Table 9)
# ═════════════════════════════════════════════════════════════
def write_config():
    cfg = f"""
[NEAT]
fitness_criterion      = max
fitness_threshold      = 999999
pop_size               = {POP_SIZE}
reset_on_extinction    = True
no_fitness_termination = True

[DefaultGenome]
num_inputs             = {NUM_INPUTS}
num_outputs            = {NUM_OUTPUTS}
num_hidden             = 0
feed_forward           = True
initial_connection     = full_direct

node_add_prob          = 0.2
node_delete_prob       = 0.2
conn_add_prob          = 0.5
conn_delete_prob       = 0.5

activation_default     = tanh
activation_mutate_rate = 0.1
activation_options     = tanh sigmoid relu

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean        = 0.0
bias_init_stdev       = 1.0
bias_init_type        = gaussian
bias_max_value        = 30.0
bias_min_value        = -30.0
bias_mutate_power     = 0.5
bias_mutate_rate      = 0.7
bias_replace_rate     = 0.1

response_init_mean    = 1.0
response_init_stdev   = 0.0
response_init_type    = gaussian
response_max_value    = 30.0
response_min_value    = -30.0
response_mutate_power = 0.0
response_mutate_rate  = 0.0
response_replace_rate = 0.0

weight_init_mean      = 0.0
weight_init_stdev     = 1.0
weight_init_type      = gaussian
weight_max_value      = 30.0
weight_min_value      = -30.0
weight_mutate_power   = 0.5
weight_mutate_rate    = 0.8
weight_replace_rate   = 0.1

enabled_default           = True
enabled_mutate_rate       = 0.01
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
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
"""
    Path(CONFIG_PATH).write_text(cfg.strip() + "\n", encoding="utf-8")
    print(f"[config] wrote {CONFIG_PATH}  (inputs={NUM_INPUTS}, outputs={NUM_OUTPUTS})")


# ═════════════════════════════════════════════════════════════
# Reporter with live metrics
# ═════════════════════════════════════════════════════════════
class PaperReporter(neat.reporting.BaseReporter):
    def __init__(self):
        self.generations = []
        self.best_fitness = []
        self.avg_fitness  = []
        self.solvability  = []
        self.t0 = None

    def start_generation(self, generation):
        if self.t0 is None:
            self.t0 = time.time()

    def post_evaluate(self, config, population, species, best_genome):
        gen = len(self.generations)
        self.generations.append(gen)

        fits = [g.fitness for g in population.values() if g.fitness is not None]
        best, avg = max(fits), float(np.mean(fits))
        self.best_fitness.append(best)
        self.avg_fitness.append(avg)

        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        sample = [generate_level(net) for _ in range(MAPS_PER_GENOME)]
        solv = solvability_fitness(sample)
        self.solvability.append(solv)

        elapsed = time.time() - self.t0
        eta = (elapsed / (gen + 1)) * (MAX_GEN - gen - 1) if gen > 0 else 0
        bar_n = int(40 * (gen + 1) / MAX_GEN)
        bar = "█" * bar_n + "░" * (40 - bar_n)
        print(f"\rGen {gen:3d}/{MAX_GEN} |{bar}| "
              f"Best={best:.3f} Avg={avg:.3f} Solv={solv:.2f} ETA={eta/60:.1f}m",
              end="", flush=True)
        if (gen + 1) % 20 == 0:
            print()


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════
def train():
    global novelty_archive
    novelty_archive = []

    write_config()
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )
    pop = neat.Population(config)
    reporter = PaperReporter()
    pop.add_reporter(reporter)
    pop.add_reporter(neat.StatisticsReporter())

    winner = pop.run(eval_genomes, MAX_GEN)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(winner, f)
    print(f"\n[train] saved winner → {MODEL_PATH}")
    return winner, config, reporter


def evaluate(winner, config, n_maps: int = 50):
    print(f"\n[eval] generating {n_maps} levels from winner …")
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    levels = [generate_level(net) for _ in range(n_maps)]

    solv_list  = [is_solvable(l) for l in levels]
    solvables  = [l for l, s in zip(levels, solv_list) if s]

    print("[eval] computing pairwise + per-map metrics …")
    comp_dist = mean_pairwise_ncd(levels)
    astar_div = astar_diversity(levels)
    vis_div   = mean_pairwise_visual_diversity(levels)
    leni      = float(np.mean([leniency(l)         for l in solvables])) if solvables else 0.0
    adiff     = float(np.mean([astar_difficulty(l) for l in solvables])) if solvables else 0.0

    print("\n" + "═" * 56)
    print("  PCGNN Maze — Paper Reproduction Report")
    print("═" * 56)
    print(f"  Maps generated        : {len(levels)}")
    print()
    print("  ── Quality ──────────────────────────────────")
    print(f"  Solvability           : {np.mean(solv_list)*100:.1f}%   (paper 100%)")
    print()
    print("  ── Diversity (Table 3) ──────────────────────")
    print(f"  Compression Distance  : {comp_dist:.3f}   (paper 0.488 ± 0.002)")
    print(f"  A* Diversity          : {astar_div:.3f}   (paper 0.13 ± 0.17)")
    print(f"  Visual Diversity      : {vis_div:.3f}")
    print()
    print("  ── Difficulty (Table 4) ─────────────────────")
    print(f"  Leniency              : {leni:.3f}   (paper 0.70 ± 0.08)")
    print(f"  A* Difficulty         : {adiff:.3f}   (paper 0.06 ± 0.08)")
    print("═" * 56)

    # Save levels
    out = Path(MAP_TXT_DIR)
    out.mkdir(exist_ok=True)
    for i, lvl in enumerate(levels):
        sym = {WALL: "#", FLOOR: "."}
        text = "\n".join("".join(sym[int(t)] for t in row) for row in lvl)
        (out / f"maze_{i:02d}.txt").write_text(text, encoding="utf-8")
    print(f"[eval] saved levels → {out}/")

    return levels


def visualize(levels, n_show: int = 20, save_path: str = "maze_samples.png"):
    cols = 10
    rows = math.ceil(n_show / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axs = np.array(axs).flatten()
    for i in range(n_show):
        lvl = levels[i]
        img = np.ones((*lvl.shape, 3))
        img[lvl == WALL] = [0.15, 0.15, 0.15]
        img[lvl == FLOOR] = [0.95, 0.95, 0.95]
        img[START] = [0.2, 0.8, 0.2]
        img[GOAL]  = [0.9, 0.2, 0.2]
        axs[i].imshow(img, interpolation="nearest")
        axs[i].axis("off")
    for j in range(n_show, len(axs)):
        axs[j].axis("off")
    plt.suptitle("PCGNN Maze — sampled levels (green = start, red = goal)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved → {save_path}")


def plot_training(reporter, save_path: str = "training_curves.png"):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot(reporter.generations, reporter.best_fitness, label="best")
    ax[0].plot(reporter.generations, reporter.avg_fitness,  label="avg", linestyle="--")
    ax[0].set(xlabel="generation", ylabel="fitness", title="NEAT fitness")
    ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(reporter.generations, reporter.solvability, color="#2E7D32")
    ax[1].set(xlabel="generation", ylabel="solvability", title="Best-genome solvability",
              ylim=(-0.05, 1.05))
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved → {save_path}")


if __name__ == "__main__":
    winner, config, reporter = train()
    plot_training(reporter)
    levels = evaluate(winner, config, n_maps=50)
    visualize(levels, n_show=20)
