"""Patch improve2.ipynb: add MAP-Elites (path_len x wall_dens)"""
import json

with open('improve2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Cell 3332459d — Constants ──────────────────────────────
new_3332459d = """\
# ── Map size ──────────────────────────────────────────────
MIN_WIDTH,  MAX_WIDTH  = 14, 14
MIN_HEIGHT, MAX_HEIGHT = 14, 14

# ── Training ──────────────────────────────────────────────
MAPS_PER_GENOME = 24
MAX_GEN         = 200
POP_SIZE        = 50
LAMBDA_ARCHIVE  = 1
MODEL_PATH      = "neat_winner.pkl"
MAP_TXT_DIR     = "saved_maps_txt"
SEEDS           = [0]              # test 1 seed; doi thanh [0,1,2,3,4] khi train day du

# ── Tile types ────────────────────────────────────────────
WALL   = 0
FLOOR  = 1
PLAYER = 2
ENEMY  = 3
N_TILE_TYPES = 4

TILE_COLORS = {
    WALL:   [0.2,  0.2,  0.2],
    FLOOR:  [0.95, 0.95, 0.95],
    PLAYER: [0.2,  0.8,  0.2],
    ENEMY:  [0.9,  0.2,  0.2],
}

# ── Novelty Search weights ────────────────────────────────
NOVELTY_K           = 15
NOVELTY_W_SOLVABLE  = 0.20
NOVELTY_W_INTER     = 0.37
NOVELTY_W_INTRA     = 0.38
NOVELTY_W_ME        = 0.05
ARCHIVE_MAX_SIZE    = 500

# ── MAP-Elites grid (path_len x wall_dens) ───────────────
ME_PATH_BINS   = [0.5, 1.0, 1.5]   # 4 bins: <0.5 | 0.5-1.0 | 1.0-1.5 | >=1.5
ME_WALL_BINS   = [0.2, 0.4, 0.6]   # 4 bins: <0.2 | 0.2-0.4 | 0.4-0.6 | >=0.6
ME_ROWS        = len(ME_PATH_BINS) + 1   # 4
ME_COLS        = len(ME_WALL_BINS) + 1   # 4
ME_TOTAL_CELLS = ME_ROWS * ME_COLS       # 16

# ── Input encoding ────────────────────────────────────────
CONTEXT_SIZE      = 1
CTX_TILES         = (2*CONTEXT_SIZE+1)**2 - 1   # 8
NUM_RANDOM_INPUTS = 4
NUM_INPUTS        = CTX_TILES + NUM_RANDOM_INPUTS  # 12
NUM_OUTPUTS       = 1

_w_sum = NOVELTY_W_SOLVABLE + NOVELTY_W_INTER + NOVELTY_W_INTRA + NOVELTY_W_ME
print(f"Inputs : {NUM_INPUTS}  |  Seeds: {SEEDS}")
print(f"Maps/genome: {MAPS_PER_GENOME}")
print(f"Weights: solve={NOVELTY_W_SOLVABLE}, inter={NOVELTY_W_INTER}, intra={NOVELTY_W_INTRA}, me={NOVELTY_W_ME}")
print(f"Weight sum = {_w_sum:.4f}  {'OK' if abs(_w_sum - 1.0) < 1e-6 else 'NOT 1.0'}")
print(f"MAP-Elites: {ME_ROWS}x{ME_COLS} = {ME_TOTAL_CELLS} cells")
"""

# ── Cell 092b7fc4 — Novelty/Fitness ───────────────────────
new_092b7fc4 = """\
import copy
novelty_archive: list = []

# MAP-Elites archive: {(row, col): (fitness, levels)}
me_archive: dict = {}


def _generator_distance(levels_i, levels_j) -> float:
    n = min(len(levels_i), len(levels_j))
    if n == 0: return 0.0
    return float(np.mean([bc_distance(levels_i[k], levels_j[k]) for k in range(n)]))


def inter_novelty_score(levels_i, all_reference_level_lists, k=NOVELTY_K) -> float:
    if not all_reference_level_lists:
        return 0.0
    dists = sorted(_generator_distance(levels_i, ref) for ref in all_reference_level_lists)
    k_eff = min(k, len(dists))
    return float(np.mean(dists[:k_eff])) if k_eff > 0 else 0.0


def intra_novelty_score(levels, k=2) -> float:
    if len(levels) < 2:
        return 0.0
    scores = []
    for i, lvl_i in enumerate(levels):
        dists = sorted(bc_distance(lvl_i, lvl_j)
                       for j, lvl_j in enumerate(levels) if i != j)
        k_eff = min(k, len(dists))
        scores.append(np.mean(dists[:k_eff]))
    return float(np.mean(scores))


def solvability_fitness(levels) -> float:
    return float(np.mean([float(is_solvable(lvl)) for lvl in levels]))


# ── MAP-Elites helpers ─────────────────────────────────────
def _me_cell(level) -> tuple:
    sp = shortest_path_bfs(level)
    path_norm = min(2.0, (sp / (MAX_HEIGHT + MAX_WIDTH)) if sp else 0.0)
    wall = interior_wall_density(level)
    row = int(np.searchsorted(ME_PATH_BINS, path_norm))
    col = int(np.searchsorted(ME_WALL_BINS, wall))
    return row, col


def me_genome_coverage(levels) -> float:
    cells = set(_me_cell(l) for l in levels)
    return len(cells) / ME_TOTAL_CELLS


def update_me_archive(genome_levels_dict, genome_fitness_dict):
    for gid, levels in genome_levels_dict.items():
        fit = genome_fitness_dict.get(gid, 0.0)
        for level in levels:
            cell = _me_cell(level)
            if cell not in me_archive or fit > me_archive[cell][0]:
                me_archive[cell] = (fit, levels)


def eval_genomes(genomes, config):
    global novelty_archive, me_archive

    genome_levels = {}
    for gid, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        genome_levels[gid] = [generate_level(net) for _ in range(MAPS_PER_GENOME)]

    # Inter-novelty reference = ME archive cells + recent novelty archive
    me_refs  = [levels for (_, levels) in me_archive.values()]
    all_refs = me_refs + novelty_archive[-50:]

    genome_fitness = {}
    for gid, genome in genomes:
        levels = genome_levels[gid]
        f_solve = solvability_fitness(levels)
        f_intra = intra_novelty_score(levels, k=min(NOVELTY_K, len(levels) - 1))
        f_inter = inter_novelty_score(levels, all_refs) if all_refs else 0.0
        f_me    = me_genome_coverage(levels)
        genome.fitness = (NOVELTY_W_SOLVABLE * f_solve
                         + NOVELTY_W_INTRA   * f_intra
                         + NOVELTY_W_INTER   * f_inter
                         + NOVELTY_W_ME      * f_me)
        genome_fitness[gid] = genome.fitness

    update_me_archive(genome_levels, genome_fitness)

    sorted_g = sorted(genomes, key=lambda x: x[1].fitness or 0, reverse=True)
    for gid, genome in sorted_g[:5]:
        novelty_archive.append(genome_levels[gid])
    if len(novelty_archive) > ARCHIVE_MAX_SIZE:
        novelty_archive = novelty_archive[-ARCHIVE_MAX_SIZE:]


print(f"Fitness: {NOVELTY_W_SOLVABLE}*solve + {NOVELTY_W_INTRA}*intra + {NOVELTY_W_INTER}*inter + {NOVELTY_W_ME}*me_coverage")
"""

# ── Cell 8367aa57 — Reporter & Training ───────────────────
new_8367aa57 = """\
class PCGNNReporter(neat.reporting.BaseReporter):
    def __init__(self, log_interval=10, live_plot_interval=5, seed=0):
        self.log_interval       = log_interval
        self.live_plot_interval = live_plot_interval
        self.generations  = []
        self.best_fitness = []
        self.avg_fitness  = []
        self.intra_scores = []
        self.inter_scores = []
        self.solve_scores = []
        self.start_time   = None

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 4))
        self.bl, = self.ax1.plot([], [], label="Best fitness", color="#2196F3")
        self.al, = self.ax1.plot([], [], label="Avg fitness",  color="#FF9800", linestyle="--")
        self.ax1.set(xlabel="Generation", ylabel="Fitness", title=f"NEAT Fitness (seed={seed})")
        self.ax1.legend(); self.ax1.grid(True, alpha=0.3)
        self.ax2.set(xlabel="Generation", ylabel="Score", title="Fitness Components")
        self.ax2.grid(True, alpha=0.3)

        self.display_id = f"pcgnn-live-{seed}"
        display(self.fig, display_id=self.display_id)

    def start_generation(self, generation):
        if generation == 0:
            self.start_time = time.time()

    def post_evaluate(self, config, population, species, best_genome):
        gen = len(self.generations)
        self.generations.append(gen)

        fits = [g.fitness for g in population.values() if g.fitness is not None]
        best = max(fits); avg = np.mean(fits)
        self.best_fitness.append(best); self.avg_fitness.append(avg)

        best_net = neat.nn.RecurrentNetwork.create(best_genome, config)
        sample_levels = [generate_level(best_net) for _ in range(MAPS_PER_GENOME)]
        intra = intra_novelty_score(sample_levels, k=10)
        inter = inter_novelty_score(sample_levels, novelty_archive) if novelty_archive else 0.0
        solve = solvability_fitness(sample_levels)
        self.intra_scores.append(intra); self.inter_scores.append(inter)
        self.solve_scores.append(solve)

        elapsed = time.time() - self.start_time
        eta = (elapsed / (gen+1)) * (MAX_GEN - gen - 1) if gen > 0 else 0
        bar = '#' * int(50*(gen+1)/MAX_GEN) + '.' * (50 - int(50*(gen+1)/MAX_GEN))
        print(f"\\rGen {gen:3d}/{MAX_GEN} |{bar}| Best:{best:6.3f} Avg:{avg:5.3f} "
              f"Solve:{solve:.2f} Intra:{intra:.3f} Inter:{inter:.3f} "
              f"ME:{len(me_archive)}/{ME_TOTAL_CELLS} ETA:{eta/60:.1f}m",
              end='', flush=True)
        if gen % self.log_interval == 0:
            print()

        if gen % self.live_plot_interval == 0:
            self.bl.set_data(self.generations, self.best_fitness)
            self.al.set_data(self.generations, self.avg_fitness)
            self.ax1.relim(); self.ax1.autoscale_view()

            self.ax2.cla()
            self.ax2.plot(self.generations, self.solve_scores,  label="Solvability", color="#F44336")
            self.ax2.plot(self.generations, self.intra_scores,  label="Intra-novelty", color="#9C27B0")
            self.ax2.plot(self.generations, self.inter_scores,  label="Inter-novelty", color="#4CAF50")
            self.ax2.set(xlabel="Generation", ylabel="Score", title="Fitness Components")
            self.ax2.legend(); self.ax2.grid(True, alpha=0.3)
            update_display(self.fig, display_id=self.display_id)


# ─────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────
all_winners   = {}
all_reporters = {}

for seed in SEEDS:
    print(f"\\n{'='*60}")
    print(f"  Seed {seed}  ({SEEDS.index(seed)+1}/{len(SEEDS)})")
    print(f"{'='*60}")

    random.seed(seed)
    np.random.seed(seed)
    novelty_archive = []
    me_archive.clear()
    reset_level_cache()
    reset_bc_cache()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )
    pop      = neat.Population(config)
    reporter = PCGNNReporter(log_interval=20, live_plot_interval=5, seed=seed)
    pop.add_reporter(reporter)
    pop.add_reporter(neat.StatisticsReporter())

    winner = pop.run(eval_genomes, MAX_GEN)

    all_winners[seed]   = (winner, config)
    all_reporters[seed] = reporter

    path = f"neat_winner_seed{seed}.pkl"
    with open(path, "wb") as f:
        pickle.dump(winner, f)
    print(f"\\nSeed {seed} done -> {path}")

print(f"\\nAll {len(SEEDS)} seeds trained!")
"""

# Apply changes
count = 0
for cell in nb['cells']:
    if cell['id'] == '3332459d':
        cell['source'] = new_3332459d
        print('Updated 3332459d (Constants)')
        count += 1
    elif cell['id'] == '092b7fc4':
        cell['source'] = new_092b7fc4
        print('Updated 092b7fc4 (Novelty/Fitness)')
        count += 1
    elif cell['id'] == '8367aa57':
        cell['source'] = new_8367aa57
        print('Updated 8367aa57 (Reporter & Training)')
        count += 1

with open('improve2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Done. {count}/3 cells updated.')
