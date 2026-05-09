"""
Run experiment with a selectable variant.

Variants test different approaches to fix mode collapse (diagonal stripes + near-empty/full).
All variants build on top of `run_experiment.py` — they monkey-patch BC and/or fitness components.

Usage:
    python run_variant.py <variant_id> <exp_id>
    python run_variant.py V1 1   # run V1, save to .lab/workspace/expv-V1-1/
"""
import sys
import json
import time
import pickle
import random
from pathlib import Path

import numpy as np
import neat

sys.path.insert(0, str(Path(__file__).parent))
import run_experiment as RE
from run_experiment import (
    generate_level, _get_astar_result, count_path_turns,
    is_solvable, shortest_path_bfs, astar_difficulty, astar_diversity,
    path_diversity_fitness, solvability_fitness,
    NEAT_CONFIG_TEXT, MAX_GEN, POP_SIZE, MAPS_PER_GENOME, SEED, N_FINAL_MAPS,
    NOVELTY_K, NOVELTY_W_SOLVABLE, NOVELTY_W_INTER, NOVELTY_W_INTRA,
    NOVELTY_W_PATH_DIV, ARCHIVE_MAX_SIZE, LAMBDA_ARCHIVE,
    WALL, FLOOR, PLAYER, ENEMY,
)
from collections import deque


# =====================================================================
# HELPER METRICS (for BC + penalty)
# =====================================================================
def spatial_entropy(level):
    """Wall pattern entropy. Diagonal stripes have low entropy; organic high."""
    h, w = level.shape
    walls = (level == WALL)
    if walls.sum() < 4:
        return 0.0
    # Divide into 4x4 cells, count walls per cell, compute entropy of distribution
    cell_h = max(1, h // 4)
    cell_w = max(1, w // 4)
    counts = []
    for r in range(0, h, cell_h):
        for c in range(0, w, cell_w):
            counts.append(int(walls[r:r+cell_h, c:c+cell_w].sum()))
    counts = np.array(counts, dtype=float)
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log2(p)))
    max_entropy = np.log2(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def corridor_width_var(level):
    """Variance of corridor widths along walkable rows. Diagonal stripes have width=1 uniformly → low var."""
    h, w = level.shape
    widths = []
    for r in range(h):
        in_corridor = False
        cur_w = 0
        for c in range(w):
            if level[r, c] != WALL:
                if in_corridor:
                    cur_w += 1
                else:
                    in_corridor = True; cur_w = 1
            else:
                if in_corridor and cur_w > 0:
                    widths.append(cur_w)
                in_corridor = False; cur_w = 0
        if in_corridor and cur_w > 0:
            widths.append(cur_w)
    if len(widths) < 2:
        return 0.0
    return float(min(1.0, np.var(widths) / 4.0))  # normalize


def wall_density_balance(level):
    """1.0 when wall_dens in [0.3, 0.5], 0.0 at extremes. Encourages balanced maze."""
    interior = level[1:-1, 1:-1]
    dens = float(np.sum(interior == WALL) / max(1, interior.size))
    if 0.3 <= dens <= 0.5:
        return 1.0
    if dens < 0.3:
        return max(0.0, dens / 0.3)
    return max(0.0, (1.0 - dens) / 0.5)


def detect_diagonal_stripes(level):
    """0..1 — fraction of walls that have NW-SE diagonal neighbor wall. Stripes → high."""
    h, w = level.shape
    walls = (level == WALL)
    if walls.sum() < 8:
        return 0.0
    diag_pairs = 0
    total = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if walls[r, c]:
                total += 1
                if walls[r + 1, c + 1]:
                    diag_pairs += 1
    return diag_pairs / max(1, total)


def density_extreme_penalty(level):
    """Returns penalty in [0, 1]. 0 = balanced, 1 = extreme empty/full."""
    interior = level[1:-1, 1:-1]
    dens = float(np.sum(interior == WALL) / max(1, interior.size))
    if dens < 0.15:
        return 1.0 - (dens / 0.15)
    if dens > 0.6:
        return min(1.0, (dens - 0.6) / 0.4)
    return 0.0


# =====================================================================
# BC VARIANTS
# =====================================================================
def bc_v_baseline(level):
    """V0: BC 7-dim winner from exp #5 (current)."""
    return RE.behavior_characterization(level)


def bc_v_entropy(level):
    """V_E: BC 8-dim — adds spatial_entropy."""
    base = RE.behavior_characterization(level)
    e = spatial_entropy(level)
    return np.append(base, e)


def bc_v_corridor(level):
    """V_C: BC 8-dim — adds corridor_width_var."""
    base = RE.behavior_characterization(level)
    return np.append(base, corridor_width_var(level))


def bc_v_balance(level):
    """V_B: BC 8-dim — adds wall_density_balance."""
    base = RE.behavior_characterization(level)
    return np.append(base, wall_density_balance(level))


def bc_v_combined(level):
    """V_X: BC 10-dim — adds entropy + corridor + balance."""
    base = RE.behavior_characterization(level)
    return np.append(base, [
        spatial_entropy(level),
        corridor_width_var(level),
        wall_density_balance(level),
    ])


# =====================================================================
# DISTANCE VARIANTS (per BC variant size)
# =====================================================================
def make_bc_distance(bc_fn, weights):
    weights = np.array(weights, dtype=float)
    norm = float(np.sqrt(weights.sum()))
    def d(a, b):
        ba = bc_fn(a); bb = bc_fn(b)
        diff = ba - bb
        return float(np.sqrt(np.sum(weights * diff * diff)) / norm)
    return d


# =====================================================================
# FITNESS PENALTIES
# =====================================================================
def penalty_diagonal(levels):
    """Mean diagonal-stripe score across levels. Higher = more stripes."""
    return float(np.mean([detect_diagonal_stripes(l) for l in levels]))


def penalty_density(levels):
    """Mean density-extreme penalty."""
    return float(np.mean([density_extreme_penalty(l) for l in levels]))


# =====================================================================
# VARIANT REGISTRY
# Each variant: (description, bc_fn, weights, fitness_penalty_fn or None, penalty_strength)
# =====================================================================
VARIANTS = {
    "V0": dict(
        desc="Control: BC 7-dim winner from exp #5 (re-run for sanity)",
        bc_fn=bc_v_baseline,
        weights=[1, 1, 1, 1, 1, 1, 7.84],
        fitness_penalty=None,
        penalty_strength=0.0,
    ),
    "V1": dict(
        desc="BC + diagonal penalty in fitness",
        bc_fn=bc_v_baseline,
        weights=[1, 1, 1, 1, 1, 1, 7.84],
        fitness_penalty="diagonal",
        penalty_strength=0.3,
    ),
    "V2": dict(
        desc="BC + density-extreme penalty in fitness",
        bc_fn=bc_v_baseline,
        weights=[1, 1, 1, 1, 1, 1, 7.84],
        fitness_penalty="density",
        penalty_strength=0.3,
    ),
    "V3": dict(
        desc="BC + combined (diagonal + density) penalty in fitness",
        bc_fn=bc_v_baseline,
        weights=[1, 1, 1, 1, 1, 1, 7.84],
        fitness_penalty="combined",
        penalty_strength=0.3,
    ),
    "V4": dict(
        desc="BC 8-dim with spatial_entropy",
        bc_fn=bc_v_entropy,
        weights=[1, 1, 1, 1, 1, 1, 7.84, 4.0],
        fitness_penalty=None,
        penalty_strength=0.0,
    ),
    "V5": dict(
        desc="BC 8-dim with corridor_width_var",
        bc_fn=bc_v_corridor,
        weights=[1, 1, 1, 1, 1, 1, 7.84, 4.0],
        fitness_penalty=None,
        penalty_strength=0.0,
    ),
    "V6": dict(
        desc="Best-of-both: reduce turns weight to 4 (2x), add combined penalty",
        bc_fn=bc_v_baseline,
        weights=[1, 1, 1, 1, 1, 1, 4.0],
        fitness_penalty="combined",
        penalty_strength=0.3,
    ),
    "V7": dict(
        desc="BC 10-dim combined + small combined penalty",
        bc_fn=bc_v_combined,
        weights=[1, 1, 1, 1, 1, 1, 7.84, 3.0, 3.0, 3.0],
        fitness_penalty="combined",
        penalty_strength=0.2,
    ),
    # Phase 2 — combinations + refinements (run if Phase 1 compound < 0.55)
    "V8": dict(
        desc="BC 9-dim entropy + corridor (combine V4 + V5)",
        bc_fn=lambda lvl: np.append(
            RE.behavior_characterization(lvl),
            [spatial_entropy(lvl), corridor_width_var(lvl)],
        ),
        weights=[1, 1, 1, 1, 1, 1, 7.84, 4.0, 4.0],
        fitness_penalty=None,
        penalty_strength=0.0,
    ),
    "V9": dict(
        desc="V6 + entropy dim (best low-turn balance + structure)",
        bc_fn=bc_v_entropy,
        weights=[1, 1, 1, 1, 1, 1, 4.0, 4.0],
        fitness_penalty="combined",
        penalty_strength=0.3,
    ),
    "V10": dict(
        desc="V3 with STRONGER combined penalty (0.5)",
        bc_fn=bc_v_baseline,
        weights=[1, 1, 1, 1, 1, 1, 7.84],
        fitness_penalty="combined",
        penalty_strength=0.5,
    ),
    "V11": dict(
        desc="Inverted: turns weight 1x + combined penalty (low BC pressure)",
        bc_fn=bc_v_baseline,
        weights=[1, 1, 1, 1, 1, 1, 1.0],
        fitness_penalty="combined",
        penalty_strength=0.4,
    ),
    # Phase 3 — fundamental shifts (run if Phase 2 compound < 0.55)
    "V12": dict(
        desc="Ablate turns dim entirely — test if turns is even needed",
        bc_fn=lambda lvl: RE.behavior_characterization(lvl)[:6],  # drop dim 7
        weights=[1, 1, 1, 1, 1, 1],
        fitness_penalty="combined",
        penalty_strength=0.3,
    ),
    "V13": dict(
        desc="Minimal BC: density_balance + corridor + entropy (3-dim, no turns)",
        bc_fn=lambda lvl: np.array([
            wall_density_balance(lvl),
            corridor_width_var(lvl),
            spatial_entropy(lvl),
        ]),
        weights=[2, 2, 2],
        fitness_penalty="combined",
        penalty_strength=0.3,
    ),
    "V14": dict(
        desc="V6 with stronger combined penalty (0.5) + balance dim",
        bc_fn=bc_v_balance,
        weights=[1, 1, 1, 1, 1, 1, 4.0, 4.0],
        fitness_penalty="combined",
        penalty_strength=0.5,
    ),
    "V15": dict(
        desc="V8 (entropy+corridor) + heavy combined penalty (0.5)",
        bc_fn=lambda lvl: np.append(
            RE.behavior_characterization(lvl),
            [spatial_entropy(lvl), corridor_width_var(lvl)],
        ),
        weights=[1, 1, 1, 1, 1, 1, 4.0, 4.0, 4.0],  # reduce turns to 2x
        fitness_penalty="combined",
        penalty_strength=0.5,
    ),
}


# =====================================================================
# CUSTOM EVAL_GENOMES (allows penalty hooks)
# =====================================================================
def make_eval_genomes(bc_fn, weights, penalty_kind, penalty_strength):
    """Build an eval_genomes closure with given BC + optional fitness penalty."""
    bc_dist = make_bc_distance(bc_fn, weights)
    archive = []

    def _generator_distance(levels_i, levels_j):
        n = min(len(levels_i), len(levels_j))
        if n == 0: return 0.0
        return float(np.mean([bc_dist(levels_i[k], levels_j[k]) for k in range(n)]))

    def _intra_novelty(levels, k=10):
        if len(levels) < 2: return 0.0
        scores = []
        for i, lvl_i in enumerate(levels):
            dists = sorted(bc_dist(lvl_i, lvl_j) for j, lvl_j in enumerate(levels) if i != j)
            k_eff = min(k, len(dists))
            scores.append(np.mean(dists[:k_eff]))
        return float(np.mean(scores))

    def _penalty(levels):
        if penalty_kind is None:
            return 0.0
        if penalty_kind == "diagonal":
            return penalty_diagonal(levels)
        if penalty_kind == "density":
            return penalty_density(levels)
        if penalty_kind == "combined":
            return 0.5 * (penalty_diagonal(levels) + penalty_density(levels))
        return 0.0

    def eval_genomes(genomes, config):
        nonlocal archive
        genome_levels = {}
        for gid, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome_levels[gid] = [generate_level(net) for _ in range(MAPS_PER_GENOME)]

        gids = [gid for gid, _ in genomes]
        for gid, genome in genomes:
            levels = genome_levels[gid]
            f_solve = solvability_fitness(levels)
            f_intra = _intra_novelty(levels, k=10)
            f_path_div = path_diversity_fitness(levels)
            pop_ref = [genome_levels[other] for other in gids if other != gid]
            all_ref = pop_ref + archive
            if all_ref:
                dists = sorted(_generator_distance(levels, ref) for ref in all_ref)
                k_eff = min(NOVELTY_K, len(dists))
                f_inter = float(np.mean(dists[:k_eff]))
            else:
                f_inter = 0.0
            base_fit = (NOVELTY_W_SOLVABLE * f_solve
                        + NOVELTY_W_INTER * f_inter
                        + NOVELTY_W_INTRA * f_intra
                        + NOVELTY_W_PATH_DIV * f_path_div)
            pen = penalty_strength * _penalty(levels)
            genome.fitness = max(0.0, base_fit - pen)

        if LAMBDA_ARCHIVE > 0 and len(gids) >= LAMBDA_ARCHIVE:
            chosen = random.sample(gids, LAMBDA_ARCHIVE)
            for cid in chosen:
                archive.append(genome_levels[cid])
        if len(archive) > ARCHIVE_MAX_SIZE:
            archive[:] = archive[-ARCHIVE_MAX_SIZE:]

    return eval_genomes


# =====================================================================
# DETAILED EVALUATION (100 maps)
# =====================================================================
def evaluate_detailed(maps):
    n = len(maps)
    turns = [count_path_turns(m) for m in maps]
    solvable = [t >= 0 for t in turns]
    non_l = [t >= 2 for t in turns]
    diff = [astar_difficulty(m) for m in maps]
    diag = [detect_diagonal_stripes(m) for m in maps]
    densities = []
    for m in maps:
        interior = m[1:-1, 1:-1]
        densities.append(float(np.sum(interior == WALL) / max(1, interior.size)))

    near_empty = sum(1 for d in densities if d < 0.15)
    near_full = sum(1 for d in densities if d > 0.6)
    diag_count = sum(1 for d in diag if d > 0.4)

    # Pairwise diversity (50 random pairs)
    random.seed(0)
    pairs = [(random.randrange(n), random.randrange(n)) for _ in range(50)]
    pairs = [(a, b) for a, b in pairs if a != b][:50]
    div_pairs = [astar_diversity(maps[a], maps[b]) for a, b in pairs]

    return {
        "n_maps": n,
        "non_l_pattern_rate": sum(non_l) / n,
        "solvability": sum(solvable) / n,
        "L_pattern_count": sum(1 for t in turns if 0 <= t <= 1),
        "unsolvable_count": sum(1 for t in turns if t == -1),
        "diagonal_stripes_count": diag_count,
        "near_empty_count": near_empty,
        "near_full_count": near_full,
        "astar_diff_mean": float(np.mean(diff)),
        "astar_div_pairs_mean": float(np.mean(div_pairs)),
        "wall_density_mean": float(np.mean(densities)),
        "wall_density_std": float(np.std(densities)),
        "diag_score_mean": float(np.mean(diag)),
        "turns_histogram": [int(sum(1 for t in turns if t == k)) for k in range(8)] + [
            int(sum(1 for t in turns if t >= 8)),
            int(sum(1 for t in turns if t == -1)),
        ],
    }


# =====================================================================
# MAIN
# =====================================================================
def main():
    if len(sys.argv) < 3:
        print("Usage: run_variant.py <variant_id> <exp_id>")
        print("Available variants:", ", ".join(VARIANTS.keys()))
        sys.exit(1)
    variant_id = sys.argv[1]
    exp_id = sys.argv[2]
    if variant_id not in VARIANTS:
        print(f"Unknown variant {variant_id}. Available: {list(VARIANTS.keys())}")
        sys.exit(1)
    cfg = VARIANTS[variant_id]
    out_dir = Path(__file__).parent / f"expv-{variant_id}-{exp_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{variant_id}] {cfg['desc']}")
    print(f"[{variant_id}] BC dim = {len(cfg['weights'])}, weights = {cfg['weights']}")
    print(f"[{variant_id}] Penalty = {cfg['fitness_penalty']} × {cfg['penalty_strength']}")

    random.seed(SEED)
    np.random.seed(SEED)

    # NEAT config
    cfg_path = out_dir / "config.txt"
    cfg_path.write_text(NEAT_CONFIG_TEXT)
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, str(cfg_path),
    )
    pop = neat.Population(config)

    eval_fn = make_eval_genomes(
        cfg["bc_fn"], cfg["weights"],
        cfg["fitness_penalty"], cfg["penalty_strength"],
    )

    t0 = time.time()
    print(f"[{variant_id}] training {MAX_GEN} gen × {POP_SIZE} pop × {MAPS_PER_GENOME} maps...", flush=True)
    winner = pop.run(eval_fn, MAX_GEN)
    train_secs = time.time() - t0
    print(f"[{variant_id}] training done in {train_secs:.1f}s", flush=True)

    # Generate 100 final maps for detailed eval
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    random.seed(SEED + 100)  # different seed for inference variety
    np.random.seed(SEED + 100)
    final_maps = [generate_level(net) for _ in range(100)]
    with open(out_dir / "maps.pkl", "wb") as f:
        pickle.dump(final_maps, f)
    with open(out_dir / "winner.pkl", "wb") as f:
        pickle.dump(winner, f)

    # Detailed metrics
    metrics = evaluate_detailed(final_maps)
    metrics["variant_id"] = variant_id
    metrics["exp_id"] = exp_id
    metrics["description"] = cfg["desc"]
    metrics["train_seconds"] = train_secs
    with open(out_dir / "result.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print()
    print(f"[{variant_id}] non_l_pattern_rate    = {metrics['non_l_pattern_rate']:.4f}")
    print(f"[{variant_id}] solvability           = {metrics['solvability']:.4f}")
    print(f"[{variant_id}] diagonal_stripes_pct  = {metrics['diagonal_stripes_count']}/{metrics['n_maps']}")
    print(f"[{variant_id}] near_empty_pct        = {metrics['near_empty_count']}/{metrics['n_maps']}")
    print(f"[{variant_id}] near_full_pct         = {metrics['near_full_count']}/{metrics['n_maps']}")
    print(f"[{variant_id}] astar_div_pairs_mean  = {metrics['astar_div_pairs_mean']:.4f}")
    print(f"[{variant_id}] astar_diff_mean       = {metrics['astar_diff_mean']:.4f}")


if __name__ == "__main__":
    main()
