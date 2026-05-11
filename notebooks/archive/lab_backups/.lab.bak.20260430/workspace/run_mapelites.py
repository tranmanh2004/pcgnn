"""
MAP-Elites experiment runner for PCGNN maze generation.

Archive grid:
  Axis 0 — turns_bin  : [0-1]  [2-4]  [5-8]  [9+]          → 4 bins
  Axis 1 — density_bin: [<0.15][0.15-0.30][0.30-0.45][0.45-0.60][>0.60] → 5 bins
  Total: 4 × 5 = 20 cells

Variants (ME_VARIANT env or CLI arg):
  ME0 — pure MAP-Elites, no penalty
  ME1 — MAP-Elites + V10 mode-collapse penalty (0.5)
  ME2 — MAP-Elites + direct turns fitness term (no penalty)
  ME3 — MAP-Elites + penalty + direct turns          [best so far: compound 0.840]
  ME4  — ME3 but tortuosity_fitness replaces turns_fitness
  ME5  — ME3 + segment_length_penalty max_seg=4  [KEEP: seg 5.30]
  ME5b — ME3 + segment_length_penalty max_seg=2  (stronger)
  ME6  — ME3 + dead_end_fitness (reward maze-like dead ends)
  ME7  — ME3 + tortuosity + dead_end combined
  ME8  — ME3 + branching_fitness (T/+ junctions, w=0.10)
  ME9  — ME3 + path_coverage_fitness (path/floor ratio, w=0.10)
  ME10 — ME5 + branching_fitness (seg_pen + branching combined) [KEEP*: 0.862]
  ME11 — ME8 + higher branching weight 0.15 (reduce intra to 0.20) [KEEP: 0.920]
  ME12 — ME10 + stronger branching target 0.15 (keep same weights as ME10) [KEEP*: 0.902]
  ME13 — branch weight 0.20 (reduce intra to 0.15), target 0.10
  ME14 — ME11 + seg_penalty(max_seg=4) (keep compound while reducing seg)
  ME15 — ME11 config, MAX_GEN=75 (longer training, same hyperparams)

Usage:
  python run_mapelites.py ME4 [out_dir]

Output:
  <out_dir>/maps.pkl      — N_FINAL_MAPS maps
  <out_dir>/result.json   — metrics + archive stats
  <out_dir>/sample.png    — 10×10 grid of first 100 maps
  <out_dir>/archive.json  — archive cell occupancy
"""
import sys
import os
import json
import time
import pickle
import heapq
import random
import math
from pathlib import Path
from collections import deque, Counter

import numpy as np
import neat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── reuse helpers from run_experiment ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import (
    generate_level, count_path_turns, _get_astar_result,
    is_solvable, shortest_path_bfs, astar_difficulty, astar_diversity,
    behavior_characterization, bc_distance, intra_novelty_score,
    solvability_fitness, path_diversity_fitness,
    WALL, FLOOR, PLAYER, ENEMY,
    MAX_HEIGHT, MAX_WIDTH, MAPS_PER_GENOME, POP_SIZE, SEED,
    N_FINAL_MAPS, NUM_INPUTS, NUM_OUTPUTS, PERTURB_SIZE,
    NEAT_CONFIG_TEXT,
)


def interior_wall_density(level):
    interior = level[1:-1, 1:-1]
    return float(np.sum(interior == WALL) / max(1, interior.size))

# ─── MAP-Elites config ────────────────────────────────────────────────────────
MAX_GEN_ME = 50           # ligeiramente mais que lab (30) para dar tempo ao archive
PENALTY_STRENGTH = 0.5    # V10 strength

TURNS_BINS  = [0, 1, 4, 8]   # boundaries: <=1 → bin0, 2-4 → bin1, 5-8 → bin2, 9+ → bin3
# Density grid: only "good" range [0.15, 0.60] → 3 bins. Genomes outside this range
# return None from density_bin() and are NOT placed in archive (no reward → pruned naturally).
DENS_GOOD_MIN = 0.15
DENS_GOOD_MAX = 0.60
DENS_BIN_EDGES = [0.30, 0.45]   # splits [0.15,0.60] into 3 bins
N_TURNS_BINS = len(TURNS_BINS)      # 4
N_DENS_BINS  = len(DENS_BIN_EDGES) + 1   # 3
TOTAL_CELLS  = N_TURNS_BINS * N_DENS_BINS  # 12

# Coverage bonus weights
BONUS_EMPTY   = 1.5   # fills empty cell
BONUS_IMPROVE = 1.2   # improves occupied cell
BONUS_WORSE   = 0.8   # doesn't improve

# Fitness weights (no inter-novelty — archive handles diversity)
ME_W_SOLVE   = 0.50
ME_W_INTRA   = 0.30
ME_W_PATH    = 0.10
ME_W_TURNS   = 0.10   # used only in ME2/ME3


def turns_bin(t):
    """Bin index for number of turns. t=-1 (unsolvable) → bin 0."""
    if t < 0 or t <= 1:
        return 0
    if t <= 4:
        return 1
    if t <= 8:
        return 2
    return 3


def density_bin(d):
    """Bin index for wall density within good range [0.15, 0.60].
    Returns None if outside good range — genome will not enter archive."""
    if d < DENS_GOOD_MIN or d > DENS_GOOD_MAX:
        return None
    for i, b in enumerate(DENS_BIN_EDGES):
        if d < b:
            return i
    return len(DENS_BIN_EDGES)


def genome_cell(levels):
    """MAP-Elites cell for a genome: bin of (mean_turns, mean_density) over maps.
    Returns None if mean density is outside good range [0.15, 0.60]."""
    valid_turns = [count_path_turns(l) for l in levels if is_solvable(l)]
    mean_turns = float(np.mean(valid_turns)) if valid_turns else -1.0
    mean_dens  = float(np.mean([interior_wall_density(l) for l in levels]))
    db = density_bin(mean_dens)
    if db is None:
        return None
    return (turns_bin(mean_turns), db)


# ─── mode-collapse helpers (V10) ─────────────────────────────────────────────
def detect_diagonal_stripes(level):
    h, w = level.shape
    walls = (level == WALL)
    if walls.sum() < 8:
        return 0.0
    diag_pairs = total = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if walls[r, c]:
                total += 1
                if walls[r + 1, c + 1]:
                    diag_pairs += 1
    return diag_pairs / max(1, total)


def density_extreme_penalty(level):
    interior = level[1:-1, 1:-1]
    dens = float(np.sum(interior == WALL) / max(1, interior.size))
    if dens < 0.15:
        return 1.0 - (dens / 0.15)
    if dens > 0.60:
        return min(1.0, (dens - 0.60) / 0.40)
    return 0.0


def mode_collapse_penalty(levels):
    diags = [detect_diagonal_stripes(l) for l in levels]
    dens  = [density_extreme_penalty(l) for l in levels]
    return 0.5 * (float(np.mean(diags)) + float(np.mean(dens)))


def turns_fitness(levels, target_turns=6):
    """Direct turns fitness: reward avg_turns toward target (capped)."""
    valid = [count_path_turns(l) for l in levels if is_solvable(l)]
    if not valid:
        return 0.0
    return float(np.mean([min(t, target_turns) / target_turns for t in valid]))


def _path_metrics(level):
    """Returns (tortuosity, mean_seg_len, path_coverage) or None if unsolvable."""
    import math
    path, _ = _get_astar_result(level)
    if path is None or len(path) < 3:
        return None
    euc = math.sqrt((path[-1][0]-path[0][0])**2 + (path[-1][1]-path[0][1])**2)
    tortuosity = len(path) / max(1.0, euc)
    # segment lengths between turns
    segs, seg = [], 1
    for i in range(1, len(path)-1):
        d1 = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        d2 = (path[i+1][0]-path[i][0],  path[i+1][1]-path[i][1])
        if d1 != d2: segs.append(seg); seg = 1
        else: seg += 1
    segs.append(seg)
    mean_seg = float(np.mean(segs))
    # path coverage
    floor_cells = max(1, int((level != WALL).sum()))
    coverage = len(path) / floor_cells
    return tortuosity, mean_seg, coverage


def tortuosity_fitness(levels, target=2.5):
    """Reward path tortuosity: path_len / euclidean_dist. Target 2.5."""
    vals = [_path_metrics(l) for l in levels]
    vals = [v[0] for v in vals if v is not None]
    if not vals:
        return 0.0
    return float(np.mean([min(t, target) / target for t in vals]))


def segment_length_penalty(levels, max_seg=4.0):
    """Penalise long straight segments. 0=good (all segs ≤ max_seg), 1=bad."""
    vals = [_path_metrics(l) for l in levels]
    vals = [v[1] for v in vals if v is not None]
    if not vals:
        return 0.0
    return float(np.mean([min(1.0, max(0.0, (s - max_seg) / max_seg)) for s in vals]))


def dead_end_fitness(levels, target=0.10):
    """Reward dead-end density (dead_ends / floor_cells). Target 0.10."""
    scores = []
    for lvl in levels:
        h, w = lvl.shape
        floor, de = 0, 0
        for r in range(h):
            for c in range(w):
                if lvl[r, c] != WALL:
                    floor += 1
                    nb = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                             if 0<=r+dr<h and 0<=c+dc<w and lvl[r+dr,c+dc] != WALL)
                    if nb == 1: de += 1
        scores.append(min(1.0, (de / max(1, floor)) / target))
    return float(np.mean(scores)) if scores else 0.0


def branching_fitness(levels, target=0.10):
    """Reward branch points (T/+ junctions). Target 0.10 branches/floor_cells.
    Horizontal stripes have near-zero branching; real mazes have 0.10+."""
    scores = []
    for lvl in levels:
        h, w = lvl.shape
        floor, branches = 0, 0
        for r in range(h):
            for c in range(w):
                if lvl[r, c] != WALL:
                    floor += 1
                    nb = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                             if 0<=r+dr<h and 0<=c+dc<w and lvl[r+dr,c+dc] != WALL)
                    if nb >= 3: branches += 1
        scores.append(min(1.0, (branches / max(1, floor)) / target))
    return float(np.mean(scores)) if scores else 0.0


def path_coverage_fitness(levels, target=0.35):
    """Reward path using more of the floor (path_len / floor_cells). Target 0.35.
    Corridor maps: ~0.20, maze-like maps: 0.30-0.50."""
    scores = []
    for lvl in levels:
        pm = _path_metrics(lvl)
        if pm is None: continue
        scores.append(min(1.0, pm[2] / target))
    return float(np.mean(scores)) if scores else 0.0


# ─── global archive ───────────────────────────────────────────────────────────
me_archive = {}   # (t_bin, d_bin) → {"fitness": float, "levels": list, "gen": int}


def eval_genomes_me(genomes, config, variant, gen_idx):
    global me_archive

    genome_levels = {}
    for gid, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome_levels[gid] = [generate_level(net) for _ in range(MAPS_PER_GENOME)]

    for gid, genome in genomes:
        levels = genome_levels[gid]

        f_solve = solvability_fitness(levels)
        f_intra = intra_novelty_score(levels, k=min(10, len(levels) - 1))
        f_path  = path_diversity_fitness(levels)

        # Variant-specific extras
        mc_pen   = 0.0
        f_turns  = 0.0
        seg_pen  = 0.0
        f_tort   = 0.0
        f_dead   = 0.0
        f_branch = 0.0
        f_cover  = 0.0

        if variant in ("ME1","ME3","ME5","ME5b","ME6","ME7","ME8","ME9","ME10","ME11","ME12","ME13","ME14","ME15"):
            mc_pen = PENALTY_STRENGTH * mode_collapse_penalty(levels)

        if variant in ("ME2","ME3","ME5","ME5b","ME6","ME8","ME9","ME10","ME11","ME12","ME13","ME14","ME15"):
            f_turns = turns_fitness(levels)

        if variant == "ME4":
            f_tort = tortuosity_fitness(levels)

        if variant == "ME5":
            seg_pen = segment_length_penalty(levels, max_seg=4.0)

        if variant == "ME5b":
            seg_pen = segment_length_penalty(levels, max_seg=2.0)

        if variant in ("ME10", "ME12", "ME14"):
            seg_pen = segment_length_penalty(levels, max_seg=4.0)

        if variant in ("ME6",):
            f_dead = dead_end_fitness(levels)

        if variant == "ME7":
            f_tort = tortuosity_fitness(levels)
            f_dead = dead_end_fitness(levels)

        if variant in ("ME8", "ME10", "ME11", "ME13", "ME14", "ME15"):
            f_branch = branching_fitness(levels, target=0.10)

        if variant == "ME12":
            f_branch = branching_fitness(levels, target=0.15)

        if variant == "ME9":
            f_cover = path_coverage_fitness(levels)

        if variant in ("ME11", "ME14", "ME15"):
            # higher branching weight: 0.50 solve + 0.20 intra + 0.10 path + 0.05 turns + 0.15 branch
            base = (0.50 * f_solve + 0.20 * f_intra
                    + 0.10 * f_path + 0.05 * f_turns + 0.15 * f_branch)
        elif variant == "ME13":
            # even higher branching: 0.50 solve + 0.15 intra + 0.10 path + 0.05 turns + 0.20 branch
            base = (0.50 * f_solve + 0.15 * f_intra
                    + 0.10 * f_path + 0.05 * f_turns + 0.20 * f_branch)
        elif variant in ("ME8", "ME10", "ME12"):
            # branching replaces half of intra, turns weight shared with branching
            base = (0.50 * f_solve + 0.25 * f_intra
                    + 0.10 * f_path + 0.05 * f_turns + 0.10 * f_branch)
        elif variant == "ME9":
            # path coverage replaces half of intra
            base = (0.50 * f_solve + 0.25 * f_intra
                    + 0.10 * f_path + 0.05 * f_turns + 0.10 * f_cover)
        elif variant in ("ME2","ME3","ME5","ME5b","ME6"):
            base = (ME_W_SOLVE * f_solve
                    + ME_W_INTRA * f_intra
                    + ME_W_PATH  * f_path
                    + ME_W_TURNS * f_turns)
        elif variant == "ME4":
            base = (ME_W_SOLVE * f_solve
                    + ME_W_INTRA * f_intra
                    + ME_W_PATH  * f_path
                    + ME_W_TURNS * f_tort)
        elif variant == "ME7":
            base = (ME_W_SOLVE * f_solve
                    + ME_W_INTRA * f_intra
                    + ME_W_PATH  * f_path
                    + 0.05 * f_tort + 0.05 * f_dead)
        else:
            base = (ME_W_SOLVE * f_solve
                    + (ME_W_INTRA + ME_W_TURNS) * f_intra
                    + ME_W_PATH  * f_path)

        base = max(0.0, base - mc_pen - 0.3 * seg_pen)

        # MAP-Elites coverage bonus
        cell = genome_cell(levels)
        if cell is None:
            # Outside good density range — penalize strongly, don't archive
            genome.fitness = base * 0.3
        elif cell not in me_archive:
            genome.fitness = base * BONUS_EMPTY
            me_archive[cell] = {"fitness": base, "levels": levels, "gen": gen_idx}
        elif base > me_archive[cell]["fitness"]:
            genome.fitness = base * BONUS_IMPROVE
            me_archive[cell] = {"fitness": base, "levels": levels, "gen": gen_idx}
        else:
            genome.fitness = base * BONUS_WORSE


# ─── main ─────────────────────────────────────────────────────────────────────
def run(variant, out_dir, max_gen=None):
    global me_archive
    me_archive = {}
    n_gen = max_gen if max_gen is not None else MAX_GEN_ME

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)

    # Write NEAT config
    cfg_path = out_dir / "config.txt"
    cfg_path.write_text(NEAT_CONFIG_TEXT.replace(
        f"pop_size               = {POP_SIZE}",
        f"pop_size               = {POP_SIZE}"
    ))
    # Override MAX_GEN in config text (fitness_threshold handles termination)
    cfg_text = NEAT_CONFIG_TEXT
    cfg_path.write_text(cfg_text)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        str(cfg_path),
    )

    pop = neat.Population(config)

    print(f"[{variant}] Starting MAP-Elites — {n_gen} gen, POP={POP_SIZE}")
    print(f"[{variant}] Archive grid: {N_TURNS_BINS}x{N_DENS_BINS} = {TOTAL_CELLS} cells")
    print(f"[{variant}] Penalty: {variant in ('ME1','ME3')}  | Turns fitness: {variant in ('ME2','ME3')}")
    t0 = time.time()
    gen_counter = [0]

    def fitness_fn(genomes, config):
        g = gen_counter[0]
        eval_genomes_me(genomes, config, variant, g)
        if g % 10 == 0 or g == n_gen - 1:
            coverage = len(me_archive)
            best_cell = max(me_archive, key=lambda c: me_archive[c]["fitness"]) if me_archive else None
            best_fit  = me_archive[best_cell]["fitness"] if best_cell else 0.0
            elapsed   = time.time() - t0
            print(f"  gen {g:3d}  archive={coverage:2d}/{TOTAL_CELLS}  best_fit={best_fit:.3f}  {elapsed:.0f}s")
        gen_counter[0] += 1

    pop.run(fitness_fn, n_gen)
    elapsed_total = time.time() - t0
    print(f"[{variant}] Training done in {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

    # ─── collect final maps from archive ────────────────────────────────────
    # Only use cells with turns_bin >= 1 (turns >= 2) — L-pattern cells excluded.
    # Archive still guides evolution via all 12 cells; output is only quality cells.
    print(f"[{variant}] Collecting maps from archive ({len(me_archive)} cells occupied)...")
    good_cells = {cell: v for cell, v in me_archive.items() if cell[0] >= 1}
    print(f"[{variant}] Good cells (turns>=2): {len(good_cells)}/{len(me_archive)}")

    archive_maps = []
    for cell in sorted(good_cells.keys()):
        archive_maps.extend(good_cells[cell]["levels"])

    # Fill up to N_FINAL_MAPS using best good-cell genome
    if good_cells:
        best_good_cell = max(good_cells, key=lambda c: good_cells[c]["fitness"])
        best_net_levels = good_cells[best_good_cell]["levels"]
    elif me_archive:
        best_good_cell = max(me_archive, key=lambda c: me_archive[c]["fitness"])
        best_net_levels = me_archive[best_good_cell]["levels"]
    else:
        best_net_levels = []

    while len(archive_maps) < N_FINAL_MAPS and best_net_levels:
        archive_maps.extend(best_net_levels)
    final_maps = archive_maps[:N_FINAL_MAPS]

    # ─── evaluate ───────────────────────────────────────────────────────────
    print(f"[{variant}] Evaluating {len(final_maps)} maps...")

    def classify_map(lvl):
        t = count_path_turns(lvl)
        d = interior_wall_density(lvl)
        diag = detect_diagonal_stripes(lvl)
        if d < 0.10:           return "near-empty"
        if d > 0.70:           return "near-full"
        if t < 0:              return "unsolvable"
        if t <= 1:             return "L-pattern"
        if diag > 0.40:        return "diagonal-stripes"
        if 2 <= t <= 7:        return "moderate-organic"
        if t >= 8:             return "very-zigzag"
        return "other"

    per_map = []
    for lvl in final_maps:
        t = count_path_turns(lvl)
        d = interior_wall_density(lvl)
        per_map.append({
            "turns": int(t),
            "solvable": int(t >= 0),
            "wall_density": float(d),
            "diag_score": float(detect_diagonal_stripes(lvl)),
            "class": classify_map(lvl),
        })

    cls_counter = Counter(m["class"] for m in per_map)
    n = len(per_map)
    solv_rate   = sum(m["solvable"] for m in per_map) / n
    non_l_rate  = sum(1 for m in per_map if m["turns"] >= 2) / n
    diag_count  = cls_counter.get("diagonal-stripes", 0)
    empty_count = cls_counter.get("near-empty", 0)
    full_count  = cls_counter.get("near-full", 0)
    compound    = non_l_rate * solv_rate * (1 - diag_count/n) * (1 - (empty_count+full_count)/n)

    # tortuosity metrics on final maps
    import math as _math
    tort_vals, seg_vals, de_vals = [], [], []
    for lvl in final_maps:
        pm = _path_metrics(lvl)
        if pm: tort_vals.append(pm[0]); seg_vals.append(pm[1])
        h,w=lvl.shape; fl=de=0
        for r in range(h):
            for c in range(w):
                if lvl[r,c]!=WALL:
                    fl+=1
                    nb=sum(1 for dr,dc in[(-1,0),(1,0),(0,-1),(0,1)]
                           if 0<=r+dr<h and 0<=c+dc<w and lvl[r+dr,c+dc]!=WALL)
                    if nb==1: de+=1
        if fl: de_vals.append(de/fl)
    tort_mean   = float(np.mean(tort_vals)) if tort_vals else 0.0
    seg_mean    = float(np.mean(seg_vals))  if seg_vals  else 0.0
    de_mean     = float(np.mean(de_vals))   if de_vals   else 0.0
    tort_factor = min(1.0, tort_mean / 2.0)
    tort_compound = compound * tort_factor

    # Archive stats
    archive_stats = {
        str(cell): {
            "fitness": round(me_archive[cell]["fitness"], 4),
            "gen_added": me_archive[cell]["gen"],
        }
        for cell in sorted(me_archive.keys())
    }

    result = {
        "variant": variant,
        "max_gen": n_gen,
        "elapsed_s": round(elapsed_total, 1),
        "archive_coverage": f"{len(me_archive)}/{TOTAL_CELLS}",
        "archive_coverage_pct": round(len(me_archive) / TOTAL_CELLS, 3),
        "n_maps_evaluated": n,
        "summary": {
            "solvability_rate": round(solv_rate, 4),
            "non_l_pattern_rate": round(non_l_rate, 4),
            "diagonal_stripes_count": diag_count,
            "near_empty_count": empty_count,
            "near_full_count": full_count,
        },
        "compound": round(compound, 4),
        "tortuosity_mean": round(tort_mean, 4),
        "seg_length_mean": round(seg_mean, 4),
        "dead_end_density": round(de_mean, 4),
        "tort_compound": round(tort_compound, 4),
        "classification": dict(cls_counter.most_common()),
        "archive": archive_stats,
    }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    with open(out_dir / "maps.pkl", "wb") as f:
        pickle.dump(final_maps, f)

    print(f"\n{'='*55}")
    print(f"[{variant}] Results")
    print(f"{'='*55}")
    print(f"  Archive coverage : {len(me_archive)}/{TOTAL_CELLS} cells ({len(me_archive)/TOTAL_CELLS*100:.0f}%)")
    print(f"  Solvability      : {solv_rate*100:.1f}%")
    print(f"  Non-L-pattern    : {non_l_rate*100:.1f}%")
    print(f"  Diagonal stripes : {diag_count}/{n}")
    print(f"  Near-empty       : {empty_count}/{n}")
    print(f"  Near-full        : {full_count}/{n}")
    print(f"  COMPOUND         : {compound:.4f}")
    print(f"  Tortuosity mean  : {tort_mean:.3f}  (target>=2.0, baseline=1.47)")
    print(f"  Seg length mean  : {seg_mean:.2f}   (target<=3.0, baseline=7.80)")
    print(f"  Dead-end density : {de_mean:.3f}  (target>=0.10, baseline=0.025)")
    print(f"  TORT_COMPOUND    : {tort_compound:.4f}  (compound * min(1, tort/2.0))")
    print(f"\n  Classification:")
    for cls, cnt in cls_counter.most_common():
        print(f"    {cls:<20}: {cnt:3d}  {'#'*cnt}")
    print(f"\n  Archive grid (turns x density, good range only):")
    print(f"    turns\\dens  .15-.30  .30-.45  .45-.60")
    for tb in range(N_TURNS_BINS):
        label = ["0-1","2-4","5-8","9+"][tb]
        row = ""
        for db in range(N_DENS_BINS):
            cell = (tb, db)
            if cell in me_archive:
                row += f"  {me_archive[cell]['fitness']:.2f} "
            else:
                row += "   --  "
        print(f"    turns={label}   {row}")
    print(f"\n  Result: {out_dir/'result.json'}")

    # ─── PNG grid ────────────────────────────────────────────────────────────
    n_show = min(100, len(final_maps))
    cols, rows = 10, (n_show + 9) // 10
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
    COLORS = {WALL:[0,0,0], FLOOR:[1,1,1], PLAYER:[0,0.7,0], ENEMY:[0.9,0.1,0.1]}
    for idx in range(n_show):
        ax = axes.flat[idx]
        lvl = final_maps[idx]
        img = np.zeros((*lvl.shape, 3))
        for tile, col in COLORS.items():
            img[lvl == tile] = col
        ax.imshow(img, interpolation="nearest")
        m = per_map[idx]
        ax.set_title(f"t={m['turns']} {m['class'][:6]}", fontsize=5)
        ax.axis("off")
    for ax in axes.flat[n_show:]:
        ax.axis("off")
    plt.suptitle(f"{variant} — compound={compound:.3f}  archive={len(me_archive)}/{TOTAL_CELLS}", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(out_dir / "sample.png"), dpi=110)
    plt.close()
    print(f"  PNG: {out_dir/'sample.png'}")

    return result


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "ME0"
    valid = ("ME0","ME1","ME2","ME3","ME4","ME5","ME5b","ME6","ME7","ME8","ME9","ME10","ME11","ME12","ME13","ME14","ME15")
    if variant not in valid:
        print(f"Unknown variant {variant}. Use: {', '.join(valid)}")
        sys.exit(1)
    out_dir = sys.argv[2] if len(sys.argv) > 2 else f".lab/workspace/expme-{variant}"
    max_gen = 75 if variant == "ME15" else None
    run(variant, out_dir, max_gen=max_gen)
