"""
Sync ME3v3 MAP-Elites winner to improve2.ipynb.

Changes:
1. Cell 936b2eef — ADD MAP-Elites constants + helpers after mode_collapse_penalty
2. Cell 092b7fc4 — REPLACE eval_genomes with ME3v3 MAP-Elites version
3. Cell 8367aa57 — UPDATE reporter: show me_archive coverage, reset archive per seed
"""
import json, sys
from pathlib import Path

NB_PATH = Path("improve2.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

backup = NB_PATH.with_suffix(".ipynb.bak4")
backup.write_bytes(NB_PATH.read_bytes())
print(f"backup: {backup}")


# ═══════════════════════════════════════════════════════════════════
# 1. Cell 936b2eef — append MAP-Elites helpers
# ═══════════════════════════════════════════════════════════════════
ME_HELPERS = '''
# -----------------------------------------------------
# 4.13  MAP-Elites archive — ME3v3 winner (2026-04-27)
# Grid: 4 turns-bins × 3 density-bins = 12 cells
# Only "good" density [0.15, 0.60] → bad-density genomes penalised, not archived.
# Output at evaluation: collect only from turns_bin>=1 (turns>=2) cells.
# -----------------------------------------------------

# Archive grid config
_ME_DENS_GOOD_MIN  = 0.15
_ME_DENS_GOOD_MAX  = 0.60
_ME_DENS_BIN_EDGES = [0.30, 0.45]   # 3 bins within good range
_ME_N_TURNS_BINS   = 4
_ME_N_DENS_BINS    = 3
_ME_TOTAL_CELLS    = _ME_N_TURNS_BINS * _ME_N_DENS_BINS   # 12

# Coverage bonuses
_ME_BONUS_EMPTY   = 1.5
_ME_BONUS_IMPROVE = 1.2
_ME_BONUS_WORSE   = 0.8
_ME_BONUS_BAD_DENS = 0.3

# Fitness weights (no inter-novelty — archive handles diversity)
_ME_W_SOLVE  = 0.50
_ME_W_INTRA  = 0.30
_ME_W_PATH   = 0.10
_ME_W_TURNS  = 0.10

me_archive: dict = {}   # (turns_bin, dens_bin) -> {"fitness", "levels", "gen"}


def _me_turns_bin(t: float) -> int:
    """Bin index for number of turns. t<0 (unsolvable) → bin 0."""
    if t < 0 or t <= 1: return 0
    if t <= 4:           return 1
    if t <= 8:           return 2
    return 3


def _me_density_bin(d: float):
    """Bin index for wall density in good range. Returns None if outside [0.15, 0.60]."""
    if d < _ME_DENS_GOOD_MIN or d > _ME_DENS_GOOD_MAX:
        return None
    for i, b in enumerate(_ME_DENS_BIN_EDGES):
        if d < b:
            return i
    return len(_ME_DENS_BIN_EDGES)


def me_genome_cell(levels):
    """MAP-Elites cell for a genome based on mean turns and mean density.
    Returns None if mean density is outside good range."""
    valid_turns = [count_path_turns(l) for l in levels if is_solvable(l)]
    mean_turns = float(np.mean(valid_turns)) if valid_turns else -1.0
    interior_dens = [
        float(np.sum(l[1:-1, 1:-1] == WALL) / max(1, l[1:-1, 1:-1].size))
        for l in levels
    ]
    mean_dens = float(np.mean(interior_dens))
    db = _me_density_bin(mean_dens)
    if db is None:
        return None
    return (_me_turns_bin(mean_turns), db)


def _me_turns_fitness(levels, target_turns: int = 6) -> float:
    """Direct turns fitness: reward avg_turns toward target."""
    valid = [count_path_turns(l) for l in levels if is_solvable(l)]
    if not valid:
        return 0.0
    return float(np.mean([min(t, target_turns) / target_turns for t in valid]))


print(f"✅ MAP-Elites helpers loaded (ME3v3 winner)")
print(f"   Grid: {_ME_N_TURNS_BINS} turns-bins x {_ME_N_DENS_BINS} density-bins = {_ME_TOTAL_CELLS} cells")
print(f"   Good density range: [{_ME_DENS_GOOD_MIN}, {_ME_DENS_GOOD_MAX}]")

'''

cell8 = next(c for c in nb["cells"] if c.get("id") == "936b2eef")
src8 = "".join(cell8["source"])

# Insert before interior_wall_density (or at end before final print)
marker = "def interior_wall_density"
idx = src8.find(marker)
if idx == -1:
    # Fallback: insert before last print block
    idx = src8.rfind('\nprint(')
if idx == -1:
    sys.exit("ERROR: cannot find insertion point in cell 936b2eef")

new_src8 = src8[:idx] + ME_HELPERS + src8[idx:]
cell8["source"] = new_src8.splitlines(keepends=True)
cell8["outputs"] = []
cell8["execution_count"] = None
print(f"Cell 936b2eef: {len(src8)} -> {len(new_src8)} chars")


# ═══════════════════════════════════════════════════════════════════
# 2. Cell 092b7fc4 — replace eval_genomes with ME3v3 version
# ═══════════════════════════════════════════════════════════════════
NEW_EVAL_GENOMES = '''
def eval_genomes(genomes, config):
    """
    ME3v3 MAP-Elites fitness (2026-04-27, compound 0.840 vs V10 0.679).

    Fitness = (0.50*f_solve + 0.30*f_intra + 0.10*f_path + 0.10*f_turns
               - 0.5*mode_collapse_penalty) * coverage_bonus

    coverage_bonus:
      1.5  — fills an empty archive cell
      1.2  — improves an occupied cell
      0.8  — doesn't improve
      0.3  — genome mean density outside good range [0.15, 0.60] (not archived)

    Archive grid: 4 turns-bins x 3 density-bins = 12 cells.
    Only good density [0.15, 0.60] cells are archived; extreme density genomes
    receive 0.3x penalty and are pruned naturally by NEAT selection.
    """
    global me_archive

    genome_levels = {}
    for gid, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome_levels[gid] = [generate_level(net) for _ in range(MAPS_PER_GENOME)]

    gen_idx = len(me_archive)   # approximate generation proxy

    for gid, genome in genomes:
        levels = genome_levels[gid]

        f_solve = solvability_fitness(levels)
        f_intra = intra_novelty_score(levels, k=min(10, len(levels) - 1))
        f_path  = path_diversity_fitness(levels)
        f_turns = _me_turns_fitness(levels)
        mc_pen  = 0.5 * mode_collapse_penalty(levels)

        base = max(0.0, (_ME_W_SOLVE * f_solve
                         + _ME_W_INTRA * f_intra
                         + _ME_W_PATH  * f_path
                         + _ME_W_TURNS * f_turns) - mc_pen)

        cell = me_genome_cell(levels)
        if cell is None:
            genome.fitness = base * _ME_BONUS_BAD_DENS
        elif cell not in me_archive:
            genome.fitness = base * _ME_BONUS_EMPTY
            me_archive[cell] = {"fitness": base, "levels": levels, "gen": gen_idx}
        elif base > me_archive[cell]["fitness"]:
            genome.fitness = base * _ME_BONUS_IMPROVE
            me_archive[cell] = {"fitness": base, "levels": levels, "gen": gen_idx}
        else:
            genome.fitness = base * _ME_BONUS_WORSE

'''

cell5 = next(c for c in nb["cells"] if c.get("id") == "092b7fc4")
src5 = "".join(cell5["source"])

# Replace only the def eval_genomes block (keep novelty helper functions above)
old_marker = "def eval_genomes(genomes, config):"
idx_start = src5.find(old_marker)
if idx_start == -1:
    sys.exit("ERROR: eval_genomes not found in cell 092b7fc4")

# Find the end: next top-level def or end of string
import re
# Find next top-level definition or end
rest = src5[idx_start:]
# Look for next def/class at column 0 after the function
next_top = re.search(r'\n(?=\S)', rest[1:])
if next_top:
    idx_end = idx_start + 1 + next_top.start()
else:
    idx_end = len(src5)

new_src5 = src5[:idx_start] + NEW_EVAL_GENOMES + src5[idx_end:]

# Update print at end
old_print = 'print("✅ Novelty search loaded (paper + 2 cải tiến)")'
new_print = 'print("✅ ME3v3 MAP-Elites loaded (compound 0.840 — best result 2026-04-27)")'
new_src5 = new_src5.replace(old_print, new_print)

old_weights_print = 'print(f"   Weights : solve={NOVELTY_W_SOLVABLE}, inter={NOVELTY_W_INTER}, intra={NOVELTY_W_INTRA}, path_div={NOVELTY_W_PATH_DIV}")'
new_weights_print = 'print(f"   Weights : solve={_ME_W_SOLVE}, intra={_ME_W_INTRA}, path={_ME_W_PATH}, turns={_ME_W_TURNS} | penalty=0.5")'
new_src5 = new_src5.replace(old_weights_print, new_weights_print)

old_k_print = 'print(f"   K={NOVELTY_K}, λ={LAMBDA_ARCHIVE}, path_div normalize=H+W=28")'
new_k_print = 'print(f"   Archive: {_ME_TOTAL_CELLS} cells | coverage_bonus: empty={_ME_BONUS_EMPTY}, improve={_ME_BONUS_IMPROVE}")'
new_src5 = new_src5.replace(old_k_print, new_k_print)

cell5["source"] = new_src5.splitlines(keepends=True)
cell5["outputs"] = []
cell5["execution_count"] = None
print(f"Cell 092b7fc4: {len(src5)} -> {len(new_src5)} chars")


# ═══════════════════════════════════════════════════════════════════
# 3. Cell 8367aa57 — update reporter: show me_archive, reset per seed
# ═══════════════════════════════════════════════════════════════════
cell7 = next(c for c in nb["cells"] if c.get("id") == "8367aa57")
src7 = "".join(cell7["source"])

# Replace "global novelty_archive" → also reset me_archive at training start
old_reset = "novelty_archive = []"
new_reset = "novelty_archive = []\n    me_archive.clear()   # reset MAP-Elites archive between seeds"
new_src7 = src7.replace(old_reset, new_reset, 1)

# Replace archive size display in reporter: novelty_archive → me_archive
old_archive_display = "Archive:{len(novelty_archive)}"
new_archive_display = "Archive:{len(me_archive)}/{_ME_TOTAL_CELLS}"
new_src7 = new_src7.replace(old_archive_display, new_archive_display)

cell7["source"] = new_src7.splitlines(keepends=True)
cell7["outputs"] = []
cell7["execution_count"] = None
print(f"Cell 8367aa57: {len(src7)} -> {len(new_src7)} chars")


# ═══════════════════════════════════════════════════════════════════
# WRITE BACK
# ═══════════════════════════════════════════════════════════════════
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"\nOK: synced ME3v3 to {NB_PATH}")

src_all = NB_PATH.read_text(encoding="utf-8")
print("\nVerify markers:")
print("  me_archive global      :", "me_archive: dict = {}" in src_all)
print("  me_genome_cell         :", "def me_genome_cell" in src_all)
print("  _me_turns_fitness      :", "def _me_turns_fitness" in src_all)
print("  ME3v3 eval_genomes     :", "ME3v3 MAP-Elites fitness" in src_all)
print("  mode_collapse_penalty  :", "mode_collapse_penalty" in src_all)
print("  me_archive.clear()     :", "me_archive.clear()" in src_all)
print("  coverage_bonus EMPTY   :", "_ME_BONUS_EMPTY" in src_all)
