"""
Sync ME11 winner to improve2.ipynb.

Changes vs ME3v3 (current notebook):
1. Cell 936b2eef — add branching_fitness helper + update weight constants
2. Cell 092b7fc4 — replace eval_genomes with ME11 version
   (compound 0.920 vs ME3v3 0.840)
"""
import json, sys
from pathlib import Path

NB_PATH = Path("improve2.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

backup = NB_PATH.with_suffix(".ipynb.bak5")
backup.write_bytes(NB_PATH.read_bytes())
print(f"backup: {backup}")


# ═══════════════════════════════════════════════════════════════════
# 1. Cell 936b2eef — add branching_fitness + update constants
# ═══════════════════════════════════════════════════════════════════
cell8 = next(c for c in nb["cells"] if c.get("id") == "936b2eef")
src8 = "".join(cell8["source"])

# Update weight constants: intra 0.30→0.20, turns 0.10→0.05, add branch 0.15
OLD_WEIGHTS = """# Fitness weights (no inter-novelty — archive handles diversity)
_ME_W_SOLVE  = 0.50
_ME_W_INTRA  = 0.30
_ME_W_PATH   = 0.10
_ME_W_TURNS  = 0.10"""

NEW_WEIGHTS = """# Fitness weights — ME11 winner (compound 0.920 vs ME3v3 0.840)
# Added branching_fitness (w=0.15): rewards T/+ junctions, breaks stripe bias.
# Reduced intra 0.30→0.20 to make room; turns reduced 0.10→0.05.
_ME_W_SOLVE  = 0.50
_ME_W_INTRA  = 0.20
_ME_W_PATH   = 0.10
_ME_W_TURNS  = 0.05
_ME_W_BRANCH = 0.15"""

if OLD_WEIGHTS not in src8:
    sys.exit("ERROR: weight constants not found in cell 936b2eef")

src8 = src8.replace(OLD_WEIGHTS, NEW_WEIGHTS)

# Add branching_fitness helper before the print block at end
BRANCH_HELPER = '''
def _me_branching_fitness(levels, target=0.10) -> float:
    """Reward branch points (T/+ junctions). Target 0.10 branches/floor_cells.
    Horizontal stripes have near-zero branching; real mazes have 0.10+.
    ME11 winner: w=0.15, target=0.10."""
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
                    if nb >= 3:
                        branches += 1
        scores.append(min(1.0, (branches / max(1, floor)) / target))
    return float(np.mean(scores)) if scores else 0.0

'''

# Insert before the final print block
insert_marker = '\nprint(f"✅ MAP-Elites helpers loaded'
idx = src8.find(insert_marker)
if idx == -1:
    sys.exit("ERROR: cannot find insertion point in cell 936b2eef")

src8 = src8[:idx] + BRANCH_HELPER + src8[idx:]

# Update the print to reflect ME11
src8 = src8.replace(
    'print(f"✅ MAP-Elites helpers loaded (ME3v3 winner)")',
    'print(f"✅ MAP-Elites helpers loaded (ME11 winner — compound 0.920)")'
)

cell8["source"] = src8.splitlines(keepends=True)
cell8["outputs"] = []
cell8["execution_count"] = None
print(f"Cell 936b2eef: weights+branching_fitness updated")


# ═══════════════════════════════════════════════════════════════════
# 2. Cell 092b7fc4 — replace eval_genomes with ME11 version
# ═══════════════════════════════════════════════════════════════════
NEW_EVAL = '''def eval_genomes(genomes, config):
    """
    ME11 MAP-Elites fitness (2026-04-27, compound 0.920 vs ME3v3 0.840).

    Fitness = (0.50*f_solve + 0.20*f_intra + 0.10*f_path
               + 0.05*f_turns + 0.15*f_branch
               - 0.5*mode_collapse_penalty) * coverage_bonus

    Key change vs ME3v3: added branching_fitness (w=0.15) which rewards
    T/+ junctions, directly breaking horizontal-stripe attractor.
    Intra-novelty reduced 0.30→0.20 to accommodate branching weight.

    coverage_bonus:
      1.5  — fills an empty archive cell
      1.2  — improves an occupied cell
      0.8  — doesn't improve
      0.3  — genome mean density outside good range [0.15, 0.60] (not archived)
    """
    global me_archive

    genome_levels = {}
    for gid, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome_levels[gid] = [generate_level(net) for _ in range(MAPS_PER_GENOME)]

    gen_idx = len(me_archive)

    for gid, genome in genomes:
        levels = genome_levels[gid]

        f_solve  = solvability_fitness(levels)
        f_intra  = intra_novelty_score(levels, k=min(10, len(levels) - 1))
        f_path   = path_diversity_fitness(levels)
        f_turns  = _me_turns_fitness(levels)
        f_branch = _me_branching_fitness(levels)
        mc_pen   = 0.5 * mode_collapse_penalty(levels)

        base = max(0.0, (_ME_W_SOLVE  * f_solve
                         + _ME_W_INTRA  * f_intra
                         + _ME_W_PATH   * f_path
                         + _ME_W_TURNS  * f_turns
                         + _ME_W_BRANCH * f_branch) - mc_pen)

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

old_marker = "def eval_genomes(genomes, config):"
idx_start = src5.find(old_marker)
if idx_start == -1:
    sys.exit("ERROR: eval_genomes not found in cell 092b7fc4")

import re
rest = src5[idx_start:]
next_top = re.search(r'\n(?=\S)', rest[1:])
if next_top:
    idx_end = idx_start + 1 + next_top.start()
else:
    idx_end = len(src5)

new_src5 = src5[:idx_start] + NEW_EVAL + src5[idx_end:]

# Update print statements
new_src5 = new_src5.replace(
    'print("✅ ME3v3 MAP-Elites loaded (compound 0.840 — best result 2026-04-27)")',
    'print("✅ ME11 MAP-Elites loaded (compound 0.920 — best result 2026-04-27)")'
)
new_src5 = new_src5.replace(
    f'print(f"   Weights : solve={{_ME_W_SOLVE}}, intra={{_ME_W_INTRA}}, path={{_ME_W_PATH}}, turns={{_ME_W_TURNS}} | penalty=0.5")',
    f'print(f"   Weights : solve={{_ME_W_SOLVE}}, intra={{_ME_W_INTRA}}, path={{_ME_W_PATH}}, turns={{_ME_W_TURNS}}, branch={{_ME_W_BRANCH}} | penalty=0.5")'
)

cell5["source"] = new_src5.splitlines(keepends=True)
cell5["outputs"] = []
cell5["execution_count"] = None
print(f"Cell 092b7fc4: eval_genomes replaced with ME11 version")


# ═══════════════════════════════════════════════════════════════════
# WRITE BACK
# ═══════════════════════════════════════════════════════════════════
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"\nOK: synced ME11 to {NB_PATH}")

src_all = NB_PATH.read_text(encoding="utf-8")
print("\nVerify:")
print("  _ME_W_BRANCH = 0.15   :", "_ME_W_BRANCH = 0.15" in src_all)
print("  _ME_W_INTRA  = 0.20   :", "_ME_W_INTRA  = 0.20" in src_all)
print("  _ME_W_TURNS  = 0.05   :", "_ME_W_TURNS  = 0.05" in src_all)
print("  _me_branching_fitness  :", "def _me_branching_fitness" in src_all)
print("  ME11 eval_genomes      :", "ME11 MAP-Elites fitness" in src_all)
print("  f_branch in eval       :", "f_branch = _me_branching_fitness" in src_all)
print("  _ME_W_BRANCH * f_branch:", "_ME_W_BRANCH * f_branch" in src_all)
