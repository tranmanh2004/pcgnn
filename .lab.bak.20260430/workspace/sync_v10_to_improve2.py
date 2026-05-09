"""
Sync V10 winner (overnight 2026-04-27) to improve2.ipynb.

Changes:
1. Cell 8 (936b2eef): append 2 helper functions after bc_distance:
   - detect_diagonal_stripes(level)
   - density_extreme_penalty(level)
2. Cell 10 (092b7fc4): modify eval_genomes to subtract penalty from fitness.
"""
import json
import sys
from pathlib import Path

NB_PATH = Path("improve2.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Backup
backup = NB_PATH.with_suffix('.ipynb.bak2')
backup.write_bytes(NB_PATH.read_bytes())
print(f"backup saved: {backup}")


# ─────────── HELPERS to insert in Cell 8 ───────────
NEW_HELPERS = '''
# -----------------------------------------------------
# 4.11 Anti-mode-collapse helpers (V10 winner — overnight 2026-04-27)
# Used as fitness penalty in eval_genomes to break diagonal-stripes
# and density-extreme attractors.
# -----------------------------------------------------
def detect_diagonal_stripes(level) -> float:
    """0..1 — fraction of walls that have NW-SE diagonal neighbor wall.
    Diagonal-stripe maps score high; organic maps score low.
    """
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


def density_extreme_penalty(level) -> float:
    """0..1 — 0 when interior wall density in [0.15, 0.6] (good range),
    ramps to 1.0 at extremes (near-empty or near-full).
    """
    interior = level[1:-1, 1:-1]
    dens = float(np.sum(interior == WALL) / max(1, interior.size))
    if dens < 0.15:
        return 1.0 - (dens / 0.15)
    if dens > 0.6:
        return min(1.0, (dens - 0.6) / 0.4)
    return 0.0


def mode_collapse_penalty(levels) -> float:
    """V10 combined penalty per generator: average over levels of
    0.5 × (diag_score + density_extreme). Used in eval_genomes:
        fitness = max(0, base_fitness - penalty_strength * mode_collapse_penalty(levels))
    where penalty_strength = 0.5 in V10.
    """
    diags = [detect_diagonal_stripes(l) for l in levels]
    dens = [density_extreme_penalty(l) for l in levels]
    return 0.5 * (float(np.mean(diags)) + float(np.mean(dens)))


'''


# ─────────── MODIFY Cell 8 — append helpers after bc_distance ───────────
cell8 = next(c for c in nb["cells"] if c.get("id") == "936b2eef")
src8 = "".join(cell8["source"]) if isinstance(cell8["source"], list) else cell8["source"]

# Find the `def interior_wall_density` (right after bc_distance) — insert helpers BEFORE it
marker = "def interior_wall_density"
idx = src8.find(marker)
if idx == -1:
    sys.exit("ERROR: 'def interior_wall_density' not found in cell 8")

new_src8 = src8[:idx] + NEW_HELPERS + src8[idx:]
cell8["source"] = [line + "\n" for line in new_src8.split("\n")[:-1]] + (
    [new_src8.split("\n")[-1]] if not new_src8.endswith("\n") else []
)
cell8["outputs"] = []
cell8["execution_count"] = None

print(f"Cell 8 updated: {len(src8)} -> {len(new_src8)} chars")


# ─────────── MODIFY Cell 10 — patch eval_genomes ───────────
cell10 = next(c for c in nb["cells"] if c.get("id") == "092b7fc4")
src10 = "".join(cell10["source"]) if isinstance(cell10["source"], list) else cell10["source"]

# Old fitness assignment (uniform across uses) — replace
OLD = """        genome.fitness = (NOVELTY_W_SOLVABLE  * f_solve
                        + NOVELTY_W_INTER     * f_inter
                        + NOVELTY_W_INTRA     * f_intra
                        + NOVELTY_W_PATH_DIV  * f_path_div)"""

NEW = """        base_fitness = (NOVELTY_W_SOLVABLE  * f_solve
                        + NOVELTY_W_INTER     * f_inter
                        + NOVELTY_W_INTRA     * f_intra
                        + NOVELTY_W_PATH_DIV  * f_path_div)
        # V10 winner: subtract mode-collapse penalty (anti-stripes + anti-extreme-density).
        # penalty_strength = 0.5 from research overnight 2026-04-27 (compound 0.679 vs baseline 0.105).
        mc_penalty = 0.5 * mode_collapse_penalty(levels)
        genome.fitness = max(0.0, base_fitness - mc_penalty)"""

if OLD not in src10:
    # Try alternate whitespace
    print("WARNING: exact OLD pattern not found, trying flexible match")
    print("Cell 10 first 200 chars around fitness assignment:")
    fit_idx = src10.find("genome.fitness = (NOVELTY_W")
    if fit_idx == -1:
        sys.exit("ERROR: cannot find fitness assignment in cell 10")
    print(repr(src10[fit_idx:fit_idx + 300]))
    sys.exit("Manual review needed — OLD pattern mismatch")

new_src10 = src10.replace(OLD, NEW)
cell10["source"] = [line + "\n" for line in new_src10.split("\n")[:-1]] + (
    [new_src10.split("\n")[-1]] if not new_src10.endswith("\n") else []
)
cell10["outputs"] = []
cell10["execution_count"] = None

print(f"Cell 10 updated: {len(src10)} -> {len(new_src10)} chars")


# ─────────── WRITE BACK ───────────
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"OK: synced V10 to {NB_PATH}")
print()
print("Verify markers:")
src_all = NB_PATH.read_text(encoding="utf-8")
print("  detect_diagonal_stripes:", "detect_diagonal_stripes" in src_all)
print("  density_extreme_penalty:", "density_extreme_penalty" in src_all)
print("  mode_collapse_penalty:", "mode_collapse_penalty" in src_all)
print("  base_fitness assignment:", "base_fitness = (NOVELTY_W_SOLVABLE" in src_all)
print("  penalty subtract:", "max(0.0, base_fitness - mc_penalty)" in src_all)
