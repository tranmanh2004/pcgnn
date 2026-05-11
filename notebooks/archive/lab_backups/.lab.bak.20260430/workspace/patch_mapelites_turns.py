"""
Patch: replace count_path_turns references in MAP-Elites helpers
with inline computation using _get_astar_result (already in notebook).
"""
import json, sys
from pathlib import Path

NB_PATH = Path("improve2.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

cell8 = next(c for c in nb["cells"] if c.get("id") == "936b2eef")
src = "".join(cell8["source"])

# ── 1. Add _count_turns helper that uses _get_astar_result ──────────────────
TURNS_HELPER = '''
def _count_turns(level) -> int:
    """Count direction changes in A* path. Returns -1 if unsolvable."""
    path, _ = _get_astar_result(level)
    if path is None or len(path) < 3:
        return -1 if path is None else 0
    turns = 0
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        d2 = (path[i+1][0] - path[i][0],  path[i+1][1] - path[i][1])
        if d1 != d2:
            turns += 1
    return turns

'''

# ── 2. Replace _me_turns_fitness ─────────────────────────────────────────────
OLD_TURNS_FN = '''def _me_turns_fitness(levels, target_turns: int = 6) -> float:
    """Direct turns fitness: reward avg_turns toward target."""
    valid = [count_path_turns(l) for l in levels if is_solvable(l)]
    if not valid:
        return 0.0
    return float(np.mean([min(t, target_turns) / target_turns for t in valid]))'''

NEW_TURNS_FN = '''def _me_turns_fitness(levels, target_turns: int = 6) -> float:
    """Direct turns fitness: reward avg_turns toward target."""
    valid = [_count_turns(l) for l in levels if _count_turns(l) >= 0]
    if not valid:
        return 0.0
    return float(np.mean([min(t, target_turns) / target_turns for t in valid]))'''

# ── 3. Replace me_genome_cell ────────────────────────────────────────────────
OLD_GENOME_CELL = '''def me_genome_cell(levels):
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
    return (_me_turns_bin(mean_turns), db)'''

NEW_GENOME_CELL = '''def me_genome_cell(levels):
    """MAP-Elites cell for a genome based on mean turns and mean density.
    Returns None if mean density is outside good range."""
    all_turns = [_count_turns(l) for l in levels]
    valid_turns = [t for t in all_turns if t >= 0]
    mean_turns = float(np.mean(valid_turns)) if valid_turns else -1.0
    interior_dens = [
        float(np.sum(l[1:-1, 1:-1] == WALL) / max(1, l[1:-1, 1:-1].size))
        for l in levels
    ]
    mean_dens = float(np.mean(interior_dens))
    db = _me_density_bin(mean_dens)
    if db is None:
        return None
    return (_me_turns_bin(mean_turns), db)'''

if OLD_TURNS_FN not in src:
    sys.exit("ERROR: _me_turns_fitness not found as expected")
if OLD_GENOME_CELL not in src:
    sys.exit("ERROR: me_genome_cell not found as expected")

# Insert _count_turns before _me_turns_fitness
insert_pt = src.find(OLD_TURNS_FN)
new_src = src[:insert_pt] + TURNS_HELPER + src[insert_pt:]
new_src = new_src.replace(OLD_TURNS_FN, NEW_TURNS_FN)
new_src = new_src.replace(OLD_GENOME_CELL, NEW_GENOME_CELL)

cell8["source"] = new_src.splitlines(keepends=True)
cell8["outputs"] = []
cell8["execution_count"] = None

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("OK — patch applied")
src_all = NB_PATH.read_text(encoding="utf-8")
print("  _count_turns defined   :", "def _count_turns(level)" in src_all)
print("  count_path_turns refs  :", src_all.count("count_path_turns"))   # should be 0 in new code
print("  _me_turns_fitness OK   :", "_count_turns(l) for l in levels" in src_all)
print("  me_genome_cell OK      :", "all_turns = [_count_turns" in src_all)
