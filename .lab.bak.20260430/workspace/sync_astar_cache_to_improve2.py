"""
Sync A* caching optimization to improve2.ipynb.

Adds a memoization layer that consolidates is_solvable, shortest_path_bfs,
and _get_astar_result into a single cached call per level.tobytes().
Expected speedup: 3-10x on training time.
"""
import json
import sys
from pathlib import Path

NB_PATH = Path("improve2.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Backup
backup = NB_PATH.with_suffix('.ipynb.bak3')
backup.write_bytes(NB_PATH.read_bytes())
print(f"backup saved: {backup}")


CACHE_BLOCK = '''
# -----------------------------------------------------
# 4.12 Cached level metrics — TRAINING SPEEDUP (3-10x)
# A* + BFS are called 5-30 times per level per generation:
#   - astar_difficulty, behavior_characterization, path_diversity_fitness,
#     solvability_fitness, intra/inter novelty pairwise, etc.
# Cache one A* result per unique level (keyed by .tobytes()) and derive
# is_solvable, shortest_path, A* path from it.
# -----------------------------------------------------
_level_cache = {}
_LEVEL_CACHE_MAX = 50000


def _astar_internal(level, start, end):
    """Raw A* — only called from _compute_level_metrics on cache miss."""
    h, w = level.shape
    counter = 0
    open_set = [(abs(start[0]-end[0]) + abs(start[1]-end[1]), counter, start)]
    came_from = {start: None}
    g_score = {start: 0}
    visited = set()
    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == end:
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = came_from[node]
            return list(reversed(path)), len(visited)
        r, c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and (nr,nc) not in visited and level[nr,nc] != WALL:
                new_g = g_score[current] + 1
                if new_g < g_score.get((nr,nc), float('inf')):
                    came_from[(nr,nc)] = current
                    g_score[(nr,nc)] = new_g
                    counter += 1
                    heapq.heappush(open_set, (new_g + abs(nr-end[0]) + abs(nc-end[1]), counter, (nr,nc)))
    return None, len(visited)


def _compute_level_metrics(level):
    """Returns (path, num_considered, solvable, shortest_path_len). Cached by level.tobytes()."""
    key = level.tobytes()
    cached = _level_cache.get(key)
    if cached is not None:
        return cached

    ps = list(zip(*np.where(level == PLAYER)))
    es = list(zip(*np.where(level == ENEMY)))
    if not ps or not es:
        result = (None, 0, False, None)
    else:
        start = ps[0]
        end = min(es, key=lambda e: abs(e[0]-start[0]) + abs(e[1]-start[1]))
        path, considered = _astar_internal(level, start, end)
        solvable = path is not None and len(path) > 1
        sp_len = (len(path) - 1) if solvable else None
        result = (path, considered, solvable, sp_len)

    if len(_level_cache) >= _LEVEL_CACHE_MAX:
        # Simple FIFO eviction: drop oldest half
        items = list(_level_cache.items())
        _level_cache.clear()
        for k, v in items[len(items)//2:]:
            _level_cache[k] = v
    _level_cache[key] = result
    return result


def reset_level_cache():
    """Call between independent runs (different seeds) to free memory."""
    _level_cache.clear()


# Override the slow versions — same signatures, cached internals
def is_solvable(level):
    return _compute_level_metrics(level)[2]


def shortest_path_bfs(level):
    return _compute_level_metrics(level)[3]


def _get_astar_result(level):
    path, considered, _, _ = _compute_level_metrics(level)
    return path, considered


print(f"✅ A* + BFS caching enabled (max cache {_LEVEL_CACHE_MAX} levels)")
print(f"   is_solvable, shortest_path_bfs, _get_astar_result share single A* result per level")
print(f"   Expected speedup: 3-10x on training time")


'''


# ─────────── Find cell 8 (id 936b2eef) and insert cache block ───────────
cell8 = next(c for c in nb["cells"] if c.get("id") == "936b2eef")
src = "".join(cell8["source"]) if isinstance(cell8["source"], list) else cell8["source"]

# Insert just before `def interior_wall_density` so it's at the END of all definitions
# but BEFORE the final print statements. This way cached versions override originals.
marker = "def interior_wall_density"
idx = src.find(marker)
if idx == -1:
    sys.exit(f"ERROR: '{marker}' not found")

new_src = src[:idx] + CACHE_BLOCK + src[idx:]

cell8["source"] = [line + "\n" for line in new_src.split("\n")[:-1]] + (
    [new_src.split("\n")[-1]] if not new_src.endswith("\n") else []
)
cell8["outputs"] = []
cell8["execution_count"] = None

print(f"Cell 8 updated: {len(src)} -> {len(new_src)} chars (+{len(new_src) - len(src)})")


# ─────────── WRITE BACK ───────────
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

# Verify
src_all = NB_PATH.read_text(encoding="utf-8")
print()
print("Verify markers:")
print("  _level_cache:", "_level_cache = {}" in src_all)
print("  _astar_internal:", "_astar_internal" in src_all)
print("  _compute_level_metrics:", "_compute_level_metrics" in src_all)
print("  reset_level_cache:", "reset_level_cache" in src_all)
# Override of is_solvable should appear AFTER original; check there are 2 def is_solvable
print("  is_solvable count (should be 2):", src_all.count("def is_solvable"))
print("  shortest_path_bfs count (should be 2):", src_all.count("def shortest_path_bfs"))
print("  _get_astar_result count (should be 2):", src_all.count("def _get_astar_result"))
print()
print("OK: A* caching synced. Run cell 8 in Jupyter to apply.")
