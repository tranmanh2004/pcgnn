"""
Experiment #1 BC variant — H1: Add num_turns_norm as 7th dimension.

Replaces `behavior_characterization` in run_experiment.py.
bc_distance unchanged (still Euclidean / sqrt(dim)).
"""
import numpy as np
from collections import deque

# These constants must match run_experiment.py
WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3


def behavior_characterization(level, _shortest_path_bfs, _astar_difficulty, _get_astar_result):
    """7-dim BC vector — H1: baseline + num_turns_norm."""
    h, w = level.shape
    walkable = {FLOOR, PLAYER, ENEMY}

    interior = level[1:-1, 1:-1]
    wall_dens = float(np.sum(interior == WALL) / max(1, interior.size))

    sp = _shortest_path_bfs(level)
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

    diff = min(1.0, _astar_difficulty(level))

    # NEW dim — num_turns / max_possible_turns
    path, _ = _get_astar_result(level)
    if path is None or len(path) < 3:
        turns_norm = 0.0
    else:
        turns = 0
        for i in range(1, len(path) - 1):
            d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            if d1 != d2: turns += 1
        # max possible turns ≈ len(path) - 2
        turns_norm = min(1.0, turns / max(1, len(path) - 2))

    return np.array([wall_dens, path_norm, dead_norm, branch_norm, regions_norm, diff, turns_norm], dtype=float)
