"""Sync winning BC variant (exp #5) into improve2.ipynb cell 8 (id 936b2eef)."""
import json
import sys
from pathlib import Path

NEW_BC_CODE = '''def behavior_characterization(level):
    h, w = level.shape
    walkable = {FLOOR, PLAYER, ENEMY}

    # 1. Wall density (interior)
    interior = level[1:-1, 1:-1]
    wall_dens = float(np.sum(interior == WALL) / max(1, interior.size))

    # 2. Path length normalized
    sp = shortest_path_bfs(level)
    path_norm = (sp / (h + w)) if sp else 0.0
    path_norm = min(1.0, path_norm)

    # 3. Dead ends + branches
    dead_ends = 0
    branches = 0
    for r in range(h):
        for c in range(w):
            if level[r, c] not in walkable: continue
            nb = 0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<h and 0<=nc<w and level[nr,nc] in walkable:
                    nb += 1
            if nb == 1: dead_ends += 1
            elif nb >= 3: branches += 1
    total_walk = max(1, int((level != WALL).sum()))
    dead_norm = dead_ends / total_walk
    branch_norm = branches / total_walk

    # 4. Number of connected regions
    visited = np.zeros((h, w), dtype=bool)
    regions = 0
    for r0 in range(h):
        for c0 in range(w):
            if level[r0, c0] in walkable and not visited[r0, c0]:
                regions += 1
                q = deque([(r0, c0)])
                visited[r0, c0] = True
                while q:
                    r, c = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and level[nr,nc] in walkable:
                            visited[nr,nc] = True
                            q.append((nr,nc))
    regions_norm = min(1.0, regions / 10.0)

    # 5. A* difficulty
    diff = min(1.0, astar_difficulty(level))

    # 6. NEW (BC tuning research winner — exp #5): turns_norm as 7th dim
    # Capture path complexity directly. Reduces L-pattern from 58% (baseline) to 2%.
    path, _ = _get_astar_result(level)
    if path is None or len(path) < 3:
        turns_norm = 0.0
    else:
        turns = 0
        for i in range(1, len(path) - 1):
            d1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            d2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            if d1 != d2: turns += 1
        turns_norm = min(1.0, turns / max(1, len(path) - 2))

    return np.array([wall_dens, path_norm, dead_norm, branch_norm, regions_norm, diff, turns_norm], dtype=float)


def bc_distance(level_a, level_b) -> float:
    # Research winner: weighted Euclidean. Turns dim weighted 7.84 (effective sqrt = 2.8x).
    # Boosting weight on turns dim breaks L-pattern attractor.
    bc_a = behavior_characterization(level_a)
    bc_b = behavior_characterization(level_b)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.84])
    diff = bc_a - bc_b
    return float(np.sqrt(np.sum(weights * diff * diff)) / np.sqrt(weights.sum()))


'''

NB_PATH = Path('improve2.ipynb')
nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

target_cell = None
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == '936b2eef':
        target_cell = cell
        target_idx = i
        break

if target_cell is None:
    sys.exit('cell 936b2eef not found')

src = target_cell['source']
text = ''.join(src) if isinstance(src, list) else src

start_marker = '# -----------------------------------------------------\n# 4.10 Behavior Characterization (BC)'
end_marker = 'def interior_wall_density'

i_start = text.find(start_marker)
if i_start == -1:
    # Try alternate marker
    i_start = text.find('def behavior_characterization')
    if i_start == -1:
        sys.exit('behavior_characterization not found')
    # back up to comment block above (search for last newline-newline before)
    i_block_start = text.rfind('\n\n# ', 0, i_start)
    i_start = i_block_start + 2 if i_block_start != -1 else i_start

i_end = text.find(end_marker)
if i_end == -1:
    sys.exit('interior_wall_density not found')

before = text[:i_start]
after = text[i_end:]

# Build replacement: header comment + new code
header = '''# -----------------------------------------------------
# 4.10 Behavior Characterization (BC) — Novelty Search chuan (Lehman 2011)
# WINNER from BC tuning research (2026-04-26/27, .lab/summary.md):
#   - 7-dim BC: [wall_dens, path_norm, dead_norm, branch_norm, regions_norm, astar_diff, turns_norm]
#   - bc_distance: weighted Euclidean, weights=[1,1,1,1,1,1,7.84] (turns 2.8x effective)
#   - Result: non_l_pattern_rate 0.20 -> 0.88 (96% L-pattern reduction), solvability 0.78 -> 0.90
# -----------------------------------------------------
'''

new_text = before + header + NEW_BC_CODE + after

# Backup
backup = NB_PATH.with_suffix('.ipynb.bak')
backup.write_bytes(NB_PATH.read_bytes())
print(f'backup saved: {backup}')

# Write back as list of lines for nbformat compatibility
target_cell['source'] = [line + '\n' for line in new_text.split('\n')[:-1]] + ([new_text.split('\n')[-1]] if not new_text.endswith('\n') else [])
# Clear outputs of this cell since code changed
target_cell['outputs'] = []
target_cell['execution_count'] = None

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'OK: synced cell {target_idx} (id 936b2eef)')
print(f'old length: {len(text)} chars')
print(f'new length: {len(new_text)} chars')
