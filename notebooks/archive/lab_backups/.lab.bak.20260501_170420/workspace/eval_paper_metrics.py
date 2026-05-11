"""
Evaluate D3 (recurrent NEAT) vs B17 (feedforward) using paper metrics.
Uses only maze-level modules (no gym dependency).

Run from src/:
  cd src && PYTHONPATH=$(pwd):$(pwd)/external/gym-pcgrl python ../.lab/workspace/eval_paper_metrics.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
# stub out gym_pcgrl so mario imports don't explode
sys.modules.setdefault('gym', type(sys)('gym'))
sys.modules.setdefault('gym.envs', type(sys)('gym.envs'))
sys.modules.setdefault('gym.envs.registration', type(sys)('gym.envs.registration'))

import pickle
import numpy as np
from itertools import combinations

LAB = os.path.join(os.path.dirname(__file__), '..')

def load_maps(name):
    path = os.path.join(LAB, 'workspace', name, 'maps.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

maps_b17 = load_maps('B17')
maps_d3  = load_maps('exp-D3')
print(f"B17: {len(maps_b17)} maps  |  D3: {len(maps_d3)} maps")

# ── tile encoding ──────────────────────────────────────────────────────────
WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3

def binarize(m):
    """passable → 1, wall → 0"""
    return (m != WALL).astype(np.int32)

# ── A* on binarized maze (0=wall, 1=floor) ─────────────────────────────────
from collections import defaultdict
from queue import PriorityQueue

def a_star_maze(grid, start=(0,0), goal=None):
    """Returns (path_or_None, nodes_expanded).  grid: 0=wall, 1=floor."""
    h, w = grid.shape
    if goal is None:
        goal = (w-1, h-1)

    if grid[start[1], start[0]] == 0:
        return None, 0

    def heur(pos):
        return abs(pos[0]-goal[0]) + abs(pos[1]-goal[1])

    def neighbours(pos):
        x, y = pos
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] != 0:
                yield (nx, ny)

    open_q = PriorityQueue()
    open_q.put((heur(start), 0, start))
    g = defaultdict(lambda: float('inf'))
    g[start] = 0
    came = {}
    visited = set()
    nodes = 0

    while not open_q.empty():
        _, cost, cur = open_q.get()
        if cur in visited:
            continue
        visited.add(cur)
        nodes += 1
        if cur == goal:
            path = []
            while cur in came:
                path.append(cur); cur = came[cur]
            path.append(start)
            return list(reversed(path)), nodes
        for nb in neighbours(cur):
            ng = cost + 1
            if ng < g[nb]:
                g[nb] = ng
                came[nb] = cur
                open_q.put((ng + heur(nb), ng, nb))
    return None, nodes

# ── trajectory distance ─────────────────────────────────────────────────────
def traj_distance(t1, t2):
    """Normalised position-set symmetric difference."""
    if t1 is None and t2 is None: return 0.0
    if t1 is None or  t2 is None: return 1.0
    s1, s2 = set(t1), set(t2)
    return len(s1.symmetric_difference(s2)) / max(len(s1 | s2), 1)

def edit_distance(t1, t2):
    """Edit distance on trajectory sequences, normalised by max length."""
    if t1 is None and t2 is None: return 0.0
    if t1 is None or  t2 is None: return 1.0
    n, m = len(t1), len(t2)
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            prev, dp[j] = dp[j], prev if t1[i-1]==t2[j-1] else 1 + min(prev, dp[j], dp[j-1])
    return dp[m] / max(n, m)

# ── evaluate one set of maps ───────────────────────────────────────────────
def evaluate(maps, label, n_pairs=200, seed=42):
    rng = np.random.default_rng(seed)
    bins = [binarize(m) for m in maps]
    n = len(maps)

    paths = [a_star_maze(b) for b in bins]
    trajs = [p for p, _ in paths]
    diffs = [d for _, d in paths]
    solvable = [1 if t is not None else 0 for t in trajs]

    # sample pairs for diversity
    all_pairs = list(combinations(range(n), 2))
    if len(all_pairs) > n_pairs:
        idx = rng.choice(len(all_pairs), n_pairs, replace=False)
        pairs = [all_pairs[i] for i in idx]
    else:
        pairs = all_pairs

    div   = np.mean([traj_distance(trajs[i], trajs[j]) for i,j in pairs])
    ed    = np.mean([edit_distance(trajs[i],  trajs[j]) for i,j in pairs])

    # visual diversity
    vd = np.mean([np.mean(bins[i] != bins[j]) for i,j in pairs])

    sol = np.mean(solvable)
    dif = np.mean([d for d in diffs if d > 0])
    path_len = np.mean([len(t) for t in trajs if t])

    print(f"\n{label}:")
    print(f"  Solvability:       {sol:.4f}  ({int(sum(solvable))}/{n})")
    print(f"  A* Diversity:      {div:.4f}")
    print(f"  A* Edit Diversity: {ed:.4f}")
    print(f"  Visual Diversity:  {vd:.4f}")
    print(f"  A* Difficulty:     {dif:.1f}  nodes expanded (mean)")
    print(f"  Path Length:       {path_len:.1f}  steps (mean)")

    return dict(label=label, sol=sol, div=div, ed=ed, vd=vd, dif=dif, path_len=path_len)

r_b17 = evaluate(maps_b17, 'B17 (feedforward, 50 gen)')
r_d3  = evaluate(maps_d3,  'D3  (recurrent,  200 gen)')

print("\n\n=== COMPARISON TABLE ===")
print(f"{'Metric':<24} {'B17':>8} {'D3':>8} {'Δ (D3−B17)':>13}")
print("-" * 57)
for key, label in [
    ('sol',      'Solvability'),
    ('div',      'A* Diversity'),
    ('ed',       'A* Edit Diversity'),
    ('vd',       'Visual Diversity'),
    ('dif',      'A* Difficulty'),
    ('path_len', 'Path Length'),
]:
    b = r_b17[key]; d = r_d3[key]
    print(f"  {label:<22} {b:8.4f} {d:8.4f} {d-b:+13.4f}")
