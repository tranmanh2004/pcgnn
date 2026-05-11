"""Re-evaluate T13 archive maps with FIXED P/E at corners (0,0) -> (13,13)."""
import pickle, heapq, json, sys
import numpy as np

WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3

def _astar(level, start, end):
    h, w = level.shape
    cnt = 0
    heap = [(abs(start[0]-end[0])+abs(start[1]-end[1]), cnt, start)]
    came, g = {start: None}, {start: 0}
    vis = set()
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in vis: continue
        vis.add(cur)
        if cur == end:
            path, node = [], end
            while node: path.append(node); node = came[node]
            return list(reversed(path))
        r, c = cur
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and (nr,nc) not in vis and level[nr,nc]!=WALL:
                ng = g[cur]+1
                if ng < g.get((nr,nc), 1e9):
                    came[(nr,nc)] = cur; g[(nr,nc)] = ng; cnt+=1
                    heapq.heappush(heap, (ng+abs(nr-end[0])+abs(nc-end[1]), cnt, (nr,nc)))
    return None

pkl_path = sys.argv[1]
with open(pkl_path, 'rb') as f:
    maps = pickle.load(f)

results = []
for lvl in maps:
    lvl = lvl.copy()
    h, w = lvl.shape
    # replace any existing P/E with floor, place fixed corners
    lvl[lvl == PLAYER] = FLOOR
    lvl[lvl == ENEMY] = FLOOR
    if lvl[0, 0] != WALL: lvl[0, 0] = PLAYER
    if lvl[h-1, w-1] != WALL: lvl[h-1, w-1] = ENEMY
    start = (0, 0); end = (h-1, w-1)
    if lvl[0,0] == WALL or lvl[h-1,w-1] == WALL:
        results.append({'solv': 0, 'tort': None}); continue
    path = _astar(lvl, start, end)
    if path is None or len(path) < 2:
        results.append({'solv': 0, 'tort': None}); continue
    manhattan = abs(end[0]-start[0]) + abs(end[1]-start[1])
    tort = (len(path)-1) / manhattan if manhattan > 0 else None
    results.append({'solv': 1, 'tort': tort})

n = len(results)
solv = sum(r['solv'] for r in results) / n
tort_vals = [r['tort'] for r in results if r['tort'] is not None]
mean_tort = np.mean(tort_vals) if tort_vals else 0
high_tort = sum(1 for t in tort_vals if t > 1.5) / n
print(f"Fixed P/E corner evaluation ({n} maps):")
print(f"  solvability:       {solv*100:.1f}%")
print(f"  mean_tortuosity:   {mean_tort:.4f}")
print(f"  high_tort_rate:    {high_tort*100:.1f}%  (tortuosity>1.5)")
