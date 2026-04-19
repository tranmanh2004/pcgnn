"""Inspect actual maze levels from paper PCGNN seed results."""
import pickle, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external/gym-pcgrl'))

SEED_FILE = '../results/experiments/experiment_105_a/Maze/NEAT/2021-10-31_13-23-23/50/200/1/seed_1_name_experiment_105_a_2021-10-31_13-42-14.p'

with open(SEED_FILE, 'rb') as f:
    res = pickle.load(f)

levels = res['levels']
lvl0 = levels[0]
empty_val  = lvl0.tile_types_reversed['empty']   # 0 = floor
filled_val = lvl0.tile_types_reversed['filled']  # 1 = wall

print(f"Số levels: {len(levels)}")
print(f"tile_types: {lvl0.tile_types}  (empty={empty_val}, filled={filled_val})")
print(f"Map shape: {lvl0.map.shape}")

# start/end defaults trong do_astar_from_level khi không có attribute
# start = (0, 0) dạng (x=col, y=row) → map[y,x] = map[0,0]
# end   = (width-1, height-1)       → map[h-1,w-1]

wall_densities  = []
start_is_floor  = 0
end_is_floor    = 0

for l in levels:
    m  = l.map
    h, w = m.shape

    interior = m[1:-1, 1:-1]
    wd = float((interior == filled_val).sum()) / interior.size
    wall_densities.append(wd)

    # start=(x=0,y=0) → map[0,0];  end=(x=w-1,y=h-1) → map[h-1,w-1]
    if m[0, 0] == empty_val:
        start_is_floor += 1
    if m[h-1, w-1] == empty_val:
        end_is_floor += 1

n = len(levels)
print(f"\n{'='*50}")
print(f"  Phân tích {n} levels — PCGNN paper (seed 1)")
print(f"{'='*50}")
print(f"  Wall density (interior, 1=wall): {np.mean(wall_densities):.3f} ± {np.std(wall_densities):.3f}")
print(f"  Start (0,0)  là FLOOR: {start_is_floor}/{n} ({100*start_is_floor/n:.0f}%)")
print(f"  End ({h-1},{w-1}) là FLOOR: {end_is_floor}/{n}   ({100*end_is_floor/n:.0f}%)")
print(f"  So sánh notebook: wall_density≈0.517, start/end luôn FLOOR (100%)")

print("\n=== Sample maze 1 (. = floor, # = wall) ===")
m = levels[0].map
for row in m:
    print('  ' + ''.join('.' if v == empty_val else '#' for v in row))

print("\n=== Sample maze 2 ===")
m = levels[5].map
for row in m:
    print('  ' + ''.join('.' if v == empty_val else '#' for v in row))
