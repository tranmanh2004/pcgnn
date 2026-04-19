"""Inspect paper maze results and individual level files."""
import pickle, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external/gym-pcgrl'))

PICKLE = 'results/maze/metrics_and_levels/2021-10-31_13-47-17/data.p'

with open(PICKLE, 'rb') as f:
    data = pickle.load(f)

key = list(data.keys())[0]
info = data[key]
print('=== PCGNN experiment key ===')
print(key)
print()

print('=== names (individual result files) ===')
names = info['names']
print(f'  Count: {len(names)}')
print(f'  Sample: {names[:2]}')
print()

print('=== results_all keys ===')
results_all = info['results_all']
for k, v in results_all.items():
    print(f'  {k}: {len(v)} seeds, type={type(v[0]).__name__}')
print()

print('=== AStarDifficulty per-level values (seed 0) ===')
diff_vals = results_all.get('AStarDifficultyMetric', [[]])[0]
print(f'  Count: {len(diff_vals)}')
print(f'  Values (first 20): {[round(v,4) for v in diff_vals[:20]]}')
print(f'  Mean: {np.mean(diff_vals):.4f}')
print(f'  Zeros: {sum(1 for v in diff_vals if v == 0)} / {len(diff_vals)}')
print()

print('=== Leniency per-level values (seed 0) ===')
len_vals = results_all.get('LeniencyMetric', [[]])[0]
print(f'  Count: {len(len_vals)}')
print(f'  Mean: {np.mean(len_vals):.4f}')
print()

# Try loading an individual result file to get actual levels
if names:
    sample_name = names[0]
    if os.path.exists(sample_name):
        print(f'=== Loading individual result: {sample_name} ===')
        with open(sample_name, 'rb') as f:
            res = pickle.load(f)
        print(f'  Type: {type(res)}')
        if hasattr(res, '__dict__'):
            print(f'  Attrs: {list(res.__dict__.keys())[:10]}')
        if hasattr(res, 'levels') and res.levels:
            lvl = res.levels[0]
            print(f'  Level type: {type(lvl)}')
            if hasattr(lvl, 'map'):
                m = lvl.map
                h, w = m.shape
                wall_d = (m[1:-1,1:-1] == 1).sum() / (h-2) / (w-2)
                print(f'  Map shape: {m.shape}')
                print(f'  Unique values: {np.unique(m)}')
                print(f'  Wall density (interior, 1=wall): {wall_d:.3f}')
                print(f'  Start (0,0) tile: {m[0,0]} (0=floor)')
                print(f'  End ({h-1},{w-1}) tile: {m[h-1,w-1]} (0=floor)')
                print(f'  level.start: {lvl.start}')
                print(f'  level.end: {lvl.end}')
                print()
                print('  Sample maze (. = floor, # = wall):')
                for row in m:
                    print('  ' + ''.join('.' if v == 0 else '#' for v in row))
    else:
        print(f'Individual file not found: {sample_name}')
        # Try relative path
        base = os.path.dirname(PICKLE)
        alt = os.path.join(base, os.path.basename(sample_name))
        print(f'Trying: {alt}, exists={os.path.exists(alt)}')
