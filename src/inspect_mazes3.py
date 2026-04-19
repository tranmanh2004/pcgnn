"""Load individual seed pickle to get actual maze levels."""
import pickle, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external/gym-pcgrl'))

PICKLE = 'results/maze/metrics_and_levels/2021-10-31_13-47-17/data.p'

with open(PICKLE, 'rb') as f:
    data = pickle.load(f)

key = list(data.keys())[0]
info = data[key]

# Thử các đường dẫn khác nhau cho file seed
names = info['names']
found_file = None
for name in names[:3]:
    candidates = [
        name,
        name.replace('../', 'src/'),
        name.replace('../results/', 'results/'),
        os.path.join('results', os.path.basename(os.path.dirname(name)), os.path.basename(name)),
    ]
    for c in candidates:
        if os.path.exists(c):
            found_file = c
            break
    if found_file:
        break

if found_file:
    print(f"Found: {found_file}")
    with open(found_file, 'rb') as f:
        res = pickle.load(f)
    print(f"Type: {type(res)}")
    if isinstance(res, dict):
        print(f"Keys: {list(res.keys())[:10]}")
    elif hasattr(res, '__dict__'):
        print(f"Attrs: {list(res.__dict__.keys())[:10]}")
        if hasattr(res, 'levels'):
            print(f"Levels type: {type(res.levels)}, len: {len(res.levels)}")
else:
    print("Individual files not found. Searching for maze files...")
    # Search for any pickle with maze levels
    for root, dirs, files in os.walk('results/experiments/experiment_105_a'):
        for f in files:
            if f.endswith('.p'):
                fpath = os.path.join(root, f)
                print(f"  Found: {fpath}")
                try:
                    with open(fpath, 'rb') as fp:
                        obj = pickle.load(fp)
                    print(f"    Type: {type(obj)}")
                    if hasattr(obj, 'levels') and obj.levels:
                        lvls = obj.levels
                        print(f"    Levels: {len(lvls)}, type: {type(lvls[0])}")
                        if hasattr(lvls[0], 'map'):
                            m = lvls[0].map
                            h, w = m.shape
                            wall_d = float((m[1:-1,1:-1] == 1).sum()) / ((h-2)*(w-2))
                            print(f"    Map shape: {m.shape}")
                            print(f"    Wall density (1=wall, interior): {wall_d:.3f}")
                            print(f"    Unique values: {np.unique(m)}")
                            print(f"    Start={lvls[0].start}, End={lvls[0].end}")
                            print(f"    start tile={m[lvls[0].start[1], lvls[0].start[0]]}")
                            print(f"    end tile={m[lvls[0].end[1], lvls[0].end[0]]}")
                            print()
                            print("    Sample maze (. = floor=0, # = wall=1):")
                            for row in m:
                                print("    " + ''.join('.' if v == 0 else '#' for v in row))
                        break
                except Exception as e:
                    print(f"    Error: {e}")
                break
        else:
            continue
        break
