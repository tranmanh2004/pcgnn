"""
Kiểm tra cấu trúc maze gốc từ pickle của paper PCGNN.
Chạy: ./run.sh inspect_mazes.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external/gym-pcgrl'))

import pickle
import numpy as np

PICKLE = 'results/maze/metrics_and_levels/2021-10-31_13-47-17/data.p'

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def inspect_experiment(exp_path, info):
    print(f"\n{'='*60}")
    print(f"  {exp_path.split('/')[-3] if exp_path.count('/') >= 3 else exp_path}")
    print(f"{'='*60}")

    # Thử lấy levels từ các field có thể có
    levels = None
    for field in ['levels', 'all_levels', 'parent_levels']:
        if field in info:
            levels = info[field]
            print(f"  Field '{field}' found: {type(levels)}")
            break

    if levels is None:
        # Thử đọc từ individual result pickles
        print(f"  Keys available: {list(info.keys())}")
        # In thông tin về results_single nếu có
        if 'results_single' in info:
            rs = info['results_single']
            print(f"  results_single keys: {list(rs.keys())[:10]}")
        return

    # Phân tích levels
    all_maps = []
    if isinstance(levels, list):
        for seed_levels in levels:
            if isinstance(seed_levels, list):
                for lvl in seed_levels:
                    if hasattr(lvl, 'map'):
                        all_maps.append(lvl.map)
            elif hasattr(seed_levels, 'map'):
                all_maps.append(seed_levels.map)

    if not all_maps:
        print(f"  Không tìm được maps từ levels")
        return

    print(f"  Số levels: {len(all_maps)}")

    # Kích thước
    shapes = set(m.shape for m in all_maps)
    print(f"  Kích thước (h, w): {shapes}")

    # Tile encoding
    all_vals = set(np.unique(m) for m in all_maps[:5])
    print(f"  Unique tile values (sample): {all_vals}")

    # Wall density (giả sử 1 = wall/filled, 0 = floor/empty theo MazeLevel)
    wall_densities = []
    start_is_floor = 0
    end_is_floor = 0

    for m in all_maps:
        h, w = m.shape
        interior = m[1:-1, 1:-1]
        wall_density = float((interior == 1).sum() / interior.size)
        wall_densities.append(wall_density)

        # Check start=(0,0) và end=(w-1,h-1) trong (x,y) → map[y,x] = map[0,0] và map[h-1,w-1]
        if m[0, 0] == 0:   # 0=empty=FLOOR trong MazeLevel
            start_is_floor += 1
        if m[h-1, w-1] == 0:
            end_is_floor += 1

    n = len(all_maps)
    print(f"\n  Wall density (interior): {np.mean(wall_densities):.3f} ± {np.std(wall_densities):.3f}")
    print(f"  Start (0,0)   là FLOOR : {start_is_floor}/{n} ({100*start_is_floor/n:.0f}%)")
    print(f"  End (h-1,w-1) là FLOOR : {end_is_floor}/{n} ({100*end_is_floor/n:.0f}%)")

    # Visualize 1 maze mẫu
    sample = all_maps[0]
    h, w = sample.shape
    print(f"\n  Maze mẫu ({h}×{w}), 0=floor '.', 1=wall '#':")
    for row in sample:
        print("  " + ''.join('.' if v == 0 else '#' for v in row))


def main():
    if not os.path.exists(PICKLE):
        print(f"Không tìm thấy: {PICKLE}")
        return

    print(f"Loading pickle...")
    data = load_pickle(PICKLE)
    print(f"Số experiments: {len(data)}")
    print(f"Keys: {list(data.keys())[:5]}")

    # Tập trung vào PCGNN (experiment_105_a)
    for exp_path, info in data.items():
        if 'experiment_105_a' in exp_path or 'pcgrl' in exp_path.lower():
            inspect_experiment(exp_path, info)
            break  # Chỉ xem 1 experiment đầu tiên

    # Nếu không tìm thấy 105_a thì xem experiment đầu tiên
    print("\n\n=== Tất cả experiments ===")
    for exp_path in data.keys():
        label = exp_path.split('/')[-3] if exp_path.count('/') >= 3 else exp_path
        print(f"  {label}")


if __name__ == '__main__':
    main()
