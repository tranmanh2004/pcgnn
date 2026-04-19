"""
Xuất metrics từ pickle ra CSV để xem bằng Excel/VSCode.
Chạy: ./run.sh export_metrics.py

Output:
  output/metrics_summary.csv   — bảng tổng hợp mean±std (như paper)
  output/metrics_per_level.csv — giá trị thô từng level
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external/gym-pcgrl'))

import pickle
import numpy as np
import pandas as pd

PICKLES = {
    'Maze':  'results/maze/metrics_and_levels/2021-10-31_13-47-17/data.p',
    'Mario': 'results/mario/metrics_and_levels/2021-10-29_07-43-24/data.p',
}

METHOD_LABELS = {
    'experiment_105_a':                              'PCGNN',
    'experiment_204e':                               'PCGNN',
    'experiment_102_f_visual_diversity_rerun_batch': 'DirectGA (Novelty)',
    'experiment_102_aaa_rerun_only_best':            'DirectGA+',
    'all_pcgrl/binary/wide':                         'PCGRL (Wide)',
    'all_pcgrl/binary/turtle':                       'PCGRL (Turtle)',
}

METRIC_DISPLAY = {
    'SolvabilityMetric':               'Solvability',
    'CompressionDistanceMetric':       'Compression Distance',
    'AStarDiversityMetric':            'A* Diversity',
    'AStarEditDistanceDiversityMetric':'A* Edit Diversity',
    'AStarDifficultyMetric':           'A* Difficulty',
    'LeniencyMetric':                  'Leniency',
    'generation_time':                 'Gen Time (s)',
    'train_time':                      'Train Time (s)',
}

def get_method_label(path):
    for k, v in METHOD_LABELS.items():
        if k in path:
            return v
    return path.split('/')[-3] if path.count('/') >= 3 else path

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def build_summary_rows(game, data):
    rows = []
    for exp_path, info in data.items():
        method = get_method_label(exp_path)
        results = info.get('results_single', {})
        for metric_key, display_name in METRIC_DISPLAY.items():
            if metric_key not in results:
                continue
            vals = [v for v in results[metric_key] if v is not None]
            if not vals:
                continue
            row = {
                'game':   game,
                'method': method,
                'metric': display_name,
                'mean':   round(np.mean(vals), 6),
                'std':    round(np.std(vals), 6),
            }
            for i, v in enumerate(vals):
                row[f'seed_{i}'] = round(v, 6)
            rows.append(row)
    return rows

def build_per_level_rows(game, data):
    rows = []
    for exp_path, info in data.items():
        method = get_method_label(exp_path)
        results_all = info.get('results_all', {})
        for metric_key, display_name in METRIC_DISPLAY.items():
            if metric_key not in results_all:
                continue
            for seed_idx, level_vals in enumerate(results_all[metric_key]):
                if not hasattr(level_vals, '__iter__'):
                    level_vals = [level_vals]
                for level_idx, val in enumerate(level_vals):
                    rows.append({
                        'game':      game,
                        'method':    method,
                        'metric':    display_name,
                        'seed':      seed_idx,
                        'level_idx': level_idx,
                        'value':     round(float(val), 6) if val is not None else None,
                    })
    return rows

def main():
    os.makedirs('output', exist_ok=True)
    summary_rows = []
    per_level_rows = []

    for game, path in PICKLES.items():
        if not os.path.exists(path):
            print(f"Không tìm thấy: {path}")
            continue
        print(f"Loading {game} pickle...")
        data = load_pickle(path)
        summary_rows.extend(build_summary_rows(game, data))
        per_level_rows.extend(build_per_level_rows(game, data))

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('output/metrics_summary.csv', index=False)
    print(f"\nDa luu: output/metrics_summary.csv  ({len(df_summary)} rows)")

    df_per_level = pd.DataFrame(per_level_rows)
    df_per_level.to_csv('output/metrics_per_level.csv', index=False)
    print(f"Da luu: output/metrics_per_level.csv ({len(df_per_level)} rows)")

    print("\n=== Preview metrics_summary.csv ===")
    print(df_summary.to_string(index=False))

if __name__ == '__main__':
    main()
