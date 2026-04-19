"""
Đọc và in metrics từ pickle đã được tính sẵn.
Chạy: ./run.sh show_metrics.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external/gym-pcgrl'))
import pickle
import numpy as np
from pprint import pprint

MAZE_PICKLE = 'results/maze/metrics_and_levels/2021-10-31_13-47-17/data.p'
MARIO_PICKLE = 'results/mario/metrics_and_levels/2021-10-29_07-43-24/data.p'

METRIC_DISPLAY = {
    'SolvabilityMetric':              'Solvability',
    'CompressionDistanceMetric':      'Compression Distance',
    'AStarDiversityMetric':           'A* Diversity',
    'AStarEditDistanceDiversityMetric': 'A* Edit Diversity',
    'AStarDifficultyMetric':          'A* Difficulty',
    'LeniencyMetric':                 'Leniency',
    'generation_time':                'Gen Time (s)',
    'train_time':                     'Train Time (s)',
}

METHOD_LABELS = {
    'experiment_105_a':                                         'PCGNN',
    'experiment_102_f_visual_diversity_rerun_batch':            'DirectGA (Novelty)',
    'experiment_102_aaa_rerun_only_best':                       'DirectGA+',
    'all_pcgrl/binary/wide':                                    'PCGRL (Wide)',
    'all_pcgrl/binary/turtle':                                  'PCGRL (Turtle)',
}

def label(key):
    for k, v in METHOD_LABELS.items():
        if k in key:
            return v
    return key.split('/')[-3] if '/' in key else key

def show(pickle_path, title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    for path, info in data.items():
        method = label(path)
        results = info['results_single']
        print(f"\n[{method}]")
        for metric_key, display_name in METRIC_DISPLAY.items():
            if metric_key in results:
                vals = results[metric_key]
                # Bỏ None/0 cho các metric bị lỗi
                valid = [v for v in vals if v is not None]
                if valid:
                    mean = np.mean(valid)
                    std  = np.std(valid)
                    print(f"  {display_name:<30} {mean:.4f}  ± {std:.4f}")

if __name__ == '__main__':
    show(MAZE_PICKLE,  'MAZE Metrics')
    show(MARIO_PICKLE, 'MARIO Metrics')
