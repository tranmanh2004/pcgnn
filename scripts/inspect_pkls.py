from pathlib import Path
import pickle
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src" / "external" / "gym-pcgrl"))

files = {
    "first.pkl (50gen)": PROJECT_DIR / "models" / "first.pkl",
    "seed0 (100gen)": PROJECT_DIR / "models" / "neat_winner_seed0 (5).pkl",
    "seed1 (100gen)": PROJECT_DIR / "models" / "neat_winner_seed1 (1).pkl",
    "seed2 (100gen)": PROJECT_DIR / "models" / "neat_winner_seed2 (1).pkl",
    "seed3 (100gen)": PROJECT_DIR / "models" / "neat_winner_seed3 (1).pkl",
    "seed4 (100gen)": PROJECT_DIR / "models" / "neat_winner_seed4 (1).pkl",
}

for name, path in files.items():
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f'{name}: nodes={len(data.nodes)}, conns={len(data.connections)}, fitness={data.fitness:.4f}')
