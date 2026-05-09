import pickle, sys
sys.path.insert(0, 'D:/Project/PCGNN/src')
sys.path.insert(0, 'D:/Project/PCGNN/external/gym-pcgrl')
import os; os.chdir('D:/Project/PCGNN/src')

import neat

files = {
    'first.pkl (50gen)': '../first.pkl',
    'seed0 (100gen)': '../neat_winner_seed0 (5).pkl',
    'seed1 (100gen)': '../neat_winner_seed1 (1).pkl',
    'seed2 (100gen)': '../neat_winner_seed2 (1).pkl',
    'seed3 (100gen)': '../neat_winner_seed3 (1).pkl',
    'seed4 (100gen)': '../neat_winner_seed4 (1).pkl',
}

for name, path in files.items():
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f'{name}: nodes={len(data.nodes)}, conns={len(data.connections)}, fitness={data.fitness:.4f}')
