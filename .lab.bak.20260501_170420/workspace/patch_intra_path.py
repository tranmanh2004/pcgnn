"""Patch improve2.ipynb: add path shape (astar_edit_diversity) to intra_novelty distance"""
import json

with open('improve2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

NEW_INTRA_FUNCS = """
# ── Intra distance: BC structural + A* path shape ─────────
_aed_cache = {}

def _aed_cached(level_a, level_b) -> float:
    ka = level_a.tobytes(); kb = level_b.tobytes()
    key = (ka, kb) if ka <= kb else (kb, ka)
    if key not in _aed_cache:
        _aed_cache[key] = astar_edit_diversity(level_a, level_b)
    return _aed_cache[key]


def intra_distance(level_a, level_b, alpha=0.6) -> float:
    return alpha * bc_distance(level_a, level_b) \\
         + (1.0 - alpha) * _aed_cached(level_a, level_b)


def reset_aed_cache():
    _aed_cache.clear()

"""

changes = 0
for cell in nb['cells']:
    src = ''.join(cell['source'])

    # Cell 936b2eef: insert new functions after reset_bc_cache()
    if cell['id'] == '936b2eef':
        if 'intra_distance' not in src:
            src = src.replace(
                'def reset_bc_cache():\n    _bc_cache.clear()',
                'def reset_bc_cache():\n    _bc_cache.clear()\n' + NEW_INTRA_FUNCS
            )
            cell['source'] = src
            print('Updated 936b2eef: added _aed_cache + intra_distance + reset_aed_cache')
            changes += 1

    # Cell 092b7fc4: switch intra_novelty_score to use intra_distance
    elif cell['id'] == '092b7fc4':
        if 'intra_distance' not in src:
            src = src.replace(
                '        dists = sorted(bc_distance(lvl_i, lvl_j)\n'
                '                       for j, lvl_j in enumerate(levels) if i != j)',
                '        dists = sorted(intra_distance(lvl_i, lvl_j)\n'
                '                       for j, lvl_j in enumerate(levels) if i != j)'
            )
            cell['source'] = src
            print('Updated 092b7fc4: intra_novelty_score uses intra_distance')
            changes += 1

    # Cell 8367aa57: add reset_aed_cache() to training loop reset
    elif cell['id'] == '8367aa57':
        if 'reset_aed_cache' not in src:
            src = src.replace(
                '    reset_bc_cache()',
                '    reset_bc_cache()\n    reset_aed_cache()'
            )
            cell['source'] = src
            print('Updated 8367aa57: added reset_aed_cache() to training loop')
            changes += 1

with open('improve2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Done. {changes}/3 cells updated.')
