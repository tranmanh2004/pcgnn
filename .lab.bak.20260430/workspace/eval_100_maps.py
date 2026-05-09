"""Detailed evaluation of 100 maps generated from trained winner."""
import sys
import json
import pickle
import random
from pathlib import Path
from collections import Counter

import numpy as np
import neat

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import (
    generate_level, count_path_turns, _get_astar_result,
    is_solvable, shortest_path_bfs, astar_difficulty, astar_diversity,
    behavior_characterization, bc_distance,
    WALL, FLOOR, PLAYER, ENEMY,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PKL_PATH = PROJECT_ROOT / "neat_winner_seed0 (1).pkl"
NEAT_CONFIG_PATH = PROJECT_ROOT / ".lab" / "workspace" / "exp-5" / "config.txt"
OUT_PATH = PROJECT_ROOT / ".lab" / "workspace" / "test_output" / "eval_100.json"

N_MAPS = 100
SEED = 42

print(f"Loading {PKL_PATH.name}...")
with open(PKL_PATH, "rb") as f:
    winner = pickle.load(f)

config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    str(NEAT_CONFIG_PATH),
)
net = neat.nn.FeedForwardNetwork.create(winner, config)

random.seed(SEED)
np.random.seed(SEED)
print(f"Generating {N_MAPS} maps (seed={SEED})...")
maps = [generate_level(net) for _ in range(N_MAPS)]


# ─────────── PER-MAP ANALYSIS ───────────
def detect_diagonal_stripes(level):
    """Heuristic: count cells where wall and diagonal-wall are paired (NW-SE diagonal lines)."""
    h, w = level.shape
    walls = (level == WALL)
    if walls.sum() < 10:
        return 0.0
    diag_pairs = 0
    total_walls = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if walls[r, c]:
                total_walls += 1
                if walls[r + 1, c + 1]:
                    diag_pairs += 1
    return diag_pairs / max(1, total_walls)


def count_regions(level):
    from collections import deque
    h, w = level.shape
    walkable = (level != WALL)
    seen = np.zeros_like(walkable, dtype=bool)
    regions = 0
    for r0 in range(h):
        for c0 in range(w):
            if walkable[r0, c0] and not seen[r0, c0]:
                regions += 1
                q = deque([(r0, c0)])
                seen[r0, c0] = True
                while q:
                    r, c = q.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and walkable[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
    return regions


print("Per-map analysis...")
per_map = []
for i, lvl in enumerate(maps):
    turns = count_path_turns(lvl)
    sp = shortest_path_bfs(lvl)
    wall_count = int((lvl == WALL).sum())
    wall_dens = wall_count / lvl.size
    per_map.append({
        "id": i + 1,
        "turns": int(turns),
        "solvable": int(turns >= 0),
        "path_len": int(sp) if sp else 0,
        "wall_count": wall_count,
        "wall_density": float(wall_dens),
        "regions": int(count_regions(lvl)),
        "diff": float(astar_difficulty(lvl)),
        "diag_score": float(detect_diagonal_stripes(lvl)),
    })


# ─────────── CLASSIFY ───────────
def classify(m):
    if m["wall_density"] < 0.1:
        return "near-empty"
    if m["wall_density"] > 0.7:
        return "near-full"
    if not m["solvable"]:
        return "unsolvable"
    if m["turns"] <= 1:
        return "L-pattern"
    if m["diag_score"] > 0.4:
        return "diagonal-stripes"
    if m["regions"] > 3:
        return "fragmented"
    if 2 <= m["turns"] <= 7:
        return "moderate-organic"
    if m["turns"] >= 8:
        return "very-zigzag"
    return "other"


for m in per_map:
    m["class"] = classify(m)

class_counter = Counter(m["class"] for m in per_map)


# ─────────── AGGREGATE ───────────
solvable = [m for m in per_map if m["solvable"]]
def agg(key, src=per_map):
    vals = [m[key] for m in src]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "median": float(np.median(vals)),
    }

# Pairwise diversity (sample 30 random pairs to keep fast)
print("Pairwise diversity (50 random pairs)...")
random.seed(0)
pairs = [(random.randrange(N_MAPS), random.randrange(N_MAPS)) for _ in range(50)]
pairs = [(a, b) for a, b in pairs if a != b][:50]
astar_div_pairs = [astar_diversity(maps[a], maps[b]) for a, b in pairs]
bc_div_pairs = [bc_distance(maps[a], maps[b]) for a, b in pairs]


# ─────────── REPORT ───────────
report = {
    "n_maps": N_MAPS,
    "seed": SEED,
    "pkl": PKL_PATH.name,
    "summary": {
        "solvability_rate": sum(m["solvable"] for m in per_map) / N_MAPS,
        "non_l_pattern_rate": sum(1 for m in per_map if m["turns"] >= 2) / N_MAPS,
        "L_pattern_count": sum(1 for m in per_map if 0 <= m["turns"] <= 1),
        "unsolvable_count": sum(1 for m in per_map if m["turns"] == -1),
        "diagonal_stripes_count": class_counter.get("diagonal-stripes", 0),
        "near_empty_count": class_counter.get("near-empty", 0),
        "near_full_count": class_counter.get("near-full", 0),
    },
    "classification": dict(class_counter.most_common()),
    "stats_solvable_only": {
        "turns": agg("turns", solvable),
        "path_len": agg("path_len", solvable),
        "diff": agg("diff", solvable),
    },
    "stats_all": {
        "wall_density": agg("wall_density"),
        "regions": agg("regions"),
        "diag_score": agg("diag_score"),
    },
    "pairwise_diversity": {
        "astar_trajectory": {
            "mean": float(np.mean(astar_div_pairs)),
            "std": float(np.std(astar_div_pairs)),
        },
        "bc_distance": {
            "mean": float(np.mean(bc_div_pairs)),
            "std": float(np.std(bc_div_pairs)),
        },
    },
    "per_map_first_20": per_map[:20],
}

with open(OUT_PATH, "w") as f:
    json.dump(report, f, indent=2)

# ─────────── PRINT ───────────
print()
print("=" * 60)
print(f"Evaluation of {N_MAPS} maps (seed={SEED})")
print("=" * 60)
s = report["summary"]
print(f"\nSolvability rate     : {s['solvability_rate']*100:.1f}% ({sum(m['solvable'] for m in per_map)}/{N_MAPS})")
print(f"Non-L-pattern rate   : {s['non_l_pattern_rate']*100:.1f}% ({sum(1 for m in per_map if m['turns']>=2)}/{N_MAPS})")
print(f"L-pattern (<=1 turn) : {s['L_pattern_count']}/{N_MAPS}")
print(f"Unsolvable           : {s['unsolvable_count']}/{N_MAPS}")
print(f"Diagonal stripes     : {s['diagonal_stripes_count']}/{N_MAPS}")
print(f"Near-empty maze      : {s['near_empty_count']}/{N_MAPS}")
print(f"Near-full maze       : {s['near_full_count']}/{N_MAPS}")

print(f"\n--- Classification ---")
for cls, cnt in class_counter.most_common():
    bar = "#" * cnt
    print(f"  {cls:<18}: {cnt:3d} {bar}")

st = report["stats_solvable_only"]
print(f"\n--- Stats (solvable maps only) ---")
print(f"  Turns        : mean={st['turns']['mean']:5.2f}  median={st['turns']['median']:5.1f}  range=[{st['turns']['min']:.0f}, {st['turns']['max']:.0f}]")
print(f"  Path length  : mean={st['path_len']['mean']:5.2f}  median={st['path_len']['median']:5.1f}  range=[{st['path_len']['min']:.0f}, {st['path_len']['max']:.0f}]")
print(f"  A* difficulty: mean={st['diff']['mean']:.3f}  std={st['diff']['std']:.3f}")

sa = report["stats_all"]
print(f"\n--- Stats (all maps) ---")
print(f"  Wall density : mean={sa['wall_density']['mean']:.3f}  std={sa['wall_density']['std']:.3f}  range=[{sa['wall_density']['min']:.3f}, {sa['wall_density']['max']:.3f}]")
print(f"  Regions      : mean={sa['regions']['mean']:.2f}  median={sa['regions']['median']:.0f}  max={sa['regions']['max']:.0f}")
print(f"  Diagonal score: mean={sa['diag_score']['mean']:.3f}")

pd = report["pairwise_diversity"]
print(f"\n--- Diversity (50 random pairs) ---")
print(f"  A* trajectory dist: mean={pd['astar_trajectory']['mean']:.3f}  std={pd['astar_trajectory']['std']:.3f}")
print(f"  BC distance       : mean={pd['bc_distance']['mean']:.3f}  std={pd['bc_distance']['std']:.3f}")

print(f"\nFull report: {OUT_PATH}")
