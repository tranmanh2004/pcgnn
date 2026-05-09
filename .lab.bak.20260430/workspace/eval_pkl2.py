"""Evaluate neat_winner_seed0 (2).pkl — 100 maps + PNG grid."""
import sys, json, pickle, random
from pathlib import Path
from collections import Counter

import numpy as np
import neat
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import (
    generate_level, count_path_turns, _get_astar_result,
    is_solvable, shortest_path_bfs, astar_difficulty, astar_diversity,
    behavior_characterization, bc_distance,
    WALL, FLOOR, PLAYER, ENEMY,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PKL_PATH = PROJECT_ROOT / "neat_winner_seed0 (2).pkl"
NEAT_CONFIG_PATH = PROJECT_ROOT / ".lab" / "workspace" / "exp-5" / "config.txt"
OUT_DIR = PROJECT_ROOT / ".lab" / "workspace" / "eval_pkl2_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_MAPS = 100
SEED = 42

print(f"Loading {PKL_PATH.name} ...")
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
print(f"Generating {N_MAPS} maps (seed={SEED}) ...")
maps = [generate_level(net) for _ in range(N_MAPS)]


# ─── helpers ──────────────────────────────────────────────────────────────────
def detect_diagonal_stripes(level):
    h, w = level.shape
    walls = (level == WALL)
    if walls.sum() < 10:
        return 0.0
    diag_pairs = total = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if walls[r, c]:
                total += 1
                if walls[r + 1, c + 1]:
                    diag_pairs += 1
    return diag_pairs / max(1, total)


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
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and walkable[nr,nc] and not seen[nr,nc]:
                            seen[nr,nc] = True
                            q.append((nr,nc))
    return regions


# ─── per-map analysis ─────────────────────────────────────────────────────────
print("Per-map analysis ...")
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

# ─── aggregate ────────────────────────────────────────────────────────────────
solvable = [m for m in per_map if m["solvable"]]
def agg(key, src=per_map):
    vals = [m[key] for m in src]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "min": float(np.min(vals)), "max": float(np.max(vals)),
            "median": float(np.median(vals))}

print("Pairwise diversity (50 pairs) ...")
random.seed(0)
pairs = [(a, b) for a, b in [(random.randrange(N_MAPS), random.randrange(N_MAPS)) for _ in range(70)] if a != b][:50]
astar_div_pairs = [astar_diversity(maps[a], maps[b]) for a, b in pairs]
bc_div_pairs = [bc_distance(maps[a], maps[b]) for a, b in pairs]

s = {
    "solvability_rate": sum(m["solvable"] for m in per_map) / N_MAPS,
    "non_l_pattern_rate": sum(1 for m in per_map if m["turns"] >= 2) / N_MAPS,
    "L_pattern_count": sum(1 for m in per_map if 0 <= m["turns"] <= 1),
    "unsolvable_count": sum(1 for m in per_map if m["turns"] == -1),
    "diagonal_stripes_count": class_counter.get("diagonal-stripes", 0),
    "near_empty_count": class_counter.get("near-empty", 0),
    "near_full_count": class_counter.get("near-full", 0),
}
non_l = s["non_l_pattern_rate"]
solv = s["solvability_rate"]
diag_frac = s["diagonal_stripes_count"] / N_MAPS
extreme_frac = (s["near_empty_count"] + s["near_full_count"]) / N_MAPS
compound = non_l * solv * (1 - diag_frac) * (1 - extreme_frac)

report = {
    "pkl": PKL_PATH.name,
    "n_maps": N_MAPS,
    "seed": SEED,
    "summary": s,
    "compound": compound,
    "classification": dict(class_counter.most_common()),
    "stats_solvable_only": {"turns": agg("turns", solvable), "path_len": agg("path_len", solvable), "diff": agg("diff", solvable)},
    "stats_all": {"wall_density": agg("wall_density"), "regions": agg("regions"), "diag_score": agg("diag_score")},
    "pairwise_diversity": {
        "astar_trajectory": {"mean": float(np.mean(astar_div_pairs)), "std": float(np.std(astar_div_pairs))},
        "bc_distance": {"mean": float(np.mean(bc_div_pairs)), "std": float(np.std(bc_div_pairs))},
    },
    "per_map_first_20": per_map[:20],
}

with open(OUT_DIR / "eval_100.json", "w") as f:
    json.dump(report, f, indent=2)

# ─── print ────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"Evaluation: {PKL_PATH.name}  ({N_MAPS} maps, seed={SEED})")
print("=" * 60)
print(f"Solvability rate     : {s['solvability_rate']*100:.1f}%")
print(f"Non-L-pattern rate   : {s['non_l_pattern_rate']*100:.1f}%")
print(f"L-pattern (<=1 turn) : {s['L_pattern_count']}/{N_MAPS}")
print(f"Unsolvable           : {s['unsolvable_count']}/{N_MAPS}")
print(f"Diagonal stripes     : {s['diagonal_stripes_count']}/{N_MAPS}")
print(f"Near-empty maze      : {s['near_empty_count']}/{N_MAPS}")
print(f"Near-full maze       : {s['near_full_count']}/{N_MAPS}")
print(f"COMPOUND             : {compound:.3f}")
print()
print("--- Classification ---")
for cls, cnt in class_counter.most_common():
    print(f"  {cls:<20}: {cnt:3d}  {'#'*cnt}")

st = report["stats_solvable_only"]
print()
print("--- Stats (solvable only) ---")
print(f"  Turns      : mean={st['turns']['mean']:.2f}  median={st['turns']['median']:.0f}  range=[{st['turns']['min']:.0f},{st['turns']['max']:.0f}]")
print(f"  Path len   : mean={st['path_len']['mean']:.2f}  median={st['path_len']['median']:.0f}")
print(f"  A* diff    : mean={st['diff']['mean']:.3f}")

sa = report["stats_all"]
print()
print("--- Stats (all) ---")
print(f"  Wall dens  : mean={sa['wall_density']['mean']:.3f}  std={sa['wall_density']['std']:.3f}  range=[{sa['wall_density']['min']:.3f},{sa['wall_density']['max']:.3f}]")
print(f"  Regions    : mean={sa['regions']['mean']:.2f}  max={sa['regions']['max']:.0f}")
print(f"  Diag score : mean={sa['diag_score']['mean']:.3f}")

pd = report["pairwise_diversity"]
print()
print("--- Diversity ---")
print(f"  A* traj dist : mean={pd['astar_trajectory']['mean']:.3f}")
print(f"  BC distance  : mean={pd['bc_distance']['mean']:.3f}")
print()
print(f"Report: {OUT_DIR/'eval_100.json'}")


# ─── PNG 10x10 grid ───────────────────────────────────────────────────────────
print("Generating PNG grid ...")
COLS = 10
ROWS = (N_MAPS + COLS - 1) // COLS
fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 1.6, ROWS * 1.6))

COLORS = {WALL: [0, 0, 0], FLOOR: [1, 1, 1], PLAYER: [0, 0.7, 0], ENEMY: [0.9, 0.1, 0.1]}

for idx, (ax, lvl) in enumerate(zip(axes.flat, maps)):
    img = np.zeros((*lvl.shape, 3))
    for tile, col in COLORS.items():
        img[lvl == tile] = col
    ax.imshow(img, interpolation="nearest")
    cls = per_map[idx]["class"]
    t = per_map[idx]["turns"]
    ax.set_title(f"#{idx+1} {cls[:8]}\nt={t}", fontsize=5)
    ax.axis("off")

for ax in axes.flat[N_MAPS:]:
    ax.axis("off")

plt.suptitle(f"{PKL_PATH.name} — 100 maps  compound={compound:.3f}", fontsize=9)
plt.tight_layout()
png_path = OUT_DIR / "100_maps.png"
plt.savefig(str(png_path), dpi=120)
plt.close()
print(f"PNG: {png_path}")
