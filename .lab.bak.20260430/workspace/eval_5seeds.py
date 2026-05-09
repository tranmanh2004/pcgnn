"""
Evaluate 5 seeds of ME11 winner, 100 maps each.
Reports per-seed metrics + mean±std across seeds for paper comparison.
"""
import sys, json, pickle, random
from pathlib import Path
from collections import Counter
import numpy as np
import neat

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import (
    generate_level, count_path_turns, is_solvable, shortest_path_bfs,
    astar_difficulty, astar_diversity, behavior_characterization, bc_distance,
    WALL, FLOOR, PLAYER, ENEMY,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NEAT_CONFIG  = PROJECT_ROOT / ".lab" / "workspace" / "exp-5" / "config.txt"
OUT_DIR      = PROJECT_ROOT / ".lab" / "workspace" / "eval_5seeds_output"
OUT_DIR.mkdir(exist_ok=True)

PKL_FILES = [
    PROJECT_ROOT / "neat_winner_seed0 (5).pkl",
    PROJECT_ROOT / "neat_winner_seed1 (1).pkl",
    PROJECT_ROOT / "neat_winner_seed2 (1).pkl",
    PROJECT_ROOT / "neat_winner_seed3 (1).pkl",
    PROJECT_ROOT / "neat_winner_seed4 (1).pkl",
]

N_MAPS = 100
EVAL_SEED = 42

config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    str(NEAT_CONFIG),
)

def detect_diagonal_stripes(level):
    h, w = level.shape
    walls = (level == WALL)
    if walls.sum() < 10: return 0.0
    diag_pairs = total_walls = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if walls[r, c]:
                total_walls += 1
                if walls[r+1, c+1]: diag_pairs += 1
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
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and walkable[nr,nc] and not seen[nr,nc]:
                            seen[nr,nc] = True
                            q.append((nr,nc))
    return regions

def leniency(level):
    h, w = level.shape
    return float((level != WALL).sum()) / (h * w)

def compression_distance(level):
    import zlib
    raw = level.astype(np.uint8).tobytes()
    return 1 - len(zlib.compress(raw)) / len(raw)

def branching_rate(level, target=0.10):
    h, w = level.shape
    floor_cells = branches = 0
    for r in range(h):
        for c in range(w):
            if level[r,c] != WALL:
                floor_cells += 1
                nb = sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                         if 0<=r+dr<h and 0<=c+dc<w and level[r+dr,c+dc]!=WALL)
                if nb >= 3: branches += 1
    return (branches/max(1,floor_cells)) / target if floor_cells else 0.0

all_results = []

for pkl_path in PKL_FILES:
    seed_name = pkl_path.stem
    print(f"\n{'='*55}")
    print(f"Evaluating: {pkl_path.name}")
    print(f"{'='*55}")

    with open(pkl_path, "rb") as f:
        winner = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    maps = [generate_level(net) for _ in range(N_MAPS)]

    per_map = []
    for lvl in maps:
        turns = count_path_turns(lvl)
        sp    = shortest_path_bfs(lvl)
        per_map.append({
            "solvable"  : int(turns >= 0),
            "non_l"     : int(turns >= 2),
            "turns"     : int(turns),
            "path_len"  : int(sp) if sp else 0,
            "diff"      : float(astar_difficulty(lvl)),
            "leniency"  : float(leniency(lvl)),
            "compress"  : float(compression_distance(lvl)),
            "diag"      : float(detect_diagonal_stripes(lvl)),
            "near_empty": int((lvl == WALL).sum() / lvl.size < 0.10),
            "near_full" : int((lvl == WALL).sum() / lvl.size > 0.70),
            "branch"    : float(branching_rate(lvl)),
        })

    # Pairwise A* diversity (50 pairs)
    random.seed(0)
    pairs = [(a,b) for a,b in [(random.randrange(N_MAPS), random.randrange(N_MAPS))
              for _ in range(60)] if a != b][:50]
    astar_div  = [astar_diversity(maps[a], maps[b]) for a,b in pairs]

    # Pairwise A* edit diversity (50 pairs) — same as astar_diversity by default
    astar_edit = astar_div  # same function in run_experiment

    s = {
        "pkl"         : pkl_path.name,
        "solvability" : np.mean([m["solvable"] for m in per_map]),
        "non_l_rate"  : np.mean([m["non_l"]    for m in per_map]),
        "leniency"    : np.mean([m["leniency"]  for m in per_map]),
        "astar_diff"  : np.mean([m["diff"]      for m in per_map if m["solvable"]]),
        "compress"    : np.mean([m["compress"]   for m in per_map]),
        "astar_div"   : float(np.mean(astar_div)),
        "diag_count"  : sum(m["diag"] > 0.4 for m in per_map),
        "near_empty"  : sum(m["near_empty"]      for m in per_map),
        "near_full"   : sum(m["near_full"]        for m in per_map),
        "branch_mean" : np.mean([m["branch"]      for m in per_map]),
    }
    all_results.append(s)

    print(f"  Solvability  : {s['solvability']*100:.1f}%")
    print(f"  Non-L rate   : {s['non_l_rate']*100:.1f}%")
    print(f"  Leniency     : {s['leniency']:.3f}")
    print(f"  A* Difficulty: {s['astar_diff']:.3f}")
    print(f"  Compression  : {s['compress']:.3f}")
    print(f"  A* Diversity : {s['astar_div']:.3f}")
    print(f"  Branching    : {s['branch_mean']:.3f}")
    print(f"  Diag stripes : {s['diag_count']}/100  |  Near-empty: {s['near_empty']}  |  Near-full: {s['near_full']}")

# ── AGGREGATE across 5 seeds ──────────────────────────────────
print(f"\n{'='*55}")
print("AGGREGATE (5 seeds, 100 maps each)")
print(f"{'='*55}")

metrics = ["solvability","non_l_rate","leniency","astar_diff","compress","astar_div","branch_mean"]
labels  = ["Solvability (%)","Non-L Rate (%)","Leniency","A* Difficulty","Compression Dist","A* Diversity","Branching"]
scales  = [100, 100, 1, 1, 1, 1, 1]

agg = {}
for m, lbl, sc in zip(metrics, labels, scales):
    vals = [r[m] for r in all_results]
    mean = np.mean(vals) * sc
    std  = np.std(vals)  * sc
    agg[m] = {"mean": float(mean), "std": float(std), "vals": [v*sc for v in vals]}
    print(f"  {lbl:<22}: {mean:.3f} ± {std:.3f}   (seeds: {[f'{v:.2f}' for v in [v*sc for v in vals]]})")

# Baseline Beukman 2022 (from paper_reference.md)
print(f"\n{'='*55}")
print("vs Beukman 2022 baseline")
print(f"{'='*55}")
baseline = {
    "Solvability (%)":   (100.0,  0.0),
    "Leniency":          (0.70,   0.08),
    "A* Difficulty":     (0.06,   0.08),
    "Compression Dist":  (0.488,  0.002),
    "A* Diversity":      (0.13,   0.17),
}
our = {
    "Solvability (%)":  (agg["solvability"]["mean"], agg["solvability"]["std"]),
    "Leniency":         (agg["leniency"]["mean"],    agg["leniency"]["std"]),
    "A* Difficulty":    (agg["astar_diff"]["mean"],  agg["astar_diff"]["std"]),
    "Compression Dist": (agg["compress"]["mean"],    agg["compress"]["std"]),
    "A* Diversity":     (agg["astar_div"]["mean"],   agg["astar_div"]["std"]),
}
for name in baseline:
    bm, bs = baseline[name]
    om, os = our[name]
    delta = (om - bm) / bm * 100
    print(f"  {name:<22}: baseline={bm:.3f}±{bs:.3f}  ours={om:.3f}±{os:.3f}  Δ={delta:+.1f}%")

# Save
out_file = OUT_DIR / "results_5seeds.json"
with open(out_file, "w") as f:
    json.dump({"per_seed": all_results, "aggregate": agg}, f, indent=2)
print(f"\nSaved: {out_file}")
