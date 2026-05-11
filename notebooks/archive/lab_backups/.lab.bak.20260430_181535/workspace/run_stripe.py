"""
run_stripe.py — Research: escape horizontal stripe local optimum.

All variants build on ME11 (best known: compound_v1=0.920).
New metric: compound_v2 = solvability × non_l_rate × dir_balance
  dir_balance = min(h_trans, v_trans) / max(h_trans, v_trans)

Variants:
  S0  — ME11 exact baseline, measure compound_v2
  S1  — ME11 + archive winner (pick best-base genome from archive, not pop.run())
  S2  — ME11 + 48-cell archive (8×6 bins)
  S3  — S1 + S2
  S4  — ME11 + dir_balance fitness term (w=0.10, reduce intra 0.20→0.10)
  S5  — ME11 + horizontal stripe penalty (penalize long horiz floor runs)
  S6  — ME11 + column-first scan order
  S7  — ME11 + random scan order
  S8  — S3 + S4
  S9  — ME11 + CONTEXT_SIZE=2 (24 inputs, requires NEAT num_inputs=28)

Usage:
  python run_stripe.py <VARIANT> [out_dir]
"""
import sys, os, json, time, pickle, heapq, random, math
from pathlib import Path
from collections import deque, Counter
import numpy as np
import neat
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── fixed params ─────────────────────────────────────────────────────────────
MIN_H = MAX_H = 14
MIN_W = MAX_W = 14
MAPS_PER_GENOME = 8
MAX_GEN = 50
POP_SIZE = 50
SEED = 0
N_FINAL = 50
PERTURB_SIZE = 0.1565
WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3

# ─── generation ───────────────────────────────────────────────────────────────
def generate_level(net, ctx=1, scan="row", map_h=MAX_H, map_w=MAX_W, perturb=True):
    half = ctx
    n_ctx = (2*half+1)**2 - 1
    n_rand = 4
    padded = np.full((map_h + 2*half, map_w + 2*half), -1.0)
    noise = [random.gauss(0, 1) for _ in range(n_rand)]

    if scan == "row":
        cells = [(r, c) for r in range(half, map_h+half) for c in range(half, map_w+half)]
    elif scan == "col":
        cells = [(r, c) for c in range(half, map_w+half) for r in range(half, map_h+half)]
    elif scan == "random":
        cells = [(r, c) for r in range(half, map_h+half) for c in range(half, map_w+half)]
        random.shuffle(cells)
    else:
        raise ValueError(f"Unknown scan: {scan}")

    for r, c in cells:
        ctx_vals = []
        for dr in range(-half, half+1):
            for dc in range(-half, half+1):
                if dr == 0 and dc == 0: continue
                ctx_vals.append(padded[r+dr, c+dc])
        inputs = ctx_vals + noise
        if perturb:
            inputs = [x + random.gauss(0, PERTURB_SIZE) for x in inputs]
        out = net.activate(inputs)[0]
        padded[r, c] = 1.0 if out > 0.5 else 0.0

    level = padded[half:half+map_h, half:half+map_w].astype(int)
    if level[0, 0] == FLOOR: level[0, 0] = PLAYER
    if level[map_h-1, map_w-1] == FLOOR: level[map_h-1, map_w-1] = ENEMY
    return level

# ─── A* + BFS (cached) ────────────────────────────────────────────────────────
_cache = {}

def _astar(level, start, end):
    h, w = level.shape
    cnt = 0
    heap = [(abs(start[0]-end[0])+abs(start[1]-end[1]), cnt, start)]
    came, g = {start: None}, {start: 0}
    vis = set()
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in vis: continue
        vis.add(cur)
        if cur == end:
            path, node = [], end
            while node: path.append(node); node = came[node]
            return list(reversed(path)), len(vis)
        r, c = cur
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and (nr,nc) not in vis and level[nr,nc]!=WALL:
                ng = g[cur]+1
                if ng < g.get((nr,nc), 1e9):
                    came[(nr,nc)] = cur; g[(nr,nc)] = ng; cnt+=1
                    heapq.heappush(heap, (ng+abs(nr-end[0])+abs(nc-end[1]), cnt, (nr,nc)))
    return None, len(vis)

def _metrics(level):
    key = level.tobytes()
    if key in _cache: return _cache[key]
    ps = list(zip(*np.where(level==PLAYER)))
    es = list(zip(*np.where(level==ENEMY)))
    if not ps or not es:
        r = (None, 0, False, None)
    else:
        st = ps[0]; en = min(es, key=lambda e: abs(e[0]-st[0])+abs(e[1]-st[1]))
        path, considered = _astar(level, st, en)
        solv = path is not None and len(path)>1
        r = (path, considered, solv, len(path)-1 if solv else None)
    if len(_cache) > 40000: _cache.clear()
    _cache[key] = r
    return r

def is_solvable(l): return _metrics(l)[2]
def shortest_path(l): return _metrics(l)[3]
def get_astar(l): m=_metrics(l); return m[0], m[1]

def count_turns(level):
    path, _ = get_astar(level)
    if path is None or len(path) < 3: return -1 if path is None else 0
    t = 0
    for i in range(1, len(path)-1):
        d1 = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        d2 = (path[i+1][0]-path[i][0],  path[i+1][1]-path[i][1])
        if d1 != d2: t += 1
    return t

# ─── new metric: directional balance ─────────────────────────────────────────
def directional_balance(level):
    """min(h_trans, v_trans) / max(h_trans, v_trans) over floor cells.
    Stripe pattern → ~0.0, balanced maze → ~1.0."""
    h, w = level.shape
    floor = {FLOOR, PLAYER, ENEMY}
    h_t = v_t = 0
    for r in range(h):
        for c in range(w):
            if level[r,c] in floor:
                if c+1 < w and level[r,c+1] in floor: h_t += 1
                if r+1 < h and level[r+1,c] in floor: v_t += 1
    if max(h_t, v_t) == 0: return 0.0
    return min(h_t, v_t) / max(h_t, v_t)

# ─── stripe penalty ───────────────────────────────────────────────────────────
def horizontal_stripe_penalty(level):
    """Fraction of rows that have a horizontal floor run ≥ 60% of width.
    Stripe pattern → high, organic maze → low."""
    h, w = level.shape
    floor = {FLOOR, PLAYER, ENEMY}
    thresh = max(2, int(w * 0.6))
    stripe_rows = 0
    for r in range(h):
        run = 0
        for c in range(w):
            if level[r,c] in floor: run += 1
            else: run = 0
            if run >= thresh: stripe_rows += 1; break
    return stripe_rows / h

def stripe_pen_fitness(levels):
    return float(np.mean([horizontal_stripe_penalty(l) for l in levels]))

def dir_balance_fitness(levels):
    return float(np.mean([directional_balance(l) for l in levels]))

# ─── standard fitness components ──────────────────────────────────────────────
def solvability_fitness(levels):
    return float(np.mean([float(is_solvable(l)) for l in levels]))

def path_diversity_fitness(levels):
    lens = [shortest_path(l) for l in levels if shortest_path(l) is not None]
    if len(lens) < 2: return 0.0
    return float(min(1.0, np.std(lens) / (MAX_H + MAX_W)))

def bc_distance(la, lb):
    """7-dim BC with turns weight=7.84 (ME11 winner)."""
    def bc(level):
        h, w = level.shape; wk = {FLOOR, PLAYER, ENEMY}
        interior = level[1:-1, 1:-1]
        wall_dens = float(np.sum(interior==WALL)/max(1,interior.size))
        sp = shortest_path(level)
        path_norm = min(1.0, (sp/(h+w)) if sp else 0.0)
        dead = branches = 0
        for r in range(h):
            for c in range(w):
                if level[r,c] not in wk: continue
                nb = sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                         if 0<=r+dr<h and 0<=c+dc<w and level[r+dr,c+dc] in wk)
                if nb==1: dead+=1
                elif nb>=3: branches+=1
        tw = max(1, int((level!=WALL).sum()))
        vis = np.zeros((h,w),bool); regs = 0
        for r0 in range(h):
            for c0 in range(w):
                if level[r0,c0] in wk and not vis[r0,c0]:
                    regs+=1; q=deque([(r0,c0)]); vis[r0,c0]=True
                    while q:
                        r,c=q.popleft()
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr,nc=r+dr,c+dc
                            if 0<=nr<h and 0<=nc<w and not vis[nr,nc] and level[nr,nc] in wk:
                                vis[nr,nc]=True; q.append((nr,nc))
        path,considered=get_astar(level)
        diff = min(1.0, max(considered-len(path),0)/max(1,int((level!=WALL).sum())-len(path if path else []))) if path else 0.0
        turns_n = 0.0
        if path and len(path)>=3:
            t=0
            for i in range(1,len(path)-1):
                d1=(path[i][0]-path[i-1][0],path[i][1]-path[i-1][1])
                d2=(path[i+1][0]-path[i][0],path[i+1][1]-path[i][1])
                if d1!=d2: t+=1
            turns_n = min(1.0, t/max(1,len(path)-2))
        return np.array([wall_dens, path_norm, dead/tw, branches/tw, min(1.0,regs/10.0), diff, turns_n])
    w = np.array([1,1,1,1,1,1,7.84])
    d = bc(la)-bc(lb)
    return float(np.sqrt(np.sum(w*d*d))/np.sqrt(w.sum()))

def intra_novelty(levels, k=2):
    if len(levels) < 2: return 0.0
    scores = []
    for i, li in enumerate(levels):
        dists = sorted(bc_distance(li, lj) for j,lj in enumerate(levels) if i!=j)
        scores.append(np.mean(dists[:min(k,len(dists))]))
    return float(np.mean(scores))

# ─── mode-collapse penalty (ME11) ────────────────────────────────────────────
def diag_penalty(level):
    walls = (level==WALL); h,w = level.shape
    if walls.sum()<8: return 0.0
    dp=tot=0
    for r in range(h-1):
        for c in range(w-1):
            if walls[r,c]: tot+=1; dp+= int(walls[r+1,c+1])
    return dp/max(1,tot)

def dens_penalty(level):
    interior=level[1:-1,1:-1]; d=float(np.sum(interior==WALL)/max(1,interior.size))
    if d<0.15: return 1.0-(d/0.15)
    if d>0.60: return min(1.0,(d-0.60)/0.40)
    return 0.0

def mode_collapse_penalty(levels):
    return 0.5*(float(np.mean([diag_penalty(l) for l in levels]))+
                float(np.mean([dens_penalty(l) for l in levels])))

def turns_fitness(levels, target=6):
    vals = [count_turns(l) for l in levels if count_turns(l)>=0]
    if not vals: return 0.0
    return float(np.mean([min(t,target)/target for t in vals]))

def branching_fitness(levels, target=0.10):
    scores = []
    for lvl in levels:
        h,w=lvl.shape; fl=br=0
        for r in range(h):
            for c in range(w):
                if lvl[r,c]!=WALL:
                    fl+=1
                    nb=sum(1 for dr,dc in[(-1,0),(1,0),(0,-1),(0,1)]
                           if 0<=r+dr<h and 0<=c+dc<w and lvl[r+dr,c+dc]!=WALL)
                    if nb>=3: br+=1
        scores.append(min(1.0,(br/max(1,fl))/target))
    return float(np.mean(scores)) if scores else 0.0

def wall_density(level):
    i=level[1:-1,1:-1]; return float(np.sum(i==WALL)/max(1,i.size))

# ─── MAP-Elites archive config ────────────────────────────────────────────────
# S0-S1-S4-S5-S6-S7-S8 (12 cells): TURNS_BINS_S = [0,1,4,8], DENS_EDGES_S = [.30,.45]
# S2-S3-S8 (48 cells): TURNS_BINS_L, DENS_EDGES_L

def make_archive_config(size="small"):
    if size == "small":
        def t_bin(t):
            if t<0 or t<=1: return 0
            if t<=4: return 1
            if t<=8: return 2
            return 3
        d_edges = [0.30, 0.45]  # 3 bins
        n_t, n_d = 4, 3
    else:  # large = 48 cells
        def t_bin(t):
            if t<0 or t==0: return 0
            if t<=2: return 1
            if t<=4: return 2
            if t<=7: return 3
            if t<=11: return 4
            if t<=16: return 5
            if t<=23: return 6
            return 7
        d_edges = [0.225, 0.30, 0.375, 0.45, 0.525]  # 6 bins
        n_t, n_d = 8, 6
    def d_bin(d):
        if d < 0.15 or d > 0.60: return None
        for i,b in enumerate(d_edges):
            if d < b: return i
        return len(d_edges)
    def genome_cell(levels):
        valid_t = [count_turns(l) for l in levels if is_solvable(l)]
        mt = float(np.mean(valid_t)) if valid_t else -1.0
        md = float(np.mean([wall_density(l) for l in levels]))
        db = d_bin(md)
        if db is None: return None
        return (t_bin(mt), db)
    return genome_cell, n_t*n_d

BONUS_EMPTY=1.5; BONUS_IMPROVE=1.2; BONUS_WORSE=0.8; BONUS_BADDNS=0.3

# ─── NEAT config text ─────────────────────────────────────────────────────────
def neat_config_text(n_in):
    return f"""
[NEAT]
fitness_criterion      = max
fitness_threshold      = 999999
pop_size               = {POP_SIZE}
reset_on_extinction    = False
no_fitness_termination = True

[DefaultGenome]
num_inputs             = {n_in}
num_outputs            = 1
num_hidden             = 0
feed_forward           = True
initial_connection     = full_direct
node_add_prob          = 0.6
node_delete_prob       = 0.2
conn_add_prob          = 0.5
conn_delete_prob       = 0.3
activation_default     = sigmoid
activation_mutate_rate = 0.15
activation_options     = sigmoid sin gauss
aggregation_default    = sum
aggregation_mutate_rate = 0.0
aggregation_options    = sum
bias_init_mean         = 0.0
bias_init_stdev        = 1.0
bias_init_type         = gaussian
bias_max_value         = 30.0
bias_min_value         = -30.0
bias_mutate_power      = 0.5
bias_mutate_rate       = 0.7
bias_replace_rate      = 0.1
response_init_mean     = 1.0
response_init_stdev    = 0.0
response_init_type     = gaussian
response_max_value     = 30.0
response_min_value     = -30.0
response_mutate_power  = 0.0
response_mutate_rate   = 0.0
response_replace_rate  = 0.0
weight_init_mean       = 0.0
weight_init_stdev      = 1.0
weight_init_type       = gaussian
weight_max_value       = 30.0
weight_min_value       = -30.0
weight_mutate_power    = 0.5
weight_mutate_rate     = 0.8
weight_replace_rate    = 0.15
enabled_default        = True
enabled_mutate_rate    = 0.02
enabled_rate_to_true_add  = 0.0
enabled_rate_to_false_add = 0.0
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
single_structural_mutation = False
structural_mutation_surer  = default

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 3
survival_threshold = 0.2
min_species_size   = 2
"""

# ─── main runner ──────────────────────────────────────────────────────────────
VARIANT_CONFIGS = {
    # (archive_size, scan, ctx, archive_winner, extra_fitness)
    # extra_fitness: None | "dir_balance" | "stripe_pen"
    "S0": ("small", "row",    1, False, None),
    "S1": ("small", "row",    1, True,  None),
    "S2": ("large", "row",    1, False, None),
    "S3": ("large", "row",    1, True,  None),
    "S4": ("small", "row",    1, False, "dir_balance"),
    "S5": ("small", "row",    1, False, "stripe_pen"),
    "S6": ("small", "col",    1, False, None),
    "S7": ("small", "random", 1, False, None),
    "S8": ("large", "row",    1, True,  "dir_balance"),
    "S9": ("small", "row",    2, False, None),
}

def run(variant, out_dir):
    if variant not in VARIANT_CONFIGS:
        print(f"Unknown variant. Choose from: {list(VARIANT_CONFIGS.keys())}"); sys.exit(1)

    arc_size, scan, ctx, use_archive_winner, extra = VARIANT_CONFIGS[variant]
    n_in = (2*ctx+1)**2 - 1 + 4  # context tiles + 4 noise

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genome_cell, total_cells = make_archive_config(arc_size)

    random.seed(SEED); np.random.seed(SEED)
    me_archive = {}  # cell → {"fitness", "levels", "gen", "genome"(if archive_winner)}

    # ── fitness function ────────────────────────────────────────────────────
    def eval_genomes(genomes, config):
        genome_levels = {}
        for gid, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome_levels[gid] = [generate_level(net, ctx=ctx, scan=scan)
                                   for _ in range(MAPS_PER_GENOME)]

        for gid, genome in genomes:
            levels = genome_levels[gid]
            f_solve  = solvability_fitness(levels)
            f_intra  = intra_novelty(levels, k=min(10, len(levels)-1))
            f_path   = path_diversity_fitness(levels)
            f_turns  = turns_fitness(levels)
            f_branch = branching_fitness(levels, target=0.10)
            mc_pen   = 0.5 * mode_collapse_penalty(levels)

            # extra fitness term
            f_extra = 0.0
            if extra == "dir_balance":
                f_extra = dir_balance_fitness(levels)
            elif extra == "stripe_pen":
                f_extra = -stripe_pen_fitness(levels)  # penalty: subtract

            # ME11 base weights + optional extra
            if extra == "dir_balance":
                # reduce intra 0.20→0.10 to make room for dir_balance 0.10
                base = max(0.0, (0.50*f_solve + 0.10*f_intra + 0.10*f_path
                                + 0.05*f_turns + 0.15*f_branch
                                + 0.10*f_extra) - mc_pen)
            elif extra == "stripe_pen":
                # add stripe penalty on top of ME11
                stripe_pen = 0.5 * stripe_pen_fitness(levels)
                base = max(0.0, (0.50*f_solve + 0.20*f_intra + 0.10*f_path
                                + 0.05*f_turns + 0.15*f_branch) - mc_pen - stripe_pen)
            else:
                base = max(0.0, (0.50*f_solve + 0.20*f_intra + 0.10*f_path
                                + 0.05*f_turns + 0.15*f_branch) - mc_pen)

            cell = genome_cell(levels)
            if cell is None:
                genome.fitness = base * BONUS_BADDNS
            elif cell not in me_archive:
                genome.fitness = base * BONUS_EMPTY
                me_archive[cell] = {"fitness": base, "levels": levels, "gen": 0}
                if use_archive_winner:
                    me_archive[cell]["genome"] = genome
            elif base > me_archive[cell]["fitness"]:
                genome.fitness = base * BONUS_IMPROVE
                me_archive[cell] = {"fitness": base, "levels": levels, "gen": 0}
                if use_archive_winner:
                    me_archive[cell]["genome"] = genome
            else:
                genome.fitness = base * BONUS_WORSE

    # ── NEAT setup ─────────────────────────────────────────────────────────
    cfg_path = str(out_dir / "neat_cfg.txt")
    with open(cfg_path, "w") as f:
        f.write(neat_config_text(n_in))
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path)
    pop = neat.Population(config)

    gen_log = []
    t0 = time.time()

    class QuietReporter(neat.reporting.BaseReporter):
        def post_evaluate(self, config, population, species, best_genome):
            g = len(gen_log)
            fits = [gn.fitness for gn in population.values() if gn.fitness is not None]
            gen_log.append({"gen": g, "best": max(fits), "avg": np.mean(fits),
                             "archive": len(me_archive)})
            if g % 10 == 0:
                elapsed = time.time()-t0
                eta = (elapsed/(g+1))*(MAX_GEN-g-1) if g>0 else 0
                print(f"  gen {g:3d}/{MAX_GEN}  archive={len(me_archive):2d}/{total_cells}"
                      f"  best={max(fits):.3f}  {elapsed:.0f}s  ETA={eta/60:.1f}m", flush=True)

    pop.add_reporter(QuietReporter())
    pop.run(eval_genomes, MAX_GEN)
    elapsed = time.time()-t0

    # ── pick winner ────────────────────────────────────────────────────────
    if use_archive_winner and me_archive:
        best_cell = max(me_archive, key=lambda c: me_archive[c]["fitness"])
        best_genome = me_archive[best_cell].get("genome")
        best_levels_source = me_archive[best_cell]["levels"]
    else:
        best_cell = max(me_archive, key=lambda c: me_archive[c]["fitness"]) if me_archive else None
        best_levels_source = me_archive[best_cell]["levels"] if best_cell else []
        best_genome = None

    # ── generate N_FINAL maps from archive (good cells first) ─────────────
    good_cells = {c: v for c, v in me_archive.items() if c[0] >= 1}
    archive_maps = []
    for c in sorted(good_cells):
        archive_maps.extend(good_cells[c]["levels"])
    while len(archive_maps) < N_FINAL and best_levels_source:
        archive_maps.extend(best_levels_source)
    final_maps = archive_maps[:N_FINAL]

    # ── evaluate ───────────────────────────────────────────────────────────
    per_map = []
    for lvl in final_maps:
        t = count_turns(lvl)
        per_map.append({
            "turns": int(t),
            "solvable": int(t >= 0),
            "wall_dens": float(wall_density(lvl)),
            "dir_balance": float(directional_balance(lvl)),
            "stripe_rate": float(horizontal_stripe_penalty(lvl)),
        })

    n = len(per_map)
    solv      = sum(m["solvable"] for m in per_map) / n
    non_l     = sum(1 for m in per_map if m["turns"] >= 2) / n
    dir_bal   = float(np.mean([m["dir_balance"] for m in per_map]))
    stripe_r  = float(np.mean([m["stripe_rate"] for m in per_map]))
    mean_t    = float(np.mean([m["turns"] for m in per_map if m["turns"]>=0]))
    compound_v2 = solv * non_l * dir_bal
    compound_v1 = solv * non_l

    # path metrics
    seg_vals = []
    for lvl in final_maps:
        path, _ = get_astar(lvl)
        if path is None or len(path) < 3: continue
        segs, s = [], 1
        for i in range(1, len(path)-1):
            d1=(path[i][0]-path[i-1][0],path[i][1]-path[i-1][1])
            d2=(path[i+1][0]-path[i][0],path[i+1][1]-path[i][1])
            if d1!=d2: segs.append(s); s=1
            else: s+=1
        segs.append(s); seg_vals.append(float(np.mean(segs)))
    seg_mean = float(np.mean(seg_vals)) if seg_vals else 0.0

    result = {
        "variant": variant, "elapsed_s": round(elapsed, 1),
        "archive_coverage": f"{len(me_archive)}/{total_cells}",
        "compound_v2": round(compound_v2, 4),
        "compound_v1": round(compound_v1, 4),
        "solvability": round(solv, 4),
        "non_l_rate": round(non_l, 4),
        "dir_balance": round(dir_bal, 4),
        "stripe_rate": round(stripe_r, 4),
        "mean_turns": round(mean_t, 2) if mean_t == mean_t else 0.0,
        "seg_mean": round(seg_mean, 2),
        "gen_log": gen_log,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    with open(out_dir / "maps.pkl", "wb") as f:
        pickle.dump(final_maps, f)

    # ── print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"[{variant}]  {elapsed:.0f}s  archive={len(me_archive)}/{total_cells}")
    print(f"  compound_v2  : {compound_v2:.4f}  (primary)")
    print(f"  compound_v1  : {compound_v1:.4f}")
    print(f"  solvability  : {solv*100:.1f}%")
    print(f"  non_l_rate   : {non_l*100:.1f}%")
    print(f"  dir_balance  : {dir_bal:.4f}  (stripe->0, maze->1)")
    print(f"  stripe_rate  : {stripe_r:.4f}  (lower=better)")
    print(f"  mean_turns   : {mean_t:.1f}")
    print(f"  seg_mean     : {seg_mean:.2f}")

    # turns histogram
    tc = Counter(m["turns"] for m in per_map)
    print(f"  turns hist   : " +
          "  ".join(f"{k}:{tc.get(k,0)}" for k in sorted(set(list(tc.keys())+[-1,0,1,2,3,4,5]))))

    # ── PNG ────────────────────────────────────────────────────────────────
    n_show = min(50, len(final_maps))
    cols = 10; rows = (n_show+9)//10
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.4, rows*1.4))
    COLORS = {WALL:[0,0,0], FLOOR:[1,1,1], PLAYER:[0,.7,0], ENEMY:[.9,.1,.1]}
    for idx in range(n_show):
        ax = axes.flat[idx]
        lvl = final_maps[idx]
        img = np.zeros((*lvl.shape, 3))
        for tile, col in COLORS.items(): img[lvl==tile] = col
        ax.imshow(img, interpolation="nearest"); ax.axis("off")
        m = per_map[idx]
        ax.set_title(f"t={m['turns']} db={m['dir_balance']:.2f}", fontsize=5)
    for ax in axes.flat[n_show:]: ax.axis("off")
    plt.suptitle(f"{variant}  compound_v2={compound_v2:.3f}  dir_bal={dir_bal:.3f}  stripe={stripe_r:.3f}",
                 fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out_dir/"sample.png"), dpi=100); plt.close()

    return result


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv)>1 else "S0"
    out_dir = sys.argv[2] if len(sys.argv)>2 else f".lab/workspace/expS-{variant}"
    print(f"[run_stripe] variant={variant}  out={out_dir}")
    run(variant, out_dir)
