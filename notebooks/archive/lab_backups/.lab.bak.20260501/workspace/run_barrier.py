"""
run_barrier.py — Research: fix archive descriptor + barrier fitness.

Root causes of T8 archive plateau at 8/12:
  1. me_genome_cell uses MEAN tortuosity — even 2 high-tort maps out of 8 average to ~1.04,
     stays in tort_bin 0. Fix: use 75th percentile (or MAX) so any map reaching high tort
     pulls the genome into a higher archive cell.
  2. No fitness gradient for barrier creation. Fix: add barrier_fitness term rewarding
     rows/cols that are ≥65% wall (these are the physical barriers that force A* to detour).

compound_v3 = solvability × high_tortuosity_rate × dir_balance

Variants:
  B0 -- T8 re-validation (MEAN tortuosity archive, S4 fitness)
  B1 -- 75th percentile archive descriptor (instead of mean)
  B2 -- Barrier fitness w=0.05 (intra 0.10→0.07), MEAN archive
  B3 -- B1 + B2 combined (75th percentile + barrier fitness)
  B4 -- MAX tortuosity archive descriptor

Usage:
  conda run -n pcgnn python .lab/workspace/run_barrier.py <VARIANT> [out_dir]
"""
import sys, os, json, time, pickle, heapq, random, math, copy
from pathlib import Path
from collections import deque, Counter
import numpy as np
import neat
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_H = MAX_H = 14
MIN_W = MAX_W = 14
MAPS_PER_GENOME = 8
MAX_GEN = 50
POP_SIZE = 50
SEED = 0
N_FINAL = 50
PERTURB_SIZE = 0.1565
WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3

TORTUOSITY_THRESHOLD = 1.5
BARRIER_WALL_FRAC = 0.65  # row/col must be ≥65% wall to count as barrier

# ── generation ────────────────────────────────────────────────────────────────
def _run_pass(net, padded, ctx, map_h, map_w, noise, perturb):
    """Single generation pass over all cells. Modifies padded in-place."""
    half = ctx
    cells = [(r, c) for r in range(half, map_h+half) for c in range(half, map_w+half)]
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

def generate_level(net, ctx=1, scan="row", map_h=MAX_H, map_w=MAX_W, perturb=True,
                   two_pass=False):
    half = ctx
    n_rand = 4
    padded = np.full((map_h + 2*half, map_w + 2*half), -1.0)
    noise = [random.gauss(0, 1) for _ in range(n_rand)]
    _run_pass(net, padded, ctx, map_h, map_w, noise, perturb)

    if two_pass:
        # Pass 2: network sees real tile values from pass 1 (floor=1.0, wall=0.0).
        # Can break L-corridors by placing walls where neighbors are all floor.
        noise2 = [random.gauss(0, 1) for _ in range(n_rand)]
        _run_pass(net, padded, ctx, map_h, map_w, noise2, perturb)

    level = padded[half:half+map_h, half:half+map_w].astype(int)
    if level[0, 0] == FLOOR: level[0, 0] = PLAYER
    if level[map_h-1, map_w-1] == FLOOR: level[map_h-1, map_w-1] = ENEMY
    return level

# ── A* ────────────────────────────────────────────────────────────────────────
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

# ── tortuosity ────────────────────────────────────────────────────────────────
def path_tortuosity(level):
    """path_length / manhattan_distance(P, E). L-path=1.0, maze>1.0."""
    ps = list(zip(*np.where(level==PLAYER)))
    es = list(zip(*np.where(level==ENEMY)))
    if not ps or not es: return None
    st = ps[0]
    en = min(es, key=lambda e: abs(e[0]-st[0])+abs(e[1]-st[1]))
    manhattan = abs(en[0]-st[0]) + abs(en[1]-st[1])
    if manhattan == 0: return None
    path, _ = _astar(level, st, en)
    if path is None or len(path) < 2: return None
    return (len(path)-1) / manhattan

def tortuosity_fitness(levels, target_thresh=3.0):
    vals = [path_tortuosity(l) for l in levels]
    vals = [v for v in vals if v is not None]
    if not vals: return 0.0
    return float(np.mean([min(v, target_thresh) / target_thresh for v in vals]))

def high_tort_fitness(levels, threshold=1.5):
    """Fraction of solvable maps with tortuosity >= threshold. Mirrors compound_v3 evaluation."""
    solv = [l for l in levels if is_solvable(l)]
    if not solv: return 0.0
    vals = [path_tortuosity(l) for l in solv]
    vals = [v for v in vals if v is not None]
    if not vals: return 0.0
    return float(sum(1 for v in vals if v >= threshold) / len(vals))

# ── barrier fitness ───────────────────────────────────────────────────────────
def barrier_fitness(levels, wall_frac=BARRIER_WALL_FRAC):
    """Fraction of rows+cols that are ≥wall_frac wall tiles.
    A barrier row/col blocks direct passage, forcing A* to detour → higher tortuosity."""
    scores = []
    for lvl in levels:
        h, w = lvl.shape
        n_barriers = 0
        for r in range(h):
            if np.sum(lvl[r, :] == WALL) / w >= wall_frac:
                n_barriers += 1
        for c in range(w):
            if np.sum(lvl[:, c] == WALL) / h >= wall_frac:
                n_barriers += 1
        scores.append(n_barriers / (h + w))
    return float(np.mean(scores)) if scores else 0.0

# ── directional balance ───────────────────────────────────────────────────────
def directional_balance(level):
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

def dir_balance_fitness(levels):
    return float(np.mean([directional_balance(l) for l in levels]))

# ── standard fitness components ───────────────────────────────────────────────
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

def horizontal_stripe_penalty(level):
    h, w = level.shape; floor = {FLOOR, PLAYER, ENEMY}
    thresh = max(2, int(w * 0.6)); stripe_rows = 0
    for r in range(h):
        run = 0
        for c in range(w):
            if level[r,c] in floor: run += 1
            else: run = 0
            if run >= thresh: stripe_rows += 1; break
    return stripe_rows / h

# ── MAP-Elites archive ────────────────────────────────────────────────────────
def make_archive_config(size="tort"):
    d_edges = [0.30, 0.45]
    n_d = 3

    def tort_bin(tort):
        if tort < 1.1: return 0
        if tort < 1.3: return 1
        if tort < 1.7: return 2
        return 3

    def d_bin(d):
        if d < 0.15 or d > 0.60: return None
        for i,b in enumerate(d_edges):
            if d < b: return i
        return len(d_edges)

    if size == "tort":
        # B0/B2: MEAN tortuosity (T8 original) — 4×3=12 cells
        def genome_cell(levels):
            tort_vals = [path_tortuosity(l) for l in levels if is_solvable(l)]
            tort_vals = [v for v in tort_vals if v is not None]
            if not tort_vals: return None
            mt = float(np.mean(tort_vals))
            md = float(np.mean([wall_density(l) for l in levels]))
            db = d_bin(md)
            if db is None: return None
            return (tort_bin(mt), db)
        return genome_cell, 4 * n_d

    elif size == "tort_p75":
        # B1/B3: 75th percentile tortuosity — 4×3=12 cells
        # Lower bar for reaching high-tort cells: even 2 high-tort maps out of 8
        # push the p75 above the threshold instead of just the mean.
        def genome_cell(levels):
            tort_vals = [path_tortuosity(l) for l in levels if is_solvable(l)]
            tort_vals = [v for v in tort_vals if v is not None]
            if not tort_vals: return None
            p75 = float(np.percentile(tort_vals, 75))
            md = float(np.mean([wall_density(l) for l in levels]))
            db = d_bin(md)
            if db is None: return None
            return (tort_bin(p75), db)
        return genome_cell, 4 * n_d

    elif size == "tort_max":
        # B4: MAX tortuosity — any single maze-quality map reaches the high-tort cell
        def genome_cell(levels):
            tort_vals = [path_tortuosity(l) for l in levels if is_solvable(l)]
            tort_vals = [v for v in tort_vals if v is not None]
            if not tort_vals: return None
            mt = float(max(tort_vals))
            md = float(np.mean([wall_density(l) for l in levels]))
            db = d_bin(md)
            if db is None: return None
            return (tort_bin(mt), db)
        return genome_cell, 4 * n_d

    elif size == "ht_rate":
        # B12: high_tort_RATE as archive dimension (fraction of solvable maps with tort>1.5)
        # Only genomes that RELIABLY produce tortuous maps reach high bins.
        # Fresh eval from those genomes should also show high compound_v3.
        def ht_rate_bin(rate):
            if rate < 0.10: return 0   # < 10%
            if rate < 0.20: return 1   # 10-20%
            if rate < 0.35: return 2   # 20-35%
            return 3                    # > 35%

        def genome_cell(levels):
            solv = [l for l in levels if is_solvable(l)]
            if not solv: return None
            tort_vals = [path_tortuosity(l) for l in solv]
            tort_vals = [v for v in tort_vals if v is not None]
            if not tort_vals: return None
            rate = sum(1 for v in tort_vals if v >= TORTUOSITY_THRESHOLD) / len(tort_vals)
            md = float(np.mean([wall_density(l) for l in levels]))
            db = d_bin(md)
            if db is None: return None
            return (ht_rate_bin(rate), db)
        return genome_cell, 4 * n_d

    else:
        raise ValueError(f"Unknown archive size: {size}")


BONUS_EMPTY=1.5; BONUS_IMPROVE=1.2; BONUS_WORSE=0.8; BONUS_BADDNS=0.3

# ── NEAT config ───────────────────────────────────────────────────────────────
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

# ── variant configs ───────────────────────────────────────────────────────────
# (archive_size, scan, ctx, extra_fitness)
VARIANT_CONFIGS = {
    "B0": ("tort",     "row", 1, None),           # T8 re-validation (MEAN archive, S4 fitness)
    "B1": ("tort_p75", "row", 1, None),           # 75th percentile archive descriptor
    "B2": ("tort",     "row", 1, "barrier_0.05"), # barrier fitness w=0.05
    "B3": ("tort_p75", "row", 1, "barrier_0.05"), # B1 + B2 combined
    "B4": ("tort_max", "row", 1, None),           # MAX tortuosity archive descriptor
    "B5": ("tort",     "row", 2, None),              # CONTEXT_SIZE=2 (5x5), S4 fitness, MEAN archive
    "B5b": ("tort",   "row", 2, "barrier_0.03"),   # B5 + barrier w=0.03 (smaller weight)
    "B6": ("tort",    "row", 1, "high_tort_0.10"), # threshold-based tort fitness (w=0.10, intra=0)
    "B7": ("tort",    "row", 2, "high_tort_0.10"),  # B5 + B6: 5x5 + high_tort fitness
    # ── two-pass branch ────────────────────────────────────────────────────────
    "P0": ("tort",    "row", 1, "two_pass"),        # 2-pass generation, B0 fitness (baseline)
    "P1": ("tort",    "row", 1, "two_pass_ht"),     # 2-pass + high_tort fitness w=0.10
    "P2": ("tort",    "row", 1, "two_pass_bar"),    # 2-pass + barrier fitness w=0.03
    # ── aggressive high_tort (solvability sacrifice OK) ─────────────────────
    "B9":  ("tort",   "row", 1, "ht_aggressive"),   # solv=0.35 high_tort=0.25 keep intra
    "B10": ("tort",   "row", 1, "ht_max"),           # solv=0.25 high_tort=0.40 minimal other
    "B11": ("tort_p75","row",1, "ht_aggressive"),   # B9 + p75 archive (lower bar for high bins)
    # ── fresh evaluation: same training as B0, evaluate from top archive genome ─
    "B0f":  ("tort",    "row", 1, "fresh_eval"),     # B0 training + fresh maps from top genome
    # ── ht_rate archive: selects for RELIABLE high-tort producers ─────────────
    "B12":  ("ht_rate", "row", 1, None),             # ht_rate archive + S4 fitness
    "B12f": ("ht_rate", "row", 1, "fresh_eval"),     # B12 + fresh eval (verify genuine capability)
    "B13":  ("ht_rate", "row", 1, "ht_small"),         # ht_rate archive + tiny ht signal w=0.03
    "B14":  ("ht_rate", "row", 1, "ht_medium"),        # ht_rate archive + ht w=0.05, keep dir_bal=0.10
    "B15":  ("ht_rate", "row", 1, "dir_boost"),        # B13 but f_dir restored to 0.10 (funded by path/turns)
    "B16":  ("ht_rate", "row", 1, "branch_fund"),      # ht_small funded from f_branch (0.15→0.12), f_dir/path/turns all at S4
    "B17":  ("ht_rate", "row", 1, "bonus2_ht_small"),  # B13 formula + BONUS_EMPTY=2.0 (stronger empty-cell incentive)
    "B18":  ("ht_rate", "row", 1, "bonus25_ht_small"), # B13 formula + BONUS_EMPTY=2.5
}

def run(variant, out_dir):
    if variant not in VARIANT_CONFIGS:
        print(f"Unknown variant. Choose from: {list(VARIANT_CONFIGS.keys())}"); sys.exit(1)

    arc_size, scan, ctx, extra = VARIANT_CONFIGS[variant]
    n_in = (2*ctx+1)**2 - 1 + 4  # 8 context + 4 noise
    use_two_pass = extra is not None and extra.startswith("two_pass")
    use_fresh_eval = extra == "fresh_eval"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genome_cell, total_cells = make_archive_config(arc_size)

    random.seed(SEED); np.random.seed(SEED)
    me_archive = {}

    def eval_genomes(genomes, config):
        genome_levels = {}
        for gid, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome_levels[gid] = [generate_level(net, ctx=ctx, scan=scan,
                                                   two_pass=use_two_pass)
                                   for _ in range(MAPS_PER_GENOME)]

        for gid, genome in genomes:
            levels = genome_levels[gid]
            f_solve  = solvability_fitness(levels)
            f_intra  = intra_novelty(levels, k=min(10, len(levels)-1))
            f_path   = path_diversity_fitness(levels)
            f_turns  = turns_fitness(levels)
            f_branch = branching_fitness(levels, target=0.10)
            mc_pen   = 0.5 * mode_collapse_penalty(levels)
            f_dir    = dir_balance_fitness(levels)

            if extra is None or extra in ("two_pass", "fresh_eval"):
                # B0/B1/B4/P0/B0f: S4 fitness (T8 baseline)
                base = max(0.0, (0.50*f_solve + 0.10*f_intra + 0.10*f_path
                                + 0.05*f_turns + 0.15*f_branch
                                + 0.10*f_dir) - mc_pen)
            elif extra == "barrier_0.05":
                # B2/B3: S4 + barrier w=0.05, intra 0.10→0.07
                f_barrier = barrier_fitness(levels)
                base = max(0.0, (0.50*f_solve + 0.07*f_intra + 0.10*f_path
                                + 0.05*f_turns + 0.15*f_branch
                                + 0.08*f_dir + 0.05*f_barrier) - mc_pen)
            elif extra in ("barrier_0.03", "two_pass_bar"):
                # B5b/P2: S4 + barrier w=0.03, preserve solvability
                f_barrier = barrier_fitness(levels)
                base = max(0.0, (0.50*f_solve + 0.09*f_intra + 0.10*f_path
                                + 0.05*f_turns + 0.15*f_branch
                                + 0.08*f_dir + 0.03*f_barrier) - mc_pen)
            elif extra in ("high_tort_0.10", "two_pass_ht"):
                # B6/B7/P1: threshold-based high_tort fitness
                f_ht = high_tort_fitness(levels, threshold=1.5)
                base = max(0.0, (0.50*f_solve + 0.00*f_intra + 0.10*f_path
                                + 0.05*f_turns + 0.15*f_branch
                                + 0.10*f_dir + 0.10*f_ht) - mc_pen)
            elif extra == "ht_aggressive":
                # B9/B11: reduce solv weight, strong high_tort signal, KEEP intra for diversity
                # Key insight: B0's intra_novelty helps create diverse maps, some of which
                # happen to be tortuous. Removing intra (B6) actually REDUCED high_tort.
                f_ht = high_tort_fitness(levels, threshold=1.5)
                base = max(0.0, (0.35*f_solve + 0.08*f_intra + 0.08*f_path
                                + 0.04*f_turns + 0.12*f_branch
                                + 0.08*f_dir + 0.25*f_ht) - mc_pen)
            elif extra == "ht_max":
                # B10: maximum high_tort weight — solv barely considered
                f_ht = high_tort_fitness(levels, threshold=1.5)
                base = max(0.0, (0.25*f_solve + 0.08*f_intra + 0.05*f_path
                                + 0.02*f_turns + 0.10*f_branch
                                + 0.10*f_dir + 0.40*f_ht) - mc_pen)
            elif extra in ("ht_small", "bonus2_ht_small", "bonus25_ht_small"):
                # B13/B17: minimal high_tort signal — just enough to push, keep everything else S4
                f_ht = high_tort_fitness(levels, threshold=1.5)
                base = max(0.0, (0.50*f_solve + 0.10*f_intra + 0.10*f_path
                                + 0.05*f_turns + 0.15*f_branch
                                + 0.07*f_dir + 0.03*f_ht) - mc_pen)
            elif extra == "ht_medium":
                # B14: moderate high_tort signal — w=0.05, keep dir_bal=0.10 (funded by branching)
                f_ht = high_tort_fitness(levels, threshold=1.5)
                base = max(0.0, (0.50*f_solve + 0.10*f_intra + 0.08*f_path
                                + 0.04*f_turns + 0.13*f_branch
                                + 0.10*f_dir + 0.05*f_ht) - mc_pen)
            elif extra == "dir_boost":
                # B15: ht_small w=0.03, but f_dir restored to 0.10 (funded by f_path+f_turns, not f_dir)
                # Hypothesis: dir_balance recovers toward B12's 0.88 while high_tort holds ≥18%
                f_ht = high_tort_fitness(levels, threshold=1.5)
                base = max(0.0, (0.50*f_solve + 0.10*f_intra + 0.08*f_path
                                + 0.04*f_turns + 0.15*f_branch
                                + 0.10*f_dir + 0.03*f_ht) - mc_pen)
            elif extra == "branch_fund":
                # B16: ht_small funded from f_branch (0.15→0.12), all other S4 weights preserved
                # Key insight from B15: f_path and f_turns DRIVE dir_balance — don't cut them
                f_ht = high_tort_fitness(levels, threshold=1.5)
                base = max(0.0, (0.50*f_solve + 0.10*f_intra + 0.10*f_path
                                + 0.05*f_turns + 0.12*f_branch
                                + 0.10*f_dir + 0.03*f_ht) - mc_pen)

            if extra == "bonus2_ht_small": be = 2.0
            elif extra == "bonus25_ht_small": be = 2.5
            else: be = BONUS_EMPTY
            cell = genome_cell(levels)
            if cell is None:
                genome.fitness = base * BONUS_BADDNS
            elif cell not in me_archive:
                genome.fitness = base * be
                me_archive[cell] = {"fitness": base, "levels": levels,
                                    "genome": copy.deepcopy(genome) if use_fresh_eval else None}
            elif base > me_archive[cell]["fitness"]:
                genome.fitness = base * BONUS_IMPROVE
                me_archive[cell] = {"fitness": base, "levels": levels,
                                    "genome": copy.deepcopy(genome) if use_fresh_eval else None}
            else:
                genome.fitness = base * BONUS_WORSE

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
            gen_log.append({"gen": g, "best": max(fits), "avg": float(np.mean(fits)),
                             "archive": len(me_archive)})
            if g % 10 == 0:
                elapsed = time.time()-t0
                eta = (elapsed/(g+1))*(MAX_GEN-g-1) if g>0 else 0
                print(f"  gen {g:3d}/{MAX_GEN}  archive={len(me_archive):2d}/{total_cells}"
                      f"  best={max(fits):.3f}  {elapsed:.0f}s  ETA={eta/60:.1f}m", flush=True)

    pop.add_reporter(QuietReporter())
    pop.run(eval_genomes, MAX_GEN)
    elapsed = time.time()-t0

    # pick winner from archive
    best_cell = max(me_archive, key=lambda c: me_archive[c]["fitness"]) if me_archive else None
    best_levels_source = me_archive[best_cell]["levels"] if best_cell else []

    if use_fresh_eval:
        # Fresh evaluation: generate N_FRESH maps from top archive genomes with new random noise.
        # Top cells = highest tort_bin first, then highest fitness.
        top_cells = sorted([c for c in me_archive if c[0] >= 1],
                           key=lambda c: (c[0], me_archive[c]["fitness"]), reverse=True)
        fresh_maps = []
        maps_per_top = max(10, N_FINAL // max(1, len(top_cells[:5])))
        for cell in top_cells[:5]:
            g = me_archive[cell].get("genome")
            if g is not None:
                net = neat.nn.FeedForwardNetwork.create(g, config)
                fresh_maps.extend([generate_level(net, ctx=ctx, scan=scan,
                                                   two_pass=use_two_pass)
                                   for _ in range(maps_per_top)])
        if len(fresh_maps) >= N_FINAL:
            final_maps = fresh_maps[:N_FINAL]
        else:
            # fallback: pad from training maps
            archive_maps = []
            for c in sorted({c: v for c, v in me_archive.items() if c[0] >= 1},
                            reverse=True):
                archive_maps.extend(me_archive[c]["levels"])
            final_maps = (fresh_maps + archive_maps)[:N_FINAL]
        print(f"  [fresh_eval] generated {len(fresh_maps)} fresh maps from {len(top_cells[:5])} top cells")
    else:
        # Standard evaluation: pool training maps from good archive cells
        good_cells = {c: v for c, v in me_archive.items() if c[0] >= 1}
        archive_maps = []
        for c in sorted(good_cells):
            archive_maps.extend(good_cells[c]["levels"])
        while len(archive_maps) < N_FINAL and best_levels_source:
            archive_maps.extend(best_levels_source)
        final_maps = archive_maps[:N_FINAL]

    # evaluate
    per_map = []
    for lvl in final_maps:
        t = count_turns(lvl)
        tort = path_tortuosity(lvl)
        per_map.append({
            "turns": int(t),
            "solvable": int(t >= 0),
            "wall_dens": float(wall_density(lvl)),
            "dir_balance": float(directional_balance(lvl)),
            "stripe_rate": float(horizontal_stripe_penalty(lvl)),
            "tortuosity": float(tort) if tort is not None else 0.0,
            "barrier_rows": int(sum(1 for r in range(lvl.shape[0])
                                   if np.sum(lvl[r,:]==WALL)/lvl.shape[1] >= BARRIER_WALL_FRAC)),
        })

    n = len(per_map)
    solv           = sum(m["solvable"] for m in per_map) / n
    non_l          = sum(1 for m in per_map if m["turns"] >= 2) / n
    dir_bal        = float(np.mean([m["dir_balance"] for m in per_map]))
    stripe_r       = float(np.mean([m["stripe_rate"] for m in per_map]))
    mean_t         = float(np.mean([m["turns"] for m in per_map if m["turns"]>=0]))
    mean_tort      = float(np.mean([m["tortuosity"] for m in per_map if m["tortuosity"]>0]))
    high_tort_rate = sum(1 for m in per_map if m["tortuosity"] >= TORTUOSITY_THRESHOLD) / n
    mean_barriers  = float(np.mean([m["barrier_rows"] for m in per_map]))
    compound_v2    = solv * non_l * dir_bal
    compound_v3    = solv * high_tort_rate * dir_bal

    result = {
        "variant": variant, "elapsed_s": round(elapsed, 1),
        "archive_size": arc_size,
        "archive_coverage": f"{len(me_archive)}/{total_cells}",
        "compound_v3": round(compound_v3, 4),
        "compound_v2": round(compound_v2, 4),
        "solvability": round(solv, 4),
        "high_tortuosity_rate": round(high_tort_rate, 4),
        "mean_tortuosity": round(mean_tort, 4),
        "non_l_rate": round(non_l, 4),
        "dir_balance": round(dir_bal, 4),
        "stripe_rate": round(stripe_r, 4),
        "mean_turns": round(mean_t, 2),
        "mean_barrier_rows": round(mean_barriers, 2),
        "gen_log": gen_log,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    with open(out_dir / "maps.pkl", "wb") as f:
        pickle.dump(final_maps, f)

    print(f"\n{'='*60}")
    print(f"[{variant}]  archive={arc_size}  {elapsed:.0f}s  cells={len(me_archive)}/{total_cells}")
    print(f"  compound_v3        : {compound_v3:.4f}  (PRIMARY: solv x high_tort x dir_bal)")
    print(f"  compound_v2        : {compound_v2:.4f}  (old: solv x non_l x dir_bal)")
    print(f"  solvability        : {solv*100:.1f}%")
    print(f"  high_tort_rate     : {high_tort_rate*100:.1f}%  (tortuosity>{TORTUOSITY_THRESHOLD})")
    print(f"  mean_tortuosity    : {mean_tort:.3f}  (L-path=1.0)")
    print(f"  mean_barrier_rows  : {mean_barriers:.1f}  (rows with ≥{BARRIER_WALL_FRAC:.0%} wall)")
    print(f"  non_l_rate         : {non_l*100:.1f}%")
    print(f"  dir_balance        : {dir_bal:.4f}")
    print(f"  stripe_rate        : {stripe_r:.4f}")
    print(f"  mean_turns         : {mean_t:.1f}")

    # visualize
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
        ax.set_title(f"t={m['turns']} tor={m['tortuosity']:.2f}", fontsize=5)
    for ax in axes.flat[n_show:]: ax.axis("off")
    plt.suptitle(f"{variant} [{arc_size}]  compound_v3={compound_v3:.3f}  "
                 f"high_tort={high_tort_rate:.2f}  dir_bal={dir_bal:.3f}", fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out_dir/"sample.png"), dpi=100); plt.close()

    return result


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv)>1 else "B0"
    out_dir = sys.argv[2] if len(sys.argv)>2 else f".lab/workspace/expB-{variant}"
    if len(sys.argv) > 3:
        MAX_GEN = int(sys.argv[3])
        print(f"[run_barrier] MAX_GEN overridden to {MAX_GEN}")
    if len(sys.argv) > 4:
        SEED = int(sys.argv[4])
        print(f"[run_barrier] SEED overridden to {SEED}")
    print(f"[run_barrier] variant={variant}  archive={VARIANT_CONFIGS[variant][0]}  out={out_dir}")
    run(variant, out_dir)
