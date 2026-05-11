"""Quick smoke test of improve2.ipynb logic (ME12-D6 formula), 10 gen."""
import sys, os, random, pickle, heapq, gzip, math, time
from collections import deque
from pathlib import Path
import numpy as np

# ── same constants as notebook ──────────────────────────────────────
MIN_WIDTH = MAX_WIDTH = MIN_HEIGHT = MAX_HEIGHT = 14
MAPS_PER_GENOME = 8   # reduced from 24 for speed
MAX_GEN = 10          # quick test
POP_SIZE = 50
SEEDS = [0]
WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3
CONTEXT_SIZE = 1
CTX_TILES = (2*CONTEXT_SIZE+1)**2 - 1
NUM_RANDOM_INPUTS = 4
NUM_INPUTS = CTX_TILES + NUM_RANDOM_INPUTS
NUM_OUTPUTS = 1
NOVELTY_K = 15
PERTURB_SIZE = 0.1565

import neat, tempfile

# ── inline all helpers from notebook cells ───────────────────────────
def generate_level(net, map_h=14, map_w=14, perturb=True):
    if hasattr(net, 'reset'): net.reset()
    half = CONTEXT_SIZE
    padded = np.full((map_h + 2*half, map_w + 2*half), -1.0, dtype=float)
    noise = [random.gauss(0, 1) for _ in range(NUM_RANDOM_INPUTS)]
    for r in range(half, map_h + half):
        for c in range(half, map_w + half):
            ctx = [padded[r+dr, c+dc] for dr in range(-half, half+1)
                   for dc in range(-half, half+1) if not (dr==0 and dc==0)]
            inputs = [x + random.gauss(0, PERTURB_SIZE) for x in ctx + noise] if perturb else ctx + noise
            padded[r, c] = 1.0 if net.activate(inputs)[0] > 0.5 else 0.0
    level = padded[half:half+map_h, half:half+map_w].astype(int)
    if level[0, 0] == FLOOR: level[0, 0] = PLAYER
    if level[-1, -1] == FLOOR: level[-1, -1] = ENEMY
    return level

_cache = {}
def _astar_internal(level, start, end):
    h, w = level.shape; cnt = 0
    heap = [(abs(start[0]-end[0])+abs(start[1]-end[1]), cnt, start)]
    came, g = {start: None}, {start: 0}; vis = set()
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

def _compute(level):
    key = level.tobytes()
    if key in _cache: return _cache[key]
    ps = list(zip(*np.where(level==PLAYER))); es = list(zip(*np.where(level==ENEMY)))
    if not ps or not es: r = (None, 0, False, None)
    else:
        st = ps[0]; en = min(es, key=lambda e: abs(e[0]-st[0])+abs(e[1]-st[1]))
        path, cnt = _astar_internal(level, st, en)
        solv = path is not None and len(path)>1
        r = (path, cnt, solv, len(path)-1 if solv else None)
    if len(_cache) > 20000: _cache.clear()
    _cache[key] = r; return r

def is_solvable(l): return _compute(l)[2]
def shortest_path_bfs(l): return _compute(l)[3]
def _get_astar_result(l): m = _compute(l); return m[0], m[1]
def reset_level_cache(): _cache.clear()

def path_diversity_fitness(levels):
    lens = [shortest_path_bfs(l) for l in levels if shortest_path_bfs(l) is not None]
    return float(min(1.0, np.std(lens) / 28)) if len(lens) >= 2 else 0.0

def solvability_fitness(levels):
    return float(np.mean([float(is_solvable(l)) for l in levels]))

def high_tort_fitness(levels, threshold=1.5):
    scores = []
    for level in levels:
        path, _ = _get_astar_result(level)
        if path is None: scores.append(0.0); continue
        ps = list(zip(*np.where(level==PLAYER))); es = list(zip(*np.where(level==ENEMY)))
        if not ps or not es: scores.append(0.0); continue
        manhattan = abs(ps[0][0]-es[0][0]) + abs(ps[0][1]-es[0][1])
        if manhattan == 0: scores.append(0.0); continue
        scores.append(1.0 if (len(path)-1)/manhattan >= threshold else 0.0)
    return float(np.mean(scores)) if scores else 0.0

def horizontal_stripe_penalty(levels, threshold=0.75):
    scores=[]
    floor_tiles={FLOOR,PLAYER,ENEMY}
    for lvl in levels:
        h,w=lvl.shape
        stripe_rows=sum(1 for r in range(h)
                        if sum(1 for c in range(w) if lvl[r,c] in floor_tiles)/w>threshold)
        scores.append(stripe_rows/h)
    return float(np.mean(scores))

def dir_balance_fitness(levels):
    scores = []; ft = {FLOOR, PLAYER, ENEMY}
    for lvl in levels:
        h, w = lvl.shape; ht = vt = 0
        for r in range(h):
            for c in range(w):
                if lvl[r,c] in ft:
                    if c+1<w and lvl[r,c+1] in ft: ht+=1
                    if r+1<h and lvl[r+1,c] in ft: vt+=1
        mx = max(ht, vt); scores.append(min(ht,vt)/mx if mx>0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0

def bc_distance(la, lb):
    def bc(level):
        h, w = level.shape; wk = {FLOOR, PLAYER, ENEMY}
        interior = level[1:-1,1:-1]; wall_dens = float(np.sum(interior==WALL)/max(1,interior.size))
        sp = shortest_path_bfs(level); path_norm = min(1.0,(sp/(h+w)) if sp else 0.0)
        dead = branches = 0
        for r in range(h):
            for c in range(w):
                if level[r,c] not in wk: continue
                nb = sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                         if 0<=r+dr<h and 0<=c+dc<w and level[r+dr,c+dc] in wk)
                if nb==1: dead+=1
                elif nb>=3: branches+=1
        tw = max(1,int((level!=WALL).sum()))
        vis2 = np.zeros((h,w),bool); regs = 0
        for r0 in range(h):
            for c0 in range(w):
                if level[r0,c0] in wk and not vis2[r0,c0]:
                    regs+=1; q=deque([(r0,c0)]); vis2[r0,c0]=True
                    while q:
                        r,c=q.popleft()
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr,nc=r+dr,c+dc
                            if 0<=nr<h and 0<=nc<w and not vis2[nr,nc] and level[nr,nc] in wk:
                                vis2[nr,nc]=True; q.append((nr,nc))
        path,_ = _get_astar_result(level); diff = 0.0
        if path and len(path)>=3:
            t=0
            for i in range(1,len(path)-1):
                d1=(path[i][0]-path[i-1][0],path[i][1]-path[i-1][1])
                d2=(path[i+1][0]-path[i][0],path[i+1][1]-path[i][1])
                if d1!=d2: t+=1
            turns_n = min(1.0,t/max(1,len(path)-2))
        else: turns_n = 0.0
        return np.array([wall_dens,path_norm,dead/tw,branches/tw,min(1.0,regs/10.0),diff,turns_n])
    w = np.array([1,1,1,1,1,1,7.84]); d = bc(la)-bc(lb)
    return float(np.sqrt(np.sum(w*d*d))/np.sqrt(w.sum()))

def intra_novelty_score(levels, k=2):
    if len(levels)<2: return 0.0
    scores = []
    for i,li in enumerate(levels):
        dists = sorted(bc_distance(li,lj) for j,lj in enumerate(levels) if i!=j)
        scores.append(np.mean(dists[:min(k,len(dists))]))
    return float(np.mean(scores))

def mode_collapse_penalty(levels):
    def diag(l):
        walls=(l==WALL); h,w=l.shape
        if walls.sum()<8: return 0.0
        dp=tot=0
        for r in range(h-1):
            for c in range(w-1):
                if walls[r,c]: tot+=1; dp+=int(walls[r+1,c+1])
        return dp/max(1,tot)
    def dens(l):
        i=l[1:-1,1:-1]; d=float(np.sum(i==WALL)/max(1,i.size))
        if d<0.15: return 1.0-(d/0.15)
        if d>0.60: return min(1.0,(d-0.60)/0.40)
        return 0.0
    return 0.5*(float(np.mean([diag(l) for l in levels]))+float(np.mean([dens(l) for l in levels])))

def _count_turns(level):
    path,_ = _get_astar_result(level)
    if path is None or len(path)<3: return -1 if path is None else 0
    t=0
    for i in range(1,len(path)-1):
        d1=(path[i][0]-path[i-1][0],path[i][1]-path[i-1][1])
        d2=(path[i+1][0]-path[i][0],path[i+1][1]-path[i][1])
        if d1!=d2: t+=1
    return t

def _me_turns_fitness(levels, target=6):
    vals=[_count_turns(l) for l in levels if _count_turns(l)>=0]
    return float(np.mean([min(t,target)/target for t in vals])) if vals else 0.0

def _me_branching_fitness(levels, target=0.10):
    scores=[]
    for lvl in levels:
        h,w=lvl.shape; fl=br=0
        for r in range(h):
            for c in range(w):
                if lvl[r,c]!=WALL:
                    fl+=1; nb=sum(1 for dr,dc in[(-1,0),(1,0),(0,-1),(0,1)]
                                  if 0<=r+dr<h and 0<=c+dc<w and lvl[r+dr,c+dc]!=WALL)
                    if nb>=3: br+=1
        scores.append(min(1.0,(br/max(1,fl))/target))
    return float(np.mean(scores)) if scores else 0.0

# ME12-D7 archive: ht_rate x density (matches run_barrier.py "ht_rate" archive + notebook fix)
_ME_DENS_GOOD_MIN=0.15; _ME_DENS_GOOD_MAX=0.60
_ME_DENS_BIN_EDGES=[0.30,0.45]; _ME_N_HT_BINS=4; _ME_N_DENS_BINS=3
_ME_TOTAL_CELLS=12; _ME_BONUS_EMPTY=1.5; _ME_BONUS_IMPROVE=1.2
_ME_BONUS_WORSE=0.8; _ME_BONUS_BAD_DENS=0.3
_ME_W_SOLVE=0.40; _ME_W_INTRA=0.15; _ME_W_PATH=0.08
_ME_W_TURNS=0.02; _ME_W_BRANCH=0.05; _ME_W_HT=0.15; _ME_W_DIR=0.05; _ME_W_STRIPE=0.20
me_archive = {}

def _me_ht_rate_bin(rate):
    if rate<0.10: return 0
    if rate<0.20: return 1
    if rate<0.35: return 2
    return 3

def _me_density_bin(d):
    if d<_ME_DENS_GOOD_MIN or d>_ME_DENS_GOOD_MAX: return None
    for i,b in enumerate(_ME_DENS_BIN_EDGES):
        if d<b: return i
    return len(_ME_DENS_BIN_EDGES)

def me_genome_cell(levels):
    solv=[l for l in levels if is_solvable(l)]
    if not solv: return None
    tort_vals=[]
    for level in solv:
        path,_=_get_astar_result(level)
        if path is None: continue
        ps=list(zip(*np.where(level==PLAYER))); es=list(zip(*np.where(level==ENEMY)))
        if not ps or not es: continue
        manhattan=abs(ps[0][0]-es[0][0])+abs(ps[0][1]-es[0][1])
        if manhattan==0: continue
        tort_vals.append((len(path)-1)/manhattan)
    if not tort_vals: return None
    rate=sum(1 for v in tort_vals if v>=1.5)/len(tort_vals)
    dens=[float(np.sum(l[1:-1,1:-1]==WALL)/max(1,l[1:-1,1:-1].size)) for l in levels]
    db=_me_density_bin(float(np.mean(dens)))
    if db is None: return None
    return (_me_ht_rate_bin(rate),db)

import copy
_best_cv3_genome = None
_best_cv3_val = 0.0

def eval_genomes(genomes, config):
    global me_archive, _best_cv3_genome, _best_cv3_val
    genome_levels={}
    for gid,genome in genomes:
        net=neat.nn.RecurrentNetwork.create(genome,config)
        genome_levels[gid]=[generate_level(net) for _ in range(MAPS_PER_GENOME)]
    gen_idx=len(me_archive)
    for gid,genome in genomes:
        levels=genome_levels[gid]
        f_solve=solvability_fitness(levels); f_intra=intra_novelty_score(levels,k=min(10,len(levels)-1))
        f_path=path_diversity_fitness(levels); f_turns=_me_turns_fitness(levels)
        f_branch=_me_branching_fitness(levels); f_ht=high_tort_fitness(levels)
        f_dir=dir_balance_fitness(levels); mc_pen=0.5*mode_collapse_penalty(levels)
        f_stripe=horizontal_stripe_penalty(levels)
        base=max(0.0,(_ME_W_SOLVE*f_solve+_ME_W_INTRA*f_intra+_ME_W_PATH*f_path
                      +_ME_W_TURNS*f_turns+_ME_W_BRANCH*f_branch
                      +_ME_W_HT*f_ht+_ME_W_DIR*f_dir
                      -_ME_W_STRIPE*f_stripe)-mc_pen)
        cv3=f_solve*f_ht*f_dir
        if cv3>_best_cv3_val:
            _best_cv3_val=cv3; _best_cv3_genome=copy.deepcopy(genome)
        cell=me_genome_cell(levels)
        if cell is None: genome.fitness=base*_ME_BONUS_BAD_DENS
        elif cell not in me_archive:
            genome.fitness=base*_ME_BONUS_EMPTY
            me_archive[cell]={"fitness":base,"levels":levels,"gen":gen_idx,"genome":copy.deepcopy(genome)}
        elif base>me_archive[cell]["fitness"]:
            genome.fitness=base*_ME_BONUS_IMPROVE
            me_archive[cell]={"fitness":base,"levels":levels,"gen":gen_idx,"genome":copy.deepcopy(genome)}
        else: genome.fitness=base*_ME_BONUS_WORSE

# ── NEAT config ─────────────────────────────────────────────────────
config_str = f"""
[NEAT]
fitness_criterion=max
fitness_threshold=999999
pop_size={POP_SIZE}
reset_on_extinction=False
no_fitness_termination=True
[DefaultGenome]
num_inputs={NUM_INPUTS}
num_outputs={NUM_OUTPUTS}
num_hidden=0
feed_forward=False
initial_connection=full_direct
node_add_prob=0.6
node_delete_prob=0.2
conn_add_prob=0.5
conn_delete_prob=0.3
activation_default=sigmoid
activation_mutate_rate=0.15
activation_options=sigmoid sin gauss
aggregation_default=sum
aggregation_mutate_rate=0.0
aggregation_options=sum
bias_init_mean=0.0
bias_init_stdev=1.0
bias_init_type=gaussian
bias_max_value=30.0
bias_min_value=-30.0
bias_mutate_power=0.5
bias_mutate_rate=0.7
bias_replace_rate=0.1
response_init_mean=1.0
response_init_stdev=0.0
response_init_type=gaussian
response_max_value=30.0
response_min_value=-30.0
response_mutate_power=0.0
response_mutate_rate=0.0
response_replace_rate=0.0
weight_init_mean=0.0
weight_init_stdev=1.0
weight_init_type=gaussian
weight_max_value=30.0
weight_min_value=-30.0
weight_mutate_power=0.5
weight_mutate_rate=0.8
weight_replace_rate=0.15
enabled_default=True
enabled_mutate_rate=0.02
enabled_rate_to_true_add=0.0
enabled_rate_to_false_add=0.0
compatibility_disjoint_coefficient=1.0
compatibility_weight_coefficient=0.5
single_structural_mutation=False
structural_mutation_surer=default
[DefaultSpeciesSet]
compatibility_threshold=3.0
[DefaultStagnation]
species_fitness_func=max
max_stagnation=20
species_elitism=2
[DefaultReproduction]
elitism=3
survival_threshold=0.2
min_species_size=2
"""

import tempfile, os
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write(config_str); cfg_path = f.name

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path)
os.unlink(cfg_path)

# ── run ─────────────────────────────────────────────────────────────
print(f"ME12-D6 smoke test: {MAX_GEN} gen, {POP_SIZE} pop, {MAPS_PER_GENOME} maps/genome")
print(f"Weights: solve={_ME_W_SOLVE}, intra={_ME_W_INTRA}, path={_ME_W_PATH}, "
      f"turns={_ME_W_TURNS}, branch={_ME_W_BRANCH}, ht={_ME_W_HT}, dir={_ME_W_DIR}")
print(f"Recurrent: feed_forward=False, RecurrentNetwork")
print()

t0 = time.time()
random.seed(0); np.random.seed(0)
pop = neat.Population(config)

class Reporter(neat.reporting.BaseReporter):
    def post_evaluate(self, config, population, species, best_genome):
        g = len(gen_log)
        fits = [gn.fitness for gn in population.values() if gn.fitness is not None]
        best_net = neat.nn.RecurrentNetwork.create(best_genome, config)
        sample = [generate_level(best_net) for _ in range(16)]
        sol = float(np.mean([float(is_solvable(l)) for l in sample]))
        ht = high_tort_fitness(sample)
        db = dir_balance_fitness(sample)
        gen_log.append({"g":g,"best":max(fits),"sol":sol,"ht":ht,"dir":db,"archive":len(me_archive)})
        print(f"  gen {g:2d}/{MAX_GEN}  best={max(fits):.3f}  sol={sol:.2f}  ht={ht:.2f}  dir={db:.3f}  archive={len(me_archive)}/{_ME_TOTAL_CELLS}")

gen_log = []
pop.add_reporter(Reporter())
winner = pop.run(eval_genomes, MAX_GEN)
elapsed = time.time() - t0

# D7: override winner with best compound_v3 genome
if _best_cv3_genome is not None and _best_cv3_val > 0:
    winner = _best_cv3_genome
    print(f"\n   cv3 override: best compound_v3={_best_cv3_val:.4f} (ht>0 genome selected)")
else:
    print(f"\n   NEAT winner (no ht>0 genome found)")

print(f"Smoke test done in {elapsed:.0f}s")
print(f"   Final archive: {len(me_archive)}/{_ME_TOTAL_CELLS} cells")
print(f"   Winner nodes={len(winner.nodes)}, conns={len(winner.connections)}")

# save winner
with open(".lab/workspace/test_winner_d7.pkl", "wb") as f:
    pickle.dump(winner, f)
print(f"   Winner saved: .lab/workspace/test_winner_d7.pkl")
