"""Visualize sample maps + compute detailed metrics from a winner pkl."""
import sys, os, pickle, random, heapq
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

PKL_PATH = sys.argv[1] if len(sys.argv) > 1 else "neat_winner_seed0.pkl"
N_MAPS   = int(sys.argv[2]) if len(sys.argv) > 2 else 50

with open(PKL_PATH, "rb") as f:
    winner = pickle.load(f)

import neat, tempfile
WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3
NUM_INPUTS, NUM_OUTPUTS, POP_SIZE = 12, 1, 50
CONTEXT_SIZE = 1; NUM_RANDOM_INPUTS = 4; PERTURB_SIZE = 0.1565

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
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write(config_str); cfg_path = f.name
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path)
os.unlink(cfg_path)
net = neat.nn.RecurrentNetwork.create(winner, config)

def generate_level():
    net.reset()
    half = CONTEXT_SIZE; map_h = map_w = 14
    padded = np.full((map_h+2*half, map_w+2*half), -1.0)
    noise = [random.gauss(0,1) for _ in range(NUM_RANDOM_INPUTS)]
    for r in range(half, map_h+half):
        for c in range(half, map_w+half):
            ctx = [padded[r+dr,c+dc] for dr in range(-half,half+1)
                   for dc in range(-half,half+1) if not(dr==0 and dc==0)]
            inputs = [x+random.gauss(0,PERTURB_SIZE) for x in ctx+noise]
            padded[r,c] = 1.0 if net.activate(inputs)[0] > 0.5 else 0.0
    lv = padded[half:half+map_h, half:half+map_w].astype(int)
    if lv[0,0]==FLOOR: lv[0,0]=PLAYER
    if lv[-1,-1]==FLOOR: lv[-1,-1]=ENEMY
    return lv

def astar(level):
    h,w=level.shape
    ps=list(zip(*np.where(level==PLAYER))); es=list(zip(*np.where(level==ENEMY)))
    if not ps or not es: return None
    start=ps[0]; goal=es[0]
    open_q=[(abs(start[0]-goal[0])+abs(start[1]-goal[1]),0,start)]
    came={}; g={start:0}; vis=set(); cnt=0
    while open_q:
        _,cost,cur=heapq.heappop(open_q)
        if cur in vis: continue
        vis.add(cur); cnt+=1
        if cur==goal:
            path=[]; c2=cur
            while c2 in came: path.append(c2); c2=came[c2]
            path.append(start); return list(reversed(path))
        r,c=cur
        for dr,dc in[(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc=r+dr,c+dc
            if 0<=nr<h and 0<=nc<w and (nr,nc) not in vis and level[nr,nc]!=WALL:
                ng=cost+1
                if ng<g.get((nr,nc),1e9):
                    came[(nr,nc)]=cur; g[(nr,nc)]=ng; cnt+=1
                    heapq.heappush(open_q,(ng+abs(nr-goal[0])+abs(nc-goal[1]),ng,(nr,nc)))
    return None

def tortuosity(level, path):
    if path is None: return 0.0
    ps=list(zip(*np.where(level==PLAYER))); es=list(zip(*np.where(level==ENEMY)))
    if not ps or not es: return 0.0
    manhattan=abs(ps[0][0]-es[0][0])+abs(ps[0][1]-es[0][1])
    return (len(path)-1)/manhattan if manhattan>0 else 0.0

def count_turns(path):
    if path is None or len(path)<3: return 0
    t=0
    for i in range(1,len(path)-1):
        if (path[i][0]-path[i-1][0],path[i][1]-path[i-1][1])!=(path[i+1][0]-path[i][0],path[i+1][1]-path[i][1]):
            t+=1
    return t

# generate
random.seed(42); np.random.seed(42)
maps=[generate_level() for _ in range(N_MAPS)]
paths=[astar(m) for m in maps]
torts=[tortuosity(maps[i],paths[i]) for i in range(N_MAPS)]
turns_list=[count_turns(p) for p in paths]

# sort by tortuosity descending — show best first
order = sorted(range(N_MAPS), key=lambda i: torts[i], reverse=True)

TILE_COLORS={WALL:[0.2,0.2,0.2],FLOOR:[0.95,0.95,0.95],PLAYER:[0.2,0.8,0.2],ENEMY:[0.9,0.2,0.2]}

# plot top-20 (most tortuous) vs bottom-5 (least)
show_top=15; show_bot=5
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
axes = axes.flatten()

for plot_i, map_i in enumerate(order[:show_top]):
    ax = axes[plot_i]
    lv = maps[map_i]; p = paths[map_i]
    img = np.zeros((14,14,3))
    for tile,col in TILE_COLORS.items(): img[lv==tile]=col
    ax.imshow(img, interpolation='nearest')
    if p:
        ys=[pt[0] for pt in p]; xs=[pt[1] for pt in p]
        ax.plot(xs, ys, 'b-', lw=1.5, alpha=0.7)
    t=torts[map_i]; turns=turns_list[map_i]
    ax.set_title(f'tort={t:.2f} turns={turns}', fontsize=8)
    ax.axis('off')

for plot_i, map_i in enumerate(order[N_MAPS-show_bot:]):
    ax = axes[show_top + plot_i]
    lv = maps[map_i]; p = paths[map_i]
    img = np.zeros((14,14,3))
    for tile,col in TILE_COLORS.items(): img[lv==tile]=col
    ax.imshow(img, interpolation='nearest')
    if p:
        ys=[pt[0] for pt in p]; xs=[pt[1] for pt in p]
        ax.plot(xs, ys, 'r-', lw=1.5, alpha=0.7)
    t=torts[map_i]; turns=turns_list[map_i]
    ax.set_title(f'tort={t:.2f} turns={turns}', fontsize=8, color='red')
    ax.axis('off')

axes[show_top-1].set_title(f'tort={torts[order[show_top-1]]:.2f} ...', fontsize=7)
plt.suptitle(f'Top-{show_top} tortuous (blue) | Bottom-{show_bot} (red)\n'
             f'sol={np.mean([p is not None for p in paths]):.0%}  '
             f'ht≥1.5={np.mean([t>=1.5 for t in torts]):.0%}  '
             f'mean_tort={np.mean(torts):.2f}  mean_turns={np.mean(turns_list):.1f}',
             fontsize=11)
plt.tight_layout()
out = PKL_PATH.replace('.pkl','_maps.png').replace(' ','_')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"Saved: {out}")

# tortuosity distribution
fig2, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
ax1.hist(torts, bins=20, color='steelblue', edgecolor='white')
ax1.axvline(1.5, color='red', linestyle='--', label='ht threshold=1.5')
ax1.set(xlabel='Tortuosity', ylabel='Count', title='Tortuosity Distribution')
ax1.legend()
ax2.hist(turns_list, bins=range(0,max(turns_list)+2), color='purple', edgecolor='white')
ax2.set(xlabel='Number of turns', ylabel='Count', title='Turns Distribution')
plt.tight_layout()
out2 = out.replace('_maps.png','_dist.png')
plt.savefig(out2, dpi=120, bbox_inches='tight')
print(f"Saved: {out2}")
plt.show()
