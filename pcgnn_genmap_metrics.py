from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
from collections import deque
from pathlib import Path

import neat
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = PROJECT_DIR / ".matplotlib"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))


# improve-v2.ipynb constants.
MIN_WIDTH, MAX_WIDTH = 14, 14
MIN_HEIGHT, MAX_HEIGHT = 14, 14
MAPS_PER_GENOME = 24
MAX_GEN = 200
POP_SIZE = 50

WALL = 0
FLOOR = 1
PLAYER = 2
ENEMY = 3

CONTEXT_SIZE = 1
CTX_TILES = (2 * CONTEXT_SIZE + 1) ** 2 - 1
NUM_RANDOM_INPUTS = 4
NUM_INPUTS = CTX_TILES + NUM_RANDOM_INPUTS
NUM_OUTPUTS = 1
PERTURB_SIZE = 0.1565


def write_improve_v2_config(path: Path) -> None:
    config_content = f"""
[NEAT]
fitness_criterion      = max
fitness_threshold      = 999999
pop_size               = {POP_SIZE}
reset_on_extinction    = False
no_fitness_termination = True

[DefaultGenome]
num_inputs             = {NUM_INPUTS}
num_outputs            = {NUM_OUTPUTS}
num_hidden             = 0
feed_forward           = False
initial_connection     = full_direct

node_add_prob          = 0.6
node_delete_prob       = 0.2
conn_add_prob          = 0.5
conn_delete_prob       = 0.3

activation_default     = sigmoid
activation_mutate_rate = 0.15
activation_options     = sigmoid sin gauss

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_init_type          = gaussian
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_init_type      = gaussian
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_init_type        = gaussian
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.15

enabled_default         = True
enabled_mutate_rate     = 0.02
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
    path.write_text(config_content.strip() + "\n", encoding="utf-8")


def generate_level(net, map_h: int | None = None, map_w: int | None = None, perturb: bool = True) -> np.ndarray:
    """Exact generation path from improve-v2.ipynb cell 6."""
    if hasattr(net, "reset"):
        net.reset()
    if map_h is None:
        map_h = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    if map_w is None:
        map_w = random.randint(MIN_WIDTH, MAX_WIDTH)

    half = CONTEXT_SIZE
    padded = np.random.randint(0, 2, (map_h + 2 * half, map_w + 2 * half)).astype(float)
    noise = [random.gauss(0, 1) for _ in range(NUM_RANDOM_INPUTS)]

    for row in range(half, map_h + half):
        for col in range(half, map_w + half):
            ctx = []
            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    if dr == 0 and dc == 0:
                        continue
                    ctx.append(padded[row + dr, col + dc])

            inputs = ctx + noise
            if perturb:
                inputs = [x + random.gauss(0, PERTURB_SIZE) for x in inputs]

            out = net.activate(inputs)[0]
            padded[row, col] = 1.0 if out > 0.5 else 0.0

    level = padded[half:half + map_h, half:half + map_w].astype(int)

    if level[0, 0] == FLOOR:
        level[0, 0] = PLAYER
    if level[map_h - 1, map_w - 1] == FLOOR:
        level[map_h - 1, map_w - 1] = ENEMY

    return level


def add_wall_border(level: np.ndarray) -> np.ndarray:
    bordered = np.full((level.shape[0] + 2, level.shape[1] + 2), WALL, dtype=np.int32)
    bordered[1:-1, 1:-1] = level.astype(np.int32)
    return bordered


def floor_only(level: np.ndarray) -> np.ndarray:
    cleaned = level.copy()
    cleaned[cleaned == PLAYER] = FLOOR
    cleaned[cleaned == ENEMY] = FLOOR
    return cleaned


def connected_components(level: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = level.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for row in range(h):
        for col in range(w):
            if seen[row, col] or level[row, col] != FLOOR:
                continue
            component = []
            queue = deque([(row, col)])
            seen[row, col] = True
            while queue:
                r, c = queue.popleft()
                component.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and level[nr, nc] == FLOOR:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(component)
    return components


def distances_from(level: np.ndarray, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    distances = {start: 0}
    queue = deque([start])
    h, w = level.shape
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            pos = (nr, nc)
            if 0 <= nr < h and 0 <= nc < w and pos not in distances and level[nr, nc] != WALL:
                distances[pos] = distances[(row, col)] + 1
                queue.append(pos)
    return distances


def place_player_enemy(level: np.ndarray) -> np.ndarray:
    """Adapter step: place P/E on the largest reachable floor component."""
    adapted = floor_only(level)
    components = connected_components(adapted)
    if not components:
        return adapted

    largest = max(components, key=len)
    if len(largest) < 2:
        return adapted

    # Two-sweep BFS gives a stable, cheap approximation of graph diameter.
    first = min(largest)
    far_a = max(distances_from(adapted, first).items(), key=lambda item: item[1])[0]
    dist_from_a = distances_from(adapted, far_a)
    far_b = max(dist_from_a.items(), key=lambda item: item[1])[0]

    adapted[far_a] = PLAYER
    adapted[far_b] = ENEMY
    return adapted


def walkable_mask(level: np.ndarray) -> np.ndarray:
    return (level == FLOOR) | (level == PLAYER) | (level == ENEMY)


def player_and_enemy(level: np.ndarray):
    players = list(zip(*np.where(level == PLAYER)))
    enemies = list(zip(*np.where(level == ENEMY)))
    if not players or not enemies:
        return None, None
    return players[0], enemies[0]


def bfs(level: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    if level[start] == WALL or level[goal] == WALL:
        return None, set()

    h, w = level.shape
    previous = {start: None}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = previous[current]
            return list(reversed(path)), set(previous.keys())

        row, col = current
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in previous and level[nr, nc] != WALL:
                previous[(nr, nc)] = current
                queue.append((nr, nc))

    return None, set(previous.keys())


def count_open_neighbors(level: np.ndarray, row: int, col: int) -> int:
    h, w = level.shape
    total = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < h and 0 <= nc < w and level[nr, nc] != WALL:
            total += 1
    return total


def reachability_ratio(level: np.ndarray) -> float:
    walkable = walkable_mask(level)
    total = int(walkable.sum())
    if total == 0:
        return 0.0

    h, w = level.shape
    visited = np.zeros_like(walkable, dtype=bool)
    best = 0
    for row in range(h):
        for col in range(w):
            if not walkable[row, col] or visited[row, col]:
                continue
            size = 0
            queue = deque([(row, col)])
            visited[row, col] = True
            while queue:
                r, c = queue.popleft()
                size += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and walkable[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            best = max(best, size)
    return best / total


def map_metrics(level: np.ndarray) -> dict[str, float | int | bool]:
    start, goal = player_and_enemy(level)
    path, visited = bfs(level, start, goal) if start is not None and goal is not None else (None, set())

    walkable = walkable_mask(level)
    floor_cells = int(walkable.sum())
    wall_cells = int(np.sum(level == WALL))
    total_cells = int(level.size)
    open_cells = [(r, c) for r in range(level.shape[0]) for c in range(level.shape[1]) if level[r, c] != WALL]

    interior = level[1:-1, 1:-1] if level.shape[0] > 2 and level.shape[1] > 2 else level
    interior_wall_density = float(np.sum(interior == WALL) / interior.size) if interior.size else 0.0
    dead_ends = sum(1 for r, c in open_cells if count_open_neighbors(level, r, c) <= 1)
    branches = sum(1 for r, c in open_cells if count_open_neighbors(level, r, c) >= 3)
    path_len = len(path) - 1 if path else 0
    max_path = max(1, level.shape[0] + level.shape[1])

    reachable_ratio = reachability_ratio(level)
    wall_ratio = wall_cells / total_cells if total_cells else 0.0
    dead_end_ratio = dead_ends / floor_cells if floor_cells else 0.0
    branching_ratio = branches / floor_cells if floor_cells else 0.0
    path_norm = path_len / max_path if path else 0.0

    if path:
        leniency = float(np.mean([1.0 if count_open_neighbors(level, r, c) >= 2 else 0.0 for r, c in path]))
        astar_difficulty = max(0.0, (len(visited) - len(path)) / floor_cells) if floor_cells else 0.0
    else:
        leniency = 0.0
        astar_difficulty = 1.0

    difficulty_score = (
        0.90 * wall_ratio
        + 0.05 * min(path_norm, 1.0)
        + 0.03 * dead_end_ratio
        + 0.02 * astar_difficulty
    )

    return {
        "has_player": start is not None,
        "has_enemy": goal is not None,
        "solvable": path is not None,
        "wall_ratio": wall_ratio,
        "interior_wall_density": interior_wall_density,
        "walkable_cells": floor_cells,
        "reachable_ratio": reachable_ratio,
        "shortest_path_length": path_len,
        "path_norm": path_norm,
        "dead_end_ratio": dead_end_ratio,
        "branching_ratio": branching_ratio,
        "leniency": leniency,
        "astar_difficulty": astar_difficulty,
        "difficulty_score": difficulty_score,
    }


def score_tier(score: float) -> str:
    if score < 0.4:
        return "easy"
    if score < 0.5:
        return "medium"
    return "hard"


def range_tier(row: dict[str, float | int | bool]) -> str:
    wall = float(row["wall_ratio"])
    path = float(row["path_norm"])
    dead = float(row["dead_end_ratio"])
    if 0.20 <= wall <= 0.35 and 0.25 <= path <= 0.40 and 0.00 <= dead <= 0.25:
        return "easy"
    if 0.35 < wall <= 0.50 and 0.40 < path <= 0.65 and 0.10 <= dead <= 0.35:
        return "medium"
    if 0.50 < wall <= 0.70 and 0.65 < path <= 1.00 and 0.20 <= dead <= 0.60:
        return "hard"
    return "unclassified"


def assign_percentile_tiers(rows: list[dict[str, object]], easy_ratio: float, medium_ratio: float) -> None:
    ordered = sorted(rows, key=lambda row: float(row["difficulty_score"]))
    easy_count = int(round(len(ordered) * easy_ratio))
    medium_count = int(round(len(ordered) * medium_ratio))

    for index, row in enumerate(ordered):
        if index < easy_count:
            row["percentile_tier"] = "easy"
        elif index < easy_count + medium_count:
            row["percentile_tier"] = "medium"
        else:
            row["percentile_tier"] = "hard"


def map_to_text(level: np.ndarray) -> str:
    symbols = {WALL: "#", FLOOR: ".", PLAYER: "P", ENEMY: "E"}
    return "\n".join("".join(symbols.get(int(tile), "?") for tile in row) for row in level) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate maps from a PCGNN genome using the improve-v2.ipynb generator.")
    parser.add_argument("--checkpoint", default="inctyseed0.pkl")
    parser.add_argument("--config", default="config-pcgnn.txt")
    parser.add_argument("--out", default="generated_maps/inctyseed0_improve_v2")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=14)
    parser.add_argument("--width", type=int, default=14)
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--add-border", action="store_true")
    parser.add_argument("--auto-spawn", action="store_true")
    parser.add_argument("--percentile-tiers", action="store_true")
    parser.add_argument("--easy-ratio", type=float, default=0.05)
    parser.add_argument("--medium-ratio", type=float, default=0.05)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config_path = Path(args.config)
    write_improve_v2_config(config_path)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    with open(args.checkpoint, "rb") as handle:
        genome = pickle.load(handle)

    net = neat.nn.RecurrentNetwork.create(genome, config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(args.count):
        level = generate_level(net, map_h=args.height, map_w=args.width, perturb=not args.no_perturb)
        if args.auto_spawn:
            level = place_player_enemy(level)
        exported = add_wall_border(level) if args.add_border else level
        filename = f"pcgnn_{index:03d}.txt"
        (out_dir / filename).write_text(map_to_text(exported), encoding="utf-8")
        metrics = map_metrics(exported)
        rows.append({
            "file": filename,
            **metrics,
            "score_tier": score_tier(float(metrics["difficulty_score"])),
            "range_tier": range_tier(metrics),
            "percentile_tier": "",
        })

    if args.percentile_tiers:
        assign_percentile_tiers(rows, args.easy_ratio, args.medium_ratio)

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    hashes = [(out_dir / row["file"]).read_bytes() for row in rows]
    unique_count = len({data for data in hashes})
    solvable_count = sum(1 for row in rows if row["solvable"])
    score_counts = {tier: sum(1 for row in rows if row["score_tier"] == tier) for tier in ("easy", "medium", "hard")}
    range_counts = {tier: sum(1 for row in rows if row["range_tier"] == tier) for tier in ("easy", "medium", "hard", "unclassified")}
    percentile_counts = {tier: sum(1 for row in rows if row["percentile_tier"] == tier) for tier in ("easy", "medium", "hard")}

    print(f"checkpoint={args.checkpoint}")
    print(f"config={config_path}")
    print("generator=improve-v2.ipynb cell 6")
    print(f"maps={args.count}")
    print(f"unique_maps={unique_count}")
    print(f"output={out_dir}")
    print(f"metrics_csv={csv_path}")
    print(f"auto_spawn={args.auto_spawn}")
    print(f"percentile_tiers={args.percentile_tiers} easy_ratio={args.easy_ratio:.3f} medium_ratio={args.medium_ratio:.3f}")
    print(f"solvability={solvable_count}/{args.count} ({100 * solvable_count / args.count:.1f}%)")
    print(f"score_tiers=easy:{score_counts['easy']} medium:{score_counts['medium']} hard:{score_counts['hard']}")
    print(f"range_tiers=easy:{range_counts['easy']} medium:{range_counts['medium']} hard:{range_counts['hard']} unclassified:{range_counts['unclassified']}")
    print(f"percentile_tiers=easy:{percentile_counts['easy']} medium:{percentile_counts['medium']} hard:{percentile_counts['hard']}")
    print(f"mean_wall_ratio={np.mean([row['wall_ratio'] for row in rows]):.3f}")
    print(f"mean_interior_wall_density={np.mean([row['interior_wall_density'] for row in rows]):.3f}")
    print(f"mean_path_length={np.mean([row['shortest_path_length'] for row in rows]):.2f}")
    print(f"mean_reachable_ratio={np.mean([row['reachable_ratio'] for row in rows]):.3f}")
    print(f"mean_dead_end_ratio={np.mean([row['dead_end_ratio'] for row in rows]):.3f}")
    print(f"mean_leniency={np.mean([row['leniency'] for row in rows]):.3f}")
    print(f"mean_astar_difficulty={np.mean([row['astar_difficulty'] for row in rows]):.3f}")
    print(f"mean_difficulty_score={np.mean([row['difficulty_score'] for row in rows]):.3f}")
    print()
    print("first_10_maps:")
    for row in rows[:10]:
        print(
            f"{row['file']}: P={row['has_player']} E={row['has_enemy']} "
            f"solvable={row['solvable']} wall={row['wall_ratio']:.3f} "
            f"interior_wall={row['interior_wall_density']:.3f} path={row['shortest_path_length']} "
            f"reach={row['reachable_ratio']:.3f} dead={row['dead_end_ratio']:.3f} "
            f"leniency={row['leniency']:.3f} astar_diff={row['astar_difficulty']:.3f} "
            f"score={row['difficulty_score']:.3f}"
            f" score_tier={row['score_tier']} range_tier={row['range_tier']}"
            f" percentile_tier={row['percentile_tier']}"
        )


if __name__ == "__main__":
    main()
