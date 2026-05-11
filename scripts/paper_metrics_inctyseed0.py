from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import sys
from pathlib import Path

import neat
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "external" / "gym-pcgrl"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.a_star.a_star_metrics import (
    AStarDifficultyMetric,
    AStarDiversityAndDifficultyMetric,
    AStarDiversityMetric,
    AStarEditDistanceDiversityMetric,
    AStarSolvabilityMetric,
)
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric

from pcgnn_genmap_metrics import (
    ENEMY,
    FLOOR,
    PLAYER,
    WALL,
    generate_level,
    place_player_enemy,
    write_improve_v2_config,
)


def to_maze_level(raw_level: np.ndarray, spawn: str) -> MazeLevel:
    level = raw_level.copy()
    if spawn == "auto":
        level = place_player_enemy(level)

    players = list(zip(*np.where(level == PLAYER)))
    enemies = list(zip(*np.where(level == ENEMY)))

    # Notebook encoding: WALL=0, FLOOR/P/E are walkable.
    # Paper MazeLevel encoding: 1=filled wall, 0=empty floor.
    maze_map = np.where(level == WALL, 1, 0).astype(np.int32)

    if spawn == "auto" and players and enemies:
        start = (int(players[0][1]), int(players[0][0]))
        end = (int(enemies[0][1]), int(enemies[0][0]))
    else:
        start = (0, 0)
        end = (maze_map.shape[1] - 1, maze_map.shape[0] - 1)

    return MazeLevel.from_map(maze_map, start=start, end=end)


def metric_summary(name: str, values) -> dict[str, object]:
    values = list(values)
    return {
        "metric": name,
        "count": len(values),
        "mean": float(np.mean(values)) if values else float("nan"),
        "std": float(np.std(values)) if values else float("nan"),
        "min": float(np.min(values)) if values else float("nan"),
        "max": float(np.max(values)) if values else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/inctyseed0.pkl")
    parser.add_argument("--config", default="configs/config-pcgnn.txt")
    parser.add_argument("--out", default="generated_maps/inctyseed0_paper_metrics_14x14_100")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=14)
    parser.add_argument("--width", type=int, default=14)
    parser.add_argument("--spawn", choices=["corner", "auto"], default="corner")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config_path = PROJECT_DIR / args.config
    write_improve_v2_config(config_path)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    checkpoint_path = PROJECT_DIR / args.checkpoint
    with checkpoint_path.open("rb") as handle:
        genome = pickle.load(handle)
    net = neat.nn.RecurrentNetwork.create(genome, config)

    levels = [
        to_maze_level(generate_level(net, map_h=args.height, map_w=args.width), args.spawn)
        for _ in range(args.count)
    ]

    game = MazeGame(MazeLevel(args.width, args.height))
    parent = AStarDiversityAndDifficultyMetric(game, number_of_times_to_do_evaluation=5)
    metrics = [
        ("CompressionDistanceMetric", CompressionDistanceMetric(game)),
        ("LeniencyMetric", LeniencyMetric(game)),
        ("SolvabilityMetric", AStarSolvabilityMetric(game, parent)),
        ("AStarDiversityMetric", AStarDiversityMetric(game, parent)),
        ("AStarDifficultyMetric", AStarDifficultyMetric(game, parent)),
        ("AStarEditDistanceDiversityMetric", AStarEditDistanceDiversityMetric(game, parent)),
    ]

    out_dir = PROJECT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_values = {}
    for name, metric in metrics:
        values = metric.evaluate(levels)
        all_values[name] = list(map(float, values))
        summaries.append(metric_summary(name, values))

    summary_csv = out_dir / "paper_metrics_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "count", "mean", "std", "min", "max"])
        writer.writeheader()
        writer.writerows(summaries)

    values_csv = out_dir / "paper_metrics_values.csv"
    max_len = max(len(v) for v in all_values.values())
    with values_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_values.keys()))
        writer.writeheader()
        for i in range(max_len):
            writer.writerow({k: v[i] if i < len(v) else "" for k, v in all_values.items()})

    print(f"checkpoint={args.checkpoint}")
    print(f"spawn={args.spawn}")
    print(f"levels={len(levels)}")
    print(f"output={out_dir}")
    print(f"summary_csv={summary_csv}")
    for row in summaries:
        print(f"{row['metric']}: mean={row['mean']:.6f} std={row['std']:.6f} count={row['count']}")


if __name__ == "__main__":
    main()
