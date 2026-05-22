"""Wrappers around the existing baseline + improved map generators.

Imports the generator functions directly from the project root so the gen path
stays identical to `pcgnn_genmap_metrics.py` (improved, RecurrentNetwork) and
`render_model_map_samples.py::generate_baseline_level` (baseline, FeedForward).
"""

from __future__ import annotations

import random
import sys
import threading
from pathlib import Path
from typing import Any

import neat
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pcgnn_genmap_metrics as improved  # noqa: E402
import render_model_map_samples as renderer  # noqa: E402

BASELINE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "baseline" / "neat_winner_seed0.pkl"
IMPROVED_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "improved" / "inctyseed0.pkl"
BASELINE_CONFIG_PATH = PROJECT_ROOT / "_render_baseline_config.txt"
IMPROVED_CONFIG_PATH = PROJECT_ROOT / "_render_improved_config.txt"

_BASELINE_NET = None
_IMPROVED_NET = None
_LOCK = threading.Lock()


def _load_baseline_net():
    """Baseline: try FFN with the improved config (same num_inputs/outputs),
    probe one map, fall back to RNN if it comes out all walls.

    This mirrors `render_model_map_samples.load_baseline_net` but uses our
    deterministic `_render_improved_config.txt` path instead of the user's
    working `config-pcgnn.txt` (which the metrics CLI overwrites at runtime).
    """
    import pickle

    # Make sure the improved config exists on disk (written by improved.write_improve_v2_config).
    improved.write_improve_v2_config(IMPROVED_CONFIG_PATH)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(IMPROVED_CONFIG_PATH),
    )
    with BASELINE_CHECKPOINT.open("rb") as handle:
        genome = pickle.load(handle)

    ff_net = neat.nn.FeedForwardNetwork.create(genome, config)
    # Probe with the renderer's baseline generator (border = -1 sentinel).
    # Seed deterministically so the probe outcome is stable across server restarts.
    random.seed(11)
    np.random.seed(11)
    probe = renderer.generate_baseline_level(ff_net)
    if int(np.sum(probe == renderer.BASELINE_WALL)) < probe.size:
        return ff_net
    return neat.nn.RecurrentNetwork.create(genome, config)


def _load_improved_net():
    """Improved: feed_forward=False config + RecurrentNetwork.

    Reuses `render_model_map_samples.load_improved_net` so the config-writing
    side effect stays identical to what the rendering pipeline does.
    """
    return renderer.load_improved_net(IMPROVED_CHECKPOINT)


def get_net(model: str):
    """Lazy + cached. The genome and config never change at runtime."""
    global _BASELINE_NET, _IMPROVED_NET
    with _LOCK:
        if model == "baseline":
            if _BASELINE_NET is None:
                _BASELINE_NET = _load_baseline_net()
            return _BASELINE_NET
        if model == "improved":
            if _IMPROVED_NET is None:
                _IMPROVED_NET = _load_improved_net()
            return _IMPROVED_NET
        raise ValueError(f"unknown model: {model!r}")


def generate_map(model: str, height: int = 14, width: int = 14, perturb: bool = True) -> np.ndarray:
    """Single map generation. Caller is responsible for seeding RNG."""
    net = get_net(model)
    if model == "baseline":
        return renderer.generate_baseline_level(net, map_h=height, map_w=width, perturb=perturb)
    return improved.generate_level(net, map_h=height, map_w=width, perturb=perturb)


def generate_batch(
    model: str,
    count: int,
    seed: int,
    height: int = 14,
    width: int = 14,
    perturb: bool = True,
) -> list[np.ndarray]:
    """Seed once, then draw `count` maps. Mirrors `sample_*_maps` in the renderer."""
    random.seed(seed)
    np.random.seed(seed)
    return [generate_map(model, height, width, perturb) for _ in range(count)]


def level_to_grid(level: np.ndarray) -> list[list[int]]:
    """Frontend-friendly: nested int list of WALL=0/FLOOR=1/PLAYER=2/ENEMY=3."""
    return level.astype(int).tolist()


def level_metrics(level: np.ndarray) -> dict[str, Any]:
    """Reuse the improved-pipeline metrics so numbers match metrics.csv."""
    metrics = improved.map_metrics(level)
    return {
        key: (float(value) if isinstance(value, (np.floating, float)) else
              int(value) if isinstance(value, (np.integer, int)) else
              bool(value) if isinstance(value, (np.bool_, bool)) else value)
        for key, value in metrics.items()
    }


def classify_rows(rows: list[dict[str, Any]], easy_ratio: float, medium_ratio: float) -> None:
    """In-place: write `percentile_tier` on each row. Mirrors improved.assign_percentile_tiers."""
    improved.assign_percentile_tiers(rows, easy_ratio, medium_ratio)
