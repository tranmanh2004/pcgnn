from __future__ import annotations

import math
import pickle
import random
from pathlib import Path

import neat
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pcgnn_genmap_metrics as improved


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "rendered_map_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAPS_PER_MODEL = 12
MAP_COLUMNS = 4
TILE_SIZE = 16
MAP_GAP = 18
INNER_PAD = 18
HEADER_H = 72
LABEL_H = 26
FOOTER_H = 54

BG = "#ffffff"
PANEL_BG = "#f8fafc"
PANEL_BORDER = "#d6dce5"
TEXT = "#1f2937"
SUBTEXT = "#475569"
WALL = "#1f2937"
FLOOR = "#f8fafc"
PLAYER = "#22c55e"
ENEMY = "#ef4444"
BASELINE_ACCENT = "#6366f1"
IMPROVED_ACCENT = "#16a34a"

BASELINE_WALL = 0
BASELINE_FLOOR = 1
BASELINE_PLAYER = 2
BASELINE_ENEMY = 3
BASELINE_CONTEXT_SIZE = 1
BASELINE_NUM_RANDOM_INPUTS = 4
BASELINE_NUM_INPUTS = 8 + BASELINE_NUM_RANDOM_INPUTS
BASELINE_NUM_OUTPUTS = 1
BASELINE_PERTURB_SIZE = 0.1565


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = load_font(28, bold=True)
FONT_SUBTITLE = load_font(17)
FONT_LABEL = load_font(18, bold=True)
FONT_NOTE = load_font(16)


def load_baseline_net(model_path: Path):
    config_path = ROOT / "config-pcgnn.txt"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    with model_path.open("rb") as handle:
        genome = pickle.load(handle)
    ff_net = neat.nn.FeedForwardNetwork.create(genome, config)
    probe_level = generate_baseline_level(ff_net)
    if int(np.sum(probe_level == BASELINE_WALL)) < probe_level.size:
        return ff_net
    rec_net = neat.nn.RecurrentNetwork.create(genome, config)
    return rec_net


def generate_baseline_level(net, map_h: int = 14, map_w: int = 14, perturb: bool = True) -> np.ndarray:
    if hasattr(net, "reset"):
        net.reset()
    half = BASELINE_CONTEXT_SIZE
    padded = np.full((map_h + 2 * half, map_w + 2 * half), -1.0, dtype=float)
    noise = [random.gauss(0, 1) for _ in range(BASELINE_NUM_RANDOM_INPUTS)]

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
                inputs = [x + random.gauss(0, BASELINE_PERTURB_SIZE) for x in inputs]

            out = net.activate(inputs)[0]
            padded[row, col] = 1.0 if out > 0.5 else 0.0

    level = padded[half:half + map_h, half:half + map_w].astype(int)
    if level[0, 0] == BASELINE_FLOOR:
        level[0, 0] = BASELINE_PLAYER
    if level[map_h - 1, map_w - 1] == BASELINE_FLOOR:
        level[map_h - 1, map_w - 1] = BASELINE_ENEMY
    return level


def load_improved_net(model_path: Path):
    config_path = ROOT / "_render_improved_config.txt"
    improved.write_improve_v2_config(config_path)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    with model_path.open("rb") as handle:
        genome = pickle.load(handle)
    return neat.nn.RecurrentNetwork.create(genome, config)


def sample_baseline_maps(model_path: Path, count: int) -> list[np.ndarray]:
    random.seed(11)
    np.random.seed(11)
    net = load_baseline_net(model_path)
    selected: list[np.ndarray] = []
    fallback: list[np.ndarray] = []
    attempts = count * 40
    for _ in range(attempts):
        level = generate_baseline_level(net)
        fallback.append(level)
        wall_count = int(np.sum(level == BASELINE_WALL))
        if 8 <= wall_count <= (level.size - 8):
            selected.append(level)
            if len(selected) >= count:
                return selected
    if len(selected) < count:
        selected.extend(fallback[: count - len(selected)])
    return selected[:count]


def sample_improved_maps(model_path: Path, count: int) -> list[np.ndarray]:
    random.seed(22)
    np.random.seed(22)
    net = load_improved_net(model_path)
    maps = []
    for _ in range(count):
        level = improved.generate_level(net, map_h=14, map_w=14, perturb=True)
        maps.append(level)
    return maps


def tile_color(value: int, kind: str) -> str:
    if kind == "baseline":
        if value == BASELINE_WALL:
            return WALL
        if value == BASELINE_PLAYER:
            return PLAYER
        if value == BASELINE_ENEMY:
            return ENEMY
        return FLOOR
    if value == improved.WALL:
        return WALL
    if value == improved.PLAYER:
        return PLAYER
    if value == improved.ENEMY:
        return ENEMY
    return FLOOR


def draw_single_map(draw: ImageDraw.ImageDraw, level: np.ndarray, x0: int, y0: int, kind: str) -> None:
    rows, cols = level.shape
    for r in range(rows):
        for c in range(cols):
            x1 = x0 + c * TILE_SIZE
            y1 = y0 + r * TILE_SIZE
            x2 = x1 + TILE_SIZE
            y2 = y1 + TILE_SIZE
            draw.rectangle([x1, y1, x2, y2], fill=tile_color(int(level[r, c]), kind), outline="#dbe2ea")


def render_grid(levels: list[np.ndarray], title: str, subtitle: str, kind: str, accent: str, out_path: Path) -> None:
    rows = math.ceil(len(levels) / MAP_COLUMNS)
    map_h, map_w = levels[0].shape
    map_px_w = map_w * TILE_SIZE
    map_px_h = map_h * TILE_SIZE
    width = INNER_PAD * 2 + MAP_COLUMNS * map_px_w + (MAP_COLUMNS - 1) * MAP_GAP
    height = HEADER_H + INNER_PAD + rows * (map_px_h + LABEL_H) + (rows - 1) * MAP_GAP + FOOTER_H

    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle([10, 10, width - 10, height - 10], radius=18, fill=PANEL_BG, outline=PANEL_BORDER, width=2)
    draw.rectangle([INNER_PAD, 26, INNER_PAD + 8, 62], fill=accent)
    draw.text((INNER_PAD + 22, 22), title, font=FONT_TITLE, fill=TEXT)
    draw.text((INNER_PAD + 22, 52), subtitle, font=FONT_SUBTITLE, fill=SUBTEXT)

    start_y = HEADER_H + 6
    for index, level in enumerate(levels):
        row = index // MAP_COLUMNS
        col = index % MAP_COLUMNS
        x = INNER_PAD + col * (map_px_w + MAP_GAP)
        y = start_y + row * (map_px_h + LABEL_H + MAP_GAP)
        draw_single_map(draw, level, x, y, kind)
        label = f"Sample {index + 1}"
        bbox = draw.textbbox((0, 0), label, font=FONT_LABEL)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (map_px_w - text_w) / 2, y + map_px_h + 4), label, font=FONT_LABEL, fill=TEXT)

    note = "Mỗi ô vuông biểu diễn một tile. Xanh lá là điểm bắt đầu, đỏ là đích/điểm địch."
    if kind == "improved":
        note = "Map cải tiến giữ cùng kích thước 14x14, có marker xanh lá cho player và đỏ cho enemy."
    bbox = draw.textbbox((0, 0), note, font=FONT_NOTE)
    note_w = bbox[2] - bbox[0]
    draw.text(((width - note_w) / 2, height - 36), note, font=FONT_NOTE, fill=SUBTEXT)

    image.save(out_path)
    print(f"[saved] {out_path}")


def render_side_by_side(baseline_levels: list[np.ndarray], improved_levels: list[np.ndarray], out_path: Path) -> None:
    subset = min(4, len(baseline_levels), len(improved_levels))
    left = baseline_levels[:subset]
    right = improved_levels[:subset]

    map_h, map_w = left[0].shape
    map_px_w = map_w * TILE_SIZE
    map_px_h = map_h * TILE_SIZE
    section_w = INNER_PAD * 2 + 2 * map_px_w + MAP_GAP
    width = section_w * 2 + 34
    height = HEADER_H + INNER_PAD + 2 * (map_px_h + LABEL_H) + MAP_GAP + FOOTER_H

    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([10, 10, width - 10, height - 10], radius=18, fill=PANEL_BG, outline=PANEL_BORDER, width=2)

    draw.text((INNER_PAD, 20), "So sánh mẫu map baseline và map cải tiến", font=FONT_TITLE, fill=TEXT)
    draw.text((INNER_PAD, 52), "Mỗi cột hiển thị cùng số lượng mẫu để tiện đối chiếu trực quan.", font=FONT_SUBTITLE, fill=SUBTEXT)

    def draw_section(origin_x: int, title: str, levels: list[np.ndarray], kind: str, accent: str) -> None:
        draw.rectangle([origin_x, 86, origin_x + 8, 118], fill=accent)
        draw.text((origin_x + 18, 84), title, font=FONT_LABEL, fill=TEXT)
        for idx, level in enumerate(levels):
            row = idx // 2
            col = idx % 2
            x = origin_x + col * (map_px_w + MAP_GAP)
            y = 126 + row * (map_px_h + LABEL_H + MAP_GAP)
            draw_single_map(draw, level, x, y, kind)
            label = f"M{idx + 1}"
            bbox = draw.textbbox((0, 0), label, font=FONT_LABEL)
            text_w = bbox[2] - bbox[0]
            draw.text((x + (map_px_w - text_w) / 2, y + map_px_h + 4), label, font=FONT_LABEL, fill=TEXT)

        block_x = origin_x + 2 * map_px_w + MAP_GAP + 10
        draw.line([block_x, 90, block_x, height - 50], fill=PANEL_BORDER, width=2)

    draw_section(INNER_PAD, "Baseline", left, "baseline", BASELINE_ACCENT)
    draw_section(INNER_PAD + section_w + 34, "Map cải tiến", right, "improved", IMPROVED_ACCENT)

    note = "Baseline dùng mạng feed-forward của bài toán maze; bản cải tiến dùng checkpoint recurrent cho generator improve-v2."
    draw.text((INNER_PAD, height - 34), note, font=FONT_NOTE, fill=SUBTEXT)
    image.save(out_path)
    print(f"[saved] {out_path}")


def main() -> None:
    baseline_path = ROOT / "checkpoints" / "baseline" / "neat_winner_seed0.pkl"
    improved_path = ROOT / "checkpoints" / "improved" / "inctyseed0.pkl"

    baseline_levels = sample_baseline_maps(baseline_path, MAPS_PER_MODEL)
    improved_levels = sample_improved_maps(improved_path, MAPS_PER_MODEL)

    render_grid(
        baseline_levels,
        title="Map baseline",
        subtitle="Sinh từ checkpoint Baseline.pkl",
        kind="baseline",
        accent=BASELINE_ACCENT,
        out_path=OUTPUT_DIR / "map_baseline_samples.png",
    )
    render_grid(
        improved_levels,
        title="Map cải tiến",
        subtitle="Sinh từ checkpoint inctyseed0.pkl",
        kind="improved",
        accent=IMPROVED_ACCENT,
        out_path=OUTPUT_DIR / "map_improved_samples.png",
    )
    render_side_by_side(
        baseline_levels,
        improved_levels,
        out_path=OUTPUT_DIR / "map_baseline_vs_improved.png",
    )


if __name__ == "__main__":
    main()
