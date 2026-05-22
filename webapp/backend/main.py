"""FastAPI app for the PCGNN web tool: generate / compare / classify."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from generators import (
    classify_rows,
    generate_batch,
    level_metrics,
    level_to_grid,
)
from models import (
    ClassifiedMap,
    ClassifyRequest,
    ClassifyResponse,
    CompareRequest,
    CompareResponse,
    GenerateRequest,
    GenerateResponse,
    MapPayload,
)

app = FastAPI(title="PCGNN Web Tool", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUMMARY_KEYS = (
    "wall_ratio",
    "interior_wall_density",
    "reachable_ratio",
    "shortest_path_length",
    "path_norm",
    "dead_end_ratio",
    "branching_ratio",
    "leniency",
    "astar_difficulty",
    "difficulty_score",
)


def _summarize(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics_list:
        return {"count": 0, "solvability": 0.0}
    summary: dict[str, Any] = {
        "count": len(metrics_list),
        "solvability": float(np.mean([1.0 if m["solvable"] else 0.0 for m in metrics_list])),
    }
    for key in SUMMARY_KEYS:
        values = [m[key] for m in metrics_list if isinstance(m.get(key), (int, float))]
        summary[key] = float(np.mean(values)) if values else 0.0
    return summary


def _build_maps(model: str, count: int, seed: int, perturb: bool) -> list[MapPayload]:
    levels = generate_batch(model, count, seed, perturb=perturb)
    out = []
    for index, level in enumerate(levels):
        metrics = level_metrics(level)
        out.append(MapPayload(index=index, grid=level_to_grid(level), metrics=metrics))
    return out


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def models_meta() -> dict[str, list[dict[str, str]]]:
    return {
        "models": [
            {
                "id": "baseline",
                "label": "Baseline (neat_winner_seed0.pkl)",
                "description": "PCGNN gốc, FeedForwardNetwork, border = sentinel -1.",
            },
            {
                "id": "improved",
                "label": "Improved (inctyseed0.pkl)",
                "description": "Bản cải tiến improve-v2, RecurrentNetwork, border = random 0/1.",
            },
        ]
    }


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        maps = _build_maps(req.model, req.count, req.seed, req.perturb)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    summary = _summarize([m.metrics for m in maps])
    return GenerateResponse(
        model=req.model,
        count=req.count,
        seed=req.seed,
        maps=maps,
        summary=summary,
    )


@app.post("/api/compare", response_model=CompareResponse)
def compare(req: CompareRequest) -> CompareResponse:
    try:
        baseline_maps = _build_maps("baseline", req.count, req.seed, req.perturb)
        improved_maps = _build_maps("improved", req.count, req.seed, req.perturb)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return CompareResponse(
        count=req.count,
        seed=req.seed,
        baseline=baseline_maps,
        improved=improved_maps,
        summary={
            "baseline": _summarize([m.metrics for m in baseline_maps]),
            "improved": _summarize([m.metrics for m in improved_maps]),
        },
    )


@app.post("/api/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    try:
        base = _build_maps(req.model, req.count, req.seed, req.perturb)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    rows: list[dict[str, Any]] = []
    for payload in base:
        row = dict(payload.metrics)
        row["_index"] = payload.index
        row["score_tier"] = _score_tier(float(row["difficulty_score"]))
        row["range_tier"] = _range_tier(row)
        row["percentile_tier"] = ""
        rows.append(row)

    classify_rows(rows, req.easy_ratio, req.medium_ratio)
    rows.sort(key=lambda r: r["_index"])

    classified = []
    for payload, row in zip(base, rows):
        classified.append(
            ClassifiedMap(
                index=payload.index,
                grid=payload.grid,
                metrics=payload.metrics,
                score_tier=row["score_tier"],
                range_tier=row["range_tier"],
                percentile_tier=row["percentile_tier"],
            )
        )

    distribution = {
        "score_tier": _count_by(classified, "score_tier", ("easy", "medium", "hard")),
        "range_tier": _count_by(classified, "range_tier", ("easy", "medium", "hard", "unclassified")),
        "percentile_tier": _count_by(classified, "percentile_tier", ("easy", "medium", "hard")),
    }

    return ClassifyResponse(
        model=req.model,
        count=req.count,
        seed=req.seed,
        easy_ratio=req.easy_ratio,
        medium_ratio=req.medium_ratio,
        maps=classified,
        distribution=distribution,
    )


def _score_tier(score: float) -> str:
    from pcgnn_genmap_metrics import score_tier  # local import; avoid top-level path noise

    return score_tier(score)


def _range_tier(row: dict[str, Any]) -> str:
    from pcgnn_genmap_metrics import range_tier

    return range_tier(row)


def _count_by(maps: list[ClassifiedMap], key: str, buckets: tuple[str, ...]) -> dict[str, int]:
    counts = {b: 0 for b in buckets}
    for m in maps:
        tier = getattr(m, key)
        if tier in counts:
            counts[tier] += 1
    return counts
