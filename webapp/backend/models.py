from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ModelName = Literal["baseline", "improved"]


class GenerateRequest(BaseModel):
    model: ModelName = "improved"
    count: int = Field(8, ge=1, le=200)
    seed: int = 0
    perturb: bool = True
    width: int = Field(14, ge=5, le=64)
    height: int = Field(14, ge=5, le=64)


class MapPayload(BaseModel):
    index: int
    grid: list[list[int]]
    metrics: dict[str, Any]


class GenerateResponse(BaseModel):
    model: ModelName
    count: int
    seed: int
    height: int
    width: int
    maps: list[MapPayload]
    summary: dict[str, Any]


class CompareRequest(BaseModel):
    count: int = Field(8, ge=1, le=100)
    seed: int = 0
    perturb: bool = True
    width: int = Field(14, ge=5, le=64)
    height: int = Field(14, ge=5, le=64)


class CompareResponse(BaseModel):
    count: int
    seed: int
    baseline: list[MapPayload]
    improved: list[MapPayload]
    summary: dict[str, dict[str, Any]]


class ClassifyRequest(BaseModel):
    model: ModelName = "improved"
    count: int = Field(60, ge=10, le=1000)
    seed: int = 0
    perturb: bool = True
    width: int = Field(14, ge=5, le=64)
    height: int = Field(14, ge=5, le=64)
    easy_ratio: float = Field(0.05, ge=0.0, le=1.0)
    medium_ratio: float = Field(0.05, ge=0.0, le=1.0)


class ClassifiedMap(MapPayload):
    score_tier: str
    range_tier: str
    percentile_tier: str


class ClassifyResponse(BaseModel):
    model: ModelName
    count: int
    seed: int
    easy_ratio: float
    medium_ratio: float
    maps: list[ClassifiedMap]
    distribution: dict[str, dict[str, int]]
