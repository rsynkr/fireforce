"""Pydantic request/response contracts for the wildfire research API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.services.model_registry import MODEL_IDS

SplitName = Literal["train", "eval", "test"]
ModelName = Literal["M1", "M2", "M3"]


class SampleSummary(BaseModel):
    sample_id: str
    split: SplitName
    shard: str
    record_index: int
    target_valid_pixels: int
    target_fire_pixels: int
    target_unknown_pixels: int
    all_unknown: bool


class SampleDetail(SampleSummary):
    available_features: list[str]
    models: list[ModelName]
    previous_fire_mask: list[list[int]]
    target_mask: list[list[int]]
    target_valid_mask: list[list[bool]]


class SampleListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[SampleSummary]


class ModelHealth(BaseModel):
    status: Literal["ready", "unavailable"]
    checkpoint: str
    in_channels: int
    threshold: float
    sha256: str | None = None
    error: str | None = None


class DataHealth(BaseModel):
    ready: bool
    manifest_version: int | None = None
    fingerprint: str | None = None
    shard_count: int | None = None
    sample_count: int | None = None
    split_counts: dict[str, int] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    device: str
    cuda_available: bool
    threshold_provenance: str
    data: DataHealth
    models: dict[str, ModelHealth]


class PredictionRequest(BaseModel):
    sample_id: str = Field(min_length=1, max_length=64)
    models: list[ModelName] = Field(default_factory=lambda: list(MODEL_IDS))

    @field_validator("models")
    @classmethod
    def models_must_be_unique_and_non_empty(cls, value: list[ModelName]) -> list[ModelName]:
        if not value:
            raise ValueError("models must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("models must not contain duplicates")
        return value


class CheckpointInfo(BaseModel):
    file: str
    sha256: str | None = None
    in_channels: int


class Metrics(BaseModel):
    available: bool
    reason: str | None = None
    valid_pixels: int
    true_positive: int | None = None
    false_positive: int | None = None
    false_negative: int | None = None
    true_negative: int | None = None
    precision: float | None = None
    recall: float | None = None
    dice: float | None = None
    iou: float | None = None


class ModelPrediction(BaseModel):
    model_id: ModelName
    status: Literal["ok", "unavailable", "error"]
    threshold: float
    threshold_provenance: str
    checkpoint: CheckpointInfo
    inference_ms: float | None = None
    probabilities: list[list[float]] | None = None
    mask: list[list[bool]] | None = None
    predicted_fire_pixels: int | None = None
    metrics: Metrics | None = None
    error: str | None = None


class GroundTruth(BaseModel):
    available: bool
    target_mask: list[list[int]]
    valid_mask: list[list[bool]]
    valid_pixels: int
    fire_pixels: int
    unknown_pixels: int


class PredictionResponse(BaseModel):
    sample_id: str
    split: SplitName
    shard: str
    record_index: int
    status: Literal["ok", "partial", "failed"]
    device: str
    duration_ms: float
    available_features: list[str]
    ground_truth: GroundTruth
    models: list[ModelPrediction]
