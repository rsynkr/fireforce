"""HTTP routes for the wildfire research prototype."""

from __future__ import annotations

import json
import os
from typing import Any

import torch
from fastapi import APIRouter, HTTPException, Query, Request

from app.schemas import (
    DataHealth,
    HealthResponse,
    ModelHealth,
    PredictionRequest,
    PredictionResponse,
    SampleDetail,
    SampleListResponse,
    SampleSummary,
    SplitName,
)
from app.services.model_registry import MODEL_IDS, THRESHOLD_PROVENANCE
from app.services.preprocessing import RAW_FEATURES, build_target
from app.services.sample_catalog import (
    ArtifactMissingError,
    InvalidSampleIdError,
    ManifestIntegrityError,
    SampleEntry,
    SampleNotFoundError,
    filter_sample_entries,
)
from app.state import AppState

router = APIRouter()


def _state(request: Request) -> AppState:
    return request.app.state.wildfire


def _summary(entry: SampleEntry) -> dict[str, Any]:
    unknown = 4096 - entry.target_valid_pixels
    return {
        "sample_id": entry.sample_id,
        "split": entry.split,
        "shard": entry.shard,
        "record_index": entry.record_index,
        "target_valid_pixels": entry.target_valid_pixels,
        "target_fire_pixels": entry.target_fire_pixels,
        "target_unknown_pixels": unknown,
        "all_unknown": entry.target_valid_pixels == 0,
    }


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> dict[str, Any]:
    state = _state(request)
    registry = state.registry
    statuses = registry.statuses()

    if state.catalog is None:
        data = DataHealth(ready=False, error=state.catalog_error)
        overall = "unavailable"
    else:
        data = DataHealth(
            ready=True,
            manifest_version=state.catalog.manifest["manifest_version"],
            fingerprint=state.catalog.fingerprint,
            shard_count=state.catalog.shard_count,
            sample_count=state.catalog.entry_count,
            split_counts=state.catalog.split_counts(),
        )
        ready = sum(1 for status in statuses.values() if status.ready)
        if ready == len(statuses):
            overall = "ok"
        elif ready > 0:
            overall = "degraded"
        else:
            overall = "unavailable"

    return {
        "status": overall,
        "device": str(registry.device),
        "cuda_available": torch.cuda.is_available(),
        "threshold_provenance": THRESHOLD_PROVENANCE,
        "data": data,
        "models": {
            model_id: ModelHealth(**statuses[model_id].to_dict())
            for model_id in MODEL_IDS
        },
    }


@router.get("/samples", response_model=SampleListResponse)
def list_samples(
    request: Request,
    split: SplitName | None = None,
    search: str | None = Query(default=None, max_length=64),
    min_fire_pixels: int | None = Query(default=None, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    state = _state(request)
    if state.catalog is None:
        raise HTTPException(
            status_code=503,
            detail=f"sample catalog unavailable: {state.catalog_error}",
        )

    entries = filter_sample_entries(
        state.catalog.list_entries(),
        split=split,
        search=search,
        min_fire_pixels=min_fire_pixels,
    )

    page = entries[offset : offset + limit]
    return {
        "total": len(entries),
        "limit": limit,
        "offset": offset,
        "items": [_summary(entry) for entry in page],
    }


@router.get("/samples/{sample_id}", response_model=SampleDetail)
def sample_detail(request: Request, sample_id: str) -> dict[str, Any]:
    state = _state(request)
    if state.catalog is None:
        raise HTTPException(
            status_code=503,
            detail=f"sample catalog unavailable: {state.catalog_error}",
        )

    try:
        entry = state.catalog.get_entry(sample_id)
    except InvalidSampleIdError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SampleNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        raw = state.catalog.load_features(sample_id)
    except (ArtifactMissingError, ManifestIntegrityError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    target = build_target(raw["FireMask"])
    previous_fire = raw["PrevFireMask"].reshape(64, 64).astype(int)

    return {
        **_summary(entry),
        "available_features": list(RAW_FEATURES),
        "models": list(MODEL_IDS),
        "previous_fire_mask": previous_fire.tolist(),
        "target_mask": target.fire_mask.astype(int).tolist(),
        "target_valid_mask": target.valid_mask.tolist(),
    }


@router.get("/evaluation/summary")
def evaluation_summary(request: Request) -> dict[str, Any]:
    """Serve the generated deterministic evaluation summary."""
    state = _state(request)
    path = state.settings.evaluation_summary_path
    if path is None or not os.path.isfile(path):
        raise HTTPException(
            status_code=503,
            detail=(
                "Evaluation summary not found. Run "
                "`python backend/tools/run_evaluation.py` and retry."
            ),
        )
    try:
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=503, detail=f"Evaluation summary could not be read: {exc}"
        ) from exc

    required = {"evaluation_version", "split", "models", "rankings", "summary", "counts"}
    if not isinstance(summary, dict) or not required.issubset(summary):
        raise HTTPException(
            status_code=503, detail="Evaluation summary is incomplete."
        )
    return summary


@router.post("/predictions", response_model=PredictionResponse)
def create_predictions(request: Request, payload: PredictionRequest) -> dict[str, Any]:
    state = _state(request)
    if state.catalog is None or state.inference is None:
        raise HTTPException(
            status_code=503,
            detail=f"sample catalog unavailable: {state.catalog_error}",
        )

    try:
        result = state.inference.predict(payload.sample_id, payload.models)
    except InvalidSampleIdError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SampleNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ArtifactMissingError, ManifestIntegrityError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=500, detail=f"sample could not be preprocessed: {exc}"
        ) from exc

    if result["status"] == "failed":
        raise HTTPException(
            status_code=503,
            detail={
                "message": "no requested model is available",
                "models": result["models"],
            },
        )
    return result
