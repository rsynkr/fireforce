"""Sample-level inference orchestration: one record read, shared preprocessing."""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

from app.services.metrics import masked_binary_metrics
from app.services.model_registry import (
    MODEL_IDS,
    THRESHOLD_PROVENANCE,
    InferenceError,
    ModelRegistry,
)
from app.services.preprocessing import (
    RAW_FEATURES,
    build_model_input,
    build_target,
    preprocess_features,
)
from app.services.sample_catalog import SampleCatalog


class InferenceService:
    def __init__(self, catalog: SampleCatalog, registry: ModelRegistry):
        self.catalog = catalog
        self.registry = registry

    def predict(self, sample_id: str, model_ids: Iterable[str]) -> dict[str, Any]:
        """Run the requested models on one sample.

        The TFRecord is read and preprocessed once; each model only assembles
        its own channel group. Raises catalog errors unchanged; per-model
        unavailability is reported in the response instead of aborting.
        """
        requested = tuple(model_ids)
        for model_id in requested:
            self.registry.spec(model_id)

        started = time.perf_counter()
        entry = self.catalog.get_entry(sample_id)
        raw = self.catalog.load_features(sample_id)
        features = preprocess_features(raw)
        target = build_target(features["FireMask"])

        target_mask = target.fire_mask.astype(np.uint8)
        valid_mask = target.valid_mask
        valid_pixels = int(valid_mask.sum())
        fire_pixels = int(target_mask.sum())

        models: list[dict[str, Any]] = []
        for model_id in requested:
            spec = self.registry.spec(model_id)
            status = self.registry.status(model_id)
            base = {
                "model_id": model_id,
                "threshold": spec.threshold,
                "threshold_provenance": THRESHOLD_PROVENANCE,
                "checkpoint": {
                    "file": spec.checkpoint_file,
                    "sha256": status.sha256,
                    "in_channels": spec.in_channels,
                },
            }

            if not status.ready:
                models.append(
                    {
                        **base,
                        "status": "unavailable",
                        "inference_ms": None,
                        "probabilities": None,
                        "mask": None,
                        "predicted_fire_pixels": None,
                        "metrics": None,
                        "error": status.error,
                    }
                )
                continue

            try:
                model_input = build_model_input(features, model_id)
                model_started = time.perf_counter()
                probabilities = self.registry.predict(model_id, model_input)
                inference_ms = (time.perf_counter() - model_started) * 1000.0
            except (InferenceError, ValueError) as exc:
                models.append(
                    {
                        **base,
                        "status": "error",
                        "inference_ms": None,
                        "probabilities": None,
                        "mask": None,
                        "predicted_fire_pixels": None,
                        "metrics": None,
                        "error": str(exc),
                    }
                )
                continue

            mask = probabilities >= spec.threshold
            metrics = masked_binary_metrics(mask, target_mask, valid_mask)
            models.append(
                {
                    **base,
                    "status": "ok",
                    "inference_ms": inference_ms,
                    "probabilities": probabilities.tolist(),
                    "mask": mask.tolist(),
                    "predicted_fire_pixels": int(mask.sum()),
                    "metrics": metrics,
                    "error": None,
                }
            )

        succeeded = sum(1 for model in models if model["status"] == "ok")
        overall = (
            "ok"
            if succeeded == len(models)
            else "partial" if succeeded > 0 else "failed"
        )

        return {
            "sample_id": entry.sample_id,
            "split": entry.split,
            "shard": entry.shard,
            "record_index": entry.record_index,
            "status": overall,
            "device": str(self.registry.device),
            "duration_ms": (time.perf_counter() - started) * 1000.0,
            "available_features": list(RAW_FEATURES),
            "ground_truth": {
                "available": valid_pixels > 0,
                "target_mask": target_mask.tolist(),
                "valid_mask": valid_mask.tolist(),
                "valid_pixels": valid_pixels,
                "fire_pixels": fire_pixels,
                "unknown_pixels": int(valid_mask.size - valid_pixels),
            },
            "models": models,
        }

    @property
    def model_ids(self) -> tuple[str, ...]:
        return MODEL_IDS
