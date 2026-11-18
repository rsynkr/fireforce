"""Deterministic 100-sample comparative evaluation for M1/M2/M3.

Sampling strategy: from the sorted sample IDs of the requested split, rank by
``sha256(f"{seed}:{sample_id}")`` and take the lowest ``count`` IDs, then sort
the selection by sample ID. This is a fixed-seed, version-independent sample and
is written into every output. It is a subset, not a full test-set estimate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import time
from typing import Any, Iterable

import numpy as np

from app.services.metrics import masked_binary_metrics
from app.services.model_registry import (
    MODEL_IDS,
    MODEL_SPECS,
    THRESHOLD_PROVENANCE,
    ModelRegistry,
)
from app.services.preprocessing import (
    MODEL_CHANNELS,
    build_model_input,
    build_target,
    preprocess_features,
)
from app.services.sample_catalog import PIXELS_PER_RECORD, SampleCatalog

EVALUATION_VERSION = "100-test-v1"
DEFAULT_SPLIT = "test"
DEFAULT_COUNT = 100
DEFAULT_SEED = 20261118
SAMPLING_STRATEGY = (
    "candidate IDs sorted; rank by sha256(f'{seed}:{sample_id}') ascending; "
    "take the lowest count; sort selection by sample_id"
)


class EvaluationError(ValueError):
    """The requested evaluation cannot be performed as specified."""


def select_evaluation_samples(
    catalog: SampleCatalog,
    split: str = DEFAULT_SPLIT,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    """Deterministically select sample IDs from one split."""
    if split not in {"train", "eval", "test"}:
        raise EvaluationError(f"unsupported split {split!r}")
    if count <= 0:
        raise EvaluationError("count must be positive")

    candidates = sorted(
        entry.sample_id for entry in catalog.list_entries() if entry.split == split
    )
    if len(candidates) < count:
        raise EvaluationError(
            f"split {split!r} has {len(candidates)} samples; cannot select {count}"
        )

    ranked = sorted(
        candidates,
        key=lambda sample_id: hashlib.sha256(
            f"{seed}:{sample_id}".encode("utf-8")
        ).hexdigest(),
    )
    return sorted(ranked[:count])


def _metric_value(
    metrics: dict[str, Any], key: str
) -> float | None:
    return metrics.get(key) if metrics.get("available") else None


def evaluate_samples(
    catalog: SampleCatalog,
    registry: ModelRegistry,
    sample_ids: Iterable[str],
    version: str = EVALUATION_VERSION,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Evaluate every model on exactly the same samples, one record read each."""
    selected = list(sample_ids)
    if len(set(selected)) != len(selected):
        raise EvaluationError("sample selection contains duplicates")

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []

    for position, sample_id in enumerate(selected, start=1):
        entry = catalog.get_entry(sample_id)
        raw = catalog.load_features(sample_id)
        features = preprocess_features(raw)
        target = build_target(features["FireMask"])
        truth_u8 = target.fire_mask.astype(np.uint8)
        valid = target.valid_mask
        actual_fire = int(truth_u8.sum())
        valid_pixels = int(valid.sum())
        unknown_pixels = int(PIXELS_PER_RECORD - valid_pixels)

        for model_id in MODEL_IDS:
            spec = MODEL_SPECS[model_id]
            model_input = build_model_input(features, model_id)
            inference_started = time.perf_counter()
            probabilities = registry.predict(model_id, model_input)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            mask = probabilities >= spec.threshold
            metrics = masked_binary_metrics(mask, truth_u8, valid)

            row: dict[str, Any] = {
                "sample_id": sample_id,
                "split": entry.split,
                "shard": entry.shard,
                "record_index": entry.record_index,
                "model_id": model_id,
                "threshold": spec.threshold,
                "actual_fire_pixels": actual_fire,
                "valid_target_pixels": valid_pixels,
                "unknown_target_pixels": unknown_pixels,
                "predicted_fire_pixels": int(mask.sum()),
                "predicted_fire_on_valid_pixels": int((mask & valid).sum()),
                "predicted_fire_on_unknown_pixels": int((mask & ~valid).sum()),
                "tp": None,
                "fp": None,
                "fn": None,
                "tn": None,
                "precision": None,
                "recall": None,
                "f1_dice": None,
                "iou": None,
                "accuracy": None,
                "inference_ms": inference_ms,
            }
            if metrics["available"]:
                row.update(
                    {
                        "tp": metrics["true_positive"],
                        "fp": metrics["false_positive"],
                        "fn": metrics["false_negative"],
                        "tn": metrics["true_negative"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_dice": metrics["dice"],
                        "iou": metrics["iou"],
                        "accuracy": (
                            (metrics["true_positive"] + metrics["true_negative"])
                            / metrics["valid_pixels"]
                        ),
                    }
                )
            rows.append(row)

        if position % 10 == 0 or position == len(selected):
            print(f"evaluated {position}/{len(selected)} samples", flush=True)

    return aggregate_evaluation(
        selected,
        rows,
        elapsed_seconds=time.perf_counter() - started,
        version=version,
        seed=seed,
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _isclose(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def rank_models(
    models: dict[str, dict[str, Any]],
    key_path: tuple[str, ...],
    direction: str,
) -> dict[str, Any]:
    """Rank models with explicit tie handling and deterministic ordering."""
    values: dict[str, float] = {}
    for model_id in MODEL_IDS:
        node: Any = models[model_id]
        for key in key_path:
            node = node[key]
        values[model_id] = float(node)

    target = max(values.values()) if direction == "max" else min(values.values())
    winners = [
        model_id
        for model_id in MODEL_IDS
        if _isclose(values[model_id], target)
    ]
    ordered = sorted(
        MODEL_IDS,
        key=lambda model_id: (
            -values[model_id] if direction == "max" else values[model_id],
            MODEL_IDS.index(model_id),
        ),
    )
    return {
        "metric": key_path[-1],
        "direction": direction,
        "winners": winners,
        "values": {model_id: values[model_id] for model_id in MODEL_IDS},
        "order": list(ordered),
    }


def _join_models(model_ids: list[str], conjunction: str = "and") -> str:
    if len(model_ids) == 1:
        return model_ids[0]
    if len(model_ids) == 2:
        return f"{model_ids[0]} {conjunction} {model_ids[1]}"
    return f"{', '.join(model_ids[:-1])}, {conjunction} {model_ids[-1]}"


def generate_summary(
    models: dict[str, dict[str, Any]],
    rankings: dict[str, Any],
    selected_count: int,
    valid_sample_count: int,
    all_unknown_count: int,
) -> list[str]:
    """Build the plain-English summary from the computed results."""
    f1 = rankings["best_f1_dice"]
    iou = rankings["best_iou"]
    most_pred = rankings["most_predicted_fire"]
    fewest_pred = rankings["fewest_predicted_fire"]
    most_tp = rankings["most_true_positives"]
    most_fp = rankings["most_false_positives"]
    best_accuracy = rankings["best_accuracy"]

    f1_winner = _join_models(f1["winners"])
    iou_winner = _join_models(iou["winners"])
    f1_value = f1["values"][f1["winners"][0]]
    iou_value = iou["values"][iou["winners"][0]]

    sentences = [
        f"{f1_winner} achieved the strongest overall F1/Dice ({f1_value:.4f}); "
        f"{iou_winner} had the best IoU ({iou_value:.4f}).",
        f"{_join_models(most_pred['winners'])} predicted the most fire pixels "
        f"({most_pred['values'][most_pred['winners'][0]]:,.0f}), while "
        f"{_join_models(fewest_pred['winners'])} predicted the fewest "
        f"({fewest_pred['values'][fewest_pred['winners'][0]]:,.0f}). "
        "Predicting more fire is not evidence of better performance.",
        f"{_join_models(most_tp['winners'])} detected the most actual fire pixels "
        f"(TP={most_tp['values'][most_tp['winners'][0]]:,.0f}); "
        f"{_join_models(most_fp['winners'])} produced the most false alarms "
        f"(FP={most_fp['values'][most_fp['winners'][0]]:,.0f}).",
        f"Accuracy was highest for {_join_models(best_accuracy['winners'])} "
        f"({best_accuracy['values'][best_accuracy['winners'][0]]:.4f}), but class "
        "imbalance makes accuracy misleading for this task.",
        f"These results cover {selected_count} samples from the test split; "
        f"{valid_sample_count} had valid target pixels and {all_unknown_count} "
        "all-unknown samples were excluded from metrics. This is a fixed-seed "
        "subset, not a full test-set evaluation or a statistical claim.",
    ]

    if set(f1["winners"]) != set(iou["winners"]):
        sentences[0] = (
            f"{f1_winner} achieved the strongest F1/Dice ({f1_value:.4f}), "
            f"while {iou_winner} had the best IoU ({iou_value:.4f})."
        )
    return sentences


def build_rankings(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "best_f1_dice": rank_models(models, ("micro", "f1_dice"), "max"),
        "best_iou": rank_models(models, ("micro", "iou"), "max"),
        "best_precision": rank_models(models, ("micro", "precision"), "max"),
        "best_recall": rank_models(models, ("micro", "recall"), "max"),
        "best_accuracy": rank_models(models, ("micro", "accuracy"), "max"),
        "most_predicted_fire": rank_models(models, ("predicted", "total"), "max"),
        "fewest_predicted_fire": rank_models(models, ("predicted", "total"), "min"),
        "most_true_positives": rank_models(models, ("counts", "true_positive"), "max"),
        "most_false_positives": rank_models(models, ("counts", "false_positive"), "max"),
    }


def aggregate_evaluation(
    selected: list[str],
    rows: list[dict[str, Any]],
    elapsed_seconds: float,
    version: str = EVALUATION_VERSION,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compute per-model micro/macro metrics, distributions and rankings."""
    if not rows:
        raise EvaluationError("no evaluation rows")

    sample_sets = {
        model_id: frozenset(
            row["sample_id"] for row in rows if row["model_id"] == model_id
        )
        for model_id in MODEL_IDS
    }
    if len(set(sample_sets.values())) != 1:
        raise EvaluationError("models were not evaluated on identical sample sets")

    models: dict[str, dict[str, Any]] = {}
    valid_sample_ids = sorted(
        {row["sample_id"] for row in rows if row["valid_target_pixels"] > 0}
    )
    all_unknown_ids = sorted(
        {row["sample_id"] for row in rows if row["valid_target_pixels"] == 0}
    )
    total_valid_pixels = sum(
        row["valid_target_pixels"] for row in rows if row["model_id"] == "M1"
    )
    total_actual_fire = sum(
        row["actual_fire_pixels"] for row in rows if row["model_id"] == "M1"
    )

    for model_id in MODEL_IDS:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        scored = [row for row in model_rows if row["valid_target_pixels"] > 0]

        tp = sum(row["tp"] for row in scored)
        fp = sum(row["fp"] for row in scored)
        fn = sum(row["fn"] for row in scored)
        tn = sum(row["tn"] for row in scored)
        valid_pixels = sum(row["valid_target_pixels"] for row in scored)

        micro = {
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1_dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1),
            "accuracy": (tp + tn) / max(valid_pixels, 1),
            "valid_pixels": valid_pixels,
        }
        macro = {
            "precision": _mean([row["precision"] for row in scored]),
            "recall": _mean([row["recall"] for row in scored]),
            "f1_dice": _mean([row["f1_dice"] for row in scored]),
            "iou": _mean([row["iou"] for row in scored]),
            "accuracy": _mean([row["accuracy"] for row in scored]),
            "samples": len(scored),
        }

        predicted = [row["predicted_fire_pixels"] for row in model_rows]
        predicted_on_valid = sum(row["predicted_fire_on_valid_pixels"] for row in model_rows)

        models[model_id] = {
            "threshold": MODEL_SPECS[model_id].threshold,
            "in_channels": len(MODEL_CHANNELS[model_id]),
            "micro": micro,
            "macro": macro,
            "counts": {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
            },
            "predicted": {
                "total": sum(predicted),
                "mean": statistics.mean(predicted),
                "median": statistics.median(predicted),
                "max": max(predicted),
                "on_valid_pixels": predicted_on_valid,
                "on_unknown_pixels": sum(
                    row["predicted_fire_on_unknown_pixels"] for row in model_rows
                ),
                "percent_of_valid_pixels": (
                    100.0 * predicted_on_valid / total_valid_pixels
                    if total_valid_pixels
                    else 0.0
                ),
            },
        }

    rankings = build_rankings(models)

    return {
        "evaluation_version": version,
        "selected_samples": selected,
        "summary": {
            "evaluation_version": version,
            "split": rows[0]["split"],
            "sampling": {
                "strategy": SAMPLING_STRATEGY,
                "seed": seed,
                "count": len(selected),
            },
            "counts": {
                "selected": len(selected),
                "valid_samples": len(valid_sample_ids),
                "all_unknown_samples": len(all_unknown_ids),
                "excluded_sample_ids": all_unknown_ids,
                "valid_pixels": total_valid_pixels,
                "actual_fire_pixels": total_actual_fire,
            },
            "thresholds": {
                model_id: MODEL_SPECS[model_id].threshold for model_id in MODEL_IDS
            },
            "threshold_provenance": THRESHOLD_PROVENANCE,
            "models": models,
            "rankings": rankings,
            "summary": generate_summary(
                models,
                rankings,
                len(selected),
                len(valid_sample_ids),
                len(all_unknown_ids),
            ),
            "elapsed_seconds": elapsed_seconds,
        },
        "rows": rows,
    }


CSV_FIELDS = [
    "sample_id",
    "split",
    "shard",
    "record_index",
    "model_id",
    "threshold",
    "actual_fire_pixels",
    "valid_target_pixels",
    "unknown_target_pixels",
    "predicted_fire_pixels",
    "predicted_fire_on_valid_pixels",
    "predicted_fire_on_unknown_pixels",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1_dice",
    "iou",
    "accuracy",
    "inference_ms",
]


def write_evaluation_outputs(result: dict[str, Any], out_dir: str) -> dict[str, str]:
    """Write versioned JSON/CSV outputs. Full rows and dashboard summary are split."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "samples": os.path.join(out_dir, "samples.json"),
        "results": os.path.join(out_dir, "results.json"),
        "results_csv": os.path.join(out_dir, "results.csv"),
        "summary": os.path.join(out_dir, "summary.json"),
        "summary_csv": os.path.join(out_dir, "summary.csv"),
    }

    with open(paths["samples"], "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "evaluation_version": result["evaluation_version"],
                "split": result["summary"]["split"],
                "sampling": result["summary"]["sampling"],
                "sample_ids": result["selected_samples"],
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    with open(paths["results"], "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    with open(paths["summary"], "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result["summary"], handle, indent=2)
        handle.write("\n")

    with open(paths["results_csv"], "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})

    with open(paths["summary_csv"], "w", encoding="utf-8", newline="") as handle:
        fields = [
            "model_id",
            "threshold",
            "precision_micro",
            "recall_micro",
            "f1_dice_micro",
            "iou_micro",
            "accuracy_micro",
            "precision_macro",
            "recall_macro",
            "f1_dice_macro",
            "iou_macro",
            "accuracy_macro",
            "tp",
            "fp",
            "fn",
            "tn",
            "predicted_fire_total",
            "predicted_fire_mean",
            "predicted_fire_median",
            "predicted_fire_max",
            "percent_valid_predicted_fire",
            "macro_samples",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_id in MODEL_IDS:
            model = result["summary"]["models"][model_id]
            writer.writerow(
                {
                    "model_id": model_id,
                    "threshold": model["threshold"],
                    "precision_micro": model["micro"]["precision"],
                    "recall_micro": model["micro"]["recall"],
                    "f1_dice_micro": model["micro"]["f1_dice"],
                    "iou_micro": model["micro"]["iou"],
                    "accuracy_micro": model["micro"]["accuracy"],
                    "precision_macro": model["macro"]["precision"],
                    "recall_macro": model["macro"]["recall"],
                    "f1_dice_macro": model["macro"]["f1_dice"],
                    "iou_macro": model["macro"]["iou"],
                    "accuracy_macro": model["macro"]["accuracy"],
                    "tp": model["counts"]["true_positive"],
                    "fp": model["counts"]["false_positive"],
                    "fn": model["counts"]["false_negative"],
                    "tn": model["counts"]["true_negative"],
                    "predicted_fire_total": model["predicted"]["total"],
                    "predicted_fire_mean": model["predicted"]["mean"],
                    "predicted_fire_median": model["predicted"]["median"],
                    "predicted_fire_max": model["predicted"]["max"],
                    "percent_valid_predicted_fire": model["predicted"][
                        "percent_of_valid_pixels"
                    ],
                    "macro_samples": model["macro"]["samples"],
                }
            )

    return paths
