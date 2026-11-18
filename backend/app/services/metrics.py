"""Masked binary metrics over valid target pixels only.

Contract: unknown target pixels (raw FireMask == -1) are excluded from every
count. If there are no valid pixels, metrics are unavailable; this module never
substitutes zeros for a missing target.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SAMPLE_SHAPE = (64, 64)


def masked_binary_metrics(
    predictions: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    """Compute TP/FP/FN/TN and scores over ``valid`` pixels.

    ``predictions`` is a binary mask, ``target`` uses {0, 1}, and ``valid`` marks
    known target pixels. Denominator conventions mirror the notebook's
    ``evaluate_global`` (max(denominator, 1)) so per-sample values stay
    comparable with Phase 0 results.
    """
    pred = np.asarray(predictions, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    mask = np.asarray(valid, dtype=bool)

    for name, array in (("predictions", pred), ("target", truth), ("valid", mask)):
        if array.shape != SAMPLE_SHAPE:
            raise ValueError(f"{name} must have shape {SAMPLE_SHAPE}, got {array.shape}")

    valid_pixels = int(mask.sum())
    if valid_pixels == 0:
        return {
            "available": False,
            "reason": "no_valid_target_pixels",
            "valid_pixels": 0,
        }

    y_true = truth[mask]
    y_pred = pred[mask]

    true_positive = int((y_pred & y_true).sum())
    false_positive = int((y_pred & ~y_true).sum())
    false_negative = int((~y_pred & y_true).sum())
    true_negative = int((~y_pred & ~y_true).sum())

    return {
        "available": True,
        "reason": None,
        "valid_pixels": valid_pixels,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": true_positive / max(true_positive + false_positive, 1),
        "recall": true_positive / max(true_positive + false_negative, 1),
        "dice": 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1),
        "iou": true_positive / max(true_positive + false_positive + false_negative, 1),
    }
