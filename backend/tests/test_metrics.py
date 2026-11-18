import numpy as np
import pytest

from app.services.metrics import masked_binary_metrics


def build_mask(indices, shape=(64, 64)):
    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    flat[list(indices)] = True
    return flat.reshape(shape)


def test_hand_checked_counts_and_scores():
    predictions = build_mask([0, 1, 2, 3])
    target = build_mask([0, 1, 10])
    valid = np.ones((64, 64), dtype=bool)

    metrics = masked_binary_metrics(predictions, target, valid)

    assert metrics["available"] is True
    assert metrics["valid_pixels"] == 4096
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 2
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 4091
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["dice"] == pytest.approx(4 / 7)
    assert metrics["iou"] == pytest.approx(2 / 5)


def test_unknown_pixels_are_excluded():
    predictions = build_mask([0, 1])
    target = build_mask([0])
    valid = np.ones((64, 64), dtype=bool)
    # Pixel 1 is predicted fire and target non-fire, but it is unknown here too.
    valid.reshape(-1)[1] = False

    metrics = masked_binary_metrics(predictions, target, valid)

    assert metrics["valid_pixels"] == 4095
    assert metrics["false_positive"] == 0
    assert metrics["true_positive"] == 1
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)


def test_all_unknown_target_is_unavailable_not_zero():
    predictions = build_mask([0, 1, 2])
    target = np.zeros((64, 64), dtype=bool)
    valid = np.zeros((64, 64), dtype=bool)

    metrics = masked_binary_metrics(predictions, target, valid)

    assert metrics == {
        "available": False,
        "reason": "no_valid_target_pixels",
        "valid_pixels": 0,
    }
    assert "dice" not in metrics
    assert "precision" not in metrics


def test_no_predictions_matches_notebook_zero_convention():
    predictions = np.zeros((64, 64), dtype=bool)
    target = build_mask([0, 1])
    valid = np.ones((64, 64), dtype=bool)

    metrics = masked_binary_metrics(predictions, target, valid)

    assert metrics["false_negative"] == 2
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["dice"] == 0.0
    assert metrics["iou"] == 0.0


def test_shape_validation():
    with pytest.raises(ValueError):
        masked_binary_metrics(
            np.zeros((10, 10), dtype=bool),
            np.zeros((64, 64), dtype=bool),
            np.ones((64, 64), dtype=bool),
        )
