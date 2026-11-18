import csv
import json
import os

import pytest

from app.services.evaluation import (
    DEFAULT_SEED,
    EvaluationError,
    aggregate_evaluation,
    build_rankings,
    evaluate_samples,
    generate_summary,
    rank_models,
    select_evaluation_samples,
    write_evaluation_outputs,
)
from app.services.model_registry import ModelRegistry
from app.services.sample_catalog import load_catalog


def make_row(
    sample_id,
    model_id,
    valid,
    actual,
    tp,
    fp,
    fn,
    tn,
    predicted,
):
    scored = valid > 0
    precision = (tp / (tp + fp)) if scored and (tp + fp) else (0.0 if scored else None)
    recall = (tp / (tp + fn)) if scored and (tp + fn) else (0.0 if scored else None)
    f1 = (
        2 * tp / max(2 * tp + fp + fn, 1) if scored else None
    )
    iou = tp / max(tp + fp + fn, 1) if scored else None
    accuracy = (tp + tn) / valid if scored else None
    return {
        "sample_id": sample_id,
        "split": "test",
        "shard": "test_00.tfrecord",
        "record_index": 0,
        "model_id": model_id,
        "threshold": 0.94,
        "actual_fire_pixels": actual,
        "valid_target_pixels": valid,
        "unknown_target_pixels": 4096 - valid,
        "predicted_fire_pixels": predicted,
        "predicted_fire_on_valid_pixels": (tp + fp) if scored else 0,
        "predicted_fire_on_unknown_pixels": predicted - ((tp + fp) if scored else 0),
        "tp": tp if scored else None,
        "fp": fp if scored else None,
        "fn": fn if scored else None,
        "tn": tn if scored else None,
        "precision": precision,
        "recall": recall,
        "f1_dice": f1,
        "iou": iou,
        "accuracy": accuracy,
        "inference_ms": 1.0,
    }


def synthetic_rows():
    specs = {
        "M1": (1, 0, 1, 8, 0, 9, 0, 1, 100),
        "M2": (2, 0, 0, 8, 0, 5, 0, 5, 50),
        "M3": (0, 0, 2, 8, 0, 0, 0, 10, 80),
    }
    rows = []
    for model_id, (atp, afp, afn, atn, btp, bfp, bfn, btn, predicted) in specs.items():
        rows.append(make_row("a", model_id, 10, 2, atp, afp, afn, atn, predicted))
        rows.append(make_row("b", model_id, 10, 0, btp, bfp, bfn, btn, predicted))
        rows.append(make_row("c", model_id, 0, 0, 0, 0, 0, 0, 5))
    return rows


def test_micro_and_macro_are_distinct():
    result = aggregate_evaluation(["a", "b", "c"], synthetic_rows(), 0.1)
    models = result["summary"]["models"]

    assert models["M1"]["counts"] == {
        "true_positive": 1,
        "false_positive": 9,
        "false_negative": 1,
        "true_negative": 9,
    }
    assert models["M1"]["micro"]["precision"] == pytest.approx(0.1)
    assert models["M1"]["micro"]["recall"] == pytest.approx(0.5)
    assert models["M1"]["micro"]["f1_dice"] == pytest.approx(1 / 6)
    assert models["M1"]["micro"]["iou"] == pytest.approx(1 / 11)
    assert models["M1"]["micro"]["accuracy"] == pytest.approx(0.5)
    assert models["M1"]["macro"]["precision"] == pytest.approx(0.5)
    assert models["M1"]["macro"]["recall"] == pytest.approx(0.25)
    assert models["M1"]["macro"]["f1_dice"] == pytest.approx(1 / 3)
    assert models["M1"]["macro"]["iou"] == pytest.approx(0.25)
    assert models["M1"]["macro"]["samples"] == 2


def test_all_unknown_samples_are_excluded_and_reported():
    result = aggregate_evaluation(["a", "b", "c"], synthetic_rows(), 0.1)
    summary = result["summary"]

    assert summary["counts"]["selected"] == 3
    assert summary["counts"]["valid_samples"] == 2
    assert summary["counts"]["all_unknown_samples"] == 1
    assert summary["counts"]["excluded_sample_ids"] == ["c"]
    assert summary["models"]["M1"]["macro"]["samples"] == 2
    assert summary["models"]["M1"]["micro"]["valid_pixels"] == 20

    unknown_rows = [row for row in result["rows"] if row["sample_id"] == "c"]
    assert unknown_rows
    assert all(row["tp"] is None and row["precision"] is None for row in unknown_rows)
    assert all(row["predicted_fire_pixels"] == 5 for row in unknown_rows)


def test_rankings_and_distribution_statistics():
    result = aggregate_evaluation(["a", "b", "c"], synthetic_rows(), 0.1)
    summary = result["summary"]

    assert summary["rankings"]["best_f1_dice"]["winners"] == ["M2"]
    assert summary["rankings"]["best_iou"]["winners"] == ["M2"]
    assert summary["rankings"]["best_precision"]["winners"] == ["M2"]
    assert summary["rankings"]["best_recall"]["winners"] == ["M2"]
    assert summary["rankings"]["best_accuracy"]["winners"] == ["M3"]
    assert summary["rankings"]["most_predicted_fire"]["winners"] == ["M1"]
    assert summary["rankings"]["fewest_predicted_fire"]["winners"] == ["M2"]
    assert summary["rankings"]["most_true_positives"]["winners"] == ["M2"]
    assert summary["rankings"]["most_false_positives"]["winners"] == ["M1"]

    assert summary["models"]["M1"]["predicted"]["total"] == 205
    assert summary["models"]["M1"]["predicted"]["mean"] == pytest.approx(205 / 3)
    assert summary["models"]["M1"]["predicted"]["median"] == 100
    assert summary["models"]["M1"]["predicted"]["max"] == 100


def test_rank_models_reports_ties_deterministically():
    models = {
        "M1": {"micro": {"f1_dice": 0.5}},
        "M2": {"micro": {"f1_dice": 0.5}},
        "M3": {"micro": {"f1_dice": 0.4}},
    }
    ranking = rank_models(models, ("micro", "f1_dice"), "max")
    assert ranking["winners"] == ["M1", "M2"]
    assert ranking["order"] == ["M1", "M2", "M3"]

    models["M1"]["micro"]["f1_dice"] = 0.1
    ranking = rank_models(models, ("micro", "f1_dice"), "max")
    assert ranking["winners"] == ["M2"]


def test_summary_is_generated_from_results():
    result = aggregate_evaluation(["a", "b", "c"], synthetic_rows(), 0.1)
    summary = result["summary"]
    text = " ".join(summary["summary"])

    assert "M2 achieved the strongest overall F1/Dice" in text
    assert "M1 predicted the most fire pixels" in text
    assert "not a full test-set evaluation" in text

    models = json.loads(json.dumps(summary["models"]))
    models["M1"]["micro"]["f1_dice"] = 0.99
    models["M1"]["micro"]["iou"] = 0.99
    rankings = build_rankings(models)
    changed = " ".join(
        generate_summary(
            models,
            rankings,
            summary["counts"]["selected"],
            summary["counts"]["valid_samples"],
            summary["counts"]["all_unknown_samples"],
        )
    )
    assert "M1 achieved the strongest overall F1/Dice" in changed
    assert changed != text


@pytest.fixture(scope="module")
def services(real_settings):
    catalog = load_catalog(
        str(real_settings.manifest_path), str(real_settings.data_root)
    )
    registry = ModelRegistry(str(real_settings.checkpoint_dir))
    return catalog, registry


def test_selection_is_deterministic_and_uses_test_split(services):
    catalog, _ = services
    first = select_evaluation_samples(catalog, count=100)
    second = select_evaluation_samples(catalog, count=100)

    assert first == second
    assert len(first) == len(set(first)) == 100
    assert all(sample_id.startswith("test_") for sample_id in first)
    for sample_id in first:
        assert catalog.get_entry(sample_id).split == "test"

    other = select_evaluation_samples(catalog, count=100, seed=DEFAULT_SEED + 1)
    assert set(other) != set(first)

    with pytest.raises(EvaluationError):
        select_evaluation_samples(catalog, count=10_000)


def test_evaluation_uses_identical_samples_and_is_reproducible(services):
    catalog, registry = services
    sample_ids = select_evaluation_samples(catalog, count=3)

    first = evaluate_samples(catalog, registry, sample_ids)
    second = evaluate_samples(catalog, registry, sample_ids)

    assert len(first["rows"]) == 9
    for model_id in ("M1", "M2", "M3"):
        rows = [row for row in first["rows"] if row["model_id"] == model_id]
        assert [row["sample_id"] for row in rows] == sorted(sample_ids)

    for row in first["rows"]:
        assert row["predicted_fire_pixels"] == (
            row["predicted_fire_on_valid_pixels"]
            + row["predicted_fire_on_unknown_pixels"]
        )
        if row["valid_target_pixels"] > 0:
            assert row["predicted_fire_on_valid_pixels"] == row["tp"] + row["fp"]

    def stable(result):
        summary = json.loads(json.dumps(result["summary"]))
        summary.pop("elapsed_seconds", None)
        rows = []
        for row in result["rows"]:
            row = dict(row)
            row.pop("inference_ms", None)
            rows.append(row)
        return summary, rows

    assert stable(first) == stable(second)


def test_outputs_are_versioned_json_and_csv(services, tmp_path):
    catalog, registry = services
    sample_ids = select_evaluation_samples(catalog, count=2)
    result = evaluate_samples(catalog, registry, sample_ids, version="test-v1", seed=7)
    paths = write_evaluation_outputs(result, str(tmp_path / "run"))

    for key in ("samples", "results", "results_csv", "summary", "summary_csv"):
        assert os.path.isfile(paths[key])

    with open(paths["summary"], encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["evaluation_version"] == "test-v1"
    assert summary["sampling"]["seed"] == 7
    assert summary["sampling"]["count"] == 2

    with open(paths["results_csv"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert set(rows[0]) >= {
        "sample_id", "model_id", "actual_fire_pixels", "valid_target_pixels",
        "predicted_fire_pixels", "tp", "fp", "fn", "tn", "precision",
        "recall", "f1_dice", "iou", "inference_ms",
    }
