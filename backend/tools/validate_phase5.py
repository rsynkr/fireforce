"""Phase 5 validation: compare the production pipeline against the frozen
Phase 0 notebook transcription on representative samples.

The oracle is ``training_reference/phase0_reference.py``, which was transcribed
from first.ipynb and is intentionally independent of the backend services.
Nothing here rewrites Phase 0 artifacts.

Usage (repository root):
    python backend/tools/validate_phase5.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "training_reference"))

import phase0_reference as oracle  # noqa: E402

from app.config import Settings  # noqa: E402
from app.services.metrics import masked_binary_metrics  # noqa: E402
from app.services.model_registry import MODEL_SPECS, ModelRegistry  # noqa: E402
from app.services.preprocessing import (  # noqa: E402
    MODEL_CHANNELS,
    build_model_input,
    build_target,
    preprocess_features,
)
from app.services.sample_catalog import load_catalog  # noqa: E402

CASES = {
    "sparse_fire": "eval_00_000027",
    "dense_fire": "train_11_000326",
    "no_fire": "train_00_000001",
    "all_unknown": "train_00_000060",
    "phase0_all_unknown": "eval_00_000009",
    "partial_unknown_dense": "train_06_000286",
    "partial_unknown_sparse": "train_04_000491",
}

OUT_PATH = os.path.join(REPO_ROOT, "artifacts", "validation", "phase5_report.json")


def compare_arrays(label, ours, theirs):
    ours = np.asarray(ours)
    theirs = np.asarray(theirs)
    if ours.shape != theirs.shape:
        raise AssertionError(f"{label}: shape {ours.shape} != {theirs.shape}")
    diff = float(np.abs(ours.astype(np.float64) - theirs.astype(np.float64)).max())
    if not np.array_equal(ours, theirs):
        raise AssertionError(f"{label}: arrays differ, max_abs_diff={diff}")
    return diff


def main() -> int:
    started = time.perf_counter()
    settings = Settings.default()
    catalog = load_catalog(str(settings.manifest_path), str(settings.data_root))
    registry = ModelRegistry(str(settings.checkpoint_dir))

    oracle_models = {}
    for model_id, spec in MODEL_SPECS.items():
        model = oracle.UNet(in_channels=spec.in_channels, out_channels=1)
        model.load_state_dict(
            torch.load(
                os.path.join(str(settings.checkpoint_dir), spec.checkpoint_file),
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        oracle_models[model_id] = model

    report = {
        "pipeline": "backend/app/services/preprocessing.py + model_registry.py",
        "oracle": "training_reference/phase0_reference.py (frozen)",
        "device": str(registry.device),
        "thresholds": {mid: spec.threshold for mid, spec in MODEL_SPECS.items()},
        "model_channels": {mid: len(ch) for mid, ch in MODEL_CHANNELS.items()},
        "cases": {},
    }

    max_feature_diff = 0.0
    max_input_diff = 0.0
    max_prob_diff = 0.0
    pairs = 0

    for case_name, sample_id in CASES.items():
        case_started = time.perf_counter()
        raw = catalog.load_features(sample_id)

        ours_features = preprocess_features(raw)
        oracle_features = oracle.preprocess_features(raw)

        for key in sorted(set(ours_features) | set(oracle_features)):
            diff = compare_arrays(
                f"{case_name}/feature/{key}", ours_features[key], oracle_features[key]
            )
            max_feature_diff = max(max_feature_diff, diff)

        target = build_target(ours_features["FireMask"])
        oracle_valid = (raw["FireMask"] != -1).reshape(64, 64)
        oracle_target = (raw["FireMask"] == 1).astype(np.float32).reshape(64, 64)
        compare_arrays(f"{case_name}/valid_mask", target.valid_mask, oracle_valid)
        compare_arrays(f"{case_name}/target_mask", target.fire_mask, oracle_target)

        model_results = {}
        for model_id in MODEL_SPECS:
            spec = MODEL_SPECS[model_id]
            ours_input = build_model_input(ours_features, model_id)
            oracle_input = oracle.build_input(oracle_features, oracle.MODEL_FEATURES[model_id])
            diff = compare_arrays(f"{case_name}/{model_id}/input", ours_input, oracle_input)
            max_input_diff = max(max_input_diff, diff)

            ours_probs = registry.predict(model_id, ours_input)
            repeat_probs = registry.predict(model_id, ours_input)
            compare_arrays(f"{case_name}/{model_id}/repeat", repeat_probs, ours_probs)

            oracle_probs = None
            with torch.inference_mode():
                oracle_logits = oracle_models[model_id](
                    torch.from_numpy(oracle_input).unsqueeze(0)
                )
                oracle_probs = torch.sigmoid(oracle_logits)[0, 0].numpy()
            diff = compare_arrays(f"{case_name}/{model_id}/probabilities", ours_probs, oracle_probs)
            max_prob_diff = max(max_prob_diff, diff)

            ours_mask = ours_probs >= spec.threshold
            oracle_mask = oracle_probs >= spec.threshold
            compare_arrays(f"{case_name}/{model_id}/mask", ours_mask, oracle_mask)

            ours_metrics = masked_binary_metrics(ours_mask, target.fire_mask, target.valid_mask)
            oracle_metrics = oracle.masked_metrics(
                oracle_probs, raw["FireMask"], oracle_valid, spec.threshold
            )
            if ours_metrics["available"] != oracle_metrics["available"]:
                raise AssertionError(f"{case_name}/{model_id}: metrics availability differs")
            if ours_metrics["available"]:
                mapping = {
                    "TP": "true_positive",
                    "FP": "false_positive",
                    "FN": "false_negative",
                    "TN": "true_negative",
                }
                for oracle_key, ours_key in mapping.items():
                    if ours_metrics[ours_key] != oracle_metrics[oracle_key]:
                        raise AssertionError(
                            f"{case_name}/{model_id}: {ours_key} differs"
                        )
                for key in ("precision", "recall", "dice", "iou"):
                    oracle_value = oracle_metrics[key]
                    ours_value = ours_metrics[key]
                    if oracle_value is None:
                        # The backend follows the original global-evaluation
                        # convention (max(denominator, 1) -> 0.0); the Phase 0
                        # per-sample helper reports undefined as None.
                        if ours_value != 0.0:
                            raise AssertionError(
                                f"{case_name}/{model_id}: {key} undefined convention "
                                f"differs (ours={ours_value})"
                            )
                    elif ours_value != oracle_value:
                        raise AssertionError(f"{case_name}/{model_id}: {key} differs")

            model_results[model_id] = {
                "input_shape": list(ours_input.shape),
                "probabilities_sha256": hashlib.sha256(ours_probs.tobytes()).hexdigest(),
                "predicted_fire_pixels": int(ours_mask.sum()),
                "metrics": ours_metrics,
                "metrics_available": ours_metrics["available"],
            }
            pairs += 1

        report["cases"][case_name] = {
            "sample_id": sample_id,
            "split": catalog.get_entry(sample_id).split,
            "target_valid_pixels": int(target.valid_mask.sum()),
            "target_fire_pixels": int(target.fire_mask.sum()),
            "target_unknown_pixels": int(4096 - target.valid_mask.sum()),
            "models": model_results,
            "elapsed_ms": (time.perf_counter() - case_started) * 1000.0,
        }
        print(
            f"PASS {case_name:24s} {sample_id:18s} "
            f"valid={int(target.valid_mask.sum()):4d} fire={int(target.fire_mask.sum()):4d}"
        )

    # Frozen Phase 0 reference tensors (read-only).
    ref_path = os.path.join(REPO_ROOT, "artifacts", "reference", "phase0_reference.npz")
    reference = np.load(ref_path)
    phase0_checks = 0
    for case_name, prefix in (("sparse_fire", "fire"), ("phase0_all_unknown", "all_unknown")):
        sample_id = CASES[case_name]
        raw = catalog.load_features(sample_id)
        features = preprocess_features(raw)
        for model_id in MODEL_SPECS:
            probs = registry.predict(model_id, build_model_input(features, model_id))
            compare_arrays(f"phase0/{prefix}/{model_id}", probs, reference[f"{prefix}_{model_id}_probs"])
            phase0_checks += 1

    report["phase0_reference_checks"] = phase0_checks
    report["max_abs_diff"] = {
        "features": max_feature_diff,
        "model_inputs": max_input_diff,
        "probabilities": max_prob_diff,
    }
    report["model_sample_pairs"] = pairs
    report["elapsed_seconds"] = time.perf_counter() - started
    report["all_passed"] = True

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nPASS phase0 reference tensors: {phase0_checks} checks")
    print(
        f"max abs diff: features={max_feature_diff} inputs={max_input_diff} "
        f"probabilities={max_prob_diff} (exact match required)"
    )
    print(f"{pairs} model-sample pairs, {report['elapsed_seconds']:.2f}s")
    print(f"report: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
