import json
import os
import shutil

import numpy as np
import pytest
import torch

from app.services.model_registry import (
    MODEL_SPECS,
    InferenceError,
    ModelRegistry,
    UNet,
)
from app.services.preprocessing import build_model_input, preprocess_features
from app.services.sample_catalog import load_catalog


@pytest.fixture(scope="module")
def registry(real_settings):
    return ModelRegistry(str(real_settings.checkpoint_dir))


@pytest.fixture(scope="module")
def reference(real_paths):
    if not os.path.isfile(real_paths["phase0_npz"]):
        pytest.skip("Phase 0 reference not available")
    with open(real_paths["phase0_report"], encoding="utf-8") as handle:
        report = json.load(handle)
    return report, np.load(real_paths["phase0_npz"])


def test_all_three_checkpoints_load_strictly(registry):
    assert registry.ready_model_ids == ("M1", "M2", "M3")
    for model_id, spec in MODEL_SPECS.items():
        status = registry.status(model_id)
        assert status.ready, status.error
        assert status.in_channels == spec.in_channels
        assert status.threshold == spec.threshold
        assert status.sha256 and len(status.sha256) == 64


def test_checkpoint_hashes_match_phase0_report(registry, reference):
    report, _ = reference
    for model_id in ("M1", "M2", "M3"):
        status = registry.status(model_id)
        assert status.sha256 == report["checkpoints"][model_id]["sha256"]


def test_architecture_is_unchanged(registry):
    model = registry._models["M3"]
    assert isinstance(model, UNet)
    assert model.enc1.block[0].in_channels == 13
    assert model.final.out_channels == 1
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_strict_loading_rejects_wrong_channel_count(tmp_path, real_settings):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    # Deliberately place the 8-channel checkpoint where M1 (1 channel) expects it.
    shutil.copy2(
        real_settings.checkpoint_dir / "model_2_best.pth",
        checkpoint_dir / "model_1_best.pth",
    )

    broken = ModelRegistry(str(checkpoint_dir))
    assert broken.status("M1").ready is False
    assert "size mismatch" in broken.status("M1").error.lower()
    with pytest.raises(InferenceError):
        broken.predict("M1", np.zeros((1, 64, 64), dtype=np.float32))


def test_empty_state_dict_is_rejected(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save({}, checkpoint_dir / "model_1_best.pth")

    broken = ModelRegistry(str(checkpoint_dir))
    assert broken.status("M1").ready is False
    assert "state dict" in broken.status("M1").error


def test_direct_strict_load_mismatch_raises(real_settings):
    state = torch.load(
        real_settings.checkpoint_dir / "model_2_best.pth",
        map_location="cpu",
        weights_only=True,
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        UNet(in_channels=1).load_state_dict(state, strict=True)


def test_predict_rejects_bad_shape_and_dtype(registry):
    with pytest.raises(InferenceError, match="expects"):
        registry.predict("M1", np.zeros((8, 64, 64), dtype=np.float32))
    with pytest.raises(InferenceError, match="float32"):
        registry.predict("M1", np.zeros((1, 64, 64), dtype=np.float64))
    with pytest.raises(InferenceError, match="non-finite"):
        bad = np.zeros((1, 64, 64), dtype=np.float32)
        bad[0, 0, 0] = np.nan
        registry.predict("M1", bad)


def test_inference_is_deterministic_and_matches_phase0(
    registry, real_settings, real_paths, reference
):
    report, ref = reference
    catalog = load_catalog(str(real_settings.manifest_path), str(real_settings.data_root))
    raw = catalog.load_features("eval_00_000027")
    features = preprocess_features(raw)

    for model_id in ("M1", "M2", "M3"):
        model_input = build_model_input(features, model_id)
        first = registry.predict(model_id, model_input)
        second = registry.predict(model_id, model_input)

        assert first.shape == (64, 64)
        assert first.dtype == np.float32
        assert np.isfinite(first).all()
        assert np.array_equal(first, second)
        assert np.array_equal(first, ref[f"fire_{model_id}_probs"])
        assert report["samples"]["fire"]["models"][model_id]["threshold"] in (0.93, 0.94)
