"""Single shared preprocessing pipeline for M1, M2 and M3.

Transcribed from ``first.ipynb`` cell 46 (cleaning/normalization) and cell 47
(channel assembly). The three models share one feature pipeline; they only
select different channel subsets, so there is no per-model preprocessing code.

Contract:
- Input: 13 raw features parsed from a TFRecord example, each 4096 float32
  values (64x64, row-major). Extra keys are allowed; all required keys must be
  present, finite, and exactly 4096 values.
- Output: ``PreparedSample.input`` is float32 with shape (channels, 64, 64),
  ``.target`` is float32 (64, 64) in {0, 1}, and ``.valid`` is bool (64, 64).
- Neither raw input arrays nor targets are mutated.

No inference, thresholds, or HTTP concerns belong in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

PIXELS_PER_RECORD = 64 * 64
SAMPLE_SHAPE = (64, 64)

# Exact normalization constants from first.ipynb cell 46. Do not edit.
TRAIN_STATS = MappingProxyType(
    {
        "tmmn": (281.852110, 18.497169),
        "tmmx": (297.716643, 19.458073),
        "pr": (0.324562, 1.532254),
        "NDVI": (5355.635208, 2165.629625),
        "pdsi": (-0.772871, 2.440719),
        "erc": (53.469103, 25.097984),
        "elevation": (896.571320, 842.610351),
        "population": (30.460330, 214.200051),
        "sph": (0.006526, 0.003736),
        "vs": (3.627848, 1.309215),
    }
)

# Training used (x - mean) / (std + 1e-8); the eval notebook dropped the epsilon.
NORMALIZATION_EPSILON = 1e-8

RAW_FEATURES = (
    "tmmn",
    "tmmx",
    "pr",
    "NDVI",
    "pdsi",
    "erc",
    "elevation",
    "population",
    "sph",
    "vs",
    "th",
    "PrevFireMask",
    "FireMask",
)

# Source feature groups exactly as written in the training notebook. "th" is a
# single source feature that expands to th_sin + th_cos when channels are built.
MODEL_FEATURES = MappingProxyType(
    {
        "M1": ("PrevFireMask",),
        "M2": ("PrevFireMask", "tmmn", "tmmx", "pr", "sph", "vs", "th"),
        "M3": (
            "PrevFireMask",
            "tmmn",
            "tmmx",
            "pr",
            "sph",
            "vs",
            "th",
            "NDVI",
            "pdsi",
            "erc",
            "elevation",
            "population",
        ),
    }
)

# Actual tensor channel order after expansion. This is the contract the trained
# checkpoints expect (1 / 8 / 13 input channels).
MODEL_CHANNELS = MappingProxyType(
    {
        "M1": ("PrevFireMask",),
        "M2": (
            "PrevFireMask",
            "tmmn",
            "tmmx",
            "pr",
            "sph",
            "vs",
            "th_sin",
            "th_cos",
        ),
        "M3": (
            "PrevFireMask",
            "tmmn",
            "tmmx",
            "pr",
            "sph",
            "vs",
            "th_sin",
            "th_cos",
            "NDVI",
            "pdsi",
            "erc",
            "elevation",
            "population",
        ),
    }
)

CHANNEL_COUNTS = MappingProxyType(
    {model: len(channels) for model, channels in MODEL_CHANNELS.items()}
)


class PreprocessingError(ValueError):
    """Raw features or a model selection are not usable by the pipeline."""


@dataclass(frozen=True)
class Target:
    fire_mask: np.ndarray  # (64, 64) float32 in {0, 1}; -1 becomes 0 by definition
    valid_mask: np.ndarray  # (64, 64) bool; True where raw FireMask != -1


@dataclass(frozen=True)
class PreparedSample:
    model: str
    input: np.ndarray  # (channels, 64, 64) float32
    target: np.ndarray  # (64, 64) float32
    valid: np.ndarray  # (64, 64) bool


def _as_flat_float32(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise PreprocessingError(f"{name} must be numeric, got dtype {array.dtype}")
    if array.size != PIXELS_PER_RECORD:
        raise PreprocessingError(
            f"{name} must contain {PIXELS_PER_RECORD} values, got {array.size}"
        )
    flat = array.astype(np.float32, copy=False).reshape(PIXELS_PER_RECORD)
    if not np.isfinite(flat).all():
        raise PreprocessingError(f"{name} contains non-finite values")
    return flat


def _validate_raw(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(raw, Mapping):
        raise PreprocessingError("raw features must be a mapping")
    missing = [name for name in RAW_FEATURES if name not in raw]
    if missing:
        raise PreprocessingError(f"missing raw features: {', '.join(missing)}")
    return {name: _as_flat_float32(raw[name], name) for name in RAW_FEATURES}


def preprocess_features(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Apply training cleaning, normalization, and wind encoding.

    Mirrors first.ipynb cell 46 step for step:
      1. pr  = max(pr, 0)
      2. NDVI = clip(NDVI, -2000, 10000)
      3. th  = clip(th, -360, 360)
      4. z-score the 10 continuous features with std + 1e-8
      5. th_sin/cos from the clipped *raw degrees* (th is not normalized)
    """
    features = _validate_raw(raw)

    features["pr"] = np.maximum(features["pr"], 0.0)
    features["NDVI"] = np.clip(features["NDVI"], -2000.0, 10000.0)
    features["th"] = np.clip(features["th"], -360.0, 360.0)

    for name, (mean, std) in TRAIN_STATS.items():
        features[name] = (features[name] - mean) / (std + NORMALIZATION_EPSILON)

    theta = np.deg2rad(features["th"])
    features["th_sin"] = np.sin(theta)
    features["th_cos"] = np.cos(theta)

    for name in ("th_sin", "th_cos"):
        if not np.isfinite(features[name]).all():
            raise PreprocessingError(f"{name} is not finite after preprocessing")
    return features


def build_model_input(features: Mapping[str, np.ndarray], model: str) -> np.ndarray:
    """Assemble one model's channels in the checkpoint's expected order."""
    if model not in MODEL_FEATURES:
        raise PreprocessingError(
            f"unsupported model {model!r}; expected one of {sorted(MODEL_FEATURES)}"
        )

    channels: list[np.ndarray] = []
    for feature in MODEL_FEATURES[model]:
        if feature == "th":
            channels.append(features["th_sin"])
            channels.append(features["th_cos"])
        elif feature == "PrevFireMask":
            # Unknown previous-fire pixels are fed to the model as no-fire.
            prev_fire = np.array(features["PrevFireMask"], dtype=np.float32, copy=True)
            prev_fire[prev_fire == -1] = 0
            channels.append(prev_fire)
        else:
            channels.append(features[feature])

    stacked = np.stack(channels, axis=0).reshape(
        CHANNEL_COUNTS[model], *SAMPLE_SHAPE
    )
    if stacked.dtype != np.float32:
        raise PreprocessingError(f"input dtype drifted to {stacked.dtype}")
    if not np.isfinite(stacked).all():
        raise PreprocessingError(f"{model} input contains non-finite values")
    return stacked


def build_target(fire_mask: Any) -> Target:
    """Split raw FireMask into supervised target and validity mask.

    Unknown pixels stay -1 in the raw data; they are excluded through
    valid_mask and never treated as non-fire.
    """
    raw = _as_flat_float32(fire_mask, "FireMask")
    valid = (raw != -1).reshape(SAMPLE_SHAPE)
    target = (raw == 1).astype(np.float32).reshape(SAMPLE_SHAPE)
    return Target(fire_mask=target, valid_mask=valid)


def prepare_sample(raw: Mapping[str, Any], model: str) -> PreparedSample:
    """Run the full shared pipeline for one model."""
    if model not in MODEL_FEATURES:
        raise PreprocessingError(
            f"unsupported model {model!r}; expected one of {sorted(MODEL_FEATURES)}"
        )
    features = preprocess_features(raw)
    target = build_target(features["FireMask"])
    return PreparedSample(
        model=model,
        input=build_model_input(features, model),
        target=target.fire_mask,
        valid=target.valid_mask,
    )
