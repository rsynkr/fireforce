"""Model registry for the three trained U-Net checkpoints.

The architecture is transcribed verbatim from ``first.ipynb`` cell 52 (training)
and ``second (1).ipynb`` cell 2 (evaluation). It must not be changed to make a
checkpoint load; channel counts are part of the checkpoint contract.

Checkpoints are loaded once when the registry is constructed (FastAPI lifespan),
kept in eval mode with gradients disabled, and reused for every request.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ARCHITECTURE_SOURCE = "first.ipynb cell 52; second (1).ipynb cell 2"

# Provisional thresholds recorded with the project notes / final test cell of
# second (1).ipynb. They have NOT been independently re-derived from the
# validation split in this build.
THRESHOLD_PROVENANCE = (
    "Validation-selected thresholds recorded with the project (M1=0.94, "
    "M2=0.94, M3=0.93). The original notebook's stored validation sweep "
    "outputs show maximum Dice at these values; this build has not re-run "
    "that sweep."
)


class InferenceError(RuntimeError):
    """A model is unavailable or received an invalid tensor."""


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    checkpoint_file: str
    in_channels: int
    threshold: float


MODEL_SPECS: dict[str, ModelSpec] = {
    "M1": ModelSpec("M1", "model_1_best.pth", 1, 0.94),
    "M2": ModelSpec("M2", "model_2_best.pth", 8, 0.94),
    "M3": ModelSpec("M3", "model_3_best.pth", 13, 0.93),
}

MODEL_IDS = tuple(MODEL_SPECS)


@dataclass(frozen=True)
class ModelStatus:
    model_id: str
    ready: bool
    checkpoint_file: str
    in_channels: int
    threshold: float
    sha256: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.ready else "unavailable",
            "checkpoint": self.checkpoint_file,
            "in_channels": self.in_channels,
            "threshold": self.threshold,
            "sha256": self.sha256,
            "error": self.error,
        }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRegistry:
    """Loads and owns all available models for the application lifetime."""

    def __init__(self, checkpoint_dir: str, device: str | None = None):
        self.checkpoint_dir = os.path.abspath(os.fspath(checkpoint_dir))
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._models: dict[str, UNet] = {}
        self._status: dict[str, ModelStatus] = {}

        for spec in MODEL_SPECS.values():
            self._load(spec)

    def _load(self, spec: ModelSpec) -> None:
        path = os.path.join(self.checkpoint_dir, spec.checkpoint_file)
        try:
            if not os.path.isfile(path):
                raise InferenceError(f"checkpoint not found: {path}")
            sha256 = _sha256_file(path)
            try:
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as exc:  # pragma: no cover - depends on corrupt files
                raise InferenceError(f"could not read checkpoint {spec.checkpoint_file}: {exc}")

            if not isinstance(state_dict, dict) or not state_dict:
                raise InferenceError(
                    f"{spec.checkpoint_file} is not a state dict"
                )

            model = UNet(in_channels=spec.in_channels, out_channels=1)
            # strict=True is deliberate: missing or unexpected keys are failures.
            model.load_state_dict(state_dict, strict=True)

            first_weight = state_dict.get("enc1.block.0.weight")
            if first_weight is None or tuple(first_weight.shape)[1] != spec.in_channels:
                raise InferenceError(
                    f"{spec.checkpoint_file} does not match {spec.in_channels} input channels"
                )

            model.to(self.device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

            self._models[spec.model_id] = model
            self._status[spec.model_id] = ModelStatus(
                model_id=spec.model_id,
                ready=True,
                checkpoint_file=spec.checkpoint_file,
                in_channels=spec.in_channels,
                threshold=spec.threshold,
                sha256=sha256,
                error=None,
            )
        except Exception as exc:
            self._status[spec.model_id] = ModelStatus(
                model_id=spec.model_id,
                ready=False,
                checkpoint_file=spec.checkpoint_file,
                in_channels=spec.in_channels,
                threshold=spec.threshold,
                sha256=None,
                error=str(exc) or exc.__class__.__name__,
            )

    def spec(self, model_id: str) -> ModelSpec:
        try:
            return MODEL_SPECS[model_id]
        except KeyError as exc:
            raise InferenceError(f"unsupported model {model_id!r}") from exc

    def status(self, model_id: str) -> ModelStatus:
        self.spec(model_id)
        return self._status[model_id]

    def statuses(self) -> dict[str, ModelStatus]:
        return dict(self._status)

    @property
    def ready_model_ids(self) -> tuple[str, ...]:
        return tuple(mid for mid in MODEL_IDS if self._status[mid].ready)

    def predict(self, model_id: str, model_input: np.ndarray) -> np.ndarray:
        """Run one forward pass and return a (64, 64) float32 probability map."""
        spec = self.spec(model_id)
        status = self._status[model_id]
        if not status.ready:
            raise InferenceError(f"{model_id} is unavailable: {status.error}")

        array = np.asarray(model_input)
        if array.shape != (spec.in_channels, 64, 64):
            raise InferenceError(
                f"{model_id} expects {(spec.in_channels, 64, 64)}, got {array.shape}"
            )
        if array.dtype != np.float32:
            raise InferenceError(f"{model_id} expects float32 input, got {array.dtype}")
        if not np.isfinite(array).all():
            raise InferenceError(f"{model_id} input contains non-finite values")

        tensor = torch.from_numpy(array).unsqueeze(0).to(self.device)
        with self._lock:
            with torch.inference_mode():
                logits = self._models[model_id](tensor)
                probabilities = torch.sigmoid(logits)[0, 0]

        result = probabilities.detach().to("cpu").numpy().astype(np.float32, copy=False)
        if result.shape != (64, 64) or not np.isfinite(result).all():
            raise InferenceError(f"{model_id} produced an invalid probability map")
        return result
