"""Phase 0 exit test: run the ORIGINAL notebook pipeline end-to-end.

Source of truth: first.ipynb (training notebook) for architecture + preprocessing;
second (1).ipynb for the final test-cell thresholds (0.94 / 0.94 / 0.93).

This script is intentionally a faithful transcription, not a refactor. It exists to
produce reference tensors for later preprocessing-parity tests.

Run: python training_reference/phase0_reference.py
Outputs: artifacts/reference/phase0_reference.npz, phase0_report.json
"""

import hashlib
import json
import os
import sys

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "archive")
CKPT_DIR = ROOT
OUT_DIR = os.path.join(ROOT, "artifacts", "reference")

SHARD = "next_day_wildfire_spread_eval_00.tfrecord"
SAMPLES = {"fire": 27, "all_unknown": 9}  # zero-based record indices, verified by scan

THRESHOLDS = {"M1": 0.94, "M2": 0.94, "M3": 0.93}

FEATURES = [
    "tmmn", "tmmx", "pr", "NDVI", "pdsi", "erc",
    "elevation", "population", "sph", "vs", "th",
    "PrevFireMask", "FireMask",
]
FEATURE_DESCRIPTION = {f: tf.io.FixedLenFeature([4096], tf.float32) for f in FEATURES}

# Verbatim from first.ipynb cell 46
TRAIN_STATS = {
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

MODEL_FEATURES = {
    "M1": ["PrevFireMask"],
    "M2": ["PrevFireMask", "tmmn", "tmmx", "pr", "sph", "vs", "th"],
    "M3": ["PrevFireMask", "tmmn", "tmmx", "pr", "sph", "vs", "th",
           "NDVI", "pdsi", "erc", "elevation", "population"],
}
EXPECTED_CHANNELS = {"M1": 1, "M2": 8, "M3": 13}


def normalize_feature(x, name):
    mean, std = TRAIN_STATS[name]
    return (x - mean) / (std + 1e-8)


def preprocess_features(features):
    features = features.copy()
    features["pr"] = np.maximum(features["pr"], 0.0)
    features["NDVI"] = np.clip(features["NDVI"], -2000.0, 10000.0)
    features["th"] = np.clip(features["th"], -360.0, 360.0)
    for name in TRAIN_STATS:
        features[name] = normalize_feature(features[name], name)
    theta = np.deg2rad(features["th"])
    features["th_sin"] = np.sin(theta)
    features["th_cos"] = np.cos(theta)
    return features


def build_input(features, input_features):
    channels = []
    for feature in input_features:
        if feature == "th":
            channels.append(features["th_sin"])
            channels.append(features["th_cos"])
        elif feature == "PrevFireMask":
            prev_fire = features["PrevFireMask"].copy()
            prev_fire[prev_fire == -1] = 0
            channels.append(prev_fire)
        else:
            channels.append(features[feature])
    return np.stack(channels, axis=0).reshape(-1, 64, 64).astype(np.float32)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
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
    def __init__(self, in_channels=1, out_channels=1):
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


def masked_metrics(probs, target, valid, threshold):
    target = target.reshape(64, 64)
    pred = probs >= threshold
    y_true = target[valid].astype(bool)
    y_pred = pred[valid]
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    tn = int((~y_pred & ~y_true).sum())
    if valid.sum() == 0:
        return {"available": False, "reason": "no_valid_target_pixels"}
    return {
        "available": True,
        "valid_pixels": int(valid.sum()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None,
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) else None,
    }


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    shard_path = os.path.join(DATA_DIR, SHARD)
    wanted = {v: k for k, v in SAMPLES.items()}

    raw_by_name = {}
    for i, raw in enumerate(tf.data.TFRecordDataset(shard_path)):
        if i in wanted:
            raw_by_name[wanted[i]] = raw.numpy()
        if len(raw_by_name) == len(wanted):
            break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name, channels in EXPECTED_CHANNELS.items():
        ckpt = os.path.join(CKPT_DIR, f"model_{name[1]}_best.pth")
        model = UNet(in_channels=channels, out_channels=1).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        models[name] = {"model": model, "path": ckpt, "sha256": sha256_file(ckpt)}

    report = {
        "phase": 0,
        "pipeline_source": "first.ipynb (architecture + preprocessing), second (1).ipynb (thresholds)",
        "normalization_epsilon": 1e-8,
        "device": str(device),
        "shard": SHARD,
        "shard_size_bytes": os.path.getsize(shard_path),
        "samples": {},
        "checkpoints": {
            k: {"file": os.path.basename(v["path"]),
                "in_channels": EXPECTED_CHANNELS[k],
                "sha256": v["sha256"]} for k, v in models.items()
        },
    }

    arrays = {}
    for sample_name, raw in raw_by_name.items():
        ex = tf.io.parse_single_example(raw, FEATURE_DESCRIPTION)
        raw_features = {k: ex[k].numpy() for k in FEATURES}
        features = preprocess_features(raw_features)

        target = raw_features["FireMask"]
        valid = (target != -1).reshape(64, 64)
        y = (target == 1).astype(np.float32).reshape(64, 64)

        sample_report = {
            "record_index": SAMPLES[sample_name],
            "split": "eval",
            "record_sha256": sha256_bytes(raw),
            "valid_pixels": int(valid.sum()),
            "fire_pixels": int((target == 1).sum()),
            "unknown_pixels": int((target == -1).sum()),
            "prev_fire_pixels": int((raw_features["PrevFireMask"] == 1).sum()),
            "prev_unknown_pixels": int((raw_features["PrevFireMask"] == -1).sum()),
            "models": {},
        }

        for name in ("M1", "M2", "M3"):
            x = build_input(features, MODEL_FEATURES[name])
            assert x.shape == (EXPECTED_CHANNELS[name], 64, 64), (name, x.shape)
            with torch.inference_mode():
                logits = models[name]["model"](torch.from_numpy(x)[None].to(device))
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            assert probs.shape == (64, 64) and np.isfinite(probs).all()

            metrics = masked_metrics(probs, target, valid, THRESHOLDS[name])
            sample_report["models"][name] = {
                "threshold": THRESHOLDS[name],
                "checkpoint_sha256": models[name]["sha256"],
                "prob_min": float(probs.min()),
                "prob_max": float(probs.max()),
                "prob_mean": float(probs.mean()),
                "predicted_fire_pixels": int((probs >= THRESHOLDS[name]).sum()),
                "metrics_masked": metrics,
            }
            arrays[f"{sample_name}_{name}_input"] = x
            arrays[f"{sample_name}_{name}_probs"] = probs

        arrays[f"{sample_name}_target"] = y
        arrays[f"{sample_name}_valid"] = valid
        report["samples"][sample_name] = sample_report

    np.savez_compressed(os.path.join(OUT_DIR, "phase0_reference.npz"), **arrays)
    with open(os.path.join(OUT_DIR, "phase0_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for sample_name, s in report["samples"].items():
        print(f"\n[{sample_name}] record {s['record_index']} | "
              f"valid={s['valid_pixels']} fire={s['fire_pixels']} unknown={s['unknown_pixels']}")
        for name, m in s["models"].items():
            mt = m["metrics_masked"]
            if not mt.get("available"):
                print(f"  {name} t={m['threshold']}: Metrics unavailable")
                continue
            print(f"  {name} t={m['threshold']} | predicted_fire={m['predicted_fire_pixels']:4d} | "
                  f"dice={mt['dice']:.4f} iou={mt['iou']:.4f} "
                  f"prec={mt['precision']:.4f} rec={mt['recall']:.4f}")
    print(f"\nWrote {OUT_DIR}\\phase0_reference.npz and phase0_report.json")


if __name__ == "__main__":
    sys.exit(main())
