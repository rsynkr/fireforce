"""Verify the Phase 0 reference is reproducible and internally consistent.

Fresh process: reloads the shard, rebuilds inputs, reruns all three checkpoints,
and compares bit-for-bit against artifacts/reference/phase0_reference.npz.
"""

import json
import os
import sys

import numpy as np
import tensorflow as tf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase0_reference as p

OUT_DIR = os.path.join(p.ROOT, "artifacts", "reference")


def main():
    with open(os.path.join(OUT_DIR, "phase0_report.json"), encoding="utf-8") as f:
        report = json.load(f)
    ref = np.load(os.path.join(OUT_DIR, "phase0_reference.npz"))

    device = torch.device("cpu")
    models = {}
    for name, ch in p.EXPECTED_CHANNELS.items():
        m = p.UNet(ch, 1).to(device)
        m.load_state_dict(torch.load(os.path.join(p.CKPT_DIR, f"model_{name[1]}_best.pth"),
                                     map_location=device))
        m.eval()
        models[name] = m

    shard = os.path.join(p.DATA_DIR, p.SHARD)
    wanted = {v: k for k, v in p.SAMPLES.items()}
    raw_by_name = {}
    for i, raw in enumerate(tf.data.TFRecordDataset(shard)):
        if i in wanted:
            raw_by_name[wanted[i]] = raw.numpy()
        if len(raw_by_name) == len(wanted):
            break

    checked = 0
    for sample_name, raw in raw_by_name.items():
        ex = tf.io.parse_single_example(raw, p.FEATURE_DESCRIPTION)
        rf = {k: ex[k].numpy() for k in p.FEATURES}
        feats = p.preprocess_features(rf)
        target = rf["FireMask"]
        valid = (target != -1).reshape(64, 64)

        assert np.array_equal(ref[f"{sample_name}_target"], (target == 1).reshape(64, 64))
        assert np.array_equal(ref[f"{sample_name}_valid"], valid)

        for name in ("M1", "M2", "M3"):
            x = p.build_input(feats, p.MODEL_FEATURES[name])
            assert np.array_equal(ref[f"{sample_name}_{name}_input"], x), "input drift"
            with torch.inference_mode():
                probs = torch.sigmoid(models[name](torch.from_numpy(x)[None]))[0, 0].numpy()
            saved = ref[f"{sample_name}_{name}_probs"]
            assert np.array_equal(saved, probs), f"probability drift: {sample_name}/{name}"

            entry = report["samples"][sample_name]["models"][name]
            got = p.masked_metrics(probs, target, valid, p.THRESHOLDS[name])
            assert got == entry["metrics_masked"], (sample_name, name, got, entry["metrics_masked"])
            checked += 1

        print(f"PASS {sample_name}: record {p.SAMPLES[sample_name]} "
              f"(valid={int(valid.sum())}, fire={int((target == 1).sum())})")

    # Depth check: channels match the registry, not just the checkpoint.
    for name, ch in p.EXPECTED_CHANNELS.items():
        assert ref[f"fire_{name}_input"].shape == (ch, 64, 64)
    print(f"PASS channels: 1/8/13")
    print(f"PASS reproducibility: {checked} model-sample pairs bit-identical")
    print("PASS metrics: recomputed TP/FP/FN/TN and scores match the report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
