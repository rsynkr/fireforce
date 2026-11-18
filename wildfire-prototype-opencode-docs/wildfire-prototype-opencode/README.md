# OpenCode handoff package

Read `TRACKER_AND_RULES.md` first. Then read PRD, TRD, app flow, design brief, schema, and implementation plan.

## You must also place/provide these artifacts
- Three trained model checkpoint files (M1, M2, M3).
- Exact U-Net architecture/model-definition source used to train them.
- Original preprocessing/inference code or notebook.
- Verified normalization statistics and feature order.
- Original TFRecord shards (or a local mount/symlink to them).
- Any existing evaluation code and the validation-selected thresholds.

Suggested local layout:
```text
artifacts/checkpoints/
artifacts/preprocessing/
data/
training_reference/
```

The docs do not contain the model binaries or dataset. Do not commit large binaries to Git. OpenCode must audit the actual artifacts before implementing inference.
