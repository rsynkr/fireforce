# Backend Schema / Data Contract

## MVP storage
No SQL database is required. Use immutable TFRecord shards, a versioned JSON sample manifest, versioned preprocessing config, and model checkpoint files.

## Manifest example
```json
{
  "manifest_version": 1,
  "dataset_name": "next-day-wildfire-spread",
  "dataset_fingerprint": "REPLACE_WITH_REAL_FINGERPRINT",
  "samples": [{
    "sample_id": "train_00_000042",
    "split": "train",
    "shard": "train_00.tfrecord",
    "record_index": 42,
    "target_valid_pixels": null,
    "metadata": {}
  }]
}
```
IDs must be unique and deterministic. Record index is zero-based. Resolve shard names only beneath a configured data root. Metadata must contain only verified source fields.

## Model registry example
```yaml
models:
  M1:
    checkpoint: artifacts/checkpoints/M1.pth
    in_channels: 1
    features: [PrevFireMask]
    threshold: 0.94
  M2:
    checkpoint: artifacts/checkpoints/M2.pth
    in_channels: 8
    features: [PrevFireMask, tmmn, tmmx, pr, sph, vs, th_sin, th_cos]
    threshold: 0.94
  M3:
    checkpoint: artifacts/checkpoints/M3.pth
    in_channels: 13
    features: [PrevFireMask, tmmn, tmmx, pr, sph, vs, th_sin, th_cos, NDVI, pdsi, erc, elevation, population]
    threshold: 0.93
```
Thresholds are recorded validation-selected values from project notes. Verify checkpoint association before use.

## Prediction response
Return sample ID, status, device, per-model status, checkpoint identity, threshold, 64×64 probabilities/mask, inference milliseconds, metrics, and ground-truth validity counts. Do not return fake placeholder values.

## Metric rules
Valid target pixels are `target != -1`. Actual fire is `target == 1`; non-fire is `target == 0`. Compute TP/FP/FN/TN only over valid pixels. If valid pixel count is zero, metrics are null with reason `no_valid_target_pixels`.

## Validation
Unknown sample ID → 404. Unsupported model → 422. Missing model/data artifacts → clear service-unavailable response. Never accept client-provided filesystem paths.
