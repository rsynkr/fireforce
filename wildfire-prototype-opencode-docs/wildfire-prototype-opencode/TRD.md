# Technical Requirements Document — Wildfire Prototype

## Recommended stack
- Frontend: React + Vite + TypeScript.
- Backend: Python + FastAPI.
- ML: PyTorch.
- Data: existing TFRecord shards.
- Sample index: versioned JSON manifest (no database for MVP).
- Tests: pytest + frontend test runner.

## Architecture
React UI → FastAPI → sample catalog / TFRecord loader / preprocessing / model registry / inference / metrics → TFRecord shards + manifest + checkpoints.

## Suggested structure
```text
frontend/
backend/app/{api,services,schemas}/
backend/tests/
artifacts/checkpoints/
artifacts/preprocessing/
artifacts/sample_manifest.json
data/                 # local mount; never commit dataset
docs/
```

## Stable IDs
Use deterministic IDs such as `train_00_000042`, derived from split, shard, and zero-based record index. A manifest maps ID → split, shard, record index, and verified optional metadata. Do not infer geography or chronology from the ID. Version the manifest and fingerprint the dataset.

## API
- `GET /api/health`: backend, data, checkpoint readiness, device.
- `GET /api/samples?split=train&search=...`: paginated sample catalog.
- `GET /api/samples/{sample_id}`: sample summary and target validity.
- `POST /api/predictions`: `{ "sample_id": "...", "models": ["M1","M2","M3"] }`.

Return prediction masks/probabilities, checkpoint identity, threshold, inference duration, model status, and metrics where valid.

## Inference
Load checkpoints once at startup; `model.eval()` and `torch.inference_mode()`. Select CUDA if available, otherwise CPU. Validate tensor dimensions and feature order. Preserve logits/probabilities separately from thresholded masks. Use validation-selected thresholds; never tune on test data.

Recorded thresholds in project notes are M1=.94, M2=.94, M3=.93; confirm they correspond to the exact checkpoints before use.

## Preprocessing
Must match training code exactly. Known research notes mention unknown previous-mask inputs mapped to 0, unknown targets masked, negative precipitation clipped to 0, and standardized continuous features. Wind direction circular encoding must be checked against actual code; do not guess whether sin/cos used raw or normalized degrees.

## Security
Bind localhost by default. Never accept arbitrary filesystem paths from API clients. Validate IDs against the manifest. Keep data/checkpoints out of Git.

## Tests
Stable lookup, record identity, feature order/channel counts, preprocessing parity, checkpoint loading, all-unknown target handling, hand-checked metrics, API errors, and CPU/CUDA behavior.
