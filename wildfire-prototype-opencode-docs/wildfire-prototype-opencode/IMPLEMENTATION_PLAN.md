# Implementation Plan

## Phase 0 — Artifact audit (blocking)
- Locate TFRecord shards and verify split names.
- Locate all three checkpoints.
- Locate exact U-Net definition.
- Locate training preprocessing and normalization statistics.
- Confirm feature order, input shape, thresholds, and checkpoint mapping.
- Reproduce one known sample through original notebook inference.

**Exit:** exact architecture and preprocessing can reproduce a known input/output.

## Phase 1 — Sample index and loader
- Stream TFRecords and generate deterministic split/shard/index IDs.
- Build versioned manifest.
- Implement `get_sample(sample_id)`.
- Test repeat lookup identity.

## Phase 2 — Preprocessing parity
- Extract shared preprocessing.
- Build M1/M2/M3 tensors in exact feature order.
- Verify normalization, clipping, missing-value policy, wind-direction encoding.
- Add shape/range/parity tests.

## Phase 3 — Inference API
- Load exact architecture and checkpoints once.
- Add device selection and inference mode.
- Apply verified per-model thresholds.
- Implement metrics masking unknown target pixels.
- Add API and failure tests.

## Phase 4 — Frontend
- Dashboard and backend status.
- Searchable sample selector.
- Input/ground-truth preview.
- Run Predictions action.
- Three comparison cards and metrics.
- Loading, empty, partial-failure, and error states.

## Phase 5 — Validation and handoff
- Test ordinary, sparse, all-fire, and all-unknown target cases.
- Compare browser output against notebook.
- Test CPU and CUDA if available.
- Document setup, data paths, commands, limitations.

## Priority
P0: artifact audit, loader, preprocessing parity, real checkpoint inference.
P1: selector, mask visualization, metrics.
P2: caching, export, richer filters.

Do not polish the UI before inference correctness is established.
