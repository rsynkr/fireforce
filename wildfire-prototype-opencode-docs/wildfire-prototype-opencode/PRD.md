# Product Requirements Document — Wildfire Prototype

## Product
A local research dashboard where a researcher selects a stable dataset sample ID, runs three trained U-Net models, and compares their next-day fire-mask predictions.

## Problem
Model inference and comparison currently require notebook execution. The prototype makes sample-level inspection repeatable and visual.

## Users
Primary: researcher/developer evaluating M1, M2, and M3.

## MVP goals
- Browse/search stable sample IDs.
- Load the exact TFRecord example for an ID.
- Apply training-identical preprocessing.
- Run all three trained checkpoints.
- Display previous-fire input, three predicted masks, and ground truth when valid.
- Calculate Dice, IoU, precision, and recall over valid target pixels only.
- Show checkpoint, threshold, sample ID, and inference status.

## Model variants
- M1: 1 channel — `PrevFireMask`.
- M2: 8 channels — `PrevFireMask`, `tmmn`, `tmmx`, `pr`, `sph`, `vs`, `th_sin`, `th_cos`.
- M3: 13 channels — M2 features plus `NDVI`, `pdsi`, `erc`, `elevation`, `population`.

Verify channel order and definitions against the actual training code before implementation.

## Out of scope
Training/fine-tuning, live satellite ingestion, emergency alerts, user accounts, cloud deployment, date/location mapping, GIS maps, and arbitrary image uploads.

## Acceptance criteria
- IDs resolve deterministically to shard + record index.
- Same ID and artifacts produce reproducible outputs.
- Each model receives the correct feature order and channel count.
- Unknown target pixels (`-1`) are excluded from metrics.
- All-unknown targets produce “Metrics unavailable”, never fabricated scores.
- Missing artifacts and model failures are reported clearly.
- No date/location claims are shown unless source metadata verifies them.

## Risks
Exact checkpoints, architecture, preprocessing, normalization, thresholds, and TFRecord paths must be recovered from the user's artifacts. This is a research prototype, not an operational wildfire forecast.
