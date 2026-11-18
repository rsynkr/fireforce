# Fireforce — Final Technical Handoff

Status: **Phase 5 complete** (2026-11-18); post-handoff extension adds the
model-evaluation summary and a fixed-seed 100-sample test-subset evaluation.
Phases 0–4 are implemented and validated. This is a local research prototype
for comparing three next-day wildfire-spread U-Net checkpoints. It is **not**
an operational forecast.

Scope boundary: no retraining, no architecture changes, no threshold tuning, no
database/auth/Docker/cloud, no live satellite ingestion, no date/location
inference from sample IDs.

---

## 1. Architecture and component responsibilities

```
React 19 SPA (Vite, TS)
   │  fetch JSON, VITE_API_BASE_URL
   ▼
FastAPI (127.0.0.1:8000)
   ├─ SampleCatalog ── manifest JSON + TFRecord shards (read-only)
   ├─ preprocessing ── 13 raw features → model-specific tensors
   ├─ ModelRegistry ── M1/M2/M3 loaded once, eval + inference_mode
   ├─ InferenceService ── one record read, shared preprocessing, per-model forward
   ├─ EvaluationService ── deterministic 100-sample run → artifacts/evaluation/
   └─ metrics ── valid-pixel-only TP/FP/FN/TN, Dice, IoU, precision, recall
```

| Component | Path | Responsibility |
|---|---|---|
| API app | `backend/app/main.py` | FastAPI factory, lifespan model load, localhost CORS |
| Settings | `backend/app/config.py` | env-driven data/manifest/checkpoint paths, device |
| Startup state | `backend/app/state.py` | build catalog + registry once |
| Routes | `backend/app/api/routes.py` | health, samples, sample detail, predictions |
| Schemas | `backend/app/schemas.py` | Pydantic request/response contracts |
| Catalog | `backend/app/services/sample_catalog.py` | manifest build/validate, deterministic IDs, record reads, masks, integrity |
| Preprocessing | `backend/app/services/preprocessing.py` | single shared pipeline for M1/M2/M3 |
| Registry | `backend/app/services/model_registry.py` | strict checkpoint load, device, forward pass |
| Inference | `backend/app/services/inference.py` | orchestration, timings, partial failure |
| Evaluation | `backend/app/services/evaluation.py` | deterministic selection, per-sample metrics, micro/macro aggregates, rankings, generated summary, JSON/CSV output |
| Evaluation tool | `backend/tools/run_evaluation.py` | runs the fixed-seed subset and writes versioned outputs |
| Metrics | `backend/app/services/metrics.py` | masked binary metrics |
| Manifest builder | `backend/tools/build_manifest.py` | reproducible manifest generation |
| Validation | `backend/tools/validate_phase5.py` | Phase 0 oracle parity across cases |
| HTTP acceptance | `backend/tools/smoke_http.py` | live endpoint/error/CORS checks |
| Benchmark | `backend/tools/benchmark_inference.py` | CPU latency measurements |
| Frozen oracle | `training_reference/phase0_reference.py` | Phase 0 transcription, not production code |
| Frontend | `frontend/src/` | catalog, preview, comparison, consensus, inspector, evaluation tab |
| Reference | `artifacts/reference/` | Phase 0 tensors and report (immutable) |
| Screenshots | `docs/screenshots/` | final browser QA states |

Runtime: checkpoints load once at application startup; requests share the same
model instances. Inference is `model.eval()` + `torch.inference_mode()`.
Each prediction request reads the TFRecord once and preprocesses once for all
three models.

---

## 2. Model specifications and checkpoints

Architecture: 3-level U-Net, transcribed verbatim from `first.ipynb` cell 52
(identical to `second (1).ipynb` cell 2). DoubleConv(3×3 conv + BatchNorm +
ReLU) ×2 per block.

- Encoder: 32 → 64 → 128 channels, MaxPool(2).
- Bottleneck: 256.
- Decoder: ConvTranspose2d + skip concat → 128 → 64 → 32.
- Head: 1×1 conv to 1 logit; sigmoid at inference.

| Model | Checkpoint | Inputs | Params | SHA-256 |
|---|---|---|---|---|
| M1 | `model_1_best.pth` | 1 | 1,927,841 | `3ec708fa1bca276f4b4d23778a4d5cb2b0e56c14913facac8c06bb589047f15f` |
| M2 | `model_2_best.pth` | 8 | 1,929,857 | `8174294b8dcacd1ca498e1d3458f9b3398108beb14f3e04ca79a5f159802daf5` |
| M3 | `model_3_best.pth` | 13 | 1,931,297 | `0bb918788309eaa65568338665b170901e860da7f6d4a035ccad06bc0644db75` |

Loading is strict (`load_state_dict(..., strict=True)`); an incompatible
checkpoint becomes an explicit unavailable/error model, never a silent load.

### Thresholds and provenance

| Model | Threshold | Validation Dice at that threshold | Origin |
|---|---|---|---|
| M1 | 0.94 | 0.3087 | 0.10–0.98 sweep, max validation Dice (`second (1).ipynb` cell 7) |
| M2 | 0.94 | 0.3250 | same sweep, max validation Dice (cell 8) |
| M3 | 0.93 | 0.3338 | staged 0.10–0.99 sweep, final max Dice (`first.ipynb` cells 92–94) |

The stored notebook outputs were inspected during Phase 5; these are genuinely
validation-selected values, not merely defaults copied from the test cell.
The sweep was **not re-run** in this build, and thresholds were not tuned here.

Recorded global test results from the original notebooks (not re-derived during
handoff): M1 Dice 0.3569 / IoU 0.2172 / precision 0.2640 / recall 0.5503;
M2 0.3780 / 0.2331 / 0.3200 / 0.4618; M3 0.3691 / 0.2263 / 0.2830 / 0.5304.

---

## 3. Dataset and manifest

Source: 19 Next-Day Wildfire Spread TFRecord shards in `archive/` (not committed).

| Split | Shards | Records |
|---|---|---|
| train | 15 | 14,979 |
| eval | 2 | 1,877 |
| test | 2 | 1,689 |
| **Total** | **19** | **18,545** |

Per-shard counts vary: `train_14` has 979, `eval_01` 877, `test_01` 689; the
rest have 1,000. Each record holds 13 features × 4,096 float32 values (64×64).

Manifest: `artifacts/sample_manifest.json` (generated, gitignored, reproducible).

- Manifest version 1, 4.23 MB
- Dataset fingerprint `3d618040c4bb0d5bdb9a321fdaaa453a7303903be0466c9e6b1bdc85833a025a`
- File SHA-256 `2634d3d3e13c212ba28ce4ac191652c4d5feb2fe1429b776b8b7aadd18dc9505`
- 44 samples have all target pixels unknown; 16,433 contain at least one fire
  pixel; 845,195 fire pixels total; densest sample has 952 fire pixels.
- There is **no all-fire target** in the dataset; dense-fire is represented by
  `train_11_000326` (952/4096).

Stable IDs: `<split>_<shard:02d>_<record_index:06d>` (zero-based), e.g.
`train_00_000042`, `eval_00_000027`. IDs encode only split/shard/record and are
never used to infer date, geography, or chronology.

Shard digests hash record payloads in order (not mtimes), so the fingerprint is
stable across copies. Manifest validation checks ID derivation, duplicates,
path safety, record bounds, `fire ≤ valid ≤ 4096`, and the referenced shards.
Every record read is rechecked against the manifest counts.

---

## 4. Preprocessing (shared by M1/M2/M3)

Raw features (13): `tmmn, tmmx, pr, NDVI, pdsi, erc, elevation, population,
sph, vs, th, PrevFireMask, FireMask`.

Exact order from `first.ipynb` cell 46:

1. `pr = max(pr, 0)`
2. `NDVI = clip(NDVI, -2000, 10000)`
3. `th = clip(th, -360, 360)`
4. z-score the 10 continuous features: `(x - mean) / (std + 1e-8)`
5. `th_sin = sin(deg2rad(th))`, `th_cos = cos(deg2rad(th))` from **clipped raw
   degrees** (th itself is not z-scored)
6. `PrevFireMask == -1 → 0` only when assembling model input
7. Target: `valid = FireMask != -1`, `target = (FireMask == 1)`; `-1` is never
   treated as non-fire

Channel order (the checkpoint contract):

- M1: `PrevFireMask`
- M2: `PrevFireMask, tmmn, tmmx, pr, sph, vs, th_sin, th_cos`
- M3: M2 + `NDVI, pdsi, erc, elevation, population`

Outputs are float32: input `(C, 64, 64)`, target `(64, 64)` in {0,1}, valid mask
`(64, 64)` bool. Non-finite raw values raise `PreprocessingError` (guard, not a
change to valid data). Inputs are not mutated.

Normalization statistics and feature definitions are in
`backend/app/services/preprocessing.py`; they must not be edited without
retraining and a new reference.

---

## 5. API endpoints and contracts

Base: `http://127.0.0.1:8000`. Docs: `/docs`. Bind localhost only.

### `GET /api/health`

```json
{
  "status": "ok | degraded | unavailable",
  "device": "cpu",
  "cuda_available": false,
  "threshold_provenance": "Validation-selected thresholds ... sweep not re-run",
  "data": {"ready": true, "manifest_version": 1, "fingerprint": "...",
           "shard_count": 19, "sample_count": 18545, "split_counts": {...},
           "error": null},
  "models": {"M1": {"status": "ready", "checkpoint": "model_1_best.pth",
                    "in_channels": 1, "threshold": 0.94, "sha256": "...",
                    "error": null}, ...}
}
```

### `GET /api/evaluation/summary`

Returns the generated evaluation summary (dashboard contract): version, split,
sampling strategy/seed/count, valid vs all-unknown counts, thresholds, per-model
micro/macro metrics, counts, predicted-fire distributions, rankings, and a
generated plain-English summary. `503` if the output has not been generated.

```bash
python backend/tools/run_evaluation.py
```

### `GET /api/samples`

Query: `split=train|eval|test`, `search=<substring>`, `limit=1..500` (default
50), `offset>=0`. Returns `{total, limit, offset, items[]}` where each item has
`sample_id, split, shard, record_index, target_valid_pixels,
target_fire_pixels, target_unknown_pixels, all_unknown`.

### `GET /api/samples/{sample_id}`

Adds `available_features[]`, `models[]`, and the preview masks:
`previous_fire_mask` (64×64, `-1/0/1`), `target_mask` (64×64, 0/1),
`target_valid_mask` (64×64 bool). This is an additive Phase 4 integration
change required to render the preview before inference.

### `POST /api/predictions`

Request:

```json
{"sample_id": "eval_00_000027", "models": ["M1", "M2", "M3"]}
```

`models` is optional and defaults to all three; duplicates/empty/unknown models
are rejected.

Response (abridged):

```json
{
  "sample_id": "...", "split": "eval", "shard": "...", "record_index": 27,
  "status": "ok | partial | failed", "device": "cpu", "duration_ms": 90.9,
  "available_features": ["..."],
  "ground_truth": {"available": true, "target_mask": [[...]],
                   "valid_mask": [[...]], "valid_pixels": 4096,
                   "fire_pixels": 21, "unknown_pixels": 0},
  "models": [{
    "model_id": "M1", "status": "ok | unavailable | error",
    "threshold": 0.94, "threshold_provenance": "...",
    "checkpoint": {"file": "model_1_best.pth", "sha256": "...",
                   "in_channels": 1},
    "inference_ms": 7.4, "probabilities": [[...64x64...]],
    "mask": [[true, false, ...]], "predicted_fire_pixels": 23,
    "metrics": {"available": true, "valid_pixels": 4096,
                "true_positive": 15, "false_positive": 8,
                "false_negative": 6, "true_negative": 4067,
                "precision": 0.6522, "recall": 0.7143,
                "dice": 0.6818, "iou": 0.5172},
    "error": null
  }]
}
```

Status codes:

| Condition | Status |
|---|---|
| malformed sample ID / unsupported model / empty or duplicate models | 422 |
| well-formed, unknown sample ID | 404 |
| missing manifest/data shard | 503 |
| no requested model available | 503 (detail lists per-model errors) |
| some models fail | 200, `status: "partial"`, successful models preserved |
| all target pixels unknown | 200, each `metrics.available=false`, scores `null`, reason `no_valid_target_pixels` |

CORS is restricted to `http://localhost:5173` and `http://127.0.0.1:5173`.
Clients never supply filesystem paths.

---

## 6. Validation results

### Phase 0 reference parity (frozen artifacts)

- `eval_00_000027` and `eval_00_000009`, all three models: probabilities
  bit-identical to `artifacts/reference/phase0_reference.npz`.
- Phase 5 extended this to 7 cases / 21 model-sample pairs using the independent
  frozen notebook transcription as oracle: **max abs diff 0.0** for features,
  model inputs, and probabilities. Metrics counts and scores match exactly
  (undefined denominators follow the notebook's `max(den,1) → 0.0` convention).
- Phase 0 artifacts were only read: npz SHA-256
  `e0403ecf50b1f17278bba7cdd226bc9b3e2504cb5397e6a2b96cb38abab11c08`,
  report SHA-256 `05c3ac5a35454d089ecd0f84641546aa801fec86acf02eed0a5b0ab655307a4f`.

### Representative cases (sample-level, valid pixels only)

| Case | Sample | Valid | Fire | M1 Dice | M2 Dice | M3 Dice |
|---|---|---|---|---|---|---|
| sparse fire | `eval_00_000027` | 4096 | 21 | 0.6818 | 0.3704 | 0.7692 |
| dense fire | `train_11_000326` | 4096 | 952 | 0.7669 | 0.8207 | 0.8805 |
| no fire | `train_00_000001` | 4096 | 0 | 0.0 | 0.0 | 0.0 |
| all unknown | `train_00_000060` | 0 | 0 | unavailable | unavailable | unavailable |
| partial unknown, dense | `train_06_000286` | 4077 | 580 | 0.2662 | 0.1224 | 0.4244 |
| partial unknown, sparse | `train_04_000491` | 201 | 2 | 0.0 | 0.0 | 0.0 |

### Sample-specific investigation — `train_00_000000`

The dashboard may show zero Dice for this sample even though all models return
predictions. Investigation:

- Catalog resolves to `next_day_wildfire_spread_train_00.tfrecord`, record `0`;
  record SHA-256 `5d135ada…6cda`. Valid pixels 4,096; target fire 7.
- Ground-truth fire coordinates (row, col): `(17,2) (17,3) (17,4) (18,1)
  (18,2) (18,3) (19,1)`.
- Preprocessed features, model inputs, and probabilities are **bit-identical**
  to the frozen notebook transcription for all three models; channel order,
  thresholds, and TP/FP/FN/TN are correct.
- Probability at the 7 actual fire pixels: M1 max 0.1005, M2 max 0.2279, M3 max
  0.3664 — all below the recorded thresholds (0.94/0.94/0.93).
- Predicted fire pixels are 11.4–27.8 pixels away from the nearest actual fire
  pixel; there is no overlap. M1 predicts 51 px, M2/M3 16 px, TP=0.

Conclusion: this is a **legitimate model miss on this sample**, not a
data-loading, preprocessing, channel-order, threshold, or metric bug. Lowering a
threshold just to catch this sample would be threshold tuning and was not done.
A regression test locks the observed counts and zero TP.

### 100-sample test-subset evaluation

Deterministic sampling: all 1,689 test IDs sorted, ranked by
`sha256("20261118:<sample_id>")`, lowest 100 selected, then sorted by ID.
Saved sample list: `artifacts/evaluation/100-test-v1/samples.json`. The selected
100 all had valid targets (0 all-unknown; the test split overall contains 2).
Every model evaluated exactly the same samples with recorded thresholds.

Micro/global metrics (aggregate counts before dividing):

| Model | Precision | Recall | F1/Dice | IoU | Accuracy | TP | FP | FN | TN | Predicted fire |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 0.2489 | 0.4882 | 0.3297 | 0.1974 | 0.9742 | 2,570 | 7,757 | 2,694 | 391,658 | 10,332 |
| M2 | 0.3367 | 0.4409 | 0.3818 | 0.2359 | 0.9814 | 2,321 | 4,573 | 2,943 | 394,842 | 6,914 |
| M3 | 0.2707 | 0.4791 | 0.3459 | 0.2091 | 0.9764 | 2,522 | 6,796 | 2,742 | 392,619 | 9,330 |

Macro averages over 100 valid samples:

| Model | Precision | Recall | F1/Dice | IoU | Accuracy |
|---|---|---|---|---|---|
| M1 | 0.2540 | 0.4847 | 0.2804 | 0.1786 | 0.9744 |
| M2 | 0.3030 | 0.4213 | 0.3136 | 0.2080 | 0.9816 |
| M3 | 0.2598 | 0.4806 | 0.2956 | 0.1912 | 0.9767 |

Predicted-fire distribution (100 samples): M1 total 10,332 / mean 103.32 /
median 39.5 / max 716 / 2.55% of valid pixels; M2 6,914 / 69.14 / 24 / 541 /
1.70%; M3 9,330 / 93.30 / 41.5 / 581 / 2.30%. Total valid pixels 404,679;
actual fire pixels 5,264.

Answers: **M1 predicts the most fire and the fewest is M2**. **M1 detects the
most actual fire (TP=2,570) but also produces the most false alarms
(FP=7,757).** **M2 performs best overall** on F1/Dice, IoU, precision, and
accuracy; **M1 has the best recall**. Accuracy is high for every model because
non-fire dominates, so it is not a headline score. These are 100-sample subset
results, not a full test-set or statistically conclusive evaluation.

Outputs: `artifacts/evaluation/100-test-v1/{samples,results,summary}.json` and
`{results,summary}.csv`. Thresholds were not tuned on these samples.

Run time was 6.8 s with a warm OS file cache; a cold rerun took 302 s because
each sample is read by scanning its TFRecord from the start. Per-model
`inference_ms` values are recorded per row. Selection and metrics were
byte-identical across both runs; only timings differ.

### HTTP acceptance (live uvicorn, port 8000)

Health, pagination, search, split filter, sample detail masks, all five
representative predictions, probability hashes vs. validation report, 404/422,
CORS allow/deny: **all passed**. The evaluation summary endpoint returns 200 with
the generated subset and 503 when output is missing. A second live instance with
M2 withheld (`WILDFIRE_CHECKPOINT_DIR`, port 8001) returned `degraded` health, 200
partial with M1/M3 preserved, and 503 when only M2 was requested. Offline
recovery was verified live: stop backend → `API offline`, 0 rows → restart →
**Retry connection** → `API online`, 18,545 samples.

### UI browser QA

Production bundle at 1440×1000 and 390×844 against the real backend. Canvas
backing store is 64×64 with `image-rendering: pixelated`; masks share identical
widths; probability maps use one shared 0–1 scale; consensus and synchronized
inspector verified. The Evaluation tab renders five tables, four comparison
charts, three confusion matrices, rankings, generated summary, and metric
definitions at both widths. Screenshots: `docs/screenshots/`.

### Test suites

- Backend: `python -m pytest backend\tests -q` → **105 passed, 0 failed**
  (12.42 s final run; 11–18 s depending on load).
- Frontend: `npm test` → **16 passed, 0 failed** (5.14 s final run after `npm ci`).
- Build: `npm run build` → 263.35 kB JS (79.73 kB gzip), 24.87 kB CSS
  (5.25 kB gzip), built in 231 ms.
- Validation: `python backend/tools/validate_phase5.py` → 7 cases, 21 pairs,
  6 Phase 0 tensor checks, all exact, 1.39 s.
- HTTP: `python backend/tools/smoke_http.py` → all checks passed.

### Baseline latency (CPU, `eval_00_000027`, after one warmup)

| Operation | Median |
|---|---|
| Load 3 checkpoints | 110 ms (startup only) |
| TFRecord read + integrity check | 80.7 ms (cold) |
| Preprocessing | 0.42 ms |
| M1 / M2 / M3 forward | 7.4 / 6.6 / 6.1 ms |
| `POST /api/predictions` (3 models, JSON) | 56.2 ms; ~366 KB response |

---

## 7. Setup and execution

Prerequisites: Python 3.13.5, Node 20+ (validated 26.13.1), ~4.2 GB of shards,
the three checkpoints, and the manifest.

Expected layout:

```text
fireforce/
  model_1_best.pth  model_2_best.pth  model_3_best.pth
  archive/next_day_wildfire_spread_{train, eval, test}_*.tfrecord
  artifacts/sample_manifest.json
  artifacts/reference/...
  backend/    frontend/    training_reference/    docs/
```

Install and build the index:

```bash
pip install -r backend/requirements-dev.txt        # runtime: backend/requirements.txt
python backend/tools/build_manifest.py             # ~10 s, 4.23 MB
python backend/tools/run_evaluation.py             # 100 test samples; seconds to minutes by cache state
```

Run the API:

```bash
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Run the UI:

```bash
cd frontend
npm ci
npm run dev        # http://127.0.0.1:5173
# production:
npm run build
npm run preview    # http://127.0.0.1:5173
```

Environment overrides:

```text
WILDFIRE_DATA_ROOT        default <repo>/archive
WILDFIRE_MANIFEST         default <repo>/artifacts/sample_manifest.json
WILDFIRE_CHECKPOINT_DIR   default <repo>
WILDFIRE_DEVICE           cpu | cuda (default auto)
WILDFIRE_EVALUATION_SUMMARY
                          default <repo>/artifacts/evaluation/100-test-v1/summary.json
VITE_API_BASE_URL         default http://127.0.0.1:8000
```

Clean-start verification performed for this handoff: manifest deleted and
rebuilt from the shards → byte-identical SHA-256; `npm ci` → 14 tests and
production build; fresh API/frontend processes → live HTTP and browser checks.

---

## 8. Testing commands

```bash
python -m pytest backend/tests -q
python backend/tools/validate_phase5.py
python backend/tools/run_evaluation.py
python backend/tools/smoke_http.py --base http://127.0.0.1:8000
python training_reference/verify_phase0.py          # Phase 0 reproducibility
python backend/tools/benchmark_inference.py --runs 5
cd frontend && npm test && npm run build
```

Do not modify `artifacts/reference/`; `verify_phase0.py` and the Phase 5
validator treat it as immutable evidence.

---

## 9. Known issues and limitations

1. **Thresholds were not re-derived.** Stored validation-sweep outputs were
   inspected and match 0.94/0.94/0.93; re-running the full sweep was out of scope.
2. **The 100-sample evaluation is a subset.** It uses a fixed-seed selection from
   the test split, contains 0 all-unknown samples (the full test split has 2), and
   is not a full-test or statistically conclusive result. Metrics are not
   comparable to the full validation/test numbers in the notebooks.
3. **`train_00_000000` is a real model miss.** All probabilities at its 7 actual
   fire pixels are below threshold; data, preprocessing, channels, thresholds,
   and metrics were verified against the frozen oracle. Do not tune thresholds to
   improve it.
4. **CPU-only validation.** `torch.cuda.is_available()` is false on the handoff
   machine; the CUDA path is implemented but untested.
5. **No all-fire target exists** in the dataset. Dense-fire coverage is
   `train_11_000326` (952/4096 fire).
6. **TFRecord lookup is sequential.** High record indices require scanning from
   the shard start; no cache/index was added because measurements did not
   justify it. Cold read of record 27 was ~81 ms.
7. **Partial-failure UI** is covered by frontend tests with a mocked contract and
   by live API tests on port 8001; a live browser screenshot of the degraded UI
   was not captured.
8. **No health polling.** Offline is detected on load or retry, not continuously.
9. **Mobile is desktop-first.** At phone widths the catalog sits above the
   workspace, so returning to a selection requires scrolling.
10. **Dev dependency advisories.** `npm audit` reports 6 high advisories in the
    Vite/Vitest toolchain; production dependencies report 0. Pinned Python
    packages returned 0 OSV advisories. `pip check` shows conflicts in unrelated
    global packages (numba/streamlit), not in this project.
11. **Legacy handoff README** in `wildfire-prototype-opencode-docs/` predates
    implementation and still says artifacts must be supplied. They are present;
    this document supersedes it.
12. **Evaluation timings are not reproducible.** Selection, counts, masks, and
    metrics are deterministic; `inference_ms` and `elapsed_seconds` vary by run.
13. No auth, database, Docker, or cloud deployment, by design. Bind localhost.

## 10. Provenance

Tracker: `wildfire-prototype-opencode-docs/wildfire-prototype-opencode/TRACKER_AND_RULES.md`.
Phase 0 reference: `artifacts/reference/phase0_report.json` (generated from
`eval_00_000027` and `eval_00_000009`). Phase 5 validation:
`artifacts/validation/phase5_report.json` (generated). Evaluation:
`artifacts/evaluation/100-test-v1/` (generated; summary plus full rows).

All research outputs; not an operational wildfire forecast.
