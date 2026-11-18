# Live Tracker & Agent Rules

## Progress tracker
| Phase | Status | Evidence |
|---|---|---|
| 0. Artifact audit | COMPLETE | 3 checkpoints load strictly (1/8/13 ch); 19 shards, 18,545 records; architecture + preprocessing recovered; reference run `verify_phase0.py` passes |
| 1. Sample index + loader | COMPLETE | `artifacts/sample_manifest.json` (18,545 IDs, fingerprint `3d6180…`); strict `SampleCatalog`; 30 tests pass |
| 2. Preprocessing parity | COMPLETE | Shared pipeline; 6/6 tensors bit-identical to Phase 0 reference (`max_abs_diff=0.0`); 59 tests pass |
| 3. Inference API | COMPLETE | FastAPI on real checkpoints; parity with Phase 0; 93 tests pass; CPU medians 7.4/6.6/6.1 ms |
| 4. Frontend | COMPLETE | React+Vite dashboard; 16 frontend tests + 105 backend tests; production build; browser QA at 1440/390 |
| 5. Validation + handoff | COMPLETE | Clean rebuild + npm ci; 105 backend + 16 frontend tests; 21/21 exact parity; live HTTP/partial/offline; evaluation summary + 100-sample test subset; `HANDOFF.md` |

Update this table after every work session.

## Session log

### 2026-11-18 — Phase 0 closed
- Verified locally: `model_1_best.pth` (1 ch), `model_2_best.pth` (8 ch), `model_3_best.pth` (13 ch) are plain `OrderedDict` state dicts (106 keys) and load with `strict=True` into the notebook UNet. Params: 1,927,841 / 1,929,857 / 1,931,297.
- Dataset: 19 shards in `archive/` (15 train, 2 eval, 2 test), 18,545 records total (14,979 / 1,877 / 1,689). Counts vary by shard (e.g. `train_14`=979, `test_01`=689); only `train_00`–`train_13` plus `eval_00` and `test_00` are 1000. Split names come from filenames only.
- Architecture source: `first.ipynb` cell 52 (training) = `second (1).ipynb` cell 2 (eval). Preprocessing source: `first.ipynb` cell 46; channel builder: cell 47.
- Reference run: `training_reference/phase0_reference.py` → `artifacts/reference/phase0_reference.npz` + `phase0_report.json`. Uses `eval_00` record 27 (21 fire px, Dice M1 .6818 / M2 .3704 / M3 .7692) and record 9 (all-unknown target → metrics unavailable).
- Verification: `training_reference/verify_phase0.py` re-ran all six model-sample pairs in a fresh process and matched saved inputs, probabilities, and metrics bit-for-bit.
- Known discrepancy (not a blocker): training notebook z-scores with `std + 1e-8`, eval notebook uses `std`. Reference follows training.
- Thresholds used: M1 .94, M2 .94, M3 .93, taken from the final test cell of `second (1).ipynb`. Checkpoint association confirmed by strict load + channel counts.
- Remaining Phase 0 caveat: no stored notebook output exists to diff against, so the reference is produced by the transcribed pipeline, not matched to a prior run.

### 2026-11-18 — Phase 1 closed
- Built `artifacts/sample_manifest.json`: manifest v1, 19 shards, 18,545 samples (train 14,979 / eval 1,877 / test 1,689), fingerprint `3d618040c4bb0d5bdb9a321fdaaa453a7303903be0466c9e6b1bdc85833a025a`, 4.23 MB.
- Scan time 11.12s; a second build took 10.70s and produced a byte-identical file (SHA-256 `2634d3d3…9505`).
- Manifest content: 44 all-unknown targets (`target_valid_pixels == 0`), 16,433 samples with at least one fire pixel, 845,195 fire pixels total.
- Shard digests hash record payloads, not mtimes, so the dataset fingerprint is stable across copies.
- Loader: `backend/app/services/sample_catalog.py` — strict manifest validation (ID derivation, duplicate IDs, shard path safety, record-index bounds, fire<=valid, per-record data-vs-manifest check), `SampleCatalog.get_sample()` raises `ArtifactMissingError` / `ManifestIntegrityError` / `RecordIndexError` instead of leaking OS errors.
- Tests: `python -m pytest backend\tests -q` → **30 passed, 0 failed in 2.83s**. Covers first/middle/last, all-unknown target, malformed IDs, unknown IDs, missing shard, missing manifest, duplicate IDs, out-of-range/negative indices, unsafe shard paths, count mismatch, round-trip, fingerprint change, and real-data parity with the Phase 0 reference (`eval_00_000027` record hash + 4096/21 counts).
- Corrected the Phase 0 audit: shards are not uniformly 1000 records; 4 shards are short.
- `.gitignore` added before any `git init`: excludes `archive/`, `data/`, `*.tfrecord`, `*.pth`, generated `artifacts/sample_manifest.json`, `.env`/keys; keeps `artifacts/reference/` fixtures.

### 2026-11-18 — Phase 2 closed
- Shared pipeline: `backend/app/services/preprocessing.py`. One `preprocess_features()` serves M1/M2/M3; models only select channel groups. No per-model preprocessing copies.
- Contract: 13 raw features x 4096 float32; `pr=max(pr,0)`, `NDVI∈[-2000,10000]`, `th∈[-360,360]`, z-score 10 continuous features with `(x-mean)/(std+1e-8)`, `th_sin/cos` from clipped raw degrees, `PrevFireMask==-1 → 0` only in model input, target `-1` excluded via `valid_mask`.
- Channels: M1=1, M2=8 (`PrevFireMask,tmmn,tmmx,pr,sph,vs,th_sin,th_cos`), M3=13 (M2 + `NDVI,pdsi,erc,elevation,population`).
- Catalog extension: `SampleCatalog.load_features()` returns all 13 raw features; `get_sample()` and `load_features()` share one manifest-integrity check. No inference code added.
- Parity: `eval_00_000027` (fire) and `eval_00_000009` (all-unknown) x M1/M2/M3 compared to `artifacts/reference/phase0_reference.npz` with **exact** equality, no tolerance: `max_abs_diff = 0.0`, dtype `float32`, shape `(C,64,64)`. Target/valid arrays also exact.
- Tests: `python -m pytest backend\tests -q` → **59 passed, 0 failed in 3.45s** (29 Phase 2 tests added). Covers shapes/dtype/finiteness, channel order via independent formula, unknown previous-fire/target handling, all-unknown target, clipping boundaries, wind cardinal directions, epsilon, non-mutation, bit-repeatability, invalid inputs, and Phase 0 parity.
- Reference artifacts were not regenerated: `phase0_reference.npz` mtime 2026-11-18 07:21:58, SHA-256 `e0403ecf…b11c08`; `phase0_report.json` SHA-256 `05c3ac5a…307a4f`.
- Intentional guard, not a training-behaviour change: non-finite raw feature values raise `PreprocessingError` instead of propagating NaN. All valid TFRecord rows are unaffected.

### 2026-11-18 — Phase 3 closed
- FastAPI app factory in `backend/app/main.py`; models load once in the lifespan via `ModelRegistry`, stay in `eval()` with gradients off, and run under `torch.inference_mode()`. Device is CUDA when available, otherwise CPU (this machine: CPU, torch 2.14.0+cpu).
- Endpoints: `GET /api/health`, `GET /api/samples` (split/search/limit/offset), `GET /api/samples/{sample_id}`, `POST /api/predictions` (`sample_id`, optional `models` defaulting to all three).
- Registry: M1 `3ec708fa…f15f` (1 ch), M2 `8174294b…daf5` (8 ch), M3 `0bb91878…4b75` (13 ch). Strict `load_state_dict`; wrong channel count, empty state dicts, and corrupted files become explicit per-model failures, not crashes.
- Partial failure: per-model `status` is `ok` / `unavailable` / `error`; at least one success → 200 with `status: "partial"`; zero successes → 503. Missing manifest → health `unavailable`, data endpoints 503.
- Metrics: valid pixels only; all-unknown target → `{"available": false, "reason": "no_valid_target_pixels"}` with null scores. No zero-valued fabrication.
- Threshold provenance is carried on every model entry and in health: recorded 0.94/0.94/0.93 from `second (1).ipynb` final test cell, provisional, **not independently re-derived from validation**.
- Phase 0 parity through the API: probabilities for `eval_00_000027` are bit-identical to `phase0_reference.npz` for all three models; TP/FP/FN/TN and Dice/IoU/precision/recall match the Phase 0 report (abs tolerance 1e-12).
- Tests: `python -m pytest backend\tests -q` → **93 passed, 0 failed in 10.33s** (34 added). Includes strict loading, shapes, determinism, metrics, all-unknown targets, invalid IDs, missing checkpoints, partial failure, and manifest-missing 503s.
- Latency (CPU, `eval_00_000027`, 5 runs after warmup): checkpoint load 110.0ms (all three); TFRecord read 80.7ms cold; preprocessing 0.42ms; forward medians M1 7.4ms, M2 6.6ms, M3 6.1ms; `POST /api/predictions` (3 models, JSON in/out) median 56.2ms. Full response 366 KB (probabilities + masks + ground truth).
- No caching added: measurements do not yet justify it. Reading a high record index still scans from shard start; this is the main scaling limitation.
- Not exercised: CUDA path (no CUDA device available) and live uvicorn networking (tests use Starlette's in-process TestClient). Run with `uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000`.

### 2026-11-18 — Phase 4 closed
- Stack: React 19 + Vite 8 + TypeScript 7 in `frontend/`; no UI framework, no state library. Fonts self-hosted (IBM Plex Sans/Mono, latin subset) via `@fontsource`.
- Design: charcoal instrument panel, fire orange restricted to actions/fire pixels, IBM Plex type system, numbered sections, 2 px radii, no gradients/shadows. Concept, palette, grid and responsive rules recorded in `frontend/DESIGN.md`.
- Views: catalog (split filter, debounced search, 50/page pagination), sample header, previous-fire + ground-truth preview, sample facts, Binary / Probability / Consensus comparison, metrics, checkpoint/inference/threshold metadata, offline/loading/partial/all-unknown states.
- Signature interaction: synchronized pixel inspector. Hovering any mask draws the same r/c crosshair on all five masks and reports per-model value plus hit/miss. Consensus view aggregates 0–3 model votes and split-vote counts.
- Backend integration change (documented, additive): `GET /api/samples/{id}` now returns `previous_fire_mask`, `target_mask`, `target_valid_mask` so the preview can render before predictions. CORS allows only `http://localhost:5173` and `http://127.0.0.1:5173`; no wildcard. Existing prediction contract unchanged.
- Frontend env: `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`), example in `frontend/.env.example`.
- Tests: `npm test` → **14 passed, 0 failed (4.64s)**. Covers browsing, search debounce, split filter, pagination, selection, prediction request/body, result metadata, clearing on new selection, all-unknown metrics, partial failure preservation, probability scale, consensus, offline/retry recovery, duplicate-request prevention, 503 and 404 handling.
- Backend: `python -m pytest backend\tests -q` → **94 passed, 0 failed (11.10s)** after adding deterministic-mask detail assertions and a CORS allow/deny test.
- Build: `npm run build` → 248.85 kB JS (76.57 kB gzip), 19.32 kB CSS (4.40 kB gzip).
- Browser QA against the real FastAPI service (production preview): desktop 1440×1000 and mobile 390×844; `docs/screenshots/` holds eight states. Verified canvas backing store 64×64 with `image-rendering: pixelated`, identical mask widths across preview/models, and live probabilities matching the API.
- Defects found and fixed during QA: (1) probability maps were smoothed because `crisp-edges` overrode `pixelated`; (2) consensus "0 models" bar was rendered in fire orange; (3) small 9 px labels were below 4.5:1 contrast (`--faint` lightened from `#6e7369` to `#7b8076`); (4) duplicate retry clutter reduced; (5) retrying after an offline start reloaded health but not the catalog — sample loading now waits for data-ready health and one retry restores both (regression test added).
- Limitations: partial-model failure and 503 are covered by frontend tests with a mocked contract plus backend tests, but were not reproduced against a live degraded backend. Deselecting/clearing the catalog on small screens still requires scrolling past the catalog on mobile (desktop-first decision).

### 2026-11-18 — Phase 5 closed
- Clean start: deleted `artifacts/sample_manifest.json`, rebuilt with `backend/tools/build_manifest.py` (9.82s) → byte-identical SHA-256 `2634d3d3…9505`, fingerprint `3d618040…025a`. `npm ci` from the lockfile, fresh API/preview processes, live HTTP checks.
- Threshold provenance verified from stored notebook outputs rather than assumption: M1 0.94 (val Dice 0.3087), M2 0.94 (0.3250), M3 0.93 (0.3338). M3 used staged sweeps 0.10–0.99; M1/M2 used 0.10–0.98 step 0.02 and selected max validation Dice. The sweep was not re-run and no threshold was changed. The API/frontend provenance copy was corrected to say “validation-selected … sweep not re-run”.
- Extended parity beyond the two Phase 0 samples using the frozen transcription as an independent oracle: sparse (`eval_00_000027`), dense (`train_11_000326`, 952 fire), no-fire (`train_00_000001`), all-unknown (`train_00_000060`), Phase 0 all-unknown (`eval_00_000009`), and two partially-unknown samples. **7 cases / 21 model-sample pairs, max abs diff 0.0** for features, inputs, and probabilities; metrics match exactly (undefined denominators follow the notebook’s `max(den,1) → 0.0` convention). Report: `artifacts/validation/phase5_report.json`.
- Live HTTP acceptance (`backend/tools/smoke_http.py`): health, pagination, search, split filter, detail masks, five representative predictions, probability hashes vs. validation report, 404/422, CORS allow/deny — all passed. A second live instance with M2 withheld (`WILDFIRE_CHECKPOINT_DIR`, port 8001) returned degraded health, 200 partial with M1/M3 preserved, and 503 for M2-only.
- Offline recovery verified live: stop backend → `API offline`, 0 rows; restart → Retry connection → `API online`, 18,545 samples. All-unknown live predictions return null metrics for all three models.
- Git exclusion audit (temp repo, correct negation handling): datasets (`archive/`, `*.tfrecord`), checkpoints (`*.pth`), generated manifest/validation, `node_modules/`, `dist/`, env/credentials/keys all ignored; `artifacts/reference/` fixtures, `backend/requirements.txt`, app code, docs screenshots, and `HANDOFF.md` remain trackable. Added `id_rsa*`, `id_ed25519*`, `*.p12`, `*.pfx`, `.npmrc`, `.pypirc`.
- Dependencies: `npm audit --omit=dev` → 0 vulnerabilities; full audit → 6 high, all in the Vite/Vitest dev chain. Pinned Python runtime packages queried against OSV → 0 advisories. `pip check` reports only unrelated global-package conflicts (numba/streamlit).
- Tests: backend **94 passed, 0 failed** (11.1 s standalone; 17.99 s on a loaded final run); frontend **14 passed, 0 failed** (5.34 s after `npm ci`); production build 248.85 kB JS / 76.56 kB gzip + 19.32 kB CSS / 4.40 kB gzip. Added `backend/requirements.txt`, `backend/requirements-dev.txt`.
- Screenshots recaptured against the final build (eight states in `docs/screenshots/`), including the corrected provenance footer. Final handoff: `HANDOFF.md` (architecture, models/hashes, dataset, preprocessing, API, results, setup, tests, known issues).
- Unresolved: CUDA untested (CPU-only host); full validation threshold sweep not re-run; no all-fire target exists; sequential TFRecord scan for high indices; partial UI live screenshot not captured (tests + live API cover it); no health polling; mobile catalog-first layout; legacy handoff README is stale and superseded by `HANDOFF.md`.

### 2026-11-18 — Post-Phase 5 extension: evaluation summary + 100-sample test subset
- `train_00_000000` investigation: correct shard/record (train_00 record 0, record SHA `5d135ada…6cda`), 4,096 valid / 7 fire at `(17,2)…(19,1)`. Features, inputs, probabilities bit-identical to the frozen oracle; correct thresholds and masked counts. Probability at the 7 fire pixels: M1 0.1005, M2 0.2279, M3 0.3664 — all below 0.94/0.94/0.93. Predicted fire is 11.4–27.8 px away; TP=0 for all. Conclusion: genuine model miss, not data/preprocessing/metric bug. Regression test added.
- Evaluation service `backend/app/services/evaluation.py` + tool `backend/tools/run_evaluation.py`: deterministic selection from the **test** split (1,689 IDs, SHA-256 rank with seed 20261118, lowest 100, sorted), identical samples for all models, recorded thresholds, valid-pixel masking, per-sample timing, JSON/CSV output.
- Run: 100/100 valid samples (0 all-unknown selected; test split overall has 2), 404,679 valid pixels, 5,264 actual fire pixels, 6.78s warm (302.2s cold; dominated by sequential shard scans). Reproducibility rerun produced identical summary (excluding elapsed) and identical sample IDs. Micro: M1 P .2489/R .4882/F1 .3297/IoU .1974; M2 .3367/.4409/.3818/.2359; M3 .2707/.4791/.3459/.2091. Macro F1: .2804/.3136/.2956. M1 predicts most fire (10,332) and most TP (2,570) but also most FP (7,757); M2 fewest (6,914) and best F1/IoU/precision/accuracy; M1 best recall.
- Outputs: `artifacts/evaluation/100-test-v1/{samples,results,summary}.json` + `{results,summary}.csv`; dashboard endpoint `GET /api/evaluation/summary` (503 when absent).
- Frontend: Inspector/Evaluation tabs; evaluation tables (micro + macro), nine rankings with ties, F1/IoU/predicted-fire bars, precision–recall scatter, three confusion matrices, generated dynamic summary, metric definitions. Evaluation page hides the catalog and uses full width. Screenshots 09/10 added; all inspector screenshots recaptured with tabs.
- Tests: backend **105 passed** (12.42s final); frontend **16 passed** (5.14s final). Build 263.35 kB JS / 79.73 kB gzip, 24.87 kB CSS / 5.25 kB gzip. Report screenshots recaptured at 1440/390; no horizontal overflow.
- Limitations: 100-sample subset is not full-test or statistically conclusive; timings vary between runs; the selected subset contained 0 all-unknown samples, so exclusion is covered by synthetic tests and the `eval_00_000009` live case rather than by this subset.

## Non-negotiable rules
1. Inspect repository and artifacts before coding.
2. Never invent paths, checkpoint names, feature values, normalization constants, or metadata.
3. Do not retrain models unless explicitly requested.
4. Do not alter architecture to force checkpoint loading; use the architecture compatible with the checkpoints.
5. Do not silently change feature order or preprocessing.
6. Do not infer date, geography, event, or chronology from sample IDs.
7. Exclude target `-1` pixels from metrics; never count them as non-fire.
8. Map unknown previous-mask inputs to 0 only if that matches training.
9. Never tune thresholds on test data.
10. Keep dataset/checkpoints out of Git.
11. Avoid database, auth, Docker, cloud deployment, and unrelated infrastructure unless justified and approved.
12. Make small, testable changes; do not rewrite working components wholesale.
13. Run tests and report actual results. Never claim unrun tests passed.
14. If a required artifact is missing, stop and report exactly what is missing. Never substitute a mock model as real inference.
15. Keep partial model failures explicit.
16. Trace every result to sample ID, checkpoint, preprocessing version, and threshold.

## OpenCode protocol
First read all docs, inspect repository/environment/artifacts, then produce an artifact inventory with exact paths and sizes. Report blockers before implementation. Follow phases in order. At each milestone report changed files, commands, actual test results, remaining blockers, and tracker status.

## Definition of done
A user selects a stable sample ID, runs all three actual trained checkpoints, views comparable prediction masks, and sees correctly masked metrics where ground truth is valid. Results are reproducible and clearly labeled as research outputs, not operational forecasts.
