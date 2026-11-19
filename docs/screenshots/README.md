# Screenshots

Captured in Chrome (agent-browser) against the production bundle served at
`http://127.0.0.1:5173` with the real FastAPI service at
`http://127.0.0.1:8000`. Recaptured after the readable-type pass, evaluation tab,
`>100` catalog fire filter, and instrument-detail refinement (mask rulers,
synchronized guides, Start here briefing, evaluation headline).

| File | State |
|---|---|
| `01-empty.png` | Inspector landing state with the “Start here” briefing and visible/indexed counts |
| `02-preview.png` | `eval_00_000014` selected via filtered search; masks include 0–64 coordinate rulers |
| `03-binary-inspector.png` | Three models run; synchronized crosshair and cell box at r35 c32 |
| `04-probability.png` | Shared 0–1 probability heatmaps |
| `05-consensus.png` | Per-pixel vote map, unanimous vs split counts, per-model fire counts |
| `06-filter-boundary.png` | Search for `train_00_000844`: exactly 101 fire pixels, included by the `>100` catalog filter |
| `07-mobile.png` | 390×844 inspector view after running predictions |
| `08-offline.png` | Backend stopped; API offline, empty catalog, retry affordance |
| `09-evaluation.png` | Evaluation tab: overall micro/macro tables, rankings, charts, confusion matrices, dynamic summary, metric definitions |
| `10-evaluation-mobile.png` | Evaluation tab at 390×844; tables scroll horizontally |

The catalog now hides samples with `target_fire_pixels <= 100` in all four
tabs. All-unknown samples (0 fire pixels) are therefore not visible in the UI;
they remain in the manifest and are still reachable through the API and
`eval_00_000009` tests. Full catalogs are available by omitting
`min_fire_pixels`.

Reproduce:

```bash
# terminal 1 (repository root)
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

# terminal 2
cd frontend
npm run build
npm run preview
```

The evaluation tab requires generated results:

```bash
python backend/tools/run_evaluation.py
```

Data: `eval_00_000014` (319 fire pixels, all targets valid),
`train_00_000844` (exactly 101 fire pixels), and the fixed-seed 100-sample test
subset. `eval_00_000009` (all-unknown) remains available through the API but is
outside the filtered catalog. These are research outputs, not operational
forecasts.
