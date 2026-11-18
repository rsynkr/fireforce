# Screenshots

Captured in Chrome (agent-browser) against the production bundle served at
`http://127.0.0.1:5173` with the real FastAPI service at
`http://127.0.0.1:8000`. Recaptured after the readable-type pass and the
evaluation tab were added.

| File | State |
|---|---|
| `01-empty.png` | Inspector landing state, 1440×1000, catalog loaded |
| `02-preview.png` | `eval_00_000027` selected, masks loaded, predictions not run |
| `03-binary-inspector.png` | Three models run; synchronized inspector at r33 c30 (M1 miss, M2/M3 hit) |
| `04-probability.png` | Shared 0–1 probability heatmaps |
| `05-consensus.png` | Per-pixel vote map, unanimous vs split counts, per-model fire counts |
| `06-all-unknown.png` | `eval_00_000009`; ground truth fully unknown, metrics unavailable |
| `07-mobile.png` | 390×844 inspector view after running predictions |
| `08-offline.png` | Backend stopped; API offline, empty catalog, retry affordance |
| `09-evaluation.png` | Evaluation tab: overall micro/macro tables, rankings, charts, confusion matrices, dynamic summary, metric definitions |
| `10-evaluation-mobile.png` | Evaluation tab at 390×844; tables scroll horizontally |

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

Data: `eval_00_000027` (21 fire pixels, all targets valid),
`eval_00_000009` (all 4,096 target pixels unknown), and the fixed-seed
100-sample test subset. These are research outputs, not operational forecasts.
