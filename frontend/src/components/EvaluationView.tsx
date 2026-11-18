import { useCallback, useEffect, useState } from "react";

import { api, ApiError } from "../api/client";
import { count, metric } from "../format";
import type {
  EvaluationModel,
  EvaluationRanking,
  EvaluationSummary,
  ModelName,
} from "../types";
import { SectionLabel, StatusPill } from "./ui";

const MODEL_IDS: ModelName[] = ["M1", "M2", "M3"];

const RANKING_LABELS: Array<[string, string, "score" | "count"]> = [
  ["best_f1_dice", "Best F1 / Dice", "score"],
  ["best_iou", "Best IoU", "score"],
  ["best_precision", "Best precision", "score"],
  ["best_recall", "Best recall", "score"],
  ["best_accuracy", "Best accuracy", "score"],
  ["most_predicted_fire", "Most predicted fire", "count"],
  ["fewest_predicted_fire", "Fewest predicted fire", "count"],
  ["most_true_positives", "Most actual fire detected", "count"],
  ["most_false_positives", "Most false alarms", "count"],
];

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "The evaluation summary could not be loaded.";
}

function joinModels(models: ModelName[]): string {
  if (models.length <= 1) return models[0] ?? "—";
  if (models.length === 2) return `${models[0]} + ${models[1]}`;
  return `${models.slice(0, -1).join(", ")} + ${models[models.length - 1]}`;
}

function rankingValue(ranking: EvaluationRanking, kind: "score" | "count"): string {
  const value = ranking.values[ranking.winners[0]];
  return kind === "count" ? count(value) : metric(value);
}

function BarChart({
  title,
  note,
  values,
  format = metric,
}: {
  title: string;
  note: string;
  values: Partial<Record<ModelName, number>>;
  format?: (value: number | null | undefined) => string;
}) {
  const max = Math.max(1e-9, ...MODEL_IDS.map((id) => values[id] ?? 0));
  return (
    <figure className="eval-chart">
      <figcaption>
        <span className="fig-title">{title}</span>
        <span className="fig-note">{note}</span>
      </figcaption>
      <div className="bar-list">
        {MODEL_IDS.map((modelId) => {
          const value = values[modelId] ?? 0;
          return (
            <div className="bar-row" key={modelId}>
              <span className="bar-label mono">{modelId}</span>
              <span className="bar-track" aria-hidden="true">
                <span style={{ width: `${(value / max) * 100}%` }} />
              </span>
              <span className="bar-value mono">{format(value)}</span>
            </div>
          );
        })}
      </div>
    </figure>
  );
}

function PrecisionRecallChart({ models }: { models: Record<ModelName, EvaluationModel> }) {
  const width = 340;
  const height = 270;
  const padLeft = 46;
  const padRight = 18;
  const padTop = 18;
  const padBottom = 42;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;
  const x = (value: number) => padLeft + value * plotWidth;
  const y = (value: number) => padTop + (1 - value) * plotHeight;

  return (
    <figure className="eval-chart is-square">
      <figcaption>
        <span className="fig-title">Precision vs recall</span>
        <span className="fig-note">micro · 0–1</span>
      </figcaption>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={MODEL_IDS.map(
          (id) =>
            `${id}: precision ${models[id].micro.precision.toFixed(3)}, recall ${models[id].micro.recall.toFixed(3)}`,
        ).join("; ")}
        className="scatter"
      >
        <rect
          x={padLeft}
          y={padTop}
          width={plotWidth}
          height={plotHeight}
          className="scatter-plot"
        />
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line
              x1={x(tick)}
              x2={x(tick)}
              y1={padTop}
              y2={padTop + plotHeight}
              className="scatter-grid"
            />
            <line
              x1={padLeft}
              x2={padLeft + plotWidth}
              y1={y(tick)}
              y2={y(tick)}
              className="scatter-grid"
            />
            <text x={x(tick)} y={padTop + plotHeight + 14} textAnchor="middle">
              {tick}
            </text>
            <text x={padLeft - 7} y={y(tick) + 3} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        <text
          x={padLeft + plotWidth / 2}
          y={height - 8}
          textAnchor="middle"
          className="scatter-axis"
        >
          Recall
        </text>
        <text
          x={12}
          y={padTop + plotHeight / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${padTop + plotHeight / 2})`}
          className="scatter-axis"
        >
          Precision
        </text>
        {MODEL_IDS.map((modelId) => {
          const point = models[modelId].micro;
          const offsets: Record<ModelName, [number, number]> = {
            M1: [-28, 16],
            M2: [8, -8],
            M3: [8, 14],
          };
          const [dx, dy] = offsets[modelId];
          const labelX = Math.min(
            padLeft + plotWidth - 10,
            Math.max(padLeft + 4, x(point.recall) + dx),
          );
          const labelY = Math.min(
            padTop + plotHeight - 4,
            Math.max(padTop + 10, y(point.precision) + dy),
          );
          return (
            <g key={modelId}>
              <circle cx={x(point.recall)} cy={y(point.precision)} r={5} />
              <text x={labelX} y={labelY} className="scatter-label">
                {modelId}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

function ConfusionMatrix({
  modelId,
  model,
}: {
  modelId: ModelName;
  model: EvaluationModel;
}) {
  const { counts } = model;
  return (
    <figure className="confusion">
      <figcaption>
        <span className="fig-title">{modelId}</span>
        <span className="fig-note">predicted vs actual</span>
      </figcaption>
      <table>
        <thead>
          <tr>
            <td />
            <th scope="col">Actual fire</th>
            <th scope="col">Actual no-fire</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Pred fire</th>
            <td className="is-tp">{count(counts.true_positive)}</td>
            <td className="is-fp">{count(counts.false_positive)}</td>
          </tr>
          <tr>
            <th scope="row">Pred no-fire</th>
            <td className="is-fn">{count(counts.false_negative)}</td>
            <td className="is-tn">{count(counts.true_negative)}</td>
          </tr>
        </tbody>
      </table>
    </figure>
  );
}

export function EvaluationView() {
  const [data, setData] = useState<EvaluationSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.evaluation());
    } catch (err) {
      setData(null);
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading && !data) {
    return (
      <section className="evaluation" aria-busy="true">
        <p className="loading-note">Loading evaluation summary…</p>
      </section>
    );
  }

  if (!data) {
    return (
      <section className="evaluation">
        <div className="banner is-error" role="alert">
          <strong>Evaluation summary unavailable</strong>
          <p>{error}</p>
          <p className="fineprint">
            Generate it with: python backend/tools/run_evaluation.py
          </p>
          <button type="button" className="text-button" onClick={() => void load()}>
            Retry
          </button>
        </div>
      </section>
    );
  }

  const { counts, models, rankings, sampling } = data;

  return (
    <section className="evaluation" aria-labelledby="evaluation-heading">
      <header className="evaluation-head">
        <div>
          <p className="eyebrow">
            Evaluation / {data.evaluation_version} / {data.split} split
          </p>
          <h1 id="evaluation-heading">Model evaluation</h1>
          <p className="lede">
            Same {counts.selected} samples, same preprocessing, same recorded
            thresholds. Metrics use valid target pixels only.
          </p>
        </div>
        <StatusPill tone="warn">100-sample subset</StatusPill>
      </header>

      <div className="eval-integrity">
        <p>
          <strong>Fixed-seed subset.</strong> {counts.selected} of the test split,
          selected by SHA-256 ranking with seed {sampling.seed}. {counts.valid_samples}{" "}
          samples have valid targets; {counts.all_unknown_samples} all-unknown
          sample{counts.all_unknown_samples === 1 ? "" : "s"} excluded from metrics.
          This is not a full test-set evaluation and is not a statistical claim.
        </p>
        <p className="fineprint">
          Sampling: {sampling.strategy}. Total valid pixels:{" "}
          {count(counts.valid_pixels)}. Actual fire pixels:{" "}
          {count(counts.actual_fire_pixels)}. Thresholds: M1{" "}
          {data.thresholds.M1.toFixed(2)} · M2 {data.thresholds.M2.toFixed(2)} · M3{" "}
          {data.thresholds.M3.toFixed(2)}.
        </p>
      </div>

      <section className="panel" aria-labelledby="eval-overall">
        <div className="section-head">
          <SectionLabel index="01">
            <span id="eval-overall">Overall comparison</span>
          </SectionLabel>
          <span className="fig-note">micro / global · counts before dividing</span>
        </div>

        <p className="table-caption">
          Micro metrics aggregate all valid pixels across the subset before
          computing ratios.
        </p>
        <div
          className="table-scroll"
          role="region"
          aria-label="Micro metrics comparison, scroll horizontally"
          tabIndex={0}
        >
          <table className="eval-table" aria-label="Micro metrics by model">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Precision</th>
                <th scope="col">Recall</th>
                <th scope="col">F1 / Dice</th>
                <th scope="col">IoU</th>
                <th scope="col">Accuracy</th>
                <th scope="col">TP</th>
                <th scope="col">FP</th>
                <th scope="col">FN</th>
                <th scope="col">TN</th>
                <th scope="col">Pred. fire</th>
              </tr>
            </thead>
            <tbody>
              {MODEL_IDS.map((modelId) => {
                const model = models[modelId];
                return (
                  <tr key={modelId}>
                    <th scope="row">{modelId}</th>
                    <td>{metric(model.micro.precision)}</td>
                    <td>{metric(model.micro.recall)}</td>
                    <td>{metric(model.micro.f1_dice)}</td>
                    <td>{metric(model.micro.iou)}</td>
                    <td>{metric(model.micro.accuracy)}</td>
                    <td>{count(model.counts.true_positive)}</td>
                    <td>{count(model.counts.false_positive)}</td>
                    <td>{count(model.counts.false_negative)}</td>
                    <td>{count(model.counts.true_negative)}</td>
                    <td>{count(model.predicted.total)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <p className="table-caption">
          Macro metrics average per-sample scores; {models.M1.macro.samples} samples
          contribute. All-unknown samples are excluded.
        </p>
        <div
          className="table-scroll eval-macro"
          role="region"
          aria-label="Macro metrics comparison, scroll horizontally"
          tabIndex={0}
        >
          <table className="eval-table" aria-label="Macro metrics by model">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Precision</th>
                <th scope="col">Recall</th>
                <th scope="col">F1 / Dice</th>
                <th scope="col">IoU</th>
                <th scope="col">Accuracy</th>
                <th scope="col">Samples</th>
              </tr>
            </thead>
            <tbody>
              {MODEL_IDS.map((modelId) => {
                const model = models[modelId];
                return (
                  <tr key={modelId}>
                    <th scope="row">{modelId}</th>
                    <td>{metric(model.macro.precision)}</td>
                    <td>{metric(model.macro.recall)}</td>
                    <td>{metric(model.macro.f1_dice)}</td>
                    <td>{metric(model.macro.iou)}</td>
                    <td>{metric(model.macro.accuracy)}</td>
                    <td>{count(model.macro.samples)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel" aria-labelledby="eval-rankings">
        <div className="section-head">
          <SectionLabel index="02">
            <span id="eval-rankings">Rankings</span>
          </SectionLabel>
          <span className="fig-note">winners shown; ties listed</span>
        </div>
        <dl className="ranking-grid">
          {RANKING_LABELS.map(([key, label, kind]) => {
            const ranking = rankings[key];
            return (
              <div key={key} className="ranking-row">
                <dt>{label}</dt>
                <dd>
                  <strong>{joinModels(ranking.winners)}</strong>
                  <span className="mono">{rankingValue(ranking, kind)}</span>
                </dd>
              </div>
            );
          })}
        </dl>
        <p className="fineprint">
          More predicted fire is not better performance. A model can lead on true
          positives while also producing the most false alarms.
        </p>
      </section>

      <section className="panel" aria-labelledby="eval-charts">
        <div className="section-head">
          <SectionLabel index="03">
            <span id="eval-charts">Visual comparison</span>
          </SectionLabel>
          <span className="fig-note">all panels share model identities</span>
        </div>

        <div className="eval-chart-grid">
          <BarChart
            title="F1 / Dice"
            note="higher is better · 0–1"
            values={Object.fromEntries(
              MODEL_IDS.map((id) => [id, models[id].micro.f1_dice]),
            )}
          />
          <BarChart
            title="IoU"
            note="higher is better · 0–1"
            values={Object.fromEntries(
              MODEL_IDS.map((id) => [id, models[id].micro.iou]),
            )}
          />
          <PrecisionRecallChart models={models} />
          <BarChart
            title="Predicted fire pixels"
            note="total across the subset"
            values={Object.fromEntries(
              MODEL_IDS.map((id) => [id, models[id].predicted.total]),
            )}
            format={(value) => count(value)}
          />
        </div>

        <div className="confusion-grid">
          {MODEL_IDS.map((modelId) => (
            <ConfusionMatrix key={modelId} modelId={modelId} model={models[modelId]} />
          ))}
        </div>
      </section>

      <section className="panel" aria-labelledby="eval-summary">
        <div className="section-head">
          <SectionLabel index="04">
            <span id="eval-summary">What the numbers say</span>
          </SectionLabel>
          <span className="fig-note">generated from this evaluation</span>
        </div>
        <div className="eval-summary">
          {data.summary.map((sentence, index) => (
            <p key={index}>{sentence}</p>
          ))}
        </div>
      </section>

      <section className="panel" aria-labelledby="eval-metrics">
        <div className="section-head">
          <SectionLabel index="05">
            <span id="eval-metrics">Metric definitions</span>
          </SectionLabel>
        </div>
        <dl className="metric-definitions">
          <div>
            <dt>Precision</dt>
            <dd>
              <code>TP / (TP + FP)</code>
              <p>
                Of the pixels predicted as fire, how many were actually fire? High
                precision means fewer false alarms; low precision means the model
                over-predicts fire.
              </p>
            </dd>
          </div>
          <div>
            <dt>Recall</dt>
            <dd>
              <code>TP / (TP + FN)</code>
              <p>
                Of the pixels that actually burned, how many did the model detect?
                High recall means fewer missed fires; low recall means the model
                misses real fire.
              </p>
            </dd>
          </div>
          <div>
            <dt>F1 / Dice</dt>
            <dd>
              <code>2TP / (2TP + FP + FN)</code>
              <p>
                The balance between precision and recall. For binary segmentation
                F1 and Dice are mathematically equivalent, so one value is shown.
              </p>
            </dd>
          </div>
          <div>
            <dt>IoU</dt>
            <dd>
              <code>TP / (TP + FP + FN)</code>
              <p>
                Intersection over union: the overlap between predicted and actual
                fire masks. IoU is stricter than Dice when overlap is small.
              </p>
            </dd>
          </div>
          <div>
            <dt>Accuracy</dt>
            <dd>
              <code>(TP + TN) / valid pixels</code>
              <p>
                The fraction of valid pixels classified correctly. Wildfire masks are
                heavily imbalanced, so a high accuracy can coexist with poor fire
                detection — do not read it as a headline score.
              </p>
            </dd>
          </div>
          <div>
            <dt>Unknown pixels</dt>
            <dd>
              <code>FireMask == -1</code>
              <p>
                Excluded from every count and ratio. They are never converted to
                non-fire, and an all-unknown sample contributes no metrics.
              </p>
            </dd>
          </div>
        </dl>
      </section>

      <p className="fineprint">
        {data.threshold_provenance} Source:{" "}
        <span className="mono">artifacts/evaluation/{data.evaluation_version}/</span>.
      </p>
    </section>
  );
}
