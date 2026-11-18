import { useMemo } from "react";

import { count, ms } from "../format";
import type {
  PredictionResponse,
  SampleDetail,
} from "../types";
import {
  consensusPixels,
  PixelGrid,
} from "./MaskCanvas";
import type { Pixel } from "./MaskCanvas";
import { ModelPanel } from "./ModelPanel";
import type { ComparisonView } from "./ModelPanel";
import { Legend, SectionLabel, StatusPill } from "./ui";

export type ViewMode = ComparisonView | "consensus";

interface ComparisonProps {
  detail: SampleDetail;
  prediction: PredictionResponse | null;
  running: boolean;
  error: string | null;
  view: ViewMode;
  onView: (view: ViewMode) => void;
  pixel: Pixel | null;
  onPixel: (pixel: Pixel | null) => void;
  onRun: () => void;
}

function targetAt(prediction: PredictionResponse, pixel: Pixel): "fire" | "non-fire" | "unknown" {
  const valid = prediction.ground_truth.valid_mask[pixel.row]?.[pixel.col];
  if (!valid) return "unknown";
  return prediction.ground_truth.target_mask[pixel.row]?.[pixel.col] === 1
    ? "fire"
    : "non-fire";
}

function PixelReadout({
  prediction,
  pixel,
  view,
}: {
  prediction: PredictionResponse | null;
  pixel: Pixel | null;
  view: ViewMode;
}) {
  if (!pixel) {
    return (
      <div className="pixel-readout" aria-live="polite">
        <span className="muted">
          Hover any mask to compare models at the same pixel. Hits and misses are
          relative to the ground truth where it is known.
        </span>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="pixel-readout" aria-live="polite">
        <span className="readout-pos">
          r{String(pixel.row).padStart(2, "0")} c{String(pixel.col).padStart(2, "0")}
        </span>
        <span className="muted">Run predictions to compare models at this pixel.</span>
      </div>
    );
  }

  const target = targetAt(prediction, pixel);

  return (
    <div className="pixel-readout" aria-live="polite">
      <span className="readout-pos">
        r{String(pixel.row).padStart(2, "0")} c{String(pixel.col).padStart(2, "0")}
      </span>
      <span>
        Target <b>{target}</b>
      </span>
      {prediction.models.map((model) => {
        if (model.status !== "ok") {
          return (
            <span key={model.model_id} className="readout-model">
              {model.model_id} <b>—</b>
            </span>
          );
        }
        if (view === "probability") {
          return (
            <span key={model.model_id} className="readout-model">
              {model.model_id}{" "}
              <b>{model.probabilities?.[pixel.row]?.[pixel.col]?.toFixed(3) ?? "—"}</b>
            </span>
          );
        }
        const predicted = model.mask?.[pixel.row]?.[pixel.col] === true;
        const label = predicted ? "fire" : "non-fire";
        const known = target !== "unknown";
        const hit = known ? (predicted ? target === "fire" : target === "non-fire") : null;
        return (
          <span key={model.model_id} className="readout-model">
            {model.model_id} <b>{label}</b>
            {hit !== null ? (
              <span className={hit ? "agreement is-hit" : "agreement is-miss"}>
                {hit ? "hit" : "miss"}
              </span>
            ) : null}
          </span>
        );
      })}
      {view === "consensus" ? (
        <span className="readout-model">
          votes{" "}
          <b>
            {prediction.models.filter(
              (model) =>
                model.status === "ok" && model.mask?.[pixel.row]?.[pixel.col] === true,
            ).length}
          </b>
        </span>
      ) : null}
    </div>
  );
}

function ConsensusPanel({
  prediction,
  pixel,
  onPixel,
}: {
  prediction: PredictionResponse;
  pixel: Pixel | null;
  onPixel: (pixel: Pixel | null) => void;
}) {
  const models = prediction.models.filter(
    (model) => model.status === "ok" && model.mask,
  );
  const maxVotes = models.length;

  const { counts, pixels } = useMemo(() => {
    const voteMap: number[][] = Array.from({ length: 64 }, () =>
      Array.from({ length: 64 }, () => 0),
    );
    const distribution = Array.from({ length: maxVotes + 1 }, () => 0);
    for (const model of models) {
      for (let row = 0; row < 64; row += 1) {
        for (let col = 0; col < 64; col += 1) {
          if (model.mask?.[row]?.[col]) voteMap[row][col] += 1;
        }
      }
    }
    for (const row of voteMap) {
      for (const value of row) distribution[value] += 1;
    }
    return {
      votes: voteMap,
      counts: distribution,
      pixels: consensusPixels(voteMap, maxVotes),
    };
  }, [models, maxVotes]);

  if (maxVotes < 2) {
    return (
      <div className="empty-comparison">
        <p className="empty-kicker">Consensus unavailable</p>
        <p>At least two successful models are required to compare agreement.</p>
      </div>
    );
  }

  const unanimous = counts[0] + counts[maxVotes];
  const split = 4096 - unanimous;

  return (
    <div className="consensus">
      <figure className="figure consensus-figure">
        <figcaption>
          <span className="fig-title">Consensus · {maxVotes} models</span>
          <span className="fig-note">votes per pixel</span>
        </figcaption>
        <PixelGrid
          pixels={pixels}
          label={`Consensus map across ${maxVotes} models: ${count(unanimous)} pixels with unanimous agreement, ${count(split)} split.`}
          hovered={pixel}
          onHover={onPixel}
        />
      </figure>

      <div className="consensus-side">
        <dl className="stat-grid">
          <div>
            <dt>Unanimous</dt>
            <dd>{count(unanimous)}</dd>
            <small>{((unanimous / 4096) * 100).toFixed(1)}% of pixels</small>
          </div>
          <div>
            <dt>Split vote</dt>
            <dd>{count(split)}</dd>
            <small>{((split / 4096) * 100).toFixed(1)}% of pixels</small>
          </div>
        </dl>

        <ul className="vote-list">
          {counts.map((value, votesCount) => (
            <li key={votesCount}>
              <span className="mono">{votesCount}</span>
              <span className={`vote-bar vote-${votesCount}`} aria-hidden="true">
                <span style={{ width: `${(value / 4096) * 100}%` }} />
              </span>
              <span className="mono">{count(value)}</span>
            </li>
          ))}
        </ul>

        <h4>Fire pixels by model</h4>
        <ul className="model-fire-counts">
          {models.map((model) => (
            <li key={model.model_id}>
              <span className="mono">{model.model_id}</span>
              <span>{count(model.predicted_fire_pixels)} px</span>
            </li>
          ))}
        </ul>

        <p className="fineprint">
          Votes count all 4,096 pixels. Metrics below remain masked to valid target
          pixels only.
        </p>
      </div>
    </div>
  );
}

export function Comparison({
  detail,
  prediction,
  running,
  error,
  view,
  onView,
  pixel,
  onPixel,
  onRun,
}: ComparisonProps) {
  const successful = prediction
    ? prediction.models.filter((model) => model.status === "ok").length
    : 0;

  return (
    <section className="panel" aria-labelledby="comparison-heading">
      <div className="section-head comparison-head">
        <SectionLabel index="03">
          <span id="comparison-heading">Comparison</span>
        </SectionLabel>

        <div className="comparison-tools">
          <div className="segmented" role="group" aria-label="Comparison view">
            {(
              [
                ["binary", "Binary"],
                ["probability", "Probability"],
                ["consensus", "Consensus"],
              ] as const
            ).map(([value, label]) => (
              <button
                key={value}
                type="button"
                className={view === value ? "is-active" : undefined}
                aria-pressed={view === value}
                onClick={() => onView(value)}
              >
                {label}
              </button>
            ))}
          </div>
          <Legend
            mode={view === "consensus" ? "consensus" : view}
            maxVotes={prediction ? successful || 1 : 3}
          />
        </div>
      </div>

      {prediction ? (
        <div className="run-strip">
          <span>
            run <b>{prediction.status}</b>
          </span>
          <span>
            device <b>{prediction.device}</b>
          </span>
          <span>
            total <b>{ms(prediction.duration_ms)}</b>
          </span>
          <span>
            sample <b>{prediction.sample_id}</b>
          </span>
        </div>
      ) : null}

      {error ? (
        <div className="banner is-error" role="alert">
          <strong>Prediction failed</strong>
          <p>{error}</p>
        </div>
      ) : null}

      {running ? (
        <div className="tri-grid model-grid" aria-busy="true">
          {["M1", "M2", "M3"].map((model) => (
            <article className="model-panel is-loading" key={model}>
              <header className="model-head">
                <div className="model-title">
                  <h3>{model}</h3>
                </div>
                <StatusPill tone="neutral">running</StatusPill>
              </header>
              <div className="model-loading">
                <span className="loading-block" aria-hidden="true" />
                <p>Running {model} on this sample…</p>
              </div>
            </article>
          ))}
        </div>
      ) : prediction ? (
        view === "consensus" ? (
          <ConsensusPanel prediction={prediction} pixel={pixel} onPixel={onPixel} />
        ) : (
          <div className="tri-grid model-grid">
            {prediction.models.map((model) => (
              <ModelPanel
                key={model.model_id}
                prediction={model}
                view={view}
                pixel={pixel}
                onPixel={onPixel}
                groundTruth={prediction.ground_truth}
              />
            ))}
          </div>
        )
      ) : (
        <div className="empty-comparison">
          <p className="empty-kicker">Not run</p>
          <p>
            Run M1, M2 and M3 on <span className="mono">{detail.sample_id}</span>.
            Inference is computed on demand and never cached.
          </p>
          <button type="button" className="button is-primary" onClick={onRun}>
            Run predictions
          </button>
        </div>
      )}

      <PixelReadout prediction={prediction} pixel={pixel} view={view} />
    </section>
  );
}
