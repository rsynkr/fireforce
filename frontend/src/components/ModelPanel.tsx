import { useMemo } from "react";

import { count, metric, ms, shortHash } from "../format";
import type { GroundTruth, ModelPrediction } from "../types";
import {
  binaryPixels,
  PixelGrid,
  probabilityPixels,
} from "./MaskCanvas";
import type { Pixel } from "./MaskCanvas";
import { StatusPill } from "./ui";

export type ComparisonView = "binary" | "probability";

function probabilityRange(probabilities: number[][]): [number, number] {
  let min = 1;
  let max = 0;
  for (const row of probabilities) {
    for (const value of row) {
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  return [min, max];
}

interface ModelPanelProps {
  prediction: ModelPrediction;
  view: ComparisonView;
  pixel: Pixel | null;
  onPixel: (pixel: Pixel | null) => void;
  groundTruth: GroundTruth;
}

export function ModelPanel({
  prediction,
  view,
  pixel,
  onPixel,
  groundTruth,
}: ModelPanelProps) {
  const available = prediction.status === "ok";
  const binary = useMemo(
    () => (prediction.mask ? binaryPixels(prediction.mask) : null),
    [prediction.mask],
  );
  const probability = useMemo(
    () =>
      prediction.probabilities ? probabilityPixels(prediction.probabilities) : null,
    [prediction.probabilities],
  );
  const pixels = available ? (view === "probability" ? probability : binary) : null;
  const range = useMemo(
    () =>
      prediction.probabilities
        ? probabilityRange(prediction.probabilities)
        : ([0, 0] as [number, number]),
    [prediction.probabilities],
  );

  const isKnown = pixel ? groundTruth.valid_mask[pixel.row]?.[pixel.col] : false;
  const targetIsFire = pixel
    ? groundTruth.target_mask[pixel.row]?.[pixel.col] === 1
    : false;
  const predictedIsFire = pixel
    ? prediction.mask?.[pixel.row]?.[pixel.col] === true
    : false;
  const agreement = pixel && isKnown
    ? predictedIsFire === targetIsFire
      ? "hit"
      : "miss"
    : null;

  const statusTone =
    prediction.status === "ok"
      ? "good"
      : prediction.status === "unavailable"
        ? "neutral"
        : "warn";

  return (
    <article className={`model-panel is-${prediction.status}`} data-testid={`model-${prediction.model_id}`}>
      <header className="model-head">
        <div className="model-title">
          <h3>{prediction.model_id}</h3>
          <span className="model-channels">{prediction.checkpoint.in_channels} ch</span>
        </div>
        <StatusPill tone={statusTone}>{prediction.status}</StatusPill>
      </header>

      {available ? (
        <>
          <figure className="figure model-figure">
            <PixelGrid
              pixels={pixels}
              label={`${prediction.model_id} ${
                view === "probability" ? "probability heatmap" : "thresholded fire mask"
              } at threshold ${prediction.threshold}. Predicted fire pixels: ${
                prediction.predicted_fire_pixels ?? 0
              }.`}
              hovered={pixel}
              onHover={onPixel}
            />
            <figcaption className="model-caption">
              {view === "probability" ? (
                <>
                  <span>
                    p {range[0].toFixed(2)}–{range[1].toFixed(2)}
                  </span>
                  <span>shared 0–1 scale</span>
                </>
              ) : (
                <>
                  <span>{count(prediction.predicted_fire_pixels)} fire px</span>
                  <span>t ≥ {prediction.threshold.toFixed(2)}</span>
                </>
              )}
            </figcaption>
          </figure>

          <div className="model-meta">
            <span title={prediction.threshold_provenance}>
              threshold <b>{prediction.threshold.toFixed(2)}</b>
            </span>
            <span>
              inference <b>{ms(prediction.inference_ms)}</b>
            </span>
            <span title={prediction.checkpoint.sha256 ?? undefined}>
              ckpt <b>#{shortHash(prediction.checkpoint.sha256)}</b>
            </span>
          </div>

          {prediction.metrics ? (
            prediction.metrics.available ? (
              <>
                <dl className="metric-grid">
                  <div>
                    <dt>
                      <abbr title="2TP / (2TP + FP + FN)">Dice</abbr>
                    </dt>
                    <dd>{metric(prediction.metrics.dice)}</dd>
                  </div>
                  <div>
                    <dt>
                      <abbr title="TP / (TP + FP + FN)">IoU</abbr>
                    </dt>
                    <dd>{metric(prediction.metrics.iou)}</dd>
                  </div>
                  <div>
                    <dt>
                      <abbr title="TP / (TP + FP)">Precision</abbr>
                    </dt>
                    <dd>{metric(prediction.metrics.precision)}</dd>
                  </div>
                  <div>
                    <dt>
                      <abbr title="TP / (TP + FN)">Recall</abbr>
                    </dt>
                    <dd>{metric(prediction.metrics.recall)}</dd>
                  </div>
                </dl>
                <p className="metric-counts">
                  TP {count(prediction.metrics.true_positive)} · FP{" "}
                  {count(prediction.metrics.false_positive)} · FN{" "}
                  {count(prediction.metrics.false_negative)} · TN{" "}
                  {count(prediction.metrics.true_negative)}
                </p>
              </>
            ) : (
              <p className="metrics-unavailable">
                Metrics unavailable — no valid target pixels in this sample.
              </p>
            )
          ) : null}

          {pixel ? (
            <p className="model-pixel">
              r{pixel.row} c{pixel.col} ·{" "}
              {view === "probability"
                ? prediction.probabilities?.[pixel.row]?.[pixel.col]?.toFixed(3)
                : predictedIsFire
                  ? "fire"
                  : "non-fire"}
              {agreement ? <span className={`agreement is-${agreement}`}> {agreement}</span> : null}
            </p>
          ) : null}
        </>
      ) : (
        <div className="model-failure" role="status">
          <p>{prediction.error ?? "This checkpoint is not available."}</p>
          <p className="fineprint">
            Other models keep their results; nothing is substituted for this one.
          </p>
        </div>
      )}
    </article>
  );
}
