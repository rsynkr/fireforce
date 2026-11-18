import { useMemo } from "react";

import { count } from "../format";
import type { HealthResponse, SampleDetail } from "../types";
import { binaryPixels, PixelGrid } from "./MaskCanvas";
import type { Pixel } from "./MaskCanvas";
import { Legend, SectionLabel } from "./ui";

function previousLabel(value: number): string {
  if (value === -1) return "unknown";
  return value === 1 ? "fire" : "non-fire";
}

function targetLabel(value: number, valid: boolean): string {
  if (!valid) return "unknown";
  return value === 1 ? "fire" : "non-fire";
}

interface MasksSectionProps {
  detail: SampleDetail;
  health: HealthResponse | null;
  pixel: Pixel | null;
  onPixel: (pixel: Pixel | null) => void;
}

export function MasksSection({ detail, health, pixel, onPixel }: MasksSectionProps) {
  const previousFire = useMemo(
    () => binaryPixels(detail.previous_fire_mask),
    [detail.previous_fire_mask],
  );
  const target = useMemo(
    () => binaryPixels(detail.target_mask, detail.target_valid_mask),
    [detail.target_mask, detail.target_valid_mask],
  );

  const focus = pixel
    ? {
        previous: previousLabel(detail.previous_fire_mask[pixel.row][pixel.col]),
        target: targetLabel(
          detail.target_mask[pixel.row][pixel.col],
          detail.target_valid_mask[pixel.row][pixel.col],
        ),
      }
    : null;

  return (
    <section className="panel" aria-labelledby="inputs-heading">
      <div className="section-head">
        <SectionLabel index="02">
          <span id="inputs-heading">Inputs &amp; target</span>
        </SectionLabel>
        <Legend mode="binary" />
      </div>

      <div className="tri-grid">
        <figure className="figure">
          <figcaption>
            <span className="fig-title">Previous fire</span>
            <span className="fig-note">PrevFireMask</span>
          </figcaption>
          <PixelGrid
            pixels={previousFire}
            label={`Previous-fire mask: ${count(detail.target_valid_pixels)} valid pixels in the target. Unknown previous-fire pixels are fed to the models as no-fire.`}
            hovered={pixel}
            onHover={onPixel}
          />
        </figure>

        <figure className="figure">
          <figcaption>
            <span className="fig-title">Ground truth · next day</span>
            <span className="fig-note">FireMask</span>
          </figcaption>
          <PixelGrid
            pixels={target}
            label={`Ground-truth mask: ${count(detail.target_fire_pixels)} fire pixels, ${count(detail.target_unknown_pixels)} unknown pixels.`}
            hovered={pixel}
            onHover={onPixel}
          />
        </figure>

        <div className="facts">
          <h3>Sample facts</h3>
          <dl className="stat-grid">
            <div>
              <dt>Valid target</dt>
              <dd>{count(detail.target_valid_pixels)}</dd>
            </div>
            <div>
              <dt>Fire pixels</dt>
              <dd>{count(detail.target_fire_pixels)}</dd>
            </div>
            <div>
              <dt>Unknown</dt>
              <dd>{count(detail.target_unknown_pixels)}</dd>
            </div>
            <div>
              <dt>Record</dt>
              <dd>{detail.record_index}</dd>
            </div>
          </dl>

          <h4>Input features · {detail.available_features.length}</h4>
          <p className="feature-list">{detail.available_features.join(" · ")}</p>

          <h4>Checkpoints</h4>
          <ul className="model-availability">
            {detail.models.map((modelId) => {
              const model = health?.models[modelId];
              return (
                <li key={modelId}>
                  <span className="mono">{modelId}</span>
                  <span className="muted">
                    {model ? `${model.in_channels} ch` : "—"}
                  </span>
                  <span className={model?.status === "ready" ? "state is-ready" : "state"}>
                    {model ? model.status : "unknown"}
                  </span>
                </li>
              );
            })}
          </ul>

          <p className="fineprint">
            Every mask is 64×64, rendered nearest-neighbour at the same scale.
          </p>
        </div>
      </div>

      <div className="pixel-readout" aria-live="polite">
        {pixel && focus ? (
          <>
            <span className="readout-pos">
              r{String(pixel.row).padStart(2, "0")} c{String(pixel.col).padStart(2, "0")}
            </span>
            <span>Previous: <b>{focus.previous}</b></span>
            <span>Target: <b>{focus.target}</b></span>
          </>
        ) : (
          <span className="muted">
            Hover any mask to inspect the same pixel across inputs and models.
          </span>
        )}
      </div>
    </section>
  );
}
