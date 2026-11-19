import { useEffect, useRef } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

export interface Pixel {
  row: number;
  col: number;
}

const FIRE: [number, number, number, number] = [240, 90, 40, 255];
const NON_FIRE: [number, number, number, number] = [27, 30, 25, 255];
const UNKNOWN_A: [number, number, number, number] = [92, 97, 87, 255];
const UNKNOWN_B: [number, number, number, number] = [66, 71, 62, 255];

const PROBABILITY_STOPS: Array<[number, [number, number, number]]> = [
  [0, [20, 23, 19]],
  [0.25, [74, 47, 35]],
  [0.5, [140, 61, 34]],
  [0.75, [209, 80, 39]],
  [1, [255, 193, 94]],
];

function probabilityColor(value: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, value));
  for (let i = 1; i < PROBABILITY_STOPS.length; i += 1) {
    const [upperT, upper] = PROBABILITY_STOPS[i];
    const [lowerT, lower] = PROBABILITY_STOPS[i - 1];
    if (clamped <= upperT) {
      const t = (clamped - lowerT) / (upperT - lowerT);
      return [
        Math.round(lower[0] + (upper[0] - lower[0]) * t),
        Math.round(lower[1] + (upper[1] - lower[1]) * t),
        Math.round(lower[2] + (upper[2] - lower[2]) * t),
      ];
    }
  }
  return PROBABILITY_STOPS[PROBABILITY_STOPS.length - 1][1];
}

function consensusColor(votes: number, maxVotes: number): [number, number, number, number] {
  if (votes <= 0) return NON_FIRE;
  if (votes >= maxVotes) return FIRE;
  const t = votes / maxVotes;
  return [
    Math.round(74 + (209 - 74) * (t - 0.2) * 1.4),
    Math.round(47 + (80 - 47) * (t - 0.2) * 1.4),
    Math.round(35 + (39 - 35) * (t - 0.2) * 1.4),
    255,
  ];
}

function setPixel(
  buffer: Uint8ClampedArray,
  row: number,
  col: number,
  color: [number, number, number, number],
) {
  const index = (row * 64 + col) * 4;
  buffer[index] = color[0];
  buffer[index + 1] = color[1];
  buffer[index + 2] = color[2];
  buffer[index + 3] = color[3];
}

export function binaryPixels(
  mask: number[][] | boolean[][],
  validMask?: boolean[][],
  unknownValue = -1,
): Uint8ClampedArray {
  const buffer = new Uint8ClampedArray(64 * 64 * 4);
  for (let row = 0; row < 64; row += 1) {
    for (let col = 0; col < 64; col += 1) {
      const value = mask[row]?.[col];
      const known = validMask ? validMask[row]?.[col] !== false : value !== unknownValue;
      if (!known || value === unknownValue) {
        setPixel(buffer, row, col, (row + col) % 2 === 0 ? UNKNOWN_A : UNKNOWN_B);
      } else {
        setPixel(buffer, row, col, value === true || value === 1 ? FIRE : NON_FIRE);
      }
    }
  }
  return buffer;
}

export function probabilityPixels(probabilities: number[][]): Uint8ClampedArray {
  const buffer = new Uint8ClampedArray(64 * 64 * 4);
  for (let row = 0; row < 64; row += 1) {
    for (let col = 0; col < 64; col += 1) {
      const [r, g, b] = probabilityColor(probabilities[row]?.[col] ?? 0);
      setPixel(buffer, row, col, [r, g, b, 255]);
    }
  }
  return buffer;
}

export function consensusPixels(votes: number[][], maxVotes: number): Uint8ClampedArray {
  const buffer = new Uint8ClampedArray(64 * 64 * 4);
  for (let row = 0; row < 64; row += 1) {
    for (let col = 0; col < 64; col += 1) {
      setPixel(buffer, row, col, consensusColor(votes[row]?.[col] ?? 0, maxVotes));
    }
  }
  return buffer;
}

const RULER_TICKS = [0, 16, 32, 48, 64];

interface PixelGridProps {
  pixels: Uint8ClampedArray | null;
  label: string;
  className?: string;
  hovered?: Pixel | null;
  onHover?: (pixel: Pixel | null) => void;
  emptyLabel?: string;
}

export function PixelGrid({
  pixels,
  label,
  className,
  hovered,
  onHover,
  emptyLabel = "No data",
}: PixelGridProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !pixels) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const image = new ImageData(new Uint8ClampedArray(pixels), 64, 64);
    context.putImageData(image, 0, 0);
  }, [pixels]);

  function handlePointer(event: ReactPointerEvent<HTMLDivElement>) {
    if (!onHover || !pixels) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const col = Math.min(63, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * 64)));
    const row = Math.min(63, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * 64)));
    onHover({ row, col });
  }

  return (
    <div className={`pixel-frame${className ? ` ${className}` : ""}`}>
      <div className="pixel-ruler is-top" aria-hidden="true">
        {RULER_TICKS.map((tick) => (
          <span key={tick} style={{ left: `${(tick / 64) * 100}%` }}>
            {tick}
          </span>
        ))}
      </div>
      <div className="pixel-ruler is-left" aria-hidden="true">
        {RULER_TICKS.map((tick) => (
          <span key={tick} style={{ top: `${(tick / 64) * 100}%` }}>
            {tick}
          </span>
        ))}
      </div>
      <div
        className={`pixel-grid${pixels ? "" : " is-empty"}`}
        role="img"
        aria-label={label}
        onPointerMove={pixels ? handlePointer : undefined}
        onPointerDown={pixels ? handlePointer : undefined}
        onPointerLeave={pixels && onHover ? () => onHover(null) : undefined}
        data-testid="pixel-grid"
      >
        {pixels ? (
          <>
            <canvas ref={canvasRef} width={64} height={64} aria-hidden="true" />
            {hovered ? (
              <>
                <span
                  aria-hidden="true"
                  className="pixel-guide is-horizontal"
                  style={{ top: `${((hovered.row + 0.5) / 64) * 100}%` }}
                />
                <span
                  aria-hidden="true"
                  className="pixel-guide is-vertical"
                  style={{ left: `${((hovered.col + 0.5) / 64) * 100}%` }}
                />
                <span
                  aria-hidden="true"
                  className="pixel-cursor"
                  style={{
                    left: `${(hovered.col / 64) * 100}%`,
                    top: `${(hovered.row / 64) * 100}%`,
                  }}
                />
              </>
            ) : null}
          </>
        ) : (
          <span className="pixel-empty">{emptyLabel}</span>
        )}
      </div>
    </div>
  );
}
