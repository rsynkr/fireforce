import type { ReactNode } from "react";

const MARK_CELLS = [
  1, 0, 0, 0,
  1, 1, 0, 0,
  0, 1, 1, 0,
  0, 0, 1, 1,
];

export function Mark() {
  return (
    <span className="mark" aria-hidden="true">
      {MARK_CELLS.map((filled, index) => (
        <span key={index} className={filled ? "mark-cell is-fire" : "mark-cell"} />
      ))}
    </span>
  );
}

export function SectionLabel({
  index,
  children,
}: {
  index: string;
  children: ReactNode;
}) {
  return (
    <div className="section-label">
      <span className="section-index">{index}</span>
      <span className="section-rule" aria-hidden="true" />
      <h2>{children}</h2>
    </div>
  );
}

export function StatusPill({
  tone,
  children,
}: {
  tone: "neutral" | "accent" | "good" | "warn";
  children: ReactNode;
}) {
  return <span className={`status-pill is-${tone}`}>{children}</span>;
}

export function Legend({
  mode,
  maxVotes = 3,
}: {
  mode: "binary" | "probability" | "consensus";
  maxVotes?: number;
}) {
  if (mode === "probability") {
    return (
      <ul className="legend" aria-label="Probability scale, shared by all models">
        <li className="legend-item">
          <span className="legend-ramp" aria-hidden="true">
            <i /><i /><i /><i /><i />
          </span>
          <span className="legend-text">Probability 0 → 1</span>
        </li>
      </ul>
    );
  }

  if (mode === "consensus") {
    return (
      <ul className="legend" aria-label="Consensus legend">
        {Array.from({ length: maxVotes + 1 }, (_, votes) => {
          const swatch =
            votes === 0
              ? "is-none"
              : votes >= maxVotes
                ? "is-fire"
                : votes === 1
                  ? "is-partial"
                  : "is-partial-strong";
          return (
            <li className="legend-item" key={votes}>
              <span className={`legend-swatch ${swatch}`} aria-hidden="true" />
              {`${votes} ${votes === 1 ? "model" : "models"}`}
            </li>
          );
        })}
      </ul>
    );
  }

  return (
    <ul className="legend" aria-label="Mask legend">
      <li className="legend-item">
        <span className="legend-swatch is-fire" aria-hidden="true" />
        Fire
      </li>
      <li className="legend-item">
        <span className="legend-swatch is-none" aria-hidden="true" />
        Non-fire
      </li>
      <li className="legend-item">
        <span className="legend-swatch is-unknown" aria-hidden="true" />
        Unknown
      </li>
    </ul>
  );
}
