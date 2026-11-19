import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";

import { api, ApiError } from "./api/client";
import { CatalogPanel } from "./components/CatalogPanel";
import type { SplitFilter } from "./components/CatalogPanel";
import { Comparison } from "./components/Comparison";
import type { ViewMode } from "./components/Comparison";
import { EvaluationView } from "./components/EvaluationView";
import { MasksSection } from "./components/MasksSection";
import type { Pixel } from "./components/MaskCanvas";
import { Mark, StatusPill } from "./components/ui";
import { count, shortHash } from "./format";
import { useDebouncedValue } from "./hooks/useDebouncedValue";
import type {
  HealthResponse,
  PredictionResponse,
  SampleDetail,
  SampleListResponse,
  SampleSummary,
} from "./types";

const PAGE_SIZE = 50;
const MIN_FIRE_PIXELS = 101; // catalog visibility: strictly more than 100 GT fire pixels

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "Something went wrong.";
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function predictionErrorMessage(error: unknown): string {
  if (!(error instanceof ApiError)) return errorMessage(error);
  const detail = error.detail;
  if (detail && typeof detail === "object" && "models" in detail) {
    const models = (detail as { models?: unknown }).models;
    if (Array.isArray(models)) {
      const reasons = models
        .map((entry) => {
          if (!entry || typeof entry !== "object") return null;
          const record = entry as Record<string, unknown>;
          if (record.status === "ok") return null;
          return `${String(record.model_id ?? "model")}: ${String(
            record.error ?? record.status ?? "unavailable",
          )}`;
        })
        .filter(Boolean);
      if (reasons.length) {
        return `${error.message} — ${reasons.join(" · ")}`;
      }
    }
  }
  if (error.status === 404) return `${error.message} Select another sample.`;
  if (error.status === 503) return `${error.message} Check /api/health and retry.`;
  return error.message;
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [split, setSplit] = useState<SplitFilter>("all");
  const [search, setSearch] = useState("");
  const debouncedSearch = useDebouncedValue(search, 250);
  const [offset, setOffset] = useState(0);
  const [list, setList] = useState<SampleListResponse | null>(null);
  const [listLoading, setListLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SampleDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  const [view, setView] = useState<ViewMode>("binary");
  const [pixel, setPixel] = useState<Pixel | null>(null);
  const [section, setSection] = useState<"inspector" | "evaluation">("inspector");

  const runSequence = useRef(0);
  const runAbort = useRef<AbortController | null>(null);
  const runningRef = useRef(false);

  const loadHealth = useCallback(async () => {
    setHealthLoading(true);
    setHealthError(null);
    try {
      const response = await api.health();
      setHealth(response);
    } catch (error) {
      setHealth(null);
      setHealthError(errorMessage(error));
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHealth();
  }, [loadHealth]);

  useEffect(() => {
    if (!health) {
      if (healthError) {
        setList(null);
        setListError(healthError);
        setListLoading(false);
      }
      return;
    }

    const controller = new AbortController();
    setListLoading(true);
    setListError(null);
    api
      .samples(
        {
          split: split === "all" ? undefined : split,
          search: debouncedSearch,
          minFirePixels: MIN_FIRE_PIXELS,
          limit: PAGE_SIZE,
          offset,
        },
        controller.signal,
      )
      .then((response) => {
        if (!controller.signal.aborted) setList(response);
      })
      .catch((error) => {
        if (controller.signal.aborted || isAbort(error)) return;
        setListError(errorMessage(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) setListLoading(false);
      });
    return () => controller.abort();
  }, [split, debouncedSearch, offset, health, healthError]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    const controller = new AbortController();
    setDetailLoading(true);
    setDetailError(null);
    api
      .sample(selectedId, controller.signal)
      .then((response) => {
        if (!controller.signal.aborted) setDetail(response);
      })
      .catch((error) => {
        if (controller.signal.aborted || isAbort(error)) return;
        setDetail(null);
        setDetailError(errorMessage(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) setDetailLoading(false);
      });
    return () => controller.abort();
  }, [selectedId]);

  function selectSample(sample: SampleSummary) {
    runSequence.current += 1;
    runAbort.current?.abort();
    runningRef.current = false;
    setRunning(false);
    setSelectedId(sample.sample_id);
    setDetail(null);
    setDetailError(null);
    setPrediction(null);
    setPredictionError(null);
    setPixel(null);
    setView("binary");
  }

  function changeSplit(next: SplitFilter) {
    setSplit(next);
    setOffset(0);
  }

  function changeSearch(next: string) {
    setSearch(next);
    setOffset(0);
  }

  function handleTabKey(event: ReactKeyboardEvent<HTMLElement>) {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    event.preventDefault();
    const next = section === "inspector" ? "evaluation" : "inspector";
    setSection(next);
    document.getElementById(`tab-${next}`)?.focus();
  }

  async function runPrediction() {
    if (!selectedId || runningRef.current) return;
    runningRef.current = true;

    const sequence = runSequence.current + 1;
    runSequence.current = sequence;
    runAbort.current?.abort();
    const controller = new AbortController();
    runAbort.current = controller;

    setRunning(true);
    setPrediction(null);
    setPredictionError(null);
    setPixel(null);

    try {
      const response = await api.predict(
        { sample_id: selectedId, models: ["M1", "M2", "M3"] },
        controller.signal,
      );
      if (sequence !== runSequence.current) return;
      setPrediction(response);
      setView("binary");
    } catch (error) {
      if (isAbort(error) || sequence !== runSequence.current) return;
      setPredictionError(predictionErrorMessage(error));
    } finally {
      if (sequence === runSequence.current) {
        runningRef.current = false;
        setRunning(false);
      }
    }
  }

  const online = health !== null && health.data.ready;
  const statusLabel = healthLoading
    ? "Checking API"
    : online
      ? health?.status === "ok"
        ? "API online"
        : "API degraded"
      : "API offline";

  return (
    <div className="app">
      <a className="skip-link" href="#workspace">
        Skip to workspace
      </a>

      <header className="app-header">
        <div className="brand">
          <Mark />
          <div>
            <p className="wordmark">FIREFORCE</p>
            <p className="tagline">Next-day wildfire spread · research console</p>
          </div>
        </div>

        <div className="header-status">
          <span className={`connection${online ? " is-online" : " is-offline"}`}>
            <span className="connection-dot" aria-hidden="true" />
            {statusLabel}
          </span>
          <span className="header-fact">
            device <b>{health?.device ?? "—"}</b>
          </span>
          <span className="header-fact">
            dataset <b>{count(health?.data.sample_count)}</b>
          </span>
          <span
            className="header-fact is-fingerprint"
            title={health?.data.fingerprint ?? undefined}
          >
            fp <b>#{shortHash(health?.data.fingerprint)}</b>
          </span>
        </div>
      </header>

      <main className={`layout${section === "evaluation" ? " is-wide" : ""}`}>
        {section === "inspector" ? (
          <CatalogPanel
            split={split}
            onSplit={changeSplit}
            search={search}
            onSearch={changeSearch}
            minFirePixels={MIN_FIRE_PIXELS}
            data={list}
            listLoading={listLoading}
            listError={listError}
            selectedId={selectedId}
            onSelect={selectSample}
            offset={offset}
            limit={PAGE_SIZE}
            onOffset={setOffset}
            onRetry={() => void loadHealth()}
          />
        ) : null}

        <div className="workspace" id="workspace">
          <nav
            className="workspace-tabs"
            role="tablist"
            aria-label="Sections"
            onKeyDown={handleTabKey}
          >
            <button
              type="button"
              role="tab"
              id="tab-inspector"
              aria-controls="panel-inspector"
              aria-selected={section === "inspector"}
              tabIndex={section === "inspector" ? 0 : -1}
              className={section === "inspector" ? "is-active" : undefined}
              onClick={() => setSection("inspector")}
            >
              Inspector
            </button>
            <button
              type="button"
              role="tab"
              id="tab-evaluation"
              aria-controls="panel-evaluation"
              aria-selected={section === "evaluation"}
              tabIndex={section === "evaluation" ? 0 : -1}
              className={section === "evaluation" ? "is-active" : undefined}
              onClick={() => setSection("evaluation")}
            >
              Evaluation
            </button>
          </nav>

          {section === "evaluation" ? (
            <div
              className="tab-panel"
              id="panel-evaluation"
              role="tabpanel"
              aria-labelledby="tab-evaluation"
            >
              <EvaluationView />
            </div>
          ) : (
            <div
              className="tab-panel"
              id="panel-inspector"
              role="tabpanel"
              aria-labelledby="tab-inspector"
            >
          {healthError ? (
            <div className="banner is-error" role="alert">
              <strong>Backend offline</strong>
              <p>
                {healthError} The catalog and predictions require the local FastAPI
                service.
              </p>
              <button type="button" className="text-button" onClick={() => void loadHealth()}>
                Retry connection
              </button>
            </div>
          ) : null}

          {!selectedId ? (
            <section className="empty-workspace">
              <div className="empty-copy">
                <p className="empty-kicker">Workspace · 64 × 64</p>
                <h1>Select a sample.</h1>
                <p className="lede">
                  Pick a stable sample ID from the catalog to inspect the previous-fire
                  input, the next-day ground truth, and compare M1, M2 and M3.
                </p>
                <p className="fineprint">
                  Research output only. Not an operational wildfire forecast.
                </p>
              </div>

              <aside className="empty-aside" aria-label="Getting started">
                <p className="empty-kicker">Start here</p>
                <ol className="empty-steps">
                  <li>
                    <span>01</span>
                    <div>
                      <b>Choose a sample</b>
                      <small>
                        Only IDs with more than 100 actual fire pixels are listed.
                      </small>
                    </div>
                  </li>
                  <li>
                    <span>02</span>
                    <div>
                      <b>Inspect inputs</b>
                      <small>Previous fire and ground truth, both 64×64.</small>
                    </div>
                  </li>
                  <li>
                    <span>03</span>
                    <div>
                      <b>Run and compare</b>
                      <small>M1, M2 and M3 on the same record.</small>
                    </div>
                  </li>
                </ol>
                <dl className="empty-stats">
                  <div>
                    <dt>Visible</dt>
                    <dd>{count(list?.total)}</dd>
                  </div>
                  <div>
                    <dt>Indexed</dt>
                    <dd>{count(health?.data.sample_count)}</dd>
                  </div>
                  <div>
                    <dt>Models</dt>
                    <dd>3</dd>
                  </div>
                  <div>
                    <dt>Device</dt>
                    <dd>{health?.device ?? "—"}</dd>
                  </div>
                </dl>
              </aside>
            </section>
          ) : detailLoading ? (
            <section className="panel" aria-busy="true">
              <p className="loading-note">Loading sample {selectedId}…</p>
            </section>
          ) : detailError ? (
            <div className="banner is-error" role="alert">
              <strong>Sample unavailable</strong>
              <p>{detailError}</p>
              <p className="fineprint">Choose another ID from the catalog.</p>
            </div>
          ) : detail ? (
            <>
              <section className="sample-header">
                <div>
                  <p className="eyebrow">
                    {detail.split} / {detail.shard} / record {detail.record_index}
                  </p>
                  <h1>{detail.sample_id}</h1>
                </div>
                <div className="sample-actions">
                  <StatusPill tone={detail.all_unknown ? "warn" : "neutral"}>
                    {detail.all_unknown
                      ? "all target pixels unknown"
                      : `${count(detail.target_fire_pixels)} fire px`}
                  </StatusPill>
                  <button
                    type="button"
                    className="button is-primary"
                    onClick={() => void runPrediction()}
                    disabled={running}
                    data-testid="run-predictions"
                  >
                    {running ? "Running…" : "Run predictions"}
                  </button>
                </div>
              </section>

              <MasksSection
                detail={detail}
                health={health}
                pixel={pixel}
                onPixel={setPixel}
              />

              <Comparison
                detail={detail}
                prediction={prediction}
                running={running}
                error={predictionError}
                view={view}
                onView={setView}
                pixel={pixel}
                onPixel={setPixel}
                onRun={() => void runPrediction()}
              />
            </>
          ) : null}
            </div>
          )}

          <footer className="app-footer">
            <span>Research prototype · not an operational forecast</span>
            <span className="mono">
              {health?.threshold_provenance
                ? "thresholds: validation-selected · sweep not re-run"
                : `api ${api.baseUrl}`}
            </span>
          </footer>
        </div>
      </main>
    </div>
  );
}
