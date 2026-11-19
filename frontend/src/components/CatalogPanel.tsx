import { count } from "../format";
import type { SampleListResponse, SampleSummary, Split } from "../types";
import { SectionLabel } from "./ui";

export type SplitFilter = Split | "all";

const SPLITS: Array<{ value: SplitFilter; label: string }> = [
  { value: "all", label: "All" },
  { value: "train", label: "Train" },
  { value: "eval", label: "Eval" },
  { value: "test", label: "Test" },
];

interface CatalogPanelProps {
  split: SplitFilter;
  onSplit: (split: SplitFilter) => void;
  search: string;
  onSearch: (search: string) => void;
  minFirePixels: number;
  data: SampleListResponse | null;
  listLoading: boolean;
  listError: string | null;
  selectedId: string | null;
  onSelect: (sample: SampleSummary) => void;
  offset: number;
  limit: number;
  onOffset: (offset: number) => void;
  onRetry: () => void;
}

export function CatalogPanel({
  split,
  onSplit,
  search,
  onSearch,
  minFirePixels,
  data,
  listLoading,
  listError,
  selectedId,
  onSelect,
  offset,
  limit,
  onOffset,
  onRetry,
}: CatalogPanelProps) {
  const total = data?.total ?? 0;
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(offset + limit, total);

  return (
    <aside className="catalog" aria-label="Sample catalog">
      <SectionLabel index="01">Catalog</SectionLabel>

      <p className="catalog-filter-note">
        Showing samples with &gt;{minFirePixels - 1} actual fire pixels
      </p>

      <div className="segmented" role="group" aria-label="Filter by split">
        {SPLITS.map((option) => (
          <button
            key={option.value}
            type="button"
            className={split === option.value ? "is-active" : undefined}
            aria-pressed={split === option.value}
            onClick={() => onSplit(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>

      <label className="search-field">
        <span>Search sample ID</span>
        <input
          type="search"
          value={search}
          placeholder="Filter by ID…"
          autoComplete="off"
          spellCheck={false}
          onChange={(event) => onSearch(event.target.value)}
        />
      </label>

      <div className="catalog-meta" aria-live="polite">
        <span>
          {count(total)} {total === 1 ? "sample" : "samples"}
        </span>
        {listLoading ? <span className="muted">Updating…</span> : null}
      </div>

      {listError ? (
        <div className="inline-error" role="alert">
          <p>{listError}</p>
          <button type="button" className="text-button" onClick={onRetry}>
            Retry
          </button>
        </div>
      ) : (
        <nav className="sample-list" aria-label="Sample IDs">
          {listLoading && !data ? (
            <div className="sample-skeletons" aria-hidden="true">
              {Array.from({ length: 8 }).map((_, index) => (
                <span key={index} className="sample-skeleton" />
              ))}
            </div>
          ) : data && data.items.length > 0 ? (
            data.items.map((sample) => (
              <button
                key={sample.sample_id}
                type="button"
                className={`sample-row${selectedId === sample.sample_id ? " is-selected" : ""}`}
                aria-current={selectedId === sample.sample_id ? "true" : undefined}
                onClick={() => onSelect(sample)}
                title={`${sample.shard} · record ${sample.record_index}`}
              >
                <span className="sample-id">{sample.sample_id}</span>
                <span className="sample-meta">
                  {sample.all_unknown ? "all unknown" : `${count(sample.target_fire_pixels)} fire`}
                </span>
              </button>
            ))
          ) : (
            <p className="empty-note">
              {listLoading
                ? "Loading sample IDs…"
                : search.trim()
                  ? `No sample IDs match this search with >${minFirePixels - 1} fire pixels.`
                  : split === "all"
                    ? `No samples have more than ${minFirePixels - 1} actual fire pixels.`
                    : `No samples in the ${split} split have more than ${minFirePixels - 1} actual fire pixels.`}
            </p>
          )}
        </nav>
      )}

      <div className="catalog-pager">
        <button
          type="button"
          className="text-button"
          disabled={offset === 0 || listLoading}
          onClick={() => onOffset(Math.max(0, offset - limit))}
        >
          ← Prev
        </button>
        <span className="pager-range">
          {total === 0 ? "0 of 0" : `${count(start)}–${count(end)} of ${count(total)}`}
        </span>
        <button
          type="button"
          className="text-button"
          disabled={end >= total || listLoading}
          onClick={() => onOffset(offset + limit)}
        >
          Next →
        </button>
      </div>
    </aside>
  );
}
