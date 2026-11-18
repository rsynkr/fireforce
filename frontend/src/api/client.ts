import type {
  EvaluationSummary,
  HealthResponse,
  ModelName,
  PredictionResponse,
  SampleDetail,
  SampleListResponse,
  Split,
} from "../types";

const BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, detail: unknown) {
    super(messageFromDetail(detail));
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function messageFromDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const first = detail[0];
    if (first && typeof first === "object" && "msg" in first) {
      return String((first as { msg: unknown }).msg);
    }
    return "Request validation failed.";
  }
  if (detail && typeof detail === "object") {
    const record = detail as Record<string, unknown>;
    if (typeof record.message === "string") return record.message;
    if ("detail" in record) return messageFromDetail(record.detail);
  }
  return "The request failed.";
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(
      0,
      `Backend unreachable at ${BASE_URL}. Start the FastAPI service and retry.`,
    );
  }

  if (!response.ok) {
    let detail: unknown = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail ?? body;
    } catch {
      // Keep the status text when the body is not JSON.
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const api = {
  baseUrl: BASE_URL,

  health(signal?: AbortSignal): Promise<HealthResponse> {
    return request<HealthResponse>("/api/health", { signal });
  },

  samples(
    params: {
      split?: Split;
      search?: string;
      limit?: number;
      offset?: number;
    },
    signal?: AbortSignal,
  ): Promise<SampleListResponse> {
    const query = new URLSearchParams();
    if (params.split) query.set("split", params.split);
    if (params.search?.trim()) query.set("search", params.search.trim());
    if (params.limit != null) query.set("limit", String(params.limit));
    if (params.offset != null) query.set("offset", String(params.offset));
    const suffix = query.size ? `?${query.toString()}` : "";
    return request<SampleListResponse>(`/api/samples${suffix}`, { signal });
  },

  sample(sampleId: string, signal?: AbortSignal): Promise<SampleDetail> {
    return request<SampleDetail>(
      `/api/samples/${encodeURIComponent(sampleId)}`,
      { signal },
    );
  },

  predict(
    body: { sample_id: string; models: ModelName[] },
    signal?: AbortSignal,
  ): Promise<PredictionResponse> {
    return request<PredictionResponse>("/api/predictions", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    });
  },

  evaluation(signal?: AbortSignal): Promise<EvaluationSummary> {
    return request<EvaluationSummary>("/api/evaluation/summary", { signal });
  },
};
