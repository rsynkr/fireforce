import { vi } from "vitest";

import type { SampleListResponse, Split } from "../types";
import {
  allSampleIds,
  allSamples,
  evaluationFixture,
  healthFixture,
  predictionFor,
  sampleDetailFixture,
} from "./fixtures";

export interface ApiCall {
  url: string;
  path: string;
  method: string;
  body: unknown;
}

export interface MockOptions {
  offline?: boolean;
  predictionMode?: "ok" | "partial" | "service-unavailable";
  detail404?: boolean;
  predictionGate?: Promise<void>;
  evaluationUnavailable?: boolean;
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export function installMockApi(options: MockOptions = {}) {
  const calls: ApiCall[] = [];

  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input));
    const method = init?.method ?? "GET";
    const body =
      typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
    calls.push({ url: String(input), path: url.pathname, method, body });

    if (options.offline) {
      throw new TypeError("fetch failed");
    }

    if (url.pathname === "/api/health") {
      return json(healthFixture);
    }

    if (url.pathname === "/api/evaluation/summary") {
      if (options.evaluationUnavailable) {
        return json(
          {
            detail:
              "Evaluation summary not found. Run `python backend/tools/run_evaluation.py` and retry.",
          },
          503,
        );
      }
      return json(evaluationFixture);
    }

    if (url.pathname === "/api/samples") {
      const split = url.searchParams.get("split") as Split | null;
      const search = (url.searchParams.get("search") ?? "").toLowerCase();
      const limit = Number(url.searchParams.get("limit") ?? 50);
      const offset = Number(url.searchParams.get("offset") ?? 0);
      const filtered = allSamples.filter(
        (sample) =>
          (!split || sample.split === split) &&
          (!search || sample.sample_id.toLowerCase().includes(search)),
      );
      const response: SampleListResponse = {
        total: filtered.length,
        limit,
        offset,
        items: filtered.slice(offset, offset + limit),
      };
      return json(response);
    }

    if (url.pathname.startsWith("/api/samples/")) {
      const sampleId = decodeURIComponent(url.pathname.split("/").pop() ?? "");
      if (!allSampleIds.includes(sampleId)) {
        return json({ detail: `unknown sample_id '${sampleId}'` }, 404);
      }
      if (options.detail404 && sampleId === "eval_00_000027") {
        return json({ detail: `unknown sample_id '${sampleId}'` }, 404);
      }
      return json(sampleDetailFixture(sampleId));
    }

    if (url.pathname === "/api/predictions") {
      if (options.predictionGate) await options.predictionGate;
      const sampleId = (body as { sample_id: string }).sample_id;
      if (options.predictionMode === "service-unavailable") {
        return json(
          {
            detail: {
              message: "no requested model is available",
              models: [
                {
                  model_id: "M1",
                  status: "unavailable",
                  error: "checkpoint not found: model_1_best.pth",
                },
              ],
            },
          },
          503,
        );
      }
      return json(
        predictionFor(
          sampleId,
          options.predictionMode === "partial" ? "partial" : "ok",
        ),
      );
    }

    return json({ detail: "not found" }, 404);
  });

  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}
