import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import App from "./App";
import { installMockApi } from "./test/mockApi";

async function renderApp() {
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText("120 samples");
  return user;
}

async function chooseEvalSample(
  user: ReturnType<typeof userEvent.setup>,
  sampleId: string,
) {
  await user.click(screen.getByRole("button", { name: "Eval" }));
  await screen.findByText("3 samples");
  await user.click(screen.getByRole("button", { name: new RegExp(sampleId) }));
  await screen.findByRole("heading", { level: 1, name: sampleId });
}

async function runPredictions(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByTestId("run-predictions"));
  await screen.findByTestId("model-M1");
}

describe("dashboard", () => {
  it("loads backend status and the sample catalog", async () => {
    installMockApi();
    await renderApp();

    expect(screen.getByText("API online")).toBeInTheDocument();
    expect(screen.getByText("cpu")).toBeInTheDocument();
    expect(screen.getByText("train_00_000000")).toBeInTheDocument();
    expect(screen.getByText("Select a sample.")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /train_00_/ }).length).toBe(50);
  });

  it("searches sample IDs and debounces the request", async () => {
    const { calls } = installMockApi();
    const user = await renderApp();

    await user.type(screen.getByLabelText("Search sample ID"), "eval_00_000027");

    await screen.findByText("1 sample");
    expect(screen.getByText("eval_00_000027")).toBeInTheDocument();
    expect(screen.queryByText("train_00_000000")).not.toBeInTheDocument();
    expect(
      calls.some(
        (call) =>
          call.path === "/api/samples" &&
          call.url.includes("search=eval_00_000027"),
      ),
    ).toBe(true);
  });

  it("filters by split and paginates", async () => {
    const { calls } = installMockApi();
    const user = await renderApp();

    await user.click(screen.getByRole("button", { name: "Eval" }));
    await screen.findByText("3 samples");
    expect(screen.getByText("eval_00_000027")).toBeInTheDocument();
    expect(screen.queryByText("train_00_000000")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "All" }));
    await screen.findByText("120 samples");
    await user.click(screen.getByRole("button", { name: /Next/ }));

    await screen.findByText("51–100 of 120");
    expect(
      calls.some(
        (call) => call.path === "/api/samples" && call.url.includes("offset=50"),
      ),
    ).toBe(true);
  });

  it("selects a sample, loads masks, and runs predictions", async () => {
    const { calls } = installMockApi();
    const user = await renderApp();
    await chooseEvalSample(user, "eval_00_000027");

    expect(screen.getByText("21 fire px")).toBeInTheDocument();
    expect(screen.getAllByTestId("pixel-grid")).toHaveLength(2);
    expect(screen.getByLabelText(/Previous-fire mask/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Ground-truth mask/)).toBeInTheDocument();

    await runPredictions(user);

    const request = calls.find((call) => call.path === "/api/predictions");
    expect(request?.method).toBe("POST");
    expect(request?.body).toEqual({
      sample_id: "eval_00_000027",
      models: ["M1", "M2", "M3"],
    });

    expect(screen.getAllByTestId("pixel-grid")).toHaveLength(5);
    expect(screen.getAllByText("0.7143").length).toBeGreaterThanOrEqual(4);
    expect(screen.getAllByText("7.4 ms")).toHaveLength(3);
    expect(screen.getByText("#11111111…")).toBeInTheDocument();
    expect(screen.getByText("56.2 ms")).toBeInTheDocument();
    for (const modelId of ["M1", "M2", "M3"]) {
      expect(
        within(screen.getByTestId(`model-${modelId}`)).getByText(/^#/),
      ).toBeInTheDocument();
    }
  });

  it("clears previous predictions when another sample is selected", async () => {
    installMockApi();
    const user = await renderApp();
    await chooseEvalSample(user, "eval_00_000027");
    await runPredictions(user);
    expect(screen.getByTestId("model-M1")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /eval_00_000009/ }));

    await screen.findByRole("heading", { level: 1, name: "eval_00_000009" });
    expect(screen.queryByTestId("model-M1")).not.toBeInTheDocument();
    expect(screen.getByText("Not run")).toBeInTheDocument();
  });

  it("renders all-unknown targets with unavailable metrics", async () => {
    installMockApi();
    const user = await renderApp();
    await chooseEvalSample(user, "eval_00_000009");

    expect(screen.getByText("all target pixels unknown")).toBeInTheDocument();
    await runPredictions(user);

    expect(
      screen.getAllByText(
        "Metrics unavailable — no valid target pixels in this sample.",
      ),
    ).toHaveLength(3);
    expect(screen.getByLabelText(/Ground-truth mask: 0 fire pixels, 4,096 unknown/)).toBeInTheDocument();
  });

  it("preserves successful models when one model fails", async () => {
    installMockApi({ predictionMode: "partial" });
    const user = await renderApp();
    await chooseEvalSample(user, "eval_00_000027");
    await runPredictions(user);

    const m1 = screen.getByTestId("model-M1");
    const m2 = screen.getByTestId("model-M2");
    const m3 = screen.getByTestId("model-M3");

    expect(within(m1).getByText("Dice")).toBeInTheDocument();
    expect(within(m1).getAllByText("0.7143").length).toBeGreaterThan(0);
    expect(within(m2).getByText("unavailable")).toBeInTheDocument();
    expect(
      within(m2).getByText(/checkpoint not found: model_2_best\.pth/),
    ).toBeInTheDocument();
    expect(within(m3).getByText("Dice")).toBeInTheDocument();
    expect(screen.getByText("partial")).toBeInTheDocument();
  });

  it("switches to a shared probability scale", async () => {
    installMockApi();
    const user = await renderApp();
    await chooseEvalSample(user, "eval_00_000027");
    await runPredictions(user);

    await user.click(screen.getByRole("button", { name: "Probability" }));
    expect(screen.getByText("Probability 0 → 1")).toBeInTheDocument();
    expect(screen.getAllByText("shared 0–1 scale")).toHaveLength(3);
  });

  it("shows a consensus map for disagreements", async () => {
    installMockApi();
    const user = await renderApp();
    await chooseEvalSample(user, "eval_00_000027");
    await runPredictions(user);

    await user.click(screen.getByRole("button", { name: "Consensus" }));

    expect(screen.getByText("Consensus · 3 models")).toBeInTheDocument();
    expect(screen.getByText("Unanimous")).toBeInTheDocument();
    expect(screen.getByText("Split vote")).toBeInTheDocument();
    expect(screen.getByLabelText(/Consensus map across 3 models/)).toBeInTheDocument();
  });

  it("renders the evaluation summary tab", async () => {
    installMockApi();
    const user = await renderApp();

    await user.click(screen.getByRole("tab", { name: "Evaluation" }));
    await screen.findByRole("heading", { name: "Model evaluation" });

    expect(screen.getByText("100-sample subset")).toBeInTheDocument();
    expect(screen.getByText("Overall comparison")).toBeInTheDocument();
    expect(screen.getAllByText("0.3818").length).toBeGreaterThan(0);
    expect(screen.getByText("Best F1 / Dice")).toBeInTheDocument();
    expect(screen.getByText(/M2 achieved the strongest overall/)).toBeInTheDocument();
    expect(screen.getByText("Metric definitions")).toBeInTheDocument();
    expect(screen.getAllByText("Pred fire")).toHaveLength(3);
    expect(screen.getAllByText(/not a full test-set evaluation/).length).toBeGreaterThan(0);

    await user.click(screen.getByRole("tab", { name: "Inspector" }));
    expect(screen.getByText("Select a sample.")).toBeInTheDocument();
  });

  it("shows an actionable error when evaluation output is missing", async () => {
    installMockApi({ evaluationUnavailable: true });
    const user = await renderApp();

    await user.click(screen.getByRole("tab", { name: "Evaluation" }));
    await screen.findByText("Evaluation summary unavailable");
    expect(screen.getAllByText(/run_evaluation\.py/).length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: "Retry" }));
    await screen.findByText("Evaluation summary unavailable");
  });

  it("handles a backend that is offline", async () => {
    const { calls } = installMockApi({ offline: true });
    render(<App />);

    await screen.findByText("Backend offline");
    expect(screen.getByText("API offline")).toBeInTheDocument();
    expect(screen.getAllByText(/Backend unreachable/).length).toBeGreaterThan(0);

    const before = calls.filter((call) => call.path === "/api/health").length;
    await userEvent.click(screen.getByRole("button", { name: "Retry connection" }));
    await waitFor(() =>
      expect(
        calls.filter((call) => call.path === "/api/health").length,
      ).toBeGreaterThan(before),
    );
  });

  it("recovers the catalog after a successful retry", async () => {
    installMockApi({ offline: true });
    render(<App />);
    await screen.findByText("Backend offline");

    installMockApi();
    await userEvent.click(
      screen.getByRole("button", { name: "Retry connection" }),
    );

    await screen.findByText("120 samples");
    expect(screen.getByText("train_00_000000")).toBeInTheDocument();
    expect(screen.getByText("API online")).toBeInTheDocument();
  });

  it("prevents duplicate in-flight prediction requests", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const { calls } = installMockApi({ predictionGate: gate });
    const user = await renderApp();
    await chooseEvalSample(user, "eval_00_000027");

    const runButton = screen.getByTestId("run-predictions");
    fireEvent.click(runButton);
    fireEvent.click(runButton);

    expect(
      calls.filter((call) => call.path === "/api/predictions"),
    ).toHaveLength(1);

    release();
    await screen.findByTestId("model-M1");
  });

  it("surfaces total failure from a 503 response", async () => {
    installMockApi({ predictionMode: "service-unavailable" });
    const user = await renderApp();
    await chooseEvalSample(user, "eval_00_000027");
    await user.click(screen.getByTestId("run-predictions"));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Prediction failed");
    expect(alert).toHaveTextContent("no requested model is available");
    expect(alert).toHaveTextContent("checkpoint not found");
    expect(screen.queryByTestId("model-M1")).not.toBeInTheDocument();
  });

  it("surfaces a 404 on sample detail", async () => {
    installMockApi({ detail404: true });
    const user = await renderApp();
    await user.click(screen.getByRole("button", { name: "Eval" }));
    await screen.findByText("3 samples");
    await user.click(screen.getByRole("button", { name: /eval_00_000027/ }));

    expect(await screen.findByText("Sample unavailable")).toBeInTheDocument();
    expect(screen.getByText(/unknown sample_id/)).toBeInTheDocument();
  });
});
