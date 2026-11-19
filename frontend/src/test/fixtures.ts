import type {
  EvaluationSummary,
  HealthResponse,
  Metrics,
  ModelName,
  PredictionResponse,
  SampleDetail,
  SampleSummary,
  Split,
} from "../types";

const ZERO_GRID = () =>
  Array.from({ length: 64 }, () => Array<number>(64).fill(0));
const TRUE_GRID = () =>
  Array.from({ length: 64 }, () => Array<boolean>(64).fill(true));

export function maskWithFire(firePixels: number): number[][] {
  const grid = ZERO_GRID();
  for (let index = 0; index < firePixels; index += 1) {
    grid[Math.floor(index / 64)][index % 64] = 1;
  }
  return grid;
}

function detailFor(id: string): SampleDetail {
  const [split, shardSuffix, record] = id.split("_") as [Split, string, string];
  const recordIndex = Number(record);
  const allUnknown = id === "eval_00_000009";
  const firePixels = id === "eval_00_000014" ? 319 : 5;
  const validMask = allUnknown
    ? Array.from({ length: 64 }, () => Array<boolean>(64).fill(false))
    : TRUE_GRID();
  const previous = ZERO_GRID();
  if (!allUnknown && id === "eval_00_000014") {
    for (let index = 0; index < 5; index += 1) previous[0][index] = 1;
    for (let index = 0; index < 13; index += 1) previous[63][63 - index] = -1;
  }
  return {
    sample_id: id,
    split,
    shard: `next_day_wildfire_spread_${split}_${shardSuffix}.tfrecord`,
    record_index: recordIndex,
    target_valid_pixels: allUnknown ? 0 : 4096,
    target_fire_pixels: allUnknown ? 0 : firePixels,
    target_unknown_pixels: allUnknown ? 4096 : 0,
    all_unknown: allUnknown,
    available_features: [
      "tmmn", "tmmx", "pr", "NDVI", "pdsi", "erc",
      "elevation", "population", "sph", "vs", "th",
      "PrevFireMask", "FireMask",
    ],
    models: ["M1", "M2", "M3"],
    previous_fire_mask: previous,
    target_mask: allUnknown ? ZERO_GRID() : maskWithFire(firePixels),
    target_valid_mask: validMask,
  };
}

function summaryFor(id: string): SampleSummary {
  const detail = detailFor(id);
  return {
    sample_id: detail.sample_id,
    split: detail.split,
    shard: detail.shard,
    record_index: detail.record_index,
    target_valid_pixels: detail.target_valid_pixels,
    target_fire_pixels: detail.target_fire_pixels,
    target_unknown_pixels: detail.target_unknown_pixels,
    all_unknown: detail.all_unknown,
  };
}

const trainIds = Array.from(
  { length: 115 },
  (_, index) => `train_00_${String(index).padStart(6, "0")}`,
);
const evalIds = ["eval_00_000000", "eval_00_000009", "eval_00_000014"];
const testIds = ["test_00_000000", "test_00_000001"];

export const allSampleIds = [...trainIds, ...evalIds, ...testIds];
export const allSamples = allSampleIds.map(summaryFor);

export const healthFixture: HealthResponse = {
  status: "ok",
  device: "cpu",
  cuda_available: false,
  threshold_provenance:
    "Validation-selected thresholds recorded with the project (M1=0.94, M2=0.94, M3=0.93). Stored sweep outputs show maximum Dice at these values; this build has not re-run the sweep.",
  data: {
    ready: true,
    manifest_version: 1,
    fingerprint: "a".repeat(64),
    shard_count: 19,
    sample_count: allSamples.length,
    split_counts: { train: trainIds.length, eval: evalIds.length, test: testIds.length },
    error: null,
  },
  models: {
    M1: {
      status: "ready",
      checkpoint: "model_1_best.pth",
      in_channels: 1,
      threshold: 0.94,
      sha256: "1".repeat(64),
      error: null,
    },
    M2: {
      status: "ready",
      checkpoint: "model_2_best.pth",
      in_channels: 8,
      threshold: 0.94,
      sha256: "2".repeat(64),
      error: null,
    },
    M3: {
      status: "ready",
      checkpoint: "model_3_best.pth",
      in_channels: 13,
      threshold: 0.93,
      sha256: "3".repeat(64),
      error: null,
    },
  },
};

function probabilityGrid(firePixels: number): number[][] {
  const grid = Array.from({ length: 64 }, () => Array<number>(64).fill(0.04));
  for (let index = 0; index < firePixels; index += 1) {
    grid[Math.floor(index / 64)][index % 64] =
      0.97 - (index % 5) * 0.01;
  }
  return grid;
}

const knownMetrics: Metrics = {
  available: true,
  reason: null,
  valid_pixels: 4096,
  true_positive: 15,
  false_positive: 6,
  false_negative: 6,
  true_negative: 4069,
  precision: 0.7143,
  recall: 0.7143,
  dice: 0.7143,
  iou: 0.5556,
};

const unavailableMetrics: Metrics = {
  available: false,
  reason: "no_valid_target_pixels",
  valid_pixels: 0,
  true_positive: null,
  false_positive: null,
  false_negative: null,
  true_negative: null,
  precision: null,
  recall: null,
  dice: null,
  iou: null,
};

export const emptySamples = summaryFor("eval_00_000027");

export function predictionFor(
  sampleId: string,
  mode: "ok" | "partial" = "ok",
): PredictionResponse {
  const detail = detailFor(sampleId);
  const probabilities: Record<ModelName, number[][]> = {
    M1: probabilityGrid(21),
    M2: probabilityGrid(18),
    M3: probabilityGrid(23),
  };
  const thresholds: Record<ModelName, number> = { M1: 0.94, M2: 0.94, M3: 0.93 };
  const hashes: Record<ModelName, string> = {
    M1: "1".repeat(64),
    M2: "2".repeat(64),
    M3: "3".repeat(64),
  };

  return {
    sample_id: sampleId,
    split: detail.split,
    shard: detail.shard,
    record_index: detail.record_index,
    status: mode === "partial" ? "partial" : "ok",
    device: "cpu",
    duration_ms: 56.2,
    available_features: detail.available_features,
    ground_truth: {
      available: !detail.all_unknown,
      target_mask: detail.target_mask,
      valid_mask: detail.target_valid_mask,
      valid_pixels: detail.target_valid_pixels,
      fire_pixels: detail.target_fire_pixels,
      unknown_pixels: detail.target_unknown_pixels,
    },
    models: (["M1", "M2", "M3"] as ModelName[]).map((modelId) => {
      if (mode === "partial" && modelId === "M2") {
        return {
          model_id: modelId,
          status: "unavailable" as const,
          threshold: thresholds[modelId],
          threshold_provenance: healthFixture.threshold_provenance,
          checkpoint: {
            file: "model_2_best.pth",
            sha256: null,
            in_channels: 8,
          },
          inference_ms: null,
          probabilities: null,
          mask: null,
          predicted_fire_pixels: null,
          metrics: null,
          error: "checkpoint not found: model_2_best.pth",
        };
      }
      const grid = probabilities[modelId];
      const mask = grid.map((row) => row.map((value) => value >= thresholds[modelId]));
      return {
        model_id: modelId,
        status: "ok" as const,
        threshold: thresholds[modelId],
        threshold_provenance: healthFixture.threshold_provenance,
        checkpoint: {
          file: `model_${modelId.slice(1)}_best.pth`,
          sha256: hashes[modelId],
          in_channels: modelId === "M1" ? 1 : modelId === "M2" ? 8 : 13,
        },
        inference_ms: 7.4,
        probabilities: grid,
        mask,
        predicted_fire_pixels: mask.flat().filter(Boolean).length,
        metrics: detail.all_unknown ? unavailableMetrics : knownMetrics,
        error: null,
      };
    }),
  };
}

export { detailFor as sampleDetailFixture };

function evaluationRanking(
  metric: string,
  direction: "max" | "min",
  winners: ModelName[],
  values: Record<ModelName, number>,
  order: ModelName[],
) {
  return { metric, direction, winners, values, order };
}

export const evaluationFixture: EvaluationSummary = {
  evaluation_version: "100-test-v1",
  split: "test",
  sampling: {
    strategy:
      "candidate IDs sorted; rank by sha256(f'{seed}:{sample_id}') ascending; take the lowest count; sort selection by sample_id",
    seed: 20261118,
    count: 100,
  },
  counts: {
    selected: 100,
    valid_samples: 100,
    all_unknown_samples: 0,
    excluded_sample_ids: [],
    valid_pixels: 404679,
    actual_fire_pixels: 5264,
  },
  thresholds: { M1: 0.94, M2: 0.94, M3: 0.93 },
  threshold_provenance: healthFixture.threshold_provenance,
  models: {
    M1: {
      threshold: 0.94,
      in_channels: 1,
      micro: { precision: 0.2489, recall: 0.4882, f1_dice: 0.3297, iou: 0.1974, accuracy: 0.9742, valid_pixels: 404679 },
      macro: { precision: 0.254, recall: 0.4847, f1_dice: 0.2804, iou: 0.1786, accuracy: 0.9744, samples: 100 },
      counts: { true_positive: 2570, false_positive: 7757, false_negative: 2694, true_negative: 391658 },
      predicted: { total: 10332, mean: 103.32, median: 39.5, max: 716, on_valid_pixels: 10327, on_unknown_pixels: 5, percent_of_valid_pixels: 2.5519 },
    },
    M2: {
      threshold: 0.94,
      in_channels: 8,
      micro: { precision: 0.3367, recall: 0.4409, f1_dice: 0.3818, iou: 0.2359, accuracy: 0.9814, valid_pixels: 404679 },
      macro: { precision: 0.303, recall: 0.4213, f1_dice: 0.3136, iou: 0.208, accuracy: 0.9816, samples: 100 },
      counts: { true_positive: 2321, false_positive: 4573, false_negative: 2943, true_negative: 394842 },
      predicted: { total: 6914, mean: 69.14, median: 24, max: 541, on_valid_pixels: 6894, on_unknown_pixels: 20, percent_of_valid_pixels: 1.7036 },
    },
    M3: {
      threshold: 0.93,
      in_channels: 13,
      micro: { precision: 0.2707, recall: 0.4791, f1_dice: 0.3459, iou: 0.2091, accuracy: 0.9764, valid_pixels: 404679 },
      macro: { precision: 0.2598, recall: 0.4806, f1_dice: 0.2956, iou: 0.1912, accuracy: 0.9767, samples: 100 },
      counts: { true_positive: 2522, false_positive: 6796, false_negative: 2742, true_negative: 392619 },
      predicted: { total: 9330, mean: 93.3, median: 41.5, max: 581, on_valid_pixels: 9318, on_unknown_pixels: 12, percent_of_valid_pixels: 2.3026 },
    },
  },
  rankings: {
    best_f1_dice: evaluationRanking("f1_dice", "max", ["M2"], { M1: 0.3297, M2: 0.3818, M3: 0.3459 }, ["M2", "M3", "M1"]),
    best_iou: evaluationRanking("iou", "max", ["M2"], { M1: 0.1974, M2: 0.2359, M3: 0.2091 }, ["M2", "M3", "M1"]),
    best_precision: evaluationRanking("precision", "max", ["M2"], { M1: 0.2489, M2: 0.3367, M3: 0.2707 }, ["M2", "M3", "M1"]),
    best_recall: evaluationRanking("recall", "max", ["M1"], { M1: 0.4882, M2: 0.4409, M3: 0.4791 }, ["M1", "M3", "M2"]),
    best_accuracy: evaluationRanking("accuracy", "max", ["M2"], { M1: 0.9742, M2: 0.9814, M3: 0.9764 }, ["M2", "M3", "M1"]),
    most_predicted_fire: evaluationRanking("total", "max", ["M1"], { M1: 10332, M2: 6914, M3: 9330 }, ["M1", "M3", "M2"]),
    fewest_predicted_fire: evaluationRanking("total", "min", ["M2"], { M1: 10332, M2: 6914, M3: 9330 }, ["M2", "M3", "M1"]),
    most_true_positives: evaluationRanking("true_positive", "max", ["M1"], { M1: 2570, M2: 2321, M3: 2522 }, ["M1", "M3", "M2"]),
    most_false_positives: evaluationRanking("false_positive", "max", ["M1"], { M1: 7757, M2: 4573, M3: 6796 }, ["M1", "M3", "M2"]),
  },
  summary: [
    "M2 achieved the strongest overall F1/Dice (0.3818); M2 had the best IoU (0.2359).",
    "M1 predicted the most fire pixels (10,332), while M2 predicted the fewest (6,914). Predicting more fire is not evidence of better performance.",
    "M1 detected the most actual fire pixels (TP=2,570); M1 produced the most false alarms (FP=7,757).",
    "Accuracy was highest for M2 (0.9814), but class imbalance makes accuracy misleading for this task.",
    "These results cover 100 samples from the test split; 100 had valid target pixels and 0 all-unknown samples were excluded from metrics. This is a fixed-seed subset, not a full test-set evaluation or a statistical claim.",
  ],
  elapsed_seconds: 6.78,
};
