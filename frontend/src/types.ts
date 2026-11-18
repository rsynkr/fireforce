export type Split = "train" | "eval" | "test";
export type ModelName = "M1" | "M2" | "M3";
export type ModelStatus = "ok" | "unavailable" | "error";

export interface SampleSummary {
  sample_id: string;
  split: Split;
  shard: string;
  record_index: number;
  target_valid_pixels: number;
  target_fire_pixels: number;
  target_unknown_pixels: number;
  all_unknown: boolean;
}

export interface SampleDetail extends SampleSummary {
  available_features: string[];
  models: ModelName[];
  previous_fire_mask: number[][];
  target_mask: number[][];
  target_valid_mask: boolean[][];
}

export interface SampleListResponse {
  total: number;
  limit: number;
  offset: number;
  items: SampleSummary[];
}

export interface ModelHealth {
  status: "ready" | "unavailable";
  checkpoint: string;
  in_channels: number;
  threshold: number;
  sha256: string | null;
  error: string | null;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "unavailable";
  device: string;
  cuda_available: boolean;
  threshold_provenance: string;
  data: {
    ready: boolean;
    manifest_version: number | null;
    fingerprint: string | null;
    shard_count: number | null;
    sample_count: number | null;
    split_counts: Record<string, number> | null;
    error: string | null;
  };
  models: Record<ModelName, ModelHealth>;
}

export interface Metrics {
  available: boolean;
  reason: string | null;
  valid_pixels: number;
  true_positive: number | null;
  false_positive: number | null;
  false_negative: number | null;
  true_negative: number | null;
  precision: number | null;
  recall: number | null;
  dice: number | null;
  iou: number | null;
}

export interface ModelPrediction {
  model_id: ModelName;
  status: ModelStatus;
  threshold: number;
  threshold_provenance: string;
  checkpoint: {
    file: string;
    sha256: string | null;
    in_channels: number;
  };
  inference_ms: number | null;
  probabilities: number[][] | null;
  mask: boolean[][] | null;
  predicted_fire_pixels: number | null;
  metrics: Metrics | null;
  error: string | null;
}

export interface GroundTruth {
  available: boolean;
  target_mask: number[][];
  valid_mask: boolean[][];
  valid_pixels: number;
  fire_pixels: number;
  unknown_pixels: number;
}

export interface PredictionResponse {
  sample_id: string;
  split: Split;
  shard: string;
  record_index: number;
  status: "ok" | "partial" | "failed";
  device: string;
  duration_ms: number;
  available_features: string[];
  ground_truth: GroundTruth;
  models: ModelPrediction[];
}

export interface EvaluationRanking {
  metric: string;
  direction: "max" | "min";
  winners: ModelName[];
  values: Record<ModelName, number>;
  order: ModelName[];
}

export interface EvaluationModel {
  threshold: number;
  in_channels: number;
  micro: {
    precision: number;
    recall: number;
    f1_dice: number;
    iou: number;
    accuracy: number;
    valid_pixels: number;
  };
  macro: {
    precision: number | null;
    recall: number | null;
    f1_dice: number | null;
    iou: number | null;
    accuracy: number | null;
    samples: number;
  };
  counts: {
    true_positive: number;
    false_positive: number;
    false_negative: number;
    true_negative: number;
  };
  predicted: {
    total: number;
    mean: number;
    median: number;
    max: number;
    on_valid_pixels: number;
    on_unknown_pixels: number;
    percent_of_valid_pixels: number;
  };
}

export interface EvaluationSummary {
  evaluation_version: string;
  split: Split;
  sampling: {
    strategy: string;
    seed: number;
    count: number;
  };
  counts: {
    selected: number;
    valid_samples: number;
    all_unknown_samples: number;
    excluded_sample_ids: string[];
    valid_pixels: number;
    actual_fire_pixels: number;
  };
  thresholds: Record<ModelName, number>;
  threshold_provenance: string;
  models: Record<ModelName, EvaluationModel>;
  rankings: Record<string, EvaluationRanking>;
  summary: string[];
  elapsed_seconds: number;
}
