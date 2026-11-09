export type ColumnType = "numeric" | "date" | "text";
export type Column = {
  name: string;
  data_type: ColumnType;
  null_count: number;
  unique_count: number;
  invalid_count: number;
};
export type DataValue = string | number | boolean | null;
export type Dataset = {
  id: string;
  filename: string;
  size_bytes: number;
  validation_status: string;
  row_count: number;
  column_count: number;
  columns: Column[];
  null_count: number;
  duplicate_count: number;
  preview?: Array<Record<string, DataValue>>;
};
export type Eda = {
  statistics: Record<string, {
    mean: number | null;
    median: number | null;
    std: number | null;
    min: number | null;
    max: number | null;
    null_count: number;
  }>;
  correlation: Record<string, Record<string, number | null>>;
  duplicate_count: number;
  null_count: number;
  series: Array<{ date: string; value: number }>;
  series_date_column: string | null;
  date_columns: string[];
  histograms: Record<string, Array<{ start: number; end: number; count: number }>>;
  boxplots: Record<string, Boxplot | null>;
  acf: { values: number[]; confidence: number | null };
  period: number | null;
  decomposition: {
    period: number;
    points: Array<{
      date: string;
      observed: number | null;
      trend: number | null;
      seasonal: number | null;
      residual: number | null;
    }>;
  } | null;
};
export type Boxplot = {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  whisker_low: number;
  whisker_high: number;
  outlier_count: number;
  outliers: number[];
};
export type Metrics = { mae: number; mse: number; rmse: number; r2: number | null };
export type SeriesFilter = { column: string; value: string; match?: "exact" | "prefix" };
export type SeriesProfile = {
  row_count: number;
  unique_dates: number;
  date_column: string | null;
  single_series: boolean;
  trainable: boolean;
  suggested_lookback: number | null;
  minimum_rows: number;
  story: string;
  categories: Array<{
    name: string;
    unique_count: number;
    truncated: boolean;
    options: Array<{ value: string; count: number; match: "exact" | "prefix" }>;
  }>;
};
export type TrainConfig = {
  date_column: string;
  target_column: string;
  feature_columns: string[];
  lookback: number;
  epochs: number;
  batch_size: number;
  units: number;
  dropout: number;
  train_ratio: number;
  validation_ratio: number;
  test_ratio: number;
  missing_strategy: "drop" | "forward_fill" | "interpolate";
  patience: number;
  forecast_steps: number;
  filters: SeriesFilter[];
};
export type TrainingJob = {
  id: string;
  dataset_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  current_epoch: number;
  total_epochs: number;
  train_loss: number | null;
  validation_loss: number | null;
  loss_history: number[];
  validation_loss_history: number[];
  error: string | null;
};
export type RunRecord = {
  id: string;
  finishedAt: string;
  target: string;
  predictors: number;
  lookback: number;
  epochs: number;
  units: number;
  metrics: Metrics;
  baseline: Metrics;
};
export type TrainingResults = {
  metrics: Metrics;
  baseline_metrics: Metrics;
  losses: { train: number[]; validation: number[] };
  predictions: Array<{ date: string; actual: number; predicted: number; naive: number }>;
  series: Array<{ date: string; actual: number }>;
  split_dates: {
    start: string;
    train_end: string;
    validation_start: string;
    validation_end: string;
    test_start: string;
    end: string;
  };
  forecast: Array<{ date: string; predicted: number }>;
  forecast_note: string | null;
  summary: string;
};
