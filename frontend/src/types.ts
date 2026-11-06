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
};
export type TrainingJob = {
  id: string;
  dataset_id: string;
  status: "queued" | "running" | "completed" | "failed";
  current_epoch: number;
  total_epochs: number;
  train_loss: number | null;
  validation_loss: number | null;
  loss_history: number[];
  validation_loss_history: number[];
  error: string | null;
};
export type TrainingResults = {
  metrics: { mae: number; mse: number; rmse: number; r2: number | null };
  losses: { train: number[]; validation: number[] };
  predictions: Array<{ date: string; actual: number; predicted: number }>;
  summary: string;
};
