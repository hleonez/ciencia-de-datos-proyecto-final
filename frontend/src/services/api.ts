import type { Dataset, Eda, TrainingJob, TrainingResults, TrainConfig } from "../types";

const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch (reason) {
    if (reason instanceof TypeError) {
      throw new Error(
        "No se pudo conectar con la API. Inicia FastAPI desde backend con «uvicorn app.main:app --reload» y vuelve a intentarlo.",
      );
    }
    throw reason;
  }
  if (!response.ok) {
    let message = `Error del servidor (${response.status}).`;
    try {
      const body = await response.json() as { detail?: string };
      message = body.detail || message;
    } catch {
      // The response may not contain JSON.
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<Dataset>("/datasets", { method: "POST", body });
  },
  eda: (datasetId: string, column: string, dateColumn = "", period = 0) => {
    const params = new URLSearchParams();
    if (column) params.set("column", column);
    if (dateColumn) params.set("date_column", dateColumn);
    if (period >= 2) params.set("period", String(period));
    const query = params.toString();
    return request<Eda>(`/datasets/${datasetId}/eda${query ? `?${query}` : ""}`);
  },
  cancelJob: (jobId: string) =>
    request<TrainingJob>(`/jobs/${jobId}/cancel`, { method: "POST" }),
  deleteDataset: (datasetId: string) =>
    request<void>(`/datasets/${datasetId}`, { method: "DELETE" }),
  startTraining: (datasetId: string, config: TrainConfig) =>
    request<{ job_id: string; status: TrainingJob["status"] }>(
      `/datasets/${datasetId}/train`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(config) },
    ),
  job: (jobId: string) => request<TrainingJob>(`/jobs/${jobId}`),
  results: (jobId: string) => request<TrainingResults>(`/jobs/${jobId}/results`),
  predictionsUrl: (jobId: string) => `${API_URL}/jobs/${jobId}/predictions.csv`,
};
