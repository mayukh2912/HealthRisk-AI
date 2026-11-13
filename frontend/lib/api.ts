import type { Analytics, BatchResponse, DriftResponse, ModelInfo, PredictionResponse } from "./types";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  details: string[];
  constructor(status: number, message: string, details: string[] = []) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

/** Normalise FastAPI error payloads ({detail: string | {detail: [...]}} | {error, detail: [...]}) */
export function extractErrors(body: unknown): string[] {
  if (!body || typeof body !== "object") return [];
  const b = body as Record<string, unknown>;
  const d = (b.detail ?? b) as unknown;
  if (typeof d === "string") return [d];
  if (Array.isArray(d)) {
    return d.map((e) =>
      typeof e === "string" ? e : e && typeof e === "object" && "message" in e
        ? `${(e as { field?: string }).field ? `${(e as { field: string }).field}: ` : ""}${(e as { message: string }).message}`
        : JSON.stringify(e),
    );
  }
  if (d && typeof d === "object") return extractErrors(d);
  return [];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "Could not reach the inference API. Check your connection or that the backend is running.");
  }
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) {
    const details = extractErrors(body);
    throw new ApiError(res.status, details[0] ?? `Request failed (${res.status})`, details);
  }
  return body as T;
}

export const api = {
  health: () => request<{ status: string; model_loaded: boolean; model_version: string | null }>("/api/health"),
  modelInfo: () => request<ModelInfo>("/api/model-info"),
  analytics: () => request<Analytics>("/api/analytics"),
  predict: (payload: Record<string, unknown>) =>
    request<PredictionResponse>("/api/predict?explain=true", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  batchPredict: (csv: string) =>
    request<BatchResponse>("/api/batch-predict", { method: "POST", headers: { "Content-Type": "text/csv" }, body: csv }),
  drift: (csv: string) =>
    request<DriftResponse>("/api/monitoring/drift", { method: "POST", headers: { "Content-Type": "text/csv" }, body: csv }),
};
