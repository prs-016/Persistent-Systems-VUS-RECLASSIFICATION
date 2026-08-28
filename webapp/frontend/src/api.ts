import type { AiExplainResponse, JobStatusResponse, RunSummary, VariantRow } from "./types";

// In production the frontend build is served by the same FastAPI process
// that exposes /api, so a relative path works. In dev, Vite's proxy (see
// vite.config.ts) forwards /api to the local backend on :8765.
const API_BASE = "";

export class ApiError extends Error {}

export async function submitClassificationJob(
  file: File,
  tissue: string,
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  form.append("tissue", tissue);

  const res = await fetch(`${API_BASE}/api/classify`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(body?.detail ?? `Upload failed (${res.status})`);
  }
  return res.json();
}

export async function fetchJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  if (!res.ok) {
    throw new ApiError(`Could not check run status (${res.status})`);
  }
  return res.json();
}

export async function fetchAiExplanation(
  summary: RunSummary,
  flagged: VariantRow[],
  tissue: string,
): Promise<AiExplainResponse> {
  const res = await fetch(`${API_BASE}/api/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ summary, flagged, tissue }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(body?.detail ?? `Could not generate explanation (${res.status})`);
  }
  return res.json();
}
