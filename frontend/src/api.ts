export type PredictResponse = {
  predicted_class: string;
  confidence: number;
};

export type MultiPredictItem = {
  filename: string;
  predicted_class: string;
  confidence: number;
};

export type MultiPredictError = {
  filename: string;
  error: string;
};

export type MultiPredictResponse = {
  count: number;
  success: number;
  failed: number;
  results: MultiPredictItem[];
  errors: MultiPredictError[];
};

export function getApiBaseUrl() {
  // En Docker Compose típico: http://backend:8000
  return (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, "") || "http://localhost:8000";
}

async function parseJsonSafe(res: Response) {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

export async function healthCheck() {
  const res = await fetch(`${getApiBaseUrl()}/`);
  const data = await parseJsonSafe(res);
  if (!res.ok) throw new Error(`Health check failed: ${res.status} ${JSON.stringify(data)}`);
  return data;
}

export async function predict(file: File): Promise<PredictResponse> {
  const fd = new FormData();
  fd.append("file", file, file.name);

  const res = await fetch(`${getApiBaseUrl()}/predict`, {
    method: "POST",
    body: fd
  });

  const data = await parseJsonSafe(res);
  if (!res.ok) throw new Error(data?.detail ?? `Predict failed: ${res.status}`);
  return data as PredictResponse;
}

export async function multipredict(files: File[]): Promise<MultiPredictResponse> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);

  const res = await fetch(`${getApiBaseUrl()}/multipredict`, {
    method: "POST",
    body: fd
  });

  const data = await parseJsonSafe(res);
  if (!res.ok) throw new Error(data?.detail ?? `MultiPredict failed: ${res.status}`);
  return data as MultiPredictResponse;
}