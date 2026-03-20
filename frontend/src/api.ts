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

function normalizeBaseUrl(url: string) {
  const trimmed = url.trim().replace(/\/+$/, "");

  // Permite "/api" (proxy) o "http://localhost:8000"
  if (trimmed.startsWith("/")) return trimmed;

  // Si viene sin http/https (ej: "backend:8000"), lo convertimos a http://...
  if (!/^https?:\/\//i.test(trimmed)) return `http://${trimmed}`;

  return trimmed;
}

/**
 * En contenedores: usa proxy nginx -> backend mediante "/api"
 * Puedes sobreescribir con VITE_API_BASE_URL (ej: "/api" o "http://localhost:8000")
 */
export function getApiBaseUrl() {
  const env = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (env && env.trim().length > 0) return normalizeBaseUrl(env);
  return "/api";
}

async function parseJsonSafe(res: Response) {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

export async function healthCheck() {
  const res = await fetch(`${getApiBaseUrl()}/`, { method: "GET" });
  const data = await parseJsonSafe(res);

  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status} ${JSON.stringify(data)}`);
  }

  return data;
}

export async function predict(file: File): Promise<PredictResponse> {
  const fd = new FormData();
  fd.append("file", file, file.name);

  const res = await fetch(`${getApiBaseUrl()}/predict`, { method: "POST", body: fd });
  const data = await parseJsonSafe(res);

  if (!res.ok) throw new Error(data?.detail ?? `Predict failed: ${res.status}`);
  return data as PredictResponse;
}

export async function multipredict(files: File[]): Promise<MultiPredictResponse> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);

  const res = await fetch(`${getApiBaseUrl()}/multipredict`, { method: "POST", body: fd });
  const data = await parseJsonSafe(res);

  if (!res.ok) throw new Error(data?.detail ?? `MultiPredict failed: ${res.status}`);
  return data as MultiPredictResponse;
}