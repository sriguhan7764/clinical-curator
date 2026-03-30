import type {
  BatchPredictItem,
  HealthResponse,
  LatentResponse,
  ModelInfo,
  PredictResponse,
} from '../types/api';

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? '';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      msg = body.detail ?? msg;
    } catch { /* ignore */ }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: (): Promise<HealthResponse> =>
    request<HealthResponse>('/api/health'),

  models: (): Promise<ModelInfo[]> =>
    request<ModelInfo[]>('/api/models'),

  predict: (image: File, modelName: string): Promise<PredictResponse> => {
    const fd = new FormData();
    fd.append('image', image);
    fd.append('model_name', modelName);
    return request<PredictResponse>('/api/predict', { method: 'POST', body: fd });
  },

  predictBatch: (image: File, modelNames: string[]): Promise<BatchPredictItem[]> => {
    const fd = new FormData();
    fd.append('image', image);
    fd.append('model_names', modelNames.join(','));
    return request<BatchPredictItem[]>('/api/predict/batch', { method: 'POST', body: fd });
  },

  latent: (modelId = 'ae_r3', imageId = 'demo'): Promise<LatentResponse> =>
    request<LatentResponse>(`/api/latent?model=${modelId}&image_id=${imageId}`),

  downloadPdf: async (predictionResult: unknown, patientId: string): Promise<void> => {
    const res = await fetch(`${BASE}/api/report/pdf`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prediction_result: predictionResult, patient_id: patientId }),
    });
    if (!res.ok) throw new Error(`PDF error: ${res.status}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `clinical_curator_${patientId}.pdf`;
    a.click();
    URL.revokeObjectURL(url);
  },
};
