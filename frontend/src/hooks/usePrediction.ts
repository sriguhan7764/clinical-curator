import { useState } from 'react';
import { api } from '../api/client';
import type { BatchPredictItem, PredictResponse } from '../types/api';

type Status = 'idle' | 'loading' | 'done' | 'error';

export function usePrediction() {
  const [status, setStatus]           = useState<Status>('idle');
  const [result, setResult]           = useState<PredictResponse | null>(null);
  const [batchResult, setBatchResult] = useState<BatchPredictItem[] | null>(null);
  const [error, setError]             = useState<string | null>(null);

  const predict = async (image: File, modelName: string): Promise<PredictResponse | null> => {
    setStatus('loading');
    setError(null);
    try {
      const res = await api.predict(image, modelName);
      setResult(res);
      setStatus('done');
      return res;
    } catch (e) {
      setError((e as Error).message);
      setStatus('error');
      return null;
    }
  };

  const predictBatch = async (image: File, modelNames: string[]): Promise<BatchPredictItem[] | null> => {
    setStatus('loading');
    setError(null);
    try {
      const res = await api.predictBatch(image, modelNames);
      setBatchResult(res);
      setStatus('done');
      return res;
    } catch (e) {
      setError((e as Error).message);
      setStatus('error');
      return null;
    }
  };

  const reset = () => {
    setStatus('idle');
    setResult(null);
    setBatchResult(null);
    setError(null);
  };

  return { status, result, batchResult, error, predict, predictBatch, reset };
}
