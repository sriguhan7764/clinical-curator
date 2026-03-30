import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { ModelInfo } from '../types/api';

export function useModels() {
  const [models, setModels]   = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  useEffect(() => {
    api.models()
      .then(setModels)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return { models, loading, error };
}
