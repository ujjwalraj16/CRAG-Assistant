import { useCallback, useEffect, useState } from 'react';
import { api } from '../services/api';

export function useHealth() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refreshHealth = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getHealth();
      setHealth(data);
    } catch (err) {
      setError(err.message || 'Backend unreachable');
      setHealth(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
  }, [refreshHealth]);

  return { health, loading, error, refreshHealth };
}

