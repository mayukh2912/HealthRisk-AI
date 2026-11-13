"use client";
import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { Analytics, ModelInfo } from "@/lib/types";

type State<T> = { data: T | null; error: string | null; loading: boolean };
const cache = new Map<string, Promise<unknown>>();

function cached<T>(key: string, fn: () => Promise<T>): Promise<T> {
  if (!cache.has(key)) {
    const p = fn().catch((e) => {
      cache.delete(key); // allow retry after a failure
      throw e;
    });
    cache.set(key, p);
  }
  return cache.get(key) as Promise<T>;
}

function useCached<T>(key: string, fn: () => Promise<T>): State<T> & { retry: () => void } {
  const [state, setState] = useState<State<T>>({ data: null, error: null, loading: true });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let alive = true;
    cached(key, fn)
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((e: unknown) =>
        alive && setState({ data: null, error: e instanceof ApiError ? e.message : "Unexpected error", loading: false }),
      );
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, attempt]);
  return { ...state, retry: () => { setState({ data: null, error: null, loading: true }); setAttempt((a) => a + 1); } };
}

export const useAnalytics = () => useCached<Analytics>("analytics", api.analytics);
export const useModelInfo = () => useCached<ModelInfo>("model-info", api.modelInfo);
export const __clearApiCache = () => cache.clear();
