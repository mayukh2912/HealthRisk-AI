"use client";
import { useCallback, useEffect, useState } from "react";

/**
 * Browser-local log of predictions made in THIS browser (no server-side state on a
 * serverless deployment). Stores only aggregate-safe fields: timestamp, probability,
 * category, model version, source. Never stores patient inputs.
 */
export type LogEntry = { ts: string; probability: number; category: string; version: string; source: "single" | "batch" };
const KEY = "healthrisk.predictionLog.v1";
const MAX = 5000;

function read(): LogEntry[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as LogEntry[]) : [];
  } catch {
    return [];
  }
}

export function appendPredictionLog(entries: LogEntry[]) {
  try {
    const next = [...read(), ...entries].slice(-MAX);
    window.localStorage.setItem(KEY, JSON.stringify(next));
    window.dispatchEvent(new Event("healthrisk-log"));
  } catch {
    /* storage unavailable (private mode) - monitoring simply shows no local history */
  }
}

export function usePredictionLog() {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  useEffect(() => {
    const sync = () => setEntries(read());
    sync();
    window.addEventListener("healthrisk-log", sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener("healthrisk-log", sync);
      window.removeEventListener("storage", sync);
    };
  }, []);
  const clear = useCallback(() => {
    try {
      window.localStorage.removeItem(KEY);
    } catch {
      /* ignore */
    }
    setEntries([]);
  }, []);
  return { entries, clear };
}
