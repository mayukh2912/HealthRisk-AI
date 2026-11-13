"use client";
import { useMemo, useState } from "react";
import { Loader2, Trash2, Upload } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useAnalytics, useModelInfo } from "@/hooks/useApi";
import { usePredictionLog } from "@/hooks/usePredictionLog";
import { api, ApiError } from "@/lib/api";
import { checkBatchCsv } from "@/lib/csv";
import type { DriftFeature, DriftResponse } from "@/lib/types";
import { Badge, Card, PageHeader, SectionLabel, StatCard } from "@/components/ui/Card";
import { Callout, ErrorState, Loading } from "@/components/ui/States";
import { DistributionCompare, PredictionHistogram, PsiChart } from "@/components/charts/DriftCharts";
import { C } from "@/components/charts/colors";
import { FEATURE_NAMES, dateTime, num, pct } from "@/lib/format";

const statusTone = (s: string) => (s === "Stable" ? "green" : s.startsWith("Moderate") ? "amber" : "red") as "green" | "amber" | "red";

function DriftPanel({ features, title }: { features: DriftFeature[]; title: string }) {
  const [sel, setSel] = useState(features[0]?.feature);
  const f = features.find((x) => x.feature === sel) ?? features[0];
  const worst = features[0];
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card title={`Feature drift — ${title}`} subtitle="PSI vs. training reference · amber ≥ 0.10 (moderate), red ≥ 0.25 (significant)"
        actions={worst && <Badge tone={statusTone(worst.status)}>max PSI {worst.psi.toFixed(3)}</Badge>}>
        <PsiChart features={features} />
      </Card>
      <Card title="Distribution comparison" actions={
        <select aria-label="Feature" value={f?.feature} onChange={(e) => setSel(e.target.value)} className="rounded border border-slate-300 px-2 py-1 text-xs">
          {features.map((x) => <option key={x.feature} value={x.feature}>{FEATURE_NAMES[x.feature] ?? x.feature}</option>)}
        </select>}>
        {f && <><p className="mb-2 text-xs text-slate-600">PSI {f.psi.toFixed(4)} · <Badge tone={statusTone(f.status)}>{f.status}</Badge></p><DistributionCompare feature={f} /></>}
      </Card>
    </div>
  );
}

export function MonitoringView() {
  const { data, error, loading, retry } = useAnalytics();
  const info = useModelInfo();
  const { entries, clear } = usePredictionLog();
  const [which, setWhich] = useState<"held_out_test" | "latest_20pct_encounters">("held_out_test");
  const [upload, setUpload] = useState<DriftResponse | null>(null);
  const [upErr, setUpErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const byDay = useMemo(() => {
    const m = new Map<string, number>();
    entries.forEach((e) => m.set(e.ts.slice(0, 10), (m.get(e.ts.slice(0, 10)) ?? 0) + 1));
    return [...m.entries()].sort().map(([day, count]) => ({ day, count }));
  }, [entries]);
  const localHist = useMemo(() => {
    const edges = Array.from({ length: 21 }, (_, i) => +(i * 0.05).toFixed(2));
    const counts = new Array(20).fill(0);
    entries.forEach((e) => { counts[Math.min(19, Math.floor(e.probability / 0.05))]++; });
    return { edges, counts };
  }, [entries]);

  if (loading) return <Loading />;
  if (error || !data) return <ErrorState message={error ?? "No data"} onRetry={retry} />;
  const base = data.monitoring.baseline_drift;
  const set = base[which];

  async function onFile(file?: File) {
    setUpErr(null); setUpload(null);
    if (!file) return;
    const text = await file.text();
    const chk = checkBatchCsv(text);
    if (!chk.ok) { setUpErr(chk.errors.join(" ")); return; }
    setBusy(true);
    try { setUpload(await api.drift(text)); }
    catch (e) { setUpErr(e instanceof ApiError ? [e.message, ...e.details.slice(1)].join(" · ") : "Drift check failed"); }
    finally { setBusy(false); }
  }

  const versions = info.data?.registry.versions ?? [];
  return (
    <div>
      <PageHeader title="Model Monitoring" description="Input-feature and prediction drift against the training distribution using the Population Stability Index (PSI), plus model version tracking." />
      <Callout tone="warn">No real-world outcome labels are collected by this prototype, so live <b>performance</b> (accuracy, recall) is not monitored — only data and prediction drift. Performance figures elsewhere come from the retrospective test set.</Callout>

      <SectionLabel>Model version tracking</SectionLabel>
      <Card title="Model registry" subtitle={info.data ? `Production: v${info.data.registry.production ?? info.data.model_version} · config ${info.data.config_sha256.slice(0, 10)}…` : undefined}>
        {info.error ? <ErrorState message={info.error} /> : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm [&_td]:px-2 [&_th]:px-2">
              <thead className="text-xs uppercase text-slate-500"><tr><th className="py-1">Version</th><th>Model</th><th>Trained</th><th className="text-right">Test PR-AUC</th><th className="text-right">Test ROC-AUC</th><th className="text-right">Recall</th><th className="text-right">Brier</th><th>Dataset hash</th></tr></thead>
              <tbody>{versions.map((v) => (
                <tr key={v.model_version} className="border-t border-slate-100">
                  <td className="py-1.5 font-medium">v{v.model_version} {v.model_version === info.data?.registry.production && <Badge tone="teal">production</Badge>}</td>
                  <td>{v.model_name}</td><td className="text-xs">{dateTime(v.trained_at)}</td>
                  <td className="text-right tabular-nums">{num(v.test_pr_auc)}</td><td className="text-right tabular-nums">{num(v.test_roc_auc)}</td>
                  <td className="text-right tabular-nums">{pct(v.test_recall)}</td><td className="text-right tabular-nums">{num(v.test_brier, 4)}</td>
                  <td className="font-mono text-xs">{v.dataset_sha256.slice(0, 12)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Card>

      <SectionLabel>Baseline drift (computed at training time)</SectionLabel>
      <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
        {(["held_out_test", "latest_20pct_encounters"] as const).map((k) => (
          <button key={k} onClick={() => setWhich(k)} className={`rounded-full px-3 py-1 ${which === k ? "bg-navy-900 text-white" : "bg-white text-slate-700 ring-1 ring-slate-200"}`}>
            {k === "held_out_test" ? "Held-out test set" : "Latest 20% of encounters"}
          </button>
        ))}
        <span className="text-xs text-slate-500">n = {set.n.toLocaleString()} · mean prediction {pct(set.mean_prediction)}</span>
      </div>
      <DriftPanel key={which} features={set.features} title={which === "held_out_test" ? "test set" : "latest encounters"} />
      <Card title="Prediction distribution" className="mt-6" subtitle="Share of encounters by predicted probability">
        <PredictionHistogram current={set.prediction_hist} currentLabel={which === "held_out_test" ? "Test set" : "Latest encounters"} />
      </Card>
      <p className="mt-2 text-xs text-slate-500">{base.notes}</p>

      <SectionLabel>Check a new batch for drift</SectionLabel>
      <Card title="Upload de-identified encounters (same CSV format as batch prediction)">
        <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-navy-900 px-4 py-2 text-sm font-semibold text-white hover:bg-navy-800">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />} Choose CSV
          <input type="file" accept=".csv" className="sr-only" onChange={(e) => onFile(e.target.files?.[0])} />
        </label>
        {upErr && <div className="mt-4"><ErrorState message={upErr} /></div>}
        {upload && (
          <div className="mt-6 space-y-6">
            <div className="grid gap-4 sm:grid-cols-3">
              <StatCard label="Rows analysed" value={upload.n_valid.toLocaleString()} hint={`${upload.n_rows - upload.n_valid} invalid rows skipped`} />
              <StatCard label="Mean predicted risk" value={pct(upload.mean_prediction)} tone="teal" />
              <StatCard label="Features with PSI ≥ 0.10" value={upload.features.filter((f) => f.psi >= 0.1).length} tone="slate" />
            </div>
            <DriftPanel features={upload.features} title="uploaded batch" />
            <Card title="Prediction distribution vs training"><PredictionHistogram current={upload.prediction_hist} reference={upload.reference_prediction_hist} currentLabel="Uploaded batch" /></Card>
          </div>
        )}
      </Card>

      <SectionLabel>Prediction volume (this browser only)</SectionLabel>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Predictions per day" subtitle={`${entries.length.toLocaleString()} predictions logged locally — the serverless API stores no prediction history`}
          actions={entries.length > 0 && <button onClick={clear} className="inline-flex items-center gap-1 text-xs text-slate-500 hover:text-red-700"><Trash2 className="h-3 w-3" /> Clear</button>}>
          {entries.length === 0 ? <p className="text-sm text-slate-500">No predictions made in this browser yet. Use the Risk Assessment or Batch pages.</p> : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={byDay}><CartesianGrid stroke={C.grid} vertical={false} /><XAxis dataKey="day" /><YAxis allowDecimals={false} /><Tooltip /><Bar dataKey="count" name="Predictions" fill={C.navy} /></BarChart>
            </ResponsiveContainer>
          )}
        </Card>
        <Card title="Local prediction distribution">
          {entries.length === 0 ? <p className="text-sm text-slate-500">—</p> : <PredictionHistogram current={localHist} currentLabel="This browser" />}
        </Card>
      </div>
    </div>
  );
}
