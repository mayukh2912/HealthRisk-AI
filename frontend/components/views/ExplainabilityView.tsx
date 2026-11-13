"use client";
import { useState } from "react";
import { Loader2, Sparkles } from "lucide-react";
import Link from "next/link";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useAnalytics } from "@/hooks/useApi";
import { api, ApiError } from "@/lib/api";
import { EXAMPLE_INPUT } from "@/lib/schema";
import { toPayload } from "@/lib/validation";
import type { PredictionResponse } from "@/lib/types";
import { Card, PageHeader, SectionLabel } from "@/components/ui/Card";
import { Callout, ErrorState, Loading } from "@/components/ui/States";
import { ImportanceChart } from "@/components/charts/ImportanceChart";
import { C } from "@/components/charts/colors";
import { PredictionResult } from "./PredictionResult";

export function ExplainabilityView() {
  const { data, error, loading, retry } = useAnalytics();
  const [example, setExample] = useState<PredictionResponse | null>(null);
  const [exErr, setExErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  if (loading) return <Loading />;
  if (error || !data) return <ErrorState message={error ?? "No data"} onRetry={retry} />;
  const x = data.explainability;
  const maxAbs = Math.max(...x.shap_distribution.map((d) => Math.max(Math.abs(d.p5), Math.abs(d.p95))));

  async function runExample() {
    setBusy(true); setExErr(null);
    try { setExample(await api.predict(toPayload(EXAMPLE_INPUT))); }
    catch (e) { setExErr(e instanceof ApiError ? e.message : "Request failed"); }
    finally { setBusy(false); }
  }

  return (
    <div>
      <PageHeader title="Explainable AI" description={`Global explanations computed offline with ${x.method} on ${x.sample_size.toLocaleString()} held-out test encounters; individual explanations are computed exactly at prediction time.`} />
      <Callout><b>Association, not causation.</b> {x.interpretation_note}</Callout>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="SHAP summary plot" subtitle="Each dot is one test encounter; colour = feature value (numeric features)">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`/${x.summary_plot}`} alt="SHAP beeswarm summary plot of the top 15 features" className="w-full rounded" />
        </Card>
        <Card title="Global feature importance" subtitle="Mean |SHAP| — average magnitude of impact on the model's log-odds">
          <ImportanceChart data={x.mean_abs_shap.slice(0, 15)} valueKey="mean_abs_shap" label="Mean |SHAP|" color={C.navy} />
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="Feature impact distribution" subtitle="Spread of per-encounter SHAP values (5th–95th percentile, box = interquartile range)">
          <div className="space-y-2">
            {x.shap_distribution.map((d) => {
              const pos = (v: number) => `${50 + (v / maxAbs) * 48}%`;
              return (
                <div key={d.feature} className="grid grid-cols-[150px_1fr_52px] items-center gap-2 text-xs">
                  <span className="truncate text-slate-700" title={d.label}>{d.label}</span>
                  <div className="relative h-4 rounded bg-slate-50" role="img" aria-label={`${d.label}: SHAP from ${d.p5.toFixed(2)} to ${d.p95.toFixed(2)}, median ${d.median.toFixed(3)}`}>
                    <div className="absolute inset-y-0 left-1/2 w-px bg-slate-300" />
                    <div className="absolute top-1/2 h-px bg-slate-400" style={{ left: pos(d.p5), right: `calc(100% - ${pos(d.p95)})` }} />
                    <div className="absolute inset-y-0.5 rounded-sm bg-teal-500/70" style={{ left: pos(d.p25), right: `calc(100% - ${pos(d.p75)})`, minWidth: 2 }} />
                    <div className="absolute inset-y-0 w-0.5 bg-navy-900" style={{ left: pos(d.median) }} />
                  </div>
                  <span className="text-right tabular-nums text-slate-500" title="share of encounters where this feature increased predicted risk">{Math.round(d.share_positive * 100)}% ↑</span>
                </div>
              );
            })}
          </div>
          <p className="mt-3 text-[11px] text-slate-500">Left of centre lowers predicted risk; right raises it. ‘% ↑’ = share of encounters where the feature pushed risk up.</p>
        </Card>
        <Card title="Permutation importance" subtitle="Drop in test PR-AUC when the feature is shuffled (model-agnostic check)">
          <ImportanceChart data={x.permutation_importance_pr_auc.slice(0, 12)} valueKey="importance" errorKey="std" label="Decrease in PR-AUC" color={C.teal} />
        </Card>
      </div>

      <SectionLabel>How predicted risk changes with key numeric features (mean SHAP by value)</SectionLabel>
      <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
        {x.dependence.map((d) => (
          <Card key={d.feature} title={d.label}>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={d.points} margin={{ top: 4, right: 8, left: -10, bottom: 4 }}>
                <CartesianGrid stroke={C.grid} />
                <XAxis dataKey="value" type="number" domain={["dataMin", "dataMax"]} />
                <YAxis tickFormatter={(v) => Number(v).toFixed(2)} />
                <ReferenceLine y={0} stroke="#94a3b8" />
                <Tooltip formatter={(v) => Number(v).toFixed(3)} labelFormatter={(l) => `${d.label} = ${l}`} />
                <Line dataKey="mean_shap" name="Mean SHAP" stroke={C.navy} strokeWidth={2} dot={{ r: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        ))}
      </div>

      <SectionLabel>Individual prediction explanation</SectionLabel>
      <Card title="Explain a prediction" subtitle="Runs the deployed model on the illustrative example input (not a real patient) and shows exact per-feature contributions">
        <div className="flex flex-wrap items-center gap-3">
          <button onClick={runExample} disabled={busy} className="inline-flex items-center gap-2 rounded-lg bg-navy-900 px-4 py-2 text-sm font-semibold text-white hover:bg-navy-800 disabled:opacity-60">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />} Explain example prediction
          </button>
          <Link href="/risk-assessment" className="text-sm font-medium text-teal-700 underline">or enter your own values →</Link>
        </div>
        {exErr && <div className="mt-4"><ErrorState message={exErr} /></div>}
        {example && <div className="mt-4"><PredictionResult result={example} /></div>}
      </Card>
    </div>
  );
}
