"use client";
import { ArrowDownRight, ArrowUpRight, Info } from "lucide-react";
import type { PredictionResponse } from "@/lib/types";
import { Badge, Card } from "@/components/ui/Card";
import { Callout } from "@/components/ui/States";
import { ContributionChart } from "@/components/charts/ContributionChart";
import { dateTime, pct } from "@/lib/format";

const tone = (c: string) => (c === "High" ? "red" : c === "Moderate" ? "amber" : "green") as "red" | "amber" | "green";
const fmt = (v: unknown) => (v === null || v === undefined ? "missing" : typeof v === "number" ? String(+v.toFixed(2)) : String(v));

export function PredictionResult({ result }: { result: PredictionResponse }) {
  const ex = result.explanation;
  return (
    <div className="space-y-4" data-testid="prediction-result">
      <Card title="Estimated 30-day readmission risk" subtitle="Statistical estimate from historical data — not a diagnosis">
        <div className="flex flex-wrap items-end gap-6">
          <div>
            <p className="text-4xl font-semibold tabular-nums text-navy-900" data-testid="probability">{pct(result.probability)}</p>
            <p className="text-xs text-slate-500">estimated probability (calibrated)</p>
          </div>
          <div className="space-y-1">
            <Badge tone={tone(result.risk_category)}><span data-testid="risk-category">{result.risk_category} risk</span></Badge>
            <p className="text-xs text-slate-600">{result.prediction_label} at threshold {result.decision_threshold.toFixed(3)}</p>
          </div>
        </div>
        <div className="mt-4 h-2.5 w-full overflow-hidden rounded-full bg-slate-100" aria-hidden>
          <div className="h-full rounded-full bg-gradient-to-r from-teal-500 to-navy-800" style={{ width: `${Math.min(100, result.probability * 100 * 2)}%` }} />
        </div>
        <p className="mt-1 text-[11px] text-slate-500">Bar scaled to 0–50%.</p>
        <dl className="mt-4 grid grid-cols-2 gap-2 text-xs text-slate-600 sm:grid-cols-3">
          <div><dt className="text-slate-400">Model</dt><dd className="font-medium text-navy-900">{result.model_name}</dd></div>
          <div><dt className="text-slate-400">Version</dt><dd className="font-medium text-navy-900" data-testid="model-version">v{result.model_version}</dd></div>
          <div><dt className="text-slate-400">Timestamp</dt><dd className="font-medium text-navy-900">{dateTime(result.timestamp)}</dd></div>
        </dl>
      </Card>

      {ex && (
        <Card title="Top contributing factors" subtitle={`${ex.method} · units: ${ex.units}`}>
          <ContributionChart items={ex.top_contributors} units={ex.units} />
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div>
              <p className="mb-1.5 flex items-center gap-1 text-xs font-semibold text-red-800"><ArrowUpRight className="h-3.5 w-3.5" /> Associated with higher predicted risk</p>
              <ul className="space-y-1 text-xs">
                {ex.increasing.length ? ex.increasing.map((c) => (
                  <li key={c.feature} className="flex justify-between gap-2"><span>{c.label}: <b>{fmt(c.value)}</b></span><span className="tabular-nums text-red-800">+{c.contribution.toFixed(3)}</span></li>
                )) : <li className="text-slate-500">None among the top factors</li>}
              </ul>
            </div>
            <div>
              <p className="mb-1.5 flex items-center gap-1 text-xs font-semibold text-teal-700"><ArrowDownRight className="h-3.5 w-3.5" /> Associated with lower predicted risk</p>
              <ul className="space-y-1 text-xs">
                {ex.decreasing.length ? ex.decreasing.map((c) => (
                  <li key={c.feature} className="flex justify-between gap-2"><span>{c.label}: <b>{fmt(c.value)}</b></span><span className="tabular-nums text-teal-700">{c.contribution.toFixed(3)}</span></li>
                )) : <li className="text-slate-500">None among the top factors</li>}
              </ul>
            </div>
          </div>
          <p className="mt-4 flex gap-2 text-xs text-slate-500">
            <Info className="h-4 w-4 shrink-0" />
            <span>Contributions start from the average prediction (base value {ex.base_value.toFixed(3)}) and sum to this
            encounter&apos;s score ({ex.output_value.toFixed(3)}). They show what drove the <i>model&apos;s</i> estimate
            (association in historical data), not causes of readmission.</span>
          </p>
        </Card>
      )}
      <Callout tone="warn">{result.disclaimer}</Callout>
    </div>
  );
}
