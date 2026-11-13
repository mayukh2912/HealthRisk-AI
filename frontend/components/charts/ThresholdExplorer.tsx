"use client";
import { useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ThresholdRow } from "@/lib/types";
import { ConfusionMatrix } from "./ConfusionMatrix";
import { C } from "./colors";

export function ThresholdExplorer({ table: fullTable, defaultThreshold }: { table: ThresholdRow[]; defaultThreshold: number }) {
  // Hide thresholds that flag < 50 test encounters: precision there is too noisy to be meaningful.
  const table = fullTable.filter((r) => r.tp + r.fp >= 50);
  const closest = (t: number) => table.reduce((a, b) => (Math.abs(b.threshold - t) < Math.abs(a.threshold - t) ? b : a));
  const [t, setT] = useState(closest(defaultThreshold).threshold);
  const row = useMemo(() => closest(t), [t]); // eslint-disable-line react-hooks/exhaustive-deps
  const total = row.tp + row.fn + row.fp + row.tn;
  const per1000 = (v: number) => Math.round((v / total) * 1000);
  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <div className="lg:col-span-3">
        <label htmlFor="thr" className="flex items-center justify-between text-sm font-medium text-navy-900">
          Probability threshold
          <span className="rounded bg-navy-900 px-2 py-0.5 font-mono text-xs text-white" data-testid="threshold-value">{row.threshold.toFixed(2)}</span>
        </label>
        <input id="thr" type="range" min={table[0].threshold} max={table[table.length - 1].threshold} step={0.01} value={t}
          onChange={(e) => setT(Number(e.target.value))} className="mt-2 w-full accent-teal-600" aria-valuetext={`threshold ${row.threshold.toFixed(2)}`} />
        <div className="mt-1 flex justify-between text-[11px] text-slate-500">
          <span>Lower: flag more patients</span>
          <button className="underline" onClick={() => setT(closest(defaultThreshold).threshold)}>Reset to model default ({defaultThreshold.toFixed(3)})</button>
          <span>Higher: fewer alerts</span>
        </div>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={table} margin={{ top: 16, right: 12, left: 0, bottom: 4 }}>
            <CartesianGrid stroke={C.grid} />
            <XAxis dataKey="threshold" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => v.toFixed(2)} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} labelFormatter={(l) => `Threshold ${Number(l).toFixed(2)}`} />
            <Legend />
            <ReferenceLine x={row.threshold} stroke={C.navy} strokeDasharray="3 3" />
            <Line dataKey="recall" name="Recall (sensitivity)" stroke={C.teal} strokeWidth={2} dot={false} />
            <Line dataKey="precision" name="Precision (PPV)" stroke={C.navy} strokeWidth={2} dot={false} />
            <Line dataKey="specificity" name="Specificity" stroke={C.grey} strokeWidth={1.5} dot={false} />
            <Line dataKey="alert_rate" name="Share flagged" stroke={C.amber} strokeWidth={1.5} strokeDasharray="5 3" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="space-y-4 lg:col-span-2">
        <div className="grid grid-cols-2 gap-2 text-sm">
          {[["Recall", row.recall], ["Precision", row.precision], ["Specificity", row.specificity], ["Flagged", row.alert_rate]].map(([k, v]) => (
            <div key={k as string} className="rounded-lg bg-slate-50 p-2.5">
              <p className="text-[11px] uppercase tracking-wide text-slate-500">{k}</p>
              <p className="text-lg font-semibold tabular-nums text-navy-900" data-testid={`thr-${String(k).toLowerCase()}`}>{((v as number) * 100).toFixed(1)}%</p>
            </div>
          ))}
        </div>
        <ConfusionMatrix cm={{ tp: row.tp, fp: row.fp, fn: row.fn, tn: row.tn }} />
        <p className="text-xs leading-relaxed text-slate-600">
          Per 1,000 test encounters at this threshold: <b>{per1000(row.tp)}</b> readmissions correctly flagged,
          <b> {per1000(row.fn)}</b> missed, and <b>{per1000(row.fp)}</b> alerts for patients who were not readmitted.
          Lowering the threshold catches more readmissions but generates more unnecessary follow-up workload.
        </p>
      </div>
    </div>
  );
}
