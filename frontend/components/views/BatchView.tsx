"use client";
import { useRef, useState } from "react";
import { Download, FileDown, Loader2, Upload } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { checkBatchCsv, downloadText, predictionsToCsv, TEMPLATE_CSV, type CsvCheck } from "@/lib/csv";
import { MAX_BATCH_ROWS, REQUIRED_CSV_COLUMNS } from "@/lib/schema";
import type { BatchResponse } from "@/lib/types";
import { appendPredictionLog } from "@/hooks/usePredictionLog";
import { Badge, Card, PageHeader, StatCard } from "@/components/ui/Card";
import { Callout, ErrorState } from "@/components/ui/States";
import { pct } from "@/lib/format";

export function BatchView() {
  const [fileName, setFileName] = useState<string | null>(null);
  const [text, setText] = useState<string | null>(null);
  const [check, setCheck] = useState<CsvCheck | null>(null);
  const [result, setResult] = useState<BatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  async function onFile(f: File | undefined) {
    setResult(null); setError(null);
    if (!f) return;
    if (!/\.csv$/i.test(f.name)) { setCheck(null); setText(null); setError("Please choose a .csv file."); return; }
    const t = await f.text();
    setFileName(f.name); setText(t); setCheck(checkBatchCsv(t));
  }

  async function run() {
    if (!text || !check?.ok) return;
    setBusy(true); setError(null);
    try {
      const res = await api.batchPredict(text);
      setResult(res);
      appendPredictionLog(res.predictions.map((p) => ({ ts: res.summary.timestamp, probability: p.predicted_probability, category: p.risk_category, version: p.model_version, source: "batch" as const })));
    } catch (e) {
      setError(e instanceof ApiError ? [e.message, ...e.details.slice(1)].join(" · ") : "Batch prediction failed");
    } finally { setBusy(false); }
  }

  return (
    <div>
      <PageHeader title="Batch Prediction" description={`Upload a CSV of de-identified encounters (max ${MAX_BATCH_ROWS.toLocaleString()} rows, 4 MB) to score them with the deployed model and download the results.`}>
        <button onClick={() => downloadText("healthrisk_template.csv", TEMPLATE_CSV)} className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium text-teal-700 ring-1 ring-teal-100 hover:bg-teal-50">
          <FileDown className="h-4 w-4" /> Download CSV template
        </button>
      </PageHeader>
      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="1 · Upload" className="lg:col-span-1">
          <label htmlFor="csv" className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-slate-300 p-6 text-center text-sm text-slate-600 hover:border-teal-500">
            <Upload className="h-6 w-6 text-teal-600" />
            <span>{fileName ?? "Choose a CSV file"}</span>
            <input ref={input} id="csv" data-testid="csv-input" type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => onFile(e.target.files?.[0])} />
          </label>
          {check && (
            <div className="mt-4 space-y-2 text-sm" data-testid="csv-check">
              <p>{check.rowCount.toLocaleString()} data rows · {check.header.length} columns</p>
              {check.errors.map((e) => <p key={e} role="alert" className="text-red-700">✕ {e}</p>)}
              {check.warnings.map((w) => <p key={w} className="text-amber-700">! {w}</p>)}
              {check.ok && <p className="text-teal-700">✓ Columns look valid</p>}
            </div>
          )}
          <button onClick={run} disabled={!check?.ok || busy} className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-navy-900 px-4 py-2 text-sm font-semibold text-white hover:bg-navy-800 disabled:opacity-50">
            {busy && <Loader2 className="h-4 w-4 animate-spin" />} Score encounters
          </button>
          <details className="mt-4 text-xs text-slate-600">
            <summary className="cursor-pointer font-medium">Required columns</summary>
            <p className="mt-2 break-words font-mono">{REQUIRED_CSV_COLUMNS.join(", ")}</p>
            <p className="mt-2">Optional: record_id (your own non-identifying reference), medical_specialty, diag_2, diag_3 and other diabetes medication columns. Names, phone numbers, addresses and other direct identifiers are rejected.</p>
          </details>
        </Card>
        <div className="space-y-4 lg:col-span-2">
          {error && <ErrorState message={error} />}
          {!result && !error && <Callout>Rows are validated individually; invalid rows are reported and skipped, valid rows are scored. Nothing is stored on the server.</Callout>}
          {result && (
            <>
              <div className="grid gap-4 sm:grid-cols-4">
                <StatCard label="Rows scored" value={result.summary.rows_scored.toLocaleString()} hint={`of ${result.summary.rows_received.toLocaleString()} received`} />
                <StatCard label="Rejected" value={result.summary.rows_rejected.toLocaleString()} tone="slate" />
                <StatCard label="Flagged" value={result.summary.flagged.toLocaleString()} hint={`threshold ${result.summary.decision_threshold}`} tone="teal" />
                <StatCard label="Mean probability" value={pct(result.summary.mean_probability)} tone="teal" />
              </div>
              <Card title="2 · Results" subtitle={`Model v${result.summary.model_version}`} actions={
                <button data-testid="download-results" onClick={() => downloadText("healthrisk_predictions.csv", predictionsToCsv(result.predictions))} disabled={!result.predictions.length}
                  className="inline-flex items-center gap-1.5 rounded-md bg-teal-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-teal-700 disabled:opacity-50">
                  <Download className="h-3.5 w-3.5" /> Download CSV
                </button>
              }>
                <div className="mb-3 flex flex-wrap gap-2 text-xs">
                  {Object.entries(result.summary.risk_categories).map(([k, v]) => <Badge key={k} tone={k === "High" ? "red" : k === "Moderate" ? "amber" : "green"}>{k}: {v}</Badge>)}
                </div>
                <div className="max-h-96 overflow-auto rounded border border-slate-100">
                  <table className="min-w-full text-left text-sm" data-testid="results-table">
                    <thead className="sticky top-0 bg-slate-50 text-xs uppercase text-slate-500">
                      <tr><th className="px-3 py-2">Record</th><th className="px-3 py-2">Probability</th><th className="px-3 py-2">Class</th><th className="px-3 py-2">Risk</th><th className="px-3 py-2">Version</th></tr>
                    </thead>
                    <tbody>
                      {result.predictions.slice(0, 200).map((p) => (
                        <tr key={p.record_id} className="border-t border-slate-100">
                          <td className="px-3 py-1.5 font-mono text-xs">{p.record_id}</td>
                          <td className="px-3 py-1.5 tabular-nums">{pct(p.predicted_probability)}</td>
                          <td className="px-3 py-1.5">{p.predicted_class ? "Flagged" : "Not flagged"}</td>
                          <td className="px-3 py-1.5">{p.risk_category}</td>
                          <td className="px-3 py-1.5 text-xs">{p.model_version}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {result.predictions.length > 200 && <p className="mt-2 text-xs text-slate-500">Showing first 200 rows — download the CSV for all {result.predictions.length.toLocaleString()}.</p>}
              </Card>
              {result.errors.length > 0 && (
                <Card title="Rejected rows" subtitle="Fix these rows and re-upload">
                  <ul className="max-h-60 space-y-1 overflow-auto text-xs text-red-800">
                    {result.errors.map((e) => <li key={e.row}>Row {e.row} ({e.record_id}): {e.errors.join("; ")}</li>)}
                  </ul>
                </Card>
              )}
              <Callout tone="warn">{result.disclaimer}</Callout>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
