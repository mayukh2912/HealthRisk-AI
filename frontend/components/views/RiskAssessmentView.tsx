"use client";
import { useState } from "react";
import { Eraser, FlaskConical, Loader2, Send } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { EMPTY_INPUT, EXAMPLE_INPUT, NUMERIC_FIELDS, SELECT_FIELDS, TEXT_FIELDS } from "@/lib/schema";
import { toPayload, validateForm, type FormErrors } from "@/lib/validation";
import type { PredictionResponse } from "@/lib/types";
import { appendPredictionLog } from "@/hooks/usePredictionLog";
import { Card, PageHeader } from "@/components/ui/Card";
import { Callout, ErrorState } from "@/components/ui/States";
import { PredictionResult } from "./PredictionResult";

const inputCls = (err?: string) =>
  `mt-1 block w-full rounded-md border bg-white px-2.5 py-1.5 text-sm text-navy-900 shadow-sm focus:border-teal-500 focus:outline-none ${err ? "border-red-400" : "border-slate-300"}`;

function Field({ name, label, error, help, children }: { name: string; label: string; error?: string; help?: string; children: React.ReactNode }) {
  return (
    <div>
      <label htmlFor={name} className="text-xs font-medium text-slate-700">{label}</label>
      {children}
      {error ? <p id={`${name}-error`} role="alert" className="mt-0.5 text-[11px] text-red-700">{error}</p>
        : help ? <p className="mt-0.5 text-[11px] text-slate-500">{help}</p> : null}
    </div>
  );
}

export function RiskAssessmentView() {
  const [values, setValues] = useState<Record<string, string>>(EMPTY_INPUT);
  const [errors, setErrors] = useState<FormErrors>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const set = (k: string, v: string) => {
    setValues((s) => ({ ...s, [k]: v }));
    if (errors[k]) setErrors((e) => { const n = { ...e }; delete n[k]; return n; });
  };

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setApiError(null);
    const errs = validateForm(values);
    setErrors(errs);
    if (Object.keys(errs).length) return;
    setBusy(true);
    try {
      const res = await api.predict(toPayload(values));
      setResult(res);
      appendPredictionLog([{ ts: res.timestamp, probability: res.probability, category: res.risk_category, version: res.model_version, source: "single" }]);
    } catch (err) {
      setResult(null);
      setApiError(err instanceof ApiError ? [err.message, ...err.details.slice(1)].join(" · ") : "Prediction failed");
    } finally {
      setBusy(false);
    }
  }

  const nErr = Object.keys(errors).length;
  return (
    <div>
      <PageHeader title="Patient Risk Assessment"
        description="Enter discharge-time encounter information to estimate the probability of readmission within 30 days. Do not enter names or other direct identifiers — none are needed." />
      <div className="grid gap-6 xl:grid-cols-5">
        <form onSubmit={submit} noValidate className="space-y-4 xl:col-span-3" aria-label="Patient risk assessment form">
          <Card title="Encounter details" actions={
            <div className="flex gap-2">
              <button type="button" onClick={() => { setValues(EXAMPLE_INPUT); setErrors({}); }} className="inline-flex items-center gap-1 rounded-md px-2.5 py-1 text-xs font-medium text-teal-700 ring-1 ring-teal-100 hover:bg-teal-50">
                <FlaskConical className="h-3.5 w-3.5" /> Load illustrative example
              </button>
              <button type="button" onClick={() => { setValues(EMPTY_INPUT); setErrors({}); setResult(null); }} className="inline-flex items-center gap-1 rounded-md px-2.5 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200 hover:bg-slate-50">
                <Eraser className="h-3.5 w-3.5" /> Clear
              </button>
            </div>
          }>
            <p className="mb-4 text-xs text-slate-500">The example values are illustrative inputs for demonstration and do not describe a real patient.</p>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {SELECT_FIELDS.slice(0, 6).map((f) => (
                <Field key={f.name} name={f.name} label={f.label} error={errors[f.name]} help={f.help}>
                  <select id={f.name} value={values[f.name]} onChange={(e) => set(f.name, e.target.value)} className={inputCls(errors[f.name])} aria-invalid={!!errors[f.name]}>
                    {f.name !== "medical_specialty" && <option value="">Select…</option>}
                    {f.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                  </select>
                </Field>
              ))}
            </div>
          </Card>
          <Card title="Utilisation and stay" subtitle="Counts for this encounter and the preceding 12 months">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {NUMERIC_FIELDS.map((f) => (
                <Field key={f.name} name={f.name} label={f.label} error={errors[f.name]}>
                  <input id={f.name} inputMode="numeric" type="number" min={f.min} max={f.max} step={1} value={values[f.name]}
                    onChange={(e) => set(f.name, e.target.value)} className={inputCls(errors[f.name])} aria-invalid={!!errors[f.name]} />
                </Field>
              ))}
            </div>
          </Card>
          <Card title="Diagnoses, labs and diabetes medication">
            <div className="grid gap-4 sm:grid-cols-3">
              {TEXT_FIELDS.map((f) => (
                <Field key={f.name} name={f.name} label={f.label} error={errors[f.name]}>
                  <input id={f.name} value={values[f.name]} placeholder={f.placeholder} onChange={(e) => set(f.name, e.target.value)} className={inputCls(errors[f.name])} aria-invalid={!!errors[f.name]} />
                </Field>
              ))}
              {SELECT_FIELDS.slice(6).map((f) => (
                <Field key={f.name} name={f.name} label={f.label} error={errors[f.name]}>
                  <select id={f.name} value={values[f.name]} onChange={(e) => set(f.name, e.target.value)} className={inputCls(errors[f.name])} aria-invalid={!!errors[f.name]}>
                    <option value="">Select…</option>
                    {f.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                  </select>
                </Field>
              ))}
            </div>
          </Card>
          {nErr > 0 && <p role="alert" className="text-sm text-red-700" data-testid="form-error-summary">Please correct {nErr} field{nErr > 1 ? "s" : ""} before submitting.</p>}
          <button type="submit" disabled={busy} className="inline-flex items-center gap-2 rounded-lg bg-navy-900 px-5 py-2.5 text-sm font-semibold text-white shadow hover:bg-navy-800 disabled:opacity-60">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />} Estimate readmission risk
          </button>
        </form>
        <div className="xl:col-span-2">
          {apiError && <ErrorState message={apiError} />}
          {result ? <PredictionResult result={result} /> : !apiError && (
            <Callout>Results appear here: calibrated probability, configurable risk category, model version, timestamp and the factors that drove the estimate.</Callout>
          )}
        </div>
      </div>
    </div>
  );
}
