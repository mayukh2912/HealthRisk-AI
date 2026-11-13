"use client";
import { Activity, Database, Gauge, Layers, Target, Users } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useAnalytics } from "@/hooks/useApi";
import { Card, PageHeader, SectionLabel, StatCard } from "@/components/ui/Card";
import { Callout, ErrorState, Loading } from "@/components/ui/States";
import { RateChart } from "@/components/charts/RateChart";
import { ImportanceChart } from "@/components/charts/ImportanceChart";
import { C } from "@/components/charts/colors";
import { int, num, pct } from "@/lib/format";

export function DashboardView() {
  const { data, error, loading, retry } = useAnalytics();
  if (loading) return <Loading label="Loading analytics…" />;
  if (error || !data) return <ErrorState message={error ?? "No analytics available"} onRetry={retry} />;
  const h = data.dataset.historical;
  const t = data.evaluation.test_metrics;
  const raw = data.dataset.historical.raw_label_distribution;
  const labelRows = [
    { label: "Readmitted < 30 days", key: "<30", color: C.red },
    { label: "Readmitted > 30 days", key: ">30", color: C.grey },
    { label: "No readmission recorded", key: "NO", color: C.navy },
  ].map((r) => ({ ...r, count: raw[r.key] ?? 0, share: (raw[r.key] ?? 0) / h.encounters }));
  const top = data.explainability.mean_abs_shap.slice(0, 10);

  return (
    <div>
      <PageHeader title="Executive Dashboard"
        description="Overview of the historical diabetes encounter cohort and the deployed 30-day readmission risk model. Historical statistics and model test metrics are shown separately." />

      <SectionLabel>Historical dataset · 130 US hospitals, 1999–2008 (modelling cohort)</SectionLabel>
      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard label="Encounters analysed" value={int(h.encounters)} hint={`${int(h.patients)} unique patients`} icon={<Database className="h-4 w-4" />} />
        <StatCard label="Historical 30-day readmission rate" value={pct(h.readmission_rate_30d)} hint={`${int(h.positives)} encounters · imbalance ${h.imbalance_ratio}:1`} icon={<Users className="h-4 w-4" />} />
        <StatCard label="Excluded (expired / hospice)" value={int(data.dataset.cohort.removed_expired_or_hospice)} hint="not at risk of readmission" icon={<Layers className="h-4 w-4" />} tone="slate" />
      </div>

      <SectionLabel>Deployed model · held-out, patient-disjoint test set</SectionLabel>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Selected model" value={<span className="text-xl">{data.model.name}</span>} hint={`Chosen by cross-validated PR-AUC`} icon={<Activity className="h-4 w-4" />} tone="teal" />
        <StatCard label="Test recall" value={pct(t.recall)} hint={`at threshold ${num(t.threshold, 3)} · precision ${pct(t.precision)}`} icon={<Target className="h-4 w-4" />} tone="teal" />
        <StatCard label="Test PR-AUC" value={num(t.pr_auc, 3)} hint={`no-skill baseline ${num(t.prevalence, 3)} · ROC-AUC ${num(t.roc_auc, 3)}`} icon={<Gauge className="h-4 w-4" />} tone="teal" />
        <StatCard label="Model version" value={`v${data.model.version}`} hint={`trained ${data.model.trained_at.slice(0, 10)}`} tone="teal" />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="Readmission outcome distribution" subtitle="Raw outcome labels; the model target is '< 30 days' vs. everything else">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={labelRows} barCategoryGap="25%" layout="vertical" margin={{ left: 20, right: 30 }}>
              <CartesianGrid stroke={C.grid} horizontal={false} />
              <XAxis type="number" tickFormatter={(v) => `${Math.round(v * 100)}%`} />
              <YAxis type="category" dataKey="label" width={170} />
              <Tooltip formatter={(v, _n, p) => `${(Number(v) * 100).toFixed(1)}% (${(p.payload as { count: number }).count.toLocaleString()})`} />
              <Bar dataKey="share" name="Share of encounters" radius={[0, 3, 3, 0]}>
                {labelRows.map((r) => <Cell key={r.key} fill={r.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Age distribution and readmission by age group" subtitle="Bars: encounters · line: 30-day readmission rate">
          <RateChart data={data.eda.univariate.age} overall={h.readmission_rate_30d} />
        </Card>
        <Card title="Previous inpatient admissions" subtitle="Admissions in the year before the index encounter">
          <RateChart data={data.eda.univariate.number_inpatient} overall={h.readmission_rate_30d} xLabel="Prior-year inpatient admissions" />
        </Card>
        <Card title="Most important model features" subtitle={`Mean |SHAP| on ${data.explainability.sample_size.toLocaleString()} test encounters`}>
          <ImportanceChart data={top} valueKey="mean_abs_shap" label="Mean |SHAP| (log-odds)" color={C.teal} />
        </Card>
      </div>

      <Card title="Key findings (computed from the data)" className="mt-6">
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-700">
          {data.eda.findings.map((f) => <li key={f}>{f}</li>)}
        </ul>
      </Card>
      <div className="mt-6">
        <Callout tone="warn">{data.evaluation.caveat}</Callout>
      </div>
    </div>
  );
}
