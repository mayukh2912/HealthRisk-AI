"use client";
import { useAnalytics } from "@/hooks/useApi";
import { Card, PageHeader, SectionLabel } from "@/components/ui/Card";
import { Callout, ErrorState, Loading } from "@/components/ui/States";
import { RateBars, RateChart } from "@/components/charts/RateChart";
import { Heatmap } from "@/components/charts/Heatmap";
import { FEATURE_NAMES, int, num, pct } from "@/lib/format";

export function InsightsView() {
  const { data, error, loading, retry } = useAnalytics();
  if (loading) return <Loading />;
  if (error || !data) return <ErrorState message={error ?? "No data"} onRetry={retry} />;
  const e = data.eda;
  const u = e.univariate, b = e.bivariate, m = e.multivariate;
  const overall = e.overview.readmission_rate_30d;
  const ageAdmTypes = Array.from(new Set(m.age_by_admission_type.map((r) => r.admission_type)));
  const ageRows = Array.from(new Set(m.age_by_admission_type.map((r) => r.age)));
  const ageAdm = ageRows.map((a) => ageAdmTypes.map((t) => {
    const r = m.age_by_admission_type.find((x) => x.age === a && x.admission_type === t);
    return r && r.count >= 30 ? r.rate : null;
  }));
  const corrCols = m.spearman_correlation.columns.map((c) => FEATURE_NAMES[c] ?? c);
  return (
    <div>
      <PageHeader title="Data Insights" description={`Exploratory analysis of ${int(e.overview.encounters)} encounters (${int(e.overview.patients)} patients). All figures are aggregates computed from the dataset by the EDA pipeline; dashed lines mark the overall ${pct(overall)} readmission rate.`} />
      <SectionLabel>Age</SectionLabel>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Encounters and readmission by age bracket"><RateChart data={u.age} overall={overall} /></Card>
        <Card title="Readmission rate by age and admission type" subtitle="Cells with fewer than 30 encounters hidden">
          <Heatmap rows={ageRows} columns={ageAdmTypes} matrix={ageAdm} format={(v) => `${(v * 100).toFixed(1)}%`} />
        </Card>
      </div>
      <SectionLabel>Hospital stay</SectionLabel>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Length of stay"><RateChart data={u.time_in_hospital} overall={overall} xLabel="Days in hospital" /></Card>
        <Card title="Lab procedures during the stay"><RateChart data={u.num_lab_procedures} overall={overall} /></Card>
      </div>
      <SectionLabel>Previous encounters (prior 12 months)</SectionLabel>
      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="Inpatient admissions"><RateChart data={u.number_inpatient} overall={overall} height={230} /></Card>
        <Card title="Emergency visits"><RateChart data={u.number_emergency} overall={overall} height={230} /></Card>
        <Card title="Outpatient visits"><RateChart data={u.number_outpatient} overall={overall} height={230} /></Card>
      </div>
      <Card title="Readmission rate by age and prior inpatient admissions" className="mt-6">
        <Heatmap rows={m.age_by_prior_inpatient.rows} columns={m.age_by_prior_inpatient.columns} matrix={m.age_by_prior_inpatient.matrix} format={(v) => `${(v * 100).toFixed(0)}%`} />
      </Card>
      <SectionLabel>Medication and glycaemic control</SectionLabel>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Number of medications"><RateChart data={u.num_medications} overall={overall} /></Card>
        <Card title="Insulin status"><RateBars data={b.insulin} overall={overall} /></Card>
        <Card title="HbA1c result"><RateBars data={b.A1Cresult} overall={overall} /></Card>
        <Card title="Max glucose serum"><RateBars data={b.max_glu_serum} overall={overall} /></Card>
      </div>
      <SectionLabel>Readmission patterns</SectionLabel>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Discharge disposition" subtitle="Groups with ≥ 300 encounters"><RateBars data={b.discharge_disposition} horizontal height={360} /></Card>
        <Card title="Admitting physician specialty" subtitle="12 most frequent (≥ 500 encounters)"><RateBars data={b.medical_specialty} horizontal height={360} /></Card>
        <Card title="Primary diagnosis group (ICD-9 chapter)"><RateBars data={b.diag_1_group} horizontal height={300} /></Card>
        <Card title="Admission type"><RateBars data={b.admission_type} horizontal height={300} /></Card>
      </div>
      <SectionLabel>Statistics</SectionLabel>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Spearman correlation" subtitle="Numeric features and the 30-day target">
          <Heatmap rows={corrCols} columns={corrCols.map((c) => c.split(" ")[0])} matrix={m.spearman_correlation.matrix} diverging format={(v) => v.toFixed(2)} />
        </Card>
        <Card title="Association with readmission (effect sizes)">
          <table className="min-w-full text-xs">
            <thead className="text-slate-500"><tr><th className="text-left">Categorical feature</th><th className="text-right">Cramér&apos;s V</th><th className="text-right">χ² p-value</th></tr></thead>
            <tbody>{e.statistical_tests.categorical_chi_square.map((r) => (
              <tr key={r.feature} className="border-t border-slate-100"><td className="py-1">{r.feature}</td><td className="text-right tabular-nums">{num(r.cramers_v)}</td><td className="text-right tabular-nums">{r.p_value < 1e-4 ? "< 0.0001" : num(r.p_value, 4)}</td></tr>
            ))}</tbody>
          </table>
          <table className="mt-4 min-w-full text-xs">
            <thead className="text-slate-500"><tr><th className="text-left">Numeric feature</th><th className="text-right">Mean (readmitted)</th><th className="text-right">Mean (not)</th><th className="text-right">Rank-biserial</th></tr></thead>
            <tbody>{e.statistical_tests.numeric_mann_whitney.map((r) => (
              <tr key={r.feature} className="border-t border-slate-100"><td className="py-1">{FEATURE_NAMES[r.feature] ?? r.feature}</td><td className="text-right tabular-nums">{num(r.mean_readmitted, 2)}</td><td className="text-right tabular-nums">{num(r.mean_not_readmitted, 2)}</td><td className="text-right tabular-nums">{num(r.rank_biserial)}</td></tr>
            ))}</tbody>
          </table>
          <p className="mt-3 text-[11px] text-slate-500">{e.statistical_tests.caveat}</p>
        </Card>
      </div>
      <Card title="Descriptive statistics (numeric inputs)" className="mt-6">
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead className="text-slate-500"><tr>{["Feature", "Mean", "Median", "Std", "Min", "Q1", "Q3", "Max", "Missing"].map((h) => <th key={h} className={h === "Feature" ? "text-left" : "text-right"}>{h}</th>)}</tr></thead>
            <tbody>{Object.entries(e.descriptive_stats).map(([k, s]) => (
              <tr key={k} className="border-t border-slate-100"><td className="py-1">{FEATURE_NAMES[k] ?? k}</td>
                {[s.mean, s.median, s.std, s.min, s.q1, s.q3, s.max].map((v, i) => <td key={i} className="text-right tabular-nums">{num(v, 2)}</td>)}
                <td className="text-right tabular-nums">{s.missing_pct}%</td></tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
      <div className="mt-6"><Callout>Missing values: {Object.entries(data.dataset.quality.missing_pct).map(([k, v]) => `${k} ${v}%`).join(" · ")}. See the data-quality report for duplicates, invalid values and cohort exclusions.</Callout></div>
    </div>
  );
}
