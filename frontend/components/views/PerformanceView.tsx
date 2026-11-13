"use client";
import { useAnalytics } from "@/hooks/useApi";
import { Badge, Card, PageHeader, SectionLabel, StatCard } from "@/components/ui/Card";
import { Callout, ErrorState, Loading } from "@/components/ui/States";
import { CalibrationChart, PrChart, RocChart } from "@/components/charts/Curves";
import { ConfusionMatrix } from "@/components/charts/ConfusionMatrix";
import { ThresholdExplorer } from "@/components/charts/ThresholdExplorer";
import { num, pct } from "@/lib/format";

export function PerformanceView() {
  const { data, error, loading, retry } = useAnalytics();
  if (loading) return <Loading />;
  if (error || !data) return <ErrorState message={error ?? "No data"} onRetry={retry} />;
  const ev = data.evaluation;
  const t = ev.test_metrics;
  const comp = [...ev.comparison].sort((a, b) => b.cv_pr_auc_mean - a.cv_pr_auc_mean);
  return (
    <div>
      <PageHeader title="Model Performance" description={`All metrics computed on a held-out test set of ${data.model.test_set.n.toLocaleString()} encounters from ${data.model.test_set.patients.toLocaleString()} patients who never appear in training. Positive class = readmitted within 30 days (prevalence ${pct(t.prevalence)}).`} />
      <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <StatCard label="ROC-AUC" value={num(t.roc_auc, 3)} />
        <StatCard label="PR-AUC" value={num(t.pr_auc, 3)} hint={`baseline ${num(t.prevalence, 3)}`} />
        <StatCard label="Recall" value={pct(t.recall)} hint="sensitivity" tone="teal" />
        <StatCard label="Precision" value={pct(t.precision)} hint="PPV" tone="teal" />
        <StatCard label="Specificity" value={pct(t.specificity)} />
        <StatCard label="F1-score" value={num(t.f1, 3)} hint={`accuracy ${pct(t.accuracy)}`} />
      </div>

      <Card title="Model comparison" subtitle={ev.comparison.length + " models · identical patient-grouped 3-fold CV and the same test set"} className="mt-6">
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm" data-testid="comparison-table">
            <thead className="text-xs uppercase text-slate-500">
              <tr className="border-b border-slate-200">
                <th className="py-2 pr-3">Model</th><th className="px-2">CV PR-AUC</th><th className="px-2">CV ROC-AUC</th>
                <th className="px-2">Test PR-AUC</th><th className="px-2">Test ROC-AUC</th><th className="px-2">Brier</th>
                <th className="px-2">Threshold</th><th className="px-2">Recall</th><th className="px-2">Precision</th><th className="px-2">F1</th><th className="px-2">Specificity</th>
              </tr>
            </thead>
            <tbody>
              {comp.map((r) => (
                <tr key={r.model} className={`border-b border-slate-100 ${r.selected ? "bg-teal-50/60" : ""}`}>
                  <td className="py-2 pr-3 font-medium text-navy-900">{r.label} {r.selected && <Badge tone="teal">selected</Badge>}</td>
                  <td className="px-2 tabular-nums">{num(r.cv_pr_auc_mean)} <span className="text-xs text-slate-400">± {num(r.cv_pr_auc_std)}</span></td>
                  <td className="px-2 tabular-nums">{num(r.cv_roc_auc_mean)}</td>
                  <td className="px-2 tabular-nums">{num(r.test_pr_auc)}</td>
                  <td className="px-2 tabular-nums">{num(r.test_roc_auc)}</td>
                  <td className="px-2 tabular-nums">{num(r.test_brier, 4)}</td>
                  <td className="px-2 tabular-nums">{num(r.threshold, 3)}</td>
                  <td className="px-2 tabular-nums">{pct(r.test_recall)}</td>
                  <td className="px-2 tabular-nums">{pct(r.test_precision)}</td>
                  <td className="px-2 tabular-nums">{num(r.test_f1)}</td>
                  <td className="px-2 tabular-nums">{pct(r.test_specificity)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-slate-500"><b>Selection rule:</b> {data.model.selection_rule} <b>Thresholds:</b> {data.model.threshold_strategy}</p>
      </Card>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card title="ROC curve"><RocChart points={ev.curves.roc} auc={t.roc_auc} /></Card>
        <Card title="Precision–recall curve" subtitle="More informative than ROC under class imbalance"><PrChart points={ev.curves.pr} ap={t.pr_auc} prevalence={t.prevalence} /></Card>
        <Card title="Calibration" subtitle={`Brier ${num(t.brier, 4)} · ECE ${num(t.ece, 4)}`}><CalibrationChart calibrated={ev.curves.calibration} uncalibrated={ev.curves.calibration_uncalibrated} /></Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card title="Confusion matrix" subtitle={`Deployed threshold ${num(t.threshold, 3)}`}><ConfusionMatrix cm={t.confusion_matrix} /></Card>
        <Card title="Risk bands on the test set" subtitle="Observed readmission rate within each configured band" className="lg:col-span-2">
          <table className="min-w-full text-sm">
            <thead className="text-xs uppercase text-slate-500"><tr><th className="py-1 text-left">Band</th><th className="text-left">Probability</th><th className="text-right">Encounters</th><th className="text-right">Share</th><th className="text-right">Observed rate</th></tr></thead>
            <tbody>
              {ev.risk_bands.map((b) => (
                <tr key={b.band} className="border-t border-slate-100">
                  <td className="py-1.5 font-medium">{b.band}</td><td>{b.lower.toFixed(2)} – {Math.min(b.upper, 1).toFixed(2)}</td>
                  <td className="text-right tabular-nums">{b.count.toLocaleString()}</td><td className="text-right tabular-nums">{pct(b.share)}</td>
                  <td className="text-right tabular-nums font-semibold">{pct(b.observed_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>

      <SectionLabel>Threshold analysis</SectionLabel>
      <Card title="Explore the alerting trade-off" subtitle="Move the threshold to see how recall, precision, false positives and false negatives change on the test set">
        <ThresholdExplorer table={ev.threshold_table} defaultThreshold={t.threshold} />
      </Card>

      <SectionLabel>Subgroup performance (fairness check)</SectionLabel>
      <div className="grid gap-6 lg:grid-cols-3">
        {Object.entries(ev.subgroups).map(([k, rows]) => (
          <Card key={k} title={`By ${k}`} subtitle={k === "race" ? "Race is not a model input; evaluated for disparity only" : undefined}>
            <table className="min-w-full text-xs">
              <thead className="text-slate-500"><tr><th className="text-left">Group</th><th className="text-right">n</th><th className="text-right">Prev.</th><th className="text-right">ROC-AUC</th><th className="text-right">Recall</th></tr></thead>
              <tbody>{rows.map((r) => (
                <tr key={r.group} className="border-t border-slate-100"><td className="py-1">{r.group}</td><td className="text-right tabular-nums">{r.n.toLocaleString()}</td>
                  <td className="text-right tabular-nums">{pct(r.prevalence)}</td><td className="text-right tabular-nums">{num(r.roc_auc)}</td><td className="text-right tabular-nums">{pct(r.recall)}</td></tr>
              ))}</tbody>
            </table>
          </Card>
        ))}
      </div>
      <div className="mt-6"><Callout tone="warn">{ev.caveat}</Callout></div>
    </div>
  );
}
