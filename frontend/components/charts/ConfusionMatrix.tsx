import type { ConfusionMatrix as CM } from "@/lib/types";

export function ConfusionMatrix({ cm }: { cm: CM }) {
  const total = cm.tn + cm.fp + cm.fn + cm.tp;
  const cell = (label: string, v: number, tone: string, desc: string) => (
    <div className={`rounded-lg p-3 ${tone}`}>
      <p className="text-[11px] font-medium uppercase tracking-wide opacity-80">{label}</p>
      <p className="text-xl font-semibold tabular-nums">{v.toLocaleString()}</p>
      <p className="text-[11px] opacity-80">{((v / total) * 100).toFixed(1)}% · {desc}</p>
    </div>
  );
  return (
    <div>
      <div className="mb-1 grid grid-cols-[80px_1fr_1fr] gap-2 text-center text-[11px] font-medium text-slate-500">
        <span />
        <span>Predicted: not flagged</span>
        <span>Predicted: flagged</span>
      </div>
      <div className="grid grid-cols-[80px_1fr_1fr] items-center gap-2">
        <span className="text-right text-[11px] font-medium text-slate-500">Actual: not readmitted</span>
        {cell("True negative", cm.tn, "bg-slate-100 text-navy-900", "correctly not flagged")}
        {cell("False positive", cm.fp, "bg-amber-50 text-amber-900", "unnecessary alert")}
        <span className="text-right text-[11px] font-medium text-slate-500">Actual: readmitted</span>
        {cell("False negative", cm.fn, "bg-red-50 text-red-900", "missed readmission")}
        {cell("True positive", cm.tp, "bg-teal-50 text-teal-700", "correctly flagged")}
      </div>
    </div>
  );
}
