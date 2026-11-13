"use client";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, ComposedChart, Tooltip, XAxis, YAxis } from "recharts";
import { C } from "./colors";

const axisLabel = (value: string, pos: "insideBottom" | "insideLeft") =>
  ({ value, position: pos, offset: pos === "insideBottom" ? -8 : 10, angle: pos === "insideLeft" ? -90 : 0, fontSize: 11, fill: "#4b5b70" });

export function RocChart({ points, auc }: { points: { fpr: number; tpr: number }[]; auc: number }) {
  const data = points.map((p) => ({ ...p, chance: p.fpr }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: 4, bottom: 18 }}>
        <CartesianGrid stroke={C.grid} />
        <XAxis dataKey="fpr" type="number" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} label={axisLabel("False positive rate (1 − specificity)", "insideBottom")} />
        <YAxis type="number" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} label={axisLabel("True positive rate (recall)", "insideLeft")} />
        <Tooltip formatter={(v) => Number(v).toFixed(3)} labelFormatter={(l) => `FPR ${Number(l).toFixed(3)}`} />
        <Legend verticalAlign="top" height={24} />
        <Line dataKey="tpr" name={`Model (ROC-AUC ${auc.toFixed(3)})`} stroke={C.navy} strokeWidth={2.5} dot={false} />
        <Line dataKey="chance" name="Chance" stroke={C.grey} strokeDasharray="4 4" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PrChart({ points, ap, prevalence }: { points: { recall: number; precision: number }[]; ap: number; prevalence: number }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={points} margin={{ top: 8, right: 12, left: 4, bottom: 18 }}>
        <CartesianGrid stroke={C.grid} />
        <XAxis dataKey="recall" type="number" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} label={axisLabel("Recall (sensitivity)", "insideBottom")} />
        <YAxis type="number" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} label={axisLabel("Precision (PPV)", "insideLeft")} />
        <Tooltip formatter={(v) => Number(v).toFixed(3)} labelFormatter={(l) => `Recall ${Number(l).toFixed(3)}`} />
        <Legend verticalAlign="top" height={24} />
        <ReferenceLine y={prevalence} stroke={C.grey} strokeDasharray="4 4" label={{ value: `No-skill = prevalence ${prevalence.toFixed(3)}`, position: "insideTopRight", fontSize: 10, fill: "#64748b" }} />
        <Line dataKey="precision" name={`Model (PR-AUC ${ap.toFixed(3)})`} stroke={C.teal} strokeWidth={2.5} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function CalibrationChart({ calibrated, uncalibrated }: {
  calibrated: { mean_predicted: number; observed_rate: number }[]; uncalibrated?: { mean_predicted: number; observed_rate: number }[];
}) {
  const max = Math.max(...calibrated.map((p) => Math.max(p.mean_predicted, p.observed_rate)), ...(uncalibrated ?? []).map((p) => Math.max(p.mean_predicted, p.observed_rate)));
  const lim = Math.min(1, Math.ceil(max * 10) / 10 + 0.05);
  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart margin={{ top: 8, right: 12, left: 4, bottom: 18 }}>
        <CartesianGrid stroke={C.grid} />
        <XAxis dataKey="mean_predicted" type="number" domain={[0, lim]} tickFormatter={(v) => v.toFixed(2)} label={axisLabel("Mean predicted probability (decile)", "insideBottom")} />
        <YAxis dataKey="observed_rate" type="number" domain={[0, lim]} tickFormatter={(v) => v.toFixed(2)} label={axisLabel("Observed readmission rate", "insideLeft")} />
        <Tooltip formatter={(v) => Number(v).toFixed(3)} />
        <Legend verticalAlign="top" height={24} />
        <Line data={[{ mean_predicted: 0, observed_rate: 0 }, { mean_predicted: lim, observed_rate: lim }]} dataKey="observed_rate" name="Perfect calibration" stroke={C.grey} strokeDasharray="4 4" dot={false} legendType="plainline" />
        {uncalibrated && <Line data={uncalibrated} dataKey="observed_rate" name="Before calibration" stroke={C.amber} strokeWidth={1.5} dot={{ r: 3 }} />}
        <Line data={calibrated} dataKey="observed_rate" name="Platt-calibrated (deployed)" stroke={C.navy} strokeWidth={2.5} dot={{ r: 4 }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
