"use client";
import { Bar, BarChart, CartesianGrid, Cell, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DriftFeature } from "@/lib/types";
import { FEATURE_NAMES } from "@/lib/format";
import { C } from "./colors";

export function PsiChart({ features }: { features: DriftFeature[] }) {
  const rows = features.map((f) => ({ ...f, label: FEATURE_NAMES[f.feature] ?? f.feature }));
  const color = (p: number) => (p >= 0.25 ? C.red : p >= 0.1 ? C.amber : C.teal);
  return (
    <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 24 + 40)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 16 }}>
        <CartesianGrid stroke={C.grid} horizontal={false} />
        <XAxis type="number" domain={[0, (max: number) => Math.max(0.3, max)]} tickFormatter={(v) => Number(v).toFixed(2)}
          label={{ value: "Population Stability Index", position: "insideBottom", offset: -8, fontSize: 11, fill: "#4b5b70" }} />
        <YAxis type="category" dataKey="label" width={160} interval={0} />
        <Tooltip formatter={(v) => Number(v).toFixed(4)} />
        <ReferenceLine x={0.1} stroke={C.amber} strokeDasharray="4 4" />
        <ReferenceLine x={0.25} stroke={C.red} strokeDasharray="4 4" />
        <Bar dataKey="psi" name="PSI" radius={[0, 3, 3, 0]}>
          {rows.map((r) => <Cell key={r.feature} fill={color(r.psi)} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DistributionCompare({ feature }: { feature: DriftFeature }) {
  const rows = feature.bins.map((b) => ({ bin: b.bin, Reference: +(b.reference * 100).toFixed(2), Current: +(b.current * 100).toFixed(2) }));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 30 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="bin" interval={0} angle={-30} textAnchor="end" height={60} />
        <YAxis unit="%" />
        <Tooltip formatter={(v) => `${v}%`} />
        <Legend verticalAlign="top" height={24} />
        <Bar dataKey="Reference" fill={C.navy} />
        <Bar dataKey="Current" fill={C.teal} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function PredictionHistogram({ current, reference, currentLabel = "Current" }: {
  current: { edges: number[]; counts: number[] }; reference?: { edges: number[]; counts: number[] }; currentLabel?: string;
}) {
  const share = (c: number[]) => { const s = c.reduce((a, b) => a + b, 0) || 1; return c.map((v) => +((v / s) * 100).toFixed(2)); };
  const cur = share(current.counts);
  const ref = reference ? share(reference.counts) : null;
  const rows = current.counts.map((_, i) => ({
    bin: `${current.edges[i].toFixed(2)}–${current.edges[i + 1].toFixed(2)}`,
    [currentLabel]: cur[i], ...(ref ? { "Training reference": ref[i] } : {}),
  })).filter((r, i) => cur[i] > 0 || (ref && ref[i] > 0));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 30 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="bin" interval={0} angle={-30} textAnchor="end" height={60} />
        <YAxis unit="%" />
        <Tooltip formatter={(v) => `${v}%`} />
        <Legend verticalAlign="top" height={24} />
        {ref && <Bar dataKey="Training reference" fill={C.navy} />}
        <Bar dataKey={currentLabel} fill={C.teal} />
      </BarChart>
    </ResponsiveContainer>
  );
}
