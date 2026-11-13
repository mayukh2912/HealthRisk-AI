"use client";
import { Bar, BarChart, Cell, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Contribution } from "@/lib/types";
import { C } from "./colors";

const fmtVal = (v: unknown) => (v === null || v === undefined ? "missing" : typeof v === "number" ? +v.toFixed(2) : String(v));

/** Diverging bar chart of per-prediction SHAP contributions. */
export function ContributionChart({ items, units }: { items: Contribution[]; units: string }) {
  const rows = items.map((c) => ({ ...c, name: `${c.label} = ${fmtVal(c.value)}` }));
  return (
    <ResponsiveContainer width="100%" height={Math.max(200, rows.length * 32 + 40)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 18 }}>
        <CartesianGrid stroke={C.grid} horizontal={false} />
        <XAxis type="number" tickFormatter={(v) => Number(v).toFixed(2)} label={{ value: `Contribution (${units})`, position: "insideBottom", offset: -10, fontSize: 11, fill: "#4b5b70" }} />
        <YAxis type="category" dataKey="name" width={230} interval={0} />
        <ReferenceLine x={0} stroke="#475569" />
        <Tooltip formatter={(v) => Number(v).toFixed(4)} />
        <Bar dataKey="contribution" name="Contribution" radius={3}>
          {rows.map((r) => <Cell key={r.feature} fill={r.contribution > 0 ? C.red : C.teal} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
