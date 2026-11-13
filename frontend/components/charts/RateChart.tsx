"use client";
import { Bar, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { RateRow } from "@/lib/types";
import { C } from "./colors";

/** Encounter counts (bars, left axis) with the 30-day readmission rate (line, right axis). */
export function RateChart({ data, height = 260, xLabel, overall }: { data: RateRow[]; height?: number; xLabel?: string; overall?: number }) {
  const rows = data.map((d) => ({ ...d, ratePct: +(d.rate * 100).toFixed(2), overallPct: overall !== undefined ? +(overall * 100).toFixed(2) : undefined }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: xLabel ? 18 : 4 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="group" interval={0} angle={rows.length > 8 ? -30 : 0} textAnchor={rows.length > 8 ? "end" : "middle"} height={rows.length > 8 ? 50 : 30}
          label={xLabel ? { value: xLabel, position: "insideBottom", offset: -12, fontSize: 11, fill: "#4b5b70" } : undefined} />
        <YAxis yAxisId="n" tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)} label={{ value: "Encounters", angle: -90, position: "insideLeft", fontSize: 11, fill: "#4b5b70" }} />
        <YAxis yAxisId="r" orientation="right" unit="%" domain={[0, "auto"]} />
        <Tooltip formatter={(v, name) => (name === "Encounters" ? Number(v).toLocaleString() : `${v}%`)} />
        <Legend verticalAlign="top" height={28} />
        <Bar yAxisId="n" dataKey="count" name="Encounters" fill={C.navy} radius={[3, 3, 0, 0]} />
        <Line yAxisId="r" dataKey="ratePct" name="30-day readmission rate" stroke={C.teal} strokeWidth={2.5} dot={{ r: 3 }} />
        {overall !== undefined && <Line yAxisId="r" dataKey="overallPct" name="Overall rate" stroke={C.grey} strokeDasharray="4 4" dot={false} />}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function RateBars({ data, height = 260, overall, horizontal = false }: { data: RateRow[]; height?: number; overall?: number; horizontal?: boolean }) {
  const rows = data.map((d) => ({ ...d, ratePct: +(d.rate * 100).toFixed(2) }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} layout={horizontal ? "vertical" : "horizontal"} margin={{ top: 8, right: 16, left: horizontal ? 40 : 0, bottom: 4 }}>
        <CartesianGrid stroke={C.grid} vertical={horizontal} horizontal={!horizontal} />
        {horizontal ? (
          <>
            <XAxis type="number" unit="%" />
            <YAxis type="category" dataKey="group" width={150} interval={0} />
          </>
        ) : (
          <>
            <XAxis dataKey="group" interval={0} />
            <YAxis unit="%" />
          </>
        )}
        <Tooltip formatter={(v) => `${v}%`} labelFormatter={(l) => {
          const r = rows.find((x) => x.group === l);
          return r ? `${l} (n = ${r.count.toLocaleString()})` : String(l);
        }} />
        <Bar dataKey="ratePct" name="30-day readmission rate" fill={C.teal} radius={horizontal ? [0, 3, 3, 0] : [3, 3, 0, 0]} />
        {overall !== undefined && !horizontal && (
          <Line dataKey={() => +(overall * 100).toFixed(2)} name="Overall" stroke={C.navy} strokeDasharray="4 4" dot={false} />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
