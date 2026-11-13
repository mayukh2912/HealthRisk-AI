"use client";
import { Bar, BarChart, CartesianGrid, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { C } from "./colors";

export function ImportanceChart({ data, valueKey, label, height, color = C.navy, errorKey }: {
  data: Record<string, unknown>[]; valueKey: string; label: string; height?: number; color?: string; errorKey?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height ?? Math.max(220, data.length * 26 + 40)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 16 }}>
        <CartesianGrid stroke={C.grid} horizontal={false} />
        <XAxis type="number" tickFormatter={(v) => Number(v).toFixed(3)} label={{ value: label, position: "insideBottom", offset: -8, fontSize: 11, fill: "#4b5b70" }} />
        <YAxis type="category" dataKey="label" width={170} interval={0} />
        <Tooltip formatter={(v) => Number(v).toFixed(4)} />
        <Bar dataKey={valueKey} name={label} fill={color} radius={[0, 3, 3, 0]}>
          {errorKey && <ErrorBar dataKey={errorKey} width={4} stroke={C.grey} />}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
