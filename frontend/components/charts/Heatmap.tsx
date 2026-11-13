/** Accessible table-based heatmap (values shown as text, colour is supplementary). */
export function Heatmap({ rows, columns, matrix, format = (v: number) => v.toFixed(2), diverging = false, caption }: {
  rows: string[]; columns: string[]; matrix: (number | null)[][]; format?: (v: number) => string; diverging?: boolean; caption?: string;
}) {
  const vals = matrix.flat().filter((v): v is number => v !== null);
  const max = Math.max(...vals.map(Math.abs));
  const min = Math.min(...vals);
  const color = (v: number | null) => {
    if (v === null) return "#f1f5f9";
    if (diverging) {
      const t = Math.min(Math.abs(v) / (max || 1), 1);
      return v >= 0 ? `rgba(180,35,24,${0.08 + 0.75 * t})` : `rgba(43,90,143,${0.08 + 0.75 * t})`;
    }
    const t = (v - min) / ((Math.max(...vals) - min) || 1);
    return `rgba(19,168,158,${0.08 + 0.85 * t})`;
  };
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full border-separate border-spacing-0.5 text-[11px]">
        {caption && <caption className="mb-2 text-left text-xs text-slate-500">{caption}</caption>}
        <thead>
          <tr>
            <th />
            {columns.map((c) => <th key={c} scope="col" className="px-1 py-1 text-center font-medium text-slate-600">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r}>
              <th scope="row" className="whitespace-nowrap pr-2 text-right font-medium text-slate-600">{r}</th>
              {matrix[i].map((v, j) => (
                <td key={j} className="min-w-12 rounded px-1 py-1.5 text-center tabular-nums text-navy-900" style={{ background: color(v) }}>
                  {v === null ? "–" : format(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
