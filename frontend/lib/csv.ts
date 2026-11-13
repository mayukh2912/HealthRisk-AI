import { FORBIDDEN_COLUMNS, MAX_BATCH_BYTES, MAX_BATCH_ROWS, OPTIONAL_CSV_COLUMNS, REQUIRED_CSV_COLUMNS } from "./schema";
import type { BatchPrediction } from "./types";

/** Minimal RFC-4180 CSV parser (quoted fields, escaped quotes, CRLF). */
export function parseCsv(text: string): { header: string[]; rows: string[][] } {
  const rows: string[][] = [];
  let field = "";
  let row: string[] = [];
  let inQuotes = false;
  const src = text.replace(/^﻿/, "");
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if (inQuotes) {
      if (c === '"') {
        if (src[i + 1] === '"') {
          field += '"';
          i++;
        } else inQuotes = false;
      } else field += c;
    } else if (c === '"') inQuotes = true;
    else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && src[i + 1] === "\n") i++;
      row.push(field);
      field = "";
      if (row.some((v) => v !== "")) rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== "" || row.length) {
    row.push(field);
    if (row.some((v) => v !== "")) rows.push(row);
  }
  const header = (rows.shift() ?? []).map((h) => h.trim());
  return { header, rows };
}

export type CsvCheck = { ok: boolean; errors: string[]; warnings: string[]; rowCount: number; header: string[] };

export function checkBatchCsv(text: string): CsvCheck {
  const errors: string[] = [];
  const warnings: string[] = [];
  const bytes = new TextEncoder().encode(text).length;
  if (bytes > MAX_BATCH_BYTES) errors.push(`File is ${(bytes / 1e6).toFixed(1)} MB; the limit is ${MAX_BATCH_BYTES / 1e6} MB.`);
  const { header, rows } = parseCsv(text);
  if (header.length === 0) {
    return { ok: false, errors: ["The file is empty."], warnings, rowCount: 0, header };
  }
  const lower = header.map((h) => h.toLowerCase());
  const forbidden = FORBIDDEN_COLUMNS.filter((f) => lower.includes(f));
  if (forbidden.length) errors.push(`Remove direct patient identifiers before upload: ${forbidden.join(", ")}`);
  const missing = REQUIRED_CSV_COLUMNS.filter((c) => !header.includes(c));
  if (missing.length) errors.push(`Missing required columns: ${missing.join(", ")}`);
  if (rows.length === 0) errors.push("The CSV has a header but no data rows.");
  if (rows.length > MAX_BATCH_ROWS) errors.push(`At most ${MAX_BATCH_ROWS.toLocaleString()} rows per upload (found ${rows.length.toLocaleString()}).`);
  if (!header.includes("record_id")) warnings.push("No 'record_id' column - rows will be labelled row_1, row_2, …");
  const known = new Set([...REQUIRED_CSV_COLUMNS, ...OPTIONAL_CSV_COLUMNS]);
  const extra = header.filter((h) => !known.has(h));
  if (extra.length) warnings.push(`${extra.length} extra column(s) will be ignored unless they are diabetes medication columns.`);
  return { ok: errors.length === 0, errors, warnings, rowCount: rows.length, header };
}

function esc(v: unknown): string {
  let s = v === null || v === undefined ? "" : String(v);
  if (/^[=+\-@]/.test(s) && !/^-?\d+(\.\d+)?$/.test(s)) s = `'${s}`; // spreadsheet formula injection guard
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function predictionsToCsv(rows: BatchPrediction[]): string {
  const cols: (keyof BatchPrediction)[] = ["record_id", "predicted_probability", "predicted_class", "risk_category", "model_version"];
  return [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n") + "\n";
}

export const TEMPLATE_CSV =
  ["record_id", ...REQUIRED_CSV_COLUMNS, "medical_specialty", "diag_2", "diag_3"].join(",") +
  "\n" +
  ["demo-001", 6, 52, 1, 18, 0, 1, 2, 9, "Female", "[70-80)", 1, 6, 7, 428, "None", "None", "Ch", "Yes", "Up", "No", "InternalMedicine", 250.02, 403].join(",") +
  "\n" +
  ["demo-002", 2, 30, 0, 9, 0, 0, 0, 5, "Male", "[50-60)", 3, 1, 1, 715, "None", "Norm", "No", "No", "No", "No", "Orthopedics", 401, 250].join(",") +
  "\n";

export function downloadText(filename: string, text: string, type = "text/csv") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
