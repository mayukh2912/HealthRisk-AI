import { ICD9_RE, NUMERIC_FIELDS, SELECT_FIELDS, TEXT_FIELDS } from "./schema";

export type FormErrors = Record<string, string>;
const OPTIONAL_SELECTS = new Set(["medical_specialty"]);

/** Validate the risk-assessment form. Returns field -> message for every problem. */
export function validateForm(values: Record<string, string>): FormErrors {
  const errors: FormErrors = {};
  for (const f of NUMERIC_FIELDS) {
    const raw = (values[f.name] ?? "").trim();
    if (raw === "") {
      errors[f.name] = "Required";
      continue;
    }
    const n = Number(raw);
    if (!Number.isFinite(n) || !Number.isInteger(n)) errors[f.name] = "Enter a whole number";
    else if (n < f.min || n > f.max) errors[f.name] = `Must be between ${f.min} and ${f.max}`;
  }
  for (const f of SELECT_FIELDS) {
    if (OPTIONAL_SELECTS.has(f.name)) continue;
    if (!values[f.name]) errors[f.name] = "Select a value";
    else if (!f.options.some((o) => o.value === values[f.name])) errors[f.name] = "Invalid option";
  }
  for (const f of TEXT_FIELDS) {
    const v = (values[f.name] ?? "").trim();
    if (!v) {
      if (f.required) errors[f.name] = "Required";
      continue;
    }
    if (!ICD9_RE.test(v)) errors[f.name] = "Use an ICD-9 code, e.g. 428, 250.01, V45";
  }
  return errors;
}

/** Convert validated string form values into the API payload. */
export function toPayload(values: Record<string, string>): Record<string, unknown> {
  const p: Record<string, unknown> = {};
  for (const f of NUMERIC_FIELDS) p[f.name] = Number(values[f.name]);
  for (const f of SELECT_FIELDS) {
    const v = values[f.name];
    if (f.name.endsWith("_id")) p[f.name] = Number(v);
    else if (f.name === "medical_specialty") p[f.name] = v ? v : null;
    else p[f.name] = v;
  }
  for (const f of TEXT_FIELDS) {
    const v = (values[f.name] ?? "").trim();
    p[f.name] = v ? v.toUpperCase() : null;
  }
  return p;
}
