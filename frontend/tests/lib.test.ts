import { describe, expect, it } from "vitest";
import { checkBatchCsv, parseCsv, predictionsToCsv, TEMPLATE_CSV } from "@/lib/csv";
import { EXAMPLE_INPUT, EMPTY_INPUT } from "@/lib/schema";
import { toPayload, validateForm } from "@/lib/validation";
import { extractErrors } from "@/lib/api";

describe("form validation", () => {
  it("flags every required field on an empty form", () => {
    const e = validateForm(EMPTY_INPUT);
    expect(e.time_in_hospital).toBe("Required");
    expect(e.gender).toBe("Select a value");
    expect(e.diag_1).toBe("Required");
    expect(e.medical_specialty).toBeUndefined();
    expect(e.diag_2).toBeUndefined();
  });
  it("accepts the illustrative example", () => {
    expect(validateForm(EXAMPLE_INPUT)).toEqual({});
  });
  it("rejects out-of-range, non-integer and malformed values", () => {
    const e = validateForm({ ...EXAMPLE_INPUT, time_in_hospital: "20", num_medications: "2.5", diag_1: "abc" });
    expect(e.time_in_hospital).toMatch(/between 1 and 14/);
    expect(e.num_medications).toMatch(/whole number/);
    expect(e.diag_1).toMatch(/ICD-9/);
  });
  it("builds a typed payload", () => {
    const p = toPayload({ ...EXAMPLE_INPUT, diag_3: "" , medical_specialty: "" });
    expect(p.time_in_hospital).toBe(6);
    expect(p.admission_type_id).toBe(1);
    expect(p.diag_3).toBeNull();
    expect(p.medical_specialty).toBeNull();
    expect(p.gender).toBe("Female");
  });
});

describe("CSV handling", () => {
  it("parses quoted fields and CRLF", () => {
    const { header, rows } = parseCsv('a,b\r\n"x, y","he said ""hi"""\r\n');
    expect(header).toEqual(["a", "b"]);
    expect(rows[0]).toEqual(["x, y", 'he said "hi"']);
  });
  it("accepts the template", () => {
    const c = checkBatchCsv(TEMPLATE_CSV);
    expect(c.ok).toBe(true);
    expect(c.rowCount).toBe(2);
  });
  it("rejects missing columns and direct identifiers", () => {
    const c = checkBatchCsv("name,gender\nJane,Female\n");
    expect(c.ok).toBe(false);
    expect(c.errors.join(" ")).toMatch(/identifiers/);
    expect(c.errors.join(" ")).toMatch(/Missing required columns/);
  });
  it("exports predictions with escaping and formula-injection guard", () => {
    const out = predictionsToCsv([
      { record_id: "=cmd()", predicted_probability: 0.12, predicted_class: 1, risk_category: "Moderate", model_version: "1.0.0" },
      { record_id: "a,b", predicted_probability: 0.05, predicted_class: 0, risk_category: "Low", model_version: "1.0.0" },
    ]);
    const lines = out.trim().split("\n");
    expect(lines[0]).toBe("record_id,predicted_probability,predicted_class,risk_category,model_version");
    expect(lines[1].startsWith("'=cmd()")).toBe(true);
    expect(lines[2].startsWith('"a,b"')).toBe(true);
  });
});

describe("API error normalisation", () => {
  it("handles FastAPI validation and plain errors", () => {
    expect(extractErrors({ error: "validation_error", detail: [{ field: "gender", message: "bad" }] })).toEqual(["gender: bad"]);
    expect(extractErrors({ detail: "nope" })).toEqual(["nope"]);
    expect(extractErrors({ detail: { error: "invalid_columns", detail: ["Missing x"] } })).toEqual(["Missing x"]);
  });
});
