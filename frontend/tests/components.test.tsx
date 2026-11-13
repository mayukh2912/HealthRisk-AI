import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import analytics from "./fixtures/analytics.json";
import { DashboardView } from "@/components/views/DashboardView";
import { RiskAssessmentView } from "@/components/views/RiskAssessmentView";
import { BatchView } from "@/components/views/BatchView";
import { PredictionResult } from "@/components/views/PredictionResult";
import { __clearApiCache } from "@/hooks/useApi";
import type { PredictionResponse } from "@/lib/types";

vi.mock("next/link", () => ({ default: ({ children, href }: { children: React.ReactNode; href: string }) => <a href={href}>{children}</a> }));

const prediction: PredictionResponse = {
  prediction: 1, prediction_label: "Elevated risk (flagged)", probability: 0.2178, risk_category: "High",
  decision_threshold: 0.11, model_name: "Gradient Boosting", model_version: "1.0.0", timestamp: "2026-11-12T19:37:04+00:00",
  disclaimer: "Educational research prototype … not a diagnosis.",
  explanation: {
    method: "TreeSHAP (exact, path-dependent)", units: "calibrated log-odds", base_value: -2.18, output_value: -1.28,
    top_contributors: [
      { feature: "number_inpatient", label: "Prior inpatient admissions", value: 2, contribution: 0.387, direction: "increases" },
      { feature: "age_years", label: "Age", value: 75, contribution: -0.05, direction: "decreases" },
    ],
    increasing: [{ feature: "number_inpatient", label: "Prior inpatient admissions", value: 2, contribution: 0.387, direction: "increases" }],
    decreasing: [{ feature: "age_years", label: "Age", value: 75, contribution: -0.05, direction: "decreases" }],
  },
};

const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));

beforeEach(() => {
  __clearApiCache();
  vi.restoreAllMocks();
});

describe("Executive dashboard", () => {
  it("loads analytics and separates historical stats from model metrics", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(() => json(analytics));
    render(<DashboardView />);
    expect(screen.getByRole("status")).toBeInTheDocument();
    await screen.findByText("Executive Dashboard");
    expect(screen.getByTestId("stat-Encounters analysed")).toHaveTextContent(analytics.dataset.historical.encounters.toLocaleString("en-US"));
    expect(screen.getByText(/Historical dataset/)).toBeInTheDocument();
    expect(screen.getByText(/held-out, patient-disjoint test set/)).toBeInTheDocument();
    expect(screen.getByTestId("stat-Selected model")).toHaveTextContent(analytics.model.name);
  });
  it("shows an error state with retry when the API fails", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(() => json({ detail: "Analytics artefact not found" }, 503));
    render(<DashboardView />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Analytics artefact not found");
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });
});

describe("Risk assessment form", () => {
  it("blocks submission and shows field errors when empty", async () => {
    const f = vi.spyOn(globalThis, "fetch");
    render(<RiskAssessmentView />);
    await userEvent.click(screen.getByRole("button", { name: /estimate readmission risk/i }));
    expect(screen.getByTestId("form-error-summary")).toHaveTextContent(/Please correct/);
    expect(f).not.toHaveBeenCalled();
  });
  it("submits the example and renders the prediction", async () => {
    const f = vi.spyOn(globalThis, "fetch").mockImplementation(() => json(prediction));
    render(<RiskAssessmentView />);
    await userEvent.click(screen.getByRole("button", { name: /load illustrative example/i }));
    await userEvent.click(screen.getByRole("button", { name: /estimate readmission risk/i }));
    expect(await screen.findByTestId("probability")).toHaveTextContent("21.8%");
    const body = JSON.parse((f.mock.calls[0][1] as RequestInit).body as string);
    expect(body.time_in_hospital).toBe(6);
    expect(body).not.toHaveProperty("name");
  });
  it("renders API validation errors", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(() => json({ error: "validation_error", detail: [{ field: "diag_1", message: "bad code" }] }, 422));
    render(<RiskAssessmentView />);
    await userEvent.click(screen.getByRole("button", { name: /load illustrative example/i }));
    await userEvent.click(screen.getByRole("button", { name: /estimate readmission risk/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("diag_1: bad code");
  });
});

describe("Prediction rendering", () => {
  it("shows probability, category, version, factors and disclaimer", () => {
    render(<PredictionResult result={prediction} />);
    expect(screen.getByTestId("probability")).toHaveTextContent("21.8%");
    expect(screen.getByTestId("risk-category")).toHaveTextContent("High risk");
    expect(screen.getByTestId("model-version")).toHaveTextContent("v1.0.0");
    expect(screen.getByText(/Associated with higher predicted risk/)).toBeInTheDocument();
    expect(screen.getByText(/Educational research prototype/)).toBeInTheDocument();
  });
});

describe("Batch upload and export", () => {
  const csv = "record_id,time_in_hospital,num_lab_procedures,num_procedures,num_medications,number_outpatient,number_emergency,number_inpatient,number_diagnoses,gender,age,admission_type_id,discharge_disposition_id,admission_source_id,diag_1,max_glu_serum,A1Cresult,change,diabetesMed,insulin,metformin\nr1,3,40,0,10,0,0,0,5,Male,[50-60),1,1,7,414,None,None,No,Yes,Steady,No\n";
  const batch = {
    summary: { rows_received: 1, rows_scored: 1, rows_rejected: 0, flagged: 0, risk_categories: { Low: 1, Moderate: 0, High: 0 }, mean_probability: 0.06, decision_threshold: 0.11, model_version: "1.0.0", timestamp: "2026-11-12T19:37:04+00:00" },
    predictions: [{ record_id: "r1", predicted_probability: 0.06, predicted_class: 0, risk_category: "Low", model_version: "1.0.0" }],
    errors: [], disclaimer: "not a diagnosis",
  };
  it("validates the file, scores it and exports a CSV", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(() => json(batch));
    const created: Blob[] = [];
    URL.createObjectURL = vi.fn((b: Blob) => { created.push(b); return "blob:x"; });
    URL.revokeObjectURL = vi.fn();
    render(<BatchView />);
    const file = new File([csv], "encounters.csv", { type: "text/csv" });
    await userEvent.upload(screen.getByTestId("csv-input"), file);
    await waitFor(() => expect(screen.getByTestId("csv-check")).toHaveTextContent("Columns look valid"));
    await userEvent.click(screen.getByRole("button", { name: /score encounters/i }));
    const table = await screen.findByTestId("results-table");
    expect(within(table).getByText("r1")).toBeInTheDocument();
    await userEvent.click(screen.getByTestId("download-results"));
    const text = await created[created.length - 1].text();
    expect(text.split("\n")[1]).toBe("r1,0.06,0,Low,1.0.0");
  });
  it("rejects files with identifier columns before upload", async () => {
    const f = vi.spyOn(globalThis, "fetch");
    render(<BatchView />);
    await userEvent.upload(screen.getByTestId("csv-input"), new File(["name,phone\nA,1\n"], "x.csv", { type: "text/csv" }));
    await waitFor(() => expect(screen.getByTestId("csv-check")).toHaveTextContent(/identifiers/));
    expect(screen.getByRole("button", { name: /score encounters/i })).toBeDisabled();
    expect(f).not.toHaveBeenCalled();
  });
});
