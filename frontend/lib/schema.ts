/** Client-side mirror of the API input contract (backend/healthrisk_runtime/features.py). */

export const AGE_BRACKETS = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"];
export const MED_STATUS = ["No", "Steady", "Up", "Down"];

export type NumericField = { name: string; label: string; min: number; max: number; help?: string };
export type SelectField = { name: string; label: string; options: { value: string; label: string }[]; help?: string };

export const NUMERIC_FIELDS: NumericField[] = [
  { name: "time_in_hospital", label: "Length of stay (days)", min: 1, max: 14 },
  { name: "num_lab_procedures", label: "Lab procedures", min: 0, max: 200 },
  { name: "num_procedures", label: "Non-lab procedures", min: 0, max: 10 },
  { name: "num_medications", label: "Distinct medications", min: 0, max: 100 },
  { name: "number_outpatient", label: "Outpatient visits (prior year)", min: 0, max: 60 },
  { name: "number_emergency", label: "Emergency visits (prior year)", min: 0, max: 80 },
  { name: "number_inpatient", label: "Inpatient admissions (prior year)", min: 0, max: 30 },
  { name: "number_diagnoses", label: "Diagnoses recorded", min: 1, max: 16 },
];

const opt = (pairs: [string | number, string][]) => pairs.map(([v, l]) => ({ value: String(v), label: l }));

export const SELECT_FIELDS: SelectField[] = [
  { name: "gender", label: "Gender", options: opt([["Female", "Female"], ["Male", "Male"]]) },
  { name: "age", label: "Age bracket", options: AGE_BRACKETS.map((a) => ({ value: a, label: a.replace("[", "").replace(")", "") + " years" })) },
  {
    name: "admission_type_id", label: "Admission type",
    options: opt([[1, "Emergency"], [2, "Urgent"], [3, "Elective"], [4, "Newborn"], [7, "Trauma center"], [5, "Not available"]]),
  },
  {
    name: "admission_source_id", label: "Admission source",
    options: opt([[7, "Emergency room"], [1, "Physician referral"], [2, "Clinic referral"], [3, "HMO referral"],
      [4, "Transfer from hospital"], [5, "Transfer from SNF"], [6, "Transfer from other facility"], [9, "Not available"]]),
  },
  {
    name: "discharge_disposition_id", label: "Discharge disposition",
    options: opt([[1, "Home"], [6, "Home with home health service"], [3, "Skilled nursing facility (SNF)"],
      [22, "Rehab facility"], [2, "Short-term hospital"], [5, "Other inpatient care"], [4, "Intermediate care facility"],
      [23, "Long-term care hospital"], [7, "Left against medical advice"], [28, "Psychiatric hospital"], [18, "Not available"]]),
    help: "Expired / hospice dispositions are out of scope (not at risk of readmission).",
  },
  {
    name: "medical_specialty", label: "Admitting physician specialty",
    options: opt([["", "Not recorded"], ["InternalMedicine", "Internal medicine"], ["Family/GeneralPractice", "Family / general practice"],
      ["Emergency/Trauma", "Emergency / trauma"], ["Cardiology", "Cardiology"], ["Surgery-General", "General surgery"],
      ["Nephrology", "Nephrology"], ["Orthopedics", "Orthopedics"], ["Pulmonology", "Pulmonology"], ["Psychiatry", "Psychiatry"],
      ["Gastroenterology", "Gastroenterology"], ["Oncology", "Oncology"], ["Other", "Other"]]),
  },
  { name: "max_glu_serum", label: "Max glucose serum", options: opt([["None", "Not measured"], ["Norm", "Normal"], [">200", "> 200 mg/dL"], [">300", "> 300 mg/dL"]]) },
  { name: "A1Cresult", label: "HbA1c result", options: opt([["None", "Not measured"], ["Norm", "Normal"], [">7", "> 7%"], [">8", "> 8%"]]) },
  { name: "insulin", label: "Insulin", options: MED_STATUS.map((s) => ({ value: s, label: s === "No" ? "Not prescribed" : s })) },
  { name: "metformin", label: "Metformin", options: MED_STATUS.map((s) => ({ value: s, label: s === "No" ? "Not prescribed" : s })) },
  { name: "change", label: "Diabetes medication changed", options: opt([["No", "No change"], ["Ch", "Changed"]]) },
  { name: "diabetesMed", label: "Diabetes medication prescribed", options: opt([["Yes", "Yes"], ["No", "No"]]) },
];

export const TEXT_FIELDS = [
  { name: "diag_1", label: "Primary diagnosis (ICD-9)", required: true, placeholder: "e.g. 428" },
  { name: "diag_2", label: "Secondary diagnosis (ICD-9)", required: false, placeholder: "optional, e.g. 250.02" },
  { name: "diag_3", label: "Additional diagnosis (ICD-9)", required: false, placeholder: "optional, e.g. 401" },
];

export const ICD9_RE = /^(V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?|\d{1,3}(\.\d{1,2})?)$/i;

/** Illustrative values for demonstrating the form - NOT a real patient record. */
export const EXAMPLE_INPUT: Record<string, string> = {
  gender: "Female", age: "[70-80)", admission_type_id: "1", admission_source_id: "7", discharge_disposition_id: "6",
  medical_specialty: "InternalMedicine", time_in_hospital: "6", num_lab_procedures: "52", num_procedures: "1",
  num_medications: "18", number_outpatient: "0", number_emergency: "1", number_inpatient: "2", number_diagnoses: "9",
  diag_1: "428", diag_2: "250.02", diag_3: "403", max_glu_serum: "None", A1Cresult: "None", insulin: "Up",
  metformin: "No", change: "Ch", diabetesMed: "Yes",
};

export const EMPTY_INPUT: Record<string, string> = {
  gender: "", age: "", admission_type_id: "", admission_source_id: "", discharge_disposition_id: "", medical_specialty: "",
  time_in_hospital: "", num_lab_procedures: "", num_procedures: "", num_medications: "", number_outpatient: "",
  number_emergency: "", number_inpatient: "", number_diagnoses: "", diag_1: "", diag_2: "", diag_3: "",
  max_glu_serum: "", A1Cresult: "", insulin: "", metformin: "", change: "", diabetesMed: "",
};

export const REQUIRED_CSV_COLUMNS = [
  "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
  "number_emergency", "number_inpatient", "number_diagnoses", "gender", "age", "admission_type_id",
  "discharge_disposition_id", "admission_source_id", "diag_1", "max_glu_serum", "A1Cresult", "change",
  "diabetesMed", "insulin", "metformin",
];
export const OPTIONAL_CSV_COLUMNS = ["record_id", "medical_specialty", "diag_2", "diag_3"];
export const FORBIDDEN_COLUMNS = ["name", "first_name", "last_name", "patient_name", "phone", "phone_number", "address", "email", "ssn", "dob", "date_of_birth", "mrn"];
export const MAX_BATCH_ROWS = 5000;
export const MAX_BATCH_BYTES = 4_000_000;
