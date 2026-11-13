import type { Metadata } from "next";
import { RiskAssessmentView } from "@/components/views/RiskAssessmentView";
export const metadata: Metadata = { title: "Patient Risk Assessment · HealthRisk AI" };
export default function Page() { return <RiskAssessmentView />; }
