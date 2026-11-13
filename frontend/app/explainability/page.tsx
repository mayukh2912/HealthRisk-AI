import type { Metadata } from "next";
import { ExplainabilityView } from "@/components/views/ExplainabilityView";
export const metadata: Metadata = { title: "Explainable AI · HealthRisk AI" };
export default function Page() { return <ExplainabilityView />; }
