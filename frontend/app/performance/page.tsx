import type { Metadata } from "next";
import { PerformanceView } from "@/components/views/PerformanceView";
export const metadata: Metadata = { title: "Model Performance · HealthRisk AI" };
export default function Page() { return <PerformanceView />; }
