import type { Metadata } from "next";
import { InsightsView } from "@/components/views/InsightsView";
export const metadata: Metadata = { title: "Data Insights · HealthRisk AI" };
export default function Page() { return <InsightsView />; }
