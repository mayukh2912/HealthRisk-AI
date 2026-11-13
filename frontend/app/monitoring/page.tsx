import type { Metadata } from "next";
import { MonitoringView } from "@/components/views/MonitoringView";
export const metadata: Metadata = { title: "Model Monitoring · HealthRisk AI" };
export default function Page() { return <MonitoringView />; }
