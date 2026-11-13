import type { Metadata } from "next";
import { BatchView } from "@/components/views/BatchView";
export const metadata: Metadata = { title: "Batch Prediction · HealthRisk AI" };
export default function Page() { return <BatchView />; }
