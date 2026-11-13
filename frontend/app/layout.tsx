import type { Metadata } from "next";
import "../styles/globals.css";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopBar } from "@/components/layout/TopBar";

export const metadata: Metadata = {
  title: "HealthRisk AI — Readmission Risk Analytics",
  description: "Educational prototype estimating 30-day hospital readmission risk with explainable machine learning.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <Sidebar />
        <div className="lg:pl-64">
          <TopBar />
          <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:py-8">{children}</main>
          <footer className="mx-auto max-w-7xl px-4 pb-8 text-xs text-slate-500 sm:px-6">
            Data: Diabetes 130-US Hospitals 1999–2008 (Strack et al., 2014; UCI ML Repository, CC BY 4.0). HealthRisk AI is
            not a medical device and must not be used for clinical decision-making.
          </footer>
        </div>
      </body>
    </html>
  );
}
