"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Activity, BarChart3, BrainCircuit, FileSpreadsheet, Gauge, HeartPulse, LayoutDashboard, Menu, Radar, X } from "lucide-react";

export const NAV = [
  { href: "/", label: "Executive Dashboard", icon: LayoutDashboard },
  { href: "/risk-assessment", label: "Patient Risk Assessment", icon: HeartPulse },
  { href: "/batch", label: "Batch Prediction", icon: FileSpreadsheet },
  { href: "/performance", label: "Model Performance", icon: Gauge },
  { href: "/explainability", label: "Explainable AI", icon: BrainCircuit },
  { href: "/insights", label: "Data Insights", icon: BarChart3 },
  { href: "/monitoring", label: "Model Monitoring", icon: Radar },
];

export function Sidebar() {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const links = (
    <nav aria-label="Main" className="space-y-1">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? path === "/" : path?.startsWith(href);
        return (
          <Link key={href} href={href} onClick={() => setOpen(false)} aria-current={active ? "page" : undefined}
            className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${active ? "bg-white/10 font-medium text-white" : "text-slate-300 hover:bg-white/5 hover:text-white"}`}>
            <Icon className={`h-4 w-4 ${active ? "text-teal-500" : ""}`} aria-hidden />
            {label}
          </Link>
        );
      })}
    </nav>
  );
  const brand = (
    <div className="flex items-center gap-2.5 px-2">
      <span className="grid h-9 w-9 place-items-center rounded-lg bg-teal-500 text-white"><Activity className="h-5 w-5" aria-hidden /></span>
      <div>
        <p className="text-sm font-semibold leading-tight text-white">HealthRisk AI</p>
        <p className="text-[11px] leading-tight text-slate-400">Readmission risk analytics</p>
      </div>
    </div>
  );
  return (
    <>
      <div className="sticky top-0 z-30 flex items-center justify-between bg-navy-900 px-4 py-3 lg:hidden">
        {brand}
        <button aria-label={open ? "Close menu" : "Open menu"} onClick={() => setOpen(!open)} className="rounded-md p-1.5 text-white hover:bg-white/10">
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>
      {open && <div className="bg-navy-900 px-3 pb-4 lg:hidden">{links}</div>}
      <aside className="fixed inset-y-0 left-0 hidden w-64 flex-col gap-8 bg-navy-900 px-3 py-5 lg:flex">
        {brand}
        {links}
        <p className="mt-auto px-3 text-[11px] leading-relaxed text-slate-400">
          Educational research prototype. Not a medical device and not clinically validated.
        </p>
      </aside>
    </>
  );
}
