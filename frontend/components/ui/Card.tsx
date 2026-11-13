import type { ReactNode } from "react";

export function Card({ title, subtitle, actions, children, className = "" }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`rounded-xl border border-slate-200 bg-white shadow-sm ${className}`}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-2 border-b border-slate-100 px-5 py-3.5">
          <div>
            {title && <h2 className="text-sm font-semibold text-navy-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function StatCard({ label, value, hint, icon, tone = "navy" }: {
  label: string; value: ReactNode; hint?: ReactNode; icon?: ReactNode; tone?: "navy" | "teal" | "slate";
}) {
  const ring = tone === "teal" ? "bg-teal-50 text-teal-700" : tone === "slate" ? "bg-slate-100 text-slate-600" : "bg-navy-900/5 text-navy-800";
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
        {icon && <span className={`rounded-lg p-1.5 ${ring}`}>{icon}</span>}
      </div>
      <p className="mt-2 text-2xl font-semibold text-navy-900 tabular-nums" data-testid={`stat-${label}`}>{value}</p>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  );
}

export function Badge({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "teal" | "navy" | "amber" | "red" | "green" }) {
  const map = {
    slate: "bg-slate-100 text-slate-700", teal: "bg-teal-50 text-teal-700 ring-1 ring-teal-100",
    navy: "bg-navy-900 text-white", amber: "bg-amber-50 text-amber-800 ring-1 ring-amber-200",
    red: "bg-red-50 text-red-800 ring-1 ring-red-200", green: "bg-emerald-50 text-emerald-800 ring-1 ring-emerald-200",
  };
  return <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${map[tone]}`}>{children}</span>;
}

export function PageHeader({ title, description, children }: { title: string; description?: ReactNode; children?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-navy-900">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-slate-600">{description}</p>}
      </div>
      {children}
    </div>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return <h3 className="mb-3 mt-8 text-xs font-semibold uppercase tracking-wider text-slate-500">{children}</h3>;
}
