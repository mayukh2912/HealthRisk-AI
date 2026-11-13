import { AlertTriangle, Loader2, RefreshCw } from "lucide-react";

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white p-8 text-sm text-slate-500">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> {label}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-wrap items-center gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
      <AlertTriangle className="h-4 w-4" aria-hidden />
      <span className="flex-1">{message}</span>
      {onRetry && (
        <button onClick={onRetry} className="inline-flex items-center gap-1 rounded-md bg-white px-2.5 py-1 text-xs font-medium ring-1 ring-red-200 hover:bg-red-100">
          <RefreshCw className="h-3 w-3" aria-hidden /> Retry
        </button>
      )}
    </div>
  );
}

export function Callout({ children, tone = "info" }: { children: React.ReactNode; tone?: "info" | "warn" }) {
  const cls = tone === "warn" ? "border-amber-200 bg-amber-50 text-amber-900" : "border-teal-100 bg-teal-50 text-navy-900";
  return <div className={`rounded-lg border px-4 py-3 text-sm ${cls}`}>{children}</div>;
}
