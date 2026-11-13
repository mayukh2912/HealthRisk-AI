"use client";
import { ShieldAlert } from "lucide-react";
import { useModelInfo } from "@/hooks/useApi";
import { Badge } from "@/components/ui/Card";

export function TopBar() {
  const { data, error } = useModelInfo();
  return (
    <div className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-2 px-4 py-2.5 sm:px-6">
        <p className="flex items-center gap-2 text-xs text-slate-600">
          <ShieldAlert className="h-4 w-4 text-amber-600" aria-hidden />
          Research prototype — predictions are statistical estimates, not diagnoses or clinical advice.
        </p>
        <div className="flex items-center gap-2 text-xs">
          {data ? (
            <>
              <Badge tone="teal">{data.model_name}</Badge>
              <Badge tone="navy">v{data.model_version}</Badge>
            </>
          ) : error ? (
            <Badge tone="red">API unavailable</Badge>
          ) : (
            <Badge>Connecting…</Badge>
          )}
        </div>
      </div>
    </div>
  );
}
