import type { NextConfig } from "next";

/**
 * In production on Vercel, `/api/*` is routed to the FastAPI service by the
 * top-level `vercel.json` (Vercel Services). For local development (and local
 * `next start` smoke tests) we proxy `/api/*` to the FastAPI dev server instead.
 */
const isDev = process.env.NODE_ENV !== "production";
const proxyTarget = process.env.API_PROXY_TARGET ?? (isDev ? "http://127.0.0.1:8000" : undefined);

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    return proxyTarget ? [{ source: "/api/:path*", destination: `${proxyTarget}/api/:path*` }] : [];
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "X-Frame-Options", value: "DENY" },
        ],
      },
    ];
  },
};

export default nextConfig;
