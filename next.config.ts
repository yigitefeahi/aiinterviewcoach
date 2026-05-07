import type { NextConfig } from "next";

/** Explicit hosts — wildcards like `http://localhost:*` are not reliably supported and can block API fetch(). */
const csp = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  [
    "connect-src 'self'",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "ws://127.0.0.1:3000",
    "ws://localhost:3000",
    "ws://127.0.0.1:3001",
    "ws://localhost:3001",
  ].join(" "),
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
].join("; ");

const baseSecurityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  {
    key: "Permissions-Policy",
    value: "camera=(self), microphone=(self), geolocation=()",
  },
];

const nextConfig: NextConfig = {
  async headers() {
    // In dev, CSP connect-src is easy to get wrong for cross-port API calls — skip CSP to avoid "Failed to fetch".
    const isDev = process.env.NODE_ENV === "development";
    const headers = isDev
      ? baseSecurityHeaders
      : [...baseSecurityHeaders, { key: "Content-Security-Policy", value: csp }];
    return [{ source: "/:path*", headers }];
  },
};

export default nextConfig;
