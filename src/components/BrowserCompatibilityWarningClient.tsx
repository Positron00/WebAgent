"use client";

import dynamic from "next/dynamic";

// Loaded only in the browser: it detects the browser, which can't run during server rendering.
// (Next.js 15 allows `ssr: false` only in a client component, not in the root layout.)
const BrowserCompatibilityWarning = dynamic(
  () => import("@/components/BrowserCompatibilityWarning"),
  { ssr: false },
);

export default BrowserCompatibilityWarning;
