"use client";

import { useApp } from "@/contexts/AppContext";

/** Shown only while the app can't reach its own server. It used to sit in the page all the time,
 *  hidden off-screen, where screen readers could still announce it as an alert. */
export default function OfflineBanner() {
  const { isOffline } = useApp();
  if (!isOffline) return null;
  return (
    <div
      id="offline-banner"
      className="fixed top-0 left-0 right-0 z-50 bg-yellow-500 text-black p-2 text-center"
      role="alert"
    >
      You are currently offline. Some features may be limited.
    </div>
  );
}
