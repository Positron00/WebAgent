/** The middleware's answer for an API call without a valid session cookie. */
export function isSignInRequired(status: number, data: unknown): boolean {
  return (
    status === 401 &&
    typeof data === "object" &&
    data !== null &&
    (data as { error?: unknown }).error === "sign_in_required"
  );
}

/** Sends the browser to the sign-in page, coming back to the current page afterwards. */
export function redirectToSignIn(): void {
  if (
    typeof window === "undefined" ||
    window.location.pathname.startsWith("/login")
  )
    return;
  const next = window.location.pathname + window.location.search;
  window.location.assign(`/login?next=${encodeURIComponent(next)}`);
}
