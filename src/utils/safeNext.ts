/** Where to go after sign-in: only a path on this site, so the redirect can't leave it. */
export function safeNextPath(next: string | null | undefined): string {
  if (
    !next ||
    !next.startsWith("/") ||
    next.startsWith("//") ||
    next.startsWith("/\\")
  )
    return "/";
  return next;
}
