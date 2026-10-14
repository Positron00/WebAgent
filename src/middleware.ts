import { NextRequest, NextResponse } from "next/server";
import {
  readAccessConfig,
  verifySession,
  SESSION_COOKIE,
  SESSION_HEADER,
} from "@/utils/session";

export const config = {
  // Static files the sign-in page needs (public/icons, public/*.svg, the manifest) stay open.
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|icons/|manifest.json|.*\\.svg$|login|api/auth/login|api/health).*)",
  ],
};

let warnedOpen = false;

export async function middleware(req: NextRequest) {
  const access = readAccessConfig();
  const isApi = req.nextUrl.pathname.startsWith("/api/");
  const headers = new Headers(req.headers);
  headers.delete(SESSION_HEADER); // never trust a browser-sent value

  if (access.kind === "misconfigured") {
    return isApi
      ? NextResponse.json(
          { error: "access_code_not_configured" },
          { status: 503 },
        )
      : NextResponse.redirect(new URL("/login?error=config", req.url));
  }
  if (access.kind !== "on") {
    if (access.kind === "off-dev" && !warnedOpen) {
      warnedOpen = true;
      console.warn(
        "WEBAGENT_ACCESS_CODE is not set: the web app is open to anyone (development only).",
      );
    }
    return NextResponse.next({ request: { headers } });
  }

  const sid = await verifySession(
    req.cookies.get(SESSION_COOKIE)?.value,
    access.secret,
  );
  if (!sid) {
    if (isApi)
      return NextResponse.json({ error: "sign_in_required" }, { status: 401 });
    const url = new URL("/login", req.url);
    url.searchParams.set("next", req.nextUrl.pathname);
    return NextResponse.redirect(url);
  }
  headers.set(SESSION_HEADER, sid);
  return NextResponse.next({ request: { headers } });
}
