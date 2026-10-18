import { NextResponse } from "next/server";
import { logger } from "@/utils/logger";
import { SESSION_HEADER } from "@/utils/session";

// Forwards /api/task requests to the FastAPI backend so the API key never reaches the browser.
const BACKEND_URL = process.env.WEBAGENT_BACKEND_URL || "http://127.0.0.1:8000";

/**
 * Headers the backend uses to identify and rate-limit this caller. The session ID comes from
 * the middleware (it deletes any browser-sent value); browser headers such as
 * X-Forwarded-For are never copied, so a browser can't pick its own client ID.
 */
export function backendClientHeaders(
  incoming?: Request,
): Record<string, string> {
  const headers: Record<string, string> = {};
  if (process.env.WEBAGENT_API_KEY) {
    headers["X-API-Key"] = process.env.WEBAGENT_API_KEY;
  }
  const sid = incoming?.headers.get(SESSION_HEADER);
  if (sid) {
    headers["X-WebAgent-Client"] = `s:${sid}`;
  }
  return headers;
}

export async function forwardToBackend(
  path: string,
  init: RequestInit,
  incoming?: Request,
): Promise<NextResponse> {
  const headers = new Headers(init.headers);
  Object.entries(backendClientHeaders(incoming)).forEach(([name, value]) =>
    headers.set(name, value),
  );

  try {
    const response = await fetch(`${BACKEND_URL}${path}`, {
      ...init,
      headers,
      cache: "no-store",
    });
    if (
      response.status === 204 ||
      response.status === 205 ||
      response.status === 304
    ) {
      // A null-body status (DELETE /api/topics/{id} answers 204) can't carry a JSON body.
      return new NextResponse(null, { status: response.status });
    }
    const body = await response.json().catch(() => ({}));
    return NextResponse.json(body, { status: response.status });
  } catch (error) {
    logger.error("Multi-agent backend unreachable", {
      path,
      error: error instanceof Error ? error.message : String(error),
    });
    return NextResponse.json(
      { error: "Multi-agent backend unavailable" },
      { status: 503, headers: { "X-WebAgent-Proxy-Error": "unreachable" } },
    );
  }
}
