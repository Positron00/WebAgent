import { NextResponse } from "next/server";
import {
  codesMatch,
  createSession,
  readAccessConfig,
  SESSION_COOKIE,
} from "@/utils/session";
import { logger } from "@/utils/logger";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MIN_RESPONSE_MS = 500;
const MAX_CODE_LENGTH = 200;
const MAX_FAILURES_PER_MINUTE = 20;
const WINDOW_MS = 60_000;

// Shared by every caller of this server process.
const failures = { windowStart: 0, count: 0 };

function failuresThisMinute(now: number): number {
  if (now - failures.windowStart >= WINDOW_MS) {
    failures.windowStart = now;
    failures.count = 0;
  }
  return failures.count;
}

async function readCode(req: Request): Promise<string | null> {
  try {
    const body = await req.json();
    const code = body?.code;
    return typeof code === "string" && code.length <= MAX_CODE_LENGTH
      ? code
      : null;
  } catch {
    return null;
  }
}

function waitUntil(start: number, ms: number): Promise<void> {
  const remaining = start + ms - Date.now();
  return remaining > 0
    ? new Promise((resolve) => setTimeout(resolve, remaining))
    : Promise.resolve();
}

export async function POST(req: Request) {
  const started = Date.now();
  const access = readAccessConfig();

  if (access.kind === "misconfigured") {
    logger.error("Sign-in is not set up", { reason: access.reason });
    return NextResponse.json(
      { error: "access_code_not_configured" },
      { status: 503 },
    );
  }
  if (access.kind !== "on") {
    return NextResponse.json({ ok: true });
  }

  const code = await readCode(req);
  // Right and wrong codes answer after the same delay.
  await waitUntil(started, MIN_RESPONSE_MS);

  // Checked after the wait, so a burst of parallel guesses is still capped.
  if (failuresThisMinute(Date.now()) >= MAX_FAILURES_PER_MINUTE) {
    return NextResponse.json({ error: "too_many_attempts" }, { status: 429 });
  }
  if (code === null || !codesMatch(code, access.code)) {
    failures.count++;
    logger.warn("Sign-in with a wrong code", {
      failuresThisMinute: failures.count,
    });
    return NextResponse.json({ error: "wrong_code" }, { status: 401 });
  }

  const session = await createSession(access.secret);
  const response = NextResponse.json({ ok: true });
  response.cookies.set({
    name: SESSION_COOKIE,
    value: session.value,
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    maxAge: session.maxAge,
    secure: process.env.NODE_ENV === "production",
  });
  return response;
}
