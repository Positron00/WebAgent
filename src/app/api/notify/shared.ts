import { createHmac, randomBytes, timingSafeEqual } from "crypto";
import { NextResponse } from "next/server";
import { forwardToBackend } from "../task/backend";
import { readLimited } from "../task/body";

// Server-side helpers for the public notify pages and proxies (REC-134). Not a route file.
//
// These calls go to the backend with the API key and *without* a session (forwardToBackend gets
// no incoming request), so every public call shares the proxy's rate-limit bucket.

export type NotifyPurpose = "confirm" | "unsubscribe";

/** Tokens are base64url(payload) "." base64url(hmac) (backend §12); anything else is junk. */
const TOKEN_RE = /^[A-Za-z0-9_.=-]{1,2048}$/;
/** A form or JSON body holds one token and, for one-click, one more field. */
export const NOTIFY_BODY_LIMIT = 8 * 1024;
/** How long a result link from a form post stays readable. */
const RECEIPT_TTL_MS = 60 * 60 * 1000;

export function isTokenShape(value: unknown): value is string {
  return typeof value === "string" && TOKEN_RE.test(value);
}

export type LookupResult =
  | { ok: true; topicName: string }
  | { ok: false; reason: "invalid" | "unavailable" };

/**
 * Reads the topic name for a token through the backend's read-only `GET /api/notify/lookup`.
 * It changes nothing; a token for the other purpose counts as invalid.
 */
export async function lookupNotifyToken(
  token: unknown,
  purpose: NotifyPurpose,
): Promise<LookupResult> {
  if (!isTokenShape(token)) return { ok: false, reason: "invalid" };
  const response = await forwardToBackend(
    `/api/notify/lookup?token=${encodeURIComponent(token)}`,
    { method: "GET" },
  );
  if (!response.ok) {
    return {
      ok: false,
      reason: isInvalidStatus(response.status) ? "invalid" : "unavailable",
    };
  }
  const body: unknown = await response.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return { ok: false, reason: "unavailable" };
  }
  const { topic_name: name, purpose: got } = body as Record<string, unknown>;
  if (typeof name !== "string" || !name || got !== purpose) {
    return { ok: false, reason: "invalid" };
  }
  return { ok: true, topicName: name };
}

const isInvalidStatus = (status: number) =>
  status === 400 || status === 404 || status === 410 || status === 422;

export type ActionResult =
  | { ok: true; status: number; body: Record<string, unknown> }
  | {
      ok: false;
      status: number;
      error: "invalid_token" | "rate_limited" | "unavailable";
    };

/** `POST /api/notify/confirm` or `/unsubscribe` with `{token}`; errors carry no backend detail. */
export async function postNotifyAction(
  purpose: NotifyPurpose,
  token: string,
): Promise<ActionResult> {
  const response = await forwardToBackend(`/api/notify/${purpose}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
  if (response.ok) {
    const body: unknown = await response.json().catch(() => ({}));
    const status =
      body && typeof body === "object"
        ? (body as Record<string, unknown>).status
        : undefined;
    return {
      ok: true,
      status: 200,
      body: {
        status:
          typeof status === "string"
            ? status
            : purpose === "confirm"
              ? "confirmed"
              : "unsubscribed",
      },
    };
  }
  if (isInvalidStatus(response.status)) {
    return { ok: false, status: 404, error: "invalid_token" };
  }
  if (response.status === 429) {
    return { ok: false, status: 429, error: "rate_limited" };
  }
  return { ok: false, status: 503, error: "unavailable" };
}

// --- Result receipts ---
//
// After a form post the proxy redirects to the page with a short signed receipt holding the
// purpose and topic name, so the result page can name the topic without a second lookup (a
// used confirm token no longer verifies) and nobody can craft a "Confirmed" page for a topic.

let processKey: Buffer | null = null;

function receiptKey(): Buffer {
  const secret =
    process.env.WEBAGENT_SESSION_SECRET || process.env.WEBAGENT_API_KEY;
  if (secret) {
    return createHmac("sha256", secret)
      .update("webagent-notify-receipt")
      .digest();
  }
  processKey ??= randomBytes(32);
  return processKey;
}

const b64url = (buf: Buffer) => buf.toString("base64url");

export function signReceipt(
  purpose: NotifyPurpose,
  topicName: string,
  now: number = Date.now(),
): string {
  const payload = b64url(
    Buffer.from(JSON.stringify({ p: purpose, n: topicName, t: now })),
  );
  const sig = b64url(
    createHmac("sha256", receiptKey()).update(payload).digest(),
  );
  return `${payload}.${sig}`;
}

/** The topic name from a receipt for this purpose, or null if it's forged, stale or foreign. */
export function verifyReceipt(
  receipt: unknown,
  purpose: NotifyPurpose,
  now: number = Date.now(),
): string | null {
  if (typeof receipt !== "string" || receipt.length > 2048) return null;
  const [payload, sig, extra] = receipt.split(".");
  if (!payload || !sig || extra !== undefined) return null;
  const expected = createHmac("sha256", receiptKey()).update(payload).digest();
  const given = Buffer.from(sig, "base64url");
  if (given.length !== expected.length || !timingSafeEqual(given, expected)) {
    return null;
  }
  try {
    const data = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
    if (
      data?.p !== purpose ||
      typeof data.n !== "string" ||
      typeof data.t !== "number" ||
      now - data.t > RECEIPT_TTL_MS ||
      data.t - now > 60_000
    ) {
      return null;
    }
    return data.n;
  } catch {
    return null;
  }
}

// --- The proxy handler shared by both routes ---

/** A relative Location, so the redirect works behind any reverse proxy or host name. */
function seeOther(location: string): NextResponse {
  return new NextResponse(null, {
    status: 303,
    headers: { Location: location, "Cache-Control": "no-store" },
  });
}

const json = (body: Record<string, unknown>, status: number) =>
  NextResponse.json(body, {
    status,
    headers: { "Cache-Control": "no-store" },
  });

/**
 * POST handler for `/api/notify/confirm` and `/api/notify/unsubscribe`.
 *
 * - `application/json` `{token}` (a script): JSON back.
 * - Form `List-Unsubscribe=One-Click` with `?token=` (RFC 8058, unsubscribe only; a mail client):
 *   JSON back, never a redirect.
 * - Any other form post (the notify page's button, with a hidden `token` field): a 303 back to
 *   the page's result view, `?receipt=…` on success or `?error=invalid|unavailable`.
 */
export async function handleNotifyPost(
  req: Request,
  purpose: NotifyPurpose,
): Promise<NextResponse> {
  const contentType = (req.headers.get("content-type") || "")
    .split(";")[0]
    .trim()
    .toLowerCase();
  const queryToken = new URL(req.url).searchParams.get("token");
  const text = await readLimited(req, NOTIFY_BODY_LIMIT);
  if (text === null) return json({ error: "request_too_large" }, 413);

  if (contentType === "application/json") {
    let token: unknown = queryToken;
    try {
      const body = text ? JSON.parse(text) : {};
      if (body && typeof body === "object" && "token" in body) {
        token = body.token;
      }
    } catch {
      return json({ error: "invalid_json" }, 400);
    }
    return jsonAction(purpose, token);
  }

  if (contentType === "application/x-www-form-urlencoded") {
    const form = new URLSearchParams(text);
    if (
      purpose === "unsubscribe" &&
      form.get("List-Unsubscribe") === "One-Click"
    ) {
      return jsonAction(purpose, queryToken);
    }
    return formAction(purpose, form.get("token") ?? queryToken);
  }

  return json({ error: "unsupported_media_type" }, 415);
}

async function jsonAction(
  purpose: NotifyPurpose,
  token: unknown,
): Promise<NextResponse> {
  if (!isTokenShape(token)) return json({ error: "invalid_token" }, 404);
  const result = await postNotifyAction(purpose, token);
  return result.ok
    ? json(result.body, result.status)
    : json({ error: result.error }, result.status);
}

async function formAction(
  purpose: NotifyPurpose,
  token: unknown,
): Promise<NextResponse> {
  const page = `/notify/${purpose}`;
  // The read-only lookup first: it names the topic for the result page and rejects a token
  // for the other purpose before anything changes.
  const lookup = await lookupNotifyToken(token, purpose);
  if (!lookup.ok) return seeOther(`${page}?error=${lookup.reason}`);
  const result = await postNotifyAction(purpose, token as string);
  if (!result.ok) {
    return seeOther(
      `${page}?error=${result.error === "invalid_token" ? "invalid" : "unavailable"}`,
    );
  }
  return seeOther(
    `${page}?receipt=${encodeURIComponent(signReceipt(purpose, lookup.topicName))}`,
  );
}
