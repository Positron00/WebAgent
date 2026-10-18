import { NextResponse } from "next/server";

/** The backend's request-size limit (10 MB by default; image attachments travel in the task body). */
export const DEFAULT_REQUEST_SIZE_LIMIT = 10 * 1024 * 1024;

export function requestSizeLimit(): number {
  const raw = Number(process.env.WEBAGENT_SECURITY_REQUEST_SIZE_LIMIT);
  return Number.isFinite(raw) && raw > 0
    ? Math.floor(raw)
    : DEFAULT_REQUEST_SIZE_LIMIT;
}

export type JsonBodyResult =
  | { ok: true; body: Record<string, unknown> }
  | { ok: false; response: NextResponse };

const tooLarge = () =>
  NextResponse.json({ error: "request_too_large" }, { status: 413 });
const invalidJson = () =>
  NextResponse.json({ error: "invalid_json" }, { status: 400 });

/** Reads the body as text, stopping as soon as it passes `limit` bytes. */
export async function readLimited(
  req: Request,
  limit: number,
): Promise<string | null> {
  const declared = Number(req.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > limit) return null;
  if (!req.body) return "";

  const reader = req.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > limit) {
      await reader.cancel().catch(() => undefined);
      return null;
    }
    chunks.push(value);
  }
  const all = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    all.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(all);
}

/**
 * Reads at most `limit` bytes (413 above it) and parses them as a JSON object (400 `invalid_json`
 * otherwise). Nothing reaches the backend on either error.
 */
export async function readJsonObject(
  req: Request,
  limit: number = requestSizeLimit(),
): Promise<JsonBodyResult> {
  const text = await readLimited(req, limit);
  if (text === null) return { ok: false, response: tooLarge() };
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, response: invalidJson() };
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return { ok: false, response: invalidJson() };
  }
  return { ok: true, body: parsed as Record<string, unknown> };
}

/** Copies only the listed fields that are present. */
export function pick(
  body: Record<string, unknown>,
  fields: readonly string[],
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const field of fields) {
    if (Object.prototype.hasOwnProperty.call(body, field))
      out[field] = body[field];
  }
  return out;
}
