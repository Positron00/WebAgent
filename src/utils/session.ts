// Signed session cookie for the access code (IMP-010).
// Web Crypto only (crypto.subtle, crypto.getRandomValues): this file runs in Edge middleware
// and in Node route handlers, so it must not import Node modules or the app logger.

export const SESSION_COOKIE = "wa_session";
export const SESSION_HEADER = "x-webagent-session";
export const SESSION_DAYS = 30;
export const SESSION_MAX_AGE = SESSION_DAYS * 24 * 60 * 60;

const MIN_CODE_LENGTH = 12;
const MIN_SECRET_LENGTH = 32;

export type AccessConfig =
  | { kind: "on"; code: string; secret: string }
  | { kind: "off-dev" }
  | { kind: "off-explicit" }
  | { kind: "misconfigured"; reason: string };

type Env = Record<string, string | undefined>;

export function readAccessConfig(env: Env = process.env): AccessConfig {
  if (env.WEBAGENT_ACCESS_CODE_DISABLED === "true")
    return { kind: "off-explicit" };

  const code = env.WEBAGENT_ACCESS_CODE ?? "";
  const secret = env.WEBAGENT_SESSION_SECRET ?? "";
  if (!code) {
    return env.NODE_ENV === "production"
      ? { kind: "misconfigured", reason: "WEBAGENT_ACCESS_CODE is not set" }
      : { kind: "off-dev" };
  }
  if (code.length < MIN_CODE_LENGTH) {
    return {
      kind: "misconfigured",
      reason: `WEBAGENT_ACCESS_CODE is shorter than ${MIN_CODE_LENGTH} characters`,
    };
  }
  if (secret.length < MIN_SECRET_LENGTH) {
    return {
      kind: "misconfigured",
      reason: `WEBAGENT_SESSION_SECRET is shorter than ${MIN_SECRET_LENGTH} characters`,
    };
  }
  return { kind: "on", code, secret };
}

function toBase64Url(bytes: Uint8Array): string {
  let binary = "";
  bytes.forEach((b) => {
    binary += String.fromCharCode(b);
  });
  return btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

function fromBase64Url(text: string): Uint8Array<ArrayBuffer> | null {
  if (!/^[A-Za-z0-9_-]*$/.test(text)) return null;
  try {
    const binary = atob(text.replace(/-/g, "+").replace(/_/g, "/"));
    const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
    // The last character can carry unused bits; reject any spelling but the canonical one,
    // so that every changed character of the cookie fails.
    return toBase64Url(bytes) === text ? bytes : null;
  } catch {
    return null;
  }
}

const encoder = new TextEncoder();

function hmacKey(secret: string, usage: "sign" | "verify"): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    [usage],
  );
}

export async function createSession(
  secret: string,
  now = Date.now(),
): Promise<{ sid: string; value: string; maxAge: number }> {
  const sid = toBase64Url(crypto.getRandomValues(new Uint8Array(16)));
  const exp = Math.floor(now / 1000) + SESSION_MAX_AGE;
  const payload = `${sid}.${exp}`;
  const key = await hmacKey(secret, "sign");
  const sig = new Uint8Array(
    await crypto.subtle.sign("HMAC", key, encoder.encode(payload)),
  );
  return {
    sid,
    value: `${payload}.${toBase64Url(sig)}`,
    maxAge: SESSION_MAX_AGE,
  };
}

/** Returns the session ID when the cookie is valid and not expired, otherwise null. */
export async function verifySession(
  value: string | undefined,
  secret: string,
  now = Date.now(),
): Promise<string | null> {
  if (!value) return null;
  const parts = value.split(".");
  if (parts.length !== 3) return null;
  const [sid, expText, sigText] = parts;
  if (!sid || !/^\d+$/.test(expText)) return null;
  if (Number(expText) * 1000 <= now) return null;

  const sig = fromBase64Url(sigText);
  if (!sig || sig.length === 0) return null;
  try {
    const key = await hmacKey(secret, "verify");
    // subtle.verify compares in constant time.
    const ok = await crypto.subtle.verify(
      "HMAC",
      key,
      sig,
      encoder.encode(`${sid}.${expText}`),
    );
    return ok ? sid : null;
  } catch {
    return null;
  }
}

/** Constant-time comparison over the longer of the two lengths. */
export function codesMatch(given: string, expected: string): boolean {
  const a = encoder.encode(given);
  const b = encoder.encode(expected);
  const length = Math.max(a.length, b.length);
  let diff = a.length ^ b.length;
  for (let i = 0; i < length; i++) {
    diff |= (a[i] ?? 0) ^ (b[i] ?? 0);
  }
  return diff === 0;
}
