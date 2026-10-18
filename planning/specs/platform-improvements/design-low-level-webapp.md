# Low-level design: web app

Paths are relative to the repo root. Requirement IDs refer to [`requirements.md`](requirements.md).

`apiClient.ts` (791 lines) and `ChatContext.tsx` (488 lines) are already far over the
200-line rule. New logic goes into new small modules, and the two big files only call
it (IMP-NFR-05).

## 1. Files

| File | New? | Purpose | Release | Approx. size |
|---|---|---|---|---|
| `src/middleware.ts` | new | Check `wa_session` on pages and API routes; set `x-webagent-session` | 3.2.7 | ~70 lines |
| `src/utils/session.ts` | new | Sign and verify session cookies with Web Crypto (Edge and Node) | 3.2.7 | ~80 lines |
| `src/app/api/auth/login/route.ts` | new | Check the code, set the cookie | 3.2.7 | ~60 lines |
| `src/app/api/auth/logout/route.ts` | new | Clear the cookie | 3.2.7 | ~15 lines |
| `src/app/login/page.tsx` | new | Sign-in form | 3.2.7 | ~80 lines |
| `src/utils/rateLimiter.ts` | changed | One bucket per key | 3.2.7 | ±30 lines |
| `src/app/api/chat/route.ts` | changed | `addRequest`; key by session; image check (3.3.0) | 3.2.7, 3.3.0 | +25 lines |
| `src/app/api/task/backend.ts` | changed | Send `X-WebAgent-Client`; stop copying `X-Forwarded-For` | 3.2.7 | ±15 lines |
| `src/app/api/task/[taskId]/cancel/route.ts` | new | Forward cancel | 3.3.0 | ~10 lines |
| `src/utils/fetchWithRetry.ts` | new (moved out of `apiClient.ts`) | Per-attempt controller and timeout | 3.2.7 | ~90 lines |
| `src/utils/taskPolling.ts` | new (moved out of `apiClient.ts`) | Poll with abort, progress callback and typed failures | 3.3.0 | ~120 lines |
| `src/types/research.ts` | new | `ResearchProgress`, `ResearchFailure` | 3.3.0 | ~30 lines |
| `src/hooks/useResearchRun.ts` | new | Current task ID, Stop, progress state for `ChatContext` | 3.3.0 | ~90 lines |
| `src/components/chat/ResearchStatus.tsx` | new | "Thinking…" with step, log line, elapsed time and Stop | 3.3.0 | ~70 lines |
| `src/components/chat/ResearchFailed.tsx` | new | Failure message with Try again / Answer directly instead | 3.3.0 | ~60 lines |
| `src/utils/safeUrl.ts` | new | Allow only `http:` and `https:` | 3.3.0 | ~15 lines |
| `src/utils/apiClient.ts` | changed | Use the moved modules; remove silent fallback; 401 → sign-in | 3.2.7, 3.3.0 | −200 / +40 |
| `src/contexts/ChatContext.tsx` | changed | Use `useResearchRun`; failure state; retry and direct actions | 3.3.0 | ±40 lines |
| `src/components/chat/MessageList.tsx` | changed | Render `ResearchStatus` / `ResearchFailed`; diagnostics → `/api/ping` | 3.3.0 | −40 / +10 |
| `src/components/chat/SourceCanvas.tsx` | changed | `safeUrl` | 3.3.0 | +3 lines |
| `src/app/api/ping/route.ts`, `src/app/api/health/route.ts` | changed | Minimal replies | 3.3.0 | −150 lines |
| `next.config.js` | changed | `headers()`; image hosts from env | 3.3.0 | +40 lines |
| `src/utils/storage.ts` | changed | Handle `QuotaExceededError` | 3.4.0 | +20 lines |
| `src/utils/apiClient.ts` (logging) | changed | Log the query's length, not its text | 3.3.0 | 2 lines |
| `src/utils/errors.ts` | changed | The single `ApiError` | 3.4.0 | small |

## 2. Session cookie (`src/utils/session.ts`, IMP-010)

Format: `wa_session=<sid>.<exp>.<sig>`

- `sid`: 16 random bytes, base64url (22 characters).
- `exp`: expiry, Unix seconds.
- `sig`: base64url HMAC-SHA256 of `"<sid>.<exp>"` with `WEBAGENT_SESSION_SECRET`.

```ts
export const SESSION_COOKIE = 'wa_session';
export const SESSION_HEADER = 'x-webagent-session';
export const SESSION_DAYS = 30;

export type AccessConfig =
  | { kind: 'on'; code: string; secret: string }
  | { kind: 'off-dev' }          // no code, development: allow, warn once
  | { kind: 'off-explicit' }     // WEBAGENT_ACCESS_CODE_DISABLED=true
  | { kind: 'misconfigured'; reason: string };   // production without code, or short secret/code

export function readAccessConfig(env = process.env): AccessConfig;
export async function createSession(secret: string, now = Date.now()): Promise<{ sid: string; value: string; maxAge: number }>;
export async function verifySession(value: string | undefined, secret: string, now = Date.now()): Promise<string | null>;  // sid or null
export function codesMatch(given: string, expected: string): boolean;   // constant time over the longer length
```

- Uses only `crypto.subtle` and `crypto.getRandomValues`, available in the Edge runtime
  (middleware) and in Node 18+ (route handlers). No new dependency (IMP-NFR-02).
- `verifySession` rejects: missing value, wrong number of parts, `exp` in the past, a
  signature that doesn't match (compared with `crypto.subtle.verify`, which is constant
  time).
- `readAccessConfig` returns `misconfigured` when the code is shorter than 12 or the
  secret shorter than 32 characters, or when `NODE_ENV === 'production'` and no code is
  set and `WEBAGENT_ACCESS_CODE_DISABLED !== 'true'`.

## 3. Middleware (`src/middleware.ts`, IMP-010)

```ts
export const config = {
  // Static files the sign-in page needs (public/icons, public/*.svg, the manifest) stay open.
  matcher: ['/((?!_next/static|_next/image|favicon.ico|icons/|manifest.json|.*\\.svg$|login|api/auth/login|api/health).*)'],
};

export async function middleware(req: NextRequest) {
  const access = readAccessConfig();
  const isApi = req.nextUrl.pathname.startsWith('/api/');
  const headers = new Headers(req.headers);
  headers.delete(SESSION_HEADER);                     // never trust a browser-sent value

  if (access.kind === 'misconfigured') {
    return isApi
      ? NextResponse.json({ error: 'access_code_not_configured' }, { status: 503 })
      : NextResponse.redirect(new URL('/login?error=config', req.url));
  }
  if (access.kind !== 'on') return NextResponse.next({ request: { headers } });

  const sid = await verifySession(req.cookies.get(SESSION_COOKIE)?.value, access.secret);
  if (!sid) {
    if (isApi) return NextResponse.json({ error: 'sign_in_required' }, { status: 401 });
    const url = new URL('/login', req.url);
    url.searchParams.set('next', req.nextUrl.pathname);
    return NextResponse.redirect(url);
  }
  headers.set(SESSION_HEADER, sid);
  return NextResponse.next({ request: { headers } });
}
```

The `next` parameter is used only if it starts with `/` and not `//`, so the redirect
after sign-in can't leave the site.

## 4. Sign-in (`/api/auth/login`, `/api/auth/logout`, `/login`)

`POST /api/auth/login` (Node runtime):

1. `readAccessConfig()`; `misconfigured` → 503 `{error: 'access_code_not_configured'}`
   (the same answer the middleware gives other API routes) and a log line naming the
   variable; not `on` → 200 (nothing to do).
2. Parse `{code}` (at most 200 characters).
3. Wait until 500 ms have passed since the request started (same time for right and
   wrong codes).
4. `codesMatch` false → 401 `{error: 'wrong_code'}`. Increment a process-wide failure
   counter; after 20 failures in a minute, answer 429 for the rest of that minute.
5. Right → `createSession`, `Set-Cookie: wa_session=…; HttpOnly; SameSite=Lax; Path=/;
   Max-Age=2592000` plus `Secure` when `NODE_ENV === 'production'`.

`POST /api/auth/logout` sets the same cookie with `Max-Age=0`.

`/login` page: one password field ("Access code"), a **Sign in** button, an error line
("That code is wrong", "Too many tries, wait a minute", "Sign-in isn't set up on this
server"), and on success `router.replace(next || '/')`. Keyboard: the field has focus on
load; Enter submits.

`apiClient`: a 401 with `error: 'sign_in_required'` sends the browser to
`/login?next=<current path>` instead of showing an error.

## 5. Rate limiter per key (`src/utils/rateLimiter.ts`, `/api/chat`, IMP-011)

```ts
class RateLimiter {
  private buckets = new Map<string, number[]>();
  isRateLimited(key: string): boolean;          // prunes the key's old timestamps
  addRequest(key: string): void;
  getTimeUntilNextAllowed(key: string): number;
  private sweep(): void;                          // drop empty buckets every 5 minutes
}
```

The per-key backoff multiplier is removed: a fixed window is simpler to reason about and
test. `/api/chat`:

```ts
const key = req.headers.get(SESSION_HEADER) ?? 'shared';
if (rateLimiter.isRateLimited(key)) { /* 429 as today */ }
rateLimiter.addRequest(key);
```

## 6. Proxy (`src/app/api/task/backend.ts`, IMP-012)

`callerAddress` is deleted. `forwardToBackend` sets:

```ts
const headers = new Headers(init.headers);       // only what the route passes, never the browser's headers
if (process.env.WEBAGENT_API_KEY) headers.set('X-API-Key', process.env.WEBAGENT_API_KEY);
const sid = incoming?.headers.get(SESSION_HEADER);
if (sid) headers.set('X-WebAgent-Client', `s:${sid}`);
```

New route `src/app/api/task/[taskId]/cancel/route.ts`:

```ts
export async function POST(req: Request, { params }: { params: { taskId: string } }) {
  return forwardToBackend(`/api/task/${encodeURIComponent(params.taskId)}/cancel`, { method: 'POST' }, req);
}
```

## 7. Retry with a fresh timeout (`src/utils/fetchWithRetry.ts`, IMP-044)

Moved out of `ApiClient` as a function; `ApiClient.fetchWithRetry` calls it.

```ts
export async function fetchWithRetry<T>(url: string, init: RequestInit, opts: {
  timeoutMs: number; maxRetries: number; signal?: AbortSignal; onAttempt?: (a: number) => void;
}): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new DOMException('timeout', 'TimeoutError')), opts.timeoutMs);
    const onOuterAbort = () => controller.abort(opts.signal?.reason);
    opts.signal?.addEventListener('abort', onOuterAbort, { once: true });
    try {
      const res = await fetch(url, { ...init, signal: controller.signal });
      // … status handling as today
      return data;
    } catch (err) {
      if (opts.signal?.aborted || !isRetryable(err) || attempt >= opts.maxRetries) throw err;
      await delay(retryDelay(attempt + 1));
    } finally {
      clearTimeout(timer);
      opts.signal?.removeEventListener('abort', onOuterAbort);
    }
  }
}
```

An abort from the caller's `signal` (the Stop button) is never retried.

## 8. Polling with progress and typed failures (`src/utils/taskPolling.ts`, IMP-051–054)

```ts
// src/types/research.ts
export interface ResearchProgress { step: string; latestLog?: string; elapsedMs: number }
export type ResearchFailureKind = 'timeout' | 'budget' | 'cancelled' | 'failed' | 'unreachable';
export class ResearchFailed extends Error { constructor(public kind: ResearchFailureKind, message: string) { super(message); } }
```

```ts
export async function pollTask(taskId: string, opts: {
  signal: AbortSignal;
  onProgress?: (p: ResearchProgress) => void;
  deadlineMs?: number;            // default 8 minutes, as today: a backstop just over the backend's 7
}): Promise<ChatCompletionResponse>;
```

- Same back-off as today (1 s, ×1.5, at most 5 s).
- On each `processing` reply with `progress`, call `onProgress({step, latestLog:
  logs.at(-1), elapsedMs})`. Without `progress`, report `step: 'Working'`.
- `failed` with `error_code`: `timeout`, `budget` and `cancelled` map to their kinds;
  `escalated` keeps today's guard path; anything else is `failed` with the backend's
  message.
- The proxy's 503 with `X-WebAgent-Proxy-Error: unreachable` maps to `unreachable`.
- `signal` aborted → throw `ResearchFailed('cancelled', 'Research stopped.')` at once,
  without another request.
- The client deadline is a backstop. With the backend limit (IMP-041) the backend
  normally reports `timeout` first.

`ApiClient.sendMultiAgentRequest` gains `opts: {signal, onProgress, onTaskId}`. It calls
`onTaskId(task_id)` as soon as the task exists (so Stop can cancel it), then
`pollTask(...)`. **The `catch` that falls back to `/api/chat` is removed** (IMP-054).
Guard refusals still pass through unchanged; every other error becomes a `ResearchFailed`.

`ApiClient.cancelTask(taskId)`: `POST /api/task/<id>/cancel`; a 409 (already finished)
is ignored.

## 9. Research state (`src/hooks/useResearchRun.ts`, `ChatContext`, IMP-051, IMP-053, IMP-054)

```ts
export function useResearchRun() {
  // state: { taskId?: string; progress?: ResearchProgress; failure?: { kind, message, retry: RetryInfo } }
  return {
    progress, failure,
    begin(): { signal: AbortSignal; onProgress; onTaskId },   // new AbortController per run
    stop(): Promise<void>,          // abort polling, then apiClient.cancelTask(taskId)
    fail(kind, message, retry): void,
    clear(): void,
  };
}
```

`ChatContext`:

- `runChatRequest` passes `research.begin()` options to `sendChatMessage` when the route
  is agentic, and calls `research.clear()` when an answer arrives.
- On `ResearchFailed`: `cancelled` → append a short assistant note "Research stopped."
  (not an error); other kinds → `research.fail(kind, message, {messages, imageBase64})`
  and `isLoading: false`.
- New context functions: `stopResearch()`, `retryResearch()` (same messages, same
  mode), `answerDirectlyInstead()` (same messages, mode `off`, so the request goes to
  `/api/chat`).
- `isLoading` goes false as soon as Stop is pressed, so a new message can be sent at once.

## 10. Components

### `ResearchStatus.tsx`

Replaces the "Thinking…" block in `MessageList.tsx:462-476` when a research run is in
progress (direct answers keep today's spinner). Shows the spinner (respecting reduced
motion), "Thinking… · <step>", the latest log line in smaller grey text, the elapsed
time as `m:ss`, and a **Stop** button (`aria-label="Stop research"`). The region keeps
`role="status"`; only the step text is in the live region, so screen readers aren't
told every second.

### `ResearchFailed.tsx`

Replaces the error box when `failure` is set. Text by kind:

| Kind | Text |
|---|---|
| `timeout` | The research took too long and was stopped. |
| `budget` | The research used its token budget and was stopped. |
| `unreachable` | The research service isn't reachable right now. |
| `failed` | Deep research failed: <message, at most 200 characters>. |

Buttons: **Try again** (`retryResearch`) and **Answer directly instead**
(`answerDirectlyInstead`).

### `MessageList.tsx`

- The diagnostics block (`MessageList.tsx:486-524`) calls `GET /api/ping` only, and shows
  "Web app reachable" or the HTTP status.
- `ReactMarkdown` is unchanged (its default URL filter already removes `javascript:`).

### `SourceCanvas.tsx`

```tsx
const href = safeUrl(source.url);       // string | null
{href ? <a href={href} target="_blank" rel="noopener noreferrer">Visit Source →</a> : <span>{source.url}</span>}
```

## 11. Image check (`/api/chat`, IMP-061)

```ts
const IMAGE_RE = /^data:image\/(png|jpeg|gif|webp);base64,([A-Za-z0-9+/=]+)$/;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
function imageError(image: unknown): string | null {
  if (image == null) return null;
  if (typeof image !== 'string') return 'Image must be a data URL';
  const m = IMAGE_RE.exec(image);
  if (!m) return 'Only PNG, JPEG, GIF or WebP images are accepted';
  if (Math.floor((m[2].length * 3) / 4) > MAX_IMAGE_BYTES) return 'Image is larger than 5 MB';
  return null;
}
```

A non-null result answers 400 `{error: <text>}`. `Chat.tsx` `handleFileSelect` checks
`file.size > MAX_IMAGE_BYTES` and shows the same text as a notification.

## 12. Headers and images (`next.config.js`, IMP-060, IMP-066)

```js
const isProd = process.env.NODE_ENV === 'production';
const csp = [
  "default-src 'self'",
  `script-src 'self' 'unsafe-inline'${isProd ? '' : " 'unsafe-eval'"}`,
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "connect-src 'self'",
  "media-src 'self' blob:",
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
].join('; ');

async headers() {
  return [{ source: '/:path*', headers: [
    { key: 'Content-Security-Policy', value: csp },
    { key: 'X-Frame-Options', value: 'DENY' },
    { key: 'X-Content-Type-Options', value: 'nosniff' },
    { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
    ...(isProd ? [{ key: 'Strict-Transport-Security', value: 'max-age=31536000' }] : []),
  ]}];
},
images: {
  remotePatterns: (process.env.WEBAGENT_IMAGE_HOSTS || '').split(',').map(h => h.trim()).filter(Boolean)
    .map(hostname => ({ protocol: 'https', hostname })),
},
```

`'unsafe-inline'` for scripts is needed for Next.js's inline bootstrap without nonces (14 and 15)
(see known limitations). `connect-src 'self'` means every browser `fetch` must stay on the
web app origin. **`clientConfig.ts`** always sets `api.baseUrl` to `'/api'` (do not point
at `NEXT_PUBLIC_VERCEL_URL`, which would break the session cookie and CSP). **`AppContext`**
treats a failed `HEAD /api/ping` as offline; it must not probe third-party hosts (for
example `google.com`). Document **`WEBAGENT_IMAGE_HOSTS`** in `.env.local.example`. The
task checks voice input, screenshot capture and image upload under the policy (IMP-060).

## 13. Minimal open endpoints (IMP-063)

- `/api/ping` → `{status: 'ok', timestamp}`.
- `/api/health` → `{status, version, timestamp}`; `status` is `'healthy'` or `'degraded'`
  from `apiClient.isHealthy()`. The metrics, logs, memory and browser details are removed
  (they were for debugging; use the server logs instead).

## 13a. No question text in logs (IMP-070, 3.3.0)

`src/utils/logger.ts` writes only to the console (no remote sink). The server routes
don't log question text today; the test keeps it that way. In the browser,
`apiClient.ts:539` logs the whole query and `:578` its first 100 characters; both change
to `{ queryChars: query.length }`, so a shared screen or a pasted console log doesn't
leak the question. `src/app/api/guard/client.ts:67,76,84` log the backend's refusal body,
which holds the reason and policy IDs, not the question; unchanged.

## 14. Storage full (`src/utils/storage.ts`, IMP-092, 3.4.0)

```ts
saveMessages(messages) {
  try { write(messages.slice(-MAX_MESSAGES)); }
  catch (e) {
    if (!isQuotaError(e)) { console.error(...); return; }
    const kept = messages.slice(-Math.max(1, Math.floor(Math.min(messages.length, MAX_MESSAGES) / 2)));
    try { write(kept); notifyTrimmed(); } catch { console.error('Could not save chat history'); }
  }
}
```

`notifyTrimmed` dispatches a `window` event that `AppContext` turns into the notification
"Older messages were removed to save space."

## 15. Tests (Jest)

| Test file | Covers |
|---|---|
| `src/__tests__/utils/session.test.ts` | Sign, verify, expired, tampered, wrong secret, short code or secret, `readAccessConfig` cases; 1,000 verifications under 5 s total (IMP-NFR-03) |
| `src/__tests__/middleware.test.ts` | API 401, page redirect, misconfigured 503, dev off, browser-sent `x-webagent-session` removed, `next` safety |
| `src/__tests__/api/authLogin.test.ts` | Right code sets cookie (flags), wrong code 401 after delay, 429 after 20 failures, logout |
| `src/__tests__/utils/rateLimiter.test.ts` | Per-key buckets (IMP-011) |
| `src/__tests__/api/chatRoute.test.ts` (extend) | 11th request from one session is 429, other session passes; image checks (IMP-061) |
| `src/__tests__/api/taskProxy.test.ts` (extend) | `X-WebAgent-Client` set; browser `X-Forwarded-For` not forwarded; cancel route |
| `src/__tests__/utils/fetchWithRetry.test.ts` | Fresh signal per attempt; timer cleared; caller abort not retried (IMP-044) |
| `src/__tests__/utils/taskPolling.test.ts` | Progress callback; each failure kind; abort; no fallback to `/api/chat` |
| `src/hooks/__tests__/useResearchRun.test.tsx` | Stop aborts and cancels; retry and direct actions |
| `src/components/chat/__tests__/ResearchStatus.test.tsx`, `ResearchFailed.test.tsx` | Rendering and buttons |
| `src/__tests__/utils/safeUrl.test.ts`, `SourceCanvas` test | IMP-062 |
| `src/__tests__/api/pingHealth.test.ts` | IMP-063 reply shapes |
| `src/__tests__/nextConfig.test.ts` | IMP-060 headers present; IMP-066 image hosts |
| `src/__tests__/utils/storage.test.ts` | IMP-092 |

Jest runs in jsdom, which lacks `crypto.subtle` on older Node; `jest.setup.js` assigns
`globalThis.crypto = require('node:crypto').webcrypto` when it's missing.

## 16. Release 3.5.0 (IMP-128)

No web app or Next.js changes. Advanced observability is backend-only (`/api/v1/observability`
behind the API key). Operators use the runbook; the chat UI continues to call `/api/task` and
does not surface drift APIs in 3.5.0.
