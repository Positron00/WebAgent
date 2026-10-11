# Low-level design: web app

Paths are relative to the repo root. Requirement IDs refer to [`requirements.md`](requirements.md).

## 1. Files

| File | New? | Change |
|---|---|---|
| `src/types/routing.ts` | new | `AgenticMode`, `Route`, `DecidedBy`, `RouteInfo`, `TriageResponse` |
| `src/types/chat.ts` | changed | `ChatMessage.route?`, `ChatMessage.imageAttached?` |
| `src/types/api.ts` | changed | `ChatCompletionResponse.route?: RouteInfo` |
| `src/utils/agenticMode.ts` | new | Read/save/migrate the stored mode (~70 lines) |
| `src/utils/storage.ts` | changed | Drop `agentic` from `AccessibilitySettings` and its default; strip a stray saved `agentic` key |
| `src/utils/routeQuestion.ts` | new | Triage call and status-code mapping (~90 lines) |
| `src/utils/apiClient.ts` | changed | `sendChatMessage` takes a mode; `previous_messages`; route info on responses |
| `src/app/api/settings/route.ts` | new | `GET` runtime default (~70 lines) |
| `src/app/api/triage/route.ts` | new | Proxy to FastAPI (~15 lines) |
| `src/app/api/task/backend.ts` | changed | `X-WebAgent-Proxy-Error: unreachable` on 503 |
| `src/contexts/AppContext.tsx` | changed | `agenticMode`, `setAgenticMode`, `modeReady` |
| `src/contexts/ChatContext.tsx` | changed | Remove mock sources; store route; `researchInDepth()` |
| `src/components/chat/AgenticModeToggle.tsx` | new | Three-way control (~90 lines) |
| `src/components/chat/RouteLabel.tsx` | new | Label + hover reason + override button (~70 lines) |
| `src/components/chat/MessageInput.tsx` | changed | Use the toggle; image note in Always |
| `src/components/chat/MessageList.tsx` | changed | Render `RouteLabel`; logs by route |
| `src/test-utils.tsx` | changed | Default mode in the test provider |
| `.env.local.example` | changed | `WEBAGENT_DEFAULT_AGENTIC_MODE=off` with notes |

New components are separate files so `MessageInput.tsx` and `MessageList.tsx` don't grow.

## 2. Types (`src/types/routing.ts`)

```ts
export type AgenticMode = 'off' | 'auto' | 'always';
export type Route = 'direct' | 'agentic';
export type DecidedBy = 'rule' | 'model' | 'fallback' | 'config' | 'mode' | 'override';

export interface RouteInfo {
  route: Route;
  decidedBy: DecidedBy;
  reason: string;          // shown on hover
  pAgentic?: number | null;
}

export interface TriageResponse {          // matches the FastAPI body
  route: Route;
  p_agentic: number | null;
  decided_by: 'rule' | 'model' | 'fallback' | 'config';   // the backend never sends mode/override here
  reason: string;
  classifier_version: string;
}

export const AGENTIC_MODES: readonly AgenticMode[] = ['off', 'auto', 'always'];
export const isAgenticMode = (v: unknown): v is AgenticMode =>
  typeof v === 'string' && (AGENTIC_MODES as readonly string[]).includes(v);
```

`ChatMessage` gets `route?: RouteInfo` and `imageAttached?: boolean` (set on user messages that carried an image, so the override button can be hidden).

## 3. Stored mode and migration (`src/utils/agenticMode.ts`)

The mode is stored under its own key, `'agentic_mode'` (same style as `accessibility_settings`; the constant lives in `agenticMode.ts`, the only module that reads or writes it), as a plain string. It is **not** part of `AccessibilitySettings`, so the `AppContext` effect that saves accessibility settings on every load can't save it by accident (REQ-051).

```ts
export function getSavedAgenticMode(): AgenticMode | null;   // null: "no saved mode"
export function saveAgenticMode(mode: AgenticMode): void;    // only called by the toggle
export function migrateLegacyAgenticSetting(): void;          // idempotent
```

`migrateLegacyAgenticSetting` (REQ-052):

1. Read the raw accessibility JSON from `localStorage`. If missing or unparseable, return.
2. If it has no `agentic` key, return (already migrated, or a new user).
3. If `agentic === true` and no mode is saved, save `'auto'`.
4. Delete `agentic` from the object and write it back.

Runs synchronously in `AppContext`'s load effect, before `getAccessibilitySettings()`. Since step 4 removes the key, a second run does nothing. `storage.getAccessibilitySettings()` also strips any stray `agentic` key, so old data can't reach the new type.

## 4. Runtime default: `GET /api/settings`

`src/app/api/settings/route.ts`:

```ts
export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';                 // needs fs

const ENV_FILE = path.join(process.cwd(), '.env.local');
const KEY = 'WEBAGENT_DEFAULT_AGENTIC_MODE';
let cache: { mtimeMs: number; value: string | undefined } | null = null;

export function readDefaultFromFile(): string | undefined | null { ... }  // null: no file
export async function GET() {
  const raw = readDefaultFromFile() ?? process.env[KEY];
  const value = (raw ?? 'off').trim().toLowerCase();
  if (!isAgenticMode(value)) {
    logger.warn('Unknown WEBAGENT_DEFAULT_AGENTIC_MODE, using off', { value: raw });
    return NextResponse.json({ defaultAgenticMode: 'off' });
  }
  return NextResponse.json({ defaultAgenticMode: value }, { headers: { 'Cache-Control': 'no-store' } });
}
```

`readDefaultFromFile`:

- `fs.statSync(ENV_FILE)`; if it throws `ENOENT`, return `null` (fall back to `process.env`). Other errors: log and return `null`.
- If `cache.mtimeMs === stat.mtimeMs`, return `cache.value`.
- Otherwise read the file and find the last line matching `^\s*(export\s+)?WEBAGENT_DEFAULT_AGENTIC_MODE\s*=\s*(.*)$`. Strip a trailing ` # comment`, then surrounding single or double quotes. Cache and return the value (`undefined` if the key isn't in the file).
- This tiny parser avoids adding `dotenv`; it only needs one key.

Why the file wins over `process.env`: Next.js copies `.env.local` into `process.env` once at startup, so `process.env` holds the stale value after an edit. When the file exists it is the newer source.

No API key, because the value isn't secret. `.env.local.example` documents that on Vercel or in containers without the file, changes need a redeploy or restart.

## 5. `AppContext` changes

```ts
agenticMode: AgenticMode;            // effective mode: saved ?? server default ?? 'off'
setAgenticMode: (m: AgenticMode) => void;   // updates state and saves
modeReady: boolean;                  // false until /api/settings answered or failed
```

Load sequence (one `useEffect` on mount):

1. `migrateLegacyAgenticSetting()`.
2. `const saved = getSavedAgenticMode()`. If set: `agenticMode = saved`, `modeReady = true`, and stop. (No network wait for people who chose.)
3. Else `fetch('/api/settings', { cache: 'no-store' })` with a 3 s timeout. On success with a valid value: `agenticMode = value`. On any failure: `'off'`. Then `modeReady = true`. The fetched default is **never** saved.

`setAgenticMode(m)` sets state and calls `saveAgenticMode(m)`. Because `ChatContext` reads `agenticMode` from context at send time, the next message uses the new mode (REQ-055).

While `modeReady` is false, `MessageInput` disables the send button and the toggle shows Off (REQ-054).

## 6. Proxy changes

`src/app/api/task/backend.ts`, in the `catch` branch:

```ts
return NextResponse.json(
  { error: 'Multi-agent backend unavailable' },
  { status: 503, headers: { 'X-WebAgent-Proxy-Error': 'unreachable' } },
);
```

`src/app/api/triage/route.ts`:

```ts
import { forwardToBackend } from '../task/backend';
export const dynamic = 'force-dynamic';
export async function POST(req: Request) {
  return forwardToBackend('/api/triage', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: await req.text(),
  }, req);
}
```

## 7. Routing (`src/utils/routeQuestion.ts`)

```ts
export type RouteOutcome =
  | { kind: 'route'; info: RouteInfo; fallbackNotice?: boolean }
  | { kind: 'rate_limited' };

export async function routeQuestion(args: {
  mode: AgenticMode;
  query: string;
  previousUserQuestion: string | null;
  hasImage: boolean;
  fetchImpl?: typeof fetch;     // for tests
  timeoutMs?: number;           // default 2000
}): Promise<RouteOutcome>;
```

Behavior:

- `off` → `{route: 'direct', decidedBy: 'mode', reason: 'Off mode'}`. No request.
- `always` → `{route: 'agentic', decidedBy: 'mode', reason: 'Always mode'}`. No request.
- `auto` → `POST /api/triage` with `{query: query.slice(0, 8000), previous_user_question: previousUserQuestion?.slice(0, 8000) ?? null, has_image}`, a single attempt (no retry), aborted after `timeoutMs`. The cut keeps long pasted text (e.g. "shorten this: …") from failing validation and being sent to research. An empty question with an image is sent as-is; the backend's rules answer it. Then map the outcome:

| Outcome | Result |
|---|---|
| 200, valid body | `route`, `decidedBy`, `reason: 'Auto · ' + reason` |
| 200, body not valid | agentic, `fallback`, "Auto · unexpected triage reply" |
| 503 with `X-WebAgent-Proxy-Error: unreachable` | direct, `fallback`, `fallbackNotice: true` |
| 401 | direct, `fallback`, `fallbackNotice: true`; `logger.error` |
| 429 | `{kind: 'rate_limited'}` |
| other status ≥ 500, or timeout (`AbortError`) | agentic, `fallback`, "Auto · triage unavailable" |
| other 4xx | agentic, `fallback`; `logger.warn` |
| `TypeError` from fetch (browser offline) | direct, `fallback`, `fallbackNotice: true` |

`routeQuestion` never throws.

## 8. `apiClient` changes

New signature (the boolean is replaced; the only caller is `ChatContext`):

```ts
async sendChatMessage(
  messages: ChatMessage[],
  image: string | null | undefined,
  promptStyle: string | undefined,
  knowledgeFocus: string | undefined,
  citeSources: boolean | undefined,
  mode: AgenticMode,
  options?: { forceAgentic?: boolean },      // override button
): Promise<ChatCompletionResponse>
```

Steps:

1. Split `messages` into the current question (last `user` message) and `history` (everything before it). `previousUserQuestion` is the last `user` message in `history`, or `null`.
2. If `options.forceAgentic`: `info = {route: 'agentic', decidedBy: 'override', reason: 'You asked for research'}`.
   Else `outcome = await routeQuestion(...)`. If `rate_limited`, throw the existing `ApiError` with status 429 (ChatContext already shows the rate-limit message).
3. If `info.route === 'agentic'` but the question has no text (an image only, in Always): `info = {route: 'direct', decidedBy: 'rule', reason: 'Image without a question'}`. The agents would have nothing to research and can't see the image.
4. `info.route === 'direct'`: the existing `/api/chat` call. If `fallbackNotice`, the reply starts with the line "I'm answering directly because research routing is temporarily unavailable.", so the notice shows whatever the model writes.
5. `info.route === 'agentic'`: `sendMultiAgentRequest(query, image, buildPreviousMessages(history), options.forceAgentic)`. The existing fallback to `/api/chat` stays; when it fires, set `info = {route: 'direct', decidedBy: 'fallback', reason: 'Research agents unavailable'}`.
6. Attach `info` as `response.route`.

```ts
export function buildPreviousMessages(history: ChatMessage[]): { role: 'user' | 'assistant'; content: string }[] {
  return history
    .filter(m => (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && m.content.trim())
    .slice(-8)
    .map(m => ({ role: m.role as 'user' | 'assistant', content: m.content.slice(0, 4000) }));
}
```

`sendMultiAgentRequest` body becomes `{query, image, agentic: true, previous_messages, triage_override}` (REQ-002, REQ-058). It keeps sending `agentic: true` explicitly. That matters after phase 5, where a request with neither `mode` nor `agentic` means `off`.

## 9. `ChatContext` changes

- Delete `generateMockSources` and its call (REQ-001). Sources are set only from `response.choices[0].message.sources` when the route is agentic.
- Pass `agenticMode` from `useApp()` instead of `accessibility.agentic`; add it to the `useCallback` dependencies.
- Store `route: response.route` on the assistant message, and `imageAttached: !!image` on the user message.
- Logs: keep `response.choices[0].message.logs` whenever `response.route?.route === 'agentic'` (not tied to the toggle).
- New `researchInDepth(assistantMessageId: string)`:
  1. Find that message and the closest earlier `user` message.
  2. Send `messages.slice(0, indexOfThatUserMessage + 1)` with `{forceAgentic: true}`.
  3. Insert the reply as a new assistant message directly after the direct answer (and after any earlier research answers to it), not at the end of the chat. The direct answer stays.
  4. Exposed on the context so `RouteLabel` can call it.

## 10. Components

### `AgenticModeToggle.tsx`

- A `role="radiogroup"` with `aria-label="Agentic mode"` and three `role="radio"` buttons: Off, Auto, Always. `aria-checked` on the selected one; roving `tabIndex`; Left/Right arrows move and select (REQ-050).
- `title` tooltips: "Answer directly, no research agents", "Use research agents when the question needs them", "Always use research agents (slower)".
- Props: `value`, `onChange`, `disabled`. Styled to match the current pill buttons in `MessageInput`, with the BeakerIcon and a visible "Agentic:" label before the three options. While `disabled` (mode not known yet), it shows Off as selected.

### `RouteLabel.tsx`

- Props: `info: RouteInfo`, `mode: AgenticMode`, `canOverride: boolean`, `onOverride: () => void`.
- Renders "Answered directly" or "Deep research" in small muted text, with `title={info.reason}`.
- Shows a "Research this in depth" text button when `mode === 'auto' && info.route === 'direct' && canOverride`. `canOverride` is false when the related user message has `imageAttached`, or a research request for it is already in progress.

### `MessageInput.tsx`

- Replace the Agentic button with `AgenticModeToggle` bound to `agenticMode` / `setAgenticMode`, disabled while `!modeReady`.
- Disable send while `!modeReady`.
- When `agenticMode === 'always'` and an image is attached, show "Images are only used by direct answers" under the input (REQ-003).

### `MessageList.tsx`

- Render `RouteLabel` under each assistant message that has `route`. Older messages without `route` show no label.
- `groupMessagesIntoTurns`: an assistant message that follows another assistant message (a research answer) becomes its own turn under the same question, instead of being dropped. Turns are keyed by the assistant message id.
- The logs block's condition changes from `accessibility.agentic` to `message.route?.route === 'agentic'` (REQ-057).

## 11. Tests (Jest)

| File | Covers |
|---|---|
| `src/__tests__/utils/agenticMode.test.ts` | Migration: `true` → auto, `false` → none, missing → none, runs once; save only on choice; `storage` drops a stray `agentic` key and still applies its colour migration (REQ-051, 052) |
| `src/__tests__/api/settingsRoute.test.ts` | File value, file edited (new mtime), file missing → `process.env`, bad value → off, quotes/comments (REQ-053) |
| `src/__tests__/api/triageProxy.test.ts` | Forwards with key; 503 + header when unreachable (REQ-011) |
| `src/__tests__/utils/routeQuestion.test.ts` | Off/Always make no request; every row of the table in section 7; query cut to 8,000 chars (REQ-056) |
| `src/__tests__/utils/apiClient.agentic.test.ts` | Direct → `/api/chat`; agentic → `/api/task` with `previous_messages` and `agentic: true`; override sets `triage_override`; an image without text in Always goes to `/api/chat`; route attached (REQ-002, 003, 056, 058) |
| `src/__tests__/contexts/AppContext.mode.test.tsx` | First paint Off and send disabled; default from `/api/settings` not saved; failure → Off; mode change applies to next send (REQ-054, 055) |
| `src/__tests__/contexts/ChatContext.test.tsx` (extend) | No mock sources; route stored; logs by route; `researchInDepth` puts the reply under its own question (REQ-001, 057, 058) |
| `src/components/chat/__tests__/MessageInput.mode.test.tsx` | Send disabled and Off shown until the mode is known; image note only in Always (REQ-003, 054) |
| `src/components/chat/__tests__/MessageList.test.tsx` (extend) | A second answer to the same question is shown, with both route labels (REQ-058) |
| `src/components/chat/__tests__/AgenticModeToggle.test.tsx` | Three options, keyboard, `aria-checked` (REQ-050) |
| `src/components/chat/__tests__/RouteLabel.test.tsx` | Labels, hover reason, override shown only in Auto + direct + no image (REQ-057, 058) |
