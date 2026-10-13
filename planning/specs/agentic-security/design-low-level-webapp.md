# Low-level design: web app

Paths are relative to the repo root. Requirement IDs refer to [`requirements.md`](requirements.md).

## 1. Files

| File | New? | Change |
|---|---|---|
| `src/types/security.ts` | new | `GuardOutcome`, `GuardRefusal`, `SecuritySummary`, `ApprovalOffer` |
| `src/config/security/patterns.json` | new | Copy of `backend/config/security/patterns.json` for the fallback (kept identical by a test) |
| `src/utils/guardPatterns.ts` | new | Load `patterns.json`; `scanUserTurn(text) -> string[]` |
| `src/app/api/guard/client.ts` | new | Server-only `callGuard(messages, hasImage, consentToken, req)` |
| `src/app/api/chat/route.ts` | changed | Guard call before Together; fallback; `consentToken` in the body |
| `src/app/api/task/backend.ts` | unchanged | Already forwards 4xx bodies and marks "unreachable" |
| `src/utils/apiClient.ts` | changed | Guard refusals are final (no fallback); `consentToken` and `approvalTokens` in requests; `security` on responses |
| `src/types/api.ts`, `src/types/chat.ts` | changed | `security?: SecuritySummary` on responses and messages; `refusal?: GuardRefusal` on messages |
| `src/contexts/ChatContext.tsx` | changed | Consent state; `sendWithConsent()`; `approveAndRerun()`; refusals shown in place of an answer |
| `src/components/chat/ConsentDialog.tsx` | new | "Send anyway" / "Edit message" |
| `src/components/chat/SecuritySummary.tsx` | new | One-line summary, reasons in a `Tooltip`, "Approve and re-run" |
| `src/components/chat/MessageList.tsx` | changed | Render `SecuritySummary` and refusals |

## 2. Types (`src/types/security.ts`)

```ts
export type GuardOutcome = 'allow' | 'verify' | 'consent' | 'decline' | 'escalate';

export interface GuardRefusal {                // from a 409 or 403 body
  code: 'consent_required' | 'declined' | 'escalated';
  reason: string;
  policyId?: string;
  consentToken?: string;                       // only for consent_required
}

export interface ApprovalOffer { action: string; reason: string; approvalToken: string; }

export interface SecuritySummary {             // from the task result's `security` block
  tainted: boolean;
  shadow: boolean;
  dropped: { source: string; ref?: string; reason: string }[];
  skipped: { action: string; reason: string }[];
  approvals: ApprovalOffer[];
}

export const isGuardRefusal = (data: unknown): data is { code: GuardRefusal['code'] } => …;
```

## 3. `/api/chat` route (SEC-090, SEC-091)

`src/app/api/guard/client.ts` (server-only):

```ts
export type GuardCall =
  | { kind: 'ok'; outcome: 'allow' | 'verify' }
  | { kind: 'refused'; status: 403 | 409; body: GuardRefusal }
  | { kind: 'unreachable' };

export async function callGuard(messages, hasImage, consentToken, incoming: Request): Promise<GuardCall>
```

- `POST {WEBAGENT_BACKEND_URL}/api/guard/input` with `X-API-Key` and the caller's `X-Forwarded-For` (same rules as `forwardToBackend`), 3 s timeout.
- 200 → `ok`; 403/409 with a guard body → `refused`; network error, timeout, or 503 → `unreachable`; any other status (401, 422, 5xx) → `unreachable` with `logger.error`, because a broken guard must not take the chat down.

In `POST /api/chat`, after validating the request and before building the Together request:

1. `const guard = await callGuard(messages, !!image, body.consentToken ?? null, req)`.
2. `refused` → return `NextResponse.json(guard.body, {status: guard.status})`. Together is not called.
3. `unreachable` → `scanUserTurn(lastUserMessage)`; a hit returns `403 {code: 'declined', reason: 'This message looks like an attempt to change the assistant\'s instructions.'}`; otherwise continue, and prepend the existing "temporarily unavailable" notice line to the answer, as the triage fallback does.
4. `ok` → continue as today.

The guard runs on every `/api/chat` call, including the fallback that `apiClient` makes when an agentic task fails.

## 4. Fast patterns in Next.js (SEC-091)

- `src/config/security/patterns.json` is a copy of the backend file. The Jest test `patternsSync.test.ts` reads both files and fails if they differ, so a change to one must be copied to the other.
- `scanUserTurn(text)` applies only the `injection_phrase` and `role_marker` patterns (the ones meaningful for a user turn); secret patterns are checkpoint 2's job.

## 5. `apiClient` (SEC-092, SEC-093, SEC-094)

- **Refusals are final.** `fetchWithRetry` already throws `ApiError` with the response body in `data` and doesn't retry 4xx. `sendChatMessage` and `sendMultiAgentRequest` re-throw an `ApiError` whose `data` passes `isGuardRefusal` **before** the existing agentic → `/api/chat` fallback, so a declined task is never re-sent as a direct chat. A polled task that fails with `error_code: "escalated"` is also final.
- **Tokens in requests.** `sendChatMessage(..., options?: { forceAgentic?: boolean; consentToken?: string; approvalTokens?: string[] })`:
  - `/api/chat` body gets `consentToken`.
  - `/api/task` body gets `consent_token` and `approval_tokens` (the backend names).
- **Security on responses.** `pollTaskResult` copies `result.security` into `ChatCompletionResponse.security` (camel-cased).

## 6. `ChatContext`

- `sendMessage` catches a guard refusal:
  - `consent_required` → keep the user message visible, store `pendingConsent = {messageId, text, image, consentToken, reason}`, clear `isLoading`, show the dialog. No error banner.
  - `declined` / `escalated` → keep the user message, add an assistant-side placeholder message with `refusal` set (rendered in place of an answer, SEC-093), clear `isLoading`.
- `sendWithConsent()` resends the pending message with `{consentToken}` through the same path (mode, routing and triage unchanged), then clears `pendingConsent`.
- `editPendingMessage()` returns the text to the input (through a callback from `Chat.tsx`), removes the pending user message and clears `pendingConsent`.
- `approveAndRerun(assistantMessageId)` takes the message's `security.approvals`, re-sends the same question (with its earlier messages) to `/api/task` with `{forceAgentic: true, approvalTokens}`, and inserts the new answer below the original, like "Research this in depth".
- Messages store `security` (assistant) and `refusal` (placeholder) so the summary survives a reload.

## 7. Components

### `ConsentDialog.tsx`

- A modal (`role="dialog"`, `aria-modal="true"`, labelled by its title, focus moves to it, Escape = "Edit message").
- Title: "Send this message?"; body: the backend's `reason`; buttons: **Send anyway** (primary) and **Edit message**.
- Rendered by `Chat.tsx` when `pendingConsent` is set.

### `SecuritySummary.tsx`

- Props: `security: SecuritySummary`, `onApprove: () => void`, `busy: boolean`.
- Renders nothing when `dropped`, `skipped` and `approvals` are all empty, or when `shadow` is true.
- Otherwise one muted line: counts joined with " · " ("2 web results left out", "1 search skipped", "code not run: needs your approval"), using the existing `Tooltip` for the list of reasons (sources and URLs, never content).
- When `approvals` is non-empty: an **Approve and re-run** button, disabled while `busy`.

### `MessageList.tsx`

- Under an assistant message: `RouteLabel` (as today), then `SecuritySummary` when `message.security` exists.
- A message with `refusal` renders the reason in the answer's place, styled as a notice ("Declined: …"), with no route label or override button.

## 8. Tests (Jest)

| File | Covers |
|---|---|
| `src/__tests__/api/chatGuard.test.ts` | Allow → Together called; 409/403 bodies passed through and Together not called; unreachable → pattern hit refused, otherwise fallback notice; 401/5xx from the guard treated as unreachable (SEC-090, SEC-091) |
| `src/__tests__/security/patternsSync.test.ts` | Web copy of `patterns.json` equals the backend file (SEC-091) |
| `src/__tests__/utils/apiClient.guard.test.ts` | Guard refusal from `/api/task` is not followed by an `/api/chat` fallback; tokens sent with the right names; `security` copied from the task result |
| `src/__tests__/contexts/ChatContext.guard.test.tsx` | Consent → dialog state; Send anyway resends with the token; Edit message returns the text; declined/escalated placeholders; approve and re-run inserts below (SEC-092–SEC-094) |
| `src/components/chat/__tests__/ConsentDialog.test.tsx` | Role, focus, Escape, buttons |
| `src/components/chat/__tests__/SecuritySummary.test.tsx` | Counts, tooltip reasons, hidden in shadow mode, approve button |
