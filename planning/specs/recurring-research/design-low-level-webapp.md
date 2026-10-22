# Low-level design: web app

Paths are relative to `src/`. Phase A is §1–§5; phase B is §6.

## 1. Files

| File | New or changed | Purpose | Requirements |
|---|---|---|---|
| `app/api/task/route.ts` | changed | Forwards only allow-listed body fields | REC-107 |
| `app/api/topics/route.ts` | new | `GET` list, `POST` create | REC-102, REC-116 |
| `app/api/topics/[topicId]/route.ts` | new | `GET`, `PATCH`, `DELETE` | REC-102 |
| `app/api/topics/[topicId]/run/route.ts`, `resume/route.ts` | new | `POST` | REC-102 |
| `app/api/topics/[topicId]/runs/route.ts`, `runs/[runId]/route.ts` | new | `GET` | REC-102 |
| `app/topics/page.tsx` | new | Topics list | REC-116 |
| `app/topics/[topicId]/page.tsx` | new | Topic detail and timeline | REC-116 |
| `components/topics/TrackThisDialog.tsx` | new | The create form | REC-116 |
| `components/topics/TopicList.tsx`, `TopicTimeline.tsx`, `TopicState.tsx` | new | List rows, run timeline, state chip | REC-116 |
| `components/chat/MessageList.tsx` | changed | A **Track this** button on research answers | REC-116 |
| `types/topics.ts` | new | Response types and parsers (as `types/security.ts` does) | REC-116 |
| `app/notify/confirm/page.tsx`, `app/notify/unsubscribe/page.tsx` | new (B) | Public pages with a button | REC-134 |
| `app/api/notify/confirm/route.ts`, `app/api/notify/unsubscribe/route.ts` | new (B) | Public proxies; one-click POST | REC-134 |
| `middleware.ts` | changed (B) | Exclude the four notify routes from sign-in | REC-134 |

## 2. Task proxy allow-list (`app/api/task/route.ts`)

```ts
const TASK_FIELDS = ["query", "mode", "agentic", "previous_messages", "triage_override", "image",
                     "document_id", "consent_token", "approval_tokens"] as const;
```

- **Filtering:** the route parses the browser's JSON and forwards `JSON.stringify(pick(body, TASK_FIELDS))` instead of `await req.text()`.
- **Errors:** invalid JSON answers 400 `invalid_json` without calling the backend.
- **Size:** the route reads at most the backend's request-size limit (`WEBAGENT_SECURITY_REQUEST_SIZE_LIMIT`, 10 MB by default, since image attachments travel in this body) before parsing, and answers 413 above it.

## 3. Topic proxies (`app/api/topics/**`)

- **The helper:** every route calls `forwardToBackend(path, init, req)` from `app/api/task/backend.ts`, which adds `X-API-Key` and `X-WebAgent-Client: s:<session>`. The path is built from fixed segments and `encodeURIComponent` IDs; UUIDs are checked before forwarding (400 otherwise).
- **Only what's listed:** each route exports only its listed methods, so Next.js answers 405 for others. There are no routes for `summarize` or `notify`, so they can't be reached from the web app at all.
- **Bodies:** `POST` and `PATCH` bodies are parsed and allow-listed:
  - create: `name`, `query`, `schedule`, `notify`, `created_from_task`;
  - patch: `name`, `schedule`, `notify`, `status`, and in phase B `add_subscriber`, `remove_subscriber`.
- **No caching:** `export const dynamic = "force-dynamic"`.
- **Sign-in:** the middleware already covers `/api/*` and `/topics`. A cross-site `POST`, `PATCH` or `DELETE` carries no session cookie (`SameSite=Lax`), so it gets 401.

## 4. Pages and components

- **Track this** (`MessageList.tsx`):
  - It shows on assistant messages with `route === "agentic"` and a completed task ID.
  - It opens `TrackThisDialog`, prefilled with `result.standalone_query`, or the user's question, and `result.title` (cut to 80 characters).
  - The dialog's fields:
    - **Frequency:** daily or weekly, default weekly.
    - **Weekday:** default today's.
    - **Time:** default `07:30`. The time zone is `Intl.DateTimeFormat().resolvedOptions().timeZone`, shown read-only.
    - **Notify when:** changes or every run.
    - **Email address:** phase B, optional, with the note "We'll send a confirmation link first."
  - It posts to `/api/topics` and shows the API's errors in words:
    - `topic_limit`: "You have 10 active topics; pause or delete one first."
    - a guard refusal: "This question can't be scheduled: <reason>."
- **`/topics`:**
  - A table of name, schedule in words ("Mondays 07:30 Europe/Paris"), `TopicState` (active, paused, held: failures, held: budget), next run, last run (with "N new sources"), and actions.
  - **Actions:** pause/resume, Run now (disabled while a run is due or running), and delete, with a dialog naming the topic.
  - **Skips:** a `skipped` state is shown as "Skipped on <date>: daily limit" or "…: budget".
  - An empty state links to the chat: "Track a research answer to see it here."
- **`/topics/[topicId]`:** the header holds the name, the question, the schedule, the state and the actions. Under it, `TopicTimeline`, newest first:
  - **Each entry:** the date, the status (or `error_code` in words, as `ResearchFailed` does), the change ("First report", "N new sources", or the summary bullets in phase B), and **Open report**.
  - **Open report** renders the stored report through the same Markdown component as chat answers (`react-markdown`, no raw HTML), with its sources panel. The run record's `report` is the result's Markdown as a string, with the run's `title` beside it (requirements, run record).
- **Polling:** the list and the detail refresh every 30 s while visible (`document.visibilityState`).
- **Accessibility:** controls have labels, the dialog traps focus, and the state is text as well as colour.

## 5. Tests (Jest)

| File | Covers |
|---|---|
| `__tests__/api/taskProxy.test.ts` (extended) | Unknown and scheduler fields are dropped; invalid JSON is a 400 |
| `__tests__/api/topicsProxy.test.ts` | Each route forwards with the key; UUID checks; method allow-list; body allow-list; no summarize or notify route exists |
| `__tests__/middleware.test.ts` (extended) | `/topics` and `/api/topics` need sign-in; in phase B only the four notify routes are public |
| `components/topics/__tests__/TrackThisDialog.test.tsx` | Prefill from `standalone_query`, the time zone, error messages |
| `components/topics/__tests__/TopicList.test.tsx`, `TopicTimeline.test.tsx` | States, skips, actions, the delete confirmation, a report with `<script>` rendered as text |

## 6. Notify pages (phase B)

- **The pages:** `/notify/confirm?token=…` and `/notify/unsubscribe?token=…` are server components that render the topic name and one button. To show the name, they call the backend's read-only `GET /api/notify/lookup?token=…` **from the server**, with the API key, through `forwardToBackend`. It verifies the token and returns the name only. There's no web route for the lookup, so the public routes stay the four below.
  - The button's form POSTs to `/api/notify/confirm` or `/api/notify/unsubscribe`. The proxy first checks the token with the read-only lookup, so a token for the other page is refused before anything changes. It then answers 303 back to the page, with a relative `?receipt=…` on success or `?error=invalid` or `?error=unavailable`.
  - **The result** comes from the receipt: a short signed `{purpose, name, time}`, valid 1 h, keyed from `WEBAGENT_SESSION_SECRET` (or the API key, or a per-process random key). It's used because a used confirm token no longer verifies, and because it means a "Confirmed" page can't be crafted.
  - The pages are `force-dynamic`, with `referrer: no-referrer` and `noindex`.
  - An invalid or expired token shows "This link has expired or isn't valid", with no detail.
- **The proxies:** `/api/notify/confirm` and `/api/notify/unsubscribe` forward `{token}` to the backend without a session. They add `X-API-Key`, so the backend accepts them, and no `X-WebAgent-Client`, so all public calls share the proxy's rate-limit bucket.
  - `/api/notify/unsubscribe` also accepts the RFC 8058 one-click form (`application/x-www-form-urlencoded`, `List-Unsubscribe=One-Click`, token in the query), and answers it with JSON, never a redirect.
  - **JSON requests** (`{token}`) get JSON back. Other content types get 415, and bodies over 8 KB get 413.
  - **JSON errors** carry no backend detail: `invalid_token` (404), `rate_limited` (429), `unavailable` (503).
- **The middleware matcher** excludes exactly `notify/confirm`, `notify/unsubscribe`, `api/notify/confirm` and `api/notify/unsubscribe`, and a test pins the list.
- **CSP:** unchanged; the forms post to the same origin (`form-action 'self'`).
