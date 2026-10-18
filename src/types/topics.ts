import type { Source } from "@/types/chat";

// Response types and parsers for the /api/topics/** proxies (REC-116). Parsers accept unknown
// JSON and return null (or skip an entry) when a required field is missing, as types/security.ts
// does, so a changed backend reply can't crash the pages.

export type Weekday = "mon" | "tue" | "wed" | "thu" | "fri" | "sat" | "sun";
export const WEEKDAYS: readonly Weekday[] = [
  "mon",
  "tue",
  "wed",
  "thu",
  "fri",
  "sat",
  "sun",
];
export const WEEKDAY_NAMES: Record<Weekday, string> = {
  mon: "Monday",
  tue: "Tuesday",
  wed: "Wednesday",
  thu: "Thursday",
  fri: "Friday",
  sat: "Saturday",
  sun: "Sunday",
};

export type NotifyWhen = "changes" | "always";
export type TopicState = "active" | "paused" | "held_failures" | "held_budget";
export type SkipReason = "daily_cap" | "budget";

export interface TopicSchedule {
  every: "day" | "week";
  weekday?: Weekday;
  time: string;
  tz: string;
}

export interface TopicLastRun {
  status?: string;
  at?: string;
  runId?: string;
}

/** An email subscriber (phase B); only the single-topic reply (the definition) lists them. */
export interface TopicSubscriber {
  email: string;
  confirmed: boolean;
  addedAt?: string;
  confirmedAt?: string;
}

/** The backend's cap (`max_subscribers_per_topic`). */
export const MAX_SUBSCRIBERS = 5;

export interface TopicSummary {
  topicId: string;
  name: string;
  query: string;
  schedule: TopicSchedule;
  notifyWhen: NotifyWhen;
  subscriberCount: number;
  state: TopicState;
  nextDue?: string;
  lastRun?: TopicLastRun;
  lastNewSources?: number;
  skipped?: { reason: SkipReason; at?: string };
  /** True when the backend says a run is starting or running. */
  running: boolean;
  /** Present when the reply listed `notify.subscribers` (the definition does; the list doesn't). */
  subscribers?: TopicSubscriber[];
}

export interface RunChange {
  first: boolean;
  newSources: number;
  newSourceList: Source[];
  droppedSources: Source[];
  changed?: boolean;
  summary?: string[];
}

export interface TopicRunEntry {
  runId: string;
  status: string;
  errorCode?: string;
  startedAt?: string;
  completedAt?: string;
  change?: RunChange;
}

export interface TopicRunDetail extends TopicRunEntry {
  report: { title?: string; content: string } | null;
  sources: Source[];
}

/** The create body; `created_from_task` links the topic to the chat answer it came from. */
export interface TopicCreateBody {
  name: string;
  query: string;
  schedule: TopicSchedule;
  notify: { when: NotifyWhen };
  created_from_task?: string;
}

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj =>
  !!v && typeof v === "object" && !Array.isArray(v);
const str = (v: unknown): string | undefined =>
  typeof v === "string" ? v : undefined;
const num = (v: unknown): number | undefined =>
  typeof v === "number" && Number.isFinite(v) ? v : undefined;

const TOPIC_STATES = new Set<TopicState>([
  "active",
  "paused",
  "held_failures",
  "held_budget",
]);
const isWeekday = (v: unknown): v is Weekday =>
  typeof v === "string" && (WEEKDAYS as readonly string[]).includes(v);

export function parseSchedule(data: unknown): TopicSchedule | null {
  if (!isObj(data)) return null;
  const every =
    data.every === "day" || data.every === "week" ? data.every : null;
  const time = str(data.time);
  const tz = str(data.tz);
  if (!every || !time || !tz) return null;
  return {
    every,
    ...(isWeekday(data.weekday) ? { weekday: data.weekday } : {}),
    time,
    tz,
  };
}

function parseLastRun(data: unknown): TopicLastRun | undefined {
  if (typeof data === "string") return { at: data };
  if (!isObj(data)) return undefined;
  return {
    status: str(data.status),
    at: str(data.completed_at) ?? str(data.started_at) ?? str(data.at),
    runId: str(data.run_id),
  };
}

const RUNNING_STATUSES = new Set([
  "starting",
  "running",
  "processing",
  "queued",
]);

function parseSubscriber(data: unknown): TopicSubscriber | null {
  if (!isObj(data)) return null;
  const email = str(data.email);
  if (!email) return null;
  const confirmedAt = str(data.confirmed_at) || undefined;
  return {
    email,
    confirmed: data.confirmed === true || !!confirmedAt,
    ...(str(data.added_at) ? { addedAt: str(data.added_at) } : {}),
    ...(confirmedAt ? { confirmedAt } : {}),
  };
}

export function parseSubscribers(data: unknown): TopicSubscriber[] | undefined {
  if (!Array.isArray(data)) return undefined;
  return data
    .map(parseSubscriber)
    .filter((s): s is TopicSubscriber => s !== null);
}

export function parseTopicSummary(data: unknown): TopicSummary | null {
  if (!isObj(data)) return null;
  const topicId = str(data.topic_id);
  const name = str(data.name);
  const schedule = parseSchedule(data.schedule);
  if (!topicId || name === undefined || !schedule) return null;

  const notify = isObj(data.notify) ? data.notify : {};
  const notifyWhen: NotifyWhen =
    notify.when === "always" || data["notify.when"] === "always"
      ? "always"
      : "changes";
  const state =
    typeof data.state === "string" && TOPIC_STATES.has(data.state as TopicState)
      ? (data.state as TopicState)
      : data.status === "paused"
        ? "paused"
        : "active";
  const lastRun = parseLastRun(data.last_run);
  const subscribers = parseSubscribers(notify.subscribers);
  const skipped =
    isObj(data.skipped) &&
    (data.skipped.reason === "daily_cap" || data.skipped.reason === "budget")
      ? { reason: data.skipped.reason as SkipReason, at: str(data.skipped.at) }
      : undefined;

  return {
    topicId,
    name,
    query: str(data.query) ?? "",
    schedule,
    notifyWhen,
    subscriberCount: num(data.subscriber_count) ?? subscribers?.length ?? 0,
    state,
    nextDue: str(data.next_due),
    lastRun,
    lastNewSources: num(data.last_new_sources),
    skipped,
    running:
      data.running === true ||
      (!!lastRun?.status && RUNNING_STATUSES.has(lastRun.status)),
    ...(subscribers ? { subscribers } : {}),
  };
}

export function parseTopicList(data: unknown): TopicSummary[] {
  const items = isObj(data) && Array.isArray(data.topics) ? data.topics : [];
  return items
    .map(parseTopicSummary)
    .filter((t): t is TopicSummary => t !== null);
}

function parseSource(data: unknown, index: number): Source | null {
  if (typeof data === "string")
    return { id: `s${index}`, title: data, url: data };
  if (!isObj(data)) return null;
  const url = str(data.url);
  const title = str(data.title) ?? url;
  if (!title) return null;
  let domain = str(data.domain);
  if (!domain && url) {
    try {
      domain = new URL(url).hostname;
    } catch {
      domain = undefined;
    }
  }
  return {
    id: str(data.id) ?? `s${index}`,
    title,
    ...(url ? { url } : {}),
    ...(str(data.snippet) ? { snippet: str(data.snippet) } : {}),
    ...(domain ? { domain } : {}),
  };
}

function parseSources(data: unknown): Source[] {
  return Array.isArray(data)
    ? data.map(parseSource).filter((s): s is Source => s !== null)
    : [];
}

function parseChange(data: unknown): RunChange | undefined {
  if (!isObj(data)) return undefined;
  const newList = parseSources(data.new_sources);
  const summary = Array.isArray(data.summary)
    ? data.summary.filter((s): s is string => typeof s === "string")
    : typeof data.summary === "string" && data.summary
      ? [data.summary]
      : undefined;
  return {
    first: data.first === true,
    newSources:
      num(data.new_sources) ??
      (Array.isArray(data.new_sources) ? data.new_sources.length : 0),
    newSourceList: newList,
    droppedSources: parseSources(data.dropped_sources),
    ...(typeof data.changed === "boolean" ? { changed: data.changed } : {}),
    ...(summary && summary.length ? { summary } : {}),
  };
}

export function parseRunEntry(data: unknown): TopicRunEntry | null {
  if (!isObj(data)) return null;
  const runId = str(data.run_id);
  if (!runId) return null;
  return {
    runId,
    status: str(data.status) ?? "unknown",
    errorCode: str(data.error_code) || undefined,
    startedAt: str(data.started_at),
    completedAt: str(data.completed_at),
    change: parseChange(data.change),
  };
}

export function parseRunList(data: unknown): TopicRunEntry[] {
  const items = isObj(data) && Array.isArray(data.runs) ? data.runs : [];
  return items.map(parseRunEntry).filter((r): r is TopicRunEntry => r !== null);
}

export function parseRunDetail(data: unknown): TopicRunDetail | null {
  const entry = parseRunEntry(data);
  if (!entry || !isObj(data)) return null;
  const report =
    isObj(data.report) && typeof data.report.content === "string"
      ? { title: str(data.report.title), content: data.report.content }
      : null;
  const sources = parseSources(
    data.sources ?? (isObj(data.report) ? data.report.sources : undefined),
  );
  return { ...entry, report, sources };
}

// --- Errors in words (TrackThisDialog and the list's actions) ---

export const TOPIC_LIMIT_MESSAGE =
  "You have 10 active topics; pause or delete one first.";
export const SUBSCRIBER_LIMIT_MESSAGE = `This topic already has ${MAX_SUBSCRIBERS} email addresses; remove one first.`;
export const EMAIL_DISABLED_MESSAGE =
  "Email isn't turned on for this WebAgent, so addresses can't be added.";
export const INVALID_EMAIL_MESSAGE = "That email address isn't valid.";

/** The backend's error body may be top-level or wrapped in FastAPI's `detail`. */
function errorBody(body: unknown): Obj {
  if (!isObj(body)) return {};
  return isObj(body.detail) ? { ...body, ...body.detail } : body;
}

const GUARD_CODES = new Set(["declined", "escalated", "consent_required"]);

/** A user-facing sentence for a failed topic call. */
export function topicErrorMessage(status: number, body: unknown): string {
  const b = errorBody(body);
  // The code may be `code`, `error` or a bare string `detail` (FastAPI's HTTPException).
  const code =
    str(b.code) ??
    str(b.error) ??
    (typeof b.detail === "string" && /^[a-z_]+$/.test(b.detail)
      ? b.detail
      : undefined);
  if (code === "topic_limit") return TOPIC_LIMIT_MESSAGE;
  if (code === "subscriber_limit") return SUBSCRIBER_LIMIT_MESSAGE;
  if (code === "email_disabled") return EMAIL_DISABLED_MESSAGE;
  if (code === "invalid_email") return INVALID_EMAIL_MESSAGE;
  if ((code && GUARD_CODES.has(code)) || (status === 403 && str(b.reason))) {
    const reason = (str(b.reason) || "it was blocked")
      .trim()
      .replace(/\.+$/, "");
    return `This question can't be scheduled: ${reason}.`;
  }
  if (status === 404) return "This topic no longer exists.";
  if (status === 401) return "Your session has ended; sign in again.";
  if (status === 503) return "The research service isn't reachable right now.";
  if (status === 422 || status === 400) {
    const detail = typeof b.detail === "string" ? b.detail : undefined;
    return detail
      ? `The topic wasn't accepted: ${detail}`
      : "The topic wasn't accepted.";
  }
  return `Something went wrong (HTTP ${status}).`;
}

// --- Words for schedules, dates and run outcomes ---

/** "Mondays 07:30 Europe/Paris" or "Daily 07:30 UTC". */
export function scheduleInWords(s: TopicSchedule): string {
  const when =
    s.every === "week" && s.weekday ? `${WEEKDAY_NAMES[s.weekday]}s` : "Daily";
  return `${when} ${s.time} ${s.tz}`;
}

export function formatDateTime(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function formatDate(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, { dateStyle: "medium" });
}

export function skipInWords(skipped: {
  reason: SkipReason;
  at?: string;
}): string {
  const reason = skipped.reason === "daily_cap" ? "daily limit" : "budget";
  return `Skipped on ${formatDate(skipped.at)}: ${reason}`;
}

/** A run's outcome, with error codes in words as ResearchFailed does. */
export function runStatusInWords(status: string, errorCode?: string): string {
  switch (errorCode) {
    case "timeout":
      return "Stopped: took too long";
    case "budget":
      return "Stopped: token budget used up";
    case "cancelled":
      return "Stopped";
    case "escalated":
      return "Held for review";
    case undefined:
      break;
    default:
      return `Failed: ${errorCode.replace(/_/g, " ")}`;
  }
  switch (status) {
    case "completed":
      return "Completed";
    case "failed":
    case "error":
      return "Failed";
    case "starting":
    case "running":
    case "processing":
      return "Running";
    default:
      return status.replace(/_/g, " ");
  }
}

export function changeInWords(change?: RunChange): string {
  if (!change) return "";
  if (change.first) return "First report";
  if (change.summary?.length) return change.summary.join(" · ");
  const n = change.newSources;
  return n === 1 ? "1 new source" : `${n} new sources`;
}

/** The backend's rule (one @, a domain with a dot, at most 254 characters), checked early. */
export function isPlausibleEmail(value: string): boolean {
  const v = value.trim();
  if (!v || v.length > 254 || /\s/.test(v)) return false;
  const parts = v.split("@");
  if (parts.length !== 2 || !parts[0]) return false;
  const domain = parts[1];
  return /^[^.]+(\.[^.]+)+$/.test(domain);
}
