import {
  parseRunDetail,
  parseRunList,
  parseTopicList,
  parseTopicSummary,
  topicErrorMessage,
  type TopicCreateBody,
  type TopicRunDetail,
  type TopicRunEntry,
  type TopicSchedule,
  type TopicSummary,
} from "@/types/topics";

// Browser calls to the /api/topics/** proxies. Each returns a result instead of throwing, with
// the error already in words for the page.

export type TopicResult<T> =
  { ok: true; value: T } | { ok: false; status: number; message: string };

async function call<T>(
  url: string,
  init: RequestInit,
  parse: (body: unknown) => T,
): Promise<TopicResult<T>> {
  let response: Response;
  try {
    response = await fetch(url, { cache: "no-store", ...init });
  } catch {
    return {
      ok: false,
      status: 0,
      message: "The web app isn't reachable right now.",
    };
  }
  const body: unknown =
    response.status === 204 ? null : await response.json().catch(() => null);
  if (!response.ok) {
    return {
      ok: false,
      status: response.status,
      message: topicErrorMessage(response.status, body),
    };
  }
  return { ok: true, value: parse(body) };
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const base = "/api/topics";
const one = (id: string) => `${base}/${encodeURIComponent(id)}`;

export const topicsApi = {
  list: (): Promise<TopicResult<TopicSummary[]>> =>
    call(base, { method: "GET" }, parseTopicList),
  get: (id: string): Promise<TopicResult<TopicSummary | null>> =>
    call(one(id), { method: "GET" }, parseTopicSummary),
  create: (body: TopicCreateBody): Promise<TopicResult<unknown>> =>
    call(base, json("POST", body), (b) => b),
  update: (
    id: string,
    change: Partial<{
      name: string;
      schedule: TopicSchedule;
      notify: { when: string };
      status: "active" | "paused";
    }>,
  ): Promise<TopicResult<unknown>> =>
    call(one(id), json("PATCH", change), (b) => b),
  remove: (id: string): Promise<TopicResult<null>> =>
    call(one(id), { method: "DELETE" }, () => null),
  runNow: (id: string): Promise<TopicResult<unknown>> =>
    call(`${one(id)}/run`, { method: "POST" }, (b) => b),
  resume: (id: string): Promise<TopicResult<unknown>> =>
    call(`${one(id)}/resume`, { method: "POST" }, (b) => b),
  runs: (id: string): Promise<TopicResult<TopicRunEntry[]>> =>
    call(`${one(id)}/runs`, { method: "GET" }, parseRunList),
  run: (
    id: string,
    runId: string,
  ): Promise<TopicResult<TopicRunDetail | null>> =>
    call(
      `${one(id)}/runs/${encodeURIComponent(runId)}`,
      { method: "GET" },
      parseRunDetail,
    ),
};
