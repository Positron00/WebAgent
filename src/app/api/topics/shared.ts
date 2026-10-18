import { NextResponse } from "next/server";

// Helpers for the /api/topics/** proxies (not a route file, so Next.js exports nothing from it).

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Topic bodies are small (a name, a question of at most 2,000 characters, a schedule). */
export const TOPIC_BODY_LIMIT = 64 * 1024;

export const TOPIC_CREATE_FIELDS = [
  "name",
  "query",
  "schedule",
  "notify",
  "created_from_task",
] as const;

export const TOPIC_PATCH_FIELDS = [
  "name",
  "schedule",
  "notify",
  "status",
] as const;

export function isUuid(value: unknown): value is string {
  return typeof value === "string" && UUID_RE.test(value);
}

export function invalidId(): NextResponse {
  return NextResponse.json({ error: "invalid_id" }, { status: 400 });
}

/** `/api/topics/<id>` from fixed segments and an encoded, already-checked UUID. */
export function topicPath(topicId: string, ...segments: string[]): string {
  return ["/api/topics", encodeURIComponent(topicId), ...segments].join("/");
}
