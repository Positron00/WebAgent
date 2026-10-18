import { forwardToBackend } from "../task/backend";
import { pick, readJsonObject } from "../task/body";
import { TOPIC_BODY_LIMIT, TOPIC_CREATE_FIELDS } from "./shared";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  return forwardToBackend("/api/topics", { method: "GET" }, req);
}

export async function POST(req: Request) {
  const parsed = await readJsonObject(req, TOPIC_BODY_LIMIT);
  if (!parsed.ok) return parsed.response;
  return forwardToBackend(
    "/api/topics",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pick(parsed.body, TOPIC_CREATE_FIELDS)),
    },
    req,
  );
}
