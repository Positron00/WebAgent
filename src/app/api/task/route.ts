import { forwardToBackend } from "./backend";
import { pick, readJsonObject } from "./body";

export const dynamic = "force-dynamic";

// Only these fields reach the backend; scheduler fields (origin, topic_id, task_id, token_budget)
// and anything else a browser adds are dropped (REC-107).
const TASK_FIELDS = [
  "query",
  "mode",
  "agentic",
  "previous_messages",
  "triage_override",
  "image",
  "document_id",
  "consent_token",
  "approval_tokens",
] as const;

export async function POST(req: Request) {
  const parsed = await readJsonObject(req);
  if (!parsed.ok) return parsed.response;
  return forwardToBackend(
    "/api/task",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pick(parsed.body, TASK_FIELDS)),
    },
    req,
  );
}
