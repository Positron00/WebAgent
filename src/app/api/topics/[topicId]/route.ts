import { forwardToBackend } from "../../task/backend";
import { pick, readJsonObject } from "../../task/body";
import {
  invalidId,
  isUuid,
  TOPIC_BODY_LIMIT,
  TOPIC_PATCH_FIELDS,
  topicPath,
} from "../shared";

export const dynamic = "force-dynamic";

type Ctx = { params: Promise<{ topicId: string }> };

export async function GET(req: Request, { params }: Ctx) {
  const { topicId } = await params; // a Promise since Next.js 15
  if (!isUuid(topicId)) return invalidId();
  return forwardToBackend(topicPath(topicId), { method: "GET" }, req);
}

export async function PATCH(req: Request, { params }: Ctx) {
  const { topicId } = await params;
  if (!isUuid(topicId)) return invalidId();
  const parsed = await readJsonObject(req, TOPIC_BODY_LIMIT);
  if (!parsed.ok) return parsed.response;
  return forwardToBackend(
    topicPath(topicId),
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pick(parsed.body, TOPIC_PATCH_FIELDS)),
    },
    req,
  );
}

export async function DELETE(req: Request, { params }: Ctx) {
  const { topicId } = await params;
  if (!isUuid(topicId)) return invalidId();
  return forwardToBackend(topicPath(topicId), { method: "DELETE" }, req);
}
