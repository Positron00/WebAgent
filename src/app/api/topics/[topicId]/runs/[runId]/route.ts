import { forwardToBackend } from "../../../../task/backend";
import { invalidId, isUuid, topicPath } from "../../../shared";

export const dynamic = "force-dynamic";

export async function GET(
  req: Request,
  { params }: { params: Promise<{ topicId: string; runId: string }> },
) {
  const { topicId, runId } = await params; // a Promise since Next.js 15
  if (!isUuid(topicId) || !isUuid(runId)) return invalidId();
  return forwardToBackend(
    topicPath(topicId, "runs", encodeURIComponent(runId)),
    { method: "GET" },
    req,
  );
}
