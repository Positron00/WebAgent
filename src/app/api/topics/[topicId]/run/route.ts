import { forwardToBackend } from "../../../task/backend";
import { invalidId, isUuid, topicPath } from "../../shared";

export const dynamic = "force-dynamic";

export async function POST(
  req: Request,
  { params }: { params: Promise<{ topicId: string }> },
) {
  const { topicId } = await params; // a Promise since Next.js 15
  if (!isUuid(topicId)) return invalidId();
  // No body is forwarded.
  return forwardToBackend(topicPath(topicId, "run"), { method: "POST" }, req);
}
