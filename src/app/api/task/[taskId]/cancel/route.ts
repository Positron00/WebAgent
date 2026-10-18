import { forwardToBackend } from "../../backend";

export const dynamic = "force-dynamic";

export async function POST(
  req: Request,
  { params }: { params: Promise<{ taskId: string }> },
) {
  const { taskId } = await params; // a Promise since Next.js 15
  return forwardToBackend(
    `/api/task/${encodeURIComponent(taskId)}/cancel`,
    { method: "POST" },
    req,
  );
}
