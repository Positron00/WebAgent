import { forwardToBackend } from "../backend";

export const dynamic = "force-dynamic";

export async function GET(
  req: Request,
  { params }: { params: Promise<{ taskId: string }> },
) {
  const { taskId } = await params; // a Promise since Next.js 15
  return forwardToBackend(
    `/api/task/${encodeURIComponent(taskId)}`,
    { method: "GET" },
    req,
  );
}
