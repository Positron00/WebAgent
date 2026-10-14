import { forwardToBackend } from "../../backend";

export const dynamic = "force-dynamic";

export async function POST(
  req: Request,
  { params }: { params: { taskId: string } },
) {
  return forwardToBackend(
    `/api/task/${encodeURIComponent(params.taskId)}/cancel`,
    { method: "POST" },
    req,
  );
}
