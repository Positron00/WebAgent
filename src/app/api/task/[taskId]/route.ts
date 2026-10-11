import { forwardToBackend } from '../backend';

export const dynamic = 'force-dynamic';

export async function GET(req: Request, { params }: { params: { taskId: string } }) {
  return forwardToBackend(`/api/task/${encodeURIComponent(params.taskId)}`, { method: 'GET' }, req);
}
