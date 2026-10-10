import { forwardToBackend } from '../backend';

export const dynamic = 'force-dynamic';

export async function GET(_req: Request, { params }: { params: { taskId: string } }) {
  return forwardToBackend(`/api/task/${encodeURIComponent(params.taskId)}`, { method: 'GET' });
}
