import { forwardToBackend } from '../task/backend';

export const dynamic = 'force-dynamic';

export async function POST(req: Request) {
  return forwardToBackend(
    '/api/triage',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: await req.text(),
    },
    req
  );
}
