import { NextResponse } from 'next/server';
import { logger } from '@/utils/logger';

// Forwards /api/task requests to the FastAPI backend so the API key never reaches the browser.
const BACKEND_URL = process.env.WEBAGENT_BACKEND_URL || 'http://127.0.0.1:8000';

// The caller's address, so the backend rate-limits each user instead of this proxy.
function callerAddress(incoming?: Request): string | null {
  if (!incoming) return null;
  const forwarded = incoming.headers.get('x-forwarded-for');
  if (forwarded) return forwarded;
  return incoming.headers.get('x-real-ip');
}

export async function forwardToBackend(
  path: string,
  init: RequestInit,
  incoming?: Request
): Promise<NextResponse> {
  const headers = new Headers(init.headers);
  if (process.env.WEBAGENT_API_KEY) {
    headers.set('X-API-Key', process.env.WEBAGENT_API_KEY);
  }
  const caller = callerAddress(incoming);
  if (caller) {
    headers.set('X-Forwarded-For', caller);
  }

  try {
    const response = await fetch(`${BACKEND_URL}${path}`, { ...init, headers, cache: 'no-store' });
    const body = await response.json().catch(() => ({}));
    return NextResponse.json(body, { status: response.status });
  } catch (error) {
    logger.error('Multi-agent backend unreachable', {
      path,
      error: error instanceof Error ? error.message : String(error),
    });
    return NextResponse.json({ error: 'Multi-agent backend unavailable' }, { status: 503 });
  }
}
