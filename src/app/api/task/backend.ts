import { NextResponse } from 'next/server';
import { logger } from '@/utils/logger';

// Forwards /api/task requests to the FastAPI backend so the API key never reaches the browser.
const BACKEND_URL = process.env.WEBAGENT_BACKEND_URL || 'http://127.0.0.1:8000';

export async function forwardToBackend(path: string, init: RequestInit): Promise<NextResponse> {
  const headers = new Headers(init.headers);
  if (process.env.WEBAGENT_API_KEY) {
    headers.set('X-API-Key', process.env.WEBAGENT_API_KEY);
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
