import { NextResponse } from 'next/server';
import { isAgenticMode } from '@/types/routing';
import { logger } from '@/utils/logger';
import { DEFAULT_MODE_KEY, readDefaultFromFile } from '@/utils/defaultAgenticMode';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET() {
  const fromFile = readDefaultFromFile();
  const raw = fromFile ?? process.env[DEFAULT_MODE_KEY];
  const value = (raw ?? 'off').trim().toLowerCase();

  if (!isAgenticMode(value)) {
    logger.warn('Unknown WEBAGENT_DEFAULT_AGENTIC_MODE, using off', { value: raw });
    return NextResponse.json(
      { defaultAgenticMode: 'off' },
      { headers: { 'Cache-Control': 'no-store' } }
    );
  }

  return NextResponse.json(
    { defaultAgenticMode: value },
    { headers: { 'Cache-Control': 'no-store' } }
  );
}
