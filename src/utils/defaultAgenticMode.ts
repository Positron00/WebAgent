import fs from 'fs';
import path from 'path';
import { logger } from '@/utils/logger';

// Server-only: reads WEBAGENT_DEFAULT_AGENTIC_MODE from .env.local for GET /api/settings.
// Lives outside the route file because Next.js route files may export only handlers and config.

export const DEFAULT_MODE_KEY = 'WEBAGENT_DEFAULT_AGENTIC_MODE';

function envFilePath(): string {
  return path.join(process.cwd(), '.env.local');
}

let cache: { mtimeMs: number; value: string | undefined } | null = null;

const LINE_RE = /^\s*(?:export\s+)?WEBAGENT_DEFAULT_AGENTIC_MODE\s*=\s*(.*)$/;

function parseEnvValue(raw: string): string {
  let value = raw.trim();
  const commentIndex = value.indexOf(' #');
  if (commentIndex !== -1) {
    value = value.slice(0, commentIndex).trim();
  }
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    value = value.slice(1, -1);
  }
  return value.trim();
}

export function readDefaultFromFile(): string | undefined | null {
  try {
    const envFile = envFilePath();
    const stat = fs.statSync(envFile);
    if (cache && cache.mtimeMs === stat.mtimeMs) {
      return cache.value;
    }

    const content = fs.readFileSync(envFile, 'utf8');
    const lines = content.split('\n');
    let found: string | undefined;

    for (const line of lines) {
      const match = LINE_RE.exec(line);
      if (match) {
        found = parseEnvValue(match[1]);
      }
    }

    cache = { mtimeMs: stat.mtimeMs, value: found };
    return found;
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && (error as NodeJS.ErrnoException).code === 'ENOENT') {
      return null;
    }
    logger.warn('Failed to read .env.local for agentic default', {
      error: error instanceof Error ? error.message : String(error),
    });
    return null;
  }
}

export function clearSettingsCache(): void {
  cache = null;
}
