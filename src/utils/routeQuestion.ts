import { AgenticMode, RouteInfo, TriageResponse } from '@/types/routing';
import { logger } from '@/utils/logger';

export type RouteOutcome =
  | { kind: 'route'; info: RouteInfo; fallbackNotice?: boolean }
  | { kind: 'rate_limited' };

const TRIAGE_URL = '/api/triage';

function isValidTriageResponse(data: unknown): data is TriageResponse {
  if (!data || typeof data !== 'object') return false;
  const d = data as TriageResponse;
  return (
    (d.route === 'direct' || d.route === 'agentic') &&
    typeof d.reason === 'string' &&
    (d.decided_by === 'rule' ||
      d.decided_by === 'model' ||
      d.decided_by === 'fallback' ||
      d.decided_by === 'config')
  );
}

export async function routeQuestion(args: {
  mode: AgenticMode;
  query: string;
  previousUserQuestion: string | null;
  hasImage: boolean;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}): Promise<RouteOutcome> {
  const { mode, query, previousUserQuestion, hasImage, fetchImpl = fetch, timeoutMs = 2000 } = args;

  if (mode === 'off') {
    return {
      kind: 'route',
      info: { route: 'direct', decidedBy: 'mode', reason: 'Off mode' },
    };
  }

  if (mode === 'always') {
    return {
      kind: 'route',
      info: { route: 'agentic', decidedBy: 'mode', reason: 'Always mode' },
    };
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetchImpl(TRIAGE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query.slice(0, 8000),
        previous_user_question: previousUserQuestion?.slice(0, 8000) ?? null,
        has_image: hasImage,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (response.status === 429) {
      return { kind: 'rate_limited' };
    }

    const proxyError = response.headers.get('X-WebAgent-Proxy-Error');

    if (response.status === 503 && proxyError === 'unreachable') {
      return {
        kind: 'route',
        info: { route: 'direct', decidedBy: 'fallback', reason: 'Auto · triage unreachable' },
        fallbackNotice: true,
      };
    }

    if (response.status === 401) {
      logger.error('Triage returned 401', { status: 401 });
      return {
        kind: 'route',
        info: { route: 'direct', decidedBy: 'fallback', reason: 'Auto · triage unauthorized' },
        fallbackNotice: true,
      };
    }

    if (response.ok) {
      let body: unknown;
      try {
        body = await response.json();
      } catch {
        body = null;
      }

      if (!isValidTriageResponse(body)) {
        return {
          kind: 'route',
          info: {
            route: 'agentic',
            decidedBy: 'fallback',
            reason: 'Auto · unexpected triage reply',
          },
        };
      }

      return {
        kind: 'route',
        info: {
          route: body.route,
          decidedBy: body.decided_by,
          reason: `Auto · ${body.reason}`,
          pAgentic: body.p_agentic,
        },
      };
    }

    if (response.status >= 500) {
      return {
        kind: 'route',
        info: {
          route: 'agentic',
          decidedBy: 'fallback',
          reason: 'Auto · triage unavailable',
        },
      };
    }

    logger.warn('Triage returned client error', { status: response.status });
    return {
      kind: 'route',
      info: {
        route: 'agentic',
        decidedBy: 'fallback',
        reason: 'Auto · triage rejected request',
      },
    };
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof Error && error.name === 'AbortError') {
      return {
        kind: 'route',
        info: {
          route: 'agentic',
          decidedBy: 'fallback',
          reason: 'Auto · triage unavailable',
        },
      };
    }

    if (error instanceof TypeError) {
      return {
        kind: 'route',
        info: { route: 'direct', decidedBy: 'fallback', reason: 'Auto · triage offline' },
        fallbackNotice: true,
      };
    }

    return {
      kind: 'route',
      info: {
        route: 'agentic',
        decidedBy: 'fallback',
        reason: 'Auto · triage unavailable',
      },
    };
  }
}
