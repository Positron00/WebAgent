import { routeQuestion } from '@/utils/routeQuestion';

jest.mock('@/utils/logger', () => ({
  logger: { debug: jest.fn(), info: jest.fn(), warn: jest.fn(), error: jest.fn() },
}));

const triageOk = (route: 'direct' | 'agentic', reason = 'rule match') =>
  jest.fn().mockResolvedValue({
    ok: true,
    status: 200,
    headers: new Headers(),
    json: async () => ({
      route,
      p_agentic: 0.2,
      decided_by: 'rule',
      reason,
      classifier_version: 'v1',
    }),
  });

describe('routeQuestion', () => {
  it('off mode does not call triage', async () => {
    const fetchImpl = jest.fn();
    const outcome = await routeQuestion({
      mode: 'off',
      query: 'hi',
      previousUserQuestion: null,
      hasImage: false,
      fetchImpl,
    });

    expect(fetchImpl).not.toHaveBeenCalled();
    expect(outcome).toEqual({
      kind: 'route',
      info: { route: 'direct', decidedBy: 'mode', reason: 'Off mode' },
    });
  });

  it('always mode does not call triage', async () => {
    const fetchImpl = jest.fn();
    const outcome = await routeQuestion({
      mode: 'always',
      query: 'hi',
      previousUserQuestion: null,
      hasImage: false,
      fetchImpl,
    });

    expect(fetchImpl).not.toHaveBeenCalled();
    expect(outcome.kind).toBe('route');
    if (outcome.kind === 'route') {
      expect(outcome.info.route).toBe('agentic');
    }
  });

  it('maps 200 triage response to route info', async () => {
    const outcome = await routeQuestion({
      mode: 'auto',
      query: 'research this',
      previousUserQuestion: 'earlier',
      hasImage: false,
      fetchImpl: triageOk('agentic', 'needs research'),
    });

    expect(outcome).toEqual({
      kind: 'route',
      info: {
        route: 'agentic',
        decidedBy: 'rule',
        reason: 'Auto · needs research',
        pAgentic: 0.2,
      },
    });
  });

  it('503 unreachable routes direct with fallback notice', async () => {
    const fetchImpl = jest.fn().mockResolvedValue({
      ok: false,
      status: 503,
      headers: new Headers({ 'X-WebAgent-Proxy-Error': 'unreachable' }),
      json: async () => ({}),
    });

    const outcome = await routeQuestion({
      mode: 'auto',
      query: 'q',
      previousUserQuestion: null,
      hasImage: false,
      fetchImpl,
    });

    expect(outcome.kind).toBe('route');
    if (outcome.kind === 'route') {
      expect(outcome.info.route).toBe('direct');
      expect(outcome.fallbackNotice).toBe(true);
    }
  });

  it('returns rate_limited on 429', async () => {
    const fetchImpl = jest.fn().mockResolvedValue({
      ok: false,
      status: 429,
      headers: new Headers(),
      json: async () => ({}),
    });

    const outcome = await routeQuestion({
      mode: 'auto',
      query: 'q',
      previousUserQuestion: null,
      hasImage: false,
      fetchImpl,
    });

    expect(outcome).toEqual({ kind: 'rate_limited' });
  });

  it('truncates query to 8000 characters in the request body', async () => {
    const longQuery = 'x'.repeat(9000);
    const fetchImpl = jest.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers(),
      json: async () => ({
        route: 'direct',
        p_agentic: null,
        decided_by: 'rule',
        reason: 'short',
        classifier_version: 'v1',
      }),
    });

    await routeQuestion({
      mode: 'auto',
      query: longQuery,
      previousUserQuestion: null,
      hasImage: false,
      fetchImpl,
    });

    const body = JSON.parse(fetchImpl.mock.calls[0][1].body as string);
    expect(body.query).toHaveLength(8000);
  });

  describe('fallback rows of the routing table', () => {
    const statusResponse = (status: number, json: () => Promise<unknown> = async () => ({})) =>
      jest.fn().mockResolvedValue({ ok: status >= 200 && status < 300, status, headers: new Headers(), json });

    const runAuto = (fetchImpl: unknown, timeoutMs?: number) =>
      routeQuestion({
        mode: 'auto',
        query: 'q',
        previousUserQuestion: null,
        hasImage: false,
        fetchImpl: fetchImpl as typeof fetch,
        timeoutMs,
      });

    it('200 with an invalid body routes agentic with "unexpected triage reply"', async () => {
      const outcome = await runAuto(statusResponse(200, async () => ({ route: 'maybe' })));

      expect(outcome).toEqual({
        kind: 'route',
        info: { route: 'agentic', decidedBy: 'fallback', reason: 'Auto · unexpected triage reply' },
      });
    });

    it('200 with a body that is not JSON routes agentic with "unexpected triage reply"', async () => {
      const outcome = await runAuto(
        statusResponse(200, async () => {
          throw new SyntaxError('Unexpected token');
        })
      );

      expect(outcome.kind === 'route' && outcome.info.reason).toBe('Auto · unexpected triage reply');
    });

    it('401 routes direct with a fallback notice and logs an error', async () => {
      const { logger } = jest.requireMock('@/utils/logger');
      logger.error.mockClear();

      const outcome = await runAuto(statusResponse(401));

      expect(outcome.kind).toBe('route');
      if (outcome.kind === 'route') {
        expect(outcome.info.route).toBe('direct');
        expect(outcome.info.decidedBy).toBe('fallback');
        expect(outcome.fallbackNotice).toBe(true);
      }
      expect(logger.error).toHaveBeenCalled();
    });

    it('500 routes agentic with "triage unavailable"', async () => {
      const outcome = await runAuto(statusResponse(500));

      expect(outcome).toEqual({
        kind: 'route',
        info: { route: 'agentic', decidedBy: 'fallback', reason: 'Auto · triage unavailable' },
      });
    });

    it('503 without the unreachable header routes agentic', async () => {
      const outcome = await runAuto(statusResponse(503));

      expect(outcome.kind === 'route' && outcome.info.route).toBe('agentic');
      expect(outcome.kind === 'route' && outcome.fallbackNotice).toBeFalsy();
    });

    it('an AbortError from fetch routes agentic with "triage unavailable"', async () => {
      const abortError = new Error('The operation was aborted');
      abortError.name = 'AbortError';

      const outcome = await runAuto(jest.fn().mockRejectedValue(abortError));

      expect(outcome).toEqual({
        kind: 'route',
        info: { route: 'agentic', decidedBy: 'fallback', reason: 'Auto · triage unavailable' },
      });
    });

    it('a real timeout aborts the request and routes agentic', async () => {
      const neverResolves = jest.fn(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            init.signal?.addEventListener('abort', () => {
              const err = new Error('aborted');
              err.name = 'AbortError';
              reject(err);
            });
          })
      );

      const outcome = await runAuto(neverResolves, 10);

      expect(outcome).toEqual({
        kind: 'route',
        info: { route: 'agentic', decidedBy: 'fallback', reason: 'Auto · triage unavailable' },
      });
    });

    it('422 routes agentic with a fallback and logs a warning', async () => {
      const { logger } = jest.requireMock('@/utils/logger');
      logger.warn.mockClear();

      const outcome = await runAuto(statusResponse(422));

      expect(outcome.kind).toBe('route');
      if (outcome.kind === 'route') {
        expect(outcome.info.route).toBe('agentic');
        expect(outcome.info.decidedBy).toBe('fallback');
        expect(outcome.fallbackNotice).toBeUndefined();
      }
      expect(logger.warn).toHaveBeenCalled();
    });

    it('a TypeError from fetch (offline) routes direct with a fallback notice', async () => {
      const outcome = await runAuto(jest.fn().mockRejectedValue(new TypeError('Failed to fetch')));

      expect(outcome.kind).toBe('route');
      if (outcome.kind === 'route') {
        expect(outcome.info.route).toBe('direct');
        expect(outcome.info.decidedBy).toBe('fallback');
        expect(outcome.fallbackNotice).toBe(true);
      }
    });
  });
});
