import { ApiClient } from '@/utils/apiClient';
import { ApiError } from '@/utils/apiClient';

jest.mock('@/config/clientConfig', () => ({
  clientConfig: {
    environment: 'test',
    version: 'test',
    api: {
      baseUrl: '',
      endpoints: { chat: '/api/chat', task: '/api/task', ping: '/api/ping', health: '/api/health' },
      timeoutMs: 60000,
    },
    together: { endpoint: '/api/chat', model: 'test-model' },
    features: { enableAgenticMode: true },
  },
}));

jest.mock('@/utils/logger', () => ({
  logger: { debug: jest.fn(), info: jest.fn(), warn: jest.fn(), error: jest.fn() },
}));

jest.mock('@/utils/routeQuestion', () => ({
  routeQuestion: jest.fn().mockResolvedValue({
    kind: 'route',
    info: { route: 'agentic', decidedBy: 'override', reason: 'test' },
  }),
}));

const jsonResponse = (body: unknown, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  statusText: status === 200 ? 'OK' : 'Error',
  headers: new Headers(),
  json: jest.fn().mockResolvedValue(body),
});

const fetchMock = jest.fn();

describe('ApiClient guard behavior', () => {
  let client: ApiClient;

  beforeEach(() => {
    fetchMock.mockReset();
    global.fetch = fetchMock as unknown as typeof fetch;
    client = ApiClient.getInstance();
    client.resetMetrics();
    (client as any).delay = jest.fn().mockResolvedValue(undefined);
  });

  it('does not fall back to /api/chat when /api/task returns a guard refusal', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ code: 'declined', reason: 'No' }, 403)
    );

    const messages = [{ id: '1', role: 'user' as const, content: 'hack', timestamp: '' }];

    await expect(
      client.sendChatMessage(messages, null, 'balanced', 'general', true, 'always')
    ).rejects.toBeInstanceOf(ApiError);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/task');
  });

  it('sends consent and approval tokens with backend field names', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ task_id: 't1' }));
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ status: 'completed', result: { content: 'done' } })
    );

    await client.sendChatMessage(
      [{ id: '1', role: 'user', content: 'q', timestamp: '' }],
      null,
      'balanced',
      'general',
      true,
      'always',
      { consentToken: 'consent-abc', approvalTokens: ['appr-1', 'appr-2'] }
    );

    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.consent_token).toBe('consent-abc');
    expect(body.approval_tokens).toEqual(['appr-1', 'appr-2']);
  });

  it('copies security from a completed task result', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ task_id: 't2' }));
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        status: 'completed',
        result: { content: 'answer' },
        security: {
          tainted: true,
          shadow: false,
          dropped: [{ source: 'web_result', ref: 'https://x.test', reason: 'injection' }],
          skipped: [],
          approvals: [],
        },
      })
    );

    const response = await client.sendChatMessage(
      [{ id: '1', role: 'user', content: 'q', timestamp: '' }],
      null,
      'balanced',
      'general',
      true,
      'always'
    );

    expect(response.security).toEqual({
      tainted: true,
      shadow: false,
      dropped: [{ source: 'web_result', ref: 'https://x.test', reason: 'injection' }],
      skipped: [],
      approvals: [],
    });
  });

  it('treats escalated task failure as final', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ task_id: 't3' }));
    fetchMock.mockResolvedValue(
      jsonResponse({ status: 'failed', error: 'Stopped', error_code: 'escalated' })
    );

    await expect(
      client.sendChatMessage(
        [{ id: '1', role: 'user', content: 'q', timestamp: '' }],
        null,
        'balanced',
        'general',
        true,
        'always'
      )
    ).rejects.toMatchObject({
      status: 403,
      data: { code: 'escalated' },
    });

    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
