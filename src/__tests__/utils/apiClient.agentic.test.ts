import { ApiClient, buildPreviousMessages } from '@/utils/apiClient';

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

const jsonResponse = (body: unknown, status = 200, headers?: Record<string, string>) => ({
  ok: status >= 200 && status < 300,
  status,
  statusText: status === 200 ? 'OK' : 'Error',
  headers: new Headers(headers),
  json: jest.fn().mockResolvedValue(body),
});

const chatReply = (content: string) =>
  jsonResponse({ choices: [{ message: { role: 'assistant', content }, finish_reason: 'stop' }] });

const conversation = [
  { id: '1', role: 'user' as const, content: 'first question', timestamp: '' },
  { id: '2', role: 'assistant' as const, content: 'first answer', timestamp: '' },
  { id: '3', role: 'user' as const, content: 'second question', timestamp: '' },
];

const fetchMock = jest.fn();

function requestTo(index: number) {
  const [url, init] = fetchMock.mock.calls[index];
  return { url, method: init?.method, body: init?.body ? JSON.parse(init.body) : undefined };
}

describe('ApiClient agentic mode', () => {
  let client: ApiClient;

  beforeEach(() => {
    fetchMock.mockReset();
    global.fetch = fetchMock as unknown as typeof fetch;
    client = ApiClient.getInstance();
    client.resetMetrics();
    (client as any).delay = jest.fn().mockResolvedValue(undefined);
  });

  it('sends only the latest question to /api/task with previous_messages in always mode', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ task_id: 'task-1', status: 'processing' }))
      .mockResolvedValueOnce(jsonResponse({ status: 'processing' }))
      .mockResolvedValueOnce(
        jsonResponse({ status: 'completed', result: { content: '# Report', logs: ['Supervisor: planning'] } })
      );

    const response = await client.sendChatMessage(
      conversation,
      null,
      'balanced',
      'general',
      true,
      'always'
    );

    expect(requestTo(0)).toEqual({
      url: '/api/task',
      method: 'POST',
      body: {
        query: 'second question',
        agentic: true,
        previous_messages: [
          { role: 'user', content: 'first question' },
          { role: 'assistant', content: 'first answer' },
        ],
      },
    });
    expect(requestTo(1).url).toBe('/api/task/task-1');
    expect(response.choices[0].message.content).toBe('# Report');
    expect(response.route?.route).toBe('agentic');
  });

  it('includes triage_override when forceAgentic is set', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ task_id: 'task-ov', status: 'processing' }));
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ status: 'completed', result: { content: 'deep' } })
    );

    await client.sendChatMessage(conversation, null, 'balanced', 'general', true, 'auto', {
      forceAgentic: true,
    });

    expect(requestTo(0).body.triage_override).toBe(true);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/task');
  });

  it('falls back to plain chat when the task fails', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ task_id: 'task-2', status: 'processing' }))
      .mockResolvedValueOnce(jsonResponse({ status: 'failed', error: 'Together Server: Error 503' }))
      .mockResolvedValueOnce(chatReply('fallback answer'));

    const response = await client.sendChatMessage(
      conversation,
      null,
      'balanced',
      'general',
      true,
      'always'
    );

    const fallback = requestTo(2);
    expect(fallback.url).toBe('/api/chat');
    expect(fallback.body.requestType).toBe('multi-agent-fallback');
    expect(response.choices[0].message.content).toBe('fallback answer');
    expect(response.route).toEqual({
      route: 'direct',
      decidedBy: 'fallback',
      reason: 'Research agents unavailable',
    });
  });

  it('calls only /api/chat when mode is off', async () => {
    fetchMock.mockResolvedValueOnce(chatReply('direct answer'));

    const response = await client.sendChatMessage(
      conversation,
      null,
      'balanced',
      'general',
      true,
      'off'
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const request = requestTo(0);
    expect(request.url).toBe('/api/chat');
    expect(request.body.agentic).toBe(false);
    expect(response.route?.route).toBe('direct');
  });

  it('always mode sends an image without text to /api/chat, not the research agents', async () => {
    fetchMock.mockResolvedValueOnce(chatReply('I see a cat'));

    const response = await client.sendChatMessage(
      [{ id: '1', role: 'user' as const, content: '', timestamp: '' }],
      'data:image/png;base64,AAAA',
      'balanced',
      'general',
      true,
      'always'
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(requestTo(0).url).toBe('/api/chat');
    expect(requestTo(0).body.image).toBe('data:image/png;base64,AAAA');
    expect(response.route).toEqual({
      route: 'direct',
      decidedBy: 'rule',
      reason: 'Image without a question',
    });
  });

  it('auto mode calls triage then routes to chat when triage picks direct', async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({
          route: 'direct',
          p_agentic: 0.1,
          decided_by: 'rule',
          reason: 'simple',
          classifier_version: 'v1',
        })
      )
      .mockResolvedValueOnce(chatReply('direct from auto'));

    const response = await client.sendChatMessage(
      conversation,
      null,
      'balanced',
      'general',
      true,
      'auto'
    );

    expect(requestTo(0).url).toBe('/api/triage');
    expect(requestTo(1).url).toBe('/api/chat');
    expect(response.route?.reason).toMatch(/^Auto ·/);
  });

  it('rejects a request with no user message and no image', async () => {
    const onlyAssistant = [{ id: '1', role: 'assistant' as const, content: 'hi', timestamp: '' }];
    await expect(
      client.sendChatMessage(onlyAssistant, null, 'balanced', 'general', true, 'always')
    ).rejects.toThrow('No user message');
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe('buildPreviousMessages', () => {
  it('keeps up to eight trimmed user/assistant turns', () => {
    const history = Array.from({ length: 10 }, (_, i) => ({
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: `msg ${i}`,
    })) as { role: 'user' | 'assistant'; content: string }[];

    const built = buildPreviousMessages(history as any);
    expect(built).toHaveLength(8);
    expect(built[0].content).toBe('msg 2');
  });
});
