import { ApiClient } from '@/utils/apiClient';

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

const jsonResponse = (body: unknown, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  statusText: status === 200 ? 'OK' : 'Error',
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
    // Skip real waiting between polls and retries
    (client as any).delay = jest.fn().mockResolvedValue(undefined);
  });

  it('sends only the latest question to /api/task and returns the polled report', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ task_id: 'task-1', status: 'processing' }))
      .mockResolvedValueOnce(jsonResponse({ status: 'processing' }))
      .mockResolvedValueOnce(
        jsonResponse({ status: 'completed', result: { content: '# Report', logs: ['Supervisor: planning'] } })
      );

    const response = await client.sendChatMessage(conversation, null, 'balanced', 'general', true, true);

    expect(requestTo(0)).toEqual({
      url: '/api/task',
      method: 'POST',
      body: { query: 'second question', agentic: true },
    });
    expect(requestTo(1).url).toBe('/api/task/task-1');
    expect(requestTo(2).url).toBe('/api/task/task-1');
    expect(response.choices[0].message.content).toBe('# Report');
    expect(response.choices[0].message.logs).toEqual(['Supervisor: planning']);
  });

  it('falls back to plain chat, marked as a fallback, when the task fails', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ task_id: 'task-2', status: 'processing' }))
      .mockResolvedValueOnce(jsonResponse({ status: 'failed', error: 'Together Server: Error 503' }))
      .mockResolvedValueOnce(chatReply('fallback answer'));

    const response = await client.sendChatMessage(conversation, null, 'balanced', 'general', true, true);

    const fallback = requestTo(2);
    expect(fallback.url).toBe('/api/chat');
    expect(fallback.body.requestType).toBe('multi-agent-fallback');
    expect(fallback.body.messages[fallback.body.messages.length - 1]).toEqual({
      role: 'user',
      content: 'second question',
    });
    expect(response.choices[0].message.content).toBe('fallback answer');
  });

  it('calls only /api/chat with the whole conversation when agentic mode is off', async () => {
    fetchMock.mockResolvedValueOnce(chatReply('direct answer'));

    const response = await client.sendChatMessage(conversation, null, 'balanced', 'general', true, false);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const request = requestTo(0);
    expect(request.url).toBe('/api/chat');
    expect(request.body.agentic).toBe(false);
    expect(request.body.messages).toHaveLength(3);
    expect(response.choices[0].message.content).toBe('direct answer');
  });

  it('rejects an agentic request with no user message', async () => {
    const onlyAssistant = [{ id: '1', role: 'assistant' as const, content: 'hi', timestamp: '' }];
    await expect(
      client.sendChatMessage(onlyAssistant, null, 'balanced', 'general', true, true)
    ).rejects.toThrow('No user message');
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
