/**
 * @jest-environment node
 */
jest.mock('@/utils/logger', () => ({
  logger: { debug: jest.fn(), info: jest.fn(), warn: jest.fn(), error: jest.fn() },
}));

export {};

const fetchMock = jest.fn();
let postChat: (req: Request) => Promise<Response>;

beforeAll(async () => {
  // config/env.ts validates the key when the module loads
  process.env.TOGETHER_API_KEY = 'test-key-0123456789abcdef0123456789abcdef';
  delete process.env.WEBAGENT_LLM_DEFAULT_MODEL;
  delete process.env.TOGETHER_MODEL;
  ({ POST: postChat } = await import('@/app/api/chat/route'));
});

beforeEach(() => {
  fetchMock.mockReset();
  global.fetch = fetchMock as unknown as typeof fetch;
  jest.spyOn(console, 'error').mockImplementation(() => undefined);
});

afterEach(() => jest.restoreAllMocks());

const chatRequest = (body: unknown) =>
  new Request('http://localhost/api/chat', { method: 'POST', body: JSON.stringify(body) });

const userMessage = { messages: [{ role: 'user', content: 'Hello' }] };

describe('/api/chat', () => {
  it('rejects messages in the wrong format', async () => {
    const response = await postChat(chatRequest({ messages: [{ text: 'no role' }] }));

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: { message: 'Invalid message format' } });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('sends a system prompt plus the conversation to Together with the server key', async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'Hi!' }, finish_reason: 'stop' }] }),
        { status: 200 }
      )
    );

    const response = await postChat(chatRequest(userMessage));

    const [url, init] = fetchMock.mock.calls[0];
    const sent = JSON.parse(init.body);
    expect(url).toBe('https://api.together.xyz/v1/chat/completions');
    expect(init.headers.Authorization).toBe(`Bearer ${process.env.TOGETHER_API_KEY}`);
    expect(sent.model).toBe('meta-models/Muse-Glimmer-30B');
    expect(sent.messages[0].role).toBe('system');
    expect(sent.messages[1]).toEqual({ role: 'user', content: 'Hello' });
    expect(response.status).toBe(200);
    expect((await response.json()).choices[0].message.content).toBe('Hi!');
  });

  it("passes Together's error status and message through", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ error: { message: 'Unable to access non-serverless model' } }), {
        status: 400,
      })
    );

    const response = await postChat(chatRequest(userMessage));

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: { message: 'Unable to access non-serverless model' } });
  });

  it('returns 503 when Together cannot be reached', async () => {
    fetchMock.mockRejectedValue(new TypeError('fetch failed'));

    const response = await postChat(chatRequest(userMessage));

    expect(response.status).toBe(503);
    expect((await response.json()).error.message).toContain('Unable to reach LLM API');
  });

  it('adds a notice when answering as a fallback for the multi-agent system', async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'Answer.' }, finish_reason: 'stop' }] }),
        { status: 200 }
      )
    );

    const response = await postChat(chatRequest({ ...userMessage, requestType: 'multi-agent-fallback' }));

    expect((await response.json()).choices[0].message.content).toMatch(/^I'm currently operating in fallback mode/);
  });
});
