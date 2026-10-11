/**
 * @jest-environment node
 */
jest.mock('@/utils/logger', () => ({
  logger: { debug: jest.fn(), info: jest.fn(), warn: jest.fn(), error: jest.fn() },
}));

export {};

const fetchMock = jest.fn();
let postTask: (req: Request) => Promise<Response>;
let getTask: (req: Request, ctx: { params: { taskId: string } }) => Promise<Response>;

beforeAll(async () => {
  // The proxy reads these when the module loads
  process.env.WEBAGENT_BACKEND_URL = 'http://backend.test:9000';
  process.env.WEBAGENT_API_KEY = 'secret-key';
  ({ POST: postTask } = await import('@/app/api/task/route'));
  ({ GET: getTask } = await import('@/app/api/task/[taskId]/route'));
});

beforeEach(() => {
  fetchMock.mockReset();
  global.fetch = fetchMock as unknown as typeof fetch;
});

function forwarded(index = 0) {
  const [url, init] = fetchMock.mock.calls[index];
  return { url, init, headers: new Headers(init.headers) };
}

describe('/api/task proxy', () => {
  it('forwards a new task to the backend with the server-side API key', async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ task_id: 't1', status: 'processing' }), { status: 200 })
    );

    const response = await postTask(
      new Request('http://localhost/api/task', {
        method: 'POST',
        body: JSON.stringify({ query: 'q', agentic: true }),
      })
    );

    const { url, init, headers } = forwarded();
    expect(url).toBe('http://backend.test:9000/api/task');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ query: 'q', agentic: true });
    expect(headers.get('X-API-Key')).toBe('secret-key');
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ task_id: 't1', status: 'processing' });
  });

  it('passes the backend status code through', async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ detail: 'Invalid API key' }), { status: 401 }));

    const response = await getTask(new Request('http://localhost/api/task/t1'), { params: { taskId: 't1' } });

    expect(forwarded().url).toBe('http://backend.test:9000/api/task/t1');
    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ detail: 'Invalid API key' });
  });

  it('encodes the task id so it cannot change the backend path', async () => {
    fetchMock.mockResolvedValue(new Response('{}', { status: 200 }));

    await getTask(new Request('http://localhost/api/task/x'), { params: { taskId: '../admin?x=1' } });

    expect(forwarded().url).toBe('http://backend.test:9000/api/task/..%2Fadmin%3Fx%3D1');
  });

  it('returns 503 when the backend cannot be reached', async () => {
    fetchMock.mockRejectedValue(new TypeError('fetch failed'));

    const response = await getTask(new Request('http://localhost/api/task/t1'), { params: { taskId: 't1' } });

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ error: 'Multi-agent backend unavailable' });
  });
});
