/**
 * @jest-environment node
 */
jest.mock("@/utils/logger", () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

export {};

const fetchMock = jest.fn();
let postTask: (req: Request) => Promise<Response>;
let getTask: (
  req: Request,
  ctx: { params: Promise<{ taskId: string }> },
) => Promise<Response>;
let cancelTask: (
  req: Request,
  ctx: { params: Promise<{ taskId: string }> },
) => Promise<Response>;

beforeAll(async () => {
  // The proxy reads these when the module loads
  process.env.WEBAGENT_BACKEND_URL = "http://backend.test:9000";
  process.env.WEBAGENT_API_KEY = "secret-key";
  ({ POST: postTask } = await import("@/app/api/task/route"));
  ({ GET: getTask } = await import("@/app/api/task/[taskId]/route"));
  ({ POST: cancelTask } = await import("@/app/api/task/[taskId]/cancel/route"));
});

beforeEach(() => {
  fetchMock.mockReset();
  global.fetch = fetchMock as unknown as typeof fetch;
});

function forwarded(index = 0) {
  const [url, init] = fetchMock.mock.calls[index];
  return { url, init, headers: new Headers(init.headers) };
}

describe("/api/task proxy", () => {
  it("forwards a new task to the backend with the server-side API key", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ task_id: "t1", status: "processing" }), {
        status: 200,
      }),
    );

    const response = await postTask(
      new Request("http://localhost/api/task", {
        method: "POST",
        body: JSON.stringify({ query: "q", agentic: true }),
      }),
    );

    const { url, init, headers } = forwarded();
    expect(url).toBe("http://backend.test:9000/api/task");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ query: "q", agentic: true });
    expect(headers.get("X-API-Key")).toBe("secret-key");
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      task_id: "t1",
      status: "processing",
    });
  });

  it("passes the backend status code through", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ detail: "Invalid API key" }), {
        status: 401,
      }),
    );

    const response = await getTask(
      new Request("http://localhost/api/task/t1"),
      { params: Promise.resolve({ taskId: "t1" }) },
    );

    expect(forwarded().url).toBe("http://backend.test:9000/api/task/t1");
    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ detail: "Invalid API key" });
  });

  it("encodes the task id so it cannot change the backend path", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 200 }));

    await getTask(new Request("http://localhost/api/task/x"), {
      params: Promise.resolve({ taskId: "../admin?x=1" }),
    });

    expect(forwarded().url).toBe(
      "http://backend.test:9000/api/task/..%2Fadmin%3Fx%3D1",
    );
  });

  it("sends the session as X-WebAgent-Client so the backend rate-limits per session", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 200 }));

    await getTask(
      new Request("http://localhost/api/task/t1", {
        headers: { "x-webagent-session": "sid-A" },
      }),
      { params: Promise.resolve({ taskId: "t1" }) },
    );

    expect(forwarded().headers.get("X-WebAgent-Client")).toBe("s:sid-A");
    expect(forwarded().headers.get("X-API-Key")).toBe("secret-key");
  });

  it("never forwards the browser's own X-Forwarded-For, X-Real-IP or other headers", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 200 }));

    await postTask(
      new Request("http://localhost/api/task", {
        method: "POST",
        body: "{}",
        headers: {
          "x-forwarded-for": "1.2.3.4",
          "x-real-ip": "1.2.3.4",
          "x-webagent-client": "s:forged",
          cookie: "wa_session=abc",
          "x-api-key": "browser-key",
        },
      }),
    );

    const { headers } = forwarded();
    expect(headers.get("X-Forwarded-For")).toBeNull();
    expect(headers.get("X-Real-IP")).toBeNull();
    expect(headers.get("Cookie")).toBeNull();
    expect(headers.get("X-WebAgent-Client")).toBeNull();
    expect(headers.get("X-API-Key")).toBe("secret-key");
  });

  it("returns 503 when the backend cannot be reached", async () => {
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));

    const response = await getTask(
      new Request("http://localhost/api/task/t1"),
      { params: Promise.resolve({ taskId: "t1" }) },
    );

    expect(response.status).toBe(503);
    expect(response.headers.get("X-WebAgent-Proxy-Error")).toBe("unreachable");
    expect(await response.json()).toEqual({
      error: "Multi-agent backend unavailable",
    });
  });

  it("forwards a cancel with the API key and passes 202 and 409 through", async () => {
    fetchMock
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ status: "cancelling" }), { status: 202 }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: "finished" }), { status: 409 }),
      );
    const request = () =>
      new Request("http://localhost/api/task/t1/cancel", {
        method: "POST",
        headers: { "x-webagent-session": "sid-A" },
      });

    const accepted = await cancelTask(request(), {
      params: Promise.resolve({ taskId: "../t1" }),
    });
    const finished = await cancelTask(request(), {
      params: Promise.resolve({ taskId: "t1" }),
    });

    const { url, init, headers } = forwarded();
    expect(url).toBe("http://backend.test:9000/api/task/..%2Ft1/cancel");
    expect(init.method).toBe("POST");
    expect(headers.get("X-API-Key")).toBe("secret-key");
    expect(headers.get("X-WebAgent-Client")).toBe("s:sid-A");
    expect(accepted.status).toBe(202);
    expect(await accepted.json()).toEqual({ status: "cancelling" });
    expect(finished.status).toBe(409);
  });
});

describe("/api/task proxy body allow-list (REC-107)", () => {
  const post = (body: BodyInit, headers: Record<string, string> = {}) =>
    postTask(
      new Request("http://localhost/api/task", {
        method: "POST",
        body,
        headers,
      }),
    );

  it("forwards every allow-listed field unchanged", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 200 }));
    const allowed = {
      query: "q",
      mode: "always",
      agentic: true,
      previous_messages: [{ role: "user", content: "hi" }],
      triage_override: true,
      image: "data:image/png;base64,AAAA",
      document_id: "doc-1",
      consent_token: "ct",
      approval_tokens: ["a1"],
    };

    await post(JSON.stringify(allowed));

    expect(JSON.parse(forwarded().init.body)).toEqual(allowed);
  });

  it("drops scheduler fields and anything else the browser adds", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 200 }));

    await post(
      JSON.stringify({
        query: "q",
        origin: "schedule",
        topic_id: "6f1d3b0e-6c1b-4f7e-9a51-2b7a1c0d9e11",
        task_id: "7a1d3b0e-6c1b-4f7e-9a51-2b7a1c0d9e11",
        token_budget: 200000,
        extra: "x",
      }),
    );

    const sent = JSON.parse(forwarded().init.body);
    expect(sent).toEqual({ query: "q" });
    expect(Object.keys(sent)).toEqual(["query"]);
  });

  it.each(["{not json", "", "[1,2]", "null", '"text"'])(
    "answers 400 invalid_json to %p without calling the backend",
    async (body) => {
      const response = await post(body);
      expect(response.status).toBe(400);
      expect(await response.json()).toEqual({ error: "invalid_json" });
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );

  it("answers 413 above the request-size limit without calling the backend", async () => {
    const saved = process.env.WEBAGENT_SECURITY_REQUEST_SIZE_LIMIT;
    process.env.WEBAGENT_SECURITY_REQUEST_SIZE_LIMIT = "100";
    try {
      const big = JSON.stringify({ query: "x".repeat(200) });
      const response = await post(big);
      expect(response.status).toBe(413);
      expect(fetchMock).not.toHaveBeenCalled();

      // A streamed body with no Content-Length is cut off too.
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(big));
          controller.close();
        },
      });
      const streamed = await postTask(
        new Request("http://localhost/api/task", {
          method: "POST",
          body: stream,
          // @ts-expect-error duplex is required for a stream body in Node
          duplex: "half",
        }),
      );
      expect(streamed.status).toBe(413);
      expect(fetchMock).not.toHaveBeenCalled();

      fetchMock.mockResolvedValue(new Response("{}", { status: 200 }));
      const small = await post(JSON.stringify({ query: "ok" }));
      expect(small.status).toBe(200);
    } finally {
      if (saved === undefined)
        delete process.env.WEBAGENT_SECURITY_REQUEST_SIZE_LIMIT;
      else process.env.WEBAGENT_SECURITY_REQUEST_SIZE_LIMIT = saved;
    }
  });

  it("allows 10 MB by default", async () => {
    const { requestSizeLimit } = await import("@/app/api/task/body");
    const saved = process.env.WEBAGENT_SECURITY_REQUEST_SIZE_LIMIT;
    delete process.env.WEBAGENT_SECURITY_REQUEST_SIZE_LIMIT;
    try {
      expect(requestSizeLimit()).toBe(10485760);
    } finally {
      if (saved !== undefined)
        process.env.WEBAGENT_SECURITY_REQUEST_SIZE_LIMIT = saved;
    }
  });
});
