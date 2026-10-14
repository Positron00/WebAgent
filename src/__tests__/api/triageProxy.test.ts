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

const fetchMock = jest.fn();
let postTriage: (req: Request) => Promise<Response>;

beforeAll(async () => {
  process.env.WEBAGENT_BACKEND_URL = "http://backend.test:9000";
  process.env.WEBAGENT_API_KEY = "secret-key";
  ({ POST: postTriage } = await import("@/app/api/triage/route"));
});

beforeEach(() => {
  fetchMock.mockReset();
  global.fetch = fetchMock as unknown as typeof fetch;
});

function forwarded(index = 0) {
  const [url, init] = fetchMock.mock.calls[index];
  return { url, init, headers: new Headers(init.headers) };
}

describe("/api/triage proxy", () => {
  it("forwards POST body to the backend with the API key", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ route: "direct", reason: "rule" }), {
        status: 200,
      }),
    );

    const body = JSON.stringify({
      query: "hello",
      previous_user_question: null,
      has_image: false,
    });

    const response = await postTriage(
      new Request("http://localhost/api/triage", { method: "POST", body }),
    );

    const { url, init, headers } = forwarded();
    expect(url).toBe("http://backend.test:9000/api/triage");
    expect(init.method).toBe("POST");
    expect(init.body).toBe(body);
    expect(headers.get("X-API-Key")).toBe("secret-key");
    expect(response.status).toBe(200);
  });

  it("returns 503 with proxy error header when backend is unreachable", async () => {
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));

    const response = await postTriage(
      new Request("http://localhost/api/triage", {
        method: "POST",
        body: JSON.stringify({ query: "q" }),
      }),
    );

    expect(response.status).toBe(503);
    expect(response.headers.get("X-WebAgent-Proxy-Error")).toBe("unreachable");
    expect(await response.json()).toEqual({
      error: "Multi-agent backend unavailable",
    });
  });

  it("sends X-WebAgent-Client from the session and drops the browser X-Forwarded-For", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 200 }));

    await postTriage(
      new Request("http://localhost/api/triage", {
        method: "POST",
        body: "{}",
        headers: {
          "x-webagent-session": "sid-B",
          "x-forwarded-for": "1.2.3.4",
        },
      }),
    );

    const { headers } = forwarded();
    expect(headers.get("X-WebAgent-Client")).toBe("s:sid-B");
    expect(headers.get("X-Forwarded-For")).toBeNull();
  });
});
