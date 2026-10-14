import { ApiClient } from "@/utils/apiClient";
import { logger } from "@/utils/logger";

jest.mock("@/config/clientConfig", () => ({
  clientConfig: {
    environment: "test",
    version: "test",
    api: {
      baseUrl: "",
      endpoints: {
        chat: "/api/chat",
        task: "/api/task",
        ping: "/api/ping",
        health: "/api/health",
      },
      timeoutMs: 60000,
    },
    together: { endpoint: "/api/chat", model: "test-model" },
    features: { enableAgenticMode: true },
  },
}));

jest.mock("@/utils/logger", () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

const jsonResponse = (
  body: unknown,
  status = 200,
  headers?: Record<string, string>,
) => ({
  ok: status >= 200 && status < 300,
  status,
  statusText: status === 200 ? "OK" : "Error",
  headers: new Headers(headers),
  json: jest.fn().mockResolvedValue(body),
});

const SECRET_QUESTION = "marker-question-7f3a";
const question = [
  { id: "1", role: "user" as const, content: SECRET_QUESTION, timestamp: "" },
];
const fetchMock = jest.fn();
const calledUrls = () => fetchMock.mock.calls.map(([url]) => url);

describe("ApiClient research runs: progress, Stop, cancel and unreachable", () => {
  let client: ApiClient;

  beforeEach(() => {
    jest.useFakeTimers();
    fetchMock.mockReset();
    Object.values(logger).forEach((fn) => (fn as jest.Mock).mockClear());
    global.fetch = fetchMock as unknown as typeof fetch;
    client = ApiClient.getInstance();
    (client as any).delay = jest.fn().mockResolvedValue(undefined);
  });
  afterEach(() => jest.useRealTimers());

  async function send(options?: Parameters<ApiClient["sendChatMessage"]>[6]) {
    const settled = client
      .sendChatMessage(
        question,
        null,
        "balanced",
        "general",
        true,
        "always",
        options,
      )
      .then(
        (value) => ({ value, error: undefined as any }),
        (error) => ({ value: undefined, error }),
      );
    await jest.runAllTimersAsync();
    return settled;
  }

  it("calls onTaskId and onProgress, then returns the report", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({ task_id: "t1", status: "processing" }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          status: "processing",
          progress: { step: "Planning", logs: ["x"] },
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({ status: "completed", result: { content: "done" } }),
      );
    const onTaskId = jest.fn();
    const onProgress = jest.fn();

    const { value } = await send({
      research: { signal: new AbortController().signal, onTaskId, onProgress },
    });

    expect(value?.choices[0].message.content).toBe("done");
    expect(onTaskId).toHaveBeenCalledWith("t1");
    expect(onProgress).toHaveBeenCalledWith(
      expect.objectContaining({ step: "Planning", latestLog: "x" }),
    );
  });

  it("maps the proxy's 503 unreachable to an 'unreachable' failure, never calling /api/chat", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ error: "Multi-agent backend unavailable" }, 503, {
        "X-WebAgent-Proxy-Error": "unreachable",
      }),
    );

    const { error } = await send();

    expect(error).toMatchObject({
      kind: "unreachable",
      message: "The research service isn't reachable right now.",
    });
    expect(calledUrls().every((url) => url === "/api/task")).toBe(true);
  });

  it("turns a plain 500 from /api/task into a 'failed' failure", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ error: "boom" }, 500));
    const { error } = await send();
    expect(error).toMatchObject({ name: "ResearchFailed", kind: "failed" });
    expect(calledUrls()).not.toContain("/api/chat");
  });

  it("keeps an escalated task as a guard refusal", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({ task_id: "t3", status: "processing" }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          status: "failed",
          error: "Needs review",
          error_code: "escalated",
        }),
      );

    const { error } = await send();

    expect(error).toMatchObject({
      name: "ApiError",
      status: 403,
      data: { code: "escalated", reason: "Needs review" },
    });
  });

  it("stops polling when the signal aborts and reports 'cancelled'", async () => {
    const controller = new AbortController();
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({ task_id: "t4", status: "processing" }),
      )
      .mockResolvedValue(jsonResponse({ status: "processing" }));

    const settled = client
      .sendChatMessage(question, null, "balanced", "general", true, "always", {
        research: { signal: controller.signal },
      })
      .catch((error) => error);
    await jest.advanceTimersByTimeAsync(1000);
    const callsBeforeStop = fetchMock.mock.calls.length;
    controller.abort();
    await jest.runAllTimersAsync();

    expect(await settled).toMatchObject({
      kind: "cancelled",
      message: "Research stopped.",
    });
    expect(fetchMock.mock.calls.length).toBe(callsBeforeStop);
  });

  it("logs the question's length, never its text", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({ task_id: "t5", status: "processing" }),
      )
      .mockResolvedValueOnce(jsonResponse({ status: "failed", error: "boom" }));

    await send();

    const logged = JSON.stringify(
      Object.values(logger).map((fn) => (fn as jest.Mock).mock.calls),
    );
    expect(logged).not.toContain(SECRET_QUESTION);
    expect(logger.info).toHaveBeenCalledWith(
      "Sending multi-agent request",
      expect.objectContaining({ queryChars: SECRET_QUESTION.length }),
    );
  });

  describe("cancelTask", () => {
    it("posts to the cancel route", async () => {
      fetchMock.mockResolvedValue(jsonResponse({ status: "cancelling" }, 202));
      await client.cancelTask("t/1");
      expect(calledUrls()).toEqual(["/api/task/t%2F1/cancel"]);
      expect(fetchMock.mock.calls[0][1].method).toBe("POST");
    });

    it("ignores 409 (the task already finished)", async () => {
      fetchMock.mockResolvedValue(jsonResponse({ detail: "finished" }, 409));
      await expect(client.cancelTask("t1")).resolves.toBeUndefined();
    });

    it("throws other errors", async () => {
      fetchMock.mockResolvedValue(jsonResponse({ detail: "unknown" }, 404));
      await expect(client.cancelTask("t1")).rejects.toMatchObject({
        status: 404,
      });
    });
  });
});
