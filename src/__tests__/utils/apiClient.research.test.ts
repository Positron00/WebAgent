import { ApiClient } from "@/utils/apiClient";
import { RESEARCH_TIMEOUT_MESSAGE } from "@/types/research";
import { isSignInRequired, redirectToSignIn } from "@/utils/signIn";

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

// jsdom can't navigate, so only the decision to redirect is checked here.
jest.mock("@/utils/signIn", () => ({
  ...jest.requireActual("@/utils/signIn"),
  redirectToSignIn: jest.fn(),
}));

const jsonResponse = (body: unknown, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  statusText: status === 200 ? "OK" : "Error",
  headers: new Headers(),
  json: jest.fn().mockResolvedValue(body),
});

const question = [
  { id: "1", role: "user" as const, content: "research this", timestamp: "" },
];
const fetchMock = jest.fn();
const calledUrls = () => fetchMock.mock.calls.map(([url]) => url);

describe("ApiClient research failures and sign-in", () => {
  let client: ApiClient;

  beforeEach(() => {
    fetchMock.mockReset();
    (redirectToSignIn as jest.Mock).mockReset();
    global.fetch = fetchMock as unknown as typeof fetch;
    client = ApiClient.getInstance();
    (client as any).delay = jest.fn().mockResolvedValue(undefined);
  });

  it("shows the timeout message and does not fall back to /api/chat", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({ task_id: "t1", status: "processing" }),
      )
      .mockResolvedValueOnce(jsonResponse({ status: "processing" }))
      .mockResolvedValueOnce(
        jsonResponse({
          status: "failed",
          error: RESEARCH_TIMEOUT_MESSAGE,
          error_code: "timeout",
        }),
      );

    await expect(
      client.sendChatMessage(
        question,
        null,
        "balanced",
        "general",
        true,
        "always",
      ),
    ).rejects.toThrow(RESEARCH_TIMEOUT_MESSAGE);

    expect(calledUrls()).toEqual(["/api/task", "/api/task/t1", "/api/task/t1"]);
  });

  it("still falls back to /api/chat for other failures (until 3.3.0)", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({ task_id: "t2", status: "processing" }),
      )
      .mockResolvedValueOnce(jsonResponse({ status: "failed", error: "boom" }))
      .mockResolvedValueOnce(
        jsonResponse({
          choices: [
            {
              message: { role: "assistant", content: "direct" },
              finish_reason: "stop",
            },
          ],
        }),
      );

    const response = await client.sendChatMessage(
      question,
      null,
      "balanced",
      "general",
      true,
      "always",
    );

    expect(calledUrls()).toEqual(["/api/task", "/api/task/t2", "/api/chat"]);
    expect(response.choices[0].message.content).toBe("direct");
  });

  it("sends the browser to sign-in on 401 sign_in_required, without retry or fallback", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ error: "sign_in_required" }, 401),
    );

    await expect(
      client.sendChatMessage(
        question,
        null,
        "balanced",
        "general",
        true,
        "always",
      ),
    ).rejects.toMatchObject({ status: 401 });

    expect(redirectToSignIn).toHaveBeenCalledTimes(1);
    expect(calledUrls()).toEqual(["/api/task"]);
  });

  it("does not redirect for other 401 replies", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ error: "Invalid API key" }, 401),
    );

    await expect(
      client.sendChatMessage(
        question,
        null,
        "balanced",
        "general",
        true,
        "off",
      ),
    ).rejects.toThrow("Invalid API key");
    expect(redirectToSignIn).not.toHaveBeenCalled();
  });
});

describe("isSignInRequired", () => {
  it("matches only a 401 with error sign_in_required", () => {
    expect(isSignInRequired(401, { error: "sign_in_required" })).toBe(true);
    expect(isSignInRequired(403, { error: "sign_in_required" })).toBe(false);
    expect(isSignInRequired(401, { error: "wrong_code" })).toBe(false);
    expect(isSignInRequired(401, null)).toBe(false);
  });
});
