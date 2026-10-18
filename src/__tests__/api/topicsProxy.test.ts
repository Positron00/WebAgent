/**
 * @jest-environment node
 */
import fs from "fs";
import path from "path";

jest.mock("@/utils/logger", () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

type Handler = (req: Request, ctx?: unknown) => Promise<Response>;
type Mod = Record<string, unknown>;

const fetchMock = jest.fn();
const TOPIC = "6f1d3b0e-6c1b-4f7e-9a51-2b7a1c0d9e11";
const RUN = "0b8d4c2a-1f3e-4a5b-8c6d-7e8f9a0b1c2d";
const TOPICS_DIR = path.join(__dirname, "../../app/api/topics");

let list: Mod;
let one: Mod;
let run: Mod;
let resume: Mod;
let runs: Mod;
let runOne: Mod;
// it.each tables are built before beforeAll runs, so they name the module and look it up later.
type ModName = "list" | "one" | "run" | "resume" | "runs" | "runOne";
const mods = (): Record<ModName, Mod> => ({
  list,
  one,
  run,
  resume,
  runs,
  runOne,
});

beforeAll(async () => {
  process.env.WEBAGENT_BACKEND_URL = "http://backend.test:9000";
  process.env.WEBAGENT_API_KEY = "secret-key";
  list = await import("@/app/api/topics/route");
  one = await import("@/app/api/topics/[topicId]/route");
  run = await import("@/app/api/topics/[topicId]/run/route");
  resume = await import("@/app/api/topics/[topicId]/resume/route");
  runs = await import("@/app/api/topics/[topicId]/runs/route");
  runOne = await import("@/app/api/topics/[topicId]/runs/[runId]/route");
});

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockImplementation(
    async () => new Response('{"ok":true}', { status: 200 }),
  );
  global.fetch = fetchMock as unknown as typeof fetch;
});

const handler = (mod: Mod | ModName, method: string) =>
  (typeof mod === "string" ? mods()[mod] : mod)[method] as Handler;
const ctx = (params: Record<string, string>) => ({
  params: Promise.resolve(params),
});
const req = (url: string, init: RequestInit = {}) =>
  new Request(`http://localhost${url}`, {
    ...init,
    headers: {
      "x-webagent-session": "sid-A",
      ...(init.headers as Record<string, string>),
    },
  });

function forwarded(index = 0) {
  const [url, init] = fetchMock.mock.calls[index];
  return { url, init, headers: new Headers(init.headers) };
}

describe("/api/topics proxies", () => {
  it.each([
    ["GET list", "list" as ModName, "GET", "/api/topics", {}, "/api/topics"],
    [
      "GET one",
      "one" as ModName,
      "GET",
      `/api/topics/${TOPIC}`,
      { topicId: TOPIC },
      `/api/topics/${TOPIC}`,
    ],
    [
      "DELETE",
      "one" as ModName,
      "DELETE",
      `/api/topics/${TOPIC}`,
      { topicId: TOPIC },
      `/api/topics/${TOPIC}`,
    ],
    [
      "Run now",
      "run" as ModName,
      "POST",
      `/api/topics/${TOPIC}/run`,
      { topicId: TOPIC },
      `/api/topics/${TOPIC}/run`,
    ],
    [
      "resume",
      "resume" as ModName,
      "POST",
      `/api/topics/${TOPIC}/resume`,
      { topicId: TOPIC },
      `/api/topics/${TOPIC}/resume`,
    ],
    [
      "runs",
      "runs" as ModName,
      "GET",
      `/api/topics/${TOPIC}/runs`,
      { topicId: TOPIC },
      `/api/topics/${TOPIC}/runs`,
    ],
    [
      "one run",
      "runOne" as ModName,
      "GET",
      `/api/topics/${TOPIC}/runs/${RUN}`,
      { topicId: TOPIC, runId: RUN },
      `/api/topics/${TOPIC}/runs/${RUN}`,
    ],
  ])(
    "%s forwards with the API key and the session",
    async (_name, mod, method, url, params, backendPath) => {
      const response = await handler(mod, method)(
        req(url, { method }),
        ctx(params),
      );

      const { url: sentUrl, init, headers } = forwarded();
      expect(sentUrl).toBe(`http://backend.test:9000${backendPath}`);
      expect(init.method).toBe(method);
      expect(headers.get("X-API-Key")).toBe("secret-key");
      expect(headers.get("X-WebAgent-Client")).toBe("s:sid-A");
      expect(response.status).toBe(200);
    },
  );

  it("passes 204 from DELETE through with no body", async () => {
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 204 }));
    const response = await handler(one, "DELETE")(
      req(`/api/topics/${TOPIC}`, { method: "DELETE" }),
      ctx({ topicId: TOPIC }),
    );
    expect(response.status).toBe(204);
    expect(await response.text()).toBe("");
  });

  it("passes the backend's topic_limit and guard refusals through", async () => {
    fetchMock
      .mockResolvedValueOnce(
        new Response('{"code":"topic_limit"}', { status: 409 }),
      )
      .mockResolvedValueOnce(
        new Response('{"code":"declined","reason":"Not allowed"}', {
          status: 403,
        }),
      );
    const body = JSON.stringify({ name: "n", query: "q" });
    const limit = await handler(
      list,
      "POST",
    )(req("/api/topics", { method: "POST", body }));
    const refused = await handler(
      list,
      "POST",
    )(req("/api/topics", { method: "POST", body }));
    expect(limit.status).toBe(409);
    expect(await limit.json()).toEqual({ code: "topic_limit" });
    expect(refused.status).toBe(403);
    expect(await refused.json()).toEqual({
      code: "declined",
      reason: "Not allowed",
    });
  });

  it.each([
    ["GET one", "one" as ModName, "GET", { topicId: "../admin" }],
    ["PATCH", "one" as ModName, "PATCH", { topicId: "not-a-uuid" }],
    ["DELETE", "one" as ModName, "DELETE", { topicId: `${TOPIC}/../x` }],
    ["Run now", "run" as ModName, "POST", { topicId: "x" }],
    ["resume", "resume" as ModName, "POST", { topicId: "" }],
    ["runs", "runs" as ModName, "GET", { topicId: "%2e%2e" }],
    [
      "one run (bad topic)",
      "runOne" as ModName,
      "GET",
      { topicId: "x", runId: RUN },
    ],
    [
      "one run (bad run)",
      "runOne" as ModName,
      "GET",
      { topicId: TOPIC, runId: "../../summarize" },
    ],
  ])(
    "%s answers 400 to a bad ID without calling the backend",
    async (_name, mod, method, params) => {
      const response = await handler(mod, method)(
        req("/api/topics/x", {
          method,
          ...(method === "PATCH" ? { body: "{}" } : {}),
        }),
        ctx(params),
      );
      expect(response.status).toBe(400);
      expect(await response.json()).toEqual({ error: "invalid_id" });
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );

  it("forwards only the create fields", async () => {
    const allowed = {
      name: "DB news",
      query: "Compare databases",
      schedule: {
        every: "week",
        weekday: "mon",
        time: "07:30",
        tz: "Europe/Paris",
      },
      notify: { when: "changes" },
      created_from_task: RUN,
    };
    await handler(
      list,
      "POST",
    )(
      req("/api/topics", {
        method: "POST",
        body: JSON.stringify({
          ...allowed,
          status: "active",
          run_requests: 99,
          origin: "schedule",
          token_budget: 1,
          topic_id: TOPIC,
        }),
      }),
    );
    const { init, headers } = forwarded();
    expect(init.method).toBe("POST");
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual(allowed);
  });

  it("forwards only the patch fields", async () => {
    await handler(one, "PATCH")(
      req(`/api/topics/${TOPIC}`, {
        method: "PATCH",
        body: JSON.stringify({
          name: "n",
          schedule: { every: "day", time: "08:00", tz: "UTC" },
          notify: { when: "always" },
          status: "paused",
          add_subscriber: "a@example.com",
          resumes: 5,
          query: "changed",
        }),
      }),
      ctx({ topicId: TOPIC }),
    );
    expect(JSON.parse(forwarded().init.body)).toEqual({
      name: "n",
      schedule: { every: "day", time: "08:00", tz: "UTC" },
      notify: { when: "always" },
      status: "paused",
    });
  });

  it("sends no body on Run now and resume, even when the browser sends one", async () => {
    await handler(run, "POST")(
      req(`/api/topics/${TOPIC}/run`, {
        method: "POST",
        body: '{"kind":"notify"}',
      }),
      ctx({ topicId: TOPIC }),
    );
    await handler(resume, "POST")(
      req(`/api/topics/${TOPIC}/resume`, { method: "POST", body: '{"x":1}' }),
      ctx({ topicId: TOPIC }),
    );
    expect(forwarded(0).init.body).toBeUndefined();
    expect(forwarded(1).init.body).toBeUndefined();
  });

  it("answers 400 invalid_json to a bad create or patch body without calling the backend", async () => {
    const created = await handler(
      list,
      "POST",
    )(req("/api/topics", { method: "POST", body: "{" }));
    const patched = await handler(one, "PATCH")(
      req(`/api/topics/${TOPIC}`, { method: "PATCH", body: "[]" }),
      ctx({ topicId: TOPIC }),
    );
    expect(created.status).toBe(400);
    expect(patched.status).toBe(400);
    expect(await created.json()).toEqual({ error: "invalid_json" });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("answers 413 to an oversized create body", async () => {
    const response = await handler(
      list,
      "POST",
    )(
      req("/api/topics", {
        method: "POST",
        body: JSON.stringify({ name: "n", query: "x".repeat(70 * 1024) }),
      }),
    );
    expect(response.status).toBe(413);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("exports only the listed methods, so Next.js answers 405 to the rest", () => {
    const methods = (mod: Mod) =>
      ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
        .filter((m) => m in mod)
        .sort();
    expect(methods(list)).toEqual(["GET", "POST"]);
    expect(methods(one)).toEqual(["DELETE", "GET", "PATCH"]);
    expect(methods(run)).toEqual(["POST"]);
    expect(methods(resume)).toEqual(["POST"]);
    expect(methods(runs)).toEqual(["GET"]);
    expect(methods(runOne)).toEqual(["GET"]);
    for (const mod of [list, one, run, resume, runs, runOne]) {
      expect(mod.dynamic).toBe("force-dynamic");
    }
  });

  it("has no route for the scheduler-only summarize and notify endpoints", () => {
    const routes: string[] = [];
    const walk = (dir: string) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(full);
        else if (/^route\.(t|j)sx?$/.test(entry.name)) {
          routes.push(
            path.relative(TOPICS_DIR, full).split(path.sep).join("/"),
          );
        }
      }
    };
    walk(TOPICS_DIR);
    expect(routes.sort()).toEqual([
      "[topicId]/resume/route.ts",
      "[topicId]/route.ts",
      "[topicId]/run/route.ts",
      "[topicId]/runs/[runId]/route.ts",
      "[topicId]/runs/route.ts",
      "route.ts",
    ]);
    expect(routes.some((r) => /summarize|notify/.test(r))).toBe(false);
  });

  it("returns 503 when the backend cannot be reached", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("fetch failed"));
    const response = await handler(list, "GET")(req("/api/topics"));
    expect(response.status).toBe(503);
  });
});
