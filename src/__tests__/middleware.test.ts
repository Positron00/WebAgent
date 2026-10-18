/**
 * @jest-environment node
 */
import { NextRequest } from "next/server";
import { config, middleware, NOTIFY_PUBLIC_PATHS } from "@/middleware";
import { createSession } from "@/utils/session";
import { safeNextPath } from "@/utils/safeNext";

const CODE = "correct-horse-battery";
const SECRET = "s".repeat(32);
const ENV_KEYS = [
  "WEBAGENT_ACCESS_CODE",
  "WEBAGENT_SESSION_SECRET",
  "WEBAGENT_ACCESS_CODE_DISABLED",
  "NODE_ENV",
];
const savedEnv: Record<string, string | undefined> = {};

function setEnv(values: Record<string, string | undefined>) {
  const env = process.env as Record<string, string | undefined>;
  ENV_KEYS.forEach((key) => delete env[key]);
  Object.assign(env, values);
}

function request(
  path: string,
  init: { cookie?: string; headers?: Record<string, string> } = {},
) {
  const headers = new Headers(init.headers);
  if (init.cookie) headers.set("cookie", `wa_session=${init.cookie}`);
  return new NextRequest(new URL(path, "http://localhost:3000"), { headers });
}

/** The request headers the middleware passes on to the route handler. */
function forwardedHeader(response: Response, name: string): string | null {
  return response.headers.get(`x-middleware-request-${name}`);
}

beforeAll(() => ENV_KEYS.forEach((key) => (savedEnv[key] = process.env[key])));
afterAll(() => setEnv(savedEnv));
beforeEach(() => {
  setEnv({
    WEBAGENT_ACCESS_CODE: CODE,
    WEBAGENT_SESSION_SECRET: SECRET,
    NODE_ENV: "test",
  });
  jest.spyOn(console, "warn").mockImplementation(() => undefined);
});
afterEach(() => jest.restoreAllMocks());

describe("middleware with an access code", () => {
  it("answers 401 sign_in_required to an API call without a cookie", async () => {
    const response = await middleware(request("/api/chat"));
    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ error: "sign_in_required" });
  });

  it("redirects a page without a cookie to /login with next", async () => {
    const response = await middleware(request("/some/page"));
    expect(response.status).toBe(307);
    const location = new URL(response.headers.get("location")!);
    expect(location.pathname).toBe("/login");
    expect(location.searchParams.get("next")).toBe("/some/page");
  });

  it("rejects an expired cookie and a cookie with a changed byte", async () => {
    const expired = await createSession(
      SECRET,
      Date.now() - 31 * 24 * 3600 * 1000,
    );
    expect(
      (await middleware(request("/api/chat", { cookie: expired.value })))
        .status,
    ).toBe(401);

    const { value } = await createSession(SECRET);
    const changed =
      value.slice(0, 3) + (value[3] === "x" ? "y" : "x") + value.slice(4);
    expect(
      (await middleware(request("/api/chat", { cookie: changed }))).status,
    ).toBe(401);
  });

  it("passes a valid cookie through with x-webagent-session set to the session ID", async () => {
    const session = await createSession(SECRET);
    const response = await middleware(
      request("/api/chat", { cookie: session.value }),
    );

    expect(response.status).toBe(200);
    expect(response.headers.get("x-middleware-next")).toBe("1");
    expect(forwardedHeader(response, "x-webagent-session")).toBe(session.sid);
  });

  it("replaces a browser-sent x-webagent-session with the real one", async () => {
    const session = await createSession(SECRET);
    const response = await middleware(
      request("/api/task", {
        cookie: session.value,
        headers: { "x-webagent-session": "forged" },
      }),
    );
    expect(forwardedHeader(response, "x-webagent-session")).toBe(session.sid);
  });
});

describe("topics need sign-in (REC-116)", () => {
  it.each([
    "/api/topics",
    "/api/topics/6f1d3b0e-6c1b-4f7e-9a51-2b7a1c0d9e11/run",
  ])("answers 401 to %s without a cookie", async (path) => {
    const response = await middleware(request(path));
    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ error: "sign_in_required" });
  });

  it.each(["/topics", "/topics/6f1d3b0e-6c1b-4f7e-9a51-2b7a1c0d9e11"])(
    "redirects %s to /login without a cookie",
    async (path) => {
      const response = await middleware(request(path));
      expect(response.status).toBe(307);
      const location = new URL(response.headers.get("location")!);
      expect(location.pathname).toBe("/login");
      expect(location.searchParams.get("next")).toBe(path);
    },
  );

  it("passes /api/topics through with the session when signed in", async () => {
    const session = await createSession(SECRET);
    const response = await middleware(
      request("/api/topics", { cookie: session.value }),
    );
    expect(response.headers.get("x-middleware-next")).toBe("1");
    expect(forwardedHeader(response, "x-webagent-session")).toBe(session.sid);
  });
});

describe("middleware without a working access code", () => {
  it("answers 503 to API calls when the code is too short", async () => {
    setEnv({ WEBAGENT_ACCESS_CODE: "short", WEBAGENT_SESSION_SECRET: SECRET });
    const response = await middleware(request("/api/task"));
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: "access_code_not_configured",
    });
  });

  it("answers 503 in production without a code, and sends pages to /login?error=config", async () => {
    setEnv({ NODE_ENV: "production" });
    expect((await middleware(request("/api/chat"))).status).toBe(503);

    const page = await middleware(request("/"));
    expect(new URL(page.headers.get("location")!).search).toBe("?error=config");
  });

  it("lets everything through in development without a code, and drops a browser-sent session", async () => {
    setEnv({ NODE_ENV: "development" });
    const response = await middleware(
      request("/api/chat", { headers: { "x-webagent-session": "forged" } }),
    );

    expect(response.headers.get("x-middleware-next")).toBe("1");
    expect(forwardedHeader(response, "x-webagent-session")).toBeNull();
  });

  it("lets everything through when explicitly disabled in production", async () => {
    setEnv({ NODE_ENV: "production", WEBAGENT_ACCESS_CODE_DISABLED: "true" });
    expect(
      (await middleware(request("/api/chat"))).headers.get("x-middleware-next"),
    ).toBe("1");
  });
});

describe("matcher", () => {
  const matcher = new RegExp(`^${config.matcher[0]}$`);

  it.each([
    "/",
    "/api/chat",
    "/api/task/t1",
    "/api/auth/logout",
    "/api/settings",
    "/some/page",
    "/topics",
    "/topics/6f1d3b0e-6c1b-4f7e-9a51-2b7a1c0d9e11",
    "/api/topics",
    "/api/topics/6f1d3b0e-6c1b-4f7e-9a51-2b7a1c0d9e11/runs",
  ])("checks %s", (path) => expect(matcher.test(path)).toBe(true));

  it.each([
    "/login",
    "/api/auth/login",
    "/api/health",
    "/_next/static/chunk.js",
    "/_next/image",
    "/favicon.ico",
    "/icons/icon-192.png",
    "/manifest.json",
    "/logo.svg",
  ])("leaves %s open", (path) => expect(matcher.test(path)).toBe(false));
});

describe("public notify routes (REC-134, RNFR-06)", () => {
  const matcher = new RegExp(`^${config.matcher[0]}$`);

  it("pins the list of public notify routes", () => {
    expect([...NOTIFY_PUBLIC_PATHS]).toEqual([
      "/notify/confirm",
      "/notify/unsubscribe",
      "/api/notify/confirm",
      "/api/notify/unsubscribe",
    ]);
    // And the matcher's exclusions are exactly the known ones plus these four
    const exclusions = config.matcher[0]
      .replace(/^\/\(\(\?!/, "")
      .replace(/\)\.\*\)$/, "")
      .split("|");
    expect(exclusions).toEqual([
      "_next/static",
      "_next/image",
      "favicon.ico",
      "icons/",
      "manifest.json",
      ".*\\.svg$",
      "login",
      "api/auth/login",
      "api/health",
      "notify/confirm$",
      "notify/unsubscribe$",
      "api/notify/confirm$",
      "api/notify/unsubscribe$",
    ]);
  });

  it.each([...NOTIFY_PUBLIC_PATHS])("leaves %s open", (path) =>
    expect(matcher.test(path)).toBe(false),
  );

  it.each([
    "/notify",
    "/notify/",
    "/notifyx",
    "/notify/confirmx",
    "/notify/confirm/",
    "/notify/confirm/extra",
    "/notify/unsubscribe2",
    "/notify/lookup",
    "/notify/other",
    "/api/notify",
    "/api/notify/lookup",
    "/api/notify/confirmx",
    "/api/notify/confirm/x",
    "/api/notify/unsubscribe/",
    "/api/notify/unsubscribe/x",
    "/api/notifyx/confirm",
    "/api/topics/6f1d3b0e-6c1b-4f7e-9a51-2b7a1c0d9e11/notify",
    "/x/notify/confirm",
    "/api/x/notify/confirm",
    "/topics/notify/confirm",
  ])("still checks %s", (path) => expect(matcher.test(path)).toBe(true));

  it("still asks for sign-in on look-alikes when the middleware runs", async () => {
    const api = await middleware(request("/api/notify/lookup?token=t"));
    expect(api.status).toBe(401);
    const page = await middleware(request("/notifyx"));
    expect(page.status).toBe(307);
  });
});

describe("safeNextPath", () => {
  it.each([
    ["/chat?x=1", "/chat?x=1"],
    ["/", "/"],
    [null, "/"],
    ["", "/"],
    ["https://evil.example", "/"],
    ["//evil.example", "/"],
    ["/\\evil.example", "/"],
    ["javascript:alert(1)", "/"],
  ])("%p -> %p", (next, expected) => expect(safeNextPath(next)).toBe(expected));
});
