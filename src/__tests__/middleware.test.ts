/**
 * @jest-environment node
 */
import { NextRequest } from "next/server";
import { config, middleware } from "@/middleware";
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
