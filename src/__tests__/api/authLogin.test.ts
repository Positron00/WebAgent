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

import { verifySession } from "@/utils/session";

const CODE = "correct-horse-battery";
const SECRET = "s".repeat(32);
const env = process.env as Record<string, string | undefined>;
const savedEnv = { ...env };

let login: (req: Request) => Promise<Response>;
let logout: () => Promise<Response>;

beforeEach(async () => {
  env.WEBAGENT_ACCESS_CODE = CODE;
  env.WEBAGENT_SESSION_SECRET = SECRET;
  delete env.WEBAGENT_ACCESS_CODE_DISABLED;
  env.NODE_ENV = "test";
  // A fresh module per test resets the process-wide failure counter.
  jest.resetModules();
  ({ POST: login } = await import("@/app/api/auth/login/route"));
  ({ POST: logout } = await import("@/app/api/auth/logout/route"));
});

afterAll(() => {
  Object.keys(env).forEach((key) => delete env[key]);
  Object.assign(env, savedEnv);
});

const signIn = (code: unknown) =>
  login(
    new Request("http://localhost/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ code }),
    }),
  );

async function timed(promise: Promise<Response>) {
  const started = Date.now();
  const response = await promise;
  return { response, ms: Date.now() - started };
}

describe("POST /api/auth/login", () => {
  it("sets a valid HttpOnly, SameSite=Lax, 30-day cookie for the right code", async () => {
    const { response, ms } = await timed(signIn(CODE));

    expect(response.status).toBe(200);
    expect(ms).toBeGreaterThanOrEqual(490);
    const cookie = response.headers.get("set-cookie")!;
    expect(cookie).toMatch(/^wa_session=[^;]+/);
    expect(cookie).toMatch(/HttpOnly/i);
    expect(cookie).toMatch(/SameSite=Lax/i);
    expect(cookie).toMatch(/Path=\//);
    expect(cookie).toMatch(/Max-Age=2592000/);
    expect(cookie).not.toMatch(/Secure/i);

    const value = cookie.split(";")[0].slice("wa_session=".length);
    expect(await verifySession(value, SECRET)).toEqual(expect.any(String));
  });

  it("marks the cookie Secure in production", async () => {
    env.NODE_ENV = "production";
    const response = await signIn(CODE);
    expect(response.headers.get("set-cookie")).toMatch(/Secure/i);
  });

  it("answers 401 wrong_code no sooner than 500 ms, without a cookie", async () => {
    const { response, ms } = await timed(signIn("not-the-code"));

    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ error: "wrong_code" });
    expect(ms).toBeGreaterThanOrEqual(490);
    expect(response.headers.get("set-cookie")).toBeNull();
  });

  it.each([[123], [null], ["x".repeat(201)]])(
    "treats the bad code %p as wrong",
    async (code) => {
      expect((await signIn(code)).status).toBe(401);
    },
  );

  it("answers 429 after 20 wrong codes in a minute, even for the right code", async () => {
    const replies = await Promise.all(
      Array.from({ length: 22 }, () => signIn("wrong-guess")),
    );
    const statuses = replies.map((r) => r.status);

    expect(statuses.filter((s) => s === 401)).toHaveLength(20);
    expect(statuses.filter((s) => s === 429)).toHaveLength(2);
    expect((await signIn(CODE)).status).toBe(429);
  });

  it("answers 503 when sign-in is not set up", async () => {
    env.WEBAGENT_SESSION_SECRET = "too-short";
    const response = await signIn(CODE);
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: "access_code_not_configured",
    });
  });

  it("answers 200 without a cookie when the access code is off", async () => {
    delete env.WEBAGENT_ACCESS_CODE;
    const response = await signIn("anything");
    expect(response.status).toBe(200);
    expect(response.headers.get("set-cookie")).toBeNull();
  });
});

describe("POST /api/auth/logout", () => {
  it("clears the cookie", async () => {
    const response = await logout();
    const cookie = response.headers.get("set-cookie")!;
    expect(response.status).toBe(200);
    expect(cookie).toMatch(/^wa_session=;/);
    expect(cookie).toMatch(/Max-Age=0/);
    expect(cookie).toMatch(/HttpOnly/i);
  });
});
