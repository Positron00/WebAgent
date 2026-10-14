/**
 * @jest-environment node
 */
import {
  codesMatch,
  createSession,
  readAccessConfig,
  SESSION_MAX_AGE,
  verifySession,
} from "@/utils/session";

const SECRET = "s".repeat(32);
const CODE = "correct-horse-battery";

describe("createSession / verifySession", () => {
  it("signs a cookie that verifies and returns its session ID", async () => {
    const session = await createSession(SECRET);

    expect(session.sid).toMatch(/^[A-Za-z0-9_-]{22}$/);
    expect(session.maxAge).toBe(30 * 24 * 60 * 60);
    expect(session.value.split(".")).toHaveLength(3);
    expect(await verifySession(session.value, SECRET)).toBe(session.sid);
  });

  it("gives each session a new random ID", async () => {
    const a = await createSession(SECRET);
    const b = await createSession(SECRET);
    expect(a.sid).not.toBe(b.sid);
  });

  it("rejects an expired cookie", async () => {
    const now = Date.now();
    const session = await createSession(SECRET, now);
    expect(
      await verifySession(
        session.value,
        SECRET,
        now + (SESSION_MAX_AGE - 1) * 1000,
      ),
    ).toBe(session.sid);
    expect(
      await verifySession(
        session.value,
        SECRET,
        now + (SESSION_MAX_AGE + 1) * 1000,
      ),
    ).toBeNull();
  });

  it("rejects a cookie signed with another secret", async () => {
    const session = await createSession(SECRET);
    expect(await verifySession(session.value, "x".repeat(32))).toBeNull();
  });

  it("rejects a cookie with any changed byte", async () => {
    const { value } = await createSession(SECRET);
    for (let i = 0; i < value.length; i++) {
      if (value[i] === ".") continue;
      const changed =
        value.slice(0, i) + (value[i] === "A" ? "B" : "A") + value.slice(i + 1);
      expect(await verifySession(changed, SECRET)).toBeNull();
    }
  });

  it("rejects a later expiry with the old signature", async () => {
    const { value } = await createSession(SECRET);
    const [sid, exp, sig] = value.split(".");
    expect(
      await verifySession(`${sid}.${Number(exp) + 1000}.${sig}`, SECRET),
    ).toBeNull();
  });

  it.each([
    undefined,
    "",
    "a.b",
    "a.b.c.d",
    "sid.notanumber.sig",
    "sid.9999999999.",
    "sid.9999999999.!!!",
  ])("rejects the malformed value %p", async (value) => {
    expect(await verifySession(value, SECRET)).toBeNull();
  });

  it("verifies 1,000 cookies well under 5 seconds", async () => {
    const { value } = await createSession(SECRET);
    const started = Date.now();
    for (let i = 0; i < 1000; i++) {
      await verifySession(value, SECRET);
    }
    expect(Date.now() - started).toBeLessThan(5000);
  });
});

describe("codesMatch", () => {
  it("matches only the exact code", () => {
    expect(codesMatch(CODE, CODE)).toBe(true);
    expect(codesMatch("correct-horse-batterY", CODE)).toBe(false);
    expect(codesMatch("", CODE)).toBe(false);
    expect(codesMatch(CODE + "x", CODE)).toBe(false);
    expect(codesMatch(CODE.slice(0, -1), CODE)).toBe(false);
  });
});

describe("readAccessConfig", () => {
  it("is on with a long enough code and secret", () => {
    expect(
      readAccessConfig({
        WEBAGENT_ACCESS_CODE: CODE,
        WEBAGENT_SESSION_SECRET: SECRET,
      }),
    ).toEqual({
      kind: "on",
      code: CODE,
      secret: SECRET,
    });
  });

  it("is off in development without a code", () => {
    expect(readAccessConfig({ NODE_ENV: "development" })).toEqual({
      kind: "off-dev",
    });
  });

  it("is misconfigured in production without a code", () => {
    const config = readAccessConfig({ NODE_ENV: "production" });
    expect(config.kind).toBe("misconfigured");
    expect(config).toHaveProperty(
      "reason",
      expect.stringContaining("WEBAGENT_ACCESS_CODE"),
    );
  });

  it("is off when explicitly disabled, even in production", () => {
    expect(
      readAccessConfig({
        NODE_ENV: "production",
        WEBAGENT_ACCESS_CODE_DISABLED: "true",
      }),
    ).toEqual({
      kind: "off-explicit",
    });
  });

  it("treats a code shorter than 12 characters as misconfigured", () => {
    const config = readAccessConfig({
      WEBAGENT_ACCESS_CODE: "short-code1",
      WEBAGENT_SESSION_SECRET: SECRET,
    });
    expect(config).toHaveProperty(
      "reason",
      expect.stringContaining("WEBAGENT_ACCESS_CODE"),
    );
  });

  it("treats a secret shorter than 32 characters as misconfigured", () => {
    const config = readAccessConfig({
      WEBAGENT_ACCESS_CODE: CODE,
      WEBAGENT_SESSION_SECRET: "s".repeat(31),
    });
    expect(config).toHaveProperty(
      "reason",
      expect.stringContaining("WEBAGENT_SESSION_SECRET"),
    );
  });
});
