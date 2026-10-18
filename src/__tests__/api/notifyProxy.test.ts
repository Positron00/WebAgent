/**
 * @jest-environment node
 */
export {}; // a module, so its names don't clash with globals such as window.confirm

jest.mock("@/utils/logger", () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

type Handler = (req: Request) => Promise<Response>;
type Mod = Record<string, unknown>;

const fetchMock = jest.fn();
const TOKEN = "eyJwIjoiY29uZmlybSJ9.c2lnbmF0dXJl";
let confirm: Mod;
let unsubscribe: Mod;
let shared: typeof import("@/app/api/notify/shared");

beforeAll(async () => {
  process.env.WEBAGENT_BACKEND_URL = "http://backend.test:9000";
  process.env.WEBAGENT_API_KEY = "secret-key";
  confirm = await import("@/app/api/notify/confirm/route");
  unsubscribe = await import("@/app/api/notify/unsubscribe/route");
  shared = await import("@/app/api/notify/shared");
});

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

/** The backend: lookup names the topic for the purpose; the POSTs succeed. */
function backend(
  purpose: "confirm" | "unsubscribe",
  overrides: { lookup?: Response; action?: Response } = {},
) {
  fetchMock.mockImplementation(async (url: string) => {
    if (String(url).includes("/api/notify/lookup")) {
      return overrides.lookup ?? json({ topic_name: "EU AI rules", purpose });
    }
    return (
      overrides.action ??
      json({ status: purpose === "confirm" ? "confirmed" : "unsubscribed" })
    );
  });
}

beforeEach(() => {
  fetchMock.mockReset();
  global.fetch = fetchMock as unknown as typeof fetch;
});

const post = (mod: Mod) => mod.POST as Handler;

/** A request as a browser or mail client sends it, including a forged session header. */
const req = (url: string, init: RequestInit = {}) =>
  new Request(`http://localhost${url}`, {
    method: "POST",
    ...init,
    headers: {
      "x-webagent-session": "forged-sid",
      cookie: "wa_session=whatever",
      ...(init.headers as Record<string, string>),
    },
  });
const form = (body: string) => ({
  body,
  headers: { "Content-Type": "application/x-www-form-urlencoded" },
});
const jsonBody = (body: unknown) => ({
  body: JSON.stringify(body),
  headers: { "Content-Type": "application/json" },
});

function calls() {
  return fetchMock.mock.calls.map(([url, init]) => ({
    url: String(url),
    method: init.method,
    headers: new Headers(init.headers),
    body: init.body ? JSON.parse(init.body) : undefined,
  }));
}

describe("/api/notify proxies", () => {
  it.each([
    ["confirm", () => confirm],
    ["unsubscribe", () => unsubscribe],
  ])("%s exports POST only", (_name, mod) => {
    const methods = ["GET", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"];
    expect(typeof mod().POST).toBe("function");
    methods.forEach((m) => expect(mod()[m]).toBeUndefined());
  });

  it("forwards a JSON {token} with the API key and no session header", async () => {
    backend("confirm");
    const response = await post(confirm)(
      req("/api/notify/confirm", jsonBody({ token: TOKEN, extra: "x" })),
    );
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ status: "confirmed" });

    const [call] = calls();
    expect(call.url).toBe("http://backend.test:9000/api/notify/confirm");
    expect(call.method).toBe("POST");
    expect(call.body).toEqual({ token: TOKEN });
    expect(call.headers.get("X-API-Key")).toBe("secret-key");
    expect(call.headers.get("X-WebAgent-Client")).toBeNull();
    expect(call.headers.get("x-webagent-session")).toBeNull();
    expect(call.headers.get("cookie")).toBeNull();
  });

  it("accepts the RFC 8058 one-click POST with the token in the query, and answers JSON", async () => {
    backend("unsubscribe");
    const response = await post(unsubscribe)(
      req(
        `/api/notify/unsubscribe?token=${encodeURIComponent(TOKEN)}`,
        form("List-Unsubscribe=One-Click"),
      ),
    );
    expect(response.status).toBe(200);
    expect(response.headers.get("location")).toBeNull();
    expect(await response.json()).toEqual({ status: "unsubscribed" });

    const all = calls();
    expect(all).toHaveLength(1); // no lookup on one-click
    expect(all[0].url).toBe("http://backend.test:9000/api/notify/unsubscribe");
    expect(all[0].body).toEqual({ token: TOKEN });
    expect(all[0].headers.get("X-WebAgent-Client")).toBeNull();
  });

  it("doesn't treat a one-click body on confirm as one-click", async () => {
    backend("confirm");
    const response = await post(confirm)(
      req(
        `/api/notify/confirm?token=${TOKEN}`,
        form("List-Unsubscribe=One-Click"),
      ),
    );
    // A form post: the read-only lookup first, then a redirect
    expect(response.status).toBe(303);
    expect(calls()[0].url).toContain("/api/notify/lookup");
  });

  it("answers the page's form post with a 303 to a signed result view", async () => {
    backend("confirm");
    const response = await post(confirm)(
      req("/api/notify/confirm", form(`token=${encodeURIComponent(TOKEN)}`)),
    );
    expect(response.status).toBe(303);
    const location = response.headers.get("location")!;
    expect(location.startsWith("/notify/confirm?receipt=")).toBe(true);
    const receipt = new URL(location, "http://x").searchParams.get("receipt");
    expect(shared.verifyReceipt(receipt, "confirm")).toBe("EU AI rules");
    expect(shared.verifyReceipt(receipt, "unsubscribe")).toBeNull();

    const [lookup, action] = calls();
    expect(lookup.url).toBe(
      `http://backend.test:9000/api/notify/lookup?token=${encodeURIComponent(TOKEN)}`,
    );
    expect(lookup.method).toBe("GET");
    expect(lookup.headers.get("X-API-Key")).toBe("secret-key");
    expect(lookup.headers.get("X-WebAgent-Client")).toBeNull();
    expect(action.body).toEqual({ token: TOKEN });
    expect(action.headers.get("X-WebAgent-Client")).toBeNull();
  });

  it("redirects an invalid token to the error view without posting", async () => {
    backend("unsubscribe", { lookup: json({ detail: "bad sig" }, 404) });
    const response = await post(unsubscribe)(
      req("/api/notify/unsubscribe", form(`token=${TOKEN}`)),
    );
    expect(response.status).toBe(303);
    expect(response.headers.get("location")).toBe(
      "/notify/unsubscribe?error=invalid",
    );
    expect(calls()).toHaveLength(1);
  });

  it("won't use a token for the other purpose", async () => {
    backend("confirm", {
      lookup: json({ topic_name: "x", purpose: "unsubscribe" }),
    });
    const response = await post(confirm)(
      req("/api/notify/confirm", form(`token=${TOKEN}`)),
    );
    expect(response.headers.get("location")).toBe(
      "/notify/confirm?error=invalid",
    );
    expect(calls()).toHaveLength(1);
  });

  it("redirects a used token (404 on the POST) to the error view", async () => {
    backend("confirm", { action: json({ detail: "used" }, 404) });
    const response = await post(confirm)(
      req("/api/notify/confirm", form(`token=${TOKEN}`)),
    );
    expect(response.headers.get("location")).toBe(
      "/notify/confirm?error=invalid",
    );
  });

  it("says unavailable when the backend can't be reached", async () => {
    fetchMock.mockRejectedValue(new Error("ECONNREFUSED"));
    const page = await post(confirm)(
      req("/api/notify/confirm", form(`token=${TOKEN}`)),
    );
    expect(page.headers.get("location")).toBe(
      "/notify/confirm?error=unavailable",
    );
    const api = await post(unsubscribe)(
      req("/api/notify/unsubscribe", jsonBody({ token: TOKEN })),
    );
    expect(api.status).toBe(503);
    expect(await api.json()).toEqual({ error: "unavailable" });
  });

  it("answers 404 invalid_token with no backend detail", async () => {
    backend("unsubscribe", {
      action: json({ detail: "signature mismatch for a@example.org" }, 404),
    });
    const response = await post(unsubscribe)(
      req("/api/notify/unsubscribe", jsonBody({ token: TOKEN })),
    );
    expect(response.status).toBe(404);
    expect(await response.json()).toEqual({ error: "invalid_token" });
  });

  it.each([
    ["a missing token", jsonBody({})],
    ["a non-string token", jsonBody({ token: 42 })],
    ["a token with odd characters", jsonBody({ token: "a b<script>" })],
    ["a too-long token", jsonBody({ token: "a".repeat(3000) })],
  ])("rejects %s without calling the backend", async (_name, init) => {
    const response = await post(confirm)(req("/api/notify/confirm", init));
    expect(response.status).toBe(404);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects a one-click POST without a query token, other content types, bad JSON and big bodies", async () => {
    const noToken = await post(unsubscribe)(
      req("/api/notify/unsubscribe", form("List-Unsubscribe=One-Click")),
    );
    expect(noToken.status).toBe(404);
    const text = await post(unsubscribe)(
      req(`/api/notify/unsubscribe?token=${TOKEN}`, {
        body: "x",
        headers: { "Content-Type": "text/plain" },
      }),
    );
    expect(text.status).toBe(415);
    const bad = await post(confirm)(
      req("/api/notify/confirm", {
        body: "{",
        headers: { "Content-Type": "application/json" },
      }),
    );
    expect(bad.status).toBe(400);
    const big = await post(confirm)(
      req("/api/notify/confirm", form(`token=${"a".repeat(20_000)}`)),
    );
    expect(big.status).toBe(413);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("result receipts", () => {
  it("rejects a forged, changed, stale or future receipt", () => {
    const now = Date.now();
    const good = shared.signReceipt("unsubscribe", "Topic", now);
    expect(shared.verifyReceipt(good, "unsubscribe", now)).toBe("Topic");

    const [payload, sig] = good.split(".");
    const otherPayload = Buffer.from(
      JSON.stringify({ p: "unsubscribe", n: "Evil", t: now }),
    ).toString("base64url");
    expect(
      shared.verifyReceipt(`${otherPayload}.${sig}`, "unsubscribe", now),
    ).toBeNull();
    expect(
      shared.verifyReceipt(`${payload}.AAAA`, "unsubscribe", now),
    ).toBeNull();
    expect(shared.verifyReceipt(`${payload}`, "unsubscribe", now)).toBeNull();
    expect(
      shared.verifyReceipt(good, "unsubscribe", now + 2 * 3600_000),
    ).toBeNull();
    expect(
      shared.verifyReceipt(
        shared.signReceipt("unsubscribe", "T", now + 3600_000),
        "unsubscribe",
        now,
      ),
    ).toBeNull();
    expect(shared.verifyReceipt(42, "unsubscribe", now)).toBeNull();
  });
});
