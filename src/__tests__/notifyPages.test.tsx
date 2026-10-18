/**
 * @jest-environment node
 */
import { renderToStaticMarkup } from "react-dom/server";

jest.mock("@/utils/logger", () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

type PageModule = {
  default: (props: {
    searchParams: Promise<Record<string, string | string[] | undefined>>;
  }) => Promise<React.ReactElement>;
};

const fetchMock = jest.fn();
const TOKEN = "eyJwIjoiY29uZmlybSJ9.c2lnbmF0dXJl";
let confirmPage: PageModule;
let unsubscribePage: PageModule;
let shared: typeof import("@/app/api/notify/shared");

beforeAll(async () => {
  process.env.WEBAGENT_BACKEND_URL = "http://backend.test:9000";
  process.env.WEBAGENT_API_KEY = "secret-key";
  confirmPage = await import("@/app/notify/confirm/page");
  unsubscribePage = await import("@/app/notify/unsubscribe/page");
  shared = await import("@/app/api/notify/shared");
});

beforeEach(() => {
  fetchMock.mockReset();
  global.fetch = fetchMock as unknown as typeof fetch;
});

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

async function render(
  mod: PageModule,
  params: Record<string, string | string[] | undefined>,
): Promise<string> {
  // The page returns <NotifyPage>, an async server component; renderToStaticMarkup can't await
  // it, so it's called here as React's server renderer would.
  let element: React.ReactElement = await mod.default({
    searchParams: Promise.resolve(params),
  });
  if (typeof element.type === "function") {
    const component = element.type as (
      p: unknown,
    ) => Promise<React.ReactElement>;
    element = await component(element.props);
  }
  return renderToStaticMarkup(element);
}

/** Only read-only GETs reached the backend. */
function expectNoStateChange() {
  for (const [url, init] of fetchMock.mock.calls) {
    expect(init.method).toBe("GET");
    expect(String(url)).toMatch(
      /^http:\/\/backend\.test:9000\/api\/notify\/lookup\?token=/,
    );
  }
}

describe.each([
  ["confirm", () => confirmPage, "Confirm", "Get emails about"],
  ["unsubscribe", () => unsubscribePage, "Unsubscribe", "Stop emails about"],
] as const)("/notify/%s", (purpose, mod, button, heading) => {
  it("names the topic and shows one button that POSTs the token, changing nothing", async () => {
    fetchMock.mockResolvedValue(
      json({ topic_name: "EU <b>AI</b> rules", purpose }),
    );
    const html = await render(mod(), { token: TOKEN });

    expect(html).toContain(heading);
    expect(html).toContain("EU &lt;b&gt;AI&lt;/b&gt; rules"); // escaped, not HTML
    expect(html).toContain(
      `<form action="/api/notify/${purpose}" method="post">`,
    );
    expect(html).toContain(
      `<input type="hidden" name="token" value="${TOKEN}"/>`,
    );
    expect(html.match(/<button/g)).toHaveLength(1);
    expect(html).toContain(`>${button}</button>`);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      `http://backend.test:9000/api/notify/lookup?token=${encodeURIComponent(TOKEN)}`,
    );
    const headers = new Headers(init.headers);
    expect(headers.get("X-API-Key")).toBe("secret-key");
    expect(headers.get("X-WebAgent-Client")).toBeNull();
    expectNoStateChange();
  });

  it.each([
    [
      "an expired or unknown token",
      () => json({ detail: "expired at ..." }, 404),
    ],
    [
      "a token for the other purpose",
      () =>
        json({
          topic_name: "T",
          purpose: purpose === "confirm" ? "unsubscribe" : "confirm",
        }),
    ],
  ])(
    "shows the generic message for %s, with no detail and no button",
    async (_n, reply) => {
      fetchMock.mockResolvedValue(reply());
      const html = await render(mod(), { token: TOKEN });
      expect(html).toContain("This link has expired or isn&#x27;t valid.");
      expect(html).not.toContain("expired at");
      expect(html).not.toContain("<form");
      expectNoStateChange();
    },
  );

  it("doesn't call the backend without a token or with a malformed one", async () => {
    for (const params of [{}, { token: "" }, { token: "<script>" }]) {
      const html = await render(mod(), params);
      expect(html).toContain("This link has expired or isn&#x27;t valid.");
    }
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("says try later when the backend is down", async () => {
    fetchMock.mockRejectedValue(new Error("ECONNREFUSED"));
    const html = await render(mod(), { token: TOKEN });
    expect(html).toContain("try the link again later");
    expect(html).not.toContain("<form");
  });

  it("shows the result from a signed receipt, without calling the backend", async () => {
    const receipt = shared.signReceipt(purpose, "EU AI rules");
    const html = await render(mod(), { receipt });
    expect(html).toContain(
      purpose === "confirm"
        ? "Confirmed: you&#x27;ll get emails about “EU AI rules”."
        : "You won&#x27;t get more emails about “EU AI rules”.",
    );
    expect(html).not.toContain("<form");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("won't show a result for a forged receipt or one for the other page", async () => {
    const other = shared.signReceipt(
      purpose === "confirm" ? "unsubscribe" : "confirm",
      "T",
    );
    for (const receipt of [other, "e30.AAAA"]) {
      const html = await render(mod(), { receipt });
      expect(html).toContain("This link has expired or isn&#x27;t valid.");
    }
  });

  it("shows the error views the proxy redirects to", async () => {
    expect(await render(mod(), { error: "invalid" })).toContain(
      "This link has expired or isn&#x27;t valid.",
    );
    expect(await render(mod(), { error: "unavailable" })).toContain(
      "try the link again later",
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
