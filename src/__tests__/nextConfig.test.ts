/**
 * @jest-environment node
 */
export {};

type Header = { key: string; value: string };
type NextConfig = {
  headers: () => Promise<{ source: string; headers: Header[] }[]>;
  images: { remotePatterns: { protocol: string; hostname: string }[] };
};

const env = process.env as Record<string, string | undefined>;
const savedEnv = { ...env };

afterEach(() => {
  env.NODE_ENV = savedEnv.NODE_ENV;
  env.WEBAGENT_IMAGE_HOSTS = savedEnv.WEBAGENT_IMAGE_HOSTS;
});

function loadConfig(nodeEnv: string, imageHosts?: string): NextConfig {
  env.NODE_ENV = nodeEnv;
  if (imageHosts === undefined) delete env.WEBAGENT_IMAGE_HOSTS;
  else env.WEBAGENT_IMAGE_HOSTS = imageHosts;
  let config: NextConfig | undefined;
  jest.isolateModules(() => {
    config = require("../../next.config.js");
  });
  return config as NextConfig;
}

async function headersFor(nodeEnv: string): Promise<Record<string, string>> {
  const rules = await loadConfig(nodeEnv).headers();
  expect(rules).toHaveLength(1);
  expect(rules[0].source).toBe("/:path*");
  return Object.fromEntries(rules[0].headers.map((h) => [h.key, h.value]));
}

describe("next.config.js security headers (IMP-060)", () => {
  it("sends the CSP and other headers on every path in production, with HSTS", async () => {
    const headers = await headersFor("production");
    const csp = headers["Content-Security-Policy"];

    expect(csp).toContain("default-src 'self'");
    expect(csp).toContain("script-src 'self' 'unsafe-inline'");
    expect(csp).not.toContain("unsafe-eval");
    expect(csp).toContain("img-src 'self' data: blob:");
    expect(csp).toContain("connect-src 'self'");
    expect(csp).toContain("frame-ancestors 'none'");
    expect(headers["X-Frame-Options"]).toBe("DENY");
    expect(headers["X-Content-Type-Options"]).toBe("nosniff");
    expect(headers["Referrer-Policy"]).toBe("strict-origin-when-cross-origin");
    expect(headers["Strict-Transport-Security"]).toBe("max-age=31536000");
  });

  it("allows eval for dev tooling and leaves out HSTS outside production", async () => {
    const headers = await headersFor("development");

    expect(headers["Content-Security-Policy"]).toContain(
      "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
    );
    expect(headers["Strict-Transport-Security"]).toBeUndefined();
    expect(headers["X-Frame-Options"]).toBe("DENY");
  });
});

describe("next.config.js image hosts (IMP-066)", () => {
  it("allows no remote image hosts when WEBAGENT_IMAGE_HOSTS is unset or empty", () => {
    expect(loadConfig("production").images.remotePatterns).toEqual([]);
    expect(loadConfig("production", " , ").images.remotePatterns).toEqual([]);
  });

  it("allows only the listed hosts, over https", () => {
    const { images } = loadConfig(
      "production",
      "images.example.com, cdn.example.org",
    );

    expect(images.remotePatterns).toEqual([
      { protocol: "https", hostname: "images.example.com" },
      { protocol: "https", hostname: "cdn.example.org" },
    ]);
    expect(images).not.toHaveProperty("domains");
  });
});
