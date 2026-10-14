/**
 * @jest-environment node
 */
import { GET as getPing, HEAD as headPing } from "@/app/api/ping/route";
import { GET as getHealth } from "@/app/api/health/route";
import { ApiClient } from "@/utils/apiClient";
import { clientConfig } from "@/config/clientConfig";

afterEach(() => jest.restoreAllMocks());

const isIsoTime = (value: unknown) =>
  typeof value === "string" && !Number.isNaN(Date.parse(value));

describe("/api/ping (IMP-063)", () => {
  it("returns only status and timestamp", async () => {
    const response = await getPing();
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(Object.keys(body).sort()).toEqual(["status", "timestamp"]);
    expect(body.status).toBe("ok");
    expect(isIsoTime(body.timestamp)).toBe(true);
  });

  it("answers HEAD with 200 and no body", async () => {
    const response = await headPing();

    expect(response.status).toBe(200);
    expect(await response.text()).toBe("");
  });
});

describe("/api/health (IMP-063)", () => {
  const mockHealthy = (healthy: boolean) =>
    jest
      .spyOn(ApiClient, "getInstance")
      .mockReturnValue({ isHealthy: () => healthy } as unknown as ApiClient);

  it("returns only status, version and timestamp", async () => {
    mockHealthy(true);

    const response = await getHealth();
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(Object.keys(body).sort()).toEqual([
      "status",
      "timestamp",
      "version",
    ]);
    expect(body.status).toBe("healthy");
    expect(body.version).toBe(clientConfig.version);
    expect(isIsoTime(body.timestamp)).toBe(true);
    expect(response.headers.get("cache-control")).toContain("no-store");
  });

  it("reports degraded when the API client is unhealthy", async () => {
    mockHealthy(false);

    const body = await (await getHealth()).json();

    expect(body.status).toBe("degraded");
  });
});
