jest.mock("@/utils/logger", () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

import { RateLimiter } from "@/utils/rateLimiter";

describe("RateLimiter", () => {
  let now: number;
  let limiter: RateLimiter;

  beforeEach(() => {
    now = 1_000_000;
    jest.spyOn(Date, "now").mockImplementation(() => now);
    limiter = new RateLimiter(60_000, 3);
  });

  afterEach(() => jest.restoreAllMocks());

  function send(key: string, times: number) {
    for (let i = 0; i < times; i++) limiter.addRequest(key);
  }

  it("limits a key once it reaches the limit inside the window", () => {
    send("A", 2);
    expect(limiter.isRateLimited("A")).toBe(false);
    send("A", 1);
    expect(limiter.isRateLimited("A")).toBe(true);
    expect(limiter.getRemainingRequests("A")).toBe(0);
  });

  it("keeps each key in its own bucket", () => {
    send("A", 3);
    expect(limiter.isRateLimited("A")).toBe(true);
    expect(limiter.isRateLimited("B")).toBe(false);
    expect(limiter.getRemainingRequests("B")).toBe(3);
  });

  it("uses a fixed window: being limited does not make the window longer", () => {
    send("A", 3);
    for (let i = 0; i < 5; i++) limiter.isRateLimited("A");
    now += 60_001;
    expect(limiter.isRateLimited("A")).toBe(false);
  });

  it("says how long until the oldest counted request leaves the window", () => {
    send("A", 1);
    now += 10_000;
    send("A", 2);
    expect(limiter.getTimeUntilNextAllowed("A")).toBe(50_000);
    expect(limiter.getTimeUntilNextAllowed("B")).toBe(0);
  });

  it("drops empty buckets in the periodic sweep", () => {
    send("A", 1);
    send("B", 1);
    expect(limiter.bucketCount()).toBe(2);
    now += 6 * 60_000;
    limiter.isRateLimited("C");
    expect(limiter.bucketCount()).toBe(0);
  });

  it("clear() empties every bucket", () => {
    send("A", 3);
    limiter.clear();
    expect(limiter.isRateLimited("A")).toBe(false);
  });
});
