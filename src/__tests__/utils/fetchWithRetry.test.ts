/**
 * @jest-environment node
 */
import { fetchWithRetry } from "@/utils/fetchWithRetry";

const fetchMock = jest.fn();
const signals: AbortSignal[] = [];

/** A fetch that only ends when its signal aborts, rejecting with the given error or the reason. */
function hangsUntilAborted(rejectWith?: () => unknown) {
  return (_url: string, init: RequestInit) => {
    const signal = init.signal as AbortSignal;
    signals.push(signal);
    return new Promise((_, reject) =>
      signal.addEventListener("abort", () =>
        reject(rejectWith ? rejectWith() : signal.reason),
      ),
    );
  };
}

function succeeds(body: unknown) {
  return (_url: string, init: RequestInit) => {
    signals.push(init.signal as AbortSignal);
    return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));
  };
}

beforeEach(() => {
  jest.useFakeTimers();
  fetchMock.mockReset();
  signals.length = 0;
  global.fetch = fetchMock as unknown as typeof fetch;
});

afterEach(() => jest.useRealTimers());

const options = { timeoutMs: 1000, maxRetries: 2, retryDelay: () => 0 };

describe("fetchWithRetry", () => {
  it("times out attempt 1, then gives attempt 2 a fresh signal that succeeds", async () => {
    fetchMock
      .mockImplementationOnce(hangsUntilAborted())
      .mockImplementationOnce(succeeds({ ok: true }));

    const result = fetchWithRetry("/x", {}, options);
    await jest.advanceTimersByTimeAsync(1001);

    await expect(result).resolves.toEqual({ ok: true });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(signals[0].aborted).toBe(true);
    expect((signals[0].reason as Error).name).toBe("TimeoutError");
    expect(signals[1]).not.toBe(signals[0]);
    expect(signals[1].aborted).toBe(false);
    expect(jest.getTimerCount()).toBe(0);
  });

  it("clears the attempt timer when the attempt ends", async () => {
    fetchMock.mockImplementationOnce(succeeds({ ok: true }));

    await expect(fetchWithRetry("/x", {}, options)).resolves.toEqual({
      ok: true,
    });
    expect(jest.getTimerCount()).toBe(0);
  });

  it("retries a timeout even when fetch rejects with a generic AbortError", async () => {
    fetchMock
      .mockImplementationOnce(
        hangsUntilAborted(() => new DOMException("aborted", "AbortError")),
      )
      .mockImplementationOnce(succeeds({ ok: 2 }));

    const result = fetchWithRetry("/x", {}, options);
    await jest.advanceTimersByTimeAsync(1001);

    await expect(result).resolves.toEqual({ ok: 2 });
  });

  it("does not retry when the caller's signal aborts", async () => {
    fetchMock.mockImplementation(hangsUntilAborted());
    const caller = new AbortController();
    const onError = jest.fn();

    const result = fetchWithRetry(
      "/x",
      {},
      { ...options, signal: caller.signal, onError },
    );
    const assertion = expect(result).rejects.toBe("stopped");
    caller.abort("stopped");
    await assertion;

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith("stopped", 0, false, 0);
    expect(jest.getTimerCount()).toBe(0);
  });

  it("makes no request when the caller already aborted", async () => {
    const caller = new AbortController();
    caller.abort("stopped");

    await expect(
      fetchWithRetry("/x", {}, { ...options, signal: caller.signal }),
    ).rejects.toBe("stopped");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("gives up after maxRetries and throws the last error", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const delay = jest.fn().mockResolvedValue(undefined);

    await expect(
      fetchWithRetry("/x", {}, { ...options, delay }),
    ).rejects.toThrow("Failed to fetch");
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(delay).toHaveBeenCalledTimes(2);
  });

  it("does not retry an error the caller marks as final", async () => {
    fetchMock.mockImplementation(succeeds({}));
    const readResponse = jest.fn().mockRejectedValue(new Error("bad request"));

    await expect(
      fetchWithRetry(
        "/x",
        {},
        { ...options, readResponse, isRetryable: () => false },
      ),
    ).rejects.toThrow("bad request");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
