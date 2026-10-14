// Fetch with retries where every attempt has its own AbortController and timeout (IMP-044).

export interface FetchRetryOptions<T> {
  timeoutMs: number;
  maxRetries: number;
  /** The caller's signal (for example a Stop button). Its abort is never retried. */
  signal?: AbortSignal;
  /** Turns a response into data, or throws. Default: `response.json()`. */
  readResponse?: (response: Response, attempt: number) => Promise<T>;
  isRetryable?: (error: unknown) => boolean;
  /** Delay before retry number `attempt` (1-based). */
  retryDelay?: (attempt: number) => number;
  delay?: (ms: number) => Promise<void>;
  onAttempt?: (attempt: number) => void;
  /** Called after every failed attempt, with whether another attempt follows. */
  onError?: (
    error: unknown,
    attempt: number,
    willRetry: boolean,
    delayMs: number,
  ) => void;
}

export function isTimeoutError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.name === "TimeoutError" || error.message.includes("timeout"))
  );
}

export function defaultIsRetryable(error: unknown): boolean {
  return error instanceof TypeError || isTimeoutError(error);
}

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

function timeoutError(timeoutMs: number): Error {
  const error = new Error(`Request timeout after ${timeoutMs} ms`);
  error.name = "TimeoutError";
  return error;
}

export async function fetchWithRetry<T>(
  url: string,
  init: RequestInit,
  opts: FetchRetryOptions<T>,
): Promise<T> {
  const readResponse =
    opts.readResponse ??
    ((response: Response) => response.json() as Promise<T>);
  const isRetryable = opts.isRetryable ?? defaultIsRetryable;
  const retryDelay =
    opts.retryDelay ??
    ((attempt: number) => Math.min(500 * 2 ** attempt, 5000));
  const delay = opts.delay ?? sleep;

  for (let attempt = 0; ; attempt++) {
    if (opts.signal?.aborted)
      throw opts.signal.reason ?? new DOMException("Aborted", "AbortError");
    opts.onAttempt?.(attempt);

    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort(timeoutError(opts.timeoutMs));
    }, opts.timeoutMs);
    const onOuterAbort = () => controller.abort(opts.signal?.reason);
    opts.signal?.addEventListener("abort", onOuterAbort, { once: true });

    let error: unknown;
    try {
      const response = await fetch(url, { ...init, signal: controller.signal });
      return await readResponse(response, attempt);
    } catch (caught) {
      // Some fetch implementations reject with a generic AbortError instead of the abort reason.
      error =
        timedOut && !isTimeoutError(caught)
          ? timeoutError(opts.timeoutMs)
          : caught;
    } finally {
      clearTimeout(timer);
      opts.signal?.removeEventListener("abort", onOuterAbort);
    }

    const willRetry =
      !opts.signal?.aborted && attempt < opts.maxRetries && isRetryable(error);
    const delayMs = willRetry ? retryDelay(attempt + 1) : 0;
    opts.onError?.(error, attempt, willRetry, delayMs);
    if (!willRetry) throw error;
    await delay(delayMs);
  }
}
