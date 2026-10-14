import { act, renderHook } from "@testing-library/react";
import { useResearchRun } from "@/hooks/useResearchRun";
import { apiClient } from "@/utils/apiClient";
import { logger } from "@/utils/logger";

jest.mock("@/utils/apiClient", () => ({
  apiClient: { cancelTask: jest.fn() },
}));

jest.mock("@/utils/logger", () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

const cancelTask = apiClient.cancelTask as jest.Mock;
const retry = {
  messages: [{ id: "u1", role: "user" as const, content: "q" }],
  imageBase64: null,
};

describe("useResearchRun", () => {
  beforeEach(() => {
    cancelTask.mockReset().mockResolvedValue(undefined);
  });

  it("tracks the task and its progress", () => {
    const { result } = renderHook(() => useResearchRun());
    let options!: ReturnType<typeof result.current.begin>;

    act(() => {
      options = result.current.begin();
    });
    expect(result.current.run).toBeNull();

    act(() => options.onTaskId!("t1"));
    expect(result.current.run).toMatchObject({ taskId: "t1" });
    expect(typeof result.current.run?.startedAt).toBe("number");

    act(() =>
      options.onProgress!({
        step: "Planning",
        latestLog: "x",
        elapsedMs: 1000,
      }),
    );
    expect(result.current.run?.progress).toEqual({
      step: "Planning",
      latestLog: "x",
      elapsedMs: 1000,
    });
  });

  it("Stop aborts polling, cancels the task and clears the run", async () => {
    const { result } = renderHook(() => useResearchRun());
    let options!: ReturnType<typeof result.current.begin>;
    act(() => {
      options = result.current.begin();
    });
    act(() => options.onTaskId!("t1"));

    await act(() => result.current.stop());

    expect(options.signal.aborted).toBe(true);
    expect(cancelTask).toHaveBeenCalledWith("t1");
    expect(result.current.run).toBeNull();
  });

  it("Stop before the task exists aborts without a cancel request", async () => {
    const { result } = renderHook(() => useResearchRun());
    let options!: ReturnType<typeof result.current.begin>;
    act(() => {
      options = result.current.begin();
    });

    await act(() => result.current.stop());

    expect(options.signal.aborted).toBe(true);
    expect(cancelTask).not.toHaveBeenCalled();
  });

  it("a failed cancel request is logged, not thrown", async () => {
    cancelTask.mockRejectedValue(new Error("offline"));
    const { result } = renderHook(() => useResearchRun());
    act(() => {
      result.current.begin().onTaskId!("t1");
    });

    await act(() => result.current.stop());

    expect(logger.warn).toHaveBeenCalledWith(
      "Could not cancel the research task",
      expect.objectContaining({ taskId: "t1", error: "offline" }),
    );
  });

  it("ignores callbacks from an older run", () => {
    const { result } = renderHook(() => useResearchRun());
    let first!: ReturnType<typeof result.current.begin>;
    act(() => {
      first = result.current.begin();
    });
    act(() => {
      result.current.begin();
    });

    expect(first.signal.aborted).toBe(true);
    act(() => first.onTaskId!("old"));
    expect(result.current.run).toBeNull();
  });

  it("fail keeps the retry info; begin and clear remove it", () => {
    const { result } = renderHook(() => useResearchRun());
    act(() => {
      result.current.begin().onTaskId!("t1");
    });

    act(() => result.current.fail("timeout", "too long", retry));
    expect(result.current.run).toBeNull();
    expect(result.current.failure).toEqual({
      kind: "timeout",
      message: "too long",
      retry,
    });

    act(() => {
      result.current.begin();
    });
    expect(result.current.failure).toBeNull();

    act(() => result.current.fail("budget", "budget", retry));
    act(() => result.current.clear());
    expect(result.current.failure).toBeNull();
  });

  it("aborts the running request when the chat unmounts", () => {
    const { result, unmount } = renderHook(() => useResearchRun());
    let options!: ReturnType<typeof result.current.begin>;
    act(() => {
      options = result.current.begin();
    });

    unmount();

    expect(options.signal.aborted).toBe(true);
  });
});
