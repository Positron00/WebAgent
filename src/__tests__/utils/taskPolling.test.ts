import { pollTask, TaskStatusReply } from "@/utils/taskPolling";
import {
  RESEARCH_BUDGET_MESSAGE,
  RESEARCH_TIMEOUT_MESSAGE,
  ResearchFailed,
} from "@/types/research";

jest.mock("@/utils/logger", () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

const processing = (
  progress?: TaskStatusReply["progress"],
): TaskStatusReply => ({
  status: "processing",
  ...(progress ? { progress } : {}),
});
const completed: TaskStatusReply = {
  status: "completed",
  result: { content: "# Report", logs: ["done"] },
};

/** Runs pollTask with fake timers until it settles. */
async function run(
  replies: Array<TaskStatusReply | Error>,
  extra: {
    onProgress?: jest.Mock;
    deadlineMs?: number;
    signal?: AbortSignal;
  } = {},
) {
  const getStatus = jest.fn();
  replies.forEach((reply) =>
    reply instanceof Error
      ? getStatus.mockRejectedValueOnce(reply)
      : getStatus.mockResolvedValueOnce(reply),
  );
  const promise = pollTask("t1", {
    signal: extra.signal ?? new AbortController().signal,
    getStatus,
    onProgress: extra.onProgress,
    deadlineMs: extra.deadlineMs,
  });
  const settled = promise.then(
    (value) => ({ value, error: undefined }),
    (error) => ({ value: undefined, error }),
  );
  await jest.runAllTimersAsync();
  return { ...(await settled), getStatus };
}

describe("pollTask", () => {
  it("keeps the task ID, the report title and the standalone question for Track this", async () => {
    jest.useFakeTimers();
    try {
      const { value } = await run([
        {
          status: "completed",
          result: {
            content: "# Report",
            title: "Database comparison",
            standalone_query: "Compare Postgres and MySQL in 2026",
          },
        },
      ]);
      expect(value?.choices[0].message.task).toEqual({
        id: "t1",
        title: "Database comparison",
        standaloneQuery: "Compare Postgres and MySQL in 2026",
      });
    } finally {
      jest.useRealTimers();
    }
  });

  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  it("returns the report and reports progress on each processing reply", async () => {
    const onProgress = jest.fn();
    const { value, getStatus } = await run(
      [
        processing(),
        processing({ step: "Web and internal research", logs: ["a", "b"] }),
        completed,
      ],
      { onProgress },
    );

    expect(value?.choices[0].message).toEqual({
      role: "assistant",
      content: "# Report",
      logs: ["done"],
      task: { id: "t1" },
    });
    expect(getStatus).toHaveBeenCalledTimes(3);
    expect(onProgress).toHaveBeenCalledTimes(2);
    expect(onProgress.mock.calls[0][0]).toMatchObject({
      step: "Working",
      latestLog: undefined,
    });
    expect(onProgress.mock.calls[1][0]).toMatchObject({
      step: "Web and internal research",
      latestLog: "b",
    });
    expect(onProgress.mock.calls[1][0].elapsedMs).toBeGreaterThan(0);
  });

  it.each([
    ["timeout", "", "timeout", RESEARCH_TIMEOUT_MESSAGE],
    ["budget", "", "budget", RESEARCH_BUDGET_MESSAGE],
    ["budget", "Used 50,000 tokens.", "budget", "Used 50,000 tokens."],
    ["cancelled", "Cancelled", "cancelled", "Research stopped."],
    [undefined, "Together error", "failed", "Together error"],
    [undefined, "", "failed", "Unknown error"],
  ])(
    "maps error_code %s with error %j to %s",
    async (code, error, kind, message) => {
      const { error: thrown } = await run([
        { status: "failed", error, ...(code ? { error_code: code } : {}) },
      ]);
      expect(thrown).toBeInstanceOf(ResearchFailed);
      expect(thrown).toMatchObject({ kind, message });
    },
  );

  it("keeps polling after a transient error", async () => {
    const { value, getStatus } = await run([
      new TypeError("network"),
      completed,
    ]);
    expect(value?.choices[0].message.content).toBe("# Report");
    expect(getStatus).toHaveBeenCalledTimes(2);
  });

  it("stops at once on a 4xx error or a ResearchFailed from getStatus", async () => {
    const refusal = Object.assign(new Error("Escalated"), { status: 403 });
    const first = await run([refusal, completed]);
    expect(first.error).toBe(refusal);
    expect(first.getStatus).toHaveBeenCalledTimes(1);

    const unreachable = new ResearchFailed("unreachable", "down");
    const second = await run([unreachable]);
    expect(second.error).toBe(unreachable);
  });

  it("gives up with a timeout after the deadline", async () => {
    const { error } = await run(Array(20).fill(processing()), {
      deadlineMs: 3000,
    });
    expect(error).toMatchObject({
      kind: "timeout",
      message: RESEARCH_TIMEOUT_MESSAGE,
    });
  });

  it("stops without another request when the signal aborts", async () => {
    const controller = new AbortController();
    const getStatus = jest.fn().mockResolvedValue(processing());
    const promise = pollTask("t1", { signal: controller.signal, getStatus });
    const caught = promise.catch((error) => error);

    await jest.advanceTimersByTimeAsync(1000);
    expect(getStatus).toHaveBeenCalledTimes(1);
    controller.abort();
    await jest.advanceTimersByTimeAsync(10_000);

    expect(await caught).toMatchObject({
      kind: "cancelled",
      message: "Research stopped.",
    });
    expect(getStatus).toHaveBeenCalledTimes(1);
  });

  it("does not poll at all when the signal is already aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    const { error, getStatus } = await run([completed], {
      signal: controller.signal,
    });
    expect(error).toMatchObject({ kind: "cancelled" });
    expect(getStatus).not.toHaveBeenCalled();
  });
});
