import { ChatCompletionResponse } from "@/types/api";
import { normalizeSecuritySummary } from "@/types/security";
import { logger } from "@/utils/logger";
import {
  RESEARCH_BUDGET_MESSAGE,
  RESEARCH_STOPPED_MESSAGE,
  RESEARCH_TIMEOUT_MESSAGE,
  ResearchFailed,
  ResearchProgress,
} from "@/types/research";

/** The backend's GET /api/task/<id> reply. */
export interface TaskStatusReply {
  status?: string;
  result?: { content?: string; logs?: string[]; security?: unknown } | null;
  error?: string | null;
  error_code?: string;
  usage?: ChatCompletionResponse["usage"];
  security?: unknown;
  progress?: { step?: string; logs?: string[]; updated_at?: string };
}

export interface PollOptions {
  signal: AbortSignal;
  /**
   * Fetches GET /api/task/<id>. A thrown ResearchFailed, or an error with an HTTP 4xx
   * `status`, ends polling as it is; other errors are retried on the next poll.
   */
  getStatus: (taskId: string) => Promise<TaskStatusReply>;
  onProgress?: (progress: ResearchProgress) => void;
  /** A backstop just over the backend's own 7-minute run limit. */
  deadlineMs?: number;
}

export const DEFAULT_DEADLINE_MS = 8 * 60 * 1000;
const FIRST_INTERVAL_MS = 1000;
const MAX_INTERVAL_MS = 5000;
const BACKOFF = 1.5;

export const stoppedError = () =>
  new ResearchFailed("cancelled", RESEARCH_STOPPED_MESSAGE);

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(stoppedError());
    const onAbort = () => {
      clearTimeout(timer);
      reject(stoppedError());
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function isFinal(error: unknown): boolean {
  if (error instanceof ResearchFailed) return true;
  const status = (error as { status?: unknown } | null)?.status;
  return typeof status === "number" && status >= 400 && status < 500;
}

function failureFor(reply: TaskStatusReply): ResearchFailed {
  const message = reply.error || "";
  switch (reply.error_code) {
    case "timeout":
      return new ResearchFailed("timeout", message || RESEARCH_TIMEOUT_MESSAGE);
    case "budget":
      return new ResearchFailed("budget", message || RESEARCH_BUDGET_MESSAGE);
    case "cancelled":
      return stoppedError();
    default:
      return new ResearchFailed("failed", message || "Unknown error");
  }
}

function progressFrom(
  reply: TaskStatusReply,
  elapsedMs: number,
): ResearchProgress {
  const logs = reply.progress?.logs ?? [];
  return {
    step: reply.progress?.step || "Working",
    latestLog: logs.length > 0 ? logs[logs.length - 1] : undefined,
    elapsedMs,
  };
}

function toResponse(reply: TaskStatusReply): ChatCompletionResponse {
  const result = reply.result!;
  const security = normalizeSecuritySummary(reply.security ?? result.security);
  return {
    choices: [
      {
        message: {
          role: "assistant",
          content: result.content || JSON.stringify(result),
          ...(result.logs && { logs: result.logs }),
        },
        finish_reason: "stop",
      },
    ],
    usage: reply.usage || {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
    ...(security ? { security } : {}),
  };
}

/** Polls a research task until it completes, fails, is stopped or passes the deadline. */
export async function pollTask(
  taskId: string,
  opts: PollOptions,
): Promise<ChatCompletionResponse> {
  const { signal, getStatus, onProgress } = opts;
  const deadlineMs = opts.deadlineMs ?? DEFAULT_DEADLINE_MS;
  const started = Date.now();
  let interval = FIRST_INTERVAL_MS;

  while (Date.now() - started < deadlineMs) {
    await sleep(interval, signal);

    let reply: TaskStatusReply;
    try {
      reply = await getStatus(taskId);
    } catch (error) {
      if (signal.aborted) throw stoppedError();
      if (isFinal(error)) throw error;
      logger.warn("Error polling task", {
        taskId,
        error: error instanceof Error ? error.message : String(error),
      });
      interval = Math.min(interval * 2, MAX_INTERVAL_MS);
      continue;
    }
    if (signal.aborted) throw stoppedError();

    if (reply.status === "completed" && reply.result) return toResponse(reply);
    if (reply.status === "failed") throw failureFor(reply);

    onProgress?.(progressFrom(reply, Date.now() - started));
    interval = Math.min(interval * BACKOFF, MAX_INTERVAL_MS);
  }
  throw new ResearchFailed("timeout", RESEARCH_TIMEOUT_MESSAGE);
}
