export type ResearchFailureKind =
  | "timeout"
  | "budget"
  | "cancelled"
  | "failed"
  | "unreachable";

export const RESEARCH_TIMEOUT_MESSAGE =
  "The research took too long and was stopped.";
export const RESEARCH_BUDGET_MESSAGE =
  "The research used its token budget and was stopped.";
export const RESEARCH_UNREACHABLE_MESSAGE =
  "The research service isn't reachable right now.";
export const RESEARCH_STOPPED_MESSAGE = "Research stopped.";

/** A research run that ended without an answer; its message is shown to the user as is. */
export class ResearchFailed extends Error {
  constructor(
    public kind: ResearchFailureKind,
    message: string,
  ) {
    super(message);
    this.name = "ResearchFailed";
  }
}

/** Shown under "Thinking…" while a research run is processing. */
export interface ResearchProgress {
  step: string;
  latestLog?: string;
  elapsedMs: number;
}

/** What `useResearchRun().begin()` hands to `apiClient.sendChatMessage` for one run. */
export interface ResearchRunOptions {
  signal: AbortSignal;
  onProgress?: (progress: ResearchProgress) => void;
  onTaskId?: (taskId: string) => void;
}
